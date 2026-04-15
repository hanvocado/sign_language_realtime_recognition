"""
Standalone MLflow model promotion script — called by the Airflow
promote_or_skip task.

Implements the champion / challenger promotion policy using MLflow
model aliases.

Ref: https://mlflow.org/docs/2.11.0/model-registry.html#migrating-from-stages

Promotion rules
───────────────
  1. challenger val_acc < promote_min_acc  →  register as "rejected", done.
  2. No champion exists yet                →  challenger becomes champion immediately.
  3. challenger val_acc > champion val_acc  →  challenger promoted to champion;
                                               old champion re-tagged "retired" and
                                               pointed at by "previous_champion" alias.
  4. challenger val_acc ≤ champion val_acc  →  challenger stays "challenger", no swap.

MLflow aliases
──────────────
  champion          → currently-serving version (at most one).
  challenger        → latest registered challenger that did not beat the champion.
  previous_champion → most recently dethroned champion (for rollback / comparison).

MLflow version tags set on every registered version
────────────────────────────────────────────────────
  role          : "champion" | "challenger" | "retired" | "rejected"
  val_acc       : float string, e.g. "0.8423"
  gold_version  : e.g. "v0003"
  promoted_by   : "training_pipeline"
  trained_at    : ISO timestamp

Usage:
    python -m src.pipeline.promote_mlflow \
        --tracking-uri http://mlflow:5000 \
        --model-name SLR_model \
        --mlflow-run-id abc123 \
        --best-val-acc 0.8423 \
        --gold-version v0003 \
        --promote-min-acc 0.20 \
        --output /tmp/promotion_result.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime
from typing import Optional

import mlflow
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)

# ── Role tag values ──────────────────────────────────────────────────
ROLE_TAG    = "role"
CHAMPION    = "champion"
CHALLENGER  = "challenger"
RETIRED     = "retired"
REJECTED    = "rejected"

# ── MLflow aliases (replacement for deprecated stages) ──────────────
CHAMPION_ALIAS          = "champion"
CHALLENGER_ALIAS        = "challenger"
PREVIOUS_CHAMPION_ALIAS = "previous_champion"


# ── Helpers ──────────────────────────────────────────────────────────

def _get_champion(client: MlflowClient, model_name: str) -> tuple[Optional[str], float]:
    """
    Return (version, val_acc) of the current champion, or (None, 0.0).

    Resolves the champion via the "champion" alias on the registered model.
    Falls back to searching by role tag (handles legacy versions that were
    tagged before aliases were adopted).
    """
    try:
        mv = client.get_model_version_by_alias(model_name, CHAMPION_ALIAS)
    except Exception:
        mv = None

    if mv is not None:
        tags = client.get_model_version(model_name, mv.version).tags or {}
        acc  = float(tags.get("val_acc", 0.0))
        return mv.version, acc

    # Fallback: no alias set yet — look for a version tagged role=champion
    try:
        champion_versions = client.search_model_versions(
            filter_string=f"name='{model_name}' and tag.{ROLE_TAG}='{CHAMPION}'"
        )
    except Exception:
        champion_versions = []

    if not champion_versions:
        return None, 0.0

    mv   = sorted(champion_versions, key=lambda v: int(v.version), reverse=True)[0]
    tags = client.get_model_version(model_name, mv.version).tags or {}
    acc  = float(tags.get("val_acc", 0.0))
    return mv.version, acc


def _set_alias(client: MlflowClient, model_name: str, alias: str, version: str) -> None:
    client.set_registered_model_alias(model_name, alias, version)


def _delete_alias(client: MlflowClient, model_name: str, alias: str) -> None:
    try:
        client.delete_registered_model_alias(model_name, alias)
    except Exception:
        pass


def _tag_version(
    client: MlflowClient,
    model_name: str,
    version: str,
    role: str,
    val_acc: float,
    gold_version: str,
    trained_at: str,
) -> None:
    tags = {
        ROLE_TAG:       role,
        "val_acc":      f"{val_acc:.4f}",
        "gold_version": gold_version,
        "promoted_by":  "training_pipeline",
        "trained_at":   trained_at,
    }
    for key, value in tags.items():
        client.set_model_version_tag(model_name, version, key, value)


def _register_version(model_name: str, mlflow_run_id: str) -> str:
    """Register the model artifact and return the new version string."""
    artifact_uri = f"runs:/{mlflow_run_id}/model"
    mv = mlflow.register_model(artifact_uri, model_name)
    return mv.version


# ── Main entrypoint ───────────────────────────────────────────────────

def promote_model(
    tracking_uri: str,
    model_name: str,
    mlflow_run_id: str,
    best_val_acc: float,
    gold_version: str,
    promote_min_acc: float,
    trained_at: Optional[str] = None,
) -> dict:
    """
    Run the champion/challenger promotion policy and return a result dict:

        {
            "promoted":       bool,
            "challenger_ver": str,
            "champion_ver":   str | None,
            "role":           "champion" | "challenger" | "rejected",
        }
    """
    trained_at = trained_at or datetime.now().isoformat()

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri)

    # ── Step 1: below floor threshold → reject immediately ──
    if best_val_acc < promote_min_acc:
        logger.info(
            f"val_acc={best_val_acc:.4f} < floor threshold={promote_min_acc} "
            f"— registering as '{REJECTED}', no promotion."
        )
        version = _register_version(model_name, mlflow_run_id)
        _tag_version(client, model_name, version, REJECTED,
                     best_val_acc, gold_version, trained_at)
        return {
            "promoted":       False,
            "challenger_ver": version,
            "champion_ver":   None,
            "role":           REJECTED,
        }

    # ── Step 2: register challenger ──
    version = _register_version(model_name, mlflow_run_id)
    logger.info(
        f"Registered challenger: '{model_name}' v{version} "
        f"(val_acc={best_val_acc:.4f})"
    )

    # ── Step 3: compare against current champion ──
    champ_version, champ_acc = _get_champion(client, model_name)

    if champ_version is None:
        # No champion yet — first model wins automatically
        logger.info(f"No champion found. Crowning v{version} as first champion.")
        _set_alias(client, model_name, CHAMPION_ALIAS, version)
        _delete_alias(client, model_name, CHALLENGER_ALIAS)
        _tag_version(client, model_name, version, CHAMPION,
                     best_val_acc, gold_version, trained_at)
        return {
            "promoted":       True,
            "challenger_ver": version,
            "champion_ver":   version,
            "role":           CHAMPION,
        }

    if best_val_acc > champ_acc:
        # Challenger beats champion → swap roles
        logger.info(
            f"Challenger v{version} ({best_val_acc:.4f}) beats "
            f"champion v{champ_version} ({champ_acc:.4f}) — promoting."
        )
        # Retire the old champion, preserving its original provenance tags
        client.set_model_version_tag(model_name, champ_version, ROLE_TAG, RETIRED)
        client.set_model_version_tag(
            model_name, champ_version, "retired_at", datetime.now().isoformat()
        )
        client.set_model_version_tag(
            model_name, champ_version, "retired_by_version", str(version)
        )
        # Point "previous_champion" at the dethroned version for rollback.
        # Setting overwrites any prior previous_champion automatically.
        _set_alias(client, model_name, PREVIOUS_CHAMPION_ALIAS, champ_version)

        # Reassign the champion alias
        _set_alias(client, model_name, CHAMPION_ALIAS, version)
        _delete_alias(client, model_name, CHALLENGER_ALIAS)
        _tag_version(client, model_name, version, CHAMPION,
                     best_val_acc, gold_version, trained_at)
        return {
            "promoted":       True,
            "challenger_ver": version,
            "champion_ver":   version,
            "role":           CHAMPION,
        }

    # Challenger did not beat the champion → stays as challenger
    logger.info(
        f"Challenger v{version} ({best_val_acc:.4f}) did not beat "
        f"champion v{champ_version} ({champ_acc:.4f}) — keeping champion."
    )
    _set_alias(client, model_name, CHALLENGER_ALIAS, version)
    _tag_version(client, model_name, version, CHALLENGER,
                 best_val_acc, gold_version, trained_at)
    return {
        "promoted":       False,
        "challenger_ver": version,
        "champion_ver":   champ_version,
        "role":           CHALLENGER,
    }


# ── CLI entrypoint ────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    p = argparse.ArgumentParser()
    p.add_argument("--tracking-uri",    required=True)
    p.add_argument("--model-name",      required=True)
    p.add_argument("--mlflow-run-id",   required=True)
    p.add_argument("--best-val-acc",    required=True, type=float)
    p.add_argument("--gold-version",    required=True)
    p.add_argument("--promote-min-acc", required=True, type=float)
    p.add_argument("--trained-at",      default=None)
    p.add_argument("--output",          default=None,
                   help="Optional path to write the promotion result JSON.")
    args = p.parse_args()

    result = promote_model(
        tracking_uri=args.tracking_uri,
        model_name=args.model_name,
        mlflow_run_id=args.mlflow_run_id,
        best_val_acc=args.best_val_acc,
        gold_version=args.gold_version,
        promote_min_acc=args.promote_min_acc,
        trained_at=args.trained_at,
    )

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)

    print(json.dumps(result))

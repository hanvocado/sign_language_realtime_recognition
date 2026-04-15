"""
Task: MLflow model promotion with champion / challenger pattern.

Uses MLflow model aliases to mark serving roles.
The registered model carries a "champion" alias that always points to
the currently-serving version. Newly promoted challengers replace the
alias target; rejected / retired versions simply have no alias.

Ref: https://mlflow.org/docs/2.11.0/model-registry.html#migrating-from-stages

Promotion rules
───────────────
  1. challenger val_acc < PROMOTE_MIN_ACC  →  register as "rejected", done.
  2. No champion exists yet               →  challenger becomes champion immediately.
  3. challenger val_acc > champion val_acc →  challenger promoted to champion;
                                              old champion re-tagged "retired".
  4. challenger val_acc ≤ champion val_acc →  challenger stays "challenger", no swap.

MLflow aliases
──────────────
  champion   → currently-serving version (at most one).
  challenger → latest registered challenger that did not beat the champion.

MLflow version tags set on every registered version
────────────────────────────────────────────────────
  role          : "champion" | "challenger" | "retired" | "rejected"
  val_acc       : float string, e.g. "0.8423"
  gold_version  : e.g. "v0003"
  promoted_by   : "training_pipeline"
  trained_at    : ISO timestamp

training sensor state is always advanced so the sensor won't re-fire
for the same Gold snapshot regardless of the promotion outcome.
"""

from datetime import datetime

import mlflow
from mlflow.tracking import MlflowClient

from training.config import (
    MLFLOW_TRACKING_URI,
    MLFLOW_MODEL_NAME,
    PROMOTE_MIN_ACC,
)
from training.utils import (
    load_training_sensor_state,
    save_training_sensor_state,
)

# Tag key used to annotate the role of a model version
_ROLE_TAG   = "role"
_CHAMPION   = "champion"
_CHALLENGER = "challenger"
_RETIRED    = "retired"
_REJECTED   = "rejected"

# MLflow aliases (replacement for deprecated stages)
_CHAMPION_ALIAS   = "champion"
_CHALLENGER_ALIAS = "challenger"


# ── Helpers ──────────────────────────────────────────────────────────

def _get_champion(client: MlflowClient) -> tuple[str | None, float]:
    """
    Return (version, val_acc) of the current champion, or (None, 0.0).

    Resolves the champion via the "champion" alias on the registered model.
    Falls back to searching by role tag (handles legacy versions that were
    tagged before aliases were adopted).
    """
    try:
        mv = client.get_model_version_by_alias(MLFLOW_MODEL_NAME, _CHAMPION_ALIAS)
    except Exception:
        mv = None

    if mv is not None:
        tags = client.get_model_version(MLFLOW_MODEL_NAME, mv.version).tags or {}
        acc  = float(tags.get("val_acc", 0.0))
        return mv.version, acc

    # Fallback: no alias set yet — look for a version tagged role=champion
    try:
        champion_versions = client.search_model_versions(
            filter_string=f"name='{MLFLOW_MODEL_NAME}' and tag.{_ROLE_TAG}='{_CHAMPION}'"
        )
    except Exception:
        champion_versions = []

    if not champion_versions:
        return None, 0.0

    mv   = sorted(champion_versions, key=lambda v: int(v.version), reverse=True)[0]
    tags = client.get_model_version(MLFLOW_MODEL_NAME, mv.version).tags or {}
    acc  = float(tags.get("val_acc", 0.0))
    return mv.version, acc


def _set_alias(client: MlflowClient, alias: str, version: str) -> None:
    client.set_registered_model_alias(MLFLOW_MODEL_NAME, alias, version)


def _delete_alias(client: MlflowClient, alias: str) -> None:
    try:
        client.delete_registered_model_alias(MLFLOW_MODEL_NAME, alias)
    except Exception:
        pass


def _tag_version(
    client: MlflowClient,
    version: str,
    role: str,
    val_acc: float,
    gold_version: str,
    trained_at: str,
) -> None:
    tags = {
        _ROLE_TAG:      role,
        "val_acc":      f"{val_acc:.4f}",
        "gold_version": gold_version,
        "promoted_by":  "training_pipeline",
        "trained_at":   trained_at,
    }
    for key, value in tags.items():
        client.set_model_version_tag(MLFLOW_MODEL_NAME, version, key, value)


def _register_version(mlflow_run_id: str) -> str:
    """Register the model artifact and return the new version string."""
    artifact_uri = f"runs:/{mlflow_run_id}/model"
    mv = mlflow.register_model(artifact_uri, MLFLOW_MODEL_NAME)
    return mv.version


# ── Task callable ─────────────────────────────────────────────────────

def promote_or_skip(**context) -> None:
    """
    XCom outputs:
        promoted        : bool   — True if challenger became champion
        challenger_ver  : str    — MLflow version of the newly registered model
        champion_ver    : str    — MLflow version of the current champion (after task)
        role            : str    — role assigned to the new version
    """
    ti              = context["task_instance"]
    mlflow_run_id   = ti.xcom_pull(task_ids="train_model",             key="mlflow_run_id")
    best_val_acc    = float(ti.xcom_pull(task_ids="train_model",       key="best_val_acc") or 0.0)
    gold_version    = ti.xcom_pull(task_ids="training_retrain_check",  key="gold_version")
    iceberg_version = int(ti.xcom_pull(task_ids="iceberg_gold_sensor", key="iceberg_version") or 0)
    trained_at      = datetime.now().isoformat()

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient(MLFLOW_TRACKING_URI)

    # ── Step 1: below floor threshold → reject immediately ──
    if best_val_acc < PROMOTE_MIN_ACC:
        print(
            f"val_acc={best_val_acc:.4f} < floor threshold={PROMOTE_MIN_ACC} "
            f"— registering as '{_REJECTED}', no promotion."
        )
        version = _register_version(mlflow_run_id)
        _tag_version(client, version, _REJECTED, best_val_acc, gold_version, trained_at)

        ti.xcom_push(key="promoted",       value=False)
        ti.xcom_push(key="challenger_ver", value=version)
        ti.xcom_push(key="champion_ver",   value=None)
        ti.xcom_push(key="role",           value=_REJECTED)
        _advance_training_sensor_state(iceberg_version, mlflow_run_id, gold_version)
        return

    # ── Step 2: register challenger ──
    version = _register_version(mlflow_run_id)
    print(f"Registered challenger: '{MLFLOW_MODEL_NAME}' v{version} (val_acc={best_val_acc:.4f})")

    # ── Step 3: compare against current champion ──
    champ_version, champ_acc = _get_champion(client)

    if champ_version is None:
        # No champion yet — first model wins automatically
        print(f"No champion found. Crowning v{version} as first champion.")
        _set_alias(client, _CHAMPION_ALIAS, version)
        _delete_alias(client, _CHALLENGER_ALIAS)
        _tag_version(client, version, _CHAMPION, best_val_acc, gold_version, trained_at)

        ti.xcom_push(key="promoted",       value=True)
        ti.xcom_push(key="challenger_ver", value=version)
        ti.xcom_push(key="champion_ver",   value=version)
        ti.xcom_push(key="role",           value=_CHAMPION)

    elif best_val_acc > champ_acc:
        # Challenger beats champion → swap roles
        print(
            f"Challenger v{version} ({best_val_acc:.4f}) beats "
            f"champion v{champ_version} ({champ_acc:.4f}) — promoting."
        )
        # Retire the old champion, preserving its original provenance tags
        client.set_model_version_tag(MLFLOW_MODEL_NAME, champ_version, _ROLE_TAG, _RETIRED)
        client.set_model_version_tag(
            MLFLOW_MODEL_NAME, champ_version, "retired_at", datetime.now().isoformat()
        )
        client.set_model_version_tag(
            MLFLOW_MODEL_NAME, champ_version, "retired_by_version", str(version)
        )

        # Reassign the champion alias (implicitly drops the old champion from the alias)
        _set_alias(client, _CHAMPION_ALIAS, version)
        _delete_alias(client, _CHALLENGER_ALIAS)
        _tag_version(client, version, _CHAMPION, best_val_acc, gold_version, trained_at)

        ti.xcom_push(key="promoted",       value=True)
        ti.xcom_push(key="challenger_ver", value=version)
        ti.xcom_push(key="champion_ver",   value=version)
        ti.xcom_push(key="role",           value=_CHAMPION)

    else:
        # Challenger did not beat the champion → stays as challenger
        print(
            f"Challenger v{version} ({best_val_acc:.4f}) did not beat "
            f"champion v{champ_version} ({champ_acc:.4f}) — keeping champion."
        )
        _set_alias(client, _CHALLENGER_ALIAS, version)
        _tag_version(client, version, _CHALLENGER, best_val_acc, gold_version, trained_at)

        ti.xcom_push(key="promoted",       value=False)
        ti.xcom_push(key="challenger_ver", value=version)
        ti.xcom_push(key="champion_ver",   value=champ_version)
        ti.xcom_push(key="role",           value=_CHALLENGER)

    _advance_training_sensor_state(iceberg_version, mlflow_run_id, gold_version)


def _advance_training_sensor_state(
    iceberg_version: int,
    mlflow_run_id: str,
    gold_version: str,
) -> None:
    """Always advance consumed-version state so sensor won't re-fire."""
    sensor_state = load_training_sensor_state()
    sensor_state.update({
        "last_consumed_version": iceberg_version,
        "last_training_run_id": mlflow_run_id,
        "last_trained_at":      datetime.now().isoformat(),
        "last_gold_version":    gold_version,
    })
    save_training_sensor_state(sensor_state)
    print(f"Training sensor state advanced to consumed version {iceberg_version}")

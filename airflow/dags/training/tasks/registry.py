"""
Task: MLflow model promotion with champion / challenger pattern.

Every model that clears PROMOTE_MIN_ACC is registered and tagged
"challenger".  It is then compared against the current "champion"
(the Production model that carries the "champion" tag).

Promotion rules
───────────────
  1. challenger val_acc < PROMOTE_MIN_ACC  →  register as "rejected", done.
  2. No champion exists yet               →  challenger becomes champion immediately.
  3. challenger val_acc > champion val_acc →  challenger promoted to champion;
                                              old champion re-tagged "retired".
  4. challenger val_acc ≤ champion val_acc →  challenger stays "challenger", no swap.

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

# Tag key used to identify the active champion version
_ROLE_TAG = "role"
_CHAMPION  = "champion"
_CHALLENGER = "challenger"
_RETIRED   = "retired"
_REJECTED  = "rejected"


# ── Helpers ──────────────────────────────────────────────────────────

def _get_champion(client: MlflowClient) -> tuple[str | None, float]:
    """
    Return (version, val_acc) of the current champion, or (None, 0.0).
    Searches all Production versions for the one tagged role=champion.
    Falls back to the latest Production version if no tag is set
    (handles the migration case where old versions have no role tag).
    """
    try:
        versions = client.get_latest_versions(MLFLOW_MODEL_NAME, stages=["Production"])
    except Exception:
        return None, 0.0

    if not versions:
        return None, 0.0

    for mv in versions:
        tags = {t.key: t.value for t in client.get_model_version(
            MLFLOW_MODEL_NAME, mv.version
        ).tags}
        if tags.get(_ROLE_TAG) == _CHAMPION:
            acc = float(tags.get("val_acc", 0.0))
            return mv.version, acc

    # Fallback: treat latest Production version as champion (no role tag yet)
    mv  = versions[0]
    run = client.get_run(mv.run_id)
    acc = run.data.metrics.get("best_val_acc", 0.0)
    return mv.version, acc


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


def _transition(client: MlflowClient, version: str, stage: str,
                archive_existing: bool = False) -> None:
    client.transition_model_version_stage(
        name=MLFLOW_MODEL_NAME,
        version=version,
        stage=stage,
        archive_existing_versions=archive_existing,
    )


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
        _transition(client, version, "Archived")
        _tag_version(client, version, _REJECTED, best_val_acc, gold_version, trained_at)

        ti.xcom_push(key="promoted",       value=False)
        ti.xcom_push(key="challenger_ver", value=version)
        ti.xcom_push(key="champion_ver",   value=None)
        ti.xcom_push(key="role",           value=_REJECTED)
        _advance_training_sensor_state(iceberg_version, mlflow_run_id, gold_version)
        return

    # ── Step 2: register challenger ──
    version = _register_version(mlflow_run_id)
    _transition(client, version, "Staging")
    print(f"Registered challenger: '{MLFLOW_MODEL_NAME}' v{version} (val_acc={best_val_acc:.4f})")

    # ── Step 3: compare against current champion ──
    champ_version, champ_acc = _get_champion(client)

    if champ_version is None:
        # No champion yet — first model wins automatically
        print(f"No champion found. Crowning v{version} as first champion.")
        _transition(client, version, "Production", archive_existing=False)
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
        # Retire the old champion first
        _tag_version(client, champ_version, _RETIRED, champ_acc, gold_version, trained_at)
        _transition(client, champ_version, "Archived", archive_existing=False)

        # Crown the new champion
        _transition(client, version, "Production", archive_existing=False)
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
        _tag_version(client, version, _CHALLENGER, best_val_acc, gold_version, trained_at)
        # Leave in Staging so it's visible but not serving traffic

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

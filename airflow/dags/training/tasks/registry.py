"""
Task: MLflow model promotion with champion / challenger pattern.

Thin Airflow adapter — the actual promotion policy (champion / challenger /
previous_champion alias management, role tagging, rollback wiring) lives in
``src/pipeline/promote_mlflow.py``. This module only bridges XCom I/O and
advances the training sensor state.

Ref: https://mlflow.org/docs/2.11.0/model-registry.html#migrating-from-stages
"""

from datetime import datetime

from src.pipeline.promote_mlflow import promote_model
from training.config import (
    MLFLOW_TRACKING_URI,
    MLFLOW_MODEL_NAME,
    PROMOTE_MIN_ACC,
)
from training.utils import (
    load_training_sensor_state,
    save_training_sensor_state,
)


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

    result = promote_model(
        tracking_uri=MLFLOW_TRACKING_URI,
        model_name=MLFLOW_MODEL_NAME,
        mlflow_run_id=mlflow_run_id,
        best_val_acc=best_val_acc,
        gold_version=gold_version,
        promote_min_acc=PROMOTE_MIN_ACC,
        trained_at=datetime.now().isoformat(),
    )

    ti.xcom_push(key="promoted",       value=result["promoted"])
    ti.xcom_push(key="challenger_ver", value=result["challenger_ver"])
    ti.xcom_push(key="champion_ver",   value=result["champion_ver"])
    ti.xcom_push(key="role",           value=result["role"])

    # Always advance consumed-version state so sensor won't re-fire,
    # regardless of promotion outcome.
    _advance_training_sensor_state(iceberg_version, mlflow_run_id, gold_version)


def _advance_training_sensor_state(
    iceberg_version: int,
    mlflow_run_id: str,
    gold_version: str,
) -> None:
    sensor_state = load_training_sensor_state()
    sensor_state.update({
        "last_consumed_version": iceberg_version,
        "last_training_run_id":  mlflow_run_id,
        "last_trained_at":       datetime.now().isoformat(),
        "last_gold_version":     gold_version,
    })
    save_training_sensor_state(sensor_state)
    print(f"Training sensor state advanced to consumed version {iceberg_version}")

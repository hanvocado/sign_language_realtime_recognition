"""
Task: train model.

Invokes train_mlflow.py as a subprocess to keep CUDA process isolation
and ensure MLflow run state is fully flushed before the next task reads it.
After the subprocess exits, queries MLflow for the run_id and best_val_acc.
"""

import mlflow
from mlflow.tracking import MlflowClient

from airflow.dags.training.config import (
    PROJECT_ROOT,
    SEQ_LEN,
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT,
    MLFLOW_MODEL_NAME,
    MODEL_TYPE,
    HIDDEN_DIM,
    NUM_LAYERS,
    DROPOUT,
    BATCH_SIZE,
    LR,
    FINETUNE_LR,
    WEIGHT_DECAY,
    LABEL_SMOOTHING,
    EPOCHS,
    PATIENCE,
    NUM_WORKERS,
)
from airflow.dags.training.utils import run_streaming, ensure_run_context

# Re-export SEQ_LEN so it is importable directly from config
from src.config.config import SEQ_LEN  # noqa: F401 (already imported above via config.py)


def train_model(**context) -> None:
    """
    XCom outputs:
        mlflow_run_id : str   — MLflow run_id of the completed training run
        best_val_acc  : float — best validation accuracy achieved
    """
    ti          = context["task_instance"]
    split_dir   = ti.xcom_pull(task_ids="split_dataset",          key="split_dir")
    decision    = ti.xcom_pull(task_ids="training_retrain_check", key="decision")
    base_ckpt   = ti.xcom_pull(task_ids="training_retrain_check", key="base_ckpt_uri")
    run_ctx     = ensure_run_context(ti)

    ckpt_dir     = f"{run_ctx['run_dir']}/checkpoints"
    effective_lr = str(FINETUNE_LR if decision == "finetune" else LR)

    cmd = [
        "python", "-u",
        f"{PROJECT_ROOT}/src/pipeline/train_mlflow.py",
        "--tracking-uri",    MLFLOW_TRACKING_URI,
        "--experiment-name", MLFLOW_EXPERIMENT,
        "--model-name",      MLFLOW_MODEL_NAME,
        "--data-dir",        split_dir,
        "--ckpt-dir",        ckpt_dir,
        "--model-type",      MODEL_TYPE,
        "--hidden-dim",      str(HIDDEN_DIM),
        "--num-layers",      str(NUM_LAYERS),
        "--dropout",         str(DROPOUT),
        "--input-dim",       "225",
        "--seq-len",         str(SEQ_LEN),
        "--batch-size",      str(BATCH_SIZE),
        "--lr",              effective_lr,
        "--weight-decay",    str(WEIGHT_DECAY),
        "--label-smoothing", str(LABEL_SMOOTHING),
        "--epochs",          str(EPOCHS),
        "--patience",        str(PATIENCE),
        "--num-workers",     str(NUM_WORKERS),
        "--normalize",       "true",
        "--augment-train",   "true",
    ]

    if decision == "finetune" and base_ckpt:
        cmd += ["--base-ckpt-uri", base_ckpt, "--finetune-lr", str(FINETUNE_LR)]

    run_streaming(cmd, cwd=PROJECT_ROOT)

    # ── Retrieve completed run from MLflow ──
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient(MLFLOW_TRACKING_URI)
    exp    = client.get_experiment_by_name(MLFLOW_EXPERIMENT)
    if exp is None:
        raise RuntimeError(f"MLflow experiment '{MLFLOW_EXPERIMENT}' not found")

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError("No MLflow runs found after training completed")

    mlflow_run_id = runs[0].info.run_id
    best_val_acc  = runs[0].data.metrics.get("best_val_acc", 0.0)

    print(f"Training complete — run_id={mlflow_run_id}, best_val_acc={best_val_acc:.4f}")
    ti.xcom_push(key="mlflow_run_id", value=mlflow_run_id)
    ti.xcom_push(key="best_val_acc",  value=best_val_acc)

"""
Task: train model.

Invokes train_mlflow.py as a subprocess to keep CUDA process isolation
and ensure MLflow run state is fully flushed before the next task reads it.
After the subprocess exits, queries MLflow for the run_id and best_val_acc.
"""

import mlflow
from mlflow.tracking import MlflowClient

from training.config import (
    PROJECT_ROOT,
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT,
    MLFLOW_MODEL_NAME,
)
from training.utils import run_streaming, ensure_run_context


def train_model(**context) -> None:
    """
    XCom outputs:
        mlflow_run_id : str   — MLflow run_id of the completed training run
        best_val_acc  : float — best validation accuracy achieved
    """
    ti          = context["task_instance"]
    params      = context["params"]
    split_dir   = ti.xcom_pull(task_ids="split_dataset",          key="split_dir")
    decision    = ti.xcom_pull(task_ids="training_retrain_check", key="decision")
    gold_version = ti.xcom_pull(task_ids="training_retrain_check", key="gold_version")
    base_ckpt   = ti.xcom_pull(task_ids="training_retrain_check", key="base_ckpt_uri")
    run_ctx     = ensure_run_context(ti)
    run_name    = f"{gold_version}_{decision}_{run_ctx['run_stamp']}"

    ckpt_dir     = f"{run_ctx['run_dir']}/checkpoints"
    run_id_file  = f"{run_ctx['run_dir']}/mlflow_run_id.txt"
    effective_lr = str(params["finetune_lr"] if decision == "finetune" else params["lr"])

    cmd = [
        "python", "-u",
        f"{PROJECT_ROOT}/src/pipeline/train_mlflow.py",
        "--tracking-uri",    MLFLOW_TRACKING_URI,
        "--experiment-name", MLFLOW_EXPERIMENT,
        "--model-name",      MLFLOW_MODEL_NAME,
        "--data-dir",        split_dir,
        "--ckpt-dir",        ckpt_dir,
        "--run-name",        run_name,
        "--run-id-output-file", run_id_file,
        "--model-type",      params["model_type"],
        "--hidden-dim",      str(params["hidden_dim"]),
        "--num-layers",      str(params["num_layers"]),
        "--dropout",         str(params["dropout"]),
        "--input-dim",       "225",
        "--seq-len",         str(params["seq_len"]),
        "--batch-size",      str(params["batch_size"]),
        "--lr",              effective_lr,
        "--weight-decay",    str(params["weight_decay"]),
        "--label-smoothing", str(params["label_smoothing"]),
        "--epochs",          str(params["epochs"]),
        "--patience",        str(params["patience"]),
        "--num-workers",     str(params["num_workers"]),
        "--normalize",       "true",
        "--augment-train",   "true",
    ]

    if decision == "finetune" and base_ckpt:
        cmd += ["--base-ckpt-uri", base_ckpt, "--finetune-lr", str(params["finetune_lr"])]

    run_streaming(cmd, cwd=PROJECT_ROOT)

    # ── Retrieve completed run_id from the output file written by the subprocess ──
    try:
        with open(run_id_file) as f:
            mlflow_run_id = f.read().strip()
    except FileNotFoundError:
        raise RuntimeError(
            f"Training subprocess did not write run_id to {run_id_file}. "
            "Check subprocess logs for errors."
        )

    if not mlflow_run_id:
        raise RuntimeError(f"run_id file {run_id_file} is empty")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient(MLFLOW_TRACKING_URI)
    run    = client.get_run(mlflow_run_id)
    best_val_acc = run.data.metrics.get("best_val_acc", 0.0)

    print(f"Training complete — run_id={mlflow_run_id}, best_val_acc={best_val_acc:.4f}")
    ti.xcom_push(key="mlflow_run_id", value=mlflow_run_id)
    ti.xcom_push(key="best_val_acc",  value=best_val_acc)

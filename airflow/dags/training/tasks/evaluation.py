"""
Task: evaluate model.

Calls evaluate_mlflow.py as a subprocess against the test split.
Logs classification report back to the MLflow run and pushes test_acc to XCom.
Gracefully skips when no test split exists.
"""

import os
import json
from pathlib import Path

from airflow.dags.training.config import (
    PROJECT_ROOT,
    MLFLOW_TRACKING_URI,
)
from airflow.dags.training.utils import run_streaming, ensure_run_context

from src.config.config import SEQ_LEN


def evaluate_model(**context) -> None:
    """
    XCom outputs:
        test_acc    : float | None
        eval_report : str   path to eval_report.json (or None)
    """
    ti            = context["task_instance"]
    mlflow_run_id = ti.xcom_pull(task_ids="train_model",   key="mlflow_run_id")
    split_dir     = ti.xcom_pull(task_ids="split_dataset", key="split_dir")
    run_ctx       = ensure_run_context(ti)

    test_dir = os.path.join(split_dir, "test")
    if not os.path.isdir(test_dir) or not any(Path(test_dir).rglob("*.npy")):
        print("No test split available — skipping evaluation.")
        ti.xcom_push(key="test_acc",    value=None)
        ti.xcom_push(key="eval_report", value=None)
        return

    eval_out = os.path.join(run_ctx["run_dir"], "eval_report.json")

    run_streaming(
        [
            "python", "-u",
            f"{PROJECT_ROOT}/src/pipeline/evaluate_mlflow.py",
            "--tracking-uri", MLFLOW_TRACKING_URI,
            "--run-id",       mlflow_run_id,
            "--split-dir",    split_dir,
            "--seq-len",      str(SEQ_LEN),
            "--output",       eval_out,
        ],
        cwd=PROJECT_ROOT,
    )

    test_acc = None
    if os.path.exists(eval_out):
        with open(eval_out) as f:
            report = json.load(f)
        test_acc = report.get("test_acc")
        print(f"Test accuracy: {test_acc:.4f}" if test_acc is not None else "test_acc not found")

    ti.xcom_push(key="test_acc",    value=test_acc)
    ti.xcom_push(key="eval_report", value=eval_out)

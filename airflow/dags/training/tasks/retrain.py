"""
Tasks: retrain decision logic.

retrain_check   — compares the current Gold snapshot against the Production
                  MLflow model and pushes one of: "skip" | "finetune" | "full"
branch_on_decision — BranchPythonOperator callable that routes the DAG
"""

import json

import mlflow
from mlflow.tracking import MlflowClient

from training.config import (
    ICEBERG_GOLD_TRAINING_TABLE,
    MLFLOW_TRACKING_URI,
    MLFLOW_MODEL_NAME,
    MIN_NEW_SAMPLES,
)
from training.utils import (
    build_spark_cmd,
    spark_query,
    ensure_run_context,
)


# ── Helpers ──────────────────────────────────────────────────────────

def _fetch_snapshot_stats(gold_version: str, output_file: str) -> dict:
    """Call Spark to get label list + total file count for a Gold snapshot."""
    cmd = build_spark_cmd(
        "snapshot_stats",
        [
            "--table",        ICEBERG_GOLD_TRAINING_TABLE,
            "--gold-version", gold_version,
            "--output-file",  output_file,
        ],
    )
    spark_query(cmd, ICEBERG_GOLD_TRAINING_TABLE)

    try:
        with open(output_file) as f:
            return json.load(f)
    except Exception:
        return {"labels": [], "total_files": 0}


def _fetch_production_model(client: MlflowClient) -> dict | None:
    """Return metadata for the current Production model version, or None."""
    try:
        versions = client.get_latest_versions(MLFLOW_MODEL_NAME, stages=["Production"])
    except Exception as exc:
        print(f"Could not query MLflow registry: {exc}")
        return None

    if not versions:
        return None

    mv     = versions[0]
    run    = client.get_run(mv.run_id)
    params = run.data.params

    label_map_json = params.get("label_map", "[]")
    return {
        "run_id":   mv.run_id,
        "version":  mv.version,
        "labels":   set(json.loads(label_map_json)),
        "total":    int(params.get("train_samples", 0)),
        "ckpt_uri": mv.source,
    }


# ── Task callables ───────────────────────────────────────────────────

def retrain_check(**context) -> None:
    """
    XCom outputs (all under task_id="training_retrain_check"):
        decision        : "skip" | "finetune" | "full"
        reason          : str
        gold_version    : str
        gold_prefix     : str
        base_run_id     : str | None
        base_ckpt_uri   : str | None
        curr_label_count: int
        curr_total      : int
    """
    ti           = context["task_instance"]
    gold_version = ti.xcom_pull(task_ids="iceberg_gold_sensor", key="gold_version")
    gold_prefix  = ti.xcom_pull(task_ids="iceberg_gold_sensor", key="gold_prefix")

    run_ctx       = ensure_run_context(ti)
    snapshot_file = f"{run_ctx['run_dir']}/gold_snapshot_stats.json"

    stats        = _fetch_snapshot_stats(gold_version, snapshot_file)
    curr_labels  = set(stats.get("labels", []))
    curr_total   = int(stats.get("total_files", 0))

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client   = MlflowClient(MLFLOW_TRACKING_URI)
    prod     = _fetch_production_model(client)

    def _push(decision, reason, base_run_id=None, base_ckpt_uri=None):
        ti.xcom_push(key="decision",         value=decision)
        ti.xcom_push(key="reason",           value=reason)
        ti.xcom_push(key="gold_version",     value=gold_version)
        ti.xcom_push(key="gold_prefix",      value=gold_prefix)
        ti.xcom_push(key="base_run_id",      value=base_run_id)
        ti.xcom_push(key="base_ckpt_uri",    value=base_ckpt_uri)
        ti.xcom_push(key="curr_label_count", value=len(curr_labels))
        ti.xcom_push(key="curr_total",       value=curr_total)
        print(f"Decision: {decision.upper()} — {reason}")

    if prod is None:
        _push("full", "No Production model in registry — training from scratch.")
        return

    new_labels    = sorted(curr_labels - prod["labels"])
    added_samples = curr_total - prod["total"]

    if new_labels:
        _push(
            "full",
            f"New labels detected: {new_labels}. Full retrain required.",
            base_run_id=prod["run_id"],
            base_ckpt_uri=prod["ckpt_uri"],
        )
        return

    if added_samples < MIN_NEW_SAMPLES:
        _push(
            "skip",
            f"Only {added_samples} new samples (threshold {MIN_NEW_SAMPLES}). Skipping.",
            base_run_id=prod["run_id"],
            base_ckpt_uri=prod["ckpt_uri"],
        )
        return

    _push(
        "finetune",
        f"{added_samples} new samples ({prod['total']} → {curr_total}). Fine-tuning.",
        base_run_id=prod["run_id"],
        base_ckpt_uri=prod["ckpt_uri"],
    )


def branch_on_decision(**context) -> str:
    """
    BranchPythonOperator callable.
    Returns the task_id to execute next.
    """
    decision = context["task_instance"].xcom_pull(
        task_ids="training_retrain_check", key="decision"
    )
    return "skip_training" if decision == "skip" else "download_gold_snapshot"

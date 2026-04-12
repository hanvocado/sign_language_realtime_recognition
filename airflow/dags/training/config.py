"""
Training pipeline configuration.

All constants are read from environment variables with sensible defaults.
Import this module everywhere instead of re-reading os.environ inline.
"""

import os
from datetime import datetime, timedelta

from src.config.config import DATASET_NAME, SEQ_LEN
from airflow.dags.preprocessing_pipeline import task_failure_email_alert

# ── MinIO ────────────────────────────────────────────────────────────
AWS_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY_ID",     "minioadmin")
AWS_SECRET_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", "minioadmin")
MINIO_BUCKET   = os.environ.get("MINIO_BUCKET",          "data")
MINIO_ENDPOINT = (
    os.environ.get("MINIO_ENDPOINT", "minio:9000")
    .replace("http://", "")
    .replace("https://", "")
)

# ── Paths ────────────────────────────────────────────────────────────
PROJECT_ROOT = os.environ.get("PYTHONPATH", "/opt/airflow/project")

MINIO_GOLD_ROOT_PREFIX = os.environ.get(
    "MINIO_GOLD_ROOT_PREFIX",
    f"lakehouse/gold/{DATASET_NAME}/npy/{SEQ_LEN}",
).strip("/")

GOLD_VERSION_STATE_PATH = os.environ.get(
    "GOLD_VERSION_STATE_PATH",
    f"{PROJECT_ROOT}/data/{DATASET_NAME}/gold_version_state.json",
)
LOCAL_TRAINING_DIR = f"{PROJECT_ROOT}/data/{DATASET_NAME}/training"
LOCAL_SPLIT_DIR    = f"{PROJECT_ROOT}/data/{DATASET_NAME}/split"

# ── Iceberg / Spark ──────────────────────────────────────────────────
ICEBERG_GOLD_TRAINING_TABLE = os.environ.get(
    "ICEBERG_GOLD_TRAINING_TABLE", "gold_training_landmarks_inventory"
)
ICEBERG_NAMESPACE    = os.environ.get("ICEBERG_NAMESPACE",  "sign_language")
SPARK_MASTER_URL     = os.environ.get("SPARK_MASTER_URL",   "spark://spark-master:7077")
SPARK_ICEBERG_JOB    = f"{PROJECT_ROOT}/src/pipeline/spark_iceberg_inventory.py"
SPARK_ICEBERG_PACKAGES = os.environ.get(
    "SPARK_ICEBERG_PACKAGES",
    "org.apache.iceberg:iceberg-spark-runtime-3.5_2.12:1.5.2,"
    "org.postgresql:postgresql:42.7.3,"
    "org.apache.hadoop:hadoop-aws:3.3.4,"
    "com.amazonaws:aws-java-sdk-bundle:1.12.262",
)

# ── MLflow ───────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MLFLOW_EXPERIMENT   = os.environ.get("MLFLOW_EXPERIMENT",   "slr_experiment")
MLFLOW_MODEL_NAME   = os.environ.get("MLFLOW_MODEL_NAME",   "SLR_model")

# ── Model hyperparameters ────────────────────────────────────────────
MODEL_TYPE      = os.environ.get("SLR_MODEL_TYPE",           "bilstm")
HIDDEN_DIM      = int(os.environ.get("SLR_HIDDEN_DIM",       "256"))
NUM_LAYERS      = int(os.environ.get("SLR_NUM_LAYERS",        "2"))
DROPOUT         = float(os.environ.get("SLR_DROPOUT",         "0.3"))
BATCH_SIZE      = int(os.environ.get("SLR_BATCH_SIZE",        "16"))
LR              = float(os.environ.get("SLR_LR",              "0.001"))
FINETUNE_LR     = float(os.environ.get("SLR_FINETUNE_LR",     "0.0003"))
WEIGHT_DECAY    = float(os.environ.get("SLR_WEIGHT_DECAY",    "0.001"))
LABEL_SMOOTHING = float(os.environ.get("SLR_LABEL_SMOOTHING", "0.1"))
EPOCHS          = int(os.environ.get("SLR_EPOCHS",            "200"))
PATIENCE        = int(os.environ.get("SLR_PATIENCE",          "20"))
NUM_WORKERS     = int(os.environ.get("SLR_NUM_WORKERS",        "0"))

# ── Pipeline thresholds ──────────────────────────────────────────────
MIN_NEW_SAMPLES = int(os.environ.get("SLR_MIN_NEW_SAMPLES",   "20"))
PROMOTE_MIN_ACC = float(os.environ.get("SLR_PROMOTE_MIN_ACC", "0.70"))

# ── Airflow default_args ─────────────────────────────────────────────
DEFAULT_ARGS = {
    "owner":               "airflow",
    "start_date":          datetime(2026, 1, 1),
    "depends_on_past":     False,
    "email_on_failure":    False,
    "email_on_retry":      False,
    "retries":             1,
    "retry_delay":         timedelta(minutes=5),
    "on_failure_callback": task_failure_email_alert,
}

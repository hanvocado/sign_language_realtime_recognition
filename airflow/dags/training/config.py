"""
Training pipeline configuration.
"""

import os

from src.config.config import (
    DATASET_NAME, SEQ_LEN,
    MODEL_TYPE, HIDDEN_DIM, NUM_LAYERS, DROPOUT, BATCH_SIZE, LR, FINETUNE_LR,
    WEIGHT_DECAY, LABEL_SMOOTHING, EPOCHS, PATIENCE, NUM_WORKERS
)
from preprocessing_pipeline import task_failure_email_alert

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

TRAINING_SENSOR_STATE_PATH = os.environ.get(
    "TRAINING_SENSOR_STATE_PATH",
    f"{PROJECT_ROOT}/data/{DATASET_NAME}/training_sensor_state.json",
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
MODEL_TYPE      = os.environ.get("SLR_MODEL_TYPE",           MODEL_TYPE)
HIDDEN_DIM      = int(os.environ.get("SLR_HIDDEN_DIM",       HIDDEN_DIM))
NUM_LAYERS      = int(os.environ.get("SLR_NUM_LAYERS",        NUM_LAYERS))
DROPOUT         = float(os.environ.get("SLR_DROPOUT",         DROPOUT))
BATCH_SIZE      = int(os.environ.get("SLR_BATCH_SIZE",        BATCH_SIZE))
LR              = float(os.environ.get("SLR_LR",              LR))
FINETUNE_LR     = float(os.environ.get("SLR_FINETUNE_LR",     FINETUNE_LR))
WEIGHT_DECAY    = float(os.environ.get("SLR_WEIGHT_DECAY",    WEIGHT_DECAY))
LABEL_SMOOTHING = float(os.environ.get("SLR_LABEL_SMOOTHING", LABEL_SMOOTHING))
EPOCHS          = int(os.environ.get("SLR_EPOCHS",            EPOCHS))
PATIENCE        = int(os.environ.get("SLR_PATIENCE",          PATIENCE))
NUM_WORKERS     = int(os.environ.get("SLR_NUM_WORKERS",        NUM_WORKERS))

# ── Pipeline thresholds ──────────────────────────────────────────────
MIN_NEW_SAMPLES = int(os.environ.get("SLR_MIN_NEW_SAMPLES",   "20"))
PROMOTE_MIN_ACC = float(os.environ.get("SLR_PROMOTE_MIN_ACC", "0.20"))

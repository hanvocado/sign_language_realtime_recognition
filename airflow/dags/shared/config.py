"""
Shared configuration for all Airflow DAGs.

All constants are read from environment variables with sensible defaults.
Import from here instead of re-reading os.environ in each DAG.
"""

import os
from datetime import datetime, timedelta

from minio import Minio

from src.config.config import DATASET_NAME, SEQ_LEN

# ── MinIO ────────────────────────────────────────────────────────────

AWS_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY_ID", "minioadmin")
AWS_SECRET_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", "minioadmin")
MINIO_BUCKET = os.environ.get("MINIO_BUCKET", "data")

_raw_endpoint = os.environ.get("MINIO_ENDPOINT", "minio:9000")
MINIO_ENDPOINT = _raw_endpoint.replace("http://", "", 1).replace("https://", "", 1)

_minio_secure_env = os.environ.get("MINIO_SECURE")
if _minio_secure_env is not None:
    MINIO_SECURE = _minio_secure_env.lower() in ("1", "true", "yes")
else:
    MINIO_SECURE = _raw_endpoint.startswith("https://")

minio_client = Minio(
    MINIO_ENDPOINT,
    access_key=AWS_ACCESS_KEY,
    secret_key=AWS_SECRET_KEY,
    secure=MINIO_SECURE,
)

# ── Paths ────────────────────────────────────────────────────────────

PROJECT_ROOT = os.environ.get("PYTHONPATH", "/opt/airflow/project")

# ── MinIO prefixes (Medallion) ───────────────────────────────────────

MINIO_BRONZE_RAW_PREFIX = os.environ.get(
    "MINIO_BRONZE_RAW_PREFIX", f"lakehouse/bronze/{DATASET_NAME}/videos/raw"
).strip("/")
MINIO_USER_UPLOAD_PREFIX = os.environ.get(
    "MINIO_USER_UPLOAD_PREFIX", f"lakehouse/bronze/{DATASET_NAME}/videos/user_uploads"
).strip("/")
MINIO_UPLOAD_PREFIX = os.environ.get(
    "MINIO_UPLOAD_PREFIX", MINIO_USER_UPLOAD_PREFIX
).strip("/")
MINIO_SILVER_PREFIX = os.environ.get(
    "MINIO_SILVER_PREFIX", f"lakehouse/silver/{DATASET_NAME}/videos/preprocessed"
).strip("/")
MINIO_GOLD_ROOT_PREFIX = os.environ.get(
    "MINIO_GOLD_ROOT_PREFIX", f"lakehouse/gold/{DATASET_NAME}/npy/{SEQ_LEN}"
).strip("/")

# ── Iceberg / Spark ──────────────────────────────────────────────────

ICEBERG_NAMESPACE = os.environ.get("ICEBERG_NAMESPACE", "sign_language")
ICEBERG_BRONZE_TABLE = os.environ.get("ICEBERG_BRONZE_TABLE", "bronze_user_upload_inventory")
ICEBERG_SILVER_RAW_TABLE = os.environ.get("ICEBERG_SILVER_RAW_TABLE", "silver_raw_inventory")
ICEBERG_GOLD_TRAINING_TABLE = os.environ.get(
    "ICEBERG_GOLD_TRAINING_TABLE", "gold_training_landmarks_inventory"
)

SPARK_MASTER_URL = os.environ.get("SPARK_MASTER_URL", "spark://spark-master:7077")
SPARK_ICEBERG_JOB = f"{PROJECT_ROOT}/src/pipeline/spark_iceberg_inventory.py"
SPARK_ICEBERG_PACKAGES = os.environ.get(
    "SPARK_ICEBERG_PACKAGES",
    "org.apache.iceberg:iceberg-spark-runtime-3.5_2.12:1.5.2,"
    "org.postgresql:postgresql:42.7.3,"
    "org.apache.hadoop:hadoop-aws:3.3.4,"
    "com.amazonaws:aws-java-sdk-bundle:1.12.262",
)

# ── Local dirs (preprocessing) ───────────────────────────────────────

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm")
LOCAL_RAW_SOURCE_DIR = f"{PROJECT_ROOT}/data/{DATASET_NAME}/videos/raw"
LOCAL_PREPROCESSED_DIR = f"{PROJECT_ROOT}/data/{DATASET_NAME}/videos/preprocessed"
LOCAL_NPY_DIR = f"{PROJECT_ROOT}/data/{DATASET_NAME}/npy/{SEQ_LEN}"
PREPROCESS_INPUT_STAGING_DIR = f"{PROJECT_ROOT}/data/{DATASET_NAME}/videos/staging"
GOLD_VERSION_STATE_PATH = os.environ.get(
    "GOLD_VERSION_STATE_PATH", f"{PROJECT_ROOT}/data/{DATASET_NAME}/gold_version_state.json"
)

# ── Airflow default_args ─────────────────────────────────────────────

# Import here (not at top) to avoid circular dependency:
# shared.alerts imports airflow.utils.email which is fine,
# but we keep the lazy import style explicit.
from shared.alerts import task_failure_email_alert

DEFAULT_ARGS = {
    "owner": "airflow",
    "start_date": datetime(2026, 1, 1),
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "on_failure_callback": task_failure_email_alert,
}

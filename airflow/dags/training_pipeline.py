"""
Sign Language Recognition - Training Pipeline DAG

Pipeline:
1. read_landmarks: Load preprocessed landmarks from MinIO/local storage
2. augmentation: Configure data augmentation params
3. normalization: Configure normalization params
4. train_model: Train model using train/val splits
5. evaluate_model: Evaluate on test split
6. model_tracking: Log metrics + model to MLflow
7. model_registry: Register model in MLflow registry (Production stage)
8. deployment: Update production manifest for Flask to consume
"""

import os
import json
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.exceptions import AirflowException

# MLflow
import mlflow
from mlflow.client import MlflowClient

# MinIO
from minio import Minio

# ================================================================
# Configuration
# ================================================================

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
AWS_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY_ID", "minioadmin")
AWS_SECRET_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", "minioadmin")
MINIO_S3_ENDPOINT = os.environ.get("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000")
PROJECT_ROOT = os.environ.get("PYTHONPATH", "/opt/airflow/project")
DATASET_VERSION = os.environ.get("DATASET_VERSION", "v1").strip("/")
ICEBERG_LANDMARKS_TABLE = os.environ.get("ICEBERG_GOLD_TRAINING_TABLE", "gold_training_landmarks_inventory")

MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "minio:9000")
if MINIO_ENDPOINT.startswith("http://"):
    MINIO_ENDPOINT = MINIO_ENDPOINT.replace("http://", "", 1)
elif MINIO_ENDPOINT.startswith("https://"):
    MINIO_ENDPOINT = MINIO_ENDPOINT.replace("https://", "", 1)

minio_client = Minio(
    MINIO_ENDPOINT,
    access_key=AWS_ACCESS_KEY,
    secret_key=AWS_SECRET_KEY,
    secure=False,
)

# Set MLflow tracking URI
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
MLFLOW_EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT", "sign_language_training")
mlflow.set_experiment(MLFLOW_EXPERIMENT)

# Default Airflow arguments
default_args = {
    "owner": "airflow",
    "start_date": datetime(2026, 1, 1),
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# ================================================================
# Task Functions
# ================================================================

def prepare_training_run(**context):
    """Prepare training run directory"""
    run_dir = f"{PROJECT_ROOT}/models/checkpoints/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(run_dir, exist_ok=True)
    
    print(f"✅ Training run directory created: {run_dir}")
    context["task_instance"].xcom_push(key="run_dir", value=run_dir)

def read_landmarks(**context):
    """Read landmarks from MinIO and prepare local train/val/test dataset"""
    import subprocess
    from pathlib import Path

    run_dir = context["task_instance"].xcom_pull(task_ids="prepare_training_run", key="run_dir")
    dag_conf = context.get("dag_run").conf if context.get("dag_run") else {}
    spark_master = os.environ.get("SPARK_MASTER_URL", "spark://spark-master:7077")
    spark_packages = os.environ.get(
        "SPARK_ICEBERG_PACKAGES",
        "org.apache.iceberg:iceberg-spark-runtime-3.5_2.12:1.5.2,org.postgresql:postgresql:42.7.3,org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.262",
    )
    spark_job = f"{PROJECT_ROOT}/src/pipeline/spark_iceberg_inventory.py"

    # Resolve run_id/prefix from dag conf, Iceberg inventory, or preprocessing manifest
    run_id = dag_conf.get("run_id")
    minio_bucket = dag_conf.get("minio_bucket", "data")
    minio_prefix = dag_conf.get("minio_prefix")

    iceberg_rows = []
    if run_id:
        try:
            iceberg_rows_file = f"{run_dir}/landmarks_inventory_{run_id}.json"
            rows_cmd = [
                "spark-submit",
                "--master",
                spark_master,
                "--packages",
                spark_packages,
                spark_job,
                "rows-by-run",
                "--table",
                ICEBERG_LANDMARKS_TABLE,
                "--run-id",
                run_id,
                "--output-file",
                iceberg_rows_file,
            ]
            rows_result = subprocess.run(rows_cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
            if rows_result.returncode == 0 and os.path.exists(iceberg_rows_file):
                with open(iceberg_rows_file, "r", encoding="utf-8") as f:
                    iceberg_rows = json.load(f)
        except Exception:
            iceberg_rows = []
    else:
        try:
            latest_cmd = [
                "spark-submit",
                "--master",
                spark_master,
                "--packages",
                spark_packages,
                spark_job,
                "latest-run",
                "--table",
                ICEBERG_LANDMARKS_TABLE,
            ]
            latest_result = subprocess.run(latest_cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
            latest_run_id = latest_result.stdout.strip() if latest_result.returncode == 0 else None
            if latest_run_id:
                run_id = latest_run_id
                iceberg_rows_file = f"{run_dir}/landmarks_inventory_{run_id}.json"
                rows_cmd = [
                    "spark-submit",
                    "--master",
                    spark_master,
                    "--packages",
                    spark_packages,
                    spark_job,
                    "rows-by-run",
                    "--table",
                    ICEBERG_LANDMARKS_TABLE,
                    "--run-id",
                    run_id,
                    "--output-file",
                    iceberg_rows_file,
                ]
                rows_result = subprocess.run(rows_cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
                if rows_result.returncode == 0 and os.path.exists(iceberg_rows_file):
                    with open(iceberg_rows_file, "r", encoding="utf-8") as f:
                        iceberg_rows = json.load(f)
        except Exception:
            iceberg_rows = []

    if not minio_prefix and not iceberg_rows:
        manifest_path = f"{PROJECT_ROOT}/data/preprocessing_manifest.json"
        if not os.path.exists(manifest_path):
            raise AirflowException(
                "No preprocessing manifest found and no minio_prefix provided in DAG config"
            )
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        run_id = run_id or manifest.get("run_id")
        minio_bucket = manifest.get("minio_bucket", minio_bucket)
        minio_prefix = manifest.get("gold_minio_prefix") or manifest.get("minio_prefix")

    if not minio_prefix and not iceberg_rows:
        raise AirflowException("Unable to resolve landmarks prefix from DAG config or manifest")

    # Download landmarks objects from MinIO
    raw_landmarks_dir = f"{run_dir}/landmarks_raw"
    split_landmarks_dir = f"{run_dir}/landmarks_split"
    os.makedirs(raw_landmarks_dir, exist_ok=True)

    downloaded_files = 0
    if iceberg_rows:
        # Delta/file-level pull using Iceberg inventory
        for row in iceberg_rows:
            object_name = row.get("object_path")
            if not object_name or not object_name.endswith(".npy"):
                continue

            if minio_prefix and object_name.startswith(minio_prefix):
                rel_path = object_name[len(minio_prefix):].lstrip("/")
            elif minio_prefix:
                rel_path = object_name.split("/", 3)[-1]
            else:
                # landmarks/<run_id>/<label>/<file>.npy
                rel_path = object_name.split("/", 2)[-1]

            local_path = os.path.join(raw_landmarks_dir, rel_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            minio_client.fget_object(minio_bucket, object_name, local_path)
            downloaded_files += 1
    else:
        # Fallback to prefix-based full pull
        for obj in minio_client.list_objects(minio_bucket, prefix=minio_prefix, recursive=True):
            if not obj.object_name.endswith(".npy"):
                continue

            rel_path = obj.object_name[len(minio_prefix):].lstrip("/")
            local_path = os.path.join(raw_landmarks_dir, rel_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            minio_client.fget_object(minio_bucket, obj.object_name, local_path)
            downloaded_files += 1

    if downloaded_files == 0:
        raise AirflowException(
            f"No landmark objects downloaded from MinIO bucket={minio_bucket}, prefix={minio_prefix}"
        )

    # Build train/val/test folder structure from downloaded landmarks
    split_cmd = [
        "python",
        f"{PROJECT_ROOT}/src/preprocess/split_dataset.py",
        "--data_dir",
        raw_landmarks_dir,
        "--output_dir",
        split_landmarks_dir,
        "--file_type",
        "npy",
        "--train_ratio",
        "0.7",
        "--val_ratio",
        "0.15",
        "--seed",
        "42",
    ]
    result = subprocess.run(split_cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        raise AirflowException(f"Split dataset failed: {result.stderr}")

    # Sanity check
    required_dirs = [
        os.path.join(split_landmarks_dir, "train"),
        os.path.join(split_landmarks_dir, "val"),
        os.path.join(split_landmarks_dir, "test"),
    ]
    for req_dir in required_dirs:
        if not os.path.isdir(req_dir):
            raise AirflowException(f"Missing split directory: {req_dir}")

    source_mode = "iceberg-delta" if iceberg_rows else "prefix-full"
    print(
        f"✅ Downloaded {downloaded_files} landmark files from MinIO "
        f"(bucket={minio_bucket}, prefix={minio_prefix}, mode={source_mode})"
    )
    context["task_instance"].xcom_push(key="run_id", value=run_id)
    context["task_instance"].xcom_push(key="minio_bucket", value=minio_bucket)
    context["task_instance"].xcom_push(key="minio_prefix", value=minio_prefix)
    context["task_instance"].xcom_push(key="landmarks_count", value=downloaded_files)
    context["task_instance"].xcom_push(key="landmarks_dir", value=split_landmarks_dir)

def configure_augmentation(**context):
    """Prepare augmentation configuration for training"""
    config = {
        "augmentation_enabled": True,
        "techniques": ["rotation", "scaling", "shift", "flip", "time_mask"],
        "rotation_range": 15,
        "scaling_range": [0.9, 1.1],
        "shift_range": 0.08,
        "flip_prob": 0.5,
        "time_mask_prob": 0.2,
        "time_mask_max": 3,
    }
    
    print(f"✅ Augmentation config: {config}")
    context["task_instance"].xcom_push(key="augmentation_config", value=config)

def configure_normalization(**context):
    """Prepare normalization configuration for training"""
    config = {
        "normalization_enabled": True,
        "method": "standard_scaler",
        "fit_on": "train_split",
    }
    
    print(f"✅ Normalization config: {config}")
    context["task_instance"].xcom_push(key="normalization_config", value=config)

def train_model(**context):
    """Train model using existing train.py"""
    import subprocess
    
    run_dir = context["task_instance"].xcom_pull(task_ids="prepare_training_run", key="run_dir")
    landmarks_dir = context["task_instance"].xcom_pull(task_ids="read_landmarks", key="landmarks_dir")
    
    augmentation_config = context["task_instance"].xcom_pull(
        task_ids="configure_augmentation", key="augmentation_config"
    ) or {}
    normalization_config = context["task_instance"].xcom_pull(
        task_ids="configure_normalization", key="normalization_config"
    ) or {}

    normalize_enabled = str(normalization_config.get("normalization_enabled", True)).lower()
    augment_enabled = str(augmentation_config.get("augmentation_enabled", True)).lower()

    scale_range = augmentation_config.get("scaling_range", [0.9, 1.1])
    if not isinstance(scale_range, (list, tuple)) or len(scale_range) != 2:
        scale_range = [0.9, 1.1]
    
    # Run training script
    result = subprocess.run(
        [
            "python", f"{PROJECT_ROOT}/src/model/train.py",
            "--data_dir", landmarks_dir,
            "--ckpt_dir", run_dir,
            "--model_type", "bilstm",
            "--epochs", "100",
            "--batch_size", "32",
            "--normalize", normalize_enabled,
            "--augment_train", augment_enabled,
            "--rotation_range", str(augmentation_config.get("rotation_range", 15)),
            "--scale_min", str(scale_range[0]),
            "--scale_max", str(scale_range[1]),
            "--shift_range", str(augmentation_config.get("shift_range", 0.08)),
            "--flip_prob", str(augmentation_config.get("flip_prob", 0.5)),
            "--time_mask_prob", str(augmentation_config.get("time_mask_prob", 0.2)),
            "--time_mask_max", str(augmentation_config.get("time_mask_max", 3)),
        ],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT
    )
    
    if result.returncode != 0:
        raise AirflowException(f"Training failed: {result.stderr}")
    
    print(f"✅ Model training complete")
    print(result.stdout)
    
    context["task_instance"].xcom_push(key="train_output", value=result.stdout)
    context["task_instance"].xcom_push(key="model_path", value=f"{run_dir}/best.pth")

def evaluate_model(**context):
    """Evaluate model on test split"""
    import subprocess
    import json
    
    run_dir = context["task_instance"].xcom_pull(task_ids="prepare_training_run", key="run_dir")
    model_path = context["task_instance"].xcom_pull(task_ids="train_model", key="model_path")

    landmarks_dir = context["task_instance"].xcom_pull(task_ids="read_landmarks", key="landmarks_dir")
    test_dir = os.path.join(landmarks_dir, "test")
    
    # Run evaluation script
    result = subprocess.run(
        [
            "python", f"{PROJECT_ROOT}/src/model/eval.py",
            "--model_path", model_path,
            "--test_dir", test_dir,
            "--output_dir", run_dir,
        ],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT
    )
    
    if result.returncode != 0:
        raise AirflowException(f"Evaluation failed: {result.stderr}")
    
    # Parse evaluation results
    eval_file = f"{run_dir}/evaluation_results.json"
    if os.path.exists(eval_file):
        with open(eval_file, "r") as f:
            eval_results = json.load(f)
        print(f"✅ Evaluation results: {eval_results}")
        context["task_instance"].xcom_push(key="eval_results", value=eval_results)
    else:
        print(f"⚠️ No evaluation results file found")

def log_to_mlflow(**context):
    """Log model and metrics to MLflow"""
    import subprocess
    
    run_dir = context["task_instance"].xcom_pull(task_ids="prepare_training_run", key="run_dir")
    model_path = context["task_instance"].xcom_pull(task_ids="train_model", key="model_path")
    eval_results = context["task_instance"].xcom_pull(task_ids="evaluate_model", key="eval_results") or {}
    augmentation_config = context["task_instance"].xcom_pull(task_ids="configure_augmentation", key="augmentation_config")
    normalization_config = context["task_instance"].xcom_pull(task_ids="configure_normalization", key="normalization_config")
    minio_bucket = context["task_instance"].xcom_pull(task_ids="read_landmarks", key="minio_bucket")
    minio_prefix = context["task_instance"].xcom_pull(task_ids="read_landmarks", key="minio_prefix")
    
    with mlflow.start_run() as run:
        # Log parameters
        mlflow.log_params({
            "model_type": "bilstm",
            "augmentation": augmentation_config.get("augmentation_enabled", True),
            "augmentation_config": json.dumps(augmentation_config, ensure_ascii=False),
            "normalization": normalization_config.get("normalization_enabled", True),
            "normalization_config": json.dumps(normalization_config, ensure_ascii=False),
            "dataset_bucket": minio_bucket,
            "dataset_prefix": minio_prefix,
            "dataset_version": DATASET_VERSION,
            "iceberg_gold_training_table": ICEBERG_LANDMARKS_TABLE,
        })
        mlflow.set_tags(
            {
                "medallion_layer": "gold",
                "dataset_version": DATASET_VERSION,
                "data_source": "gold_training_landmarks",
            }
        )
        
        # Log metrics
        for metric_name, metric_value in eval_results.items():
            if isinstance(metric_value, (int, float)):
                mlflow.log_metric(metric_name, metric_value)
        
        # Log model artifact
        mlflow.log_artifact(model_path, artifact_path=f"gold/models/{DATASET_VERSION}")
        
        # Log evaluation results
        eval_file = f"{run_dir}/evaluation_results.json"
        if os.path.exists(eval_file):
            mlflow.log_artifact(eval_file, artifact_path=f"gold/evaluation/{DATASET_VERSION}")
        
        mlflow_run_id = run.info.run_id
    
    print(f"✅ Model logged to MLflow (Run ID: {mlflow_run_id})")
    context["task_instance"].xcom_push(key="mlflow_run_id", value=mlflow_run_id)

def register_model(**context):
    """Register model in MLflow registry (Production stage)"""
    mlflow_run_id = context["task_instance"].xcom_pull(task_ids="log_to_mlflow", key="mlflow_run_id")
    eval_results = context["task_instance"].xcom_pull(task_ids="evaluate_model", key="eval_results") or {}
    
    # Quality gate: check minimum accuracy
    top1_accuracy = eval_results.get("top1_accuracy", 0)
    macro_f1 = eval_results.get("macro_f1", 0)
    
    MIN_TOP1_ACCURACY = 0.70
    MIN_MACRO_F1 = 0.65
    
    if top1_accuracy < MIN_TOP1_ACCURACY or macro_f1 < MIN_MACRO_F1:
        print(f"⚠️ Model did not meet quality gate:")
        print(f"   Top1 Accuracy: {top1_accuracy} < {MIN_TOP1_ACCURACY}")
        print(f"   Macro F1: {macro_f1} < {MIN_MACRO_F1}")
        context["task_instance"].xcom_push(key="quality_gate_pass", value=False)
        return
    
    # Register model
    try:
        client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        
        model_name = os.environ.get("MLFLOW_MODEL_NAME", "sign_language_model")

        # Create registered model once (idempotent)
        try:
            client.get_registered_model(model_name)
        except Exception:
            client.create_registered_model(model_name)

        model_version = client.create_model_version(
            name=model_name,
            source=f"runs:/{mlflow_run_id}/models",
            run_id=mlflow_run_id,
        )
        
        # Transition to Production stage
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Production",
        )
        
        print(f"✅ Model registered (v{model_version.version}) and promoted to Production")
        context["task_instance"].xcom_push(key="quality_gate_pass", value=True)
        context["task_instance"].xcom_push(key="model_version", value=model_version.version)
    except Exception as e:
        print(f"❌ Model registration failed: {e}")
        raise

def update_production_manifest(**context):
    """Update production manifest for Flask to read"""
    quality_gate_pass = context["task_instance"].xcom_pull(task_ids="register_model", key="quality_gate_pass")
    
    if not quality_gate_pass:
        print("Skipping deployment: Model did not pass quality gate")
        return
    
    run_dir = context["task_instance"].xcom_pull(task_ids="prepare_training_run", key="run_dir")
    model_path = context["task_instance"].xcom_pull(task_ids="train_model", key="model_path")
    mlflow_run_id = context["task_instance"].xcom_pull(task_ids="log_to_mlflow", key="mlflow_run_id")
    
    # Convert absolute paths to paths relative to PROJECT_ROOT so the
    # manifest is portable across containers (Airflow vs. webapp).
    project_root = Path(PROJECT_ROOT)
    try:
        rel_model_path = Path(model_path).relative_to(project_root)
        rel_run_dir = Path(run_dir).relative_to(project_root)
    except ValueError:
        # Paths are not under PROJECT_ROOT; store as-is (non-portable across containers)
        import warnings
        warnings.warn(
            f"model_path or run_dir is outside PROJECT_ROOT ({PROJECT_ROOT}). "
            "Storing absolute paths in production.json — these may not resolve in other containers.",
            RuntimeWarning,
        )
        rel_model_path = model_path
        rel_run_dir = run_dir

    manifest = {
        "model_path": str(rel_model_path),
        "run_dir": str(rel_run_dir),
        "mlflow_run_id": mlflow_run_id,
        "mlflow_tracking_uri": MLFLOW_TRACKING_URI,
        "timestamp": datetime.now().isoformat(),
        "status": "production",
    }
    
    manifest_path = f"{PROJECT_ROOT}/models/checkpoints/production.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    
    print(f"✅ Production manifest updated: {manifest_path}")
    print(f"✅ Deployment ready! Flask will load model from: {rel_model_path}")

# ================================================================
# DAG Definition
# ================================================================

with DAG(
    dag_id="training_pipeline",
    default_args=default_args,
    description="Sign Language Model Training: Landmarks → Train → MLflow Registry → Deployment",
    schedule_interval=None,  # Manual trigger
    catchup=False,
    tags=["training", "mlflow", "sign-language"],
) as dag:
    
    prepare = PythonOperator(
        task_id="prepare_training_run",
        python_callable=prepare_training_run,
    )
    
    read = PythonOperator(
        task_id="read_landmarks",
        python_callable=read_landmarks,
    )
    
    augment = PythonOperator(
        task_id="configure_augmentation",
        python_callable=configure_augmentation,
    )
    
    normalize = PythonOperator(
        task_id="configure_normalization",
        python_callable=configure_normalization,
    )
    
    train = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
    )
    
    evaluate = PythonOperator(
        task_id="evaluate_model",
        python_callable=evaluate_model,
    )
    
    tracking = PythonOperator(
        task_id="log_to_mlflow",
        python_callable=log_to_mlflow,
    )
    
    registry = PythonOperator(
        task_id="register_model",
        python_callable=register_model,
    )
    
    deploy = PythonOperator(
        task_id="update_production_manifest",
        python_callable=update_production_manifest,
    )
    
    # Task dependencies
    prepare >> [read, augment, normalize]
    [read, augment, normalize] >> train
    train >> evaluate >> tracking >> registry >> deploy

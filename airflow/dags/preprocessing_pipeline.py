"""
Sign Language Recognition - Preprocessing Pipeline DAG

Pipeline:
1. preprocess: Normalize videos (resize, fps sync)
2. extract_landmarks: Extract MediaPipe keypoints
3. store_minio: Upload raw videos + npy landmarks to MinIO
"""

import os
import json
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

import psycopg2

from airflow import DAG
from airflow.operators.python import PythonOperator
# MinIO
from minio import Minio

# ================================================================
# Configuration
# ================================================================

AWS_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY_ID", "minioadmin")
AWS_SECRET_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", "minioadmin")
MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "minio:9000")
if MINIO_ENDPOINT.startswith("http://"):
    MINIO_ENDPOINT = MINIO_ENDPOINT.replace("http://", "", 1)
elif MINIO_ENDPOINT.startswith("https://"):
    MINIO_ENDPOINT = MINIO_ENDPOINT.replace("https://", "", 1)
PROJECT_ROOT = os.environ.get("PYTHONPATH", "/opt/airflow/project")

# MinIO client
minio_client = Minio(
    MINIO_ENDPOINT, 
    access_key=AWS_ACCESS_KEY, 
    secret_key=AWS_SECRET_KEY, 
    secure=False
)

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

def _list_files(base_dir: str, exts: tuple[str, ...]) -> list[str]:
    files = []
    if not os.path.isdir(base_dir):
        return files
    for root, _, names in os.walk(base_dir):
        for name in names:
            if name.lower().endswith(exts):
                files.append(os.path.relpath(os.path.join(root, name), base_dir))
    files.sort()
    return files


def _run_command_streaming(cmd: list[str], cwd: str | None = None) -> None:
    """Run command and stream stdout/stderr to Airflow logs in real time."""
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    process = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    assert process.stdout is not None
    for line in process.stdout:
        print(line.rstrip())

    process.wait()
    if process.returncode != 0:
        raise Exception(f"Command failed with code {process.returncode}: {' '.join(cmd)}")

def prepare_preprocess_run(**context):
    """Create directory structure for preprocessing"""

    now = datetime.now()
    run_id = now.strftime("%Y%m%d_%H%M%S")
    run_month = now.strftime("%Y%m")
    run_stamp = now.strftime("%Y%m%dT%H%M%S")
    run_dir = f"{PROJECT_ROOT}/data/runs/preprocess_{run_id}"
    
    os.makedirs(run_dir, exist_ok=True)
    print(f"✅ Preprocessing run directory created: {run_dir}")
    
    # Push to XCom for downstream tasks
    context["task_instance"].xcom_push(key="run_dir", value=run_dir)
    context["task_instance"].xcom_push(key="run_id", value=run_id)
    context["task_instance"].xcom_push(key="run_month", value=run_month)
    context["task_instance"].xcom_push(key="run_stamp", value=run_stamp)

def preprocess_videos(**context):
    """Normalize videos using existing preprocessing code"""

    run_dir = context["task_instance"].xcom_pull(task_ids="prepare_preprocess_run", key="run_dir")
    raw_dir = f"{PROJECT_ROOT}/data/raw_unprocessed"
    output_dir = f"{PROJECT_ROOT}/data/raw"

    os.makedirs(output_dir, exist_ok=True)

    video_files = _list_files(raw_dir, (".mp4", ".avi", ".mov", ".mkv", ".webm"))
    print(f"📂 Raw input dir: {raw_dir}")
    print(f"🎬 Found {len(video_files)} raw video file(s)")
    for idx, rel in enumerate(video_files[:10], start=1):
        print(f"   [{idx}] {rel}")
    if len(video_files) > 10:
        print(f"   ... and {len(video_files) - 10} more")

    # Run script with unbuffered output so per-file logs are visible immediately.
    _run_command_streaming(
        [
            "python", "-u", f"{PROJECT_ROOT}/src/preprocess/preprocess_video.py",
            "--input_dir", raw_dir,
            "--output_dir", output_dir,
        ],
        cwd=PROJECT_ROOT,
    )

    print(f"✅ Video preprocessing complete")

def extract_landmarks(**context):
    """Extract MediaPipe landmarks from videos"""

    preprocessed_dir = f"{PROJECT_ROOT}/data/raw"
    landmarks_dir = f"{PROJECT_ROOT}/data/npy"

    os.makedirs(landmarks_dir, exist_ok=True)

    preprocessed_files = _list_files(preprocessed_dir, (".mp4", ".avi", ".mov", ".mkv", ".webm"))
    print(f"📂 Preprocessed input dir: {preprocessed_dir}")
    print(f"🎞️ Found {len(preprocessed_files)} preprocessed video file(s)")
    for idx, rel in enumerate(preprocessed_files[:10], start=1):
        print(f"   [{idx}] {rel}")
    if len(preprocessed_files) > 10:
        print(f"   ... and {len(preprocessed_files) - 10} more")

    _run_command_streaming(
        [
            "python", "-u", f"{PROJECT_ROOT}/src/preprocess/video2npy.py",
            "--input_dir", preprocessed_dir,
            "--output_dir", landmarks_dir,
        ],
        cwd=PROJECT_ROOT,
    )

    print(f"✅ Landmark extraction complete")

    # Push landmark dir path
    context["task_instance"].xcom_push(key="landmarks_dir", value=landmarks_dir)

def store_to_minio(**context):
    """Upload raw videos + landmarks to MinIO and create preprocessing manifest"""
    
    import subprocess

    landmarks_dir = context["task_instance"].xcom_pull(task_ids="extract_landmarks", key="landmarks_dir")
    run_id = context["task_instance"].xcom_pull(task_ids="prepare_preprocess_run", key="run_id")
    run_month = context["task_instance"].xcom_pull(task_ids="prepare_preprocess_run", key="run_month")
    run_stamp = context["task_instance"].xcom_pull(task_ids="prepare_preprocess_run", key="run_stamp")
    if not run_stamp:
        run_stamp = run_id
    run_dir = context["task_instance"].xcom_pull(task_ids="prepare_preprocess_run", key="run_dir")
    raw_dir = f"{PROJECT_ROOT}/data/raw_unprocessed"

    video_exts = (".mp4", ".avi", ".mov", ".mkv", ".webm")
    
    # Create bucket if not exists
    try:
        if not minio_client.bucket_exists("data"):
            minio_client.make_bucket("data")
    except Exception as e:
        print(f"MinIO bucket check failed: {e}")
    
    raw_inventory_rows = []
    landmark_inventory_rows = []

    # Upload raw videos to MinIO
    uploaded_raw_files = 0
    for root, dirs, files in os.walk(raw_dir):
        for file in files:
            if not file.lower().endswith(video_exts):
                continue

            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, raw_dir)
            object_name = f"raw_videos/{run_month}/{run_stamp}/{rel_path}"

            try:
                result = minio_client.fput_object("data", object_name, file_path)
                uploaded_raw_files += 1

                label = rel_path.split(os.sep)[0] if os.sep in rel_path else None
                raw_inventory_rows.append(
                    {
                        "run_id": run_id,
                        "object_path": object_name,
                        "file_type": "raw_video",
                        "label": label,
                        "size_bytes": os.path.getsize(file_path),
                        "etag": result.etag,
                    }
                )
            except Exception as e:
                print(f"Error uploading raw video {file}: {e}")

    print(f"✅ Uploaded {uploaded_raw_files} raw video files to MinIO")

    # Upload landmarks to MinIO
    uploaded_landmark_files = 0
    for root, dirs, files in os.walk(landmarks_dir):
        for file in files:
            if file.endswith(".npy"):
                file_path = os.path.join(root, file)
                # Compute relative path for MinIO object name
                rel_path = os.path.relpath(file_path, landmarks_dir)
                object_name = f"landmarks/{run_month}/{run_stamp}/{rel_path}"
                
                try:
                    result = minio_client.fput_object("data", object_name, file_path)
                    uploaded_landmark_files += 1

                    label = rel_path.split(os.sep)[0] if os.sep in rel_path else None
                    landmark_inventory_rows.append(
                        {
                            "run_id": run_id,
                            "object_path": object_name,
                            "file_type": "landmark_npy",
                            "label": label,
                            "size_bytes": os.path.getsize(file_path),
                            "etag": result.etag,
                        }
                    )
                except Exception as e:
                    print(f"Error uploading {file}: {e}")
    
    print(f"✅ Uploaded {uploaded_landmark_files} landmark files to MinIO")

    # Append inventory to Iceberg using Spark engine
    namespace = os.environ.get("ICEBERG_NAMESPACE", "sign_language")
    spark_master = os.environ.get("SPARK_MASTER_URL", "spark://spark-master:7077")
    spark_packages = os.environ.get(
        "SPARK_ICEBERG_PACKAGES",
        "org.apache.iceberg:iceberg-spark-runtime-3.5_2.12:1.5.2,org.postgresql:postgresql:42.7.3,org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.262",
    )
    spark_job = f"{PROJECT_ROOT}/src/pipeline/spark_iceberg_inventory.py"

    raw_rows_file = f"{run_dir}/raw_inventory_rows.json"
    landmarks_rows_file = f"{run_dir}/landmarks_inventory_rows.json"
    with open(raw_rows_file, "w", encoding="utf-8") as f:
        json.dump(raw_inventory_rows, f, ensure_ascii=False, indent=2)
    with open(landmarks_rows_file, "w", encoding="utf-8") as f:
        json.dump(landmark_inventory_rows, f, ensure_ascii=False, indent=2)

    raw_cmd = [
        "spark-submit",
        "--master",
        spark_master,
        "--packages",
        spark_packages,
        spark_job,
        "append",
        "--table",
        "raw_inventory",
        "--rows-file",
        raw_rows_file,
    ]
    landmark_cmd = [
        "spark-submit",
        "--master",
        spark_master,
        "--packages",
        spark_packages,
        spark_job,
        "append",
        "--table",
        "landmarks_inventory",
        "--rows-file",
        landmarks_rows_file,
    ]

    def _is_missing_iceberg_metadata(result: subprocess.CompletedProcess) -> bool:
        merged = f"{result.stdout}\n{result.stderr}"
        return "FileNotFoundException" in merged and "/metadata/" in merged

    def _repair_catalog_entry(table_name: str) -> None:
        pg_user = os.environ.get("POSTGRES_USER", "postgres")
        pg_password = os.environ.get("POSTGRES_PASSWORD", "postgres")
        pg_host = os.environ.get("ICEBERG_POSTGRES_HOST", "postgres")
        pg_port = int(os.environ.get("ICEBERG_POSTGRES_PORT", "5432"))
        pg_db = os.environ.get("ICEBERG_POSTGRES_DB", "iceberg")

        conn = psycopg2.connect(
            host=pg_host,
            port=pg_port,
            dbname=pg_db,
            user=pg_user,
            password=pg_password,
        )
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM iceberg_tables WHERE table_name = %s",
                    (table_name,),
                )
                deleted = cur.rowcount
            conn.commit()
            print(f"⚠️ Repaired stale Iceberg catalog pointer for {namespace}.{table_name} (deleted_rows={deleted})")
        finally:
            conn.close()

    def _append_with_repair(cmd: list[str], table_name: str) -> subprocess.CompletedProcess:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
        if result.returncode == 0:
            return result

        if _is_missing_iceberg_metadata(result):
            _repair_catalog_entry(table_name)
            retry_result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
            if retry_result.returncode == 0:
                print(f"✅ Spark append recovered after catalog repair for {namespace}.{table_name}")
                return retry_result
            result = retry_result

        out_tail = "\n".join((result.stdout or "").splitlines()[-120:])
        err_tail = "\n".join((result.stderr or "").splitlines()[-120:])
        raise Exception(
            f"Spark Iceberg append failed ({table_name}). "
            f"returncode={result.returncode}\n"
            f"STDOUT (tail):\n{out_tail}\n"
            f"STDERR (tail):\n{err_tail}"
        )

    _append_with_repair(raw_cmd, "raw_inventory")

    _append_with_repair(landmark_cmd, "landmarks_inventory")

    print("✅ Iceberg inventory appended via Spark engine")
    
    # Create preprocessing manifest
    manifest = {
        "run_id": run_id,
        "run_stamp": run_stamp,
        "timestamp": datetime.now().isoformat(),
        "local_raw_dir": raw_dir,
        "minio_raw_prefix": f"raw_videos/{run_month}/{run_stamp}",
        "raw_files_count": uploaded_raw_files,
        "local_landmarks_dir": landmarks_dir,
        "minio_prefix": f"landmarks/{run_month}/{run_stamp}",
        "minio_bucket": "data",
        "landmark_files_count": uploaded_landmark_files,
        "iceberg_tables": {
            "raw": f"{namespace}.raw_inventory",
            "landmarks": f"{namespace}.landmarks_inventory",
        },
    }
    
    manifest_path = f"{PROJECT_ROOT}/data/preprocessing_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    
    print(f"✅ Preprocessing manifest created: {manifest_path}")
    
    context["task_instance"].xcom_push(key="manifest_path", value=manifest_path)

# ================================================================
# DAG Definition
# ================================================================

with DAG(
    dag_id="preprocessing_pipeline",
    default_args=default_args,
    description="Sign Language Preprocessing: Videos → Landmarks → MinIO",
    schedule_interval=None,  # Manual trigger
    catchup=False,
    tags=["preprocessing", "sign-language"],
) as dag:
    
    prepare = PythonOperator(
        task_id="prepare_preprocess_run",
        python_callable=prepare_preprocess_run,
    )
    
    preprocess = PythonOperator(
        task_id="preprocess_videos",
        python_callable=preprocess_videos,
    )
    
    extract = PythonOperator(
        task_id="extract_landmarks",
        python_callable=extract_landmarks,
    )
    
    store = PythonOperator(
        task_id="store_to_minio",
        python_callable=store_to_minio,
    )
    
    # Task dependencies
    prepare >> preprocess >> extract >> store

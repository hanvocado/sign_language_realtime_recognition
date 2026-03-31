"""
Sign Language Recognition - Preprocessing Pipeline DAG

Pipeline:
1. preprocess: Normalize videos (resize, fps sync)
2. extract_landmarks: Extract MediaPipe keypoints
3. store_minio: Upload raw videos + npy landmarks to MinIO
"""

import os
import json
import glob
import subprocess
import shutil
from datetime import datetime, timedelta
from pathlib import Path

import psycopg2

from airflow import DAG
from airflow.operators.python import PythonOperator
# MinIO
from minio import Minio
from minio.commonconfig import CopySource

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
MINIO_BUCKET = os.environ.get("MINIO_BUCKET", "data")
DATASET_VERSION = os.environ.get("DATASET_VERSION", "v1").strip("/")
MINIO_BRONZE_ROOT_PREFIX = os.environ.get(
    "MINIO_BRONZE_ROOT_PREFIX", "lakehouse/bronze"
).strip("/")
MINIO_UPLOAD_PREFIX = os.environ.get(
    "MINIO_UPLOAD_PREFIX", f"{MINIO_BRONZE_ROOT_PREFIX}/user_upload"
).strip("/")
MINIO_BRONZE_LOCAL_PREFIX = os.environ.get(
    "MINIO_BRONZE_LOCAL_PREFIX", f"{MINIO_BRONZE_ROOT_PREFIX}/local_dataset"
).strip("/")
MINIO_LOCAL_DATASET_PREFIX = os.environ.get(
    "MINIO_LOCAL_DATASET_PREFIX", f"lakehouse/silver/preprocessed/{DATASET_VERSION}/raw_videos"
).strip("/")
MINIO_LANDMARKS_PREFIX = os.environ.get(
    "MINIO_LANDMARKS_PREFIX", f"lakehouse/silver/preprocessed/{DATASET_VERSION}/landmarks"
).strip("/")
MINIO_GOLD_TRAINING_PREFIX = os.environ.get(
    "MINIO_GOLD_TRAINING_PREFIX", "lakehouse/gold/training_dataset"
).strip("/")
ICEBERG_BRONZE_TABLE = os.environ.get("ICEBERG_BRONZE_TABLE", "bronze_user_upload_inventory")
ICEBERG_SILVER_RAW_TABLE = os.environ.get("ICEBERG_SILVER_RAW_TABLE", "silver_raw_inventory")
ICEBERG_SILVER_LANDMARKS_TABLE = os.environ.get("ICEBERG_SILVER_LANDMARKS_TABLE", "silver_landmarks_inventory")
ICEBERG_GOLD_TRAINING_TABLE = os.environ.get("ICEBERG_GOLD_TRAINING_TABLE", "gold_training_landmarks_inventory")
GOLD_VERSION_STATE_PATH = os.environ.get(
    "GOLD_VERSION_STATE_PATH", f"{PROJECT_ROOT}/data/gold_version_state.json"
)
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm")
LOCAL_RAW_SOURCE_DIR = f"{PROJECT_ROOT}/data/raw_unprocessed"
PREPROCESS_INPUT_STAGING_DIR = f"{PROJECT_ROOT}/data/raw_stage"

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


def _is_missing_iceberg_metadata_output(stdout: str, stderr: str) -> bool:
    merged = f"{stdout}\n{stderr}"
    return "FileNotFoundException" in merged and "/metadata/" in merged


def _repair_catalog_entry(namespace: str, table_name: str) -> None:
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
        print(
            f"⚠️ Repaired stale Iceberg catalog pointer for {namespace}.{table_name} "
            f"(deleted_rows={deleted})"
        )
    finally:
        conn.close()


def _load_processed_checkpoint_from_iceberg(run_dir: str) -> tuple[set[str], set[str]]:
    """Load processed Bronze checkpoint from Iceberg table (source of truth)."""

    namespace = os.environ.get("ICEBERG_NAMESPACE", "sign_language")
    spark_master = os.environ.get("SPARK_MASTER_URL", "spark://spark-master:7077")
    spark_packages = os.environ.get(
        "SPARK_ICEBERG_PACKAGES",
        "org.apache.iceberg:iceberg-spark-runtime-3.5_2.12:1.5.2,org.postgresql:postgresql:42.7.3,org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.262",
    )
    spark_job = f"{PROJECT_ROOT}/src/pipeline/spark_iceberg_inventory.py"
    checkpoint_file = f"{run_dir}/bronze_processed_checkpoint.json"

    cmd = [
        "spark-submit",
        "--master",
        spark_master,
        "--packages",
        spark_packages,
        spark_job,
        "checkpoint",
        "--table",
        ICEBERG_BRONZE_TABLE,
        "--output-file",
        checkpoint_file,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
    if result.returncode != 0 and _is_missing_iceberg_metadata_output(
        result.stdout or "", result.stderr or ""
    ):
        _repair_catalog_entry(namespace, ICEBERG_BRONZE_TABLE)
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)

    if result.returncode != 0:
        out_tail = "\n".join((result.stdout or "").splitlines()[-80:])
        err_tail = "\n".join((result.stderr or "").splitlines()[-80:])
        raise Exception(
            f"Cannot load Iceberg checkpoint from {namespace}.{ICEBERG_BRONZE_TABLE}. "
            f"returncode={result.returncode}\n"
            f"STDOUT (tail):\n{out_tail}\n"
            f"STDERR (tail):\n{err_tail}"
        )

    if not os.path.exists(checkpoint_file):
        return set(), set()

    with open(checkpoint_file, "r", encoding="utf-8") as f:
        rows = json.load(f)

    processed_objects: set[str] = set()
    processed_etags: set[str] = set()
    for row in rows:
        object_path = (row.get("object_path") or "").strip()
        etag = (row.get("etag") or "").strip('"')
        if object_path:
            processed_objects.add(object_path)
        if etag:
            processed_etags.add(etag)

    print(
        f"✅ Loaded Iceberg checkpoint: object_paths={len(processed_objects)}, etags={len(processed_etags)}"
    )
    return processed_objects, processed_etags


def _load_gold_state() -> dict:
    try:
        with open(GOLD_VERSION_STATE_PATH, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            return {"latest_version": 0}
        payload.setdefault("latest_version", 0)
        return payload
    except Exception:
        return {"latest_version": 0}


def _save_gold_state(state: dict) -> None:
    os.makedirs(os.path.dirname(GOLD_VERSION_STATE_PATH), exist_ok=True)
    with open(GOLD_VERSION_STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

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


def ingest_local_raw_to_bronze(**context):
    """Upload current local raw snapshot into Bronze local_dataset partition."""

    local_raw_dir = LOCAL_RAW_SOURCE_DIR
    if not os.path.isdir(local_raw_dir):
        print(f"ℹ️ Local raw directory does not exist: {local_raw_dir}")
        context["task_instance"].xcom_push(key="local_ingested_count", value=0)
        return

    uploaded = 0
    skipped = 0
    ingest_month = datetime.now().strftime("%Y%m")
    ingest_day = datetime.now().strftime("%Y%m%d")

    try:
        if not minio_client.bucket_exists(MINIO_BUCKET):
            minio_client.make_bucket(MINIO_BUCKET)
    except Exception as exc:
        print(f"⚠️ Cannot validate MinIO bucket {MINIO_BUCKET}: {exc}")

    for rel in _list_files(local_raw_dir, VIDEO_EXTENSIONS):
        src_abs = os.path.join(local_raw_dir, rel)
        rel_path = Path(rel)
        if len(rel_path.parts) < 2:
            skipped += 1
            continue

        label = rel_path.parts[0]
        filename = rel_path.name

        object_name = (
            f"{MINIO_BRONZE_LOCAL_PREFIX}/{ingest_month}/{ingest_day}/{label}/"
            f"{filename}"
        )
        try:
            minio_client.fput_object(MINIO_BUCKET, object_name, src_abs)
            uploaded += 1
        except Exception as exc:
            print(f"⚠️ Failed to ingest local raw file {src_abs}: {exc}")

    print(
        f"✅ Ingested local raw snapshot to Bronze: {uploaded} uploaded, {skipped} skipped"
    )
    context["task_instance"].xcom_push(key="local_ingested_count", value=uploaded)


def sync_pending_bronze_inputs(**context):
    """Sync all pending unprocessed videos from Bronze sources using Iceberg checkpoint."""

    raw_dir = PREPROCESS_INPUT_STAGING_DIR
    shutil.rmtree(raw_dir, ignore_errors=True)
    os.makedirs(raw_dir, exist_ok=True)

    try:
        if not minio_client.bucket_exists(MINIO_BUCKET):
            print(f"ℹ️ MinIO bucket does not exist yet: {MINIO_BUCKET}")
            context["task_instance"].xcom_push(key="new_uploads", value=[])
            context["task_instance"].xcom_push(key="synced_raw_count", value=0)
            context["task_instance"].xcom_push(key="sync_raw_dir", value=raw_dir)
            return
    except Exception as exc:
        raise Exception(f"Cannot verify MinIO bucket {MINIO_BUCKET}: {exc}")

    run_dir = context["task_instance"].xcom_pull(task_ids="prepare_run_context", key="run_dir")
    if not run_dir:
        run_dir = f"{PROJECT_ROOT}/data/runs/preprocess_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(run_dir, exist_ok=True)
    processed_names, processed_etags = _load_processed_checkpoint_from_iceberg(run_dir)

    source_configs = [
        ("local_dataset", f"{MINIO_BRONZE_LOCAL_PREFIX}/"),
        ("user_upload", f"{MINIO_UPLOAD_PREFIX}/"),
    ]

    pending_inputs = []
    skipped = 0
    scanned = 0

    for source_type, prefix in source_configs:
        for obj in minio_client.list_objects(MINIO_BUCKET, prefix=prefix, recursive=True):
            object_name = obj.object_name
            suffix = Path(object_name).suffix.lower()
            if suffix not in VIDEO_EXTENSIONS:
                continue
            scanned += 1

            etag = (obj.etag or "").strip('"')
            # If etag exists, use etag as primary checkpoint key; fallback to object path.
            if (etag and etag in processed_etags) or (not etag and object_name in processed_names):
                skipped += 1
                continue

            rel = object_name[len(prefix):].lstrip("/")
            parts = rel.split("/")
            if len(parts) >= 2:
                label = parts[-2]
            elif len(parts) == 1:
                label = "unknown"
            else:
                continue

            local_filename = f"{source_type}_{'_'.join(parts)}"
            local_abs = os.path.join(raw_dir, label, local_filename)
            os.makedirs(os.path.dirname(local_abs), exist_ok=True)
            minio_client.fget_object(MINIO_BUCKET, object_name, local_abs)

            local_rel = os.path.relpath(local_abs, raw_dir)
            pending_inputs.append(
                {
                    "source_object": object_name,
                    "source_etag": etag,
                    "source_type": source_type,
                    "label": label,
                    "local_input_rel": local_rel,
                    "local_input_abs": local_abs,
                }
            )

    print(
        f"✅ Synced pending Bronze inputs: {len(pending_inputs)} new, {skipped} skipped, {scanned} scanned"
    )

    context["task_instance"].xcom_push(key="new_uploads", value=pending_inputs)
    context["task_instance"].xcom_push(key="synced_raw_count", value=len(pending_inputs))
    context["task_instance"].xcom_push(key="sync_raw_dir", value=raw_dir)


def bronze_prepare_inputs(**context):
    """Single Bronze stage: ingest local snapshot then sync pending Bronze inputs."""
    ingest_local_raw_to_bronze(**context)
    sync_pending_bronze_inputs(**context)

    ti = context["task_instance"]
    pending_inputs = ti.xcom_pull(key="new_uploads") or []

    ti.xcom_push(key="new_uploads", value=pending_inputs)
    ti.xcom_push(key="sync_raw_dir", value=PREPROCESS_INPUT_STAGING_DIR)
    print(f"✅ Bronze pending inputs prepared: {len(pending_inputs)} item(s)")

def preprocess_videos(**context):
    """Normalize videos using existing preprocessing code"""

    new_uploads = context["task_instance"].xcom_pull(
        task_ids="bronze_ingest_data", key="new_uploads"
    ) or []
    if not new_uploads:
        print("ℹ️ No new uploaded videos to preprocess. Skipping preprocessing step.")
        return

    raw_dir = context["task_instance"].xcom_pull(
        task_ids="bronze_ingest_data", key="sync_raw_dir"
    ) or PREPROCESS_INPUT_STAGING_DIR
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
            "--skip_existing",
        ],
        cwd=PROJECT_ROOT,
    )

    # Count Silver raw outputs generated/available for current pending inputs.
    silver_raw_ready_count = 0
    for item in new_uploads:
        local_input_rel = item.get("local_input_rel", "")
        label = item.get("label")
        stem = Path(local_input_rel).stem
        pattern = os.path.join(output_dir, label, f"{stem}_*.mp4")
        segment_files = sorted(glob.glob(pattern))
        fallback_segment = os.path.join(output_dir, label, f"{stem}.mp4")
        if not segment_files and os.path.exists(fallback_segment):
            segment_files = [fallback_segment]
        silver_raw_ready_count += len(segment_files)

    print(f"✅ Silver raw outputs ready: {silver_raw_ready_count} file(s)")
    context["task_instance"].xcom_push(
        key="silver_raw_ready_count", value=silver_raw_ready_count
    )

    print(f"✅ Video preprocessing complete")

def extract_landmarks(**context):
    """Extract MediaPipe landmarks from videos"""

    new_uploads = context["task_instance"].xcom_pull(
        task_ids="bronze_ingest_data", key="new_uploads"
    ) or []
    if not new_uploads:
        print("ℹ️ No new uploaded videos to extract landmarks. Skipping extraction step.")
        context["task_instance"].xcom_push(key="landmarks_dir", value=f"{PROJECT_ROOT}/data/npy")
        return

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
            "--skip_existing",
        ],
        cwd=PROJECT_ROOT,
    )

    # Count Silver landmarks available for current pending inputs.
    silver_landmarks_ready_count = 0
    for item in new_uploads:
        local_input_rel = item.get("local_input_rel", "")
        label = item.get("label")
        stem = Path(local_input_rel).stem
        pattern = os.path.join(preprocessed_dir, label, f"{stem}_*.mp4")
        segment_files = sorted(glob.glob(pattern))
        fallback_segment = os.path.join(preprocessed_dir, label, f"{stem}.mp4")
        if not segment_files and os.path.exists(fallback_segment):
            segment_files = [fallback_segment]

        for file_path in segment_files:
            rel_path = os.path.relpath(file_path, preprocessed_dir)
            rel_npy = str(Path(rel_path).with_suffix(".npy"))
            npy_path = os.path.join(landmarks_dir, rel_npy)
            if os.path.exists(npy_path):
                silver_landmarks_ready_count += 1

    print(f"✅ Silver landmark outputs ready: {silver_landmarks_ready_count} file(s)")
    context["task_instance"].xcom_push(
        key="silver_landmarks_ready_count", value=silver_landmarks_ready_count
    )

    print(f"✅ Landmark extraction complete")

    # Push landmark dir path
    context["task_instance"].xcom_push(key="landmarks_dir", value=landmarks_dir)

def store_to_minio(**context):
    """Upload raw videos + landmarks to MinIO and create preprocessing manifest"""

    landmarks_dir = context["task_instance"].xcom_pull(task_ids="silver_extract_landmarks", key="landmarks_dir")
    run_id = context["task_instance"].xcom_pull(task_ids="prepare_run_context", key="run_id")
    run_month = context["task_instance"].xcom_pull(task_ids="prepare_run_context", key="run_month")
    run_stamp = context["task_instance"].xcom_pull(task_ids="prepare_run_context", key="run_stamp")
    if not run_stamp:
        run_stamp = run_id
    run_dir = context["task_instance"].xcom_pull(task_ids="prepare_run_context", key="run_dir")
    raw_dir = f"{PROJECT_ROOT}/data/raw"
    new_uploads = context["task_instance"].xcom_pull(
        task_ids="bronze_ingest_data", key="new_uploads"
    ) or []

    video_exts = VIDEO_EXTENSIONS
    
    # Create bucket if not exists
    try:
        if not minio_client.bucket_exists(MINIO_BUCKET):
            minio_client.make_bucket(MINIO_BUCKET)
    except Exception as e:
        print(f"MinIO bucket check failed: {e}")
    
    raw_inventory_rows = []
    landmark_inventory_rows = []
    bronze_inventory_rows = []

    # Upload only newly generated preprocessed raw segments
    uploaded_raw_files = 0
    uploaded_landmark_files = 0
    for item in new_uploads:
        local_input_rel = item.get("local_input_rel", "")
        source_object = item.get("source_object")
        source_etag = item.get("source_etag")
        source_type = item.get("source_type", "unknown")
        label = item.get("label")

        stem = Path(local_input_rel).stem
        pattern = os.path.join(raw_dir, label, f"{stem}_*.mp4")
        segment_files = sorted(glob.glob(pattern))
        fallback_segment = os.path.join(raw_dir, label, f"{stem}.mp4")
        if not segment_files and os.path.exists(fallback_segment):
            segment_files = [fallback_segment]

        if not segment_files:
            print(f"⚠️ No preprocessed segment found for upload: {local_input_rel}")

        for file_path in segment_files:
            if not file_path.lower().endswith(video_exts):
                continue

            rel_path = os.path.relpath(file_path, raw_dir)
            object_name = f"{MINIO_LOCAL_DATASET_PREFIX}/{run_month}/{run_stamp}/{rel_path}"

            try:
                result = minio_client.fput_object(MINIO_BUCKET, object_name, file_path)
                uploaded_raw_files += 1

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
                print(f"Error uploading preprocessed raw video {file_path}: {e}")

            rel_npy = str(Path(rel_path).with_suffix(".npy"))
            npy_path = os.path.join(landmarks_dir, rel_npy)
            if not os.path.exists(npy_path):
                print(f"⚠️ Missing landmark file for segment: {npy_path}")
                continue

            landmark_object_name = f"{MINIO_LANDMARKS_PREFIX}/{run_month}/{run_stamp}/{rel_npy}"
            try:
                result = minio_client.fput_object(MINIO_BUCKET, landmark_object_name, npy_path)
                uploaded_landmark_files += 1
                landmark_inventory_rows.append(
                    {
                        "run_id": run_id,
                        "object_path": landmark_object_name,
                        "file_type": "landmark_npy",
                        "label": label,
                        "size_bytes": os.path.getsize(npy_path),
                        "etag": result.etag,
                    }
                )
            except Exception as e:
                print(f"Error uploading landmark file {npy_path}: {e}")

        if source_object:
            bronze_inventory_rows.append(
                {
                    "run_id": run_id,
                    "object_path": source_object,
                    "file_type": f"bronze_{source_type}",
                    "label": label,
                    "size_bytes": os.path.getsize(item.get("local_input_abs")) if item.get("local_input_abs") and os.path.exists(item.get("local_input_abs")) else 0,
                    "etag": source_etag,
                }
            )

    print("✅ Silver publish to MinIO complete")

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
    bronze_rows_file = f"{run_dir}/bronze_inventory_rows.json"
    gold_rows_file = f"{run_dir}/gold_training_inventory_rows.json"
    with open(raw_rows_file, "w", encoding="utf-8") as f:
        json.dump(raw_inventory_rows, f, ensure_ascii=False, indent=2)
    with open(landmarks_rows_file, "w", encoding="utf-8") as f:
        json.dump(landmark_inventory_rows, f, ensure_ascii=False, indent=2)
    with open(bronze_rows_file, "w", encoding="utf-8") as f:
        json.dump(bronze_inventory_rows, f, ensure_ascii=False, indent=2)

    gold_inventory_rows = []
    gold_version = None
    gold_prefix = None
    if uploaded_landmark_files > 0:
        gold_state = _load_gold_state()
        next_version = int(gold_state.get("latest_version", 0)) + 1
        gold_version = f"v{next_version:04d}"
        gold_prefix = f"{MINIO_GOLD_TRAINING_PREFIX}/{gold_version}/landmarks"

        silver_prefix = f"{MINIO_LANDMARKS_PREFIX}/"
        for obj in minio_client.list_objects(MINIO_BUCKET, prefix=silver_prefix, recursive=True):
            if not obj.object_name.endswith(".npy"):
                continue

            rel = obj.object_name[len(silver_prefix):]
            parts = rel.split("/")
            if len(parts) >= 4:
                run_month_part, run_stamp_part, label, *rest = parts
                filename = "_".join(rest)
                gold_object_name = (
                    f"{gold_prefix}/{label}/{run_month_part}_{run_stamp_part}_{filename}"
                )
            elif len(parts) >= 2:
                label = parts[-2]
                filename = parts[-1]
                gold_object_name = f"{gold_prefix}/{label}/{filename}"
            else:
                continue

            minio_client.copy_object(
                MINIO_BUCKET,
                gold_object_name,
                CopySource(MINIO_BUCKET, obj.object_name),
            )

            gold_inventory_rows.append(
                {
                    "run_id": gold_version,
                    "object_path": gold_object_name,
                    "file_type": "gold_training_landmark",
                    "label": label,
                    "size_bytes": obj.size,
                    "etag": (obj.etag or "").strip('"'),
                }
            )

        gold_state["latest_version"] = next_version
        gold_state["last_source_run_id"] = run_id
        gold_state["last_updated_at"] = datetime.now().isoformat()
        _save_gold_state(gold_state)
        print(
            f"✅ Published Gold training dataset snapshot: {gold_version} ({len(gold_inventory_rows)} files)"
        )
    else:
        print("ℹ️ No new landmarks in this run, Gold training dataset version unchanged.")

    with open(gold_rows_file, "w", encoding="utf-8") as f:
        json.dump(gold_inventory_rows, f, ensure_ascii=False, indent=2)

    raw_cmd = [
        "spark-submit",
        "--master",
        spark_master,
        "--packages",
        spark_packages,
        spark_job,
        "append",
        "--table",
        ICEBERG_SILVER_RAW_TABLE,
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
        ICEBERG_SILVER_LANDMARKS_TABLE,
        "--rows-file",
        landmarks_rows_file,
    ]
    bronze_cmd = [
        "spark-submit",
        "--master",
        spark_master,
        "--packages",
        spark_packages,
        spark_job,
        "append",
        "--table",
        ICEBERG_BRONZE_TABLE,
        "--rows-file",
        bronze_rows_file,
    ]
    gold_cmd = [
        "spark-submit",
        "--master",
        spark_master,
        "--packages",
        spark_packages,
        spark_job,
        "append",
        "--table",
        ICEBERG_GOLD_TRAINING_TABLE,
        "--rows-file",
        gold_rows_file,
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

    _append_with_repair(bronze_cmd, ICEBERG_BRONZE_TABLE)
    _append_with_repair(raw_cmd, ICEBERG_SILVER_RAW_TABLE)
    _append_with_repair(landmark_cmd, ICEBERG_SILVER_LANDMARKS_TABLE)
    if gold_inventory_rows:
        _append_with_repair(gold_cmd, ICEBERG_GOLD_TRAINING_TABLE)

    print("✅ Iceberg inventory appended via Spark engine")
    
    # Create preprocessing manifest
    manifest = {
        "run_id": run_id,
        "run_stamp": run_stamp,
        "timestamp": datetime.now().isoformat(),
        "local_raw_dir": raw_dir,
        "minio_raw_prefix": f"{MINIO_LOCAL_DATASET_PREFIX}/{run_month}/{run_stamp}",
        "raw_files_count": uploaded_raw_files,
        "local_landmarks_dir": landmarks_dir,
        "minio_prefix": f"{MINIO_LANDMARKS_PREFIX}/{run_month}/{run_stamp}",
        "minio_bucket": MINIO_BUCKET,
        "landmark_files_count": uploaded_landmark_files,
        "synced_new_uploads": len(new_uploads),
        "dataset_version": DATASET_VERSION,
        "minio_upload_prefix": MINIO_UPLOAD_PREFIX,
        "minio_local_dataset_prefix": MINIO_LOCAL_DATASET_PREFIX,
        "minio_landmarks_prefix": MINIO_LANDMARKS_PREFIX,
        "gold_version": gold_version,
        "gold_minio_prefix": gold_prefix,
        "minio_gold_training_prefix": MINIO_GOLD_TRAINING_PREFIX,
        "iceberg_tables": {
            "bronze": f"{namespace}.{ICEBERG_BRONZE_TABLE}",
            "silver_raw": f"{namespace}.{ICEBERG_SILVER_RAW_TABLE}",
            "silver_landmarks": f"{namespace}.{ICEBERG_SILVER_LANDMARKS_TABLE}",
            "gold_training": f"{namespace}.{ICEBERG_GOLD_TRAINING_TABLE}",
        },
    }
    
    manifest_path = f"{PROJECT_ROOT}/data/preprocessing_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    
    print(f"✅ Preprocessing manifest created: {manifest_path}")
    
    context["task_instance"].xcom_push(key="manifest_path", value=manifest_path)

    # Cleanup transient staging dirs after successful publish.
    shutil.rmtree(PREPROCESS_INPUT_STAGING_DIR, ignore_errors=True)

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
        task_id="prepare_run_context",
        python_callable=prepare_preprocess_run,
    )
    
    preprocess = PythonOperator(
        task_id="silver_preprocess_videos",
        python_callable=preprocess_videos,
    )

    bronze_prepare = PythonOperator(
        task_id="bronze_ingest_data",
        python_callable=bronze_prepare_inputs,
    )
    
    extract = PythonOperator(
        task_id="silver_extract_landmarks",
        python_callable=extract_landmarks,
    )
    
    store = PythonOperator(
        task_id="gold_publish_dataset",
        python_callable=store_to_minio,
    )
    
    # Task dependencies
    prepare >> bronze_prepare >> preprocess >> extract >> store

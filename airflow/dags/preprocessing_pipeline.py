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
import html
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path

import psycopg2

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.task_group import TaskGroup
from airflow.utils.email import send_email
# MinIO
from minio import Minio
from minio.commonconfig import CopySource
from src.config.config import DATASET_NAME, SEQ_LEN

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

ICEBERG_BRONZE_TABLE = os.environ.get("ICEBERG_BRONZE_TABLE", "bronze_user_upload_inventory")
ICEBERG_SILVER_RAW_TABLE = os.environ.get("ICEBERG_SILVER_RAW_TABLE", "silver_raw_inventory")
ICEBERG_GOLD_TRAINING_TABLE = os.environ.get("ICEBERG_GOLD_TRAINING_TABLE", "gold_training_landmarks_inventory")
GOLD_VERSION_STATE_PATH = os.environ.get(
    "GOLD_VERSION_STATE_PATH", f"{PROJECT_ROOT}/data/{DATASET_NAME}/gold_version_state.json"
)
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm")
LOCAL_RAW_SOURCE_DIR = f"{PROJECT_ROOT}/data/{DATASET_NAME}/videos/raw"
LOCAL_PREPROCESSED_DIR = f"{PROJECT_ROOT}/data/{DATASET_NAME}/videos/preprocessed"
LOCAL_NPY_DIR = f"{PROJECT_ROOT}/data/{DATASET_NAME}/npy/{SEQ_LEN}"
PREPROCESS_INPUT_STAGING_DIR = f"{PROJECT_ROOT}/data/{DATASET_NAME}/videos/staging"

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


def task_failure_email_alert(context):
    """Send rich failure alert email with task/run states and processing metrics."""
    recipients_raw = os.environ.get("AIRFLOW_ALERT_EMAIL_TO", "").strip()
    if not recipients_raw:
        print("ℹ️ AIRFLOW_ALERT_EMAIL_TO is empty, skip failure email alert.")
        return

    recipients = [item.strip() for item in recipients_raw.split(",") if item.strip()]
    if not recipients:
        print("ℹ️ No valid alert recipients found, skip failure email alert.")
        return

    dag_id = context.get("dag").dag_id if context.get("dag") else "unknown_dag"
    ti = context.get("task_instance")
    task_id = ti.task_id if ti else "unknown_task"
    task_state = ti.state if ti and ti.state else "unknown"
    run_id = context.get("run_id", "unknown_run")
    logical_date = context.get("logical_date")
    exception_text = context.get("exception")
    exception_text = html.escape(str(exception_text)) if exception_text else "No exception detail"
    log_url = ti.log_url if ti else ""

    dag_run = context.get("dag_run")
    dag_run_state = dag_run.get_state() if dag_run else "unknown"

    state_counts: dict[str, int] = {}
    failed_task_ids: list[str] = []
    if dag_run:
        for task_inst in dag_run.get_task_instances():
            state = task_inst.state or "none"
            state_counts[state] = state_counts.get(state, 0) + 1
            if state == "failed":
                failed_task_ids.append(task_inst.task_id)

    def _safe_xcom(task_ids: str, key: str):
        if not ti:
            return None
        try:
            return ti.xcom_pull(task_ids=task_ids, key=key)
        except Exception:
            return None

    log_tail_text = "Log tail unavailable"
    if ti:
        log_filepath = getattr(ti, "log_filepath", None)
        if log_filepath and os.path.exists(log_filepath):
            try:
                with open(log_filepath, "r", encoding="utf-8", errors="ignore") as log_file:
                    tail_lines = deque(log_file, maxlen=60)
                log_tail_text = "".join(tail_lines).strip() or "Log file is empty"
            except Exception as exc:
                log_tail_text = f"Cannot read log tail: {exc}"

    processing_metrics = {
        "local_ingested_count": _safe_xcom("bronze_ingest_local_raw", "local_ingested_count"),
        "synced_raw_count": _safe_xcom("bronze_collect_unprocessed_inputs", "synced_raw_count"),
        "silver_raw_ready_count": _safe_xcom("silver_preprocess_videos", "silver_raw_ready_count"),
        "silver_landmarks_ready_count": _safe_xcom("gold_extract_landmarks", "silver_landmarks_ready_count"),
        "manifest_path": _safe_xcom("gold_merge_snapshot", "manifest_path"),
    }

    # Normalize values for email readability.
    normalized_metrics = {
        k: ("N/A" if v is None else html.escape(str(v))) for k, v in processing_metrics.items()
    }

    failed_tasks_text = ", ".join(failed_task_ids) if failed_task_ids else "N/A"
    state_counts_text = html.escape(json.dumps(state_counts, ensure_ascii=False)) if state_counts else "{}"
    log_tail_html = html.escape(log_tail_text)

    subject = f"[Airflow][FAILED] {dag_id}.{task_id}"
    html_content = (
        "<h3>Airflow Task Failed</h3>"
        f"<p><b>DAG:</b> {dag_id}</p>"
        f"<p><b>Task:</b> {task_id}</p>"
        f"<p><b>Task State:</b> {task_state}</p>"
        f"<p><b>Run ID:</b> {run_id}</p>"
        f"<p><b>DAG Run State:</b> {dag_run_state}</p>"
        f"<p><b>Logical Date:</b> {logical_date}</p>"
        f"<p><b>Exception:</b> {exception_text}</p>"
        f"<p><b>Failed Tasks (current run):</b> {html.escape(failed_tasks_text)}</p>"
        f"<p><b>Task State Counts:</b> {state_counts_text}</p>"
        "<h4>Processing Metrics</h4>"
        f"<p><b>Bronze local_ingested_count:</b> {normalized_metrics['local_ingested_count']}</p>"
        f"<p><b>Bronze synced_raw_count:</b> {normalized_metrics['synced_raw_count']}</p>"
        f"<p><b>Silver silver_raw_ready_count:</b> {normalized_metrics['silver_raw_ready_count']}</p>"
        f"<p><b>Gold silver_landmarks_ready_count:</b> {normalized_metrics['silver_landmarks_ready_count']}</p>"
        f"<p><b>Manifest path:</b> {normalized_metrics['manifest_path']}</p>"
        "<h4>Task Log Tail (last 60 lines)</h4>"
        f"<pre>{log_tail_html}</pre>"
        f"<p><a href=\"{log_url}\">Open Task Log</a></p>"
    )

    try:
        send_email(to=recipients, subject=subject, html_content=html_content)
        print(f"✅ Failure alert email sent to: {recipients}")
    except Exception as exc:
        print(f"⚠️ Failed to send failure email alert: {exc}")


# Apply callback at task level so each failed task can trigger alert immediately.
default_args["on_failure_callback"] = task_failure_email_alert

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

def _ensure_run_context(ti, create_if_missing: bool = False) -> dict:
    """Return run context from bronze_prepare_run_context XCom.

    If create_if_missing=True, generate and push a new run context on the current task.
    """
    CONTEXT_TASK_ID = "bronze_prepare_run_context"
    run_dir = ti.xcom_pull(task_ids=CONTEXT_TASK_ID, key="run_dir")
    run_id = ti.xcom_pull(task_ids=CONTEXT_TASK_ID, key="run_id")
    run_month = ti.xcom_pull(task_ids=CONTEXT_TASK_ID, key="run_month")
    run_stamp = ti.xcom_pull(task_ids=CONTEXT_TASK_ID, key="run_stamp")

    if run_dir and run_id and run_month and run_stamp:
        return {
            "run_dir": run_dir,
            "run_id": run_id,
            "run_month": run_month,
            "run_stamp": run_stamp,
        }

    if create_if_missing:
        now = datetime.now()
        run_id = now.strftime("%Y%m%d_%H%M%S")
        run_month = now.strftime("%Y%m")
        run_stamp = now.strftime("%Y%m%dT%H%M%S")
        run_dir = f"{PROJECT_ROOT}/data/runs/preprocess_{run_id}"
        os.makedirs(run_dir, exist_ok=True)

        ti.xcom_push(key="run_dir", value=run_dir)
        ti.xcom_push(key="run_id", value=run_id)
        ti.xcom_push(key="run_month", value=run_month)
        ti.xcom_push(key="run_stamp", value=run_stamp)
        print(f"✅ Run context prepared: run_id={run_id}, run_dir={run_dir}")

        return {
            "run_dir": run_dir,
            "run_id": run_id,
            "run_month": run_month,
            "run_stamp": run_stamp,
        }

    raise RuntimeError(
        f"Run context not found in XCom (task_ids={CONTEXT_TASK_ID!r}). "
        f"Ensure {CONTEXT_TASK_ID} has run successfully before calling this helper."
    )


def prepare_run_context(**context):
    """Prepare run context for downstream Bronze/Silver/Gold tasks."""
    _ensure_run_context(context["task_instance"], create_if_missing=True)


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

        object_name = f"{MINIO_BRONZE_RAW_PREFIX}/{ingest_month}/{ingest_day}/{label}/{filename}"
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

    run_ctx = _ensure_run_context(context["task_instance"])
    run_dir = run_ctx["run_dir"]
    processed_names, processed_etags = _load_processed_checkpoint_from_iceberg(run_dir)

    source_configs = [("bronze_raw", f"{MINIO_BRONZE_RAW_PREFIX}/")]
    if MINIO_UPLOAD_PREFIX.rstrip("/") != MINIO_BRONZE_RAW_PREFIX.rstrip("/"):
        source_configs.append(("user_upload", f"{MINIO_UPLOAD_PREFIX}/"))
    if MINIO_USER_UPLOAD_PREFIX.rstrip("/") not in {p.rstrip("/") for _, p in source_configs}:
        source_configs.append(("user_upload", f"{MINIO_USER_UPLOAD_PREFIX}/"))

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

def preprocess_videos(**context):
    """Normalize videos using existing preprocessing code"""

    new_uploads = context["task_instance"].xcom_pull(
        task_ids="bronze_collect_unprocessed_inputs", key="new_uploads"
    ) or []
    if not new_uploads:
        print("ℹ️ No new uploaded videos to preprocess. Skipping preprocessing step.")
        return

    raw_dir = context["task_instance"].xcom_pull(
        task_ids="bronze_collect_unprocessed_inputs", key="sync_raw_dir"
    ) or PREPROCESS_INPUT_STAGING_DIR
    output_dir = LOCAL_PREPROCESSED_DIR

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
        task_ids="bronze_collect_unprocessed_inputs", key="new_uploads"
    ) or []
    if not new_uploads:
        print("ℹ️ No new uploaded videos to extract landmarks. Skipping extraction step.")
        context["task_instance"].xcom_push(key="landmarks_dir", value=LOCAL_NPY_DIR)
        return

    preprocessed_dir = LOCAL_PREPROCESSED_DIR
    landmarks_dir = LOCAL_NPY_DIR

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
            "--seq_len", str(SEQ_LEN),
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

    print(f"✅ Gold landmark outputs ready: {silver_landmarks_ready_count} file(s)")
    context["task_instance"].xcom_push(
        key="silver_landmarks_ready_count", value=silver_landmarks_ready_count
    )

    print(f"✅ Landmark extraction complete")

    # Push landmark dir path
    context["task_instance"].xcom_push(key="landmarks_dir", value=landmarks_dir)

def store_to_minio(**context):
    """Upload Silver preprocessed videos and Gold landmarks to MinIO and create manifest"""
    landmarks_dir = context["task_instance"].xcom_pull(
        task_ids="gold_extract_landmarks", key="landmarks_dir"
    )
    run_id = context["task_instance"].xcom_pull(task_ids="bronze_prepare_run_context", key="run_id")
    run_month = context["task_instance"].xcom_pull(task_ids="bronze_prepare_run_context", key="run_month")
    run_stamp = context["task_instance"].xcom_pull(task_ids="bronze_prepare_run_context", key="run_stamp")
    run_dir = context["task_instance"].xcom_pull(task_ids="bronze_prepare_run_context", key="run_dir")
    if not run_id or not run_month or not run_stamp or not run_dir:
        run_ctx = _ensure_run_context(context["task_instance"])
        run_id = run_ctx["run_id"]
        run_month = run_ctx["run_month"]
        run_stamp = run_ctx["run_stamp"]
        run_dir = run_ctx["run_dir"]
    preprocessed_dir = LOCAL_PREPROCESSED_DIR
    new_uploads = context["task_instance"].xcom_pull(
        task_ids="bronze_collect_unprocessed_inputs", key="new_uploads"
    ) or []

    video_exts = VIDEO_EXTENSIONS
    
    # Create bucket if not exists
    try:
        if not minio_client.bucket_exists(MINIO_BUCKET):
            minio_client.make_bucket(MINIO_BUCKET)
    except Exception as e:
        print(f"MinIO bucket check failed: {e}")
    
    silver_inventory_rows = []
    gold_inventory_rows = []
    new_landmark_candidates = []
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
        pattern = os.path.join(preprocessed_dir, label, f"{stem}_*.mp4")
        segment_files = sorted(glob.glob(pattern))
        fallback_segment = os.path.join(preprocessed_dir, label, f"{stem}.mp4")
        if not segment_files and os.path.exists(fallback_segment):
            segment_files = [fallback_segment]

        if not segment_files:
            print(f"⚠️ No preprocessed segment found for upload: {local_input_rel}")

        for file_path in segment_files:
            if not file_path.lower().endswith(video_exts):
                continue

            rel_path = os.path.relpath(file_path, preprocessed_dir)
            object_name = f"{MINIO_SILVER_PREFIX}/{run_month}/{run_stamp}/{rel_path}"

            try:
                result = minio_client.fput_object(MINIO_BUCKET, object_name, file_path)
                uploaded_raw_files += 1

                silver_inventory_rows.append(
                    {
                        "run_id": run_id,
                        "object_path": object_name,
                        "file_type": "silver_preprocessed_video",
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

            new_landmark_candidates.append(
                {
                    "label": label,
                    "rel_npy": rel_npy,
                    "local_path": npy_path,
                    "size_bytes": os.path.getsize(npy_path),
                }
            )
            uploaded_landmark_files += 1

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

    print("✅ Silver incremental publish complete")

    # Append inventory to Iceberg using Spark engine
    namespace = os.environ.get("ICEBERG_NAMESPACE", "sign_language")
    spark_master = os.environ.get("SPARK_MASTER_URL", "spark://spark-master:7077")
    spark_packages = os.environ.get(
        "SPARK_ICEBERG_PACKAGES",
        "org.apache.iceberg:iceberg-spark-runtime-3.5_2.12:1.5.2,org.postgresql:postgresql:42.7.3,org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.262",
    )
    spark_job = f"{PROJECT_ROOT}/src/pipeline/spark_iceberg_inventory.py"

    silver_rows_file = f"{run_dir}/silver_inventory_rows.json"
    gold_rows_file = f"{run_dir}/gold_training_inventory_rows.json"
    bronze_rows_file = f"{run_dir}/bronze_inventory_rows.json"
    with open(silver_rows_file, "w", encoding="utf-8") as f:
        json.dump(silver_inventory_rows, f, ensure_ascii=False, indent=2)
    with open(bronze_rows_file, "w", encoding="utf-8") as f:
        json.dump(bronze_inventory_rows, f, ensure_ascii=False, indent=2)

    gold_state = _load_gold_state()
    gold_version = None
    gold_prefix = None
    if uploaded_landmark_files > 0:
        gold_state = _load_gold_state()
        previous_version = int(gold_state.get("latest_version", 0))
        next_version = previous_version + 1
        gold_version = f"v{next_version:04d}"
        gold_prefix = f"{MINIO_GOLD_ROOT_PREFIX}/{gold_version}"

        if previous_version > 0:
            previous_prefix = f"{MINIO_GOLD_ROOT_PREFIX}/v{previous_version:04d}/"
            copied = 0
            for obj in minio_client.list_objects(MINIO_BUCKET, prefix=previous_prefix, recursive=True):
                if not obj.object_name.endswith(".npy"):
                    continue
                rel = obj.object_name[len(previous_prefix):]
                dst = f"{gold_prefix}/{rel}"
                minio_client.copy_object(
                    MINIO_BUCKET,
                    dst,
                    CopySource(MINIO_BUCKET, obj.object_name),
                )
                copied += 1
            print(f"✅ Gold snapshot seed copied from previous version: {copied} file(s)")

        for item in new_landmark_candidates:
            dst = f"{gold_prefix}/{item['rel_npy']}"
            result = minio_client.fput_object(MINIO_BUCKET, dst, item["local_path"])
            gold_inventory_rows.append(
                {
                    "run_id": gold_version,
                    "object_path": dst,
                    "file_type": "gold_landmark_snapshot",
                    "label": item["label"],
                    "size_bytes": item["size_bytes"],
                    "etag": result.etag,
                }
            )

        # Rebuild full snapshot inventory rows so training gets complete Gold state.
        snapshot_rows = []
        full_prefix = f"{gold_prefix}/"
        for obj in minio_client.list_objects(MINIO_BUCKET, prefix=full_prefix, recursive=True):
            if not obj.object_name.endswith(".npy"):
                continue
            rel = obj.object_name[len(full_prefix):]
            label = rel.split("/")[0] if "/" in rel else "unknown"
            snapshot_rows.append(
                {
                    "run_id": gold_version,
                    "object_path": obj.object_name,
                    "file_type": "gold_landmark_snapshot",
                    "label": label,
                    "size_bytes": obj.size,
                    "etag": (obj.etag or "").strip('"'),
                }
            )

        gold_inventory_rows = snapshot_rows
        gold_state["latest_version"] = next_version
        gold_state["last_source_run_id"] = run_id
        gold_state["last_updated_at"] = datetime.now().isoformat()
        gold_state["latest_snapshot_prefix"] = gold_prefix
        _save_gold_state(gold_state)
        print(
            f"✅ Published Gold full snapshot {gold_version}: {len(gold_inventory_rows)} file(s)"
        )
    else:
        latest_version = int(gold_state.get("latest_version", 0))
        if latest_version > 0:
            gold_version = f"v{latest_version:04d}"
            gold_prefix = f"{MINIO_GOLD_ROOT_PREFIX}/{gold_version}"
        print("ℹ️ No new landmarks in this run, Gold snapshot version unchanged.")

    with open(gold_rows_file, "w", encoding="utf-8") as f:
        json.dump(gold_inventory_rows, f, ensure_ascii=False, indent=2)

    silver_cmd = [
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
        silver_rows_file,
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

    def _append_with_repair(cmd: list[str], table_name: str) -> subprocess.CompletedProcess:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
        if result.returncode == 0:
            return result

        if _is_missing_iceberg_metadata_output(result.stdout or "", result.stderr or ""):
            _repair_catalog_entry(namespace, table_name)
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
    _append_with_repair(silver_cmd, ICEBERG_SILVER_RAW_TABLE)
    if gold_inventory_rows:
        _append_with_repair(gold_cmd, ICEBERG_GOLD_TRAINING_TABLE)

    print("✅ Iceberg inventory appended via Spark engine")
    
    # Create preprocessing manifest
    manifest = {
        "run_id": run_id,
        "run_stamp": run_stamp,
        "timestamp": datetime.now().isoformat(),
        "dataset_name": DATASET_NAME,
        "seq_len": SEQ_LEN,
        "local_raw_dir": LOCAL_RAW_SOURCE_DIR,
        "local_preprocessed_dir": preprocessed_dir,
        "minio_preprocessed_prefix": f"{MINIO_SILVER_PREFIX}/{run_month}/{run_stamp}",
        "raw_files_count": uploaded_raw_files,
        "local_landmarks_dir": landmarks_dir,
        "minio_prefix": gold_prefix,
        "gold_landmarks_prefix": gold_prefix,
        "minio_bucket": MINIO_BUCKET,
        "landmark_files_count": uploaded_landmark_files,
        "synced_new_uploads": len(new_uploads),
        "minio_upload_prefix": MINIO_UPLOAD_PREFIX,
        "minio_bronze_raw_prefix": MINIO_BRONZE_RAW_PREFIX,
        "minio_user_upload_prefix": MINIO_USER_UPLOAD_PREFIX,
        "minio_silver_prefix": MINIO_SILVER_PREFIX,
        "minio_gold_root_prefix": MINIO_GOLD_ROOT_PREFIX,
        "gold_version": gold_version,
        "gold_minio_prefix": gold_prefix,
        "iceberg_tables": {
            "bronze": f"{namespace}.{ICEBERG_BRONZE_TABLE}",
            "silver_raw": f"{namespace}.{ICEBERG_SILVER_RAW_TABLE}",
            "gold_training": f"{namespace}.{ICEBERG_GOLD_TRAINING_TABLE}",
        },
    }
    
    manifest_path = f"{PROJECT_ROOT}/data/{DATASET_NAME}/preprocessing_manifest.json"
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
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
    description="Bronze raw videos -> Silver preprocessed videos -> Gold landmarks",
    schedule_interval=None,  # Manual trigger
    catchup=False,
    on_failure_callback=task_failure_email_alert,
    tags=["preprocessing", "sign-language"],
) as dag:
    start = EmptyOperator(
        task_id="start",
        doc_md="Start node for preprocessing pipeline run.",
    )
    end = EmptyOperator(
        task_id="end",
        doc_md="End node for preprocessing pipeline run.",
    )

    with TaskGroup(group_id="bronze", prefix_group_id=False) as bronze_group:
        bronze_prepare_context = PythonOperator(
            task_id="bronze_prepare_run_context",
            python_callable=prepare_run_context,
            doc_md="Prepare shared run metadata for all downstream stages.",
        )
        bronze_ingest_local = PythonOperator(
            task_id="bronze_ingest_local_raw",
            python_callable=ingest_local_raw_to_bronze,
            doc_md="Ingest local raw dataset snapshot into Bronze storage.",
        )
        bronze_sync_pending = PythonOperator(
            task_id="bronze_collect_unprocessed_inputs",
            python_callable=sync_pending_bronze_inputs,
            doc_md="Collect only unprocessed Bronze inputs using Iceberg checkpoint.",
        )

        bronze_prepare_context >> bronze_ingest_local >> bronze_sync_pending

    with TaskGroup(group_id="silver", prefix_group_id=False) as silver_group:
        silver_preprocess = PythonOperator(
            task_id="silver_preprocess_videos",
            python_callable=preprocess_videos,
            doc_md="Normalize and segment videos for Silver stage outputs.",
        )

    with TaskGroup(group_id="gold", prefix_group_id=False) as gold_group:
        gold_extract = PythonOperator(
            task_id="gold_extract_landmarks",
            python_callable=extract_landmarks,
            doc_md="Extract landmark arrays from preprocessed Silver videos.",
        )
        gold_publish = PythonOperator(
            task_id="gold_merge_snapshot",
            python_callable=store_to_minio,
            doc_md="Merge new landmarks into Gold snapshot and append Iceberg inventory.",
        )

        gold_extract >> gold_publish

    start >> bronze_group >> silver_group >> gold_group >> end

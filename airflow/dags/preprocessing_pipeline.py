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
import shutil
from datetime import datetime
from pathlib import Path

from airflow import DAG
from airflow.models.param import Param
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.task_group import TaskGroup
from minio.commonconfig import CopySource

from src.config.config import DATASET_NAME, SEQ_LEN

from shared.config import (
    PROJECT_ROOT,
    MINIO_BUCKET,
    MINIO_BRONZE_RAW_PREFIX,
    MINIO_USER_UPLOAD_PREFIX,
    MINIO_UPLOAD_PREFIX,
    MINIO_SILVER_PREFIX,
    ICEBERG_NAMESPACE,
    ICEBERG_BRONZE_TABLE,
    ICEBERG_SILVER_RAW_TABLE,
    ICEBERG_GOLD_TRAINING_TABLE,
    VIDEO_EXTENSIONS,
    LOCAL_RAW_SOURCE_DIR,
    LOCAL_PREPROCESSED_DIR,
    PREPROCESS_INPUT_STAGING_DIR,
    DEFAULT_ARGS,
    minio_client,
    build_minio_gold_root_prefix,
    build_local_npy_dir,
)
from shared.utils import (
    run_streaming,
    list_files,
    build_spark_cmd,
    spark_query,
    ensure_run_context,
    load_gold_state,
    save_gold_state,
)
from shared.alerts import task_failure_email_alert

# ================================================================
# Configuration
# ================================================================

PREPROCESS_CONTEXT_TASK_ID = "bronze_prepare_run_context"


# ================================================================
# Task Helpers
# ================================================================

def _load_processed_checkpoint_from_iceberg(run_dir: str) -> tuple[set[str], set[str]]:
    """Load processed Bronze checkpoint from Iceberg table (source of truth)."""
    checkpoint_file = f"{run_dir}/bronze_processed_checkpoint.json"

    cmd = build_spark_cmd(
        "checkpoint",
        ["--table", ICEBERG_BRONZE_TABLE, "--output-file", checkpoint_file],
    )

    spark_query(cmd, ICEBERG_BRONZE_TABLE)

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
        f"Loaded Iceberg checkpoint: object_paths={len(processed_objects)}, etags={len(processed_etags)}"
    )
    return processed_objects, processed_etags


def _ensure_preprocess_run_context(ti, create_if_missing: bool = False) -> dict:
    return ensure_run_context(
        ti,
        context_task_id=PREPROCESS_CONTEXT_TASK_ID,
        create_if_missing=create_if_missing,
        run_id_prefix="preprocess",
    )


# ================================================================
# Task Functions
# ================================================================

def prepare_run_context(**context):
    """Prepare run context for downstream Bronze/Silver/Gold tasks."""
    _ensure_preprocess_run_context(context["task_instance"], create_if_missing=True)


def ingest_local_raw_to_bronze(**context):
    """Upload current local raw snapshot into Bronze local_dataset partition."""

    local_raw_dir = LOCAL_RAW_SOURCE_DIR
    if not os.path.isdir(local_raw_dir):
        print(f"Local raw directory does not exist: {local_raw_dir}")
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
        print(f"Cannot validate MinIO bucket {MINIO_BUCKET}: {exc}")

    for rel in list_files(local_raw_dir, VIDEO_EXTENSIONS):
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
            print(f"Failed to ingest local raw file {src_abs}: {exc}")

    print(
        f"Ingested local raw snapshot to Bronze: {uploaded} uploaded, {skipped} skipped"
    )
    context["task_instance"].xcom_push(key="local_ingested_count", value=uploaded)


def sync_pending_bronze_inputs(**context):
    """Sync all pending unprocessed videos from Bronze sources using Iceberg checkpoint."""

    raw_dir = PREPROCESS_INPUT_STAGING_DIR
    shutil.rmtree(raw_dir, ignore_errors=True)
    os.makedirs(raw_dir, exist_ok=True)

    try:
        if not minio_client.bucket_exists(MINIO_BUCKET):
            print(f"MinIO bucket does not exist yet: {MINIO_BUCKET}")
            context["task_instance"].xcom_push(key="new_uploads", value=[])
            context["task_instance"].xcom_push(key="synced_raw_count", value=0)
            context["task_instance"].xcom_push(key="sync_raw_dir", value=raw_dir)
            return
    except Exception as exc:
        raise Exception(f"Cannot verify MinIO bucket {MINIO_BUCKET}: {exc}")

    run_ctx = _ensure_preprocess_run_context(context["task_instance"])
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
        f"Synced pending Bronze inputs: {len(pending_inputs)} new, {skipped} skipped, {scanned} scanned"
    )

    context["task_instance"].xcom_push(key="new_uploads", value=pending_inputs)
    context["task_instance"].xcom_push(key="synced_raw_count", value=len(pending_inputs))
    context["task_instance"].xcom_push(key="sync_raw_dir", value=raw_dir)

def preprocess_videos(**context):
    """Normalize videos using existing preprocessing code"""

    params = context["params"]
    skip_existing = bool(params["skip_existing"])

    new_uploads = context["task_instance"].xcom_pull(
        task_ids="bronze_collect_unprocessed_inputs", key="new_uploads"
    ) or []
    if not new_uploads:
        print("No new uploaded videos to preprocess. Skipping preprocessing step.")
        return

    raw_dir = context["task_instance"].xcom_pull(
        task_ids="bronze_collect_unprocessed_inputs", key="sync_raw_dir"
    ) or PREPROCESS_INPUT_STAGING_DIR
    output_dir = LOCAL_PREPROCESSED_DIR

    os.makedirs(output_dir, exist_ok=True)

    video_files = list_files(raw_dir, (".mp4", ".avi", ".mov", ".mkv", ".webm"))
    print(f"Raw input dir: {raw_dir}")
    print(f"Found {len(video_files)} raw video file(s)")
    for idx, rel in enumerate(video_files[:10], start=1):
        print(f"   [{idx}] {rel}")
    if len(video_files) > 10:
        print(f"   ... and {len(video_files) - 10} more")

    cmd = [
        "python", "-u", f"{PROJECT_ROOT}/src/preprocess/preprocess_video.py",
        "--input_dir", raw_dir,
        "--output_dir", output_dir,
    ]
    if skip_existing:
        cmd.append("--skip_existing")
    run_streaming(cmd, cwd=PROJECT_ROOT)

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

    print(f"Silver raw outputs ready: {silver_raw_ready_count} file(s)")
    context["task_instance"].xcom_push(
        key="silver_raw_ready_count", value=silver_raw_ready_count
    )

    print("Video preprocessing complete")

def extract_landmarks(**context):
    """Extract MediaPipe landmarks from videos"""

    params        = context["params"]
    seq_len       = int(params["seq_len"])
    skip_existing = bool(params["skip_existing"])
    landmarks_dir = build_local_npy_dir(seq_len)
    preprocessed_dir = LOCAL_PREPROCESSED_DIR

    os.makedirs(landmarks_dir, exist_ok=True)

    new_uploads = context["task_instance"].xcom_pull(
        task_ids="bronze_collect_unprocessed_inputs", key="new_uploads"
    ) or []

    preprocessed_files = list_files(preprocessed_dir, (".mp4", ".avi", ".mov", ".mkv", ".webm"))

    # Detect preprocessed Silver videos missing their .npy counterpart for
    # THIS seq_len. Without this check, a DAG run triggered with a new
    # ``seq_len`` Param would skip extraction whenever ``new_uploads`` is
    # empty, even though the target landmarks_dir is empty.
    missing_landmarks = [
        rel for rel in preprocessed_files
        if not os.path.exists(
            os.path.join(landmarks_dir, str(Path(rel).with_suffix(".npy")))
        )
    ]

    # Decide whether to invoke the extraction subprocess.
    if not preprocessed_files:
        print("No preprocessed videos found. Skipping extraction subprocess.")
    elif not new_uploads and not missing_landmarks:
        print(
            f"No new uploads and all {len(preprocessed_files)} preprocessed "
            f"videos already have landmarks at seq_len={seq_len}. "
            f"Skipping extraction subprocess (will still check Gold for unpublished landmarks)."
        )
    else:
        if not new_uploads and missing_landmarks:
            print(
                f"No new Bronze uploads, but {len(missing_landmarks)} preprocessed "
                f"videos are missing landmarks at seq_len={seq_len} — running extraction."
            )

        print(f"Preprocessed input dir: {preprocessed_dir}")
        print(f"Found {len(preprocessed_files)} preprocessed video file(s)")
        for idx, rel in enumerate(preprocessed_files[:10], start=1):
            print(f"   [{idx}] {rel}")
        if len(preprocessed_files) > 10:
            print(f"   ... and {len(preprocessed_files) - 10} more")

        cmd = [
            "python", "-u", f"{PROJECT_ROOT}/src/preprocess/video2npy.py",
            "--input_dir", preprocessed_dir,
            "--output_dir", landmarks_dir,
            "--seq_len", str(seq_len),
        ]
        if skip_existing:
            cmd.append("--skip_existing")
        run_streaming(cmd, cwd=PROJECT_ROOT)

    # Count every preprocessed Silver video whose landmark counterpart now
    # exists at this seq_len — not just those tied to new_uploads, so a
    # re-extraction triggered by a seq_len change is reported correctly.
    silver_landmarks_ready_count = sum(
        1 for rel in preprocessed_files
        if os.path.exists(
            os.path.join(landmarks_dir, str(Path(rel).with_suffix(".npy")))
        )
    )

    print(f"Gold landmark outputs ready: {silver_landmarks_ready_count} file(s)")
    context["task_instance"].xcom_push(
        key="silver_landmarks_ready_count", value=silver_landmarks_ready_count
    )

    # ── Detect local landmarks not yet published to the latest Gold snapshot ──
    # Handles "repair" runs: no new Bronze uploads and every preprocessed video
    # already has a local .npy at this seq_len, but the landmarks never reached
    # MinIO Gold (e.g. a previous run failed after extraction, or the landmarks
    # were produced at a new seq_len for the first time). These candidates are
    # forwarded to store_to_minio so it can merge them into a Gold snapshot.
    #
    # The result is serialized to a JSON file under run_dir (not XCom) because
    # the list can grow to thousands of entries on repair / new-seq_len runs
    # and would otherwise risk hitting the XCom size limit.
    run_ctx         = _ensure_preprocess_run_context(context["task_instance"])
    run_dir         = run_ctx["run_dir"]
    os.makedirs(run_dir, exist_ok=True)

    gold_state      = load_gold_state()
    latest_version  = int(gold_state.get("latest_version", 0))
    minio_gold_root = build_minio_gold_root_prefix(seq_len)

    published_rel_npys: set[str] = set()
    if latest_version > 0:
        latest_prefix = f"{minio_gold_root}/v{latest_version:04d}/"
        try:
            for obj in minio_client.list_objects(
                MINIO_BUCKET, prefix=latest_prefix, recursive=True
            ):
                if obj.object_name.endswith(".npy"):
                    published_rel_npys.add(obj.object_name[len(latest_prefix):])
        except Exception as exc:
            print(f"Cannot list latest Gold snapshot prefix {latest_prefix}: {exc}")

    candidates_file = os.path.join(run_dir, "unpublished_gold_candidates.jsonl")
    unpublished_count = 0
    with open(candidates_file, "w", encoding="utf-8") as fout:
        for rel in preprocessed_files:
            rel_npy    = str(Path(rel).with_suffix(".npy"))
            local_path = os.path.join(landmarks_dir, rel_npy)
            if not os.path.exists(local_path):
                continue
            if rel_npy in published_rel_npys:
                continue
            parts = Path(rel_npy).parts
            label = parts[0] if len(parts) > 1 else "unknown"
            fout.write(
                json.dumps(
                    {
                        "label":      label,
                        "rel_npy":    rel_npy,
                        "local_path": local_path,
                        "size_bytes": os.path.getsize(local_path),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            unpublished_count += 1

    if unpublished_count:
        print(
            f"Detected {unpublished_count} local landmark(s) not yet published "
            f"to the latest Gold snapshot (seq_len={seq_len}). "
            f"Written to {candidates_file}"
        )
    else:
        print("All local landmarks are already present in the latest Gold snapshot.")

    context["task_instance"].xcom_push(
        key="unpublished_gold_candidates_file", value=candidates_file
    )
    context["task_instance"].xcom_push(
        key="unpublished_gold_candidates_count", value=unpublished_count
    )

    print("Landmark extraction complete")
    context["task_instance"].xcom_push(key="landmarks_dir", value=landmarks_dir)

def store_to_minio(**context):
    """Upload Silver preprocessed videos and Gold landmarks to MinIO and create manifest"""
    params                 = context["params"]
    seq_len                = int(params["seq_len"])
    minio_gold_root_prefix = build_minio_gold_root_prefix(seq_len)

    landmarks_dir = context["task_instance"].xcom_pull(
        task_ids="gold_extract_landmarks", key="landmarks_dir"
    )
    run_ctx = _ensure_preprocess_run_context(context["task_instance"])
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
            print(f"No preprocessed segment found for upload: {local_input_rel}")

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
                print(f"Missing landmark file for segment: {npy_path}")
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

    # Merge in local landmarks that extract_landmarks detected as missing from
    # the latest Gold snapshot (repair path / first-time seq_len publish).
    # The candidate list is streamed from a JSON-lines file in run_dir to
    # avoid XCom size limits when the list is large.
    unpublished_candidates_file = context["task_instance"].xcom_pull(
        task_ids="gold_extract_landmarks", key="unpublished_gold_candidates_file"
    )
    if unpublished_candidates_file and os.path.exists(unpublished_candidates_file):
        already_queued = {item["rel_npy"] for item in new_landmark_candidates}
        merged = 0
        with open(unpublished_candidates_file, "r", encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                try:
                    cand = json.loads(line)
                except json.JSONDecodeError as exc:
                    print(f"Skipping malformed candidate line: {exc}")
                    continue
                if cand["rel_npy"] in already_queued:
                    continue
                if not os.path.exists(cand["local_path"]):
                    print(f"Skipping missing local landmark: {cand['local_path']}")
                    continue
                new_landmark_candidates.append(cand)
                already_queued.add(cand["rel_npy"])
                uploaded_landmark_files += 1
                merged += 1
        if merged:
            print(
                f"Queued {merged} unpublished landmark(s) from repair path "
                f"for Gold publish."
            )

    print("Silver incremental publish complete")

    # Write inventory rows to JSON files for Spark
    silver_rows_file = f"{run_dir}/silver_inventory_rows.json"
    gold_rows_file = f"{run_dir}/gold_training_inventory_rows.json"
    bronze_rows_file = f"{run_dir}/bronze_inventory_rows.json"
    with open(silver_rows_file, "w", encoding="utf-8") as f:
        json.dump(silver_inventory_rows, f, ensure_ascii=False, indent=2)
    with open(bronze_rows_file, "w", encoding="utf-8") as f:
        json.dump(bronze_inventory_rows, f, ensure_ascii=False, indent=2)

    gold_state = load_gold_state()
    gold_version = None
    gold_prefix = None
    if uploaded_landmark_files > 0:
        gold_state = load_gold_state()
        previous_version = int(gold_state.get("latest_version", 0))
        next_version = previous_version + 1
        gold_version = f"v{next_version:04d}"
        gold_prefix = f"{minio_gold_root_prefix}/{gold_version}"

        if previous_version > 0:
            previous_prefix = f"{minio_gold_root_prefix}/v{previous_version:04d}/"
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
            print(f"Gold snapshot seed copied from previous version: {copied} file(s)")

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
        save_gold_state(gold_state)
        print(
            f"Published Gold full snapshot {gold_version}: {len(gold_inventory_rows)} file(s)"
        )
    else:
        latest_version = int(gold_state.get("latest_version", 0))
        if latest_version > 0:
            gold_version = f"v{latest_version:04d}"
            gold_prefix = f"{minio_gold_root_prefix}/{gold_version}"
        print("No new landmarks in this run, Gold snapshot version unchanged.")

    with open(gold_rows_file, "w", encoding="utf-8") as f:
        json.dump(gold_inventory_rows, f, ensure_ascii=False, indent=2)

    # Append inventory to Iceberg via Spark
    bronze_cmd = build_spark_cmd("append", ["--table", ICEBERG_BRONZE_TABLE, "--rows-file", bronze_rows_file])
    silver_cmd = build_spark_cmd("append", ["--table", ICEBERG_SILVER_RAW_TABLE, "--rows-file", silver_rows_file])
    gold_cmd = build_spark_cmd("append", ["--table", ICEBERG_GOLD_TRAINING_TABLE, "--rows-file", gold_rows_file])

    spark_query(bronze_cmd, ICEBERG_BRONZE_TABLE)
    spark_query(silver_cmd, ICEBERG_SILVER_RAW_TABLE)
    if gold_inventory_rows:
        spark_query(gold_cmd, ICEBERG_GOLD_TRAINING_TABLE)

    print("Iceberg inventory appended via Spark engine")

    # Create preprocessing manifest
    manifest = {
        "run_id": run_id,
        "run_stamp": run_stamp,
        "timestamp": datetime.now().isoformat(),
        "dataset_name": DATASET_NAME,
        "seq_len": seq_len,
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
        "minio_gold_root_prefix": minio_gold_root_prefix,
        "gold_version": gold_version,
        "gold_minio_prefix": gold_prefix,
        "iceberg_tables": {
            "bronze": f"{ICEBERG_NAMESPACE}.{ICEBERG_BRONZE_TABLE}",
            "silver_raw": f"{ICEBERG_NAMESPACE}.{ICEBERG_SILVER_RAW_TABLE}",
            "gold_training": f"{ICEBERG_NAMESPACE}.{ICEBERG_GOLD_TRAINING_TABLE}",
        },
    }

    manifest_path = f"{PROJECT_ROOT}/data/{DATASET_NAME}/preprocessing_manifest.json"
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Preprocessing manifest created: {manifest_path}")

    context["task_instance"].xcom_push(key="manifest_path", value=manifest_path)

    # Cleanup transient staging dirs after successful publish.
    shutil.rmtree(PREPROCESS_INPUT_STAGING_DIR, ignore_errors=True)


# ================================================================
# DAG Definition
# ================================================================

with DAG(
    dag_id="preprocessing_pipeline",
    default_args=DEFAULT_ARGS,
    description="Bronze raw videos -> Silver preprocessed videos -> Gold landmarks",
    schedule_interval=None,  # Manual trigger
    catchup=False,
    on_failure_callback=task_failure_email_alert,
    tags=["preprocessing", "sign-language"],
    params={
        "seq_len": Param(
            SEQ_LEN,
            type="integer",
            minimum=1,
            title="Sequence length",
            description=(
                "Frames to be sampled per videoclip."
            ),
        ),
        "skip_existing": Param(
            True,
            type="boolean",
            title="Skip existing outputs",
            description=(
                "Files already produced are not re-generated."
            ),
        ),
    },
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

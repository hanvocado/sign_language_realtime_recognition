"""
Tasks: data acquisition and splitting.

download_gold_snapshot — pulls the versioned npy snapshot from MinIO
                         using an etag manifest to skip unchanged files.
split_dataset          — runs split_dataset.py to produce train/val/test.
"""

import os
import json
import shutil

from training.config import (
    MINIO_BUCKET,
    LOCAL_TRAINING_DIR,
    LOCAL_SPLIT_DIR,
    PROJECT_ROOT,
)
from training.utils import run_streaming
from shared.config import minio_client


# ── Task callables ───────────────────────────────────────────────────

def download_gold_snapshot(**context) -> None:
    """
    Download the full Gold npy snapshot from MinIO into LOCAL_TRAINING_DIR.
    Skips files whose etag matches the local manifest (incremental sync).

    XCom output (key="local_npy_dir"): absolute path to downloaded snapshot.
    """
    ti           = context["task_instance"]
    gold_version = ti.xcom_pull(task_ids="training_retrain_check", key="gold_version")
    gold_prefix  = ti.xcom_pull(task_ids="training_retrain_check", key="gold_prefix")

    client = minio_client
    local_dir     = os.path.join(LOCAL_TRAINING_DIR, gold_version)
    manifest_path = os.path.join(LOCAL_TRAINING_DIR, f".manifest_{gold_version}.json")
    os.makedirs(local_dir, exist_ok=True)

    manifest: dict[str, str] = {}
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)

    prefix     = gold_prefix.rstrip("/") + "/"
    downloaded = skipped = 0

    for obj in client.list_objects(MINIO_BUCKET, prefix=prefix, recursive=True):
        if not obj.object_name.endswith(".npy"):
            continue

        etag       = (obj.etag or "").strip('"')
        rel        = obj.object_name[len(prefix):]
        local_path = os.path.join(local_dir, rel)

        if etag and manifest.get(obj.object_name) == etag and os.path.exists(local_path):
            skipped += 1
            continue

        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        client.fget_object(MINIO_BUCKET, obj.object_name, local_path)
        manifest[obj.object_name] = etag
        downloaded += 1

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Gold snapshot {gold_version}: {downloaded} downloaded, {skipped} skipped")
    ti.xcom_push(key="local_npy_dir", value=local_dir)


def split_dataset(**context) -> None:
    """
    Run split_dataset.py on the downloaded Gold snapshot.
    Clears any previous split for this gold_version before running.

    XCom output (key="split_dir"): absolute path to the split root.
    """
    ti            = context["task_instance"]
    local_npy_dir = ti.xcom_pull(task_ids="download_gold_snapshot", key="local_npy_dir")
    gold_version  = ti.xcom_pull(task_ids="training_retrain_check", key="gold_version")

    split_out = os.path.join(LOCAL_SPLIT_DIR, gold_version)
    shutil.rmtree(split_out, ignore_errors=True)
    os.makedirs(split_out, exist_ok=True)

    run_streaming(
        [
            "python", "-u",
            f"{PROJECT_ROOT}/src/preprocess/split_dataset.py",
            "--data_dir",    local_npy_dir,
            "--output_dir",  split_out,
            "--file_type",   "npy",
            "--train_ratio", "0.70",
            "--val_ratio",   "0.15",
            "--seed",        "42",
        ],
        cwd=PROJECT_ROOT,
    )

    ti.xcom_push(key="split_dir", value=split_out)

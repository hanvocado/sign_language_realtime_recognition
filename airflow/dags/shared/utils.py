"""
Shared utility functions for all Airflow DAGs.

Subprocess helpers, Spark/Iceberg operations, run context management,
and Gold state I/O used by both preprocessing and training pipelines.
"""

import os
import json
import subprocess
from datetime import datetime

import psycopg2

from shared.config import (
    PROJECT_ROOT,
    GOLD_VERSION_STATE_PATH,
    SPARK_MASTER_URL,
    SPARK_ICEBERG_PACKAGES,
    SPARK_ICEBERG_JOB,
)


# ── Subprocess ───────────────────────────────────────────────────────

def run_streaming(cmd: list[str], cwd: str | None = None) -> None:
    """Run a command and stream its stdout/stderr to Airflow task logs."""
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen(
        cmd, cwd=cwd, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    assert proc.stdout
    for line in proc.stdout:
        print(line.rstrip())
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}")


def list_files(base_dir: str, exts: tuple[str, ...]) -> list[str]:
    """Recursively list files under base_dir matching given extensions."""
    files = []
    if not os.path.isdir(base_dir):
        return files
    for root, _, names in os.walk(base_dir):
        for name in names:
            if name.lower().endswith(exts):
                files.append(os.path.relpath(os.path.join(root, name), base_dir))
    files.sort()
    return files


# ── Iceberg / Spark ──────────────────────────────────────────────────

def is_missing_iceberg_metadata(stdout: str, stderr: str) -> bool:
    merged = f"{stdout}\n{stderr}"
    return "FileNotFoundException" in merged and "/metadata/" in merged


def repair_catalog_entry(table_name: str) -> None:
    """Delete a stale Iceberg catalog pointer from the Postgres JDBC catalog."""
    conn = psycopg2.connect(
        host=os.environ.get("ICEBERG_POSTGRES_HOST", "postgres"),
        port=int(os.environ.get("ICEBERG_POSTGRES_PORT", "5432")),
        dbname=os.environ.get("ICEBERG_POSTGRES_DB", "iceberg"),
        user=os.environ.get("POSTGRES_USER", "postgres"),
        password=os.environ.get("POSTGRES_PASSWORD", "postgres"),
    )
    try:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM iceberg_tables WHERE table_name = %s", (table_name,)
            )
            print(f"Repaired catalog for {table_name} (deleted {cur.rowcount} rows)")
        conn.commit()
    finally:
        conn.close()


def build_spark_cmd(subcommand: str, extra_args: list[str]) -> list[str]:
    """Build a spark-submit command for spark_iceberg_inventory.py."""
    return [
        "spark-submit",
        "--master", SPARK_MASTER_URL,
        "--packages", SPARK_ICEBERG_PACKAGES,
        SPARK_ICEBERG_JOB,
        subcommand,
        *extra_args,
    ]


def spark_query(cmd: list[str], table_name: str) -> subprocess.CompletedProcess:
    """Run a spark-submit command with one automatic catalog-repair retry."""
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
    if result.returncode == 0:
        return result

    if is_missing_iceberg_metadata(result.stdout or "", result.stderr or ""):
        repair_catalog_entry(table_name)
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)

    if result.returncode != 0:
        tail_out = "\n".join((result.stdout or "").splitlines()[-120:])
        tail_err = "\n".join((result.stderr or "").splitlines()[-120:])
        raise RuntimeError(
            f"Spark job failed for {table_name}.\n"
            f"STDOUT:\n{tail_out}\nSTDERR:\n{tail_err}"
        )
    return result


# ── Run context ──────────────────────────────────────────────────────

def ensure_run_context(
    ti,
    context_task_id: str,
    create_if_missing: bool = False,
    run_id_prefix: str = "preprocess",
) -> dict:
    """Pull or create the shared run context (run_id, run_dir, run_month, run_stamp).

    Parameters
    ----------
    ti : TaskInstance
    context_task_id : upstream task_id that originally pushed the context keys.
    create_if_missing : if True, generate and push a new run context.
    run_id_prefix : prefix for run_id when creating (e.g. "preprocess", "train").
    """
    run_dir = ti.xcom_pull(task_ids=context_task_id, key="run_dir")
    run_id = ti.xcom_pull(task_ids=context_task_id, key="run_id")
    run_month = ti.xcom_pull(task_ids=context_task_id, key="run_month")
    run_stamp = ti.xcom_pull(task_ids=context_task_id, key="run_stamp")

    if run_dir and run_id and run_month and run_stamp:
        return {
            "run_dir": run_dir,
            "run_id": run_id,
            "run_month": run_month,
            "run_stamp": run_stamp,
        }

    if create_if_missing:
        now = datetime.now()
        run_id = f"{run_id_prefix}_{now.strftime('%Y%m%d_%H%M%S')}"
        run_month = now.strftime("%Y%m")
        run_stamp = now.strftime("%Y%m%dT%H%M%S")
        run_dir = f"{PROJECT_ROOT}/data/runs/{run_id}"
        os.makedirs(run_dir, exist_ok=True)

        ti.xcom_push(key="run_dir", value=run_dir)
        ti.xcom_push(key="run_id", value=run_id)
        ti.xcom_push(key="run_month", value=run_month)
        ti.xcom_push(key="run_stamp", value=run_stamp)
        print(f"Run context: run_id={run_id}, run_dir={run_dir}")
        return {
            "run_dir": run_dir,
            "run_id": run_id,
            "run_month": run_month,
            "run_stamp": run_stamp,
        }

    raise RuntimeError(
        f"Run context not found (task_ids={context_task_id!r}). "
        f"Ensure {context_task_id} ran successfully first."
    )


# ── Gold state I/O ───────────────────────────────────────────────────

def load_gold_state() -> dict:
    try:
        with open(GOLD_VERSION_STATE_PATH, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            return {"latest_version": 0}
        payload.setdefault("latest_version", 0)
        return payload
    except Exception:
        return {"latest_version": 0}


def save_gold_state(state: dict) -> None:
    os.makedirs(os.path.dirname(GOLD_VERSION_STATE_PATH), exist_ok=True)
    with open(GOLD_VERSION_STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

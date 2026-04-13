"""
Shared Airflow failure alert callbacks.
"""

import os
import json
import html
from collections import deque

from airflow.utils.email import send_email


def task_failure_email_alert(context):
    """Send rich failure alert email with task/run states and processing metrics."""
    recipients_raw = os.environ.get("AIRFLOW_ALERT_EMAIL_TO", "").strip()
    if not recipients_raw:
        print("AIRFLOW_ALERT_EMAIL_TO is empty, skip failure email alert.")
        return

    recipients = [item.strip() for item in recipients_raw.split(",") if item.strip()]
    if not recipients:
        print("No valid alert recipients found, skip failure email alert.")
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
        print(f"Failure alert email sent to: {recipients}")
    except Exception as exc:
        print(f"Failed to send failure email alert: {exc}")

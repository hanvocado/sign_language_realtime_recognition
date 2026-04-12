"""
Task: prepare run context.

Stamps run_id / run_dir / run_month / run_stamp into XCom so every
downstream task can call ensure_run_context(ti) without re-creating them.
"""

from airflow.dags.training.utils import ensure_run_context


def prepare_run_context(**context) -> None:
    ensure_run_context(context["task_instance"], create_if_missing=True)

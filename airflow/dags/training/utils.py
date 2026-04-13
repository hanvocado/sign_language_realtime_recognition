"""
Training pipeline utilities.

Shared helpers (subprocess, Spark, Iceberg) are re-exported from
shared.utils.  Training-specific helpers (run context, sensor state)
live here.
"""

import os
import json

from training.config import (
    PROJECT_ROOT,
    TRAINING_SENSOR_STATE_PATH,
)

# Re-export shared utilities so existing imports keep working.
from shared.utils import (  # noqa: F401
    run_streaming,
    list_files,
    build_spark_cmd,
    spark_query,
    is_missing_iceberg_metadata,
    repair_catalog_entry,
    ensure_run_context as _shared_ensure_run_context,
)


# ── Run context (training-specific wrapper) ──────────────────────────

CONTEXT_TASK_ID = "training_prepare_run_context"


def ensure_run_context(ti, create_if_missing: bool = False) -> dict:
    """Training-specific wrapper: delegates to shared ensure_run_context
    with the training context task ID and run_id prefix."""
    return _shared_ensure_run_context(
        ti,
        context_task_id=CONTEXT_TASK_ID,
        create_if_missing=create_if_missing,
        run_id_prefix="train",
    )


# ── Training sensor state ───────────────────────────────────────────

def load_training_sensor_state() -> dict:
    """Read training sensor state used to track last consumed Gold version."""
    try:
        with open(TRAINING_SENSOR_STATE_PATH) as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            return {"last_consumed_version": 0}

        # Backward compatibility for older state keys.
        if "last_consumed_version" not in payload:
            payload["last_consumed_version"] = int(payload.get("latest_version", 0) or 0)

        return payload
    except Exception:
        return {"last_consumed_version": 0}


def save_training_sensor_state(state: dict) -> None:
    os.makedirs(os.path.dirname(TRAINING_SENSOR_STATE_PATH), exist_ok=True)
    with open(TRAINING_SENSOR_STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)

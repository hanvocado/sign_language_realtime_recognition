"""
Custom Airflow sensor for the training pipeline.

Polls the Iceberg gold_training_landmarks_inventory table and fires when
a new Gold snapshot version is available (numeric version > last known).
"""

import os
import json

from airflow.sensors.base import BaseSensorOperator

from airflow.dags.training.config import ICEBERG_GOLD_TRAINING_TABLE
from airflow.dags.training.utils import build_spark_cmd, spark_query, load_gold_state


class IcebergGoldVersionSensor(BaseSensorOperator):
    """
    Succeeds when the Iceberg inventory table contains a gold_version
    higher than the last version recorded in gold_version_state.json.

    On success pushes to XCom:
        gold_version     : str  e.g. "v0003"
        gold_prefix      : str  MinIO object prefix for the snapshot
        iceberg_version  : int  raw integer version number
    """

    def poke(self, context) -> bool:
        from airflow.dags.training.config import MINIO_GOLD_ROOT_PREFIX

        gold_state   = load_gold_state()
        latest_known = int(gold_state.get("latest_version", 0))

        output_file = f"/tmp/slr_gold_sensor_{context['run_id']}.json"
        cmd = build_spark_cmd(
            "latest_version",
            [
                "--table",       ICEBERG_GOLD_TRAINING_TABLE,
                "--output-file", output_file,
            ],
        )

        try:
            spark_query(cmd, ICEBERG_GOLD_TRAINING_TABLE)
        except RuntimeError as exc:
            self.log.warning(f"Spark query failed during poke: {exc}")
            return False

        if not os.path.exists(output_file):
            self.log.info("No output file — table may be empty.")
            return False

        with open(output_file) as f:
            data = json.load(f)

        iceberg_version = int(data.get("latest_version", 0))
        self.log.info(f"Iceberg version={iceberg_version}, last known={latest_known}")

        if iceberg_version > latest_known:
            gold_version = f"v{iceberg_version:04d}"
            gold_prefix  = f"{MINIO_GOLD_ROOT_PREFIX}/{gold_version}"

            ti = context["task_instance"]
            ti.xcom_push(key="gold_version",    value=gold_version)
            ti.xcom_push(key="gold_prefix",     value=gold_prefix)
            ti.xcom_push(key="iceberg_version", value=iceberg_version)
            self.log.info(f"New Gold version detected: {gold_version}")
            return True

        return False

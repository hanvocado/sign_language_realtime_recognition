"""Utilities for writing and reading Iceberg inventory tables on MinIO.

This module uses:
- SQL catalog metadata in PostgreSQL
- Object files in MinIO (S3-compatible warehouse)
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Dict, List, Optional

import pyarrow as pa
from pyiceberg.catalog import load_catalog
from pyiceberg.exceptions import NoSuchNamespaceError, NoSuchTableError
from pyiceberg.schema import Schema
from pyiceberg.types import NestedField, StringType, LongType, TimestampType


def _build_catalog():
    postgres_user = os.environ.get("POSTGRES_USER", "postgres")
    postgres_password = os.environ.get("POSTGRES_PASSWORD", "postgres")
    postgres_host = os.environ.get("ICEBERG_POSTGRES_HOST", "postgres")
    postgres_port = os.environ.get("ICEBERG_POSTGRES_PORT", "5432")
    postgres_db = os.environ.get("ICEBERG_POSTGRES_DB", "iceberg")

    minio_endpoint = os.environ.get("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000")
    minio_access = os.environ.get("AWS_ACCESS_KEY_ID", "minioadmin")
    minio_secret = os.environ.get("AWS_SECRET_ACCESS_KEY", "minioadmin")
    warehouse = os.environ.get("ICEBERG_WAREHOUSE", "s3://data/iceberg")
    namespace = os.environ.get("ICEBERG_NAMESPACE", "sign_language")

    return load_catalog(
        f"{namespace}_catalog",
        type="sql",
        uri=f"postgresql+psycopg2://{postgres_user}:{postgres_password}@{postgres_host}:{postgres_port}/{postgres_db}",
        warehouse=warehouse,
        **{
            "s3.endpoint": minio_endpoint,
            "s3.access-key-id": minio_access,
            "s3.secret-access-key": minio_secret,
            "s3.path-style-access": "true",
            "s3.region": "us-east-1",
        },
    )


def _ensure_table(table_name: str):
    catalog = _build_catalog()
    namespace_name = os.environ.get("ICEBERG_NAMESPACE", "sign_language")
    namespace = (namespace_name,)
    identifier = (namespace_name, table_name)

    try:
        catalog.load_namespace_properties(namespace)
    except NoSuchNamespaceError:
        catalog.create_namespace(namespace)

    try:
        return catalog.load_table(identifier)
    except NoSuchTableError:
        schema = Schema(
            NestedField(field_id=1, name="run_id", field_type=StringType(), required=True),
            NestedField(field_id=2, name="object_path", field_type=StringType(), required=True),
            NestedField(field_id=3, name="file_type", field_type=StringType(), required=True),
            NestedField(field_id=4, name="label", field_type=StringType(), required=False),
            NestedField(field_id=5, name="size_bytes", field_type=LongType(), required=False),
            NestedField(field_id=6, name="etag", field_type=StringType(), required=False),
            NestedField(field_id=7, name="inserted_at", field_type=TimestampType(), required=True),
        )
        return catalog.create_table(identifier=identifier, schema=schema)


def append_inventory_rows(table_name: str, rows: List[Dict]) -> int:
    """Append inventory rows into an Iceberg table. Returns appended row count."""
    if not rows:
        return 0

    now = datetime.now(timezone.utc)
    normalized = []
    for row in rows:
        normalized.append(
            {
                "run_id": row.get("run_id"),
                "object_path": row.get("object_path"),
                "file_type": row.get("file_type"),
                "label": row.get("label"),
                "size_bytes": row.get("size_bytes"),
                "etag": row.get("etag"),
                "inserted_at": row.get("inserted_at") or now,
            }
        )

    table = _ensure_table(table_name)
    arrow_table = pa.Table.from_pylist(normalized)
    table.append(arrow_table)
    return len(normalized)


def resolve_latest_run_id(table_name: str) -> Optional[str]:
    """Return the latest run_id from an inventory table, or None if empty."""
    import pyarrow.compute as pc
    namespace = os.environ.get("ICEBERG_NAMESPACE", "sign_language")
    catalog = _build_catalog()
    table = catalog.load_table((namespace, table_name))
    # Project only the two columns needed so we avoid reading the full table
    arrow = table.scan(selected_fields=("run_id", "inserted_at")).to_arrow()
    if arrow.num_rows == 0:
        return None

    inserted_at = arrow["inserted_at"]
    max_ts = pc.max(inserted_at)
    if max_ts.is_valid is False or max_ts.as_py() is None:
        return None
    mask = pc.equal(inserted_at, max_ts)
    filtered = arrow.filter(mask)
    if filtered.num_rows == 0:
        return None
    return filtered["run_id"][0].as_py()


def get_run_rows(table_name: str, run_id: str) -> List[Dict]:
    """Fetch all rows for a specific run_id from inventory table."""
    from pyiceberg.expressions import EqualTo
    namespace = os.environ.get("ICEBERG_NAMESPACE", "sign_language")
    catalog = _build_catalog()
    table = catalog.load_table((namespace, table_name))
    arrow = table.scan(row_filter=EqualTo("run_id", run_id)).to_arrow()
    if arrow.num_rows == 0:
        return []
    return arrow.to_pylist()

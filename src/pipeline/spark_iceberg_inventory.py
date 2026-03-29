"""Spark jobs for Iceberg inventory tables on MinIO.

Commands:
- append: append inventory rows from a JSON file
- latest-run: print latest run_id in a table
- rows-by-run: export rows for a run_id to JSON file
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Optional

from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp


def build_spark(app_name: str, master: Optional[str] = None) -> SparkSession:
    postgres_user = os.environ.get("POSTGRES_USER", "postgres")
    postgres_password = os.environ.get("POSTGRES_PASSWORD", "postgres")
    postgres_host = os.environ.get("ICEBERG_POSTGRES_HOST", "postgres")
    postgres_port = os.environ.get("ICEBERG_POSTGRES_PORT", "5432")
    postgres_db = os.environ.get("ICEBERG_POSTGRES_DB", "iceberg")

    minio_endpoint = os.environ.get("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000")
    minio_access = os.environ.get("AWS_ACCESS_KEY_ID", "minioadmin")
    minio_secret = os.environ.get("AWS_SECRET_ACCESS_KEY", "minioadmin")
    warehouse = os.environ.get("ICEBERG_WAREHOUSE", "s3a://data/iceberg")
    if warehouse.startswith("s3://"):
        warehouse = warehouse.replace("s3://", "s3a://", 1)

    builder = SparkSession.builder.appName(app_name)
    if master:
        builder = builder.master(master)

    spark = (
        builder.config("spark.sql.catalog.vsl", "org.apache.iceberg.spark.SparkCatalog")
        .config("spark.sql.catalog.vsl.type", "jdbc")
        .config(
            "spark.sql.catalog.vsl.uri",
            f"jdbc:postgresql://{postgres_host}:{postgres_port}/{postgres_db}",
        )
        .config("spark.sql.catalog.vsl.jdbc.user", postgres_user)
        .config("spark.sql.catalog.vsl.jdbc.password", postgres_password)
        .config("spark.sql.catalog.vsl.warehouse", warehouse)
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.endpoint", minio_endpoint)
        .config("spark.hadoop.fs.s3a.access.key", minio_access)
        .config("spark.hadoop.fs.s3a.secret.key", minio_secret)
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .getOrCreate()
    )

    spark.sql("CREATE NAMESPACE IF NOT EXISTS vsl")
    return spark


def ensure_inventory_table(spark: SparkSession, table: str) -> None:
    def _create_table() -> None:
        spark.sql(
            f"""
            CREATE TABLE IF NOT EXISTS vsl.{table} (
              run_id STRING,
              object_path STRING,
              file_type STRING,
              label STRING,
              size_bytes BIGINT,
              etag STRING,
              inserted_at TIMESTAMP
            ) USING iceberg
            """
        )

    exists = spark.sql(f"SHOW TABLES IN vsl LIKE '{table}'").count() > 0
    if not exists:
        _create_table()
        return

    try:
        # Force metadata resolution so we can detect broken catalog pointers early.
        spark.sql(f"SELECT 1 FROM vsl.{table} LIMIT 1").collect()
    except Exception as exc:
        message = str(exc)
        missing_metadata = "FileNotFoundException" in message and "/metadata/" in message
        table_missing = "TABLE_OR_VIEW_NOT_FOUND" in message or "not found" in message.lower()

        if table_missing:
            _create_table()
            return

        if not missing_metadata:
            raise

        print(
            f"WARN: Iceberg table vsl.{table} has missing metadata. "
            "Dropping stale table entry and recreating it."
        )
        spark.sql(f"DROP TABLE IF EXISTS vsl.{table}")
        _create_table()


def cmd_append(args: argparse.Namespace) -> None:
    spark = build_spark("append-iceberg-inventory", args.master)
    ensure_inventory_table(spark, args.table)

    df = spark.read.option("multiline", "true").json(args.rows_file)
    if df.limit(1).count() == 0:
        print(f"APPENDED 0 rows to vsl.{args.table}")
        spark.stop()
        return

    if "inserted_at" not in df.columns:
        df = df.withColumn("inserted_at", current_timestamp())

    df.cache()
    row_count = df.count()
    df.writeTo(f"vsl.{args.table}").append()
    print(f"APPENDED {row_count} rows to vsl.{args.table}")
    spark.stop()


def cmd_latest_run(args: argparse.Namespace) -> None:
    spark = build_spark("latest-run-iceberg", args.master)
    ensure_inventory_table(spark, args.table)

    result = spark.sql(
        f"""
        SELECT run_id, MAX(inserted_at) AS max_ts
        FROM vsl.{args.table}
        GROUP BY run_id
        ORDER BY max_ts DESC
        LIMIT 1
        """
    ).collect()

    if result:
        print(result[0]["run_id"])
    spark.stop()


def cmd_rows_by_run(args: argparse.Namespace) -> None:
    spark = build_spark("rows-by-run-iceberg", args.master)
    ensure_inventory_table(spark, args.table)

    rows = spark.sql(
        f"""
        SELECT run_id, object_path, file_type, label, size_bytes, etag, inserted_at
        FROM vsl.{args.table}
        WHERE run_id = '{args.run_id}'
        """
    ).toJSON().collect()

    parsed = [json.loads(r) for r in rows]
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(parsed, f, ensure_ascii=False, indent=2)

    print(f"EXPORTED {len(parsed)} rows to {args.output_file}")
    spark.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Spark Iceberg inventory jobs")
    parser.add_argument("command", choices=["append", "latest-run", "rows-by-run"])
    parser.add_argument("--master", default=os.environ.get("SPARK_MASTER_URL", "spark://spark-master:7077"))
    parser.add_argument("--table", required=True)
    parser.add_argument("--rows-file")
    parser.add_argument("--run-id")
    parser.add_argument("--output-file")

    args = parser.parse_args()

    if args.command == "append":
        if not args.rows_file:
            raise ValueError("--rows-file is required for append")
        cmd_append(args)
    elif args.command == "latest-run":
        cmd_latest_run(args)
    elif args.command == "rows-by-run":
        if not args.run_id or not args.output_file:
            raise ValueError("--run-id and --output-file are required for rows-by-run")
        cmd_rows_by_run(args)


if __name__ == "__main__":
    main()

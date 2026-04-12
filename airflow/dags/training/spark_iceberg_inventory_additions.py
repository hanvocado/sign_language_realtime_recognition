"""
spark_iceberg_inventory.py  — additions for training_pipeline.py

Adds two new subcommands to whatever existing spark_iceberg_inventory.py
you already have.  Paste these handlers into the existing file's
subcommand dispatch block, or run this as a standalone script.

New subcommands
───────────────
latest_version
    Reads the gold_training_landmarks_inventory table, finds the highest
    numeric version embedded in run_id (e.g. "v0003" → 3), and writes:
        {"latest_version": 3}

snapshot_stats
    Filters the table by --gold-version, counts distinct labels and total
    files, and writes:
        {"labels": ["HELLO", "THANK_YOU", ...], "total_files": 1234}

Both commands write JSON to --output-file and exit 0 on success.
"""

import sys
import json
import argparse
import os

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# ─────────────────────────────────────────────
# Spark session (mirrors existing project setup)
# ─────────────────────────────────────────────

def _get_spark() -> SparkSession:
    aws_key    = os.environ.get("AWS_ACCESS_KEY_ID",     "minioadmin")
    aws_secret = os.environ.get("AWS_SECRET_ACCESS_KEY", "minioadmin")
    endpoint   = os.environ.get("MINIO_ENDPOINT",        "http://minio:9000")
    pg_host    = os.environ.get("ICEBERG_POSTGRES_HOST", "postgres")
    pg_port    = os.environ.get("ICEBERG_POSTGRES_PORT", "5432")
    pg_db      = os.environ.get("ICEBERG_POSTGRES_DB",   "iceberg")
    pg_user    = os.environ.get("POSTGRES_USER",         "postgres")
    pg_pw      = os.environ.get("POSTGRES_PASSWORD",     "postgres")
    namespace  = os.environ.get("ICEBERG_NAMESPACE",     "sign_language")
    bucket     = os.environ.get("MINIO_BUCKET",          "data")

    jdbc_url = (
        f"jdbc:postgresql://{pg_host}:{pg_port}/{pg_db}"
        f"?user={pg_user}&password={pg_pw}"
    )
    warehouse = f"s3a://{bucket}/lakehouse/iceberg_warehouse"

    return (
        SparkSession.builder.appName("slr_iceberg_inventory")
        .config("spark.sql.extensions",
                "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions")
        .config("spark.sql.catalog.local", "org.apache.iceberg.spark.SparkCatalog")
        .config("spark.sql.catalog.local.type", "jdbc")
        .config("spark.sql.catalog.local.uri", jdbc_url)
        .config("spark.sql.catalog.local.warehouse", warehouse)
        .config("spark.hadoop.fs.s3a.endpoint",              endpoint)
        .config("spark.hadoop.fs.s3a.access.key",            aws_key)
        .config("spark.hadoop.fs.s3a.secret.key",            aws_secret)
        .config("spark.hadoop.fs.s3a.path.style.access",     "true")
        .config("spark.hadoop.fs.s3a.impl",
                "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .getOrCreate()
    )


# ─────────────────────────────────────────────
# Subcommand: latest_version
# ─────────────────────────────────────────────

def cmd_latest_version(args, spark: SparkSession) -> None:
    """
    Find the highest gold version number in the inventory table.
    run_id values look like "v0003"; we extract the integer suffix.
    """
    namespace = os.environ.get("ICEBERG_NAMESPACE", "sign_language")
    table_ref = f"local.{namespace}.{args.table}"

    try:
        df = spark.table(table_ref)
    except Exception:
        # Table does not exist yet — version 0
        result = {"latest_version": 0}
        _write_output(args.output_file, result)
        return

    # Filter to gold_landmark_snapshot rows and extract the version integer
    version_df = (
        df.filter(F.col("file_type") == "gold_landmark_snapshot")
        .select(
            F.regexp_extract(F.col("run_id"), r"v(\d+)", 1)
            .cast("int")
            .alias("ver")
        )
        .agg(F.max("ver").alias("max_ver"))
    )

    row = version_df.collect()
    latest = int(row[0]["max_ver"] or 0) if row else 0
    _write_output(args.output_file, {"latest_version": latest})
    print(f"latest_version={latest}")


# ─────────────────────────────────────────────
# Subcommand: snapshot_stats
# ─────────────────────────────────────────────

def cmd_snapshot_stats(args, spark: SparkSession) -> None:
    """
    Return label list and total file count for a specific gold_version.
    gold_version is the run_id value, e.g. "v0003".
    """
    namespace = os.environ.get("ICEBERG_NAMESPACE", "sign_language")
    table_ref = f"local.{namespace}.{args.table}"

    try:
        df = spark.table(table_ref)
    except Exception:
        _write_output(args.output_file, {"labels": [], "total_files": 0})
        return

    snapshot_df = df.filter(
        (F.col("file_type") == "gold_landmark_snapshot") &
        (F.col("run_id")    == args.gold_version)
    )

    total = snapshot_df.count()
    labels = sorted([
        row["label"]
        for row in snapshot_df.select("label").distinct().collect()
        if row["label"]
    ])

    result = {"labels": labels, "total_files": total, "gold_version": args.gold_version}
    _write_output(args.output_file, result)
    print(f"snapshot_stats: labels={len(labels)}, total_files={total}")


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _write_output(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ─────────────────────────────────────────────
# CLI dispatch  (add to existing script's main)
# ─────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Iceberg inventory Spark job")
    sub = p.add_subparsers(dest="command", required=True)

    # latest_version
    sv = sub.add_parser("latest_version")
    sv.add_argument("--table",       required=True)
    sv.add_argument("--output-file", required=True)

    # snapshot_stats
    ss = sub.add_parser("snapshot_stats")
    ss.add_argument("--table",        required=True)
    ss.add_argument("--gold-version", required=True)
    ss.add_argument("--output-file",  required=True)

    # Existing subcommands (checkpoint, append) should be preserved in the
    # real file — add only the new ones here as an extension.

    args = p.parse_args()
    spark = _get_spark()

    try:
        if args.command == "latest_version":
            cmd_latest_version(args, spark)
        elif args.command == "snapshot_stats":
            cmd_snapshot_stats(args, spark)
        else:
            print(f"Unknown command: {args.command}", file=sys.stderr)
            sys.exit(1)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()

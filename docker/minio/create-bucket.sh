#!/bin/bash
# Configure MinIO Client
mc alias set minioserver http://minio:9000 ${MINIO_ROOT_USER} ${MINIO_ROOT_PASSWORD}

# Create required buckets (idempotent)
mc mb --ignore-existing minioserver/mlflow
mc mb --ignore-existing minioserver/data

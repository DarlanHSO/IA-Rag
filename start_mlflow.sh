#!/bin/bash

python3 - <<'PYEOF'
import boto3
import os

s3 = boto3.client(
    "s3",
    endpoint_url=os.environ.get("MLFLOW_S3_ENDPOINT_URL"),
    aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
)

bucket = os.environ.get("BUCKET_NAME")

try:
    s3.create_bucket(Bucket=bucket)
    print("Bucket criado com sucesso!")
except Exception as e:
    if "BucketAlreadyOwnedByYou" in str(e) or "BucketAlreadyExists" in str(e):
        print("Bucket já existe.")
    else:
        print(f"Erro ao criar bucket: {e}")
PYEOF

exec mlflow server \
  --backend-store-uri "$MLFLOW_TRACKING_URI" \
  --default-artifact-root "s3://$BUCKET_NAME/" \
  --host 0.0.0.0 \
  --port 3000
import os
import tempfile
import pandas as pd
from pathlib import Path
from minio import Minio
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

MINIO_HOST       = os.getenv("MINIO_HOST", "localhost")
MINIO_PORT       = os.getenv("MINIO_PORT", "9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "")

client = Minio(
    f"{MINIO_HOST}:{MINIO_PORT}",
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False,
)

print("iniciando agregação gold...")

with tempfile.TemporaryDirectory() as temp_dir:
    silver_path = os.path.join(temp_dir, "trending_silver.parquet")

    client.fget_object("silver", "trending_silver.parquet", silver_path)
    df_gold = pd.read_parquet(silver_path)

    if "channel_id" in df_gold.columns:
        df_gold = df_gold.drop(columns=["channel_id"])

    # texto_rag: string concatenada usada como input do embedding
    df_gold["texto_rag"] = (
        "título: " + df_gold["title"].astype(str) +
        ". categoria: " + df_gold["category_name"].astype(str) +
        ". canal: " + df_gold["channel_title"].astype(str) +
        ". views: " + df_gold["views"].astype(str) +
        ". tags: " + df_gold["tags"].astype(str) +
        ". descrição: " + df_gold["description"].astype(str)
    )

    gold_path = os.path.join(temp_dir, "trending_gold.parquet")
    df_gold.to_parquet(gold_path, index=False)
    
    client.fput_object("gold", "trending_gold.parquet", gold_path)

print("agregação gold finalizada.")
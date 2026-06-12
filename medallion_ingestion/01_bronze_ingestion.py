import os
import tempfile
from pathlib import Path
from minio import Minio
from kaggle.api.kaggle_api_extended import KaggleApi
from dotenv import load_dotenv
from db_metadata import log_ingestion

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

for bucket in ["bronze", "silver", "gold"]:
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)

api = KaggleApi()
api.authenticate()

DATASET_ID = "bsthere/youtube-trending-videos-stats-2026"

with tempfile.TemporaryDirectory() as temp_dir:
    api.dataset_download_files(DATASET_ID, path=temp_dir, unzip=True)

    arquivos_validos = [
        "IN_Trending.csv", "RU_Trending.csv", "JP_Trending.csv",
        "GB_Trending.csv", "FR_Trending.csv", "DE_Trending.csv",
        "US_Trending.csv", "CA_Trending.csv", "MX_Trending.csv",
        "BR_Trending.csv", "KR_Trending.csv"
    ]

    for filename in os.listdir(temp_dir):
        if filename in arquivos_validos:
            file_path = os.path.join(temp_dir, filename)
            client.fput_object("bronze", filename, file_path)
            print(f"salvo na bronze: {filename}")

categories_path = BASE_DIR / "data" / "bronze" / "categories_official.json"

if categories_path.exists():
    client.fput_object("bronze", "categories_official.json", str(categories_path))
    print("salvo na bronze: categories_official.json")
else:
    print(f"aviso: dicionário não encontrado em {categories_path}. execute get_categorias.py antes.")

arquivos_bronze = sum(1 for _ in client.list_objects("bronze"))
log_ingestion("bronze", arquivos_bronze, source=DATASET_ID)
print("ingestão bronze finalizada.")

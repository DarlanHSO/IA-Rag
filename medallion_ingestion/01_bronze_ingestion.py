import os
import tempfile
from pathlib import Path
from minio import Minio
from kaggle.api.kaggle_api_extended import KaggleApi
from dotenv import load_dotenv

# impede erro de "file not found" caso o script seja executado a partir de outro diretório
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

client = Minio("localhost:9000", access_key="minio", secret_key="minio123", secure=False)

# garante a resiliência do pipeline caso o data lake tenha sido recriado do zero
for bucket in ["bronze", "silver", "gold"]:
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)

api = KaggleApi()
api.authenticate()

DATASET_ID = "bsthere/youtube-trending-videos-stats-2026"

# evita o acúmulo de artefatos residuais (.zip) e previne o esgotamento do armazenamento em disco local
with tempfile.TemporaryDirectory() as temp_dir:
    api.dataset_download_files(DATASET_ID, path=temp_dir, unzip=True)
    
    arquivos_validos = [
        "IN_Trending.csv", "RU_Trending.csv", "JP_Trending.csv", 
        "GB_Trending.csv", "FR_Trending.csv", "DE_Trending.csv", 
        "US_Trending.csv", "CA_Trending.csv", "MX_Trending.csv", 
        "BR_Trending.csv", "KR_Trending.csv"
    ]
    
    # restringe a ingestão aos dados relevantes para o escopo do rag, otimizando o storage na camada bronze
    for filename in os.listdir(temp_dir):
        if filename in arquivos_validos:
            file_path = os.path.join(temp_dir, filename)
            client.fput_object("bronze", filename, file_path)
            print(f"salvo na bronze: {filename}")

categories_path = BASE_DIR / "data" / "bronze" / "categories_official.json"

# centraliza o dicionário no data lake para atuar como fonte única da verdade para as próximas camadas
if categories_path.exists():
    client.fput_object("bronze", "categories_official.json", str(categories_path))
    print("salvo na bronze: categories_official.json")
else:
    print(f"aviso: dicionário não encontrado em {categories_path}. execute get_categorias.py antes.")

print("ingestão bronze finalizada.")

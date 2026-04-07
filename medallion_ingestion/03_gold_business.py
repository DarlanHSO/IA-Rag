import os
import tempfile
import pandas as pd
from minio import Minio

client = Minio("localhost:9000", access_key="minio", secret_key="minio123", secure=False)

print("iniciando agregação gold...")

# centraliza as operações em diretório efêmero para garantir a premissa de processamento stateless
with tempfile.TemporaryDirectory() as temp_dir:
    silver_path = os.path.join(temp_dir, "trending_silver.parquet")
    
    client.fget_object("silver", "trending_silver.parquet", silver_path)
    df_gold = pd.read_parquet(silver_path)
    
    # elimina identificadores estruturais que não agregam valor semântico ao banco de vetores
    if "channel_id" in df_gold.columns:
        df_gold = df_gold.drop(columns=["channel_id"])

    # consolida os metadados em uma string densa para otimizar a indexação e a recuperação pelo modelo de linguagem
    df_gold["texto_rag"] = (
        "título: " + df_gold["title"].astype(str) +
        ". categoria: " + df_gold["category_name"].astype(str) +
        ". canal: " + df_gold["channel_title"].astype(str) +
        ". tags: " + df_gold["tags"].astype(str) +
        ". descrição: " + df_gold["description"].astype(str)
    )

    gold_path = os.path.join(temp_dir, "trending_gold.parquet")
    df_gold.to_parquet(gold_path, index=False)
    
    client.fput_object("gold", "trending_gold.parquet", gold_path)

print("agregação gold finalizada.")
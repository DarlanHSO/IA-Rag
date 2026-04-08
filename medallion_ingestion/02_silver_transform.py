import os
import json
import tempfile
import pandas as pd
from minio import Minio

client = Minio("localhost:9000", access_key="minio", secret_key="minio123", secure=False)

print("iniciando transformação silver...")

with tempfile.TemporaryDirectory() as temp_dir:
    client.fget_object("bronze", "categories_official.json", os.path.join(temp_dir, "categories.json"))
    with open(os.path.join(temp_dir, "categories.json"), "r", encoding="utf-8") as f:
        category_map = {int(k): v for k, v in json.load(f).items()}

    objetos_bronze = client.list_objects("bronze")
    dfs = []
    
    for obj in objetos_bronze:
        # ignora o dicionário de categorias e o arquivo data_dictionary.csv que causou o erro
        if "categories_official.json" in obj.object_name or "data_dictionary" in obj.object_name:
            continue
            
        if not obj.object_name.lower().endswith(".csv"):
            continue
            
        print(f"lendo arquivo: {obj.object_name}")
        
        file_path = os.path.join(temp_dir, obj.object_name)
        client.fget_object("bronze", obj.object_name, file_path)
        
        df = pd.read_csv(file_path, encoding="utf-8-sig")
        df.columns = [c.strip() for c in df.columns]
        
        df["country"] = obj.object_name.split("_")[0]
        df["category_name"] = df["category_id"].map(category_map).fillna("não informado")

        if "dislikes" in df.columns:
            df = df.drop(columns=["dislikes"])

        df = df[df["views"] > 0]
        
        df["tags"] = df["tags"].replace("[none]", "não informado").fillna("não informado")
        df["description"] = df["description"].fillna("não informado")
        df["tags"] = df["tags"].str.replace("|", ", ", regex=False)
        
        df["trending_date"] = pd.to_datetime(df["trending_date"], format="%y.%d.%m", errors="coerce")
        df["publish_time"] = pd.to_datetime(df["publish_time"], errors="coerce")
        df["views"] = pd.to_numeric(df["views"], errors="coerce").fillna(0).astype(int)
        df["likes"] = pd.to_numeric(df["likes"], errors="coerce").fillna(0).astype(int)
        
        col_comments = "comments" if "comments" in df.columns else "comment_count"
        if col_comments in df.columns:
            df[col_comments] = pd.to_numeric(df[col_comments], errors="coerce").fillna(0).astype(int)
            
        df = df.drop_duplicates()
        dfs.append(df)

    if not dfs:
        print("erro: nenhum arquivo csv de tendência foi processado.")
        exit(1)

    df_silver = pd.concat(dfs, ignore_index=True)
    silver_path = os.path.join(temp_dir, "trending_silver.parquet")
    df_silver.to_parquet(silver_path, index=False)
    client.fput_object("silver", "trending_silver.parquet", silver_path)

print("transformação silver finalizada.")
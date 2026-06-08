import os
from pathlib import Path
import pandas as pd
import ollama
from dotenv import load_dotenv

from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility
)

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

MILVUS_HOST     = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT     = os.getenv("MILVUS_PORT", "19530")
COLLECTION_NAME = os.getenv("MILVUS_COLLECTION", "youtube_trending")
EMBED_MODEL     = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# =========================================================
# PATHS
# =========================================================

DATASET_PATH = BASE_DIR / "data" / "gold" / "trending_gold.parquet"

print(f"Dataset path: {DATASET_PATH}")

# =========================================================
# Ler dataset parquet
# =========================================================

LIMIT = 500

df = pd.read_parquet(DATASET_PATH)
df = df.head(LIMIT)

print("Dataset carregado!")
print(f"Quantidade de linhas: {len(df)}")

# =========================================================
# Conectar no Milvus
# =========================================================

connections.connect(
    alias="default",
    host=MILVUS_HOST,
    port=MILVUS_PORT,
)

# =========================================================
# Gerar embeddings
# =========================================================


texts = []
embeddings = []

print("Gerando embeddings...")

for index, row in df.iterrows():

    texto_rag = row["texto_rag"]

    texts.append(texto_rag)

    response = ollama.embeddings(
        model=EMBED_MODEL,
        prompt=texto_rag
    )

    embeddings.append(
        response["embedding"]
    )

    if index % 50 == 0:
        print(f"{index}/{LIMIT} embeddings gerados...")

print("Embeddings finalizados!")

# =========================================================
# Criar collection se não existir
# =========================================================

if not utility.has_collection(COLLECTION_NAME):

    fields = [

        FieldSchema(
            name="id",
            dtype=DataType.INT64,
            is_primary=True,
            auto_id=True
        ),

        FieldSchema(
            name="video_id",
            dtype=DataType.VARCHAR,
            max_length=100
        ),

        FieldSchema(
            name="title",
            dtype=DataType.VARCHAR,
            max_length=1000
        ),

        FieldSchema(
            name="channel_title",
            dtype=DataType.VARCHAR,
            max_length=500
        ),

        FieldSchema(
            name="category_name",
            dtype=DataType.VARCHAR,
            max_length=200
        ),

        FieldSchema(
            name="country",
            dtype=DataType.VARCHAR,
            max_length=50
        ),

        FieldSchema(
            name="texto_rag",
            dtype=DataType.VARCHAR,
            max_length=20000
        ),

        FieldSchema(
            name="embedding",
            dtype=DataType.FLOAT_VECTOR,
            dim=len(embeddings[0])
        )
    ]

    schema = CollectionSchema(
        fields=fields,
        description="Youtube trending videos embeddings"
    )

    collection = Collection(
        name=COLLECTION_NAME,
        schema=schema
    )

    # =====================================================
    # Índice vetorial
    # =====================================================

    index_params = {
        "metric_type": "COSINE",

        "index_type": "HNSW",

        "params": {
            "M": 16,
            "efConstruction": 200
        }
    }

    collection.create_index(
        field_name="embedding",
        index_params=index_params
    )

    print("Collection criada!")

else:
    collection = Collection(COLLECTION_NAME)

# =========================================================
# Inserir dados
# =========================================================

data = [

    # video_id
    df["video_id"].astype(str).tolist(),

    # title
    df["title"].astype(str).tolist(),

    # channel_title
    df["channel_title"].astype(str).tolist(),

    # category_name
    df["category_name"].astype(str).tolist(),

    # country
    df["country"].astype(str).tolist(),

    # texto_rag
    texts,

    # embeddings
    embeddings
]

print("Inserindo no Milvus...")

result = collection.insert(data)

collection.flush()

print("===================================")
print("Dados inseridos com sucesso!")
print(f"Quantidade inserida: {len(result.primary_keys)}")
print("===================================")
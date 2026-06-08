import os
from pathlib import Path

import ollama
from dotenv import load_dotenv

from pymilvus import (
    connections,
    Collection
)

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# =========================================================
# CONFIG
# =========================================================

COLLECTION_NAME = os.getenv("MILVUS_COLLECTION", "youtube_trending")
EMBED_MODEL     = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
MILVUS_HOST     = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT     = os.getenv("MILVUS_PORT", "19530")

# =========================================================
# Conectar no Milvus
# =========================================================

connections.connect(
    alias="default",
    host=MILVUS_HOST,
    port=MILVUS_PORT,
)

collection = Collection(COLLECTION_NAME)
collection.load()

print("Milvus conectado e collection carregada!")

# =========================================================
# INPUT DO USUÁRIO
# =========================================================

query = input("\nDigite sua busca: ")

# Exemplo:
# "videos virais de minecraft com alta retenção"

# =========================================================
# Gerar embedding da query
# =========================================================

response = ollama.embeddings(
    model=EMBED_MODEL,
    prompt=query
)

query_embedding = response["embedding"]

# =========================================================
# Buscar no Milvus
# =========================================================

results = collection.search(
    data=[query_embedding],

    anns_field="embedding",

    param={
        "metric_type": "COSINE",
        "params": {
            "ef": 64
        }
    },

    limit=5,

    output_fields=[
        "title",
        "channel_title",
        "category_name"
    ]
)

# =========================================================
# Mostrar resultados
# =========================================================

print("\n===================================")
print("RESULTADOS ENCONTRADOS")
print("===================================\n")

for hits in results:

    for hit in hits:

        print(f"Score (similaridade): {hit.distance:.4f}")
        print(f"Título: {hit.entity.get('title')}")
        print(f"Canal: {hit.entity.get('channel_title')}")
        print(f"Categoria: {hit.entity.get('category_name')}")
        print("-----------------------------------")
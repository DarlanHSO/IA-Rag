import ollama

from pymilvus import (
    connections,
    Collection
)

# =========================================================
# CONFIG
# =========================================================

COLLECTION_NAME = "youtube_trending"
EMBED_MODEL = "nomic-embed-text"

# =========================================================
# Conectar no Milvus
# =========================================================

connections.connect(
    alias="default",
    host="localhost",
    port="19530"
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
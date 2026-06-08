import os
from pathlib import Path

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility
)

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")


model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


sentences = [
    "The weather is lovely today.",
    "It's so sunny outside!",
    "He drove to the stadium.",
]

# ==========================================
# Gerar embeddings
# ==========================================

embeddings = model.encode(
    sentences,
    normalize_embeddings=True
)

print(embeddings.shape)
# [3, 384]

# ==========================================
# Conectar no Milvus
# ==========================================

connections.connect(
    alias="default",
    host=MILVUS_HOST,
    port=MILVUS_PORT,
)

COLLECTION_NAME = "sentences"

if not utility.has_collection(COLLECTION_NAME):

    fields = [
        FieldSchema(
            name="id",
            dtype=DataType.INT64,
            is_primary=True,
            auto_id=True
        ),

        FieldSchema(
            name="text",
            dtype=DataType.VARCHAR,
            max_length=1000
        ),

        FieldSchema(
            name="embedding",
            dtype=DataType.FLOAT_VECTOR,
            dim=384
        )
    ]

    schema = CollectionSchema(
        fields=fields,
        description="Sentence embeddings"
    )

    collection = Collection(
        name=COLLECTION_NAME,
        schema=schema
    )

    # ==========================================
    # Criar índice vetorial
    # ==========================================

    index_params = {
        "metric_type": "COSINE",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128}
    }

    collection.create_index(
        field_name="embedding",
        index_params=index_params
    )

else:
    collection = Collection(COLLECTION_NAME)

# ==========================================
# Inserir dados
# ==========================================

data = [
    sentences,
    embeddings.tolist()
]

insert_result = collection.insert(data)

collection.flush()

print("Dados inseridos com sucesso!")
print(insert_result.primary_keys)

# ==========================================
# Carregar collection para busca
# ==========================================

collection.load()

# ==========================================
# Buscar similaridade
# ==========================================

query = "It is a beautiful sunny day"

query_embedding = model.encode(
    [query],
    normalize_embeddings=True
)

results = collection.search(
    data=query_embedding.tolist(),
    anns_field="embedding",
    param={
        "metric_type": "COSINE",
        "params": {"nprobe": 10}
    },
    limit=2,
    output_fields=["text"]
)

# ==========================================
# Mostrar resultados
# ==========================================

for hits in results:
    for hit in hits:
        print("Texto:", hit.entity.get("text"))
        print("Distância:", hit.distance)
        print("------")
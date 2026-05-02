import ollama

from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility
)

# ==========================================
# Modelo de embedding do Ollama
# ==========================================

OLLAMA_EMBED_MODEL = "nomic-embed-text"

# ==========================================
# Textos
# ==========================================

sentences = [
    "The weather is lovely today.",
    "It's so sunny outside!",
    "He drove to the stadium.",
]

# ==========================================
# Gerar embeddings usando Ollama
# ==========================================

embeddings = []

for sentence in sentences:

    response = ollama.embeddings(
        model=OLLAMA_EMBED_MODEL,
        prompt=sentence
    )

    embeddings.append(
        response["embedding"]
    )

print(f"Quantidade de embeddings: {len(embeddings)}")
print(f"Dimensão do embedding: {len(embeddings[0])}")

# ==========================================
# Conectar no Milvus
# ==========================================

connections.connect(
    alias="default",
    host="localhost",  # use "milvus" se estiver dentro do docker
    port="19530"
)

COLLECTION_NAME = "sentences"

# ==========================================
# Criar collection se não existir
# ==========================================

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

            # nomic-embed-text normalmente usa 768 dimensões
            dim=len(embeddings[0])
        )
    ]

    schema = CollectionSchema(
        fields=fields,
        description="Sentence embeddings with Ollama"
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
    embeddings
]

insert_result = collection.insert(data)

collection.flush()

print("Dados inseridos com sucesso!")
print(insert_result.primary_keys)

# ==========================================
# Carregar collection
# ==========================================

collection.load()

# ==========================================
# Query
# ==========================================

query = "It is a beautiful sunny day"

# ==========================================
# Embedding da query usando Ollama
# ==========================================

query_embedding = ollama.embeddings(
    model=OLLAMA_EMBED_MODEL,
    prompt=query
)["embedding"]

# ==========================================
# Busca vetorial
# ==========================================

results = collection.search(
    data=[query_embedding],
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
import math
import time
import ollama
import pandas as pd

from pathlib import Path

from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed
)

from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility
)

# =========================================================
# CONFIG
# =========================================================

BATCH_SIZE = 300

THREADS = 3

EMBED_MODEL = "nomic-embed-text"

COLLECTION_NAME = "youtube_trending"

# =========================================================
# DATASET
# =========================================================

print("\n===================================")
print("INICIANDO PIPELINE")
print("===================================\n")

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = (
    BASE_DIR /
    "data" /
    "gold" /
    "trending_gold.parquet"
)

print(f"Dataset localizado em:")
print(DATASET_PATH)

print("\nLendo parquet...")

df = pd.read_parquet(DATASET_PATH)

df = df.fillna("")

print("Dataset carregado com sucesso!")

print(f"Quantidade total de registros: {len(df)}")

# =========================================================
# MILVUS
# =========================================================

print("\n===================================")
print("CONECTANDO NO MILVUS")
print("===================================\n")

connections.connect(
    alias="default",
    host="localhost",
    port="19530"
)

print("Conectado no Milvus!")

# =========================================================
# CRIAR COLLECTION
# =========================================================

print("\n===================================")
print("VERIFICANDO COLLECTION")
print("===================================\n")

if not utility.has_collection(COLLECTION_NAME):

    print("Collection não existe.")
    print("Criando collection...")

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
            dim=768
        )
    ]

    schema = CollectionSchema(
        fields=fields,
        description="Youtube trending embeddings"
    )

    collection = Collection(
        name=COLLECTION_NAME,
        schema=schema
    )

    print("Criando índice vetorial...")

    collection.create_index(
        field_name="embedding",

        index_params={
            "metric_type": "COSINE",

            "index_type": "HNSW",

            "params": {
                "M": 16,
                "efConstruction": 200
            }
        }
    )

    print("Índice criado!")

    print("Collection criada com sucesso!")

else:

    print("Collection já existe.")

    collection = Collection(COLLECTION_NAME)

# =========================================================
# MOSTRAR SCHEMA
# =========================================================

print("\n===================================")
print("SCHEMA DA COLLECTION")
print("===================================\n")

for field in collection.schema.fields:
    print(f"- {field.name}")

# =========================================================
# WORKER
# =========================================================

def process_chunk(chunk_df, worker_id):

    print(
        f"[WORKER {worker_id}] "
        f"Iniciando processamento "
        f"({len(chunk_df)} registros)"
    )

    worker_start = time.time()

    video_ids = []
    titles = []
    channels = []
    categories = []
    countries = []
    texts = []
    embeddings = []

    total = len(chunk_df)

    for idx, (_, row) in enumerate(chunk_df.iterrows()):

        text = str(row["texto_rag"])

        try:

            response = ollama.embeddings(
                model=EMBED_MODEL,
                prompt=text
            )

            video_ids.append(
                str(row["video_id"])
            )

            titles.append(
                str(row["title"])
            )

            channels.append(
                str(row["channel_title"])
            )

            categories.append(
                str(row["category_name"])
            )

            countries.append(
                str(row["country"])
            )

            texts.append(text)

            embeddings.append(
                response["embedding"]
            )

        except Exception as e:

            print(
                f"[WORKER {worker_id}] "
                f"ERRO embedding: {e}"
            )

        # progresso do worker
        if idx % 10 == 0:

            print(
                f"[WORKER {worker_id}] "
                f"{idx}/{total} embeddings gerados"
            )

    worker_end = time.time()

    print(
        f"[WORKER {worker_id}] "
        f"Finalizado em "
        f"{worker_end - worker_start:.2f}s"
    )

    return {
        "video_ids": video_ids,
        "titles": titles,
        "channels": channels,
        "categories": categories,
        "countries": countries,
        "texts": texts,
        "embeddings": embeddings
    }

# =========================================================
# TOTAL BATCHES
# =========================================================

total_batches = math.ceil(
    len(df) / BATCH_SIZE
)

print("\n===================================")
print("CONFIGURAÇÃO DO PIPELINE")
print("===================================\n")

print(f"Batch size: {BATCH_SIZE}")
print(f"Threads: {THREADS}")
print(f"Total batches: {total_batches}")

# =========================================================
# PIPELINE
# =========================================================

pipeline_start = time.time()

for batch_index in range(total_batches):

    print("\n===================================")
    print(
        f"INICIANDO BATCH "
        f"{batch_index + 1}/{total_batches}"
    )
    print("===================================\n")

    batch_start_time = time.time()

    # -----------------------------------------------------
    # Slice batch
    # -----------------------------------------------------

    start = batch_index * BATCH_SIZE
    end = start + BATCH_SIZE

    print(f"Slice dataset: {start} -> {end}")

    batch_df = df.iloc[start:end]

    print(
        f"Registros neste batch: "
        f"{len(batch_df)}"
    )

    # -----------------------------------------------------
    # Split em threads
    # -----------------------------------------------------

    print("\nDividindo registros entre workers...")

    split_dfs = [
        batch_df.iloc[i::THREADS]
        for i in range(THREADS)
    ]

    for i, split in enumerate(split_dfs):

        print(
            f"Worker {i}: "
            f"{len(split)} registros"
        )

    all_video_ids = []
    all_titles = []
    all_channels = []
    all_categories = []
    all_countries = []
    all_texts = []
    all_embeddings = []

    # -----------------------------------------------------
    # Thread pool
    # -----------------------------------------------------

    print("\nIniciando ThreadPoolExecutor...")

    with ThreadPoolExecutor(
        max_workers=THREADS
    ) as executor:

        futures = [

            executor.submit(
                process_chunk,
                split_df,
                worker_id
            )

            for worker_id, split_df
            in enumerate(split_dfs)
        ]

        for future in as_completed(futures):

            print(
                "\nWorker concluído!"
            )

            result = future.result()

            all_video_ids.extend(
                result["video_ids"]
            )

            all_titles.extend(
                result["titles"]
            )

            all_channels.extend(
                result["channels"]
            )

            all_categories.extend(
                result["categories"]
            )

            all_countries.extend(
                result["countries"]
            )

            all_texts.extend(
                result["texts"]
            )

            all_embeddings.extend(
                result["embeddings"]
            )

            print(
                f"Embeddings acumulados: "
                f"{len(all_embeddings)}"
            )

    # =====================================================
    # INSERT MILVUS
    # =====================================================

    print("\n===================================")
    print("INSERINDO NO MILVUS")
    print("===================================\n")

    print(
        f"Quantidade embeddings: "
        f"{len(all_embeddings)}"
    )

    insert_start = time.time()

    data = [

        all_video_ids,

        all_titles,

        all_channels,

        all_categories,

        all_countries,

        all_texts,

        all_embeddings
    ]

    print("Executando insert...")

    collection.insert(data)

    print("Executando flush...")

    collection.flush()

    insert_end = time.time()

    print(
        f"Insert finalizado em "
        f"{insert_end - insert_start:.2f}s"
    )

    batch_end_time = time.time()

    print("\n===================================")
    print(
        f"BATCH {batch_index + 1} FINALIZADO"
    )
    print("===================================\n")

    print(
        f"Tempo batch: "
        f"{batch_end_time - batch_start_time:.2f}s"
    )

    print(
        f"Total embeddings inseridos: "
        f"{len(all_embeddings)}"
    )

pipeline_end = time.time()

# =========================================================
# FINAL
# =========================================================

print("\n===================================")
print("PIPELINE FINALIZADA")
print("===================================\n")

print(
    f"Tempo total: "
    f"{pipeline_end - pipeline_start:.2f}s"
)

print(
    f"Total registros processados: "
    f"{len(df)}"
)
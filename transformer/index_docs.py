import os
import io
from pathlib import Path
from dotenv import load_dotenv
import ollama
from minio import Minio
from pymilvus import connections, Collection, utility

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

EMBED_MODEL      = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
COLLECTION_NAME  = os.getenv("MILVUS_COLLECTION", "youtube_trending")
MILVUS_HOST      = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT      = os.getenv("MILVUS_PORT", "19530")
MINIO_HOST       = os.getenv("MINIO_HOST", "localhost")
MINIO_PORT       = os.getenv("MINIO_PORT", "9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "")

MIN_CHUNK_CHARS = 60

_minio = Minio(
    f"{MINIO_HOST}:{MINIO_PORT}",
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False,
)

connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)

if not utility.has_collection(COLLECTION_NAME):
    print(f"Collection '{COLLECTION_NAME}' não encontrada. Execute make index primeiro.")
    exit(1)

collection = Collection(COLLECTION_NAME)
print(f"Collection '{COLLECTION_NAME}' encontrada. Indexando documentação por chunks...\n")

BUCKETS = ["bronze", "silver", "gold"]

video_ids, titles, channels, categories, countries, texts, embeddings = [], [], [], [], [], [], []


def chunk_text(text: str) -> list[str]:
    """Divide o texto em parágrafos, filtrando chunks muito curtos."""
    chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
    return [c for c in chunks if len(c) >= MIN_CHUNK_CHARS]


for bucket in BUCKETS:
    objects = list(_minio.list_objects(bucket))
    txt_files = [o for o in objects if o.object_name.endswith(".txt")]

    for obj in txt_files:
        response = _minio.get_object(bucket, obj.object_name)
        content = response.read().decode("utf-8")
        response.close()

        chunks = chunk_text(content)
        print(f"Indexando {bucket}/{obj.object_name} → {len(chunks)} chunks")

        for i, chunk in enumerate(chunks):
            embed_resp = ollama.embeddings(model=EMBED_MODEL, prompt=chunk)

            doc_id = f"doc_{bucket}_{obj.object_name.replace('.txt', '')}_{i}"
            video_ids.append(doc_id)
            titles.append(f"Documentação {bucket.capitalize()} — {obj.object_name} (chunk {i+1})")
            channels.append("Plataforma RAG")
            categories.append("Documentação")
            countries.append("BR")
            texts.append(chunk[:19990])
            embeddings.append(embed_resp["embedding"])

if not embeddings:
    print("Nenhum documento .txt encontrado nos buckets.")
    exit(0)

collection.insert([video_ids, titles, channels, categories, countries, texts, embeddings])
collection.flush()

print(f"\n{len(embeddings)} chunk(s) indexado(s) com sucesso.")

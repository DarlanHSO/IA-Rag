import io
import os
from pathlib import Path
from minio import Minio
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = Path(__file__).resolve().parent / "docs"

load_dotenv(BASE_DIR / ".env")

client = Minio(
    f"{os.getenv('MINIO_HOST', 'localhost')}:{os.getenv('MINIO_PORT', '9000')}",
    access_key=os.getenv("MINIO_ACCESS_KEY", ""),
    secret_key=os.getenv("MINIO_SECRET_KEY", ""),
    secure=False,
)


def upload_doc(bucket: str, filename: str, content: str):
    data = content.encode("utf-8")
    client.put_object(
        bucket_name=bucket,
        object_name=filename,
        data=io.BytesIO(data),
        length=len(data),
        content_type="text/plain",
    )
    print(f"  Enviado: {bucket}/{filename}")


print("Enviando documentação para os buckets MinIO...")

for bucket_dir in sorted(DOCS_DIR.iterdir()):
    if not bucket_dir.is_dir():
        continue
    bucket = bucket_dir.name
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)
    for doc_file in sorted(bucket_dir.glob("*.txt")):
        upload_doc(bucket, doc_file.name, doc_file.read_text(encoding="utf-8"))

print("Documentação enviada com sucesso.")

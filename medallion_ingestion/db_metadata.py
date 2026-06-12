import os
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv
import sqlalchemy as sa

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

_engine = None


def _get_engine():
    global _engine
    if _engine is None:
        url = (
            f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}"
            f"@{os.getenv('POSTGRES_HOST', 'postgres')}:{os.getenv('POSTGRES_PORT', '5432')}"
            f"/{os.getenv('POSTGRES_DB')}"
        )
        _engine = sa.create_engine(url)
        with _engine.connect() as conn:
            conn.execute(sa.text("""
                CREATE TABLE IF NOT EXISTS dataset_versions (
                    id          SERIAL PRIMARY KEY,
                    layer       VARCHAR(10)  NOT NULL,
                    source      VARCHAR(255),
                    record_count INTEGER,
                    ingested_at TIMESTAMP    NOT NULL DEFAULT NOW(),
                    status      VARCHAR(20)  NOT NULL DEFAULT 'success'
                )
            """))
            conn.commit()
    return _engine


def log_ingestion(layer: str, record_count: int, source: str = None):
    try:
        engine = _get_engine()
        with engine.connect() as conn:
            conn.execute(sa.text("""
                INSERT INTO dataset_versions (layer, source, record_count, ingested_at)
                VALUES (:layer, :source, :record_count, :ingested_at)
            """), {
                "layer":        layer,
                "source":       source,
                "record_count": record_count,
                "ingested_at":  datetime.now(timezone.utc),
            })
            conn.commit()
        print(f"[postgres] versão registrada: {layer} — {record_count} registros")
    except Exception as e:
        print(f"[postgres] aviso: não foi possível registrar metadados — {e}")

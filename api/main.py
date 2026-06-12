import os
import uuid
import time
import tempfile
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import mlflow
from fastapi import FastAPI, HTTPException, Query
from minio import Minio
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pymilvus import connections, Collection
from ollama import Client as OllamaClient

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

MILVUS_HOST      = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT      = os.getenv("MILVUS_PORT", "19530")
COLLECTION_NAME  = os.getenv("MILVUS_COLLECTION", "youtube_trending")
EMBED_MODEL      = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
LLM_MODEL        = os.getenv("OLLAMA_LLM_MODEL", "phi3:mini")
MINIO_HOST       = os.getenv("MINIO_HOST", "localhost")
MINIO_PORT       = os.getenv("MINIO_PORT", "9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "")
MLFLOW_URI       = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:3000")
OLLAMA_HOST      = os.getenv("OLLAMA_HOST", "http://localhost:11434")

AVAILABLE_MODELS = list(dict.fromkeys(filter(None, [
    LLM_MODEL,
    os.getenv("OLLAMA_LLM_MODEL_2"),
    os.getenv("OLLAMA_LLM_MODEL_3"),
])))

_ollama = OllamaClient(host=OLLAMA_HOST)
_col: Collection | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _col
    connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
    _col = Collection(COLLECTION_NAME)
    _col.load()
    mlflow.set_tracking_uri(MLFLOW_URI)
    yield
    connections.disconnect("default")


app = FastAPI(
    title="YouTube Viral RAG API",
    description="Plataforma RAG Enterprise para análise de conteúdo viral no YouTube.",
    version="1.0.0",
    lifespan=lifespan,
)

# modelos de contrato

class QueryRequest(BaseModel):
    question: str
    top_k: int = Field(default=5, ge=1, le=20)
    model: Optional[str] = None


class ModelInfo(BaseModel):
    llm: str
    embedding: str


class RetrievedDocument(BaseModel):
    rank: int
    score: float
    video_id: str
    title: str
    channel: str
    category: str
    country: str
    texto_rag: str


class Latency(BaseModel):
    embedding_ms: float
    search_ms: float
    llm_ms: float
    total_ms: float


class QueryResponse(BaseModel):
    query_id: str
    timestamp: str
    question: str
    expanded_query: str
    answer: str
    models: ModelInfo
    collection: str
    top_k: int
    retrieved_documents: list[RetrievedDocument]
    latency: Latency


class VectorStoreInfo(BaseModel):
    host: str
    port: int
    collection: str
    metric: str
    index_type: str


class DataLakeInfo(BaseModel):
    host: str
    buckets: list[str]


class MetadataResponse(BaseModel):
    timestamp: str
    default_models: ModelInfo
    available_llm_models: list[str]
    vector_store: VectorStoreInfo
    mlflow_uri: str
    data_lake: DataLakeInfo


class MLflowRunInfo(BaseModel):
    run_id: str
    run_name: str
    status: str
    metrics: dict
    params: dict
    start_time: str


class MLflowExperimentInfo(BaseModel):
    experiment_id: str
    name: str
    runs: list[MLflowRunInfo]


class VideoInfo(BaseModel):
    rank: int
    video_id: str
    title: str
    channel: str
    category: str
    country: str
    views: int


class TopVideosResponse(BaseModel):
    timestamp: str
    limit: int
    videos: list[VideoInfo]


# helpers

def _minio() -> Minio:
    return Minio(
        f"{MINIO_HOST}:{MINIO_PORT}",
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False,
    )


def _collection() -> Collection:
    if _col is None:
        raise RuntimeError("Milvus collection not initialized")
    return _col


_PLATFORM_KW = {
    "quem é você", "o que você é", "plataforma", "arquitetura",
    "pré-processamento", "preprocessamento", "medallion", "camada bronze",
    "camada silver", "camada gold", "modelos treinados", "como funciona",
    "infraestrutura", "rag enterprise", "como você funciona",
}

_MLFLOW_KW = {"modelos treinados", "métricas", "acurácia", "accuracy", "f1", "precision", "recall", "experimento", "mlflow"}


def _get_mlflow_context() -> str:
    try:
        client = MlflowClient()
        runs = client.search_runs(
            experiment_ids=[client.get_experiment_by_name("youtube_trending_classification").experiment_id]
        )
        lines = ["Resultados reais dos experimentos MLflow:"]
        for r in runs:
            m = r.data.metrics
            lines.append(
                f"- {r.info.run_name or r.data.params.get('model_name', r.info.run_id)}: "
                f"accuracy={m.get('accuracy', 0):.4f}, "
                f"precision={m.get('precision', 0):.4f}, "
                f"recall={m.get('recall', 0):.4f}, "
                f"f1_score={m.get('f1_score', 0):.4f}"
            )
        return "\n".join(lines)
    except Exception:
        return ""


def _expand_query(question: str) -> str:
    if any(kw in question.lower() for kw in _PLATFORM_KW):
        return question
    return (
        f"Usuário busca ideias de conteúdo viral para YouTube.\n\n"
        f"Pedido: {question}\n\n"
        "Buscar: vídeos similares, títulos engajantes, temas virais, "
        "conteúdo com alta retenção, tendências relacionadas, "
        "formatos que aumentam CTR, vídeos com potencial viral."
    )


def _build_context(docs: list[RetrievedDocument]) -> str:
    lines = []
    for doc in docs:
        lines.append(
            f"VÍDEO #{doc.rank} (score {doc.score})\n"
            f"Título: {doc.title}\n"
            f"Canal: {doc.channel} | Categoria: {doc.category} | País: {doc.country}\n"
            f"Conteúdo: {doc.texto_rag}\n"
            "---"
        )
    return "\n".join(lines)


# endpoints

@app.get("/health", tags=["System"])
def health():
    """Verifica se a API está no ar."""
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/metadata", response_model=MetadataResponse, tags=["System"])
def metadata():
    """Retorna a configuração atual do sistema: modelos, banco vetorial e Data Lake."""
    try:
        buckets = [b.name for b in _minio().list_buckets()]
    except Exception:
        buckets = []

    return MetadataResponse(
        timestamp=datetime.now(timezone.utc).isoformat(),
        default_models=ModelInfo(llm=LLM_MODEL, embedding=EMBED_MODEL),
        available_llm_models=AVAILABLE_MODELS,
        vector_store=VectorStoreInfo(
            host=MILVUS_HOST,
            port=int(MILVUS_PORT),
            collection=COLLECTION_NAME,
            metric="COSINE",
            index_type="HNSW",
        ),
        mlflow_uri=MLFLOW_URI,
        data_lake=DataLakeInfo(
            host=f"{MINIO_HOST}:{MINIO_PORT}",
            buckets=buckets,
        ),
    )


@app.get("/models", tags=["System"])
def list_models():
    """Lista todos os modelos LLM disponíveis no Ollama."""
    try:
        return {
            "available": AVAILABLE_MODELS,
            "default": LLM_MODEL,
            "embedding": EMBED_MODEL,
            "ollama": _ollama.list(),
        }
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Ollama error: {exc}")


@app.post("/query", response_model=QueryResponse, tags=["RAG"])
def query(request: QueryRequest):
    """
    Endpoint principal do RAG. Gera embedding da pergunta, busca os documentos mais similares
    no Milvus e gera a resposta via LLM. Use o campo 'model' para escolher o LLM.
    """
    active_model = request.model or LLM_MODEL

    if request.model and request.model not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{request.model}' not available. Choose from: {AVAILABLE_MODELS}",
        )

    total_start = time.time()
    query_id    = str(uuid.uuid4())
    timestamp   = datetime.now(timezone.utc).isoformat()
    expanded    = _expand_query(request.question)

    # embedding
    t0 = time.time()
    try:
        embed_resp = _ollama.embeddings(model=EMBED_MODEL, prompt=expanded)
        embedding  = embed_resp["embedding"]
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Embedding error: {exc}")
    embedding_ms = (time.time() - t0) * 1000

    # busca vetorial
    is_platform = any(kw in request.question.lower() for kw in _PLATFORM_KW)
    t0 = time.time()
    try:
        col     = _collection()
        results = col.search(
            data=[embedding],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"ef": 128}},
            limit=request.top_k,
            expr="category_name == 'Documentação'" if is_platform else None,
            output_fields=[
                "video_id", "title", "channel_title",
                "category_name", "country", "texto_rag",
            ],
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Milvus error: {exc}")
    search_ms = (time.time() - t0) * 1000

    retrieved: list[RetrievedDocument] = []
    for hits in results:
        for idx, hit in enumerate(hits):
            e = hit.entity
            retrieved.append(
                RetrievedDocument(
                    rank=idx + 1,
                    score=round(hit.distance, 4),
                    video_id=e.get("video_id", ""),
                    title=e.get("title", ""),
                    channel=e.get("channel_title", ""),
                    category=e.get("category_name", ""),
                    country=e.get("country", ""),
                    texto_rag=e.get("texto_rag", ""),
                )
            )

    context = _build_context(retrieved)

    if any(kw in request.question.lower() for kw in _MLFLOW_KW):
        mlflow_ctx = _get_mlflow_context()
        if mlflow_ctx:
            context = f"{context}\n\n{mlflow_ctx}" if context else mlflow_ctx

    rag_prompt = (
        "Você é o assistente da plataforma YouTube Viral RAG Enterprise.\n\n"
        f"O usuário perguntou:\n{request.question}\n\n"
        f"Contexto recuperado do banco vetorial:\n{context}\n\n"
        "REGRAS:\n"
        "- Responda em português brasileiro\n"
        "- Seja direto e objetivo, máximo 100 palavras\n"
        "- Responda APENAS com base no contexto fornecido, sem inventar informações\n"
        "- Se o contexto contiver documentação (categoria 'Documentação'), extraia a resposta exata dela\n"
        "- Para perguntas sobre pré-processamento, liste as etapas da camada Silver mencionadas no contexto\n"
        "- Para perguntas sobre modelos, liste os modelos e métricas mencionados no contexto\n"
        "- Para perguntas sobre identidade, descreva a plataforma como apresentada no contexto\n"
        "- Se o contexto contiver vídeos do YouTube, analise padrões e faça sugestões de conteúdo\n"
        "- Se não encontrar a informação no contexto, diga isso claramente"
    )

    # geração da resposta
    t0 = time.time()
    try:
        llm_resp = _ollama.chat(
            model=active_model,
            messages=[
                {
                    "role": "system",
                    "content": "Você é a plataforma YouTube Viral RAG Enterprise. Responda SOMENTE com informações presentes no contexto fornecido. NÃO invente, complete ou adicione nada que não esteja explicitamente no contexto.",
                },
                {"role": "user", "content": rag_prompt},
            ],
            options={"num_ctx": 2048, "temperature": 0.1},
        )
        answer = llm_resp["message"]["content"]
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"LLM error: {exc}")
    llm_ms    = (time.time() - t0) * 1000
    total_ms  = (time.time() - total_start) * 1000

    return QueryResponse(
        query_id=query_id,
        timestamp=timestamp,
        question=request.question,
        expanded_query=expanded,
        answer=answer,
        models=ModelInfo(llm=active_model, embedding=EMBED_MODEL),
        collection=COLLECTION_NAME,
        top_k=request.top_k,
        retrieved_documents=retrieved,
        latency=Latency(
            embedding_ms=round(embedding_ms, 2),
            search_ms=round(search_ms, 2),
            llm_ms=round(llm_ms, 2),
            total_ms=round(total_ms, 2),
        ),
    )


class MLflowQueryRequest(BaseModel):
    question: str
    model: Optional[str] = None


class MLflowQueryResponse(BaseModel):
    question: str
    answer: str
    model: str
    context_runs: int
    llm_ms: float


@app.post("/query/mlflow", response_model=MLflowQueryResponse, tags=["RAG"])
def query_mlflow(request: MLflowQueryRequest):
    """
    RAG sobre experimentos MLflow. Busca todos os runs registrados,
    monta o contexto e gera uma resposta via LLM.
    """
    active_model = request.model or LLM_MODEL

    if request.model and request.model not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{request.model}' not available. Choose from: {AVAILABLE_MODELS}",
        )

    try:
        client      = MlflowClient()
        experiments = client.search_experiments()
        context_lines = []
        total_runs = 0
        for exp in experiments:
            runs = client.search_runs(experiment_ids=[exp.experiment_id])
            for r in runs:
                total_runs += 1
                metrics_str = " | ".join(f"{k}={round(v,4)}" for k, v in r.data.metrics.items())
                params_str  = " | ".join(f"{k}={v}" for k, v in r.data.params.items()
                                         if k != "model_name")
                context_lines.append(
                    f"EXPERIMENTO: {exp.name}\n"
                    f"RUN: {r.info.run_name or r.data.params.get('model_name', r.info.run_id)}\n"
                    f"Status: {r.info.status}\n"
                    f"Métricas: {metrics_str or 'nenhuma'}\n"
                    f"Parâmetros: {params_str or 'nenhum'}\n"
                    "---"
                )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"MLflow error: {exc}")

    if not context_lines:
        raise HTTPException(status_code=404, detail="Nenhum experimento encontrado no MLflow.")

    context = "\n".join(context_lines)
    prompt = (
        "Você é um assistente de ciência de dados especialista em MLflow e experimentos de Machine Learning.\n\n"
        f"O usuário perguntou:\n{request.question}\n\n"
        f"Experimentos registrados no MLflow:\n{context}\n\n"
        "REGRAS:\n"
        "- Responda em português brasileiro\n"
        "- Seja direto e preciso com os números\n"
        "- Se a pergunta for sobre o melhor modelo, compare as métricas objetivamente\n"
        "- Use listas quando adequado\n"
        "- Se a informação não estiver nos dados, diga isso claramente"
    )

    t0 = time.time()
    try:
        llm_resp = _ollama.chat(
            model=active_model,
            messages=[
                {"role": "system", "content": "Você é um especialista em análise de experimentos de Machine Learning."},
                {"role": "user", "content": prompt},
            ],
            options={"num_ctx": 2048, "temperature": 0.1},
        )
        answer = llm_resp["message"]["content"]
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"LLM error: {exc}")
    llm_ms = (time.time() - t0) * 1000

    return MLflowQueryResponse(
        question=request.question,
        answer=answer,
        model=active_model,
        context_runs=total_runs,
        llm_ms=round(llm_ms, 2),
    )


@app.get("/mlflow/experiments", response_model=list[MLflowExperimentInfo], tags=["MLflow"])
def list_experiments():
    """Lista todos os experimentos e runs do MLflow com métricas."""
    try:
        client      = MlflowClient()
        experiments = client.search_experiments()
        result      = []
        for exp in experiments:
            runs     = client.search_runs(experiment_ids=[exp.experiment_id])
            run_list = [
                MLflowRunInfo(
                    run_id=r.info.run_id,
                    run_name=r.info.run_name or r.data.params.get("model_name", r.info.run_id),
                    status=r.info.status,
                    metrics=r.data.metrics,
                    params=r.data.params,
                    start_time=datetime.fromtimestamp(
                        r.info.start_time / 1000, timezone.utc
                    ).isoformat(),
                )
                for r in runs
            ]
            result.append(
                MLflowExperimentInfo(
                    experiment_id=exp.experiment_id,
                    name=exp.name,
                    runs=run_list,
                )
            )
        return result
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"MLflow error: {exc}")


@app.get("/mlflow/metrics", tags=["MLflow"])
def get_metrics(
    experiment_name: str = Query(..., description="MLflow experiment name"),
):
    """Retorna todos os runs e métricas de um experimento pelo nome."""
    try:
        client     = MlflowClient()
        experiment = client.get_experiment_by_name(experiment_name)
        if not experiment:
            raise HTTPException(
                status_code=404,
                detail=f"Experiment '{experiment_name}' not found.",
            )
        runs = client.search_runs(experiment_ids=[experiment.experiment_id])
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "experiment": experiment_name,
            "runs": [
                {
                    "run_id": r.info.run_id,
                    "run_name": r.info.run_name
                    or r.data.params.get("model_name", r.info.run_id),
                    "status": r.info.status,
                    "metrics": r.data.metrics,
                    "params": r.data.params,
                    "start_time": datetime.fromtimestamp(
                        r.info.start_time / 1000, timezone.utc
                    ).isoformat(),
                }
                for r in runs
            ],
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"MLflow error: {exc}")


@app.get("/videos/top", response_model=TopVideosResponse, tags=["Data"])
def top_videos(
    limit: int = Query(default=10, ge=1, le=100, description="Number of videos to return"),
):
    """Retorna os N vídeos do YouTube com mais views, lidos da camada Gold do Data Lake."""
    try:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "trending_gold.parquet")
            _minio().fget_object("gold", "trending_gold.parquet", path)
            df = pd.read_parquet(path)

        top = df.nlargest(limit, "views")[
            ["video_id", "title", "channel_title", "category_name", "country", "views"]
        ]

        videos = [
            VideoInfo(
                rank=i + 1,
                video_id=str(row["video_id"]),
                title=str(row["title"]),
                channel=str(row["channel_title"]),
                category=str(row["category_name"]),
                country=str(row["country"]),
                views=int(row["views"]),
            )
            for i, (_, row) in enumerate(top.iterrows())
        ]

        return TopVideosResponse(
            timestamp=datetime.now(timezone.utc).isoformat(),
            limit=limit,
            videos=videos,
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Data error: {exc}")

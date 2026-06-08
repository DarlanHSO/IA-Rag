# API Contract — YouTube Viral RAG

## Onde está o código

| O quê | Arquivo |
|---|---|
| Definição dos endpoints | `api/main.py` — seção `ENDPOINTS` (linha 208+) |
| Modelos Pydantic (request/response) | `api/main.py` — seção `CONTRACT` (linha 67+) |
| Lógica de startup (conexão Milvus) | `api/main.py` — função `lifespan` (linha 46) |
| Helpers internos (expand query, build context) | `api/main.py` — seção `HELPERS` (linha 165+) |
| Dockerfile da API | `api/Dockerfile` |
| Configuração do serviço no stack | `docker-compose.yaml` — serviço `api` |

---

## Base URL

```
http://localhost:8000
```

Documentação interativa disponível em `http://localhost:8000/docs` (Swagger UI).

---

## Endpoints

### Sistema

#### `GET /health`
Liveness check — verifica se a API está no ar.

**Response `200`**
```json
{
  "status": "ok",
  "timestamp": "2026-06-06T18:00:00+00:00"
}
```

---

#### `GET /metadata`
Retorna a configuração atual do sistema: modelos, vector store e data lake.

**Response `200` — `MetadataResponse`**
```json
{
  "timestamp": "2026-06-06T18:00:00+00:00",
  "default_models": {
    "llm": "phi3:mini",
    "embedding": "nomic-embed-text"
  },
  "available_llm_models": ["phi3:mini", "llama3"],
  "vector_store": {
    "host": "milvus",
    "port": 19530,
    "collection": "youtube_trending",
    "metric": "COSINE",
    "index_type": "HNSW"
  },
  "mlflow_uri": "http://mlflow:3000",
  "data_lake": {
    "host": "minio:9000",
    "buckets": ["bronze", "silver", "gold", "mlflow"]
  }
}
```

---

#### `GET /models`
Lista todos os modelos LLM disponíveis no Ollama.

**Response `200`**
```json
{
  "available": ["phi3:mini", "llama3"],
  "default": "phi3:mini",
  "embedding": "nomic-embed-text",
  "ollama": { ... }
}
```

**Response `503`** — Ollama inacessível
```json
{ "detail": "Ollama error: <mensagem>" }
```

---

### RAG

#### `POST /query/mlflow`
RAG sobre experimentos MLflow. Busca todos os runs registrados, monta o contexto e gera uma resposta via LLM.

**Request Body — `MLflowQueryRequest`**
```json
{
  "question": "Qual foi o melhor modelo?",
  "model": "phi3:mini"
}
```

| Campo | Tipo | Obrigatório | Default |
|---|---|---|---|
| `question` | string | sim | — |
| `model` | string | não | `null` (usa o default) |

**Response `200` — `MLflowQueryResponse`**
```json
{
  "question": "Qual foi o melhor modelo?",
  "answer": "O RandomForest obteve o melhor F1-score de 0.92...",
  "model": "phi3:mini",
  "context_runs": 3,
  "llm_ms": 1420.5
}
```

---

#### `POST /query`
Endpoint principal. Embeda a pergunta, busca os vídeos mais similares no Milvus e gera uma resposta via LLM.

**Request Body — `QueryRequest`**
```json
{
  "question": "que tipo de vídeo está viral no Brasil?",
  "top_k": 5,
  "model": "phi3:mini"
}
```

| Campo | Tipo | Obrigatório | Default | Validação |
|---|---|---|---|---|
| `question` | string | sim | — | — |
| `top_k` | int | não | `5` | 1 ≤ top_k ≤ 20 |
| `model` | string | não | `null` (usa o default) | deve estar em `available_llm_models` |

**Response `200` — `QueryResponse`**
```json
{
  "query_id": "uuid-v4",
  "timestamp": "2026-06-06T18:00:00+00:00",
  "question": "que tipo de vídeo está viral no Brasil?",
  "expanded_query": "Usuário busca ideias de conteúdo viral...",
  "answer": "Com base nos vídeos encontrados...",
  "models": {
    "llm": "phi3:mini",
    "embedding": "nomic-embed-text"
  },
  "collection": "youtube_trending",
  "top_k": 5,
  "retrieved_documents": [
    {
      "rank": 1,
      "score": 0.9821,
      "video_id": "abc123",
      "title": "10 Receitas Virais",
      "channel": "Canal X",
      "category": "Food",
      "country": "BR",
      "texto_rag": "..."
    }
  ],
  "latency": {
    "embedding_ms": 45.2,
    "search_ms": 12.1,
    "llm_ms": 1823.4,
    "total_ms": 1880.7
  }
}
```

**Response `400`** — modelo inválido
```json
{ "detail": "Model 'xyz' not available. Choose from: ['phi3:mini']" }
```

**Response `503`** — falha em embedding, Milvus ou LLM
```json
{ "detail": "Embedding error: <mensagem>" }
```

---

### MLflow

#### `GET /mlflow/experiments`
Lista todos os experimentos MLflow e seus runs com métricas.

**Response `200` — `list[MLflowExperimentInfo]`**
```json
[
  {
    "experiment_id": "1",
    "name": "rag_evaluation",
    "runs": [
      {
        "run_id": "abc...",
        "run_name": "phi3:mini",
        "status": "FINISHED",
        "metrics": { "precision": 0.87 },
        "params": { "top_k": "5" },
        "start_time": "2026-06-06T18:00:00+00:00"
      }
    ]
  }
]
```

**Response `503`** — MLflow inacessível
```json
{ "detail": "MLflow error: <mensagem>" }
```

---

#### `GET /mlflow/metrics?experiment_name={name}`
Retorna todos os runs e métricas de um experimento pelo nome.

**Query Params**

| Parâmetro | Tipo | Obrigatório |
|---|---|---|
| `experiment_name` | string | sim |

**Response `200`**
```json
{
  "timestamp": "2026-06-06T18:00:00+00:00",
  "experiment": "rag_evaluation",
  "runs": [ { ... } ]
}
```

**Response `404`** — experimento não encontrado
```json
{ "detail": "Experiment 'xyz' not found." }
```

---

### Data

#### `GET /videos/top?limit={n}`
Retorna os N vídeos com mais views da camada Gold (MinIO).

**Query Params**

| Parâmetro | Tipo | Obrigatório | Default | Validação |
|---|---|---|---|---|
| `limit` | int | não | `10` | 1 ≤ limit ≤ 100 |

**Response `200` — `TopVideosResponse`**
```json
{
  "timestamp": "2026-06-06T18:00:00+00:00",
  "limit": 10,
  "videos": [
    {
      "rank": 1,
      "video_id": "abc123",
      "title": "Vídeo Mais Visto",
      "channel": "Canal X",
      "category": "Entertainment",
      "country": "BR",
      "views": 98000000
    }
  ]
}
```

**Response `503`** — falha ao ler o parquet do MinIO
```json
{ "detail": "Data error: <mensagem>" }
```

---

## Variáveis de ambiente da API

Configuradas no `docker-compose.yaml` (serviço `api`):

| Variável | Descrição | Default no código |
|---|---|---|
| `MILVUS_HOST` | Host do Milvus | `localhost` |
| `MILVUS_PORT` | Porta do Milvus | `19530` |
| `MILVUS_COLLECTION` | Nome da collection | `youtube_trending` |
| `OLLAMA_HOST` | URL do Ollama | `http://localhost:11434` |
| `OLLAMA_EMBED_MODEL` | Modelo de embedding | `nomic-embed-text` |
| `OLLAMA_LLM_MODEL` | LLM principal | `phi3:mini` |
| `OLLAMA_LLM_MODEL_2` | LLM alternativo 2 | — |
| `OLLAMA_LLM_MODEL_3` | LLM alternativo 3 | — |
| `MINIO_HOST` | Host do MinIO | `localhost` |
| `MINIO_PORT` | Porta do MinIO | `9000` |
| `MINIO_ACCESS_KEY` | Access key do MinIO | — |
| `MINIO_SECRET_KEY` | Secret key do MinIO | — |
| `MLFLOW_TRACKING_URI` | URI do MLflow | `http://localhost:3000` |

# YouTube Viral RAG Enterprise

Plataforma RAG para análise de vídeos virais no YouTube. Dataset com trending videos de 11 países via Kaggle.

**Disciplina:** Inteligência Artificial — UniFacens  
**Professor:** Adson Nogueira Alves  
**Dataset:** [youtube-trending-videos-stats-2026](https://www.kaggle.com/datasets/bsthere/youtube-trending-videos-stats-2026)

---

## Serviços

| Container | Tecnologia | Porta | Função |
|-----------|-----------|-------|--------|
| `rag_api` | FastAPI | 8000 | API REST principal |
| `rag_interface` | Gradio | 7860 | Interface web |
| `mlflow-server` | MLflow 2.6.0 | 3000 | Rastreamento de experimentos |
| `mlflow-minio` | MinIO | 9000 / 9001 | Data Lake (bronze, silver, gold) |
| `mlflow-postgres` | PostgreSQL 13 | 5433 | Metadados e versões do dataset |
| `rag_milvus` | Milvus v2.4.0 | 19530 | Banco vetorial (HNSW, COSINE) |
| `rag_ollama` | Ollama | 11434 | LLM e embeddings locais |
| `rag_attu` | Attu | 8888 | Interface visual do Milvus |
| `milvus-etcd` | etcd | — | Coordenação interna do Milvus |

---

## Como rodar

### Pré-requisitos

- Docker + Docker Compose
- GPU NVIDIA (recomendado)
- Arquivo `.env` configurado (veja `.env.example`)

### 1. Subir os containers

```bash
make up
```

Na primeira execução o Ollama vai baixar os modelos. Aguarde antes de continuar.

```bash
docker logs rag_ollama -f
```

### 2. Rodar o pipeline completo

```bash
make pipeline
```

Executa em sequência:
- `ingest` — faz upload dos docs de arquitetura + baixa dados do Kaggle + processa bronze → silver → gold
- `index` — gera embeddings e indexa no Milvus
- `train` — treina os 3 modelos ML e registra no MLflow

### 3. Testar

```bash
make test
```

### Outros comandos

```bash
make down       # desce os containers (volumes preservados)
make restart    # down + up
make logs       # acompanha logs em tempo real
make build      # rebuild das imagens + up
```

---

## Endpoints da API

| Método | Rota | Descrição |
|--------|------|-----------|
| GET | `/health` | Liveness check |
| GET | `/metadata` | Configuração do sistema |
| GET | `/models` | LLMs disponíveis no Ollama |
| POST | `/query` | RAG principal — pergunta → embedding → Milvus → LLM |
| POST | `/query/mlflow` | RAG sobre experimentos MLflow |
| GET | `/mlflow/experiments` | Lista experimentos e runs |
| GET | `/mlflow/metrics` | Métricas de um experimento |
| GET | `/videos/top` | Top N vídeos por views (camada Gold) |
| GET | `/dataset/versions` | Histórico de ingestões no PostgreSQL |

Documentação interativa: `http://localhost:8000/docs`

---

## Arquitetura Medallion

| Camada | Bucket | Conteúdo |
|--------|--------|----------|
| Bronze | `bronze` | CSVs originais do Kaggle — imutável |
| Silver | `silver` | Dados limpos e normalizados em Parquet |
| Gold | `gold` | Campo `texto_rag` pronto para embeddings e ML |

---

## Modelos ML

LogisticRegression, DecisionTreeClassifier e RandomForestClassifier treinados para classificação binária de vídeos virais (views > 1M). Experimentos e artefatos registrados no MLflow, modelos salvos no MinIO.

---

## Time

| Nome | RA | Função | Turma |
|------|----|--------|-------|
| César Augusto de Almeida | 222909 | Dev | CP901TIN2 |
| Darlan Henrique de Souza Oliveira | 211926 | Scrum Master | CP901TIN2 |
| Grazielly Almeida Rolle | 211871 | Dev | CP901TIN2 |
| Gustavo Eiji Tamezava | 222226 | Scrum Master | CP901TIN2 |
| Kevyn Feitosa Rocha | 223535 | Dev | CP901TIN2 |
| Leonardo Almeida Proença | 222241 | Dev | CP901TIN2 |
| Lucas Nascimento de Campos | 223324 | Tester | CP901TIN2 |
| Natale Tagliaferro Neto | 212182 | Dev | CP901TIN2 |
| Thiago Jun Honma | 222628 | Tester | CP901TIN2 |
| Vinicius Matheus Nunes Araújo | 211973 | PO | CP901TIN2 |
| Felipe Roberto de Souza Silva | 226752 | Dev | CP901TIN3 |

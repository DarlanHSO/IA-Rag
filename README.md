# YouTube Viral RAG Enterprise

Plataforma de análise de conteúdo viral no YouTube usando RAG (Retrieval-Augmented Generation). Dataset com vídeos trending de 11 países via Kaggle.

**Disciplina:** Inteligência Artificial — UniFacens  
**Professor:** Adson Nogueira Alves  
**Dataset:** [youtube-trending-videos-stats-2026](https://www.kaggle.com/datasets/bsthere/youtube-trending-videos-stats-2026)

---

## Stack

| Serviço | Tecnologia | Porta |
|---------|-----------|-------|
| Data Lake | MinIO | 9000 / 9001 |
| Banco vetorial | Milvus v2.4.0 | 19530 |
| LLM / Embeddings | Ollama | 11434 |
| Experimentos ML | MLflow 2.6.0 | 3000 |
| API | FastAPI | 8000 |
| Interface | Gradio | 7860 |
| Banco relacional | PostgreSQL 13 | 5433 |

---

## Como rodar

```bash
# Sobe os 9 containers
make up

# Ingesta dados + indexa embeddings + treina modelos
make pipeline

# Testa todos os endpoints
make test
```

Interface disponível em `http://localhost:7860`  
API + docs em `http://localhost:8000/docs`  
MLflow em `http://localhost:3000`

---

## Arquitetura Medallion

- **Bronze** — CSVs originais do Kaggle, imutável
- **Silver** — dados limpos e normalizados em Parquet
- **Gold** — campo `texto_rag` pronto para embeddings e treino ML

---

## Modelos treinados

LogisticRegression, DecisionTreeClassifier e RandomForestClassifier para classificação binária de vídeos virais (views > 1M). Experimentos registrados no MLflow.

---

## Time

| Nome | RA | Turma |
|------|----|-------|
| César Augusto de Almeida | 222909 | CP901TIN2 |
| Darlan Henrique de Souza Oliveira | 211926 | CP901TIN2 |
| Grazielly Almeida Rolle | 211871 | CP901TIN2 |
| Gustavo Eiji Tamezava | 222226 | CP901TIN2 |
| Kevyn Feitosa Rocha | 223535 | CP901TIN2 |
| Leonardo Almeida Proença | 222241 | CP901TIN2 |
| Lucas Nascimento de Campos | 223324 | CP901TIN2 |
| Natale Tagliaferro Neto | 212182 | CP901TIN2 |
| Thiago Jun Honma | 222628 | CP901TIN2 |
| Vinicius Matheus Nunes Araújo | 211973 | CP901TIN2 |
| Felipe Roberto de Souza Silva | 226752 | CP901TIN3 |

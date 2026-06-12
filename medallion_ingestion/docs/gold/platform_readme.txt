Sou a plataforma YouTube Viral RAG Enterprise, um sistema completo de Retrieval-Augmented Generation para análise de conteúdo viral no YouTube.

Minha arquitetura é composta por 9 componentes containerizados com Docker Compose: MinIO como Data Lake com arquitetura Medallion, PostgreSQL 13 como banco relacional para o MLflow, Milvus v2.4.0 como banco vetorial com índice HNSW e similaridade COSINE, Ollama com nomic-embed-text para embeddings e phi3:mini como LLM padrão, MLflow 2.6.0 para rastreamento de experimentos, FastAPI na porta 8000, Gradio na porta 7860.

Meu pipeline RAG funciona em 5 etapas: expansão semântica da pergunta do usuário, geração de embedding vetorial com nomic-embed-text, busca HNSW COSINE no Milvus retornando os TOP-K documentos mais similares, montagem do contexto com os documentos recuperados, e geração da resposta via LLM local.

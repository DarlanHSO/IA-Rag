YouTube Viral RAG Enterprise — Documentação da Camada Gold

A camada Gold é a terceira e última camada da arquitetura Medallion. Contém os dados finais prontos para consumo direto em machine learning e no pipeline RAG.

Transformações aplicadas nesta camada:
- Remoção da coluna channel_id (irrelevante para análise de conteúdo)
- Criação do campo texto_rag: concatenação estruturada de título, categoria, canal, views, tags e descrição de cada vídeo
- O campo texto_rag é o texto usado para gerar os embeddings vetoriais indexados no Milvus
- Armazenamento em Parquet no bucket gold do MinIO

Uso da camada Gold:
- Treino dos modelos de machine learning (LogisticRegression, DecisionTree, RandomForest) para classificação de vídeos virais (views acima de 1 milhão)
- Indexação vetorial no Milvus para o pipeline RAG de consulta semântica
- Servida pela API REST via endpoint /videos/top

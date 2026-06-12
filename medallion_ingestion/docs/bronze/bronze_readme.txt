YouTube Viral RAG Enterprise — Documentação da Camada Bronze

A camada Bronze é a primeira camada da arquitetura Medallion desta plataforma RAG Enterprise. Ela armazena os dados brutos sem nenhuma transformação, exatamente como vieram da fonte original.

Fonte dos dados: dataset do Kaggle bsthere/youtube-trending-videos-stats-2026, baixado via Kaggle API.
Países incluídos: Brasil, Estados Unidos, Canadá, Reino Unido, França, Alemanha, Japão, Coreia do Sul, México, Rússia e Índia (11 países).
Formato: arquivos CSV separados por país (BR_Trending.csv, US_Trending.csv, etc.).
Armazenamento: bucket bronze no MinIO (Data Lake).
Objetivo: preservar os dados originais para garantir rastreabilidade, auditoria e possibilidade de reprocessamento sem perda de informação.

Nenhum dado é modificado nesta camada. A filosofia é: ingerir primeiro, transformar depois.

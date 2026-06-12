A camada Silver é a segunda camada da arquitetura Medallion. Contém os dados tratados e normalizados provenientes da camada Bronze.

O pré-processamento realizado na camada Silver inclui: concatenação dos CSVs de 11 países em um único DataFrame consolidado, remoção de duplicatas com base em video_id e country, remoção de linhas com valores nulos em colunas críticas (video_id, title, views), conversão de tipos (views/likes/comments para inteiro, trending_date para datetime), normalização de strings com strip e lowercase em campos categóricos, geração de matriz de correlação entre features numéricas.

Os dados da camada Silver são armazenados em formato Parquet no bucket silver do MinIO, garantindo eficiência de leitura e compressão antes de chegar à camada Gold.

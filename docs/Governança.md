# Governança de Dados — YouTube Viral RAG Enterprise

Este documento define as regras de arquitetura, qualidade e segurança aplicadas ao pipeline de dados da plataforma.

---

## 1. Arquitetura Medallion e Linhagem de Dados

O fluxo de dados utiliza o MinIO como Data Lake, dividido em três camadas com restrições distintas de acesso e mutabilidade:

| Camada | Tipo | Descrição |
|--------|------|-----------|
| **Bronze** | Raw | Camada imutável. Armazena os CSVs originais do Kaggle. Fonte única da verdade (SSOT). |
| **Silver** | Trusted | Camada de qualidade. Dados consolidados, limpos e convertidos para Parquet. |
| **Gold** | Business | Camada de consumo. Dados formatados para indexação vetorial no Milvus e treino de modelos ML. |

---

## 2. Regras de Qualidade (Data Quality)

Para um dado ser promovido da camada Bronze até a Gold, ele passa obrigatoriamente pelos seguintes contratos:

- **Remoção de viés:** A coluna `dislikes` é removida do pipeline por ter sido descontinuada pelo YouTube, evitando viés analítico.
- **Filtragem de ruído:** Registros com `views <= 0` e linhas duplicadas são descartados por serem anomalias estatísticas.
- **Completude semântica:** Campos vazios ou corrompidos (como `[none]`) em tags e descrições são padronizados para `"não informado"`.
- **Tipagem estrita:** Datas são convertidas para `datetime` e métricas para `inteiro`, prevenindo quebras de esquema nas camadas de consumo.
- **Enriquecimento obrigatório:** O ID numérico da categoria é mapeado para seu nome em texto via dicionário oficial da API do YouTube.

---

## 3. Regras de Negócio para Inteligência Artificial (Camada Gold)

- **Descarte estrutural:** Colunas sem valor semântico para busca, como `channel_id`, são removidas.
- **Densidade de contexto:** Os metadados úteis (título, categoria, canal, tags e descrição) são consolidados em um único campo `texto_rag`, otimizando a geração de embeddings.

---

## 4. Segurança e Infraestrutura

- **Proteção de credenciais:** Nenhuma chave de acesso possui hardcode no repositório. Todas as credenciais (Kaggle, MinIO, PostgreSQL) são injetadas exclusivamente via variáveis de ambiente (`.env`).
- **Processamento stateless:** Arquivos temporários (ZIPs, CSVs intermediários) são processados em memória e descartados, sem resíduos em disco.
- **Isolamento de serviços:** Todos os 9 componentes da plataforma executam em containers Docker isolados, comunicando-se apenas pela rede interna `mlflow-network`.
- **Rastreabilidade de modelos:** Todos os experimentos de machine learning são registrados no MLflow com parâmetros, métricas e artefatos, garantindo reprodutibilidade.

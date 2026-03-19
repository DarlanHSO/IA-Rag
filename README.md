# IA-Rag
Jira: 

Link Dataset: https://www.kaggle.com/datasets/bsthere/youtube-trending-videos-stats-2026

- César Augusto de Almeida - RA 222909 - Dev / CP901TIN2
- Darlan Henrique de Souza Oliveira - RA 211926 - Scrum / CP901TIN2
- Grazielly Almeida Rolle - RA 211871 - Dev / CP901TIN2
- Gustavo Eiji Tamezava - RA 222226 - Scrum / CP901TIN2
- Kevyn Feitosa Rocha - RA 223535 - Dev / CP901TIN2
- Leonardo Almeida Proença - RA 222241 - Dev / CP901TIN2
- Lucas Nascimento de Campos - RA 223324 - Tester / CP901TIN2
- Natale Tagliaferro Neto - RA 212182 - Dev / CP901TIN2
- Thiago Jun Honma RA - 222628 - Tester / CP901TIN2
- Vinicius Matheus Nunes Araújo - RA 211973 - PO / CP901TIN2
- Felipe Roberto de Souza Silva - RA 226752 - Dev / CP901TIN3

# Product Backlog

Camadas da arquitetura do projeto:

- Camada de Dados
- Camada de IA
- Camada de Aplicação
- Camada de MLOps
- Infraestrutura

---

# 🗂 Estrutura do Backlog

Cada item do backlog possui:

- **ID** — Identificador da tarefa  
- **User Story** — Descrição da necessidade do usuário ou desenvolvedor  
- **Camada** — Parte da arquitetura relacionada  
- **Critério de Aceite** — Condição para considerar a tarefa concluída  

---

# Epic 1 — Preparação e Organização de Dados

**Objetivo:** estruturar e preparar o dataset de dados para utilização no sistema de IA.

| ID | User Story | Camada | Critério de Aceite |
|----|-----------|--------|--------------------|
| PB01 | Como desenvolvedor, quero armazenar o dataset bruto de dados no Data Lake para centralizar os dados do projeto | Dados (MinIO - Bronze) | Dataset armazenado no bucket bronze |
| PB02 | Como desenvolvedor, quero organizar e limpar os metadados das trends para melhorar a qualidade dos dados | Dados (Silver) | Metadados estruturados e inconsistências removidas |
| PB03 | Como desenvolvedor, quero gerar uma versão tratada do dataset pronta para consumo pela IA | Dados (Gold) | Dataset padronizado e validado |
| PB04 | Como desenvolvedor, quero registrar metadados e versões do dataset no banco para auditoria | Dados (PostgreSQL) | Dataset registrado com versão |

---

# Epic 2 — Processamento de IA

**Objetivo:** preparar e processar os dados utilizando técnicas de IA e pipeline RAG.

| ID   | User Story                                                                 | Camada              | Critério de Aceite                                      |
|------|---------------------------------------------------------------------------|---------------------|---------------------------------------------------------|
| PB05 | Como desenvolvedor, quero dividir os dados tabulares em partes menores para melhorar o processamento | IA (Chunking)       | Pipeline gera chunks de registros corretamente           |
| PB06 | Como desenvolvedor, quero gerar embeddings a partir dos dados tabulares para permitir busca semântica | IA (Embedding)      | Embeddings gerados com sucesso a partir dos registros    |
| PB07 | Como desenvolvedor, quero armazenar embeddings no banco vetorial para consultas eficientes | Dados (Milvus)      | Vetores indexados corretamente no banco                 |
| PB08 | Como usuário, quero que o sistema recupere registros similares a partir de uma consulta | IA (Recuperação)    | Sistema retorna resultados relevantes com base nos dados |
|

---

# Epic 3 — API e Aplicação

**Objetivo:** disponibilizar os serviços de análise de dados tabulares (trends do YouTube) através de uma API e interface de usuário.

| ID   | User Story                                                                 | Camada              | Critério de Aceite                                                  |
|------|---------------------------------------------------------------------------|---------------------|---------------------------------------------------------------------|
| PB09 | Como desenvolvedor, quero criar endpoints para consulta de dados via API  | Aplicação (FastAPI) | Endpoint `/search` retorna registros relevantes do dataset          |
| PB10 | Como usuário, quero consultar dados através de uma interface simples      | Aplicação (Gradio)  | Interface funcional permitindo buscas e filtros                     |
| PB11 | Como usuário, quero visualizar os resultados da análise dos dados         | Aplicação           | Resultados exibidos corretamente (ex: título, canal, métricas, etc) |

---

# Epic 4 — Monitoramento e Experimentos

**Objetivo:** acompanhar desempenho, experimentos e evolução dos modelos.

| ID | User Story | Camada | Critério de Aceite |
|----|-----------|--------|--------------------|
| PB12 | Como desenvolvedor, quero registrar experimentos do modelo para análise futura | MLOps (MLflow) | Experimentos registrados |
| PB13 | Como desenvolvedor, quero versionar os modelos utilizados no sistema | MLOps | Versões de modelo registradas |
| PB14 | Como desenvolvedor, quero registrar métricas de avaliação do sistema | MLOps | Métricas armazenadas |

---

# Epic 5 — Infraestrutura

**Objetivo:** garantir execução, reprodutibilidade e documentação do sistema.

| ID | User Story | Camada | Critério de Aceite |
|----|-----------|--------|--------------------|
| PB15 | Como desenvolvedor, quero criar containers para os serviços do sistema | Infraestrutura (Docker) | Containers funcionando |
| PB16 | Como desenvolvedor, quero orquestrar os serviços utilizando Docker Compose | Infraestrutura | Serviços executando corretamente |
| PB17 | Como desenvolvedor, quero documentar a arquitetura do sistema | Infraestrutura | README técnico completo |

---

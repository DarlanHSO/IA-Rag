# Sprint 4 — Modelagem e Treinamento de Modelos (ML + MLflow)

## Visão Geral

Nesta etapa foi desenvolvido um pipeline completo de **Machine Learning**, integrado ao **MLflow**, com base em uma arquitetura de dados no padrão **Medallion (Bronze → Silver → Gold)**.

O objetivo é classificar vídeos do YouTube como **virais ou não virais**, utilizando dados estruturados e enriquecidos ao longo do pipeline.

---

# Arquitetura do Pipeline (Medallion)

O projeto segue uma arquitetura moderna de dados dividida em três camadas:

## Bronze (Ingestão)

* Dados brutos obtidos via API do Kaggle
* Arquivos CSV por país
* Armazenamento no MinIO (Data Lake)

✔ Sem transformação
✔ Dados originais preservados

---

## Silver (Transformação)

* Limpeza e padronização dos dados
* Tratamento de nulos
* Conversão de tipos
* Enriquecimento com categorias

✔ Dados confiáveis e estruturados
✔ Prontos para análise

---

## Gold (Camada Analítica)

* Criação de features
* Consolidação dos dados
* Preparação para Machine Learning
* Criação de campo textual (`texto_rag`)

✔ Dados prontos para modelagem

---

# Definição do Problema

Problema de **classificação binária**:

* **1 (Viral):** vídeos com alto número de visualizações
* **0 (Não viral):** vídeos com menor alcance

```python
viral = views > 1_000_000
```

---

# Pipeline de Treinamento

## 1. Leitura dos dados

* Fonte: camada Gold (`trending_gold.parquet`)
* Armazenamento: MinIO

---

## 2. Pré-processamento

* Tratamento de valores nulos
* Seleção de features
* Criação do target

### Features utilizadas:

```text
likes
comments
title_length
num_tags
description_length
likes_ratio
comments_ratio
```

---

## 3. Divisão dos dados

* 80% treino
* 20% teste

---

## 4. Modelos treinados

* Logistic Regression
* Decision Tree
* Random Forest

---

## 5. Avaliação

Métricas utilizadas:

* Accuracy
* Precision
* Recall
* F1-score

---

## 6. MLflow

Experimentos registrados automaticamente:

* parâmetros
* métricas
* comparação entre modelos

![Experimentos MLflow](/docs/experimentos_MLflow.png)

Acesso local:
http://localhost:3000

---

# Resultados

| Modelo              | Accuracy   | Precision  | Recall     | F1-score   |
| ------------------- | ---------- | ---------- | ---------- | ---------- |
| Logistic Regression | 0.9474     | 0.9792     | 0.7785     | 0.8674     |
| Decision Tree       | 0.9973     | 0.9948     | 0.9932     | 0.9940     |
| Random Forest       | **0.9977** | **0.9964** | **0.9934** | **0.9948** |

---

# Melhor Modelo

**Random Forest**

Motivos:

* Melhor F1-score
* Alto equilíbrio entre precisão e recall
* Melhor desempenho geral

---

# Observações nos resultados da avaliação
Os resultados elevados podem ocorrer devido a:

* alta correlação entre variáveis de engajamento
* features derivadas diretamente do comportamento do vídeo
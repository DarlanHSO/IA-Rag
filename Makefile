.PHONY: up down build restart logs ingest index train test pipeline help

PIPELINE = docker compose --profile pipeline run --rm pipeline

# ─── Infraestrutura ───────────────────────────────────────────────────────────

up:
	docker compose up -d

down:
	docker compose down

build:
	docker compose up -d --build

restart:
	docker compose down
	docker compose up -d

logs:
	docker compose logs -f

# ─── Pipeline de dados ────────────────────────────────────────────────────────

ingest:
	$(PIPELINE) python medallion_ingestion/00_upload_docs.py
	$(PIPELINE) python medallion_ingestion/01_bronze_ingestion.py
	$(PIPELINE) python medallion_ingestion/02_silver_transform.py
	$(PIPELINE) python medallion_ingestion/03_gold_business.py

index:
	$(PIPELINE) python transformer/parallel_transformer.py
	$(PIPELINE) python transformer/index_docs.py

train:
	$(PIPELINE) sh -c "pip install boto3 -q && python treino/modelo_treino.py"

# Roda tudo em sequência: ingestão → embeddings → treino ML
pipeline: ingest index train

# ─── Testes ───────────────────────────────────────────────────────────────────

test:
	python tests/test_api.py

# ─── Help ─────────────────────────────────────────────────────────────────────

help:
	@echo ""
	@echo "Comandos disponíveis:"
	@echo ""
	@echo "  Infraestrutura:"
	@echo "    make up        Sobe todos os containers (sem rebuild)"
	@echo "    make down      Desce todos os containers"
	@echo "    make build     Rebuild das imagens e sobe"
	@echo "    make restart   Down + up"
	@echo "    make logs      Acompanha logs em tempo real"
	@echo ""
	@echo "  Pipeline de dados:"
	@echo "    make ingest    Bronze -> Silver -> Gold (container)"
	@echo "    make index     Indexa embeddings no Milvus (container)"
	@echo "    make train     Treina modelos ML e registra no MLflow (container)"
	@echo "    make pipeline  ingest + index + train em sequência"
	@echo ""
	@echo "  Testes:"
	@echo "    make test      Testa saúde de todos os endpoints da API"
	@echo ""

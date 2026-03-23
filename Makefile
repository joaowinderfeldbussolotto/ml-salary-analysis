# Makefile — Atalhos para as operações principais do projeto
# Uso: make <comando>

.PHONY: install train mlflow-ui api test clean docker-build docker-run

# Instalar dependências
install:
	pip install -r requirements.txt

# Treinar todos os modelos
train:
	python src/pipeline/train.py

# Abrir interface do MLflow
mlflow-ui:
	python mlflow_ui.py

# Subir a API
api:
	uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000

# Rodar testes
test:
	pytest tests/ -v

# Limpar artefatos gerados
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true

# Docker
docker-build:
	docker build -t salarios-ml .

docker-run:
	docker run -p 8000:8000 salarios-ml

# Pipeline completo (instalar → treinar → subir API)
all: install train api

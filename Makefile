.PHONY: instaldep install lock precommitrun format lint test train serve mlflow dvc clean

# Instalar dependências
installdep:
	uv sync
# Instalar dependências e grupos
install:
	uv sync --group eda --group model --group serving --group monitoring --group data --group experiment --group agent

# Atualizar lockfile
lock:
	uv lock

# Realizar pre-commit
precommitrun:
	uv run pre-commit run --all-files

# Formatar código
format:
	uv run black src tests
	uv run isort src tests

# Lint
lint:
	uv run flake8 src tests

# Testes
test:
	uv run pytest --cov=src --cov-fail-under=60

# Rodar treino com MLflow
train:
	PYTHONPATH=. uv run python src/models/train.py

# Subir API
serve:
	uv run uvicorn src.serving.app:app --reload --host 0.0.0.0 --port 8000

# Subir MLflow UI
mlflow:
	uv run mlflow server \
		--backend-store-uri sqlite:///mlflow.db \
		--default-artifact-root ./mlruns \
		-p 5000

# Rodar pipeline DVC
dvc:
	dvc repro

# Rodar avaliação RAGAS
evaluate:
	uv run python evaluation/ragas_eval.py

# Limpeza
clean:
	rm -rf __pycache__ .pytest_cache .coverage mlruns

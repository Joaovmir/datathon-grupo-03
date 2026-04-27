# 🧠 datathon-grupo-03

Projeto de Engenharia de Machine Learning com foco em **MLOps + LLMOps**, incluindo treinamento, avaliação, serving, monitoramento e governança.

---

## 🎯 Objetivo

Desenvolver uma solução completa de ML/LLM que seja:

* Reprodutível
* Testável
* Monitorável
* Escalável
* Segura

---

## 🏗️ Estrutura do Projeto

```
datathon-grupo-03/
├── .github/workflows/       # CI/CD (GitHub Actions)
├── data/
│   ├── raw/                 # Dados brutos (via DVC)
│   ├── processed/           # Dados processados
│   └── golden_set/          # Dataset de avaliação
├── src/
│   ├── features/            # Engenharia de features
│   ├── models/              # Treinamento e modelos
│   ├── agent/               # Agente + RAG
│   ├── serving/             # API (FastAPI)
│   ├── monitoring/          # Drift + métricas
│   └── security/            # Guardrails + PII
├── tests/                   # Testes (pytest)
├── evaluation/              # Avaliação (RAGAS, LLM judge)
├── docs/                    # Documentação
├── notebooks/               # EDA
├── configs/                 # Configurações YAML
├── Makefile                 # Automação de comandos
├── pyproject.toml           # Dependências (uv)
├── dvc.yaml                 # Pipeline de dados
└── README.md
```

---

## ⚙️ Stack Tecnológica

* Gerenciamento de dependências: `uv`
* Testes: `pytest`
* Tracking: `MLflow`
* API: `FastAPI`
* Versionamento de dados: `DVC`
* CI/CD: GitHub Actions
* Qualidade de código: pre-commit (black, isort, flake8)

---

## 🚀 Setup do Projeto

### 1. Clonar repositório

```bash
git clone https://github.com/Joaovmir/datathon-grupo-03
cd datathon-grupo-03
```

---

### 2. Instalar uv

```bash
pip install uv
```

---

### 3. Instalar dependências

Dependências base:

```bash
uv sync
```
Todas as dependências:

```bash
uv sync --group eda --group model --group serving --group monitoring --group data --group experiment
```

---

### 4. Ativar pre-commit

```bash
uv run pre-commit install
```

---

## 🧪 Testes

Rodar testes com cobertura:

```bash
uv run pytest --cov=src --cov-fail-under=60
```

Critério mínimo:

```
--cov-fail-under=60
```

---

## 🎨 Qualidade de código

Rodar lint + format:

```bash
uv run black src tests
uv run isort src tests
uv run flake8 src tests
```

---

## 🤖 Treinamento de modelo

```bash
uv run python src/models/train.py
```

Os experimentos são rastreados com **MLflow**.

---

## 📊 MLflow

Subir servidor local:

```bash
uv run mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlruns \
    -p 5000
```

Acesse:

```
http://localhost:5000
```

---

## 🌐 Servir API

```bash
uv run uvicorn src.serving.app:app --reload --host 0.0.0.0 --port 8000
```

API disponível em:

```
http://localhost:8000
```

---

## 🐳 Docker

### Build da imagem

> Requer que `dvc repro` tenha sido executado antes (artefatos precisam existir localmente).

```bash
docker build -f src/serving/Dockerfile -t credit-risk-api .
```

### Rodar o container

```bash
docker run -p 8000:8000 credit-risk-api
```

API disponível em:

```
http://localhost:8000
http://localhost:8000/docs
```

---

## 🔁 Pipeline de dados

Executar pipeline com DVC:

```bash
dvc repro
```

---

## 🔍 Pre-commit

Executa automaticamente antes de cada commit:

* Formatação (black)
* Imports (isort)
* Lint (flake8)
* Checks básicos

Rodar manualmente:

```bash
uv run pre-commit run --all-files
```

---

## 🔄 CI/CD

Pipeline configurado com GitHub Actions:

Etapas realizadas com o make (configuração Makefile):

1. Instala dependências (`make install`)
2. Executa pre-commit
3. Executa testes (`make test`)

---

## 📦 Comandos úteis

Comandos configurados com Makefile, configurado para o CI/CD ou comandos locais com Linux:

```bash
make install     # instalar dependências
make test        # rodar testes
make train       # treinar modelo
make serve       # subir API
make mlflow      # subir MLflow
make precommitrun
make lint        # rodar lint
make format      # formatar código
make dvc         # rodar pipeline de dados
```

---

## 📌 Boas práticas do projeto

* Código modular (pipeline desacoplado)
* Testes obrigatórios com cobertura mínima
* Versionamento de dados com DVC
* Experimentos rastreados com MLflow
* CI/CD obrigatório para validação
* Hooks de qualidade com pre-commit

---

## 🚧 Status do Projeto

### Etapa 1 — Dados + Baseline

- [x] EDA documentada com insights relevantes
- [x] Baseline treinado e métricas reportadas no MLflow
- [x] Pipeline versionado (DVC + Docker) e reprodutível
- [x] Métricas de negócio mapeadas para métricas técnicas
- [x] pyproject.toml com todas as dependências

### Etapa 2 — LLM + Agente

- [~] LLM servido via API com quantização aplicada
- [x] Agente ReAct funcional com ≥ 3 tools relevantes ao domínio
- [x] RAG retornando contexto relevante dos dados fornecidos
- [x] CI/CD pipeline funcional (GitHub Actions)
- [ ] Benchmark documentado com ≥ 3 configurações


### Etapa 3 — Avaliação + Observabilidade

- [x] Golden set com ≥ 20 pares relevantes ao domínio
- [x] RAGAS: 4 métricas calculadas e reportadas
- [x] LLM-as-judge com ≥ 3 critérios (incluindo critério de negócio)
- [~] Telemetria e dashboard funcionando end-to-end
- [x] Detecção de drift implementada e documentada


### Etapa 4 — Segurança + Governança

- [ ] OWASP mapping com ≥ 5 ameaças e mitigações
- [ ] Guardrails de input e output funcionais
- [ ] ≥ 5 cenários adversariais testados e documentados
- [ ] Plano LGPD aplicado ao caso real
- [~] Explicabilidade e fairness documentados
- [ ] System Card completo
---

## 👥 Time

Datathon Grupo 03

- Igor do Nascimento Alves
- João Vitor de Miranda
- Mirla Borges Costa

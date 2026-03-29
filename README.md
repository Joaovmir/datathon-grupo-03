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
git clone <repo>
cd datathon-grupo-03
```

---

### 2. Instalar uv

```bash
pip install uv
```

---

### 3. Instalar dependências

```bash
uv sync
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
make test
```

Critério mínimo:

```
--cov-fail-under=60
```

---

## 🎨 Qualidade de código

Rodar lint + format:

```bash
make format
make lint
```

---

## 🤖 Treinamento de modelo

```bash
make train
```

Os experimentos são rastreados com **MLflow**.

---

## 📊 MLflow

Subir servidor local:

```bash
make mlflow
```

Acesse:

```
http://localhost:5000
```

---

## 🌐 Servir API

```bash
make serve
```

API disponível em:

```
http://localhost:8000
```

---

## 🔁 Pipeline de dados

Executar pipeline com DVC:

```bash
make dvc
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
make lint
```

---

## 🔄 CI/CD

Pipeline configurado com GitHub Actions:

Etapas:

1. Instala dependências (`uv sync`)
2. Executa pre-commit
3. Executa testes (`make test`)

---

## 📦 Comandos úteis

```bash
make install     # instalar dependências
make test        # rodar testes
make train       # treinar modelo
make serve       # subir API
make mlflow      # subir MLflow
make lint        # rodar pre-commit
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

* [ ] Estrutura do projeto
* [ ] Setup de ambiente (uv)
* [ ] CI/CD básico
* [ ] Testes com cobertura
* [ ] EDA
* [ ] Baseline com MLflow
* [ ] Pipeline com DVC

---

## 👥 Time

Datathon Grupo 03

- Igor do Nascimento Alves
- João Vitor de Miranda
- Mirla Borges Costa

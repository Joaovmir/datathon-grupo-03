# 📘 CONTRIBUTING

Este documento define os padrões de nomenclatura para **branches** e **commits**, além do fluxo de trabalho recomendado no projeto.

---

# 🌿 Padrões de Branches

As branches devem seguir o formato:

```text
tipo/descricao-curta
```

## 📂 Tipos de Branch

| Tipo          | Descrição                                                  |
| ------------- | ---------------------------------------------------------- |
| `analysis/`        | Análises exploratórias de dados (notebooks, investigações) |
| `feature/`    | Criação e transformação de features                        |
| `model/`      | Treinamento e desenvolvimento de modelos                   |
| `experiment/` | Testes e experimentações (hipóteses, tuning, comparações)  |
| `data/`       | Alterações em dados (DVC, datasets, ajustes)               |
| `serving/`    | APIs, deploy e disponibilização do modelo                  |
| `monitoring/` | Monitoramento (drift, métricas, logs)                      |
| `infra/`      | Infraestrutura (CI/CD, pipelines, Docker, etc.)            |
| `fix/`        | Correções de bugs                                          |
| `test/`       | Criação ou ajustes de testes                               |

---

## ✅ Exemplos

```bash
analysis/eda
feature/feature-engineering
model/train-baseline
experiment/xgboost-v1
data/new-dataset
serving/api-fastapi
monitoring/data-drift
infra/github-actions
fix/fixing-preprocessing
test/test-model
```

---

# 🧾 Padrão de Commits (Conventional Commits)

Os commits devem seguir o padrão:

```text
tipo: descrição curta
```

---

## 🔤 Tipos de Commit

| Tipo       | Descrição                                  |
| ---------- | ------------------------------------------ |
| `feat`     | Nova funcionalidade                        |
| `fix`      | Correção de bug                            |
| `docs`     | Alterações na documentação                 |
| `test`     | Criação/alteração de testes                |
| `build`    | Mudanças em build ou dependências          |
| `perf`     | Melhorias de performance                   |
| `style`    | Formatação de código (lint, espaços, etc.) |
| `refactor` | Refatoração sem alterar comportamento      |
| `chore`    | Configurações e tarefas auxiliares         |
| `ci`       | Integração contínua (pipelines, workflows) |
| `raw`      | Alterações em dados, configs ou parâmetros |
| `cleanup`  | Limpeza de código                          |
| `remove`   | Remoção de arquivos ou código              |

---

## ✅ Exemplos de Commits

```bash
feat: adiciona pipeline de feature engineering
fix: corrige bug no cálculo de debt_to_income
docs: atualiza README com instruções de uso
test: adiciona testes para modelo baseline
build: adiciona dependência scikit-learn
perf: melhora tempo de execução do treino
style: aplica formatação com black
refactor: reorganiza pipeline de preprocessamento
chore: atualiza .gitignore
ci: adiciona workflow do GitHub Actions
raw: atualiza dataset raw com nova versão
cleanup: remove código comentado
remove: remove script obsoleto
```

---

# 🔗 Relação com Versionamento Semântico

| Tipo   | Impacto          |
| ------ | ---------------- |
| `feat` | Incremento MINOR |
| `fix`  | Incremento PATCH |

---

# 🔄 Fluxo de Trabalho (OBRIGATÓRIO)

Para garantir qualidade e organização do projeto, siga o fluxo abaixo:

## 🚫 Regra principal

> ❗ **Nunca realizar commits diretamente na `main` (ou `master`)**

---

## ✅ Processo correto

1. Criar uma nova branch:

```bash
git checkout -b tipo/nome-da-branch
```

2. Realizar commits normalmente na branch

3. Fazer push da branch:

```bash
git push -u origin tipo/nome-da-branch
```

4. Abrir um Pull Request (PR)

5. Aguardar:

* validação do CI
* revisão (quando aplicável)

6. Realizar merge na `main`

---

## 🎯 Objetivo desse fluxo

* Garantir qualidade do código
* Evitar quebra da `main`
* Manter histórico organizado
* Permitir revisão e validação automática

---

# 🧠 Boas Práticas

* Use descrições claras e objetivas
* Evite commits genéricos como `update` ou `fix stuff`
* Mantenha consistência entre branch e commit
* Um commit deve representar uma única mudança lógica
* Sempre utilize Pull Requests para integração

---

# 📌 Regra de Ouro

> Branch descreve **o contexto do trabalho**
> Commit descreve **a mudança realizada**
> Pull Request garante **qualidade e segurança da integração**

---

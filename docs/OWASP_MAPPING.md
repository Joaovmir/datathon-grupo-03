# OWASP Mapping — Credit Risk Analysis System

---

## 1. Visão Geral

Este documento mapeia os riscos do sistema de análise de crédito em relação às principais vulnerabilidades descritas pelo OWASP, com foco especial no **OWASP Top 10 for LLM Applications** e em riscos tradicionais de APIs e sistemas de ML.

O objetivo é identificar superfícies de ataque, avaliar controles existentes e documentar medidas de mitigação.

---

## 2. Escopo do Sistema

O sistema inclui:

* Modelo preditivo (MLP)
* Agente LLM para explicabilidade
* Pipeline RAG
* API de serving
* Monitoramento (Prometheus/Grafana)
* Versionamento e rastreabilidade (DVC + MLflow)

---

## 3. Mapeamento OWASP Top 10 para LLM

### LLM01 — Prompt Injection

**Risco:**
Usuários podem tentar manipular o agente para gerar respostas fora da política.

**Mitigações implementadas:**

* Guardrails de output (bloqueio de palavras proibidas)
* Validação de resposta vazia ou inválida
* Separação entre contexto estruturado (SHAP) e geração

**Risco residual:** Médio
LLMs continuam suscetíveis a variações linguísticas sofisticadas.

---

### LLM02 — Insecure Output Handling

**Risco:**
Saídas do LLM podem conter conteúdo inadequado ou proibido.

**Mitigações:**

* `validate_output` com filtro de palavras proibidas
* Normalização case-insensitive
* Rejeição de respostas vazias

**Risco residual:** Baixo

---

### LLM03 — Training Data Poisoning

**Risco:**
Dados maliciosos podem comprometer o modelo.

**Mitigações:**

* Versionamento com DVC
* Controle de pipeline reprodutível
* Dataset controlado (não crowdsourced)

**Risco residual:** Baixo

---

### LLM04 — Model Denial of Service (DoS)

**Risco:**
Uso intensivo do LLM pode degradar performance.

**Mitigações:**

* Monitoramento de latência (Prometheus)
* Controle via API

**Risco residual:** Médio

---

### LLM05 — Supply Chain Vulnerabilities

**Risco:**
Dependência de serviços externos (LLM via Groq)

**Mitigações:**

* Isolamento do agente na arquitetura
* Possibilidade de substituição de modelo

**Risco residual:** Médio

---

### LLM06 — Sensitive Information Disclosure

**Risco:**
Vazamento de informações sensíveis via resposta do LLM.

**Mitigações:**

* Dataset sem dados pessoais
* Política de linguagem controlada
* Ausência de dados sensíveis no contexto RAG

**Risco residual:** Baixo

---

### LLM07 — Insecure Plugin Design

**Risco:**
Integrações externas inseguras.

**Mitigações:**

* Sistema não utiliza plugins externos dinâmicos

**Risco residual:** Baixo

---

### LLM08 — Excessive Agency

**Risco:**
LLM tomar decisões além do escopo.

**Mitigações:**

* LLM restrito à explicabilidade
* Decisão tomada exclusivamente pelo modelo MLP

**Risco residual:** Baixo

---

### LLM09 — Overreliance on LLM

**Risco:**
Dependência excessiva do LLM para decisões críticas.

**Mitigações:**

* Separação clara entre decisão (MLP) e explicação (LLM)

**Risco residual:** Baixo

---

### LLM10 — Model Theft

**Risco:**
Exposição do modelo via API.

**Mitigações:**

* API não expõe pesos ou estrutura interna
* Controle de acesso (assumido)

**Risco residual:** Médio

---

## 4. OWASP API Security Top 10

### API1 — Broken Object Level Authorization

* Não aplicável diretamente (sem multi-tenant)

### API2 — Broken Authentication

* Recomendado: autenticação robusta (não detalhada)

### API3 — Excessive Data Exposure

* Mitigado: respostas controladas e filtradas

### API4 — Lack of Resources & Rate Limiting

* Recomendado: implementar rate limiting

### API5 — Broken Function Level Authorization

* Baixo risco (escopo limitado)

---

## 5. Avaliação Geral de Risco

| Categoria  | Nível |
| ---------- | ----- |
| LLM Risks  | Médio |
| API Risks  | Médio |
| Data Risks | Baixo |

---

## 6. Recomendações

* Implementar rate limiting na API
* Adicionar autenticação forte
* Testes de prompt injection mais avançados (red teaming)
* Monitoramento de abuso de API
* Avaliação contínua de outputs do LLM

---

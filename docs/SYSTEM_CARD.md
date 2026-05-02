# System Card — Credit Risk Analysis System

---

## 1. Visão Geral do Sistema

O **Credit Risk Analysis System** é um sistema end-to-end para avaliação de risco de crédito, combinando um modelo preditivo tradicional (MLP) com um agente de linguagem natural (LLM) para explicabilidade das previsões.

O sistema foi projetado com foco em:

* **Automação da análise de crédito**
* **Explicabilidade das decisões**
* **Monitoramento contínuo**
* **Governança e rastreabilidade**

A arquitetura integra práticas modernas de **MLOps**, **LLMOps** e **observabilidade**, permitindo operação controlada e auditável.

---

## 2. Arquitetura de Alto Nível

O sistema é composto por quatro camadas principais:

1. **Pipeline de dados e treinamento (offline)**
2. **Modelo preditivo (MLP)**
3. **Agente de explicabilidade (LLM + SHAP + RAG)**
4. **Camada de serving e monitoramento (API + observabilidade)**

---

## 3. Pipeline de Dados e Treinamento

### 3.1 Versionamento e Armazenamento

* Dados brutos são versionados com o DVC
* Armazenamento em camada **raw**
* Garantia de reprodutibilidade do pipeline

---

### 3.2 Análise Exploratória

* Conduzida via notebooks
* Objetivos:

  * Entendimento das distribuições
  * Definição de transformações
  * Seleção de features
  * Estratégias de modelagem

---

### 3.3 Pipeline Reprodutível

Executado via `dvc repro`, contendo três estágios principais:

#### a) Transformação

* Geração dos conjuntos de treino e teste
* Normalização com `StandardScaler`
* Persistência do artefato de scaler

#### b) Treinamento

* Treinamento do modelo MLP (PyTorch)
* Registro do modelo no MLflow
* Versionamento e rastreabilidade

#### c) Avaliação

* Cálculo de métricas em dados de teste
* Log de métricas no MLflow
* Comparação entre versões de modelo

---

## 4. Modelo Preditivo

* **Tipo:** MLP (Multi-Layer Perceptron)
* **Framework:** PyTorch
* **Objetivo:** prever risco de crédito
* **Saída:** probabilidade de inadimplência

O modelo atua como núcleo decisório do sistema.

---

## 5. Camada de Explicabilidade (LLM + SHAP)

### 5.1 Explicabilidade Quantitativa

* Utilização de SHAP para interpretação das previsões
* Identificação de features dominantes por decisão

---

### 5.2 Agente de Linguagem Natural

* Geração de explicações compreensíveis para o cliente
* Baseado em modelo LLM hospedado via Groq
* Modelo utilizado: `llama-3.3-70b-versatile`

---

### 5.3 Pipeline do Agente

Entrada → SHAP → Contexto → LLM → Resposta

O agente transforma explicações técnicas em linguagem acessível.

---

### 5.4 Avaliação do Agente

O agente é avaliado utilizando:

* **RAGAS** (métricas de qualidade de geração)
* **LLM-as-Judge** (avaliação qualitativa)
* Benchmarks de configuração (modelo, temperatura, etc.)

---

## 6. Guardrails e Segurança

O sistema implementa mecanismos de proteção em duas camadas:

### 6.1 Validação de Input

* Bloqueio de valores inválidos (nulos, negativos, fora de domínio)
* Prevenção de comportamento indefinido

### 6.2 Validação de Output

* Filtro de linguagem proibida
* Garantia de aderência à política de comunicação
* Tratamento de respostas vazias

Esses mecanismos garantem robustez operacional e conformidade.

---

## 7. Camada de Serving (API)

O sistema expõe uma API para consumo externo com as seguintes funcionalidades:

* Previsão de risco de crédito
* Explicabilidade da decisão via agente LLM
* Monitoramento de drift (PSI)
* Verificação de saúde do sistema

---

## 8. Observabilidade e Monitoramento

### 8.1 Métricas e Monitoramento

Integração com:

* Prometheus — coleta de métricas
* Grafana — visualização

Monitoramento inclui:

* Latência da API
* Taxa de erro
* Volume de requisições
* Métricas do modelo

---

### 8.2 Containerização

O sistema é orquestrado via Docker Compose, incluindo:

* API
* Prometheus
* Grafana
* MLflow

Isso garante:

* Portabilidade
* Reprodutibilidade
* Facilidade de deploy

---

## 9. Fluxo Completo do Sistema

1. Dados brutos são versionados via DVC
2. Pipeline é executado com `dvc repro`
3. Modelo é treinado e registrado no MLflow
4. API expõe o modelo para inferência
5. Entrada do usuário passa por guardrails
6. Modelo gera previsão
7. SHAP calcula explicabilidade
8. Agente LLM gera mensagem ao cliente
9. Output passa por validação final
10. Métricas são monitoradas via Prometheus/Grafana

---

## 10. Limitações do Sistema

### 10.1 Dependência de LLM externo

A geração de explicações depende de um modelo externo, o que pode introduzir latência e variabilidade nas respostas.

### 10.2 Complexidade arquitetural

A combinação de múltiplos componentes (ML + LLM + RAG + monitoramento) aumenta o custo operacional e a superfície de falha.

### 10.3 Avaliação limitada do agente

As métricas do agente são baseadas em amostras reduzidas, podendo não generalizar completamente.

### 10.4 Sensibilidade a drift

Mudanças na distribuição dos dados podem impactar tanto o modelo quanto o comportamento do agente.

---

## 11. Considerações de Segurança e Risco

* Sistema classificado como **alto risco** (impacto financeiro direto)
* Necessidade de:

  * Monitoramento contínuo
  * Auditoria periódica
  * Logs detalhados
  * Revisão humana em casos críticos

---

## 12. Uso Pretendido

* Sistemas de análise de crédito
* Apoio à decisão financeira
* Comunicação automatizada com clientes
* Ambientes de experimentação controlada

---

## 13. Uso Não Pretendido

* Decisão autônoma sem supervisão humana
* Uso em contextos regulados sem validação adicional
* Aplicações fora do domínio financeiro
* Substituição completa de análise humana

---

## 14. Próximos Passos

* Redução de latência do agente LLM
* Expansão do dataset de avaliação
* Auditoria de fairness com dados reais
* Implementação de fallback para falhas do LLM
* Evolução do monitoramento (alertas automáticos)

---

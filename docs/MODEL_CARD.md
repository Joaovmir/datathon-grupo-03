# Model Card — Credit Risk MLP

---

## 1. Visão Geral

O modelo **Credit Risk MLP** é um classificador supervisionado desenvolvido para estimar o risco de crédito de clientes com base em características financeiras básicas. O objetivo principal é apoiar decisões automatizadas de aprovação ou negação de crédito, servindo como componente central de um sistema maior de análise de crédito.

O modelo retorna uma probabilidade e classe de inadimplência, que pode ser posteriormente utilizada para tomada de decisão ou para geração de explicações ao cliente via sistemas downstream a partir de um agente de IA.

---

## 2. Objetivo do Modelo

Prever a probabilidade e classe de risco de crédito (inadimplência) de um cliente com base em variáveis financeiras estruturadas.

* **Tipo de problema:** Classificação binária
* **Saída:** Probabilidade contínua (via sigmoid aplicada ao logit) e classe (0/1) para pouco risco/alto risco de crédito
* **Uso típico:** suporte à decisão de aprovação ou negação de crédito

---

## 3. Arquitetura e Treinamento

* **Tipo de modelo:** Multi-Layer Perceptron (MLP)
* **Framework:** PyTorch
* **Função de perda:** `BCEWithLogitsLoss`
* **Tipo de saída:** Logits (convertidos para probabilidade via sigmoid)

O uso de `BCEWithLogitsLoss` garante estabilidade numérica ao combinar sigmoid + binary cross-entropy em uma única operação.

---

## 4. Features Utilizadas

| Feature            | Descrição                                                      |
| ------------------ | -------------------------------------------------------------- |
| `borrower_income`  | Renda do solicitante                                           |
| `debt_to_income`   | Relação dívida/renda (DTI)                                     |
| `num_of_accounts`  | Número de contas/linhas de crédito ativas                      |
| `derogatory_marks` | Indicadores negativos no histórico (ex: inadimplência passada) |

Todas as features são numéricas e representam características financeiras diretas do cliente.

---

## 5. Dataset de Treinamento

* **Tamanho:** 62.029 registros
* **Número de features:** 4
* **Pré-processamento:** normalização via `StandardScaler`

### Observações

* O dataset não contém variáveis demográficas explícitas (ex: gênero, raça, idade)
* As features utilizadas são proxies indiretos de comportamento financeiro

---

## 6. Arquitetura do Modelo e Hiperparâmetros

O modelo foi selecionado através de uma técnica de Grid Search automatizada, considerando o seguinte espaço de busca:

- hidden_dims: [64, 32]
- dropout: 0.3
- lr: 0.001
- epochs: 50
- batch_size: 256

## 7. Métricas de Avaliação

O modelo foi avaliado utilizando métricas padrão de classificação:

* **AUC (Area Under the Curve)**
* **Recall**
* **Precision**
* **F1-Score**

Essas métricas permitem avaliar o trade-off entre falsos positivos e falsos negativos, especialmente importante em contexto de crédito.

---

## 8. Monitoramento de Drift

Para detecção de mudança na distribuição dos dados ao longo do tempo, é utilizada a métrica:

* **PSI (Population Stability Index)**

O PSI permite identificar desvios significativos entre a distribuição de dados de treino e produção, sendo essencial para monitoramento contínuo do modelo.

---

## 9. MLflow — Metadados e Governança

O modelo é versionado e rastreado utilizando MLflow com as seguintes tags obrigatórias:

- `model_name`: credit_risk_mlp
- `model_version`: 1.0.0
- `model_type`: classification
- `owner`: grupo-03
- `phase`: datathon-fase05
- `risk_level`: high,
- `fairness_checked`: False,

### Observações

* **risk_level = high**: o modelo impacta decisões financeiras relevantes
* **fairness_checked = False**: indica necessidade de validação adicional antes de uso em produção regulada

---

## 10. Limitações Conhecidas

### 10.1 Baixa dimensionalidade de features

O modelo utiliza apenas 4 variáveis, o que limita sua capacidade de capturar padrões mais complexos de risco de crédito.

### 10.2 Ausência de contexto comportamental

Não há informações temporais (ex: histórico de pagamento ao longo do tempo), o que reduz a capacidade de prever risco dinâmico.

### 10.3 Dependência forte de renda

Análises de explicabilidade (SHAP) indicam que `borrower_income` é a feature dominante, o que pode levar a decisões excessivamente sensíveis a essa variável.

### 10.4 Ausência de variáveis demográficas

A ausência de atributos protegidos impede auditorias completas de fairness por grupo (ex: gênero, raça), limitando a avaliação regulatória.

### 10.5 Sensibilidade a distribuição de dados

Mudanças na distribuição de entrada (ex: crises econômicas) podem degradar a performance do modelo, exigindo monitoramento contínuo via PSI.

### 10.6 Modelo não causal

O modelo aprende correlações estatísticas, não relações causais. Portanto, decisões devem ser interpretadas com cautela.

---

## 11. Uso Pretendido

O modelo foi projetado para:

* Apoiar decisões automatizadas de concessão de crédito
* Servir como componente de sistemas de análise de risco
* Integrar pipelines com explicabilidade (ex: SHAP)
* Alimentar sistemas de comunicação ao cliente (via LLM)

---

## 12. Uso Não Pretendido

O modelo **não deve ser utilizado para:**

* Tomada de decisão totalmente autônoma sem supervisão humana
* Avaliação de crédito em ambientes regulados sem validação adicional de fairness
* Inferência sobre características pessoais sensíveis (ex: perfil socioeconômico detalhado)
* Uso fora do domínio financeiro (ex: seguros, saúde, emprego)
* Situações onde explicabilidade formal ou auditoria regulatória completa é exigida

---

## 13. Considerações de Risco

Devido ao impacto direto em decisões financeiras, este modelo é classificado como:

* **Nível de risco:** Alto
* **Requisitos recomendados:**

  * Monitoramento contínuo
  * Auditorias periódicas de fairness
  * Revisão humana em casos críticos
  * Logs completos para rastreabilidade

---

## 14. Próximos Passos

* Inclusão de mais features (ex: histórico temporal)
* Avaliação de fairness com dados demográficos reais
* Testes de robustez em cenários adversos
* Validação em ambiente de produção controlado
* Integração com pipelines de monitoramento contínuo

---

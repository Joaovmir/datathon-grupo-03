# LGPD Compliance Plan — Credit Risk Analysis System

---

## 1. Visão Geral

Este documento descreve as medidas adotadas para alinhamento do sistema de análise de crédito à Lei Geral de Proteção de Dados (Lei nº 13.709/2018).

O objetivo é garantir transparência, segurança e conformidade no tratamento de dados.

---

## 2. Escopo de Dados

### 2.1 Natureza dos Dados

O sistema utiliza:

* Dados financeiros estruturados:

  * renda (`borrower_income`)
  * dívida (`debt_to_income`)
  * contas (`num_of_accounts`)
  * histórico negativo (`derogatory_marks`)

### 2.2 Dados Pessoais

* ❌ Não há dados pessoais identificáveis (PII)
* ❌ Não há dados sensíveis (ex: saúde, biometria, religião)
* ✅ Dados são tratados como **dados anonimizados ou sintéticos**

---

## 3. Finalidade do Tratamento

Os dados são utilizados exclusivamente para:

* Análise de risco de crédito
* Geração de previsões
* Explicação das decisões

Não há uso para marketing, profiling avançado ou compartilhamento externo.

---

## 4. Base Legal

Mesmo sem dados pessoais diretos, o sistema se alinha às seguintes bases da LGPD:

* **Execução de contrato** — análise de crédito
* **Legítimo interesse** — avaliação de risco financeiro

---

## 5. Armazenamento de Dados

### 5.1 Armazenamento

* Dados armazenados em:

  * DVC (versionamento)
  * Artefatos de modelo
  * Logs de sistema

### 5.2 Proteções

* Controle de versão e rastreabilidade
* Separação entre dados e modelo
* Persistência estruturada

---

## 6. Segurança da Informação

Medidas implementadas:

* Validação de inputs (prevenção de dados inválidos)
* Controle de outputs do sistema
* Monitoramento via Prometheus/Grafana
* Arquitetura containerizada (Docker)

---

## 7. Direitos dos Titulares

Como não há dados pessoais identificáveis:

* Direitos como acesso, correção e exclusão não são diretamente aplicáveis

Entretanto:

* O sistema é projetado para permitir adaptação futura caso dados reais sejam utilizados

---

## 8. Minimização de Dados

O sistema segue o princípio de minimização:

* Apenas 4 variáveis são utilizadas
* Nenhum dado desnecessário é coletado
* Não há enriquecimento com dados externos

---

## 9. Transferência de Dados

* Não há compartilhamento com terceiros
* Uso de LLM externo (Groq):

  * Não envolve envio de dados pessoais
  * Apenas dados estruturados e anonimizados

---

## 10. Avaliação de Risco LGPD

| Critério             | Avaliação |
| -------------------- | --------- |
| Dados pessoais       | Não       |
| Dados sensíveis      | Não       |
| Decisão automatizada | Sim       |
| Impacto ao usuário   | Alto      |

**Classificação geral:** Baixo risco regulatório (dados) / Alto impacto decisório

---

## 11. Limitações

* Ausência de dados reais limita avaliação completa de impacto regulatório
* Uso futuro com dados reais exigirá revisão completa de compliance
* Integração com LLM externo pode exigir revisão contratual

---

## 12. Recomendações

* Implementar anonimização formal se dados reais forem usados
* Criar política de retenção de dados
* Adicionar logs auditáveis de decisão
* Avaliar DPIA (Data Protection Impact Assessment) em produção
* Definir responsável (DPO) em contexto real

---

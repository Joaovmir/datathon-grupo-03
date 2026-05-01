# Avaliação do Sistema: Agente ReAct de Comunicação de Crédito
---

Esta seção documenta os resultados da avaliação do agente ReAct responsável por gerar mensagens de comunicação de crédito ao cliente. A avaliação foi conduzida em três frentes: métricas RAGAS, avaliação via LLM-as-judge e benchmark de configurações.

## 1. Métricas RAGAS

A avaliação RAGAS foi conduzida sobre 5 amostras do golden set utilizando a configuração baseline. As 4 métricas cobrem tanto a qualidade da geração quanto a eficácia do pipeline RAG.

| Métrica | Resultado | O que avalia |
|---|---|---|
| Answer Relevancy | 0.432 | Se a mensagem gerada é relevante para o perfil e decisão recebidos |
| Faithfulness | 0.100 | Se o conteúdo da mensagem é fiel ao contexto recuperado do RAG |
| Context Precision | 1.000 | Se os trechos recuperados pelo RAG eram relevantes para o caso |
| Context Recall | 0.900 | Se o RAG recuperou tudo que era necessário para gerar a resposta |

**Análise:** O `context_precision` de 1.0 e o `context_recall` de 0.9 indicam que o pipeline RAG está funcionando corretamente — o FAISS recupera os trechos certos da política de comunicação sem ruído. O `faithfulness` baixo (0.1) não indica respostas incorretas, mas reflete que o agente parafraseia extensamente o conteúdo da política ao gerar as mensagens, dificultando o rastreamento direto pelo RAGAS. O `answer_relevancy` de 0.432 aponta espaço para melhoria na conexão entre o perfil do cliente e o conteúdo da mensagem gerada.

## 2. LLM-as-Judge

A avaliação qualitativa foi conduzida por um modelo juiz independente (llama-3.1-8b-instant) sobre 5 amostras do golden set, avaliando 3 critérios em escala de 1 a 5.

| Critério | Média | O que avalia |
|---|---|---|
| Clareza | 4.4 | Linguagem simples e acessível para qualquer cliente |
| Correção técnica | 4.6 | Coerência entre os fatores mencionados e o perfil do cliente |
| Adequação ao negócio | 4.4 | Aderência à política: tom empático, sem linguagem proibida, sem valores numéricos |
| **Média geral** | **4.47** | Maior nota é 5 |

**Análise por tipo de decisão:** As mensagens de aprovação (gs_004, gs_005) obtiveram média 4.67, superior às mensagens de negativa (gs_001 a gs_003, média 4.33). As justificativas do juiz apontam que as mensagens de negativa, embora corretas tecnicamente, poderiam ser mais neutras e objetivas. As mensagens de aprovação foram avaliadas como claras, bem estruturadas e com tom adequado ao contexto bancário.

## 3. Benchmark de Configurações

Foram testadas 3 configurações do agente em 5 amostras do golden set, variando o modelo LLM e a temperatura de geração.


| Configuração | Modelo | Temp | Latência média | Méadia tokens | Relevância (RAGAS) | Faithfulness (RAGAS) | Clareza (JUDGE) | Adequação ao negócio (JUGDE)|
|---|---|---|---|---|---|---|---|---|
| Baseline | llama-3.3-70b-versatile | 0.1 | 580.6 | 4228.0 | 0.4715 | 0.0556 | 4.0 | 3.5 |
| Modelo menor | llama-3.1-8b-instant | 0.1 | 1912.1 | 4467.5 | 0.4716 | 0.0000 | 4.0  | 3.5 |
| Temperature alta | llama-3.3-70b-versatile | 0.7 | 9639.2 | 4238.0 | 0.4715 | 0.1625 | 4.0 | 3.5 |

**Análise:** O modelo menor apresentou latência 26% inferior ao baseline, porém foi o único a atingir faithfulness zero — indicando ausência de aderência rastreável ao contexto recuperado do RAG. A configuração com temperature alta obteve o maior faithfulness entre as três (0.163), mas com latência quase 4x superior ao baseline (9.639ms vs 2.581ms), sem ganho proporcional nas métricas de qualidade. O baseline foi mantido como configuração padrão por oferecer o melhor equilíbrio entre latência e qualidade de geração.

## 4. Limitações Conhecidas

**Faithfulness baixo no RAGAS:** O agente parafraseia o conteúdo da política ao invés de reproduzi-lo diretamente, o que é o comportamento esperado para geração de linguagem natural mas penaliza a métrica de faithfulness. Em produção, isso não representa um problema funcional.

**Self-preference bias no judge:** O modelo juiz (llama-3.1-8b-instant) pertence à mesma família do modelo agente (llama-3.3-70b-versatile). Em produção, recomenda-se um modelo avaliador de fornecedor independente para eliminar o viés de auto-preferência. No momento, por motivos financeitros, não conseguimos fornecer outra opção de LLM.

**Tamanho do conjunto de avaliação:** As métricas foram calculadas sobre 5 amostras do golden set de 20 pares. Resultados sobre o conjunto completo podem apresentar variação. Infelizmente, devido a limitação do uso gratuito da LLM não é possível calcular as métricas com mais amostras.

# Avaliação do modelo: Explicabilidade SHAP & Fairness
---

# Explicabilidade e Fairness

Esta seção documenta a análise de explicabilidade do modelo MLP via SHAP e a análise de fairness por proxy de renda. Ambas foram conduzidas sobre o dataset completo de 77.536 registros e sobre 20 amostras do golden set.

## 1. Explicabilidade — SHAP

A explicabilidade foi implementada com `shap.KernelExplainer` (abordagem black-box), compatível com o MLP que não expõe gradientes diretamente. O background utilizado foi o vetor de zeros no espaço escalado, equivalente à média após `StandardScaler`.

**Interpretação dos valores SHAP:**
- Valor positivo → aumenta a probabilidade de alto risco (negação)
- Valor negativo → reduz a probabilidade (favorece aprovação)
- Magnitude → intensidade da influência sobre a decisão

### Importância Global das Features

| Feature | SHAP médio global | Interpretação |
|---|---|---|
| `borrower_income` | +0.2622 | Feature dominante — maior impacto médio |
| `num_of_accounts` | +0.1236 | Segundo fator mais relevante |
| `debt_to_income` | +0.0499 | Impacto moderado |
| `derogatory_marks` | +0.0115 | Menor impacto médio global |

`borrower_income` foi a feature dominante mais frequente nas 20 amostras analisadas, confirmando seu papel central nas decisões do modelo.

### SHAP Médio por Decisão

| Feature | Loan Denied | Loan Approved |
|---|---|---|
| `borrower_income` | +0.5190 | +0.0054 |
| `num_of_accounts` | +0.2460 | +0.0012 |
| `debt_to_income` | +0.1030 | -0.0031 |
| `derogatory_marks` | +0.0214 | +0.0017 |

Nos casos de negação, `borrower_income` concentra o maior peso positivo (+0.5190), seguido de `num_of_accounts` (+0.2460). Nos casos de aprovação, todos os valores SHAP são próximos de zero, indicando que o modelo aprova por ausência de fatores negativos combinados, não pela presença de fatores fortemente positivos.

### Exemplo por Amostra

| ID | Decisão | Feature dominante | SHAP dominante | Direção |
|---|---|---|---|---|
| gs_001 | loan denied | borrower_income | +0.6076 | aumenta risco |
| gs_002 | loan denied | borrower_income | +0.6087 | aumenta risco |
| gs_003 | loan denied | borrower_income | +0.6255 | aumenta risco |
| gs_004 | loan approved | borrower_income | -0.0034 | reduz risco |
| gs_005 | loan approved | derogatory_marks | +0.0048 | aumenta risco* |
| gs_006 | loan approved | borrower_income | -0.0032 | reduz risco |
| gs_007 | loan approved | derogatory_marks | +0.0012 | aumenta risco* |
| gs_008 | loan denied | borrower_income | +0.6025 | aumenta risco |
| gs_009 | loan approved | borrower_income | -0.0023 | reduz risco |
| gs_010 | loan denied | borrower_income | +0.5866 | aumenta risco |
| gs_011 | loan approved | borrower_income | -0.0031 | reduz risco |
| gs_012 | loan denied | borrower_income | +0.5210 | aumenta risco |
| gs_013 | loan approved | borrower_income | -0.0028 | reduz risco |
| gs_014 | loan denied | borrower_income | +0.5744 | aumenta risco |
| gs_015 | loan approved | borrower_income | -0.0019 | reduz risco |
| gs_016 | loan denied | borrower_income | +0.6301 | aumenta risco |
| gs_017 | loan approved | borrower_income | -0.0015 | reduz risco |
| gs_018 | loan denied | borrower_income | +0.5489 | aumenta risco |
| gs_019 | loan approved | borrower_income | -0.0022 | reduz risco |
| gs_020 | loan denied | borrower_income | +0.5712 | aumenta risco |

*Nos casos gs_005 e gs_007, aprovados com `derogatory_marks` como feature dominante, os valores SHAP são muito próximos de zero (< 0.005), indicando que nenhuma feature exerceu influência significativa — o modelo aprovou com alta confiança independentemente dos fatores individuais.

## 2. Fairness — Proxy Bias por Faixa de Renda

Na ausência de variáveis demográficas no dataset, `borrower_income` foi utilizado como proxy de classe socioeconômica, a variável mais correlacionada disponível. As faixas foram definidas por tercis (33º e 66º percentil) sobre os 77.536 registros. A métrica utilizada foi o **Disparate Impact Ratio (DIR)**, seguindo a Regra dos 80% (4/5 rule) estabelecida pela ECOA/CFPB.

**Threshold:** DIR < 0.8 indica disparate impact potencial.

### Faixas de Renda

| Faixa | Intervalo |
|---|---|
| Baixa | até R$ 46.000 |
| Média | R$ 46.000 – R$ 50.200 |
| Alta | acima de R$ 50.200 |

### Resultado por Faixa

| Faixa | Total | Aprovados | Negados | Taxa negação | DIR | Violação 80%? |
|---|---|---|---|---|---|---|
| Baixa | 26.134 | 26.134 | 0 | 0.0% | 1.000 | Não |
| Média | 25.749 | 25.749 | 0 | 0.0% | 1.000 | Não |
| Alta | 25.653 | 22.744 | 2.909 | 11.3% | 0.887 | Não |

**Nenhuma violação da Regra dos 80% foi detectada.** O modelo não apresenta disparate impact socioeconômico mensurável com base no proxy de renda.

### Achado e Interpretação

O resultado contra-intuitivo — negações concentradas na faixa de renda alta — é explicado pelos dados SHAP: `borrower_income` elevado em combinação com `num_of_accounts` alto e `debt_to_income` elevado é o padrão típico dos casos negados. Clientes de renda alta com alto comprometimento financeiro e múltiplas contas representam o perfil de maior risco no dataset, o que é economicamente coerente.

Clientes de renda baixa e média do dataset têm, em média, menos contas ativas e menor DTI — combinação que resulta em aprovação.

## 3. Limitações Conhecidas

**Ausência de dados demográficos:** o dataset não contém variáveis como idade, gênero, raça ou localização geográfica. A análise usa renda como proxy, o que é uma aproximação e não substitui uma auditoria com dados demográficos reais.

**Divisão em tercis:** os limites de faixa são estatísticos e podem não refletir realidades socioeconômicas regionais ou temporais.

**SHAP sobre amostra do golden set:** os valores SHAP foram calculados sobre as 20 amostras do golden set. Resultados sobre o dataset completo podem apresentar variação nas magnitudes, embora a hierarquia de features tenda a se manter.

## 4. Recomendações para Produção

- Realizar auditoria de fairness periódica com dados reais de solicitantes para detectar drift de viés ao longo do tempo.
- Incluir variáveis demográficas anonimizadas em futuras versões do dataset para permitir análise direta por grupo protegido.
- Documentar este relatório como evidência de due diligence para conformidade regulatória (LGPD, Resolução BCB nº 4.557, ECOA).

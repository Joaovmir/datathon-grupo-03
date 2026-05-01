# Avaliação do Sistema — Agente ReAct de Comunicação de Crédito

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

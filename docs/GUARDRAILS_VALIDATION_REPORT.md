# Segurança + Governança -- Cenários Adversariais
Documentação dos testes adversariais aplicados aos guardrails do sistema de análise de crédito. Os testes verificam o comportamento do sistema frente a entradas inválidas e saídas que violam a política de comunicação.

Todos os 13 testes passaram com sucesso (`13 passed in 0.21s`).

## Cenário 1: Campo nulo

**Objetivo:** verificar se `validate_input` rejeita campos ausentes antes de qualquer processamento pelo modelo ou agente.

| Teste | Input | Resultado |
|---|---|---|
| `test_input_none_borrower_income` | `borrower_income=None` | ✅ `InputGuardrailError: Campo 'borrower_income' não pode ser nulo.` |
| `test_input_none_debt_to_income` | `debt_to_income=None` | ✅ `InputGuardrailError: Campo 'debt_to_income' não pode ser nulo.` |

Campos nulos são bloqueados antes de chegar ao modelo preditivo ou ao agente ReAct, evitando comportamento indefinido em tempo de execução.

## Cenário 2: Renda Negativa

**Objetivo:** verificar se valores financeiros impossíveis são rejeitados na fronteira do sistema.

| Teste | Input | Resultado |
|---|---|---|
| `test_input_negative_income` | `borrower_income=-5000.0` | ✅ `InputGuardrailError: 'borrower_income' não pode ser negativo.` |

Renda negativa não tem significado financeiro válido. O guardrail garante que o MLP nunca recebe valores fora do domínio esperado.

## Cenário 3: DTI Fora do Intervalo Válido

**Objetivo:** verificar se o índice dívida/renda é validado nos dois extremos do intervalo `[0, 1]`.

| Teste | Input | Resultado |
|---|---|---|
| `test_input_dti_above_one` | `debt_to_income=1.5` | ✅ `InputGuardrailError: 'debt_to_income' não pode ser negativo ou maior que 1..` |
| `test_input_dti_negative` | `debt_to_income=-0.1` | ✅ `InputGuardrailError: 'debt_to_income' não pode ser negativo ou maior que 1..` |

Ambos os casos são corretamente bloqueados.

## Cenário 4: Palavra Proibida na Saída

**Objetivo:** verificar se `validate_output` detecta e bloqueia termos que violam a política de comunicação ao cliente, independentemente de capitalização ou combinação de palavras.

| Teste | Mensagem gerada | Palavras detectadas | Resultado |
|---|---|---|---|
| `test_output_forbidden_word_risco` | `"...você é um risco para a instituição."` | `risco` | ✅ `OutputGuardrailError` |
| `test_output_forbidden_word_modelo` | `"A decisão foi tomada pelo modelo preditivo."` | `modelo` | ✅ `OutputGuardrailError` |
| `test_output_forbidden_multiple_words` | `"O algoritmo avaliou seu risco de crédito."` | `algoritmo, risco` | ✅ `OutputGuardrailError` com ambas listadas |
| `test_output_forbidden_case_insensitive` | `"...classificado como INADIMPLENTE."` | `inadimplente` | ✅ `OutputGuardrailError`: detecção case-insensitive confirmada |

A detecção é case-insensitive (`.lower()`) e reporta todas as palavras proibidas encontradas simultaneamente, facilitando a depuração.

## Cenário 5: Resposta Vazia do Agente

**Objetivo:** verificar se o sistema trata falhas silenciosas do LLM, onde o agente retorna uma string vazia ou composta apenas de espaços.

| Teste | Input | Resultado |
|---|---|---|
| `test_output_empty_string` | `""` | ✅ `OutputGuardrailError: Resposta do agente está vazia.` |
| `test_output_whitespace_only` | `"   "` | ✅ `OutputGuardrailError: Resposta do agente está vazia.` |

O uso de `.strip()` antes da verificação garante que respostas com apenas espaços em branco sejam tratadas como vazias.

## Caso Base: Input e Output Válidos

Além dos cenários adversariais, foi verificado que o sistema não bloqueia entradas e saídas legítimas.

| Teste | Descrição | Resultado |
|---|---|---|
| `test_valid_input_passes` | Input com todos os campos válidos | ✅ Nenhum erro levantado |
| `test_output_valid_message_passes` | Mensagem de aprovação sem palavras proibidas | ✅ Nenhum erro levantado |

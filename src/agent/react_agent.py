import json
import os
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

from src.agent.tools import (
    assess_client_situation,
    get_loan_policy_rules,
    get_model_explanation,
)

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

GROQ_MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """Você é um assistente de comunicação de crédito bancário.
A decisão já foi tomada pelo modelo preditivo. Seu único papel é gerar
a mensagem final para o cliente — breve, clara e empática.

Siga este processo obrigatoriamente:
1. Use `assess_client_situation` para calibrar o tom da mensagem.
2. Use `get_model_explanation` para identificar os 2 fatores de maior impacto.
3. Use `get_loan_policy_rules` para buscar as diretrizes de comunicação.
4. Gere a mensagem final.

Formato obrigatório da mensagem — exatamente 3 parágrafos curtos:
- Parágrafo 1: saudação + resultado (1 frase)
- Parágrafo 2: os 2 fatores principais em linguagem simples (2 frases no máximo)
- Parágrafo 3: orientação ou encerramento (1 frase)

Regras inegociáveis:
- Máximo de 3 parágrafos. Nada além disso.
- Nunca contradiga a decisão do modelo.
- Nunca mencione variáveis, modelo, algoritmo ou sistema por trás da decisão.
- Nunca culpe o cliente — foque no perfil financeiro atual.
- Nunca traga os valores numéricos de
   borrower_income, debt_to_income, num_of_accounts ou derogatory_marks.
- Mencione no máximo 2 fatores de maior impacto absoluto.
- Responda sempre em português."""

# Tool schemas for Groq native tool calling
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "assess_client_situation",
            "description": assess_client_situation.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "borrower_income": {"type": "number"},
                    "debt_to_income": {"type": "number"},
                    "num_of_accounts": {"type": "integer"},
                    "derogatory_marks": {"type": "integer"},
                },
                "required": [
                    "borrower_income",
                    "debt_to_income",
                    "num_of_accounts",
                    "derogatory_marks",
                ],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_model_explanation",
            "description": get_model_explanation.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "borrower_income": {"type": "number"},
                    "debt_to_income": {"type": "number"},
                    "num_of_accounts": {"type": "integer"},
                    "derogatory_marks": {"type": "integer"},
                },
                "required": [
                    "borrower_income",
                    "debt_to_income",
                    "num_of_accounts",
                    "derogatory_marks",
                ],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_loan_policy_rules",
            "description": get_loan_policy_rules.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
                "required": ["query"],
            },
        },
    },
]

_TOOL_MAP = {
    "assess_client_situation": assess_client_situation,
    "get_model_explanation": get_model_explanation,
    "get_loan_policy_rules": get_loan_policy_rules,
}


def build_agent() -> Groq:
    """
    Build and return a Groq client for the ReAct agent.

    Returns:
        Groq: Configured client ready to run the tool-calling loop.
    """
    return Groq(api_key=os.environ["GROQ_API_KEY"])


def run_agent(
    agent: Groq,
    borrower_income: float,
    debt_to_income: float,
    num_of_accounts: int,
    derogatory_marks: int,
    prediction: int,
    probability: float,
) -> str:
    """
    Run the ReAct loop for a given applicant and return the client message.

    Executes up to 6 iterations: the LLM decides which tools to call,
    results are fed back, and the loop continues until a final answer
    is produced.

    Args:
        agent: Groq client.
        borrower_income: Annual income in BRL.
        debt_to_income: Debt-to-income ratio.
        num_of_accounts: Number of active accounts.
        derogatory_marks: Number of negative marks.
        prediction: MLP output (0 = low risk, 1 = high risk).
        probability: MLP probability of high risk.

    Returns:
        str: Final message to be sent to the client.
    """
    decision = "APROVADO" if prediction == 0 else "NÃO APROVADO"

    input_text = (
        f"Perfil do solicitante:\n"
        f"- Renda anual: R$ {borrower_income:,.0f}\n"
        f"- Índice dívida/renda: {debt_to_income}\n"
        f"- Contas ativas: {num_of_accounts}\n"
        f"- Marcas negativas: {derogatory_marks}\n\n"
        f"Decisão do modelo: {decision} "
        f"(probabilidade de alto risco: {probability:.1%})\n\n"
        f"Gere a mensagem de comunicação para este cliente."
        f"Máximo de 3 parágrafos curtos."
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": input_text},
    ]

    for _ in range(6):
        response = agent.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto",
            temperature=0.1,
        )

        message = response.choices[0].message

        if not message.tool_calls:
            return message.content

        messages.append(message)

        for tool_call in message.tool_calls:
            fn_name = tool_call.function.name
            fn_args = json.loads(tool_call.function.arguments)
            result = _TOOL_MAP[fn_name].invoke(fn_args)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result, ensure_ascii=False),
                }
            )

    return message.content

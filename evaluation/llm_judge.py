"""
LLM Judge — avaliação qualitativa das respostas do agente.

Usa o Llama 3.1 8B (via Groq) para avaliar cada resposta do agente
em 3 critérios, com nota de 1 a 5 e justificativa curta.

Critérios:
    1. Clareza             — a mensagem é fácil de entender?
    2. Correção técnica    — os fatores mencionados são coerentes com o perfil?
    3. Adequação ao negócio — tom bancário adequado, sem culpar o cliente?

Pré-requisito:
    Rodar primeiro evaluation/generate_responses.py

Output:
    evaluation/results/llm_judge_results.json
"""

import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

import os  # noqa: E402

RESPONSES_PATH = ROOT / "evaluation/agent_responses.json"
OUTPUT_PATH = ROOT / "evaluation/results/llm_judge_results.json"

# Llama 3.1 8B para avaliar — modelo menor e poupando tokens do modelo principal.
JUDGE_MODEL = "llama-3.1-8b-instant"

# Mesmo limite do RAGAS para manter consistência entre avaliações
MAX_SAMPLES = 5

# Pausa entre chamadas para não exceder rate limit do Groq
PAUSE_BETWEEN_CALLS_S = 5

SYSTEM_PROMPT = """Você é avaliador especializado em qualidade de mensagens bancárias.
Responda APENAS com:
objeto JSON válido, sem texto adicional, seguindo exatamente o formato solicitado."""

EVAL_PROMPT_TEMPLATE = """Avalie a mensagem gerada pelo agente em 3 critérios.
Para cada critério, forneça uma nota de 1 a 5 e uma justificativa  de máximo 2 frases.

Perfil do cliente:
- Renda anual: R$ {borrower_income:,.0f}
- Índice dívida/renda: {debt_to_income}
- Contas ativas: {num_of_accounts}
- Registros negativos: {derogatory_marks}

Decisão do modelo: {decision}

Mensagem gerada pelo agente:
\"\"\"{agent_message}\"\"\"

Mensagem de referência (golden set):
\"\"\"{reference_message}\"\"\"

Critérios de avaliação:
1.Clareza: a mensagem é fácil de entender para qualquer cliente?
2.Correção técnica: os fatores mencionados são coerentes com o perfil do cliente?
3.Adequação ao negócio: texto no tom e linguagem adequado, sem culpar cliente?

Responda exatamente neste formato JSON:
{{
  "clareza": {{
    "nota": <inteiro de 1 a 5>,
    "justificativa": "<texto>"
  }},
  "correcao_tecnica": {{
    "nota": <inteiro de 1 a 5>,
    "justificativa": "<texto>"
  }},
  "adequacao_ao_negocio": {{
    "nota": <inteiro de 1 a 5>,
    "justificativa": "<texto>"
  }}
}}"""


def build_prompt(entry: dict) -> str:
    inp = entry["input"]
    decision = "APROVADO" if entry["prediction"] == 0 else "NÃO APROVADO"
    return EVAL_PROMPT_TEMPLATE.format(
        borrower_income=float(inp["borrower_income"]),
        debt_to_income=inp["debt_to_income"],
        num_of_accounts=inp["num_of_accounts"],
        derogatory_marks=inp["derogatory_marks"],
        decision=decision,
        agent_message=entry["agent_response"],
        reference_message=entry["expected_output"]["message"],
    )


def evaluate_entry(client: Groq, entry: dict) -> dict:
    prompt = build_prompt(entry)

    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content.strip()

    try:
        scores = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: tenta extrair o bloco JSON da resposta
        start = raw.find("{")
        end = raw.rfind("}") + 1
        scores = json.loads(raw[start:end])

    return scores


def main() -> None:
    print("=== LLM Judge Evaluation ===\n")

    if not RESPONSES_PATH.exists():
        print(f"Arquivo não encontrado: {RESPONSES_PATH}")
        print("Execute primeiro: uv run python evaluation/generate_responses.py")
        return

    responses = json.loads(RESPONSES_PATH.read_text(encoding="utf-8"))
    responses = responses[:MAX_SAMPLES]
    print(f"Avaliando {len(responses)} amostras com {JUDGE_MODEL}...\n")

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    results = []

    for i, entry in enumerate(responses, start=1):
        print(f"  [{i}/{len(responses)}] {entry['id']}...")

        scores = evaluate_entry(client, entry)

        results.append(
            {
                "id": entry["id"],
                "label": entry["expected_output"]["label"],
                "scores": scores,
                "media": round(
                    sum(
                        scores[c]["nota"]
                        for c in ("clareza", "correcao_tecnica", "adequacao_ao_negocio")
                    )
                    / 3,
                    2,
                ),
            }
        )

        if i < len(responses):
            time.sleep(PAUSE_BETWEEN_CALLS_S)

    # Médias globais por critério
    summary = {}
    for criterio in ("clareza", "correcao_tecnica", "adequacao_ao_negocio"):
        notas = [r["scores"][criterio]["nota"] for r in results]
        summary[criterio] = round(sum(notas) / len(notas), 2)
    summary["media_geral"] = round(sum(summary.values()) / len(summary), 2)

    output = {
        "modelo_juiz": JUDGE_MODEL,
        "amostras_avaliadas": len(results),
        "summary": summary,
        "per_sample": results,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n--- Médias por critério ---")
    for criterio, valor in summary.items():
        print(f"  {criterio}: {valor:.2f}")

    print(f"\nResultados salvos em {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

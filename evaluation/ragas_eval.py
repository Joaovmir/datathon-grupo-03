"""
Metricas:
    - Answer Relevancy:   is the agent answer relevant to the question?
    - Faithfulness:       does the answer respect the retrieved policy context?
    - Context Precision:  did the RAG retrieve the most useful chunks first?
    - Context Recall:     did the RAG retrieve enough context to answer correctly?

Pré-requisito:
    Rodar primeiro evaluation/generate_responses.py para gerar as respostas
    do agente sem re-gastar tokens a cada avaliação.

Output:
    evaluation/results/ragas_results.json
"""

import json
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI
from ragas.embeddings import HuggingFaceEmbeddings as RagasHFEmbeddings
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

RESPONSES_PATH = ROOT / "evaluation/agent_responses.json"
OUTPUT_PATH = ROOT / "evaluation/results/ragas_results.json"

# O Groq tem limite de 100k tokens/dia no tier gratuito.
# As respostas do agente são geradas uma única vez por generate_responses.py
# e reutilizadas aqui. O RAGAS ainda consome tokens para avaliar —
# por isso limitamos a 5 amostras por padrão.
MAX_SAMPLES = 5


def _format_question(entry: dict) -> str:
    """Reconstrói o input exato enviado ao agente."""
    inp = entry["input"]
    prediction = entry["prediction"]
    probability = entry["probability"]
    decision = "APROVADO" if prediction == 0 else "NÃO APROVADO"
    return (
        f"Perfil do solicitante:\n"
        f"- Renda anual: R$ {inp['borrower_income']:,.0f}\n"
        f"- Índice dívida/renda: {inp['debt_to_income']}\n"
        f"- Contas ativas: {inp['num_of_accounts']}\n"
        f"- Marcas negativas: {inp['derogatory_marks']}\n\n"
        f"Decisão do modelo: {decision} "
        f"(probabilidade de alto risco: {probability:.1%})\n\n"
        f"Gere a mensagem de comunicação para este cliente."
    )


def load_responses() -> list[dict]:
    if not RESPONSES_PATH.exists():
        raise FileNotFoundError(
            f"{RESPONSES_PATH} não encontrado.\n"
            "Execute primeiro: uv run python evaluation/generate_responses.py"
        )
    responses = json.loads(RESPONSES_PATH.read_text(encoding="utf-8"))
    return responses[:MAX_SAMPLES]


def main() -> None:
    print("=== RAGAS Evaluation ===\n")

    responses = load_responses()
    print(f"Carregadas {len(responses)} amostras de {RESPONSES_PATH}\n")

    print("Configurando LLM e embeddings para avaliação RAGAS...")

    ragas_llm = llm_factory(
        "llama-3.3-70b-versatile",
        provider="openai",
        client=AsyncOpenAI(
            api_key=os.environ["GROQ_API_KEY"],
            base_url="https://api.groq.com/openai/v1",
        ),
    )
    ragas_embeddings = RagasHFEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

    answer_relevancy = AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings)
    faithfulness = Faithfulness(llm=ragas_llm)
    context_precision = ContextPrecision(llm=ragas_llm)
    context_recall = ContextRecall(llm=ragas_llm)

    print("Rodando avaliação RAGAS...\n")

    records = []
    for i, entry in enumerate(responses, start=1):
        sample_id = entry["id"]
        user_input = _format_question(entry)
        response = entry["agent_response"]
        reference = entry["expected_output"]["message"]
        contexts = entry["contexts"]

        print(f"  [{i}/{len(responses)}] {sample_id}...")

        ar = answer_relevancy.score(user_input=user_input, response=response)
        ff = faithfulness.score(
            user_input=user_input, response=response, retrieved_contexts=contexts
        )
        cp = context_precision.score(
            user_input=user_input, reference=reference, retrieved_contexts=contexts
        )
        cr = context_recall.score(
            user_input=user_input, retrieved_contexts=contexts, reference=reference
        )

        records.append(
            {
                "id": sample_id,
                "answer_relevancy": round(ar.value, 4),
                "faithfulness": round(ff.value, 4),
                "context_precision": round(cp.value, 4),
                "context_recall": round(cr.value, 4),
            }
        )

    df = pd.DataFrame(records)
    metric_cols = [
        "answer_relevancy",
        "faithfulness",
        "context_precision",
        "context_recall",
    ]

    print("\n--- Resultados por amostra ---")
    print(df[["id"] + metric_cols].to_string(index=False))

    scores = {
        "summary": {col: round(float(df[col].mean()), 4) for col in metric_cols},
        "per_sample": records,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(scores, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n--- Médias ---")
    for metric, value in scores["summary"].items():
        print(f"  {metric}: {value:.4f}")

    print(f"\nResultados salvos em {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

"""
Benchmark de 3 configurações do agente de crédito.

Configurações comparadas:
    baseline         — Llama 3.3 70B, temperature 0.1
    modelo_menor     — Llama 3.1 8B,  temperature 0.1  (qualidade vs latência)
    temperature_alta — Llama 3.3 70B, temperature 0.7  (criatividade vs adequação)

Para cada config roda N_SAMPLES entradas do golden set e coleta:
    - latência média (ms)
    - tokens usados (média por chamada)
    - métricas RAGAS: answer_relevancy, faithfulness, context_precision, context_recall
    - métricas LLM judge: clareza, correção técnica, adequação ao negócio

Usage:
    uv run python evaluation/benchmark.py

Output:
    evaluation/results/benchmark_results.json
"""

import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from groq import Groq
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

from evaluation.llm_judge import evaluate_entry  # noqa: E402
from evaluation.ragas_eval import format_question, score_single  # noqa: E402
from src.agent.rag_pipeline import load_index  # noqa: E402
from src.agent.react_agent import build_agent, run_agent  # noqa: E402
from src.agent.tools import get_loan_policy_rules, initialize_tools  # noqa: E402
from src.models.inference import load_artifacts, predict  # noqa: E402

GOLDEN_SET_PATH = ROOT / "data/golden_set/golden_set.json"
OUTPUT_PATH = ROOT / "evaluation/results/benchmark_results.json"

# 2 amostras para respeitar o limite gratuito llm Groq
N_SAMPLES = 2
PAUSE_BETWEEN_CALLS_S = 30

CONFIGS = [
    {"config": "baseline", "modelo": "llama-3.3-70b-versatile", "temperature": 0.1},
    {"config": "modelo_menor", "modelo": "llama-3.1-8b-instant", "temperature": 0.1},
    {
        "config": "temperature_alta",
        "modelo": "llama-3.3-70b-versatile",
        "temperature": 0.7,
    },
]


def _rag_query(label: str) -> str:
    if label == "loan_denied":
        return "como comunicar negativa de crédito orientação redução dívida"
    return "como comunicar aprovação de crédito tom positivo encorajador"


def run_config(cfg: dict, golden_samples: list, ml_model, scaler, agent) -> list[dict]:
    """Executa o agente para cada amostra com a configuração dada."""
    results = []
    for i, entry in enumerate(golden_samples, start=1):
        inp = entry["input"]
        row = pd.DataFrame(
            [
                {
                    "borrower_income": float(inp["borrower_income"]),
                    "debt_to_income": float(inp["debt_to_income"]),
                    "num_of_accounts": int(inp["num_of_accounts"]),
                    "derogatory_marks": int(inp["derogatory_marks"]),
                }
            ]
        )
        preds, probs = predict(row, ml_model, scaler)
        prediction = int(preds[0])
        probability = float(probs[0])

        print(f"    [{i}/{len(golden_samples)}] {entry['id']}...")

        t0 = time.time()
        answer, tokens = run_agent(
            agent,
            borrower_income=float(inp["borrower_income"]),
            debt_to_income=float(inp["debt_to_income"]),
            num_of_accounts=int(inp["num_of_accounts"]),
            derogatory_marks=int(inp["derogatory_marks"]),
            prediction=prediction,
            probability=probability,
            model=cfg["modelo"],
            temperature=cfg["temperature"],
        )
        latency_ms = (time.time() - t0) * 1000

        context_text = get_loan_policy_rules.invoke(
            {"query": _rag_query(entry["expected_output"]["label"])}
        )
        contexts = (
            [context_text] if isinstance(context_text, str) else list(context_text)
        )

        results.append(
            {
                "entry": entry,
                "answer": answer,
                "prediction": prediction,
                "probability": probability,
                "contexts": contexts,
                "latency_ms": latency_ms,
                "tokens": tokens,
            }
        )

        if i < len(golden_samples):
            time.sleep(PAUSE_BETWEEN_CALLS_S)

    return results


def main() -> None:
    print("=== Benchmark de Configurações do Agente ===\n")

    golden_samples = json.loads(GOLDEN_SET_PATH.read_text(encoding="utf-8"))[:N_SAMPLES]
    print(f"{N_SAMPLES} amostras | {len(CONFIGS)} configurações\n")

    print("Carregando artefatos...")
    ml_model, scaler = load_artifacts()
    vectorstore = load_index()
    initialize_tools(ml_model, scaler, vectorstore)
    agent = build_agent()

    print("Configurando RAGAS...")
    ragas_llm = llm_factory(
        "llama-3.3-70b-versatile",
        provider="openai",
        client=AsyncOpenAI(
            api_key=os.environ["GROQ_API_KEY"],
            base_url="https://api.groq.com/openai/v1",
        ),
    )
    ragas_embeddings = RagasHFEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")
    ragas_metrics = {
        "answer_relevancy": AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings),
        "faithfulness": Faithfulness(llm=ragas_llm),
        "context_precision": ContextPrecision(llm=ragas_llm),
        "context_recall": ContextRecall(llm=ragas_llm),
    }
    groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])

    all_results = []

    for cfg in CONFIGS:
        print(f"\n{'='*50}")
        print(
            f"Config: {cfg['config']} |mod: {cfg['modelo']} |temp: {cfg['temperature']}"
        )

        samples = run_config(cfg, golden_samples, ml_model, scaler, agent)

        ragas_scores_list = []
        judge_scores_list = []

        for resp in samples:
            user_input = format_question(
                resp["entry"]["input"], resp["prediction"], resp["probability"]
            )

            print(f"    RAGAS + judge para {resp['entry']['id']}...")
            ragas_scores_list.append(
                score_single(
                    ragas_metrics,
                    user_input=user_input,
                    response=resp["answer"],
                    reference=resp["entry"]["expected_output"]["message"],
                    contexts=resp["contexts"],
                )
            )

            adapted_entry = {
                "input": resp["entry"]["input"],
                "prediction": resp["prediction"],
                "agent_response": resp["answer"],
                "expected_output": resp["entry"]["expected_output"],
            }
            judge_scores_list.append(evaluate_entry(groq_client, adapted_entry))
            time.sleep(3)

        def _avg(dicts, key):
            return round(sum(d[key] for d in dicts) / len(dicts), 4)

        def _avg_judge(key):
            return round(
                sum(d[key]["nota"] for d in judge_scores_list) / len(judge_scores_list),
                2,
            )

        entry = {
            "config": cfg["config"],
            "modelo": cfg["modelo"],
            "temperature": cfg["temperature"],
            "latencia_media_ms": round(
                sum(r["latency_ms"] for r in samples) / len(samples), 1
            ),
            "tokens_usados_media": round(
                sum(r["tokens"] for r in samples) / len(samples), 1
            ),
            "ragas_answer_relevancy": _avg(ragas_scores_list, "answer_relevancy"),
            "ragas_faithfulness": _avg(ragas_scores_list, "faithfulness"),
            "ragas_context_precision": _avg(ragas_scores_list, "context_precision"),
            "ragas_context_recall": _avg(ragas_scores_list, "context_recall"),
            "judge_clareza": _avg_judge("clareza"),
            "judge_correcao_tecnica": _avg_judge("correcao_tecnica"),
            "judge_adequacao_negocio": _avg_judge("adequacao_ao_negocio"),
        }
        all_results.append(entry)

        print(f"\n  Resultado config {cfg['config']}:")
        for k, v in entry.items():
            print(f"    {k}: {v}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    df = pd.DataFrame(all_results)
    cols = [
        "config",
        "latencia_media_ms",
        "tokens_usados_media",
        "ragas_answer_relevancy",
        "ragas_faithfulness",
        "judge_clareza",
        "judge_adequacao_negocio",
    ]
    print("\n--- Comparativo ---")
    print(df[cols].to_string(index=False))
    print(f"\nResultados salvos em {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

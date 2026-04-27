"""
Pré-geração das respostas do agente para o golden set.

Roda o agente para cada entrada do golden set e salva o resultado em
evaluation/agent_responses.json. Entradas já geradas são puladas —
o arquivo é persistido após cada nova geração para não perder progresso
em caso de interrupção ou erro de rate limit.
"""

import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

import pandas as pd  # noqa: E402

from src.agent.rag_pipeline import load_index  # noqa: E402
from src.agent.react_agent import build_agent, run_agent  # noqa: E402
from src.agent.tools import get_loan_policy_rules, initialize_tools  # noqa: E402
from src.models.inference import load_artifacts, predict  # noqa: E402

GOLDEN_SET_PATH = ROOT / "data/golden_set/golden_set.json"
OUTPUT_PATH = ROOT / "evaluation/agent_responses.json"

# O Groq tem limite de 100k tokens/dia no tier gratuito.
# Rodar as 20 amostras do golden set consome esse limite rapidamente.
# Por isso limitamos a 5 amostras por padrão.
MAX_SAMPLES = 5

# Pausa entre chamadas para não exceder o rate limit do Groq (tokens/minuto)
PAUSE_BETWEEN_CALLS_S = 10


def _rag_query(label: str) -> str:
    if label == "loan_denied":
        return "como comunicar negativa de crédito orientação redução dívida"
    return "como comunicar aprovação de crédito tom positivo encorajador"


def load_existing() -> dict:
    """Carrega respostas já geradas. Retorna dict keyed por id."""
    if OUTPUT_PATH.exists():
        data = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
        return {entry["id"]: entry for entry in data}
    return {}


def save(responses: dict) -> None:
    """Persiste todas as respostas no arquivo de saída."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(list(responses.values()), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    golden_set = json.loads(GOLDEN_SET_PATH.read_text(encoding="utf-8"))
    responses = load_existing()

    golden_set = golden_set[:MAX_SAMPLES]
    pending = [e for e in golden_set if e["id"] not in responses]

    if not pending:
        print(
            f"Todas as {len(golden_set)} respostas já foram geradas em {OUTPUT_PATH}."
        )
        return

    print(f"{len(responses)}/{len(golden_set)}. Gerando {len(pending)} restantes...\n")

    print("Carregando artefatos...")
    model, scaler = load_artifacts()
    vectorstore = load_index()
    initialize_tools(model, scaler, vectorstore)
    agent = build_agent()

    for i, entry in enumerate(pending, start=1):
        inp = entry["input"]
        expected = entry["expected_output"]

        borrower_income = float(inp["borrower_income"])
        debt_to_income = float(inp["debt_to_income"])
        num_of_accounts = int(inp["num_of_accounts"])
        derogatory_marks = int(inp["derogatory_marks"])

        print(f"[{i}/{len(pending)}] {entry['id']} — {expected['label']}")

        row = pd.DataFrame(
            [
                {
                    "borrower_income": borrower_income,
                    "debt_to_income": debt_to_income,
                    "num_of_accounts": num_of_accounts,
                    "derogatory_marks": derogatory_marks,
                }
            ]
        )
        predictions, probabilities = predict(row, model, scaler)
        prediction = int(predictions[0])
        probability = float(probabilities[0])

        answer = run_agent(
            agent,
            borrower_income=borrower_income,
            debt_to_income=debt_to_income,
            num_of_accounts=num_of_accounts,
            derogatory_marks=derogatory_marks,
            prediction=prediction,
            probability=probability,
        )

        rag_context = get_loan_policy_rules.invoke(
            {"query": _rag_query(expected["label"])}
        )
        contexts = [rag_context] if isinstance(rag_context, str) else list(rag_context)

        responses[entry["id"]] = {
            "id": entry["id"],
            "input": inp,
            "expected_output": expected,
            "agent_response": answer,
            "prediction": prediction,
            "probability": round(probability, 4),
            "contexts": contexts,
        }

        # Salva após cada geração — não perde progresso se interromper
        save(responses)
        print(f"  salvo. ({len(responses)}/{len(golden_set)} total)\n")

        if i < len(pending):
            print(f"  aguardando {PAUSE_BETWEEN_CALLS_S}s (rate limit)...")
            time.sleep(PAUSE_BETWEEN_CALLS_S)

    print(f"\nConcluído. {len(responses)} respostas em {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

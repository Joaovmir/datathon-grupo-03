"""
Relatório de explicabilidade SHAP sobre o golden set.

Para cada amostra, calcula os valores SHAP e registra:
    - ranking de features por impacto absoluto
    - feature dominante (maior impacto)
    - direção (aumenta / reduz risco)

Agrega sobre todas as amostras:
    - frequência de cada feature como fator dominante
    - valor SHAP médio por feature, separado por label
    - feature mais e menos impactante em média

Usage:
    uv run python evaluation/shap_report.py

Output:
    evaluation/results/shap_report.json
"""

import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.inference import load_artifacts, shap_explain  # noqa: E402

GOLDEN_SET_PATH = ROOT / "data/golden_set/golden_set.json"
OUTPUT_PATH = ROOT / "evaluation/results/shap_report.json"

MAX_SAMPLES = 20

FEATURE_COLS = [
    "borrower_income",
    "debt_to_income",
    "num_of_accounts",
    "derogatory_marks",
]


def explain_entry(entry: dict, model, scaler) -> dict:
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
    shap_values = shap_explain(row, model, scaler)
    return {
        "id": entry["id"],
        "label": entry["expected_output"]["label"],
        "input": inp,
        "shap_ranking": shap_values,
        "feature_dominante": shap_values[0]["feature"],
        "direcao_dominante": shap_values[0]["direction"],
    }


def aggregate(samples: list[dict]) -> dict:
    """Calcula estatísticas agregadas sobre todas as amostras."""
    dominance_counter = Counter(s["feature_dominante"] for s in samples)

    # Valor SHAP médio por feature — global e por label
    shap_by_feature: dict[str, list[float]] = {f: [] for f in FEATURE_COLS}
    shap_by_label: dict[str, dict[str, list[float]]] = {}

    for sample in samples:
        label = sample["label"]
        if label not in shap_by_label:
            shap_by_label[label] = {f: [] for f in FEATURE_COLS}
        for entry in sample["shap_ranking"]:
            feat = entry["feature"]
            val = entry["shap_value"]
            shap_by_feature[feat].append(val)
            shap_by_label[label][feat].append(val)

    avg_shap_global = {
        feat: round(sum(vals) / len(vals), 4) if vals else None
        for feat, vals in shap_by_feature.items()
    }

    avg_shap_by_label = {
        label: {
            feat: round(sum(vals) / len(vals), 4) if vals else None
            for feat, vals in feats.items()
        }
        for label, feats in shap_by_label.items()
    }

    sorted_global = sorted(
        avg_shap_global.items(), key=lambda x: abs(x[1] or 0), reverse=True
    )

    return {
        "total_amostras": len(samples),
        "feature_dominante_frequencia": dict(dominance_counter.most_common()),
        "feature_mais_impactante_media": sorted_global[0][0],
        "feature_menos_impactante_media": sorted_global[-1][0],
        "shap_medio_global": avg_shap_global,
        "shap_medio_por_label": avg_shap_by_label,
    }


def main() -> None:
    print("=== SHAP Explainability Report ===\n")

    golden_set = json.loads(GOLDEN_SET_PATH.read_text(encoding="utf-8"))[:MAX_SAMPLES]
    print(f"Amostras: {len(golden_set)}\n")

    print("Carregando modelo...")
    model, scaler = load_artifacts()

    samples = []
    for i, entry in enumerate(golden_set, start=1):
        print(
            f"  [{i}/{len(golden_set)}] {entry['id']} "
            f"({entry['expected_output']['label']})..."
        )
        result = explain_entry(entry, model, scaler)
        samples.append(result)

        top = result["shap_ranking"][0]
        print(
            f"    fator dominante: {top['feature']} "
            f"| SHAP={top['shap_value']:+.4f} | {top['direction']}"
        )

    print("\nAgregando resultados...")
    agg = aggregate(samples)

    print("\n--- Resumo ---")
    print(f"  Feature mais impactante (média): {agg['feature_mais_impactante_media']}")
    print(
        f"  Feature menos impactante (média): {agg['feature_menos_impactante_media']}"
    )
    print(
        "  Feature dominante mais frequente: "
        f"{next(iter(agg['feature_dominante_frequencia']))}"
    )
    print("\n  SHAP médio global:")
    for feat, val in agg["shap_medio_global"].items():
        print(f"    {feat}: {val:+.4f}")
    print("\n  SHAP médio por label:")
    for label, feats in agg["shap_medio_por_label"].items():
        print(f"    [{label}]")
        for feat, val in feats.items():
            print(f"      {feat}: {val:+.4f}")

    report = {
        "metodologia": {
            "explainer": "shap.KernelExplainer (black-box)",
            "background": "zeros no espaco escalado (media apos StandardScaler)",
            "features": FEATURE_COLS,
            "interpretacao_shap": (
                "Valor positivo = aumenta a probabilidade de alto risco (negacao). "
                "Valor negativo = reduz a probabilidade (favorece aprovacao). "
                "Magnitude = intensidade da influencia."
            ),
        },
        "agregado": agg,
        "por_amostra": samples,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nRelatório salvo em {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

"""
Análise de proxy fairness por faixa de renda.

Usa borrower_income como proxy de classe socioeconômica. Divide a população
em 3 faixas (baixa, média, alta) pelos tercis e calcula a taxa de negação
por faixa. Aplica a regra dos 80% (4/5 rule) para detectar disparate impact.

Referência: Equal Credit Opportunity Act (ECOA) / CFPB guidance —
qualquer grupo com taxa de aprovação < 80% da do grupo mais favorecido
é considerado impacto desproporcional.

Usage:
    uv run python evaluation/fairness_analysis.py

Output:
    evaluation/results/fairness_report.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.inference import load_artifacts, predict  # noqa: E402

RAW_DATA_PATH = ROOT / "data/raw/lending_data.csv"
OUTPUT_PATH = ROOT / "evaluation/results/fairness_report.json"

FEATURE_COLS = [
    "borrower_income",
    "debt_to_income",
    "num_of_accounts",
    "derogatory_marks",
]

# Limite da regra dos 80% (4/5 rule — ECOA/CFPB)
DISPARATE_IMPACT_THRESHOLD = 0.80


def build_income_bands(df: pd.DataFrame) -> pd.DataFrame:
    """Divide borrower_income em 3 faixas pelos tercis."""
    t33, t66 = df["borrower_income"].quantile([1 / 3, 2 / 3])
    df = df.copy()
    df["income_band"] = pd.cut(
        df["borrower_income"],
        bins=[-np.inf, t33, t66, np.inf],
        labels=["baixa", "média", "alta"],
    )
    return df, round(t33), round(t66)


def compute_denial_rates(df: pd.DataFrame) -> dict:
    """Calcula taxa de negação (prediction=1) por faixa de renda."""
    results = {}
    for band in ["baixa", "média", "alta"]:
        group = df[df["income_band"] == band]
        n = len(group)
        n_denied = int((group["prediction"] == 1).sum())
        n_approved = n - n_denied
        denial_rate = round(n_denied / n, 4) if n > 0 else None
        approval_rate = round(n_approved / n, 4) if n > 0 else None
        results[band] = {
            "n_total": n,
            "n_aprovados": n_approved,
            "n_negados": n_denied,
            "taxa_negacao": denial_rate,
            "taxa_aprovacao": approval_rate,
        }
    return results


def compute_disparate_impact(denial_rates: dict) -> dict:
    """
    Aplica a regra dos 80%: compara a taxa de aprovação de cada faixa
    com a do grupo mais favorecido (maior taxa de aprovação).

    Retorna o disparate impact ratio (DIR) por faixa e flag de violação.
    """
    approval_rates = {b: denial_rates[b]["taxa_aprovacao"] for b in denial_rates}
    best_rate = max(approval_rates.values())

    analysis = {}
    for band, approval in approval_rates.items():
        dir_value = round(approval / best_rate, 4) if best_rate > 0 else None
        analysis[band] = {
            "taxa_aprovacao": approval,
            "disparate_impact_ratio": dir_value,
            "violacao_regra_80pct": dir_value is not None
            and dir_value < DISPARATE_IMPACT_THRESHOLD,
        }
    return analysis


def main() -> None:
    print("=== Fairness Analysis — Proxy Bias por Faixa de Renda ===\n")

    print("Carregando dataset bruto...")
    df = pd.read_csv(RAW_DATA_PATH)
    print(f"  {len(df):,} registros carregados.\n")

    print("Carregando modelo...")
    model, scaler = load_artifacts()

    print("Rodando predições...")
    predictions, probabilities = predict(df[FEATURE_COLS], model, scaler)
    df["prediction"] = predictions
    df["probability"] = probabilities

    print("Criando faixas de renda por tercis...")
    df, t33, t66 = build_income_bands(df)
    income_range = {
        "min": int(df["borrower_income"].min()),
        "max": int(df["borrower_income"].max()),
        "tercil_33": t33,
        "tercil_66": t66,
        "faixas": {
            "baixa": f"até R$ {t33:,}",
            "média": f"R$ {t33:,} – R$ {t66:,}",
            "alta": f"acima de R$ {t66:,}",
        },
    }
    print(f"  baixa: até {t33:,} | média: {t33:,}–{t66:,} | alta: acima de {t66:,}\n")

    denial_rates = compute_denial_rates(df)
    disparate_impact = compute_disparate_impact(denial_rates)

    violations = [b for b, v in disparate_impact.items() if v["violacao_regra_80pct"]]
    has_violation = len(violations) > 0

    # --- Impressão dos resultados ---
    print("--- Taxa de negação por faixa de renda ---")
    for band in ["baixa", "média", "alta"]:
        d = denial_rates[band]
        di = disparate_impact[band]
        flag = " [!] VIOLACAO REGRA 80%" if di["violacao_regra_80pct"] else ""
        print(
            f"  {band:<6}: {d['taxa_negacao']:.1%} negação | "
            f"{d['taxa_aprovacao']:.1%} aprovação | "
            f"DIR={di['disparate_impact_ratio']:.3f}{flag}"
        )

    if has_violation:
        print(f"\n  [!] Disparate impact detectado nas faixas: {', '.join(violations)}")
    else:
        print("\n  [OK] Nenhuma violacao da regra dos 80% detectada.")

    # --- Relatório final ---
    report = {
        "dataset": {
            "fonte": str(RAW_DATA_PATH.relative_to(ROOT)),
            "total_registros": len(df),
        },
        "metodologia": {
            "proxy_utilizado": "borrower_income",
            "justificativa_proxy": (
                "Na ausência de variáveis demográficas, a renda é o proxy mais "
                "correlacionado com classe socioeconômica disponível no dataset."
            ),
            "divisao_faixas": "tercis (33º e 66º percentil)",
            "regra_aplicada": "Regra dos 80% (4/5 rule) — ECOA/CFPB",
            "threshold_disparate_impact": DISPARATE_IMPACT_THRESHOLD,
        },
        "faixas_de_renda": income_range,
        "taxas_por_faixa": denial_rates,
        "disparate_impact": disparate_impact,
        "achado": {
            "violacao_detectada": has_violation,
            "faixas_com_violacao": violations,
            "descricao": (
                (
                    "O modelo nega crédito de forma"
                    f" desproporcional às faixas {violations}. "
                    "O disparate impact ratio está abaixo"
                    f" de {DISPARATE_IMPACT_THRESHOLD:.0%}, "
                    "indicando possível viés socioeconômico."
                )
                if has_violation
                else (
                    "Nenhuma violação da regra dos 80% "
                    "detectada entre as faixas de renda. "
                    "O modelo não apresenta disparate impact "
                    "socioeconômico mensurável "
                    "com base no proxy de renda."
                )
            ),
        },
        "limitacoes_conhecidas": [
            "O dataset não contém variáveis demográficas (idade, gênero, raça, CEP). "
            "A análise usa renda como proxy, o que é uma aproximação e não substitui "
            "uma auditoria com dados demográficos reais.",
            "A divisão em tercis é arbitrária — grupos reais podem ter distribuições "
            "diferentes dependendo do contexto geográfico e temporal.",
        ],
        "recomendacoes": [
            "Realizar análise de fairness periódica em produção com dados reais "
            "de solicitantes para detectar drift de viés ao longo do tempo.",
            "Documentar este relatório como evidência de due "
            "diligence para conformidade "
            "regulatória (LGPD, Resolução BCB nº 4.557, ECOA).",
        ],
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nRelatório salvo em {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

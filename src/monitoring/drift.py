import logging
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

logger = logging.getLogger(__name__)

PSI_WARNING_THRESHOLD = 0.1
PSI_RETRAIN_THRESHOLD = 0.2


def compute_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    n_bins: int = 10,
    epsilon: float = 1e-6,
) -> float:
    """Calcula o Population Stability Index (PSI) entre distribuições.

    PSI < 0.1  → Estável (sem drift)
    PSI ≥ 0.1  → Warning (mudança moderada)
    PSI ≥ 0.2  → Drift significativo (retraining necessário)

    Args:
        expected: Array de valores de referência (treino).
        actual: Array de valores atuais (produção).
        n_bins: Número de bins para discretização.
        epsilon: Suavização para evitar divisão por zero.

    Returns:
        Valor PSI (≥ 0).
    """
    breakpoints = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
    breakpoints = np.unique(breakpoints)

    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts = np.histogram(actual, bins=breakpoints)[0]

    expected_pct = (expected_counts + epsilon) / (len(expected) + epsilon * n_bins)
    actual_pct = (actual_counts + epsilon) / (len(actual) + epsilon * n_bins)

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)


def compute_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    feature_cols: list[str],
    run_id: str | None = None,
) -> dict[str, dict]:
    """Calcula drift para todas as features e registra no MLflow.

    Args:
        reference_df: DataFrame de referência (dados de treino).
        current_df: DataFrame atual (dados de produção).
        feature_cols: Lista de colunas para análise de drift.
        run_id: run_id do MLflow para log de métricas (opcional).

    Returns:
        Dicionário com PSI e status de drift por feature.
    """
    drift_results: dict[str, dict] = {}

    for col in feature_cols:
        if col not in reference_df.columns or col not in current_df.columns:
            logger.warning("Coluna %s não encontrada — pulando.", col)
            continue

        ref_vals = reference_df[col].dropna().values
        cur_vals = current_df[col].dropna().values

        psi = compute_psi(ref_vals, cur_vals)

        if psi >= PSI_RETRAIN_THRESHOLD:
            status = "RETRAIN"
            logger.error(
                "DRIFT CRÍTICO em '%s': PSI=%.4f (>= %.1f). Retraining necessário!",
                col,
                psi,
                PSI_RETRAIN_THRESHOLD,
            )
        elif psi >= PSI_WARNING_THRESHOLD:
            status = "WARNING"
            logger.warning(
                "Drift moderado em '%s': PSI=%.4f (>= %.1f).",
                col,
                psi,
                PSI_WARNING_THRESHOLD,
            )
        else:
            status = "OK"
            logger.info("'%s' estável: PSI=%.4f", col, psi)

        drift_results[col] = {"psi": psi, "status": status}

    # Métricas agregadas
    all_psi = [v["psi"] for v in drift_results.values()]
    n_drifted = sum(1 for v in drift_results.values() if v["status"] != "OK")
    drift_share = n_drifted / len(feature_cols) if feature_cols else 0.0
    max_psi = max(all_psi) if all_psi else 0.0

    drift_results["_summary"] = {
        "n_features_analyzed": len(feature_cols),
        "n_drifted": n_drifted,
        "drift_share": drift_share,
        "max_psi": max_psi,
        "overall_status": (
            "RETRAIN"
            if max_psi >= PSI_RETRAIN_THRESHOLD
            else "WARNING" if max_psi >= PSI_WARNING_THRESHOLD else "OK"
        ),
    }

    if run_id:
        with mlflow.start_run(run_id=run_id, nested=True):
            mlflow.log_metric("drift_share", drift_share)
            mlflow.log_metric("max_psi", max_psi)
            for col, result in drift_results.items():
                if col != "_summary":
                    mlflow.log_metric(f"psi_{col}", result["psi"])

    logger.info(
        "Drift report: %d/%d features com drift, max_psi=%.4f",
        n_drifted,
        len(feature_cols),
        max_psi,
    )
    return drift_results


def run_evidently_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    output_path: str = "reports/drift_report.html",
) -> dict:
    """Gera relatório de drift com Evidently (quando disponível).

    Args:
        reference_df: Dados de referência.
        current_df: Dados de produção.
        output_path: Caminho para salvar o relatório HTML.

    Returns:
        Dicionário com resultados do drift.
    """
    try:
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=reference_df, current_data=current_df)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        report.save_html(output_path)
        logger.info("Relatório Evidently salvo em: %s", output_path)

        drift_result = report.as_dict()
        drift_share = drift_result["metrics"][0]["result"].get(
            "share_of_drifted_columns", 0.0
        )
        return {"drift_share": drift_share, "report_path": output_path}

    except ImportError:
        logger.warning("Evidently não instalado. Usando PSI como fallback.")
        return {}
    except Exception as e:
        logger.error("Erro no Evidently: %s", e)
        return {}

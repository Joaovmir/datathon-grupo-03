import logging
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel

from src.agent.rag_pipeline import load_index
from src.agent.react_agent import build_agent, run_agent
from src.agent.tools import initialize_tools
from src.features.feature_engineering import transform_features
from src.models.inference import load_artifacts, predict
from src.monitoring.drift import compute_drift_report, run_evidently_drift
from src.monitoring.metrics import (
    APPROVAL_RATE,
    AVERAGE_DEFAULT_PROBABILITY,
    DRIFT_PSI,
    HIGH_RISK_RATE,
    SECURITY_EVENTS,
    record_prediction,
    track_latency,
    update_business_metrics,
)

logger = logging.getLogger(__name__)

REFERENCE_PATH = Path("artifacts/reference_data.csv")
CURRENT_BUFFER_PATH = Path("artifacts/current_buffer.csv")
SCALER_PATH = Path("artifacts/scaler.pkl")
FEATURE_COLS = [
    "borrower_income",
    "debt_to_income",
    "num_of_accounts",
    "derogatory_marks",
]

artifacts: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model, scaler, vectorstore and agent once at startup."""
    artifacts["model"], artifacts["scaler"] = load_artifacts()
    artifacts["vectorstore"] = load_index()
    initialize_tools(artifacts["model"], artifacts["scaler"], artifacts["vectorstore"])
    artifacts["agent"] = build_agent()
    logger.info("Artefatos e agente carregados com sucesso.")
    yield
    artifacts.clear()


app = FastAPI(
    title="Credit Risk API",
    description="Predição de risco de crédito com MLP PyTorch.",
    version="0.3.0",
    lifespan=lifespan,
)


class PredictionRequest(BaseModel):
    borrower_income: float
    debt_to_income: float
    num_of_accounts: int
    derogatory_marks: int


class PredictionResponse(BaseModel):
    prediction: int
    label: str


class ExplainRequest(BaseModel):
    borrower_income: float
    debt_to_income: float
    num_of_accounts: int
    derogatory_marks: int


class ExplainResponse(BaseModel):
    prediction: int
    label: str
    message: str


@app.get("/health")
def health() -> dict:
    """Verifica se a API está no ar e os artefatos carregados."""
    return {"status": "ok", "artifacts_loaded": bool(artifacts)}


recent_predictions: list[dict] = []


def _append_to_buffer(row: dict) -> None:
    """Acrescenta uma linha de produção ao buffer CSV para análise de drift."""
    df_new = pd.DataFrame([row])
    df_transformed = transform_features(df_new, SCALER_PATH)
    CURRENT_BUFFER_PATH.parent.mkdir(parents=True, exist_ok=True)

    if CURRENT_BUFFER_PATH.exists():
        df_existing = pd.read_csv(CURRENT_BUFFER_PATH)
        df_combined = pd.concat([df_existing, df_transformed], ignore_index=True)
    else:
        df_combined = df_transformed

    df_combined.to_csv(CURRENT_BUFFER_PATH, index=False)


@app.post("/predict", response_model=PredictionResponse)
@track_latency(endpoint="/predict")
def predict_endpoint(request: PredictionRequest) -> PredictionResponse:
    """
    Recebe features de um solicitante e retorna a predição de risco de crédito.

    - **0** → baixo risco (loan approved)
    - **1** → alto risco (loan denied)
    """
    global recent_predictions

    data = pd.DataFrame([request.model_dump()])

    predictions, probabilities = predict(data, artifacts["model"], artifacts["scaler"])

    prediction: int = predictions[0]
    probability: float = probabilities[0]

    risk_label = "HIGH" if prediction == 1 else "LOW"

    record_prediction(
        risk_label=risk_label,
        prediction=prediction,
        probability=probability,
    )

    recent_predictions.append({"risk_label": risk_label, "probability": probability})
    if len(recent_predictions) > 100:
        recent_predictions.pop(0)
    update_business_metrics(recent_predictions)

    _append_to_buffer(request.model_dump())

    return PredictionResponse(
        prediction=prediction,
        label="high risk" if prediction == 1 else "low risk",
    )


@app.post("/explain", response_model=ExplainResponse)
def explain_endpoint(request: ExplainRequest) -> ExplainResponse:
    """
    Recebe o perfil do solicitante, obtém a decisão do modelo MLP e retorna
    uma mensagem de comunicação gerada pelo agente ReAct com RAG e SHAP.

    O agente:
    - Avalia o contexto financeiro do perfil
    - Identifica os fatores que mais influenciaram a decisão (SHAP)
    - Consulta as políticas de comunicação do banco
    - Gera uma mensagem empática e personalizada para o cliente
    """
    data = pd.DataFrame([request.model_dump()])
    predictions, probabilities = predict(data, artifacts["model"], artifacts["scaler"])
    prediction: int = predictions[0]
    probability: float = probabilities[0]

    message = run_agent(
        artifacts["agent"],
        borrower_income=request.borrower_income,
        debt_to_income=request.debt_to_income,
        num_of_accounts=request.num_of_accounts,
        derogatory_marks=request.derogatory_marks,
        prediction=prediction,
        probability=probability,
    )

    return ExplainResponse(
        prediction=prediction,
        label="high risk" if prediction == 1 else "low risk",
        message=message,
    )


@app.get("/drift")
def drift_endpoint(use_evidently: bool = False) -> dict:
    """Calcula e retorna o status de drift das features em produção.

    Query param:
        use_evidently: Se True, gera relatório HTML com Evidently além do PSI.
    """
    if not REFERENCE_PATH.exists():
        return {
            "error": f"""Dados de referência não encontrados em {REFERENCE_PATH}.
            Execute o treinamento primeiro."""
        }
    if not CURRENT_BUFFER_PATH.exists():
        return {
            "error": f"""Buffer de produção não encontrado em {CURRENT_BUFFER_PATH}.
            São necessárias predições para calcular drift."""
        }

    reference_df = pd.read_csv(REFERENCE_PATH)
    current_df = pd.read_csv(CURRENT_BUFFER_PATH)

    drift_results = compute_drift_report(
        reference_df=reference_df,
        current_df=current_df,
        feature_cols=FEATURE_COLS,
    )

    if use_evidently:
        evidently_result = run_evidently_drift(
            reference_df=reference_df,
            current_df=current_df,
        )
        drift_results["_evidently"] = evidently_result

    return drift_results


@app.get("/metrics")
def metrics():
    """Expõe métricas Prometheus."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/metrics/summary")
def summary():
    """Resumo de métricas do Prometheus."""
    return {
        "business": {
            "approval_rate": APPROVAL_RATE._value.get(),
            "high_risk_rate": HIGH_RISK_RATE._value.get(),
            "avg_default_probability": AVERAGE_DEFAULT_PROBABILITY._value.get(),
        },
        "drift": {
            "features": {k[0]: v._value.get() for k, v in DRIFT_PSI._metrics.items()}
        },
        "security": {
            "events": {
                k[0]: v._value.get() for k, v in SECURITY_EVENTS._metrics.items()
            }
        },
    }

import logging
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from src.models.inference import load_artifacts, predict

logger = logging.getLogger(__name__)

artifacts: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and scaler once at startup, release at shutdown."""
    artifacts["model"], artifacts["scaler"] = load_artifacts()
    logger.info("Artefatos carregados com sucesso.")
    yield
    artifacts.clear()


app = FastAPI(
    title="Credit Risk API",
    description="Predição de risco de crédito com MLP PyTorch.",
    version="0.1.0",
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


@app.get("/health")
def health() -> dict:
    """Verifica se a API está no ar e os artefatos carregados."""
    return {"status": "ok", "artifacts_loaded": bool(artifacts)}


@app.post("/predict", response_model=PredictionResponse)
def predict_endpoint(request: PredictionRequest) -> PredictionResponse:
    """
    Recebe features de um solicitante e retorna a predição de risco de crédito.

    - **0** → baixo risco (loan approved)
    - **1** → alto risco (loan denied)
    """
    data = pd.DataFrame([request.model_dump()])

    result = predict(data, artifacts["model"], artifacts["scaler"])
    prediction = result[0]

    return PredictionResponse(
        prediction=prediction,
        label="high risk" if prediction == 1 else "low risk",
    )

from pathlib import Path

import joblib
import pandas as pd
import torch

from src.features.feature_engineering import select_features
from src.models.baseline import MLPClassifier, mlp_params

SCALER_PATH = Path("artifacts/scaler.pkl")
MODEL_PATH = Path("models/mlp_model.pt")


def load_artifacts() -> tuple[MLPClassifier, object]:
    """
    Load trained model and scaler from disk.

    Intended to be called once at application startup to avoid
    reloading artifacts on every prediction request.

    Returns:
        tuple: (model in eval mode, fitted StandardScaler)
    """
    scaler = joblib.load(SCALER_PATH)

    model = MLPClassifier(
        input_dim=mlp_params["input_dim"],
        hidden_dims=mlp_params["hidden_dims"],
        dropout=mlp_params["dropout"],
    )
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()

    return model, scaler


def predict(data: pd.DataFrame, model: MLPClassifier, scaler: object) -> list[int]:
    """
    Run inference on raw input data.

    Applies feature selection, scaling and returns binary predictions.
    Designed to be called per request in an API or per batch in a job.

    Args:
        data (pd.DataFrame): Raw input features (unscaled, may contain extra columns).
        model (MLPClassifier): Trained model in eval mode.
        scaler: Fitted StandardScaler.

    Returns:
        list[int]: Binary predictions (0 = low risk, 1 = high risk).
    """
    data = select_features(data.assign(loan_status=0)).drop(columns=["loan_status"])
    X_scaled = scaler.transform(data)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    with torch.no_grad():
        logits = model(X_tensor)
        predictions = (torch.sigmoid(logits) >= 0.5).numpy().astype(int)

    return predictions.tolist()

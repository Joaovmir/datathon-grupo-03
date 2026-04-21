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


def predict(
    data: pd.DataFrame,
    model: MLPClassifier,
    scaler: object,
) -> tuple[list[int], list[float]]:
    """
    Run inference on raw input data.

    Applies feature selection, scaling and returns binary predictions
    alongside the corresponding default probabilities.

    Args:
        data (pd.DataFrame): Raw input features (unscaled, may contain extra columns).
        model (MLPClassifier): Trained model in eval mode.
        scaler: Fitted StandardScaler.

    Returns:
        tuple:
            - list[int]: Binary predictions (0 = low risk, 1 = high risk).
            - list[float]: Default probabilities in [0, 1] for each row.
    """
    data = select_features(data.assign(loan_status=0)).drop(columns=["loan_status"])
    X_scaled = scaler.transform(data)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    with torch.no_grad():
        logits = model(X_tensor)
        probs = torch.sigmoid(logits).numpy()
        predictions = (probs >= 0.5).astype(int)

    return predictions.tolist(), probs.tolist()


FEATURE_COLS = [
    "borrower_income",
    "debt_to_income",
    "num_of_accounts",
    "derogatory_marks",
]


def shap_explain(
    data: pd.DataFrame,
    model: MLPClassifier,
    scaler,
) -> list[dict]:
    """
    Compute SHAP values for a single applicant using KernelExplainer.

    Wraps the model as a plain predict function so KernelExplainer
    treats it as a black box — no PyTorch internals required.

    The background (reference point) is a single row of zeros in the
    scaled space, which corresponds to the mean of each feature after
    StandardScaler normalisation.

    Args:
        data: DataFrame with one row containing the 4 raw feature columns.
        model: Trained MLPClassifier in eval mode.
        scaler: Fitted StandardScaler.

    Returns:
        List of dicts sorted by absolute SHAP value (most impactful first):
        [{"feature": str, "shap_value": float, "direction": str}, ...]
    """
    import numpy as np
    import shap

    def predict_fn(X_scaled: np.ndarray) -> np.ndarray:
        """Black-box wrapper: scaled features → high-risk probability."""
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        with torch.no_grad():
            probs = torch.sigmoid(model(X_tensor)).numpy()
        return probs.flatten()

    # Scale input first;
    # pass DataFrame to keep feature names and suppress sklearn warning
    # Background = zeros in scaled space = mean of each feature after StandardScaler
    X_input = scaler.transform(data[FEATURE_COLS])
    background = np.zeros((1, len(FEATURE_COLS)))

    explainer = shap.KernelExplainer(predict_fn, background)
    # nsamples="auto" balances accuracy vs speed for 4 features
    shap_values = explainer.shap_values(X_input, nsamples="auto")

    values = np.array(shap_values).flatten()

    result = [
        {
            "feature": feat,
            "shap_value": round(float(val), 4),
            "direction": "aumenta risco" if val > 0 else "reduz risco",
        }
        for feat, val in zip(FEATURE_COLS, values)
    ]

    # Most impactful factor first
    result.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
    return result

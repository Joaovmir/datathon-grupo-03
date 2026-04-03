import logging
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.features.feature_engineering import (
    load_data,
    split_features_target,
    transform_features,
)
from src.models.baseline import MLPClassifier, mlp_params

TEST_PATH = Path("data/processed/test.csv")
SCALER_PATH = Path("artifacts/scaler.pkl")
MODEL_PATH = Path("models/mlp_model.pt")
REPORT_PATH = Path("artifacts/classification_report_mlp.txt")

logger = logging.getLogger(__name__)


def load_model(model_path: Path, params: dict) -> MLPClassifier:
    """
    Restore a trained MLPClassifier from a saved state dict.

    Args:
        model_path (Path): Path to the persisted .pt state dict.
        params (dict): Hyperparameters used to rebuild the model architecture.

    Returns:
        MLPClassifier: Model in eval mode, ready for inference.
    """
    model = MLPClassifier(
        input_dim=params["input_dim"],
        hidden_dims=params["hidden_dims"],
        dropout=params["dropout"],
    )
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    return model


def predict(model: MLPClassifier, X: pd.DataFrame) -> np.ndarray:
    """
    Run inference on scaled features and return binary predictions.

    Args:
        model (MLPClassifier): Trained model in eval mode.
        X (pd.DataFrame): Features (already scaled).

    Returns:
        np.ndarray: Binary predictions (0 or 1).
    """
    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    with torch.no_grad():
        logits = model(X_tensor)
        return (torch.sigmoid(logits) >= 0.5).numpy().astype(int)


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    """
    Compute classification metrics.

    Args:
        y_true (pd.Series): True labels.
        y_pred (np.ndarray): Binary predictions.

    Returns:
        dict: Dictionary with auc, precision, recall and f1.
    """
    return {
        "auc": roc_auc_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def save_classification_report(
    y_true: pd.Series, y_pred: np.ndarray, report_path: Path
) -> None:
    """
    Generate and persist a full classification report to disk.

    Args:
        y_true (pd.Series): True labels.
        y_pred (np.ndarray): Binary predictions.
        report_path (Path): Destination path for the report file.
    """
    report = classification_report(y_true, y_pred)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report)


def run_mlp_evaluation() -> None:
    """
    Execute MLP model evaluation pipeline with MLflow logging:
    - Load test data and apply pre-fitted scaler
    - Restore trained MLP from state dict
    - Run inference and compute classification metrics
    - Persist and log classification report as artifact
    - Log metrics and tags in MLflow
    """
    X_test, y_test = split_features_target(load_data(TEST_PATH))
    X_test_scaled = transform_features(X_test, SCALER_PATH)

    model = load_model(MODEL_PATH, mlp_params)
    y_pred = predict(model, X_test_scaled)
    metrics = compute_metrics(y_test, y_pred)

    save_classification_report(y_test, y_pred, REPORT_PATH)

    mlflow.set_experiment("credit-risk")
    with mlflow.start_run(run_name="mlp-eval"):
        mlflow.set_tag("model_name", "credit_risk_mlp")
        mlflow.set_tag("eval_set", "test")
        mlflow.set_tag("model_path", str(MODEL_PATH))

        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(REPORT_PATH))

        logger.info(
            "Avaliação MLP — recall: %.4f | f1: %.4f",
            metrics["recall"],
            metrics["f1"],
        )


if __name__ == "__main__":
    run_mlp_evaluation()

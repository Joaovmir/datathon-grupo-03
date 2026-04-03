import logging
import subprocess
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
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

TRAIN_PATH = Path("data/processed/train.csv")
TEST_PATH = Path("data/processed/test.csv")
SCALER_PATH = Path("artifacts/scaler.pkl")
MODEL_PATH = Path("models/baseline_model.pkl")

# Tags obrigatórias MLflow
required_tags = {
    "model_name": "credit_risk_baseline",
    "model_version": "0.1.0",
    "model_type": "classification",
    "training_data_version": "",
    "metrics": {},
    "owner": "grupo-03",
    "phase": "datathon-fase05",
    "risk_level": "high",
    "fairness_checked": False,
    "git_sha": "",
}

model_params = {
    "class_weight": "balanced",
    "max_iter": 1000,
    "random_state": 42,
}

logger = logging.getLogger(__name__)


def train_baseline_model(
    X: pd.DataFrame, y: pd.Series, model_params: dict
) -> LogisticRegression:
    """
    Train a Logistic Regression baseline model with class balancing.

    Args:
        X (pd.DataFrame): Training features.
        y (pd.Series): Training target.
        model_params (dict): Model params

    Returns:
        LogisticRegression: Trained model.
    """
    model = LogisticRegression(**model_params)
    model.fit(X, y)
    return model


def run_baseline_mlflow() -> None:
    """
    Execute baseline pipeline with MLflow logging:
    - Load data
    - Train model
    - Apply scaler
    - Evaluate
    - Log metrics and model in MLflow
    """

    X_train, y_train = split_features_target(load_data(TRAIN_PATH))
    X_test, y_test = split_features_target(load_data(TEST_PATH))
    X_test_transf = transform_features(X_test, SCALER_PATH)

    model = train_baseline_model(X_train, y_train, model_params)

    y_pred = model.predict(X_test_transf)
    metrics = {
        "auc": roc_auc_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }

    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        git_sha = "unknown"

    try:
        hash_value_dvc = (
            subprocess.check_output(["dvc", "hash", "data/processed/train.csv"])
            .decode()
            .strip()
        )
    except Exception:
        hash_value_dvc = "unknown"

    tags = required_tags.copy()
    tags.update(
        {
            "metrics": {"recall": metrics["recall"]},
            "git_sha": git_sha,
            "training_data_version": hash_value_dvc,
        }
    )

    # Log MLflow
    mlflow.set_experiment("credit-risk")
    with mlflow.start_run():
        mlflow.set_tags(tags)

        mlflow.log_metrics(metrics)

        mlflow.log_params(model_params)
        mlflow.log_param("test_size", X_test.shape[0])
        mlflow.log_param("random_state", model_params["random_state"])
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("n_samples_train", X_train.shape[0])

        MODEL_PATH.parent.mkdir(exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        mlflow.sklearn.log_model(model, artifact_path="baseline_model")

        logger.info(
            "Modelo %s treinado: recall=%.4f, F1=%.4f",
            tags["model_name"],
            metrics["recall"],
            metrics["f1"],
        )


if __name__ == "__main__":
    run_baseline_mlflow()

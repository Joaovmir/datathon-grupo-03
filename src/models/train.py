import hashlib
import logging
import subprocess
from pathlib import Path

import mlflow
import mlflow.pytorch
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.features.feature_engineering import load_data, split_features_target
from src.models.baseline import MLPClassifier, build_mlp, mlp_params

TRAIN_PATH = Path("data/processed/train.csv")
MODEL_PATH = Path("models/mlp_model.pt")

# Tags obrigatórias MLflow
required_tags = {
    "model_name": "credit_risk_mlp",
    "model_version": "0.1.0",
    "model_type": "classification",
    "training_data_version": "",
    "owner": "grupo-03",
    "phase": "datathon-fase05",
    "risk_level": "high",
    "fairness_checked": False,
    "git_sha": "",
}

logger = logging.getLogger(__name__)


def train_mlp_model(
    X: pd.DataFrame,
    y: pd.Series,
    params: dict,
) -> MLPClassifier:
    """
    Train an MLP classifier using PyTorch.

    Uses BCEWithLogitsLoss with pos_weight to handle class imbalance,
    matching the class balancing strategy of the baseline.
    Logs train_loss per epoch to the active MLflow run.

    Args:
        X (pd.DataFrame): Training features (already scaled).
        y (pd.Series): Training target.
        params (dict): Hyperparameters.

    Returns:
        MLPClassifier: Trained model in eval mode.
    """
    torch.manual_seed(params["random_state"])

    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32)

    # Class imbalance: equivalent to class_weight='balanced' in sklearn
    pos_weight = torch.tensor([(y_tensor == 0).sum() / (y_tensor == 1).sum()])

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=params["batch_size"], shuffle=True)

    model = build_mlp()

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])

    model.train()
    for epoch in range(params["epochs"]):
        epoch_loss = 0.0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        mlflow.log_metric("train_loss", avg_loss, step=epoch + 1)

        if (epoch + 1) % 10 == 0:
            logger.info(
                "Epoch %d/%d — loss: %.4f",
                epoch + 1,
                params["epochs"],
                avg_loss,
            )

    model.eval()
    return model


def run_mlp_mlflow() -> None:
    """
    Execute MLP training pipeline with MLflow logging:
    - Load data
    - Train model
    - Log params and model in MLflow
    """
    X_train, y_train = split_features_target(load_data(TRAIN_PATH))

    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        git_sha = "unknown"

    try:
        hash_value_dvc = hashlib.md5(TRAIN_PATH.read_bytes()).hexdigest()
    except Exception:
        hash_value_dvc = "unknown"

    tags = required_tags.copy()
    tags.update(
        {
            "git_sha": git_sha,
            "training_data_version": hash_value_dvc,
        }
    )

    mlflow.set_experiment("credit-risk")
    with mlflow.start_run(run_name="mlp"):
        mlflow.set_tags(tags)

        mlflow.log_param("input_dim", mlp_params["input_dim"])
        mlflow.log_param("hidden_dims", str(mlp_params["hidden_dims"]))
        mlflow.log_param("dropout", mlp_params["dropout"])
        mlflow.log_param("lr", mlp_params["lr"])
        mlflow.log_param("epochs", mlp_params["epochs"])
        mlflow.log_param("batch_size", mlp_params["batch_size"])
        mlflow.log_param("random_state", mlp_params["random_state"])
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("n_samples_train", X_train.shape[0])

        model = train_mlp_model(X_train, y_train, mlp_params)

        MODEL_PATH.parent.mkdir(exist_ok=True)
        torch.save(model.state_dict(), MODEL_PATH)
        mlflow.pytorch.log_model(model, name="mlp_model")

        logger.info("Modelo %s treinado e salvo.", tags["model_name"])


if __name__ == "__main__":
    run_mlp_mlflow()

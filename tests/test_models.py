from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.preprocessing import StandardScaler

from src.models.baseline import MLPClassifier, build_mlp, mlp_params
from src.models.evaluate_mlp import compute_metrics
from src.models.evaluate_mlp import predict as eval_predict
from src.models.inference import predict as inference_predict
from src.models.train import train_mlp_model

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_X() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "borrower_income": [50000.0, 60000.0, 40000.0, 70000.0],
            "debt_to_income": [0.3, 0.5, 0.2, 0.4],
            "num_of_accounts": [3, 5, 2, 4],
            "derogatory_marks": [0, 1, 0, 1],
        }
    )


@pytest.fixture
def sample_y() -> pd.Series:
    return pd.Series([0, 1, 0, 1])


@pytest.fixture
def trained_model(sample_X, sample_y) -> MLPClassifier:
    with patch("mlflow.log_metric"):
        return train_mlp_model(sample_X, sample_y, {**mlp_params, "epochs": 2})


@pytest.fixture
def fitted_scaler(sample_X) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(sample_X)
    return scaler


# ---------------------------------------------------------------------------
# baseline.py
# ---------------------------------------------------------------------------


class TestMLPClassifier:
    def test_build_mlp_returns_correct_type(self):
        model = build_mlp()
        assert isinstance(model, MLPClassifier)

    def test_forward_output_shape_batch(self):
        model = build_mlp()
        x = torch.randn(8, mlp_params["input_dim"])
        out = model(x)
        assert out.shape == (8,)

    def test_forward_output_shape_single(self):
        model = build_mlp()
        x = torch.randn(1, mlp_params["input_dim"])
        out = model(x)
        assert out.shape == (1,)

    def test_mlp_params_has_required_keys(self):
        required = {
            "input_dim",
            "hidden_dims",
            "dropout",
            "lr",
            "epochs",
            "batch_size",
            "random_state",
        }
        assert required.issubset(mlp_params.keys())


# ---------------------------------------------------------------------------
# train.py
# ---------------------------------------------------------------------------


class TestTrainMlpModel:
    def test_returns_mlp_classifier(self, sample_X, sample_y):
        with patch("mlflow.log_metric"):
            model = train_mlp_model(sample_X, sample_y, {**mlp_params, "epochs": 2})
        assert isinstance(model, MLPClassifier)

    def test_model_in_eval_mode(self, trained_model):
        assert not trained_model.training

    def test_model_produces_correct_output_shape(self, trained_model, sample_X):
        x = torch.tensor(sample_X.values, dtype=torch.float32)
        with torch.no_grad():
            out = trained_model(x)
        assert out.shape == (len(sample_X),)


# ---------------------------------------------------------------------------
# inference.py
# ---------------------------------------------------------------------------


class TestInferencePredict:
    def test_returns_list(self, trained_model, fitted_scaler, sample_X):
        preds, probs = inference_predict(sample_X, trained_model, fitted_scaler)

        assert isinstance(preds, list)
        assert isinstance(probs, list)

    def test_returns_binary_values(self, trained_model, fitted_scaler, sample_X):
        preds, probs = inference_predict(sample_X, trained_model, fitted_scaler)
        assert all(v in (0, 1) for v in preds)

    def test_output_length_matches_input(self, trained_model, fitted_scaler, sample_X):
        preds, probs = inference_predict(sample_X, trained_model, fitted_scaler)
        assert len(preds) == len(sample_X)
        assert len(probs) == len(sample_X)


# ---------------------------------------------------------------------------
# evaluate_mlp.py
# ---------------------------------------------------------------------------


class TestComputeMetrics:
    def test_returns_required_keys(self):
        y_true = pd.Series([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        metrics = compute_metrics(y_true, y_pred)
        assert set(metrics.keys()) == {"auc", "precision", "recall", "f1"}

    def test_perfect_predictions(self):
        y_true = pd.Series([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        metrics = compute_metrics(y_true, y_pred)
        assert metrics["f1"] == 1.0
        assert metrics["recall"] == 1.0

    def test_metric_values_between_0_and_1(self):
        y_true = pd.Series([0, 1, 1, 0])
        y_pred = np.array([0, 1, 0, 1])
        metrics = compute_metrics(y_true, y_pred)
        for v in metrics.values():
            assert 0.0 <= v <= 1.0


class TestEvalPredict:
    def test_returns_ndarray(self, trained_model, sample_X):
        result = eval_predict(trained_model, sample_X)
        assert isinstance(result, np.ndarray)

    def test_returns_binary_values(self, trained_model, sample_X):
        result = eval_predict(trained_model, sample_X)
        assert set(result).issubset({0, 1})

    def test_output_length_matches_input(self, trained_model, sample_X):
        result = eval_predict(trained_model, sample_X)
        assert len(result) == len(sample_X)

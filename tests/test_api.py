import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.serving import app

# ------------------------
# Fixtures
# ------------------------


@pytest.fixture
def client(monkeypatch, tmp_path):
    """
    Create a test client with fully isolated environment.
    """

    monkeypatch.setattr(app, "CURRENT_BUFFER_PATH", tmp_path / "buffer.csv")
    monkeypatch.setattr(app, "REFERENCE_PATH", tmp_path / "reference.csv")
    monkeypatch.setattr(app, "SCALER_PATH", tmp_path / "scaler.pkl")

    monkeypatch.setattr(app, "transform_features", lambda df, _: df)

    def fake_load_artifacts():
        return "model", "scaler"

    def fake_predict(data, model, scaler):
        return [0], [0.2]

    monkeypatch.setattr(app, "load_artifacts", fake_load_artifacts)
    monkeypatch.setattr(app, "predict", fake_predict)
    monkeypatch.setattr(app, "load_index", lambda: None)
    monkeypatch.setattr(app, "initialize_tools", lambda *args: None)
    monkeypatch.setattr(app, "build_agent", lambda: None)

    app.recent_predictions.clear()

    with TestClient(app.app) as c:
        yield c


@pytest.fixture
def sample_payload():
    return {
        "borrower_income": 50000,
        "debt_to_income": 0.3,
        "num_of_accounts": 5,
        "derogatory_marks": 1,
    }


@pytest.fixture
def temp_paths(tmp_path, monkeypatch):
    """
    Redirect file paths to temporary directory.
    """
    ref = tmp_path / "reference.csv"
    cur = tmp_path / "current.csv"

    monkeypatch.setattr(app, "REFERENCE_PATH", ref)
    monkeypatch.setattr(app, "CURRENT_BUFFER_PATH", cur)

    return ref, cur


# ------------------------
# Health
# ------------------------


def test_health_endpoint(client):
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "ok"
    assert data["artifacts_loaded"] is True


# ------------------------
# Predict
# ------------------------


def test_predict_endpoint_success(client, sample_payload, temp_paths):
    response = client.post("/predict", json=sample_payload)

    assert response.status_code == 200

    data = response.json()
    assert data["prediction"] == 0
    assert data["label"] == "low risk"


def test_predict_appends_to_buffer(client, sample_payload, temp_paths):
    _, buffer_path = temp_paths

    client.post("/predict", json=sample_payload)

    assert buffer_path.exists()

    df = pd.read_csv(buffer_path)
    assert len(df) == 1
    assert "borrower_income" in df.columns


def test_predict_updates_recent_predictions(client, sample_payload, temp_paths):
    app.recent_predictions.clear()

    client.post("/predict", json=sample_payload)

    assert len(app.recent_predictions) == 1
    assert app.recent_predictions[0]["risk_label"] == "LOW"


def test_predict_limits_recent_predictions(client, sample_payload, temp_paths):
    app.recent_predictions.clear()

    for _ in range(105):
        client.post("/predict", json=sample_payload)

    assert len(app.recent_predictions) == 100


# ------------------------
# Drift Endpoint
# ------------------------


def test_drift_no_reference(client, temp_paths):
    response = client.get("/drift")

    assert response.status_code == 200
    assert "error" in response.json()


def test_drift_no_current_buffer(client, temp_paths):
    ref, _ = temp_paths

    # create reference only
    pd.DataFrame({"a": [1]}).to_csv(ref, index=False)

    response = client.get("/drift")

    assert "error" in response.json()


def test_drift_success(client, temp_paths, monkeypatch):
    ref, cur = temp_paths

    df = pd.DataFrame(
        {
            "borrower_income": [1, 2],
            "debt_to_income": [0.1, 0.2],
            "num_of_accounts": [1, 2],
            "derogatory_marks": [0, 1],
        }
    )

    df.to_csv(ref, index=False)
    df.to_csv(cur, index=False)

    def fake_drift(*args, **kwargs):
        return {"feature": {"psi": 0.0, "status": "OK"}, "_summary": {}}

    monkeypatch.setattr(app, "compute_drift_report", fake_drift)

    response = client.get("/drift")

    assert response.status_code == 200
    assert "_summary" in response.json()


def test_drift_with_evidently(client, temp_paths, monkeypatch):
    ref, cur = temp_paths

    df = pd.DataFrame(
        {
            "borrower_income": [1, 2],
            "debt_to_income": [0.1, 0.2],
            "num_of_accounts": [1, 2],
            "derogatory_marks": [0, 1],
        }
    )

    df.to_csv(ref, index=False)
    df.to_csv(cur, index=False)

    monkeypatch.setattr(app, "compute_drift_report", lambda **_: {"_summary": {}})
    monkeypatch.setattr(app, "run_evidently_drift", lambda **_: {"report": "ok"})

    response = client.get("/drift?use_evidently=true")

    data = response.json()

    assert "_evidently" in data
    assert data["_evidently"]["report"] == "ok"


# ------------------------
# Metrics Endpoint
# ------------------------


def test_metrics_endpoint(client):
    response = client.get("/metrics")

    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    assert len(response.content) > 0


# ------------------------
# Internal Buffer Function
# ------------------------


def test_append_to_buffer_creates_file(tmp_path, monkeypatch):
    buffer_path = tmp_path / "buffer.csv"

    monkeypatch.setattr(app, "CURRENT_BUFFER_PATH", buffer_path)

    def fake_transform(df, path):
        return df

    monkeypatch.setattr(app, "transform_features", fake_transform)

    row = {
        "borrower_income": 1,
        "debt_to_income": 0.1,
        "num_of_accounts": 1,
        "derogatory_marks": 1,
    }

    app._append_to_buffer(row)

    assert buffer_path.exists()


def test_append_to_buffer_appends(tmp_path, monkeypatch):
    buffer_path = tmp_path / "buffer.csv"

    monkeypatch.setattr(app, "CURRENT_BUFFER_PATH", buffer_path)

    def fake_transform(df, path):
        return df

    monkeypatch.setattr(app, "transform_features", fake_transform)

    app._append_to_buffer(
        {
            "borrower_income": 1,
            "debt_to_income": 0.1,
            "num_of_accounts": 1,
            "derogatory_marks": 1,
        }
    )
    app._append_to_buffer(
        {
            "borrower_income": 2,
            "debt_to_income": 0.2,
            "num_of_accounts": 2,
            "derogatory_marks": 2,
        }
    )

    df = pd.read_csv(buffer_path)

    assert len(df) == 2

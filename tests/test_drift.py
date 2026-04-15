import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.monitoring import drift

# ------------------------
# Fixtures
# ------------------------


@pytest.fixture
def reference_data():
    np.random.seed(42)
    return pd.DataFrame(
        {
            "feature_1": np.random.normal(0, 1, 1000),
            "feature_2": np.random.normal(5, 2, 1000),
        }
    )


@pytest.fixture
def current_data_stable(reference_data):
    # mesma distribuição → sem drift
    return reference_data.copy()


@pytest.fixture
def current_data_drifted():
    np.random.seed(42)
    return pd.DataFrame(
        {
            "feature_1": np.random.normal(3, 1, 1000),  # shift forte
            "feature_2": np.random.normal(10, 2, 1000),
        }
    )


# ------------------------
# compute_psi
# ------------------------


def test_compute_psi_zero_when_identical():
    data = np.random.normal(0, 1, 1000)

    psi = drift.compute_psi(data, data)

    assert psi == pytest.approx(0.0, abs=1e-6)


def test_compute_psi_detects_drift():
    ref = np.random.normal(0, 1, 1000)
    cur = np.random.normal(3, 1, 1000)

    psi = drift.compute_psi(ref, cur)

    assert psi > 0.1


def test_compute_psi_handles_small_arrays():
    ref = np.array([1, 2, 3])
    cur = np.array([1, 2, 3])

    psi = drift.compute_psi(ref, cur)

    assert psi >= 0


# ------------------------
# compute_drift_report
# ------------------------


def test_compute_drift_report_ok(reference_data, current_data_stable):
    result = drift.compute_drift_report(
        reference_data, current_data_stable, ["feature_1"]
    )

    assert "feature_1" in result
    assert result["feature_1"]["status"] == "OK"
    assert result["_summary"]["overall_status"] == "OK"


def test_compute_drift_report_warning(reference_data):
    current = reference_data.copy()
    current["feature_1"] += 0.5  # leve drift

    result = drift.compute_drift_report(reference_data, current, ["feature_1"])

    assert result["feature_1"]["status"] in {"WARNING", "RETRAIN"}


def test_compute_drift_report_retrain(reference_data, current_data_drifted):
    result = drift.compute_drift_report(
        reference_data, current_data_drifted, ["feature_1"]
    )

    assert result["feature_1"]["status"] == "RETRAIN"
    assert result["_summary"]["overall_status"] == "RETRAIN"


def test_compute_drift_report_missing_column(caplog, reference_data):
    caplog.set_level(logging.WARNING)

    result = drift.compute_drift_report(reference_data, reference_data, ["missing_col"])

    assert "missing_col" not in result
    assert "não encontrada" in caplog.text


def test_compute_drift_report_summary_fields(reference_data, current_data_drifted):
    result = drift.compute_drift_report(
        reference_data,
        current_data_drifted,
        ["feature_1", "feature_2"],
    )

    summary = result["_summary"]

    assert summary["n_features_analyzed"] == 2
    assert "drift_share" in summary
    assert "max_psi" in summary


def test_compute_drift_report_empty_features(reference_data):
    result = drift.compute_drift_report(
        reference_data,
        reference_data,
        [],
    )

    summary = result["_summary"]

    assert summary["drift_share"] == 0.0
    assert summary["max_psi"] == 0.0


# ------------------------
# MLflow integration
# ------------------------


def test_compute_drift_report_logs_to_mlflow(monkeypatch, reference_data):
    calls = {"metrics": []}

    class DummyRun:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    def fake_start_run(*args, **kwargs):
        return DummyRun()

    def fake_log_metric(key, value):
        calls["metrics"].append((key, value))

    monkeypatch.setattr(drift.mlflow, "start_run", fake_start_run)
    monkeypatch.setattr(drift.mlflow, "log_metric", fake_log_metric)

    drift.compute_drift_report(
        reference_data,
        reference_data,
        ["feature_1"],
        run_id="test-run",
    )

    keys = [k for k, _ in calls["metrics"]]

    assert "drift_share" in keys
    assert "max_psi" in keys
    assert any(k.startswith("psi_") for k in keys)


# ------------------------
# Evidently
# ------------------------


def test_run_evidently_drift_success(monkeypatch, tmp_path, reference_data):
    class FakeReport:
        def run(self, **kwargs):
            pass

        def save_html(self, path):
            Path(path).write_text("ok")

        def as_dict(self):
            return {"metrics": [{"result": {"share_of_drifted_columns": 0.1}}]}

    monkeypatch.setattr(drift, "Report", lambda *a, **k: FakeReport())

    output_file = tmp_path / "report.html"

    result = drift.run_evidently_drift(
        reference_data,
        reference_data,
        output_path=str(output_file),
    )

    assert result["drift_share"] == 0.1
    assert output_file.exists()


def test_run_evidently_drift_exception(monkeypatch, reference_data):
    def fail(*args, **kwargs):
        raise Exception("boom")

    monkeypatch.setattr(drift.Report, "run", fail)

    result = drift.run_evidently_drift(
        reference_data,
        reference_data,
    )

    assert result == {}


def test_run_evidently_drift_import_error(monkeypatch, reference_data):
    def raise_import_error(*args, **kwargs):
        raise ImportError()

    monkeypatch.setattr(drift.Report, "__init__", raise_import_error)

    result = drift.run_evidently_drift(
        reference_data,
        reference_data,
    )

    assert result == {}

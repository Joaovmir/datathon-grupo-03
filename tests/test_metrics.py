import logging

import pytest

from src.monitoring import metrics

# ------------------------
# Fixtures
# ------------------------


@pytest.fixture(autouse=True)
def reset_metrics():
    metrics.PREDICTIONS_TOTAL._metrics.clear()
    metrics.DRIFT_PSI._metrics.clear()
    metrics.DRIFT_STATUS._metrics.clear()
    metrics.SECURITY_EVENTS._metrics.clear()
    yield


# ------------------------
# Prediction Metrics
# ------------------------


def test_record_prediction_increments_counter():
    metrics.record_prediction("LOW", 0, 0.2)

    sample = metrics.PREDICTIONS_TOTAL.labels(risk_label="LOW", prediction="0")

    assert sample._value.get() == 1


def test_record_prediction_observes_probability():
    metrics.record_prediction("HIGH", 1, 0.75)

    # forma correta:
    samples = list(metrics.PREDICTION_PROBABILITY.collect())[0].samples

    assert any(s.value > 0 for s in samples)


# ------------------------
# Drift Metrics
# ------------------------


def test_record_drift_sets_values():
    metrics.record_drift("income", 0.3, "WARNING")

    psi_metric = metrics.DRIFT_PSI.labels(feature="income")
    status_metric = metrics.DRIFT_STATUS.labels(feature="income")

    assert psi_metric._value.get() == 0.3
    assert status_metric._value.get() == 1  # WARNING → 1


def test_record_drift_unknown_status_defaults_to_zero():
    metrics.record_drift("income", 0.1, "UNKNOWN")

    status_metric = metrics.DRIFT_STATUS.labels(feature="income")

    assert status_metric._value.get() == 0


# ------------------------
# Security Events
# ------------------------


def test_record_security_event_increments_counter(caplog):
    caplog.set_level(logging.WARNING)

    metrics.record_security_event("pii_detected")

    sample = metrics.SECURITY_EVENTS.labels(event_type="pii_detected")

    assert sample._value.get() == 1
    assert "Evento de segurança registrado" in caplog.text


# ------------------------
# Business Metrics
# ------------------------


def test_update_business_metrics_correct_values():
    data = [
        {"risk_label": "LOW", "probability": 0.1},
        {"risk_label": "HIGH", "probability": 0.9},
        {"risk_label": "LOW", "probability": 0.2},
    ]

    metrics.update_business_metrics(data)

    assert metrics.APPROVAL_RATE._value.get() == pytest.approx(2 / 3)
    assert metrics.HIGH_RISK_RATE._value.get() == pytest.approx(1 / 3)
    assert metrics.AVERAGE_DEFAULT_PROBABILITY._value.get() == pytest.approx(
        (0.1 + 0.9 + 0.2) / 3
    )


def test_update_business_metrics_empty_input():
    before = metrics.APPROVAL_RATE._value.get()

    metrics.update_business_metrics([])

    after = metrics.APPROVAL_RATE._value.get()

    assert before == after


def test_update_business_metrics_missing_probability():
    data = [
        {"risk_label": "LOW"},
        {"risk_label": "HIGH"},
    ]

    metrics.update_business_metrics(data)

    assert metrics.AVERAGE_DEFAULT_PROBABILITY._value.get() == 0


# ------------------------
# Latency Decorator
# ------------------------


def test_track_latency_success():
    @metrics.track_latency(endpoint="/test")
    def dummy():
        return "ok"

    result = dummy()

    assert result == "ok"

    counter = metrics.REQUEST_TOTAL.labels(
        method="call", endpoint="/test", status_code="200"
    )
    assert counter._value.get() == 1

    samples = list(metrics.REQUEST_DURATION.collect())[0].samples

    assert any(s.value > 0 for s in samples)


def test_track_latency_exception():
    @metrics.track_latency(endpoint="/fail")
    def failing():
        raise ValueError("error")

    with pytest.raises(ValueError):
        failing()

    counter = metrics.REQUEST_TOTAL.labels(
        method="call", endpoint="/fail", status_code="500"
    )

    assert counter._value.get() == 1


def test_track_latency_preserves_function_metadata():
    @metrics.track_latency(endpoint="/meta")
    def my_function():
        """docstring"""
        return 42

    assert my_function.__name__ == "my_function"
    assert my_function.__doc__ == "docstring"

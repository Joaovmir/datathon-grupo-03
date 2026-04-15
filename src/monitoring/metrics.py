import logging
import time
from functools import wraps
from typing import Callable

from prometheus_client import Counter, Gauge, Histogram

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Métricas operacionais
# ---------------------------------------------------------------------------

REQUEST_TOTAL = Counter(
    "credit_api_requests_total",
    "Total de requisições recebidas pela API",
    ["method", "endpoint", "status_code"],
)

REQUEST_DURATION = Histogram(
    "credit_api_request_duration_seconds",
    "Latência das requisições da API",
    ["endpoint"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0],
)

# ---------------------------------------------------------------------------
# Métricas do modelo
# ---------------------------------------------------------------------------

PREDICTIONS_TOTAL = Counter(
    "credit_model_predictions_total",
    "Total de predições realizadas",
    ["risk_label", "prediction"],
)

PREDICTION_PROBABILITY = Histogram(
    "credit_model_prediction_probability",
    "Distribuição das probabilidades de inadimplência",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

MODEL_VERSION = Gauge(
    "credit_model_version_info",
    "Informação sobre a versão do modelo em produção",
    ["version", "model_type"],
)

# ---------------------------------------------------------------------------
# Métricas de negócio
# ---------------------------------------------------------------------------

APPROVAL_RATE = Gauge(
    "credit_business_approval_rate",
    "Taxa de aprovação (LOW risk) nas últimas N predições",
)

HIGH_RISK_RATE = Gauge(
    "credit_business_high_risk_rate",
    "Taxa de casos HIGH risk nas últimas N predições",
)

AVERAGE_DEFAULT_PROBABILITY = Gauge(
    "credit_business_avg_default_probability",
    "Probabilidade média de inadimplência nas últimas N predições",
)

# ---------------------------------------------------------------------------
# Métricas de drift
# ---------------------------------------------------------------------------

DRIFT_PSI = Gauge(
    "credit_drift_psi",
    "PSI (Population Stability Index) por feature",
    ["feature"],
)

DRIFT_STATUS = Gauge(
    "credit_drift_status",
    "Status de drift: 0=OK, 1=WARNING, 2=RETRAIN",
    ["feature"],
)

# ---------------------------------------------------------------------------
# Métricas de segurança
# ---------------------------------------------------------------------------

SECURITY_EVENTS = Counter(
    "credit_security_events_total",
    "Eventos de segurança detectados",
    ["event_type"],  # prompt_injection, pii_detected, rate_limit
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_STATUS_MAP = {"OK": 0, "WARNING": 1, "RETRAIN": 2}


def record_prediction(
    risk_label: str,
    prediction: int,
    probability: float,
) -> None:
    """Registra métricas de uma predição.

    Args:
        risk_label: LOW, MEDIUM ou HIGH.
        prediction: 0 ou 1.
        probability: Probabilidade de inadimplência.
    """
    PREDICTIONS_TOTAL.labels(risk_label=risk_label, prediction=str(prediction)).inc()
    PREDICTION_PROBABILITY.observe(probability)


def record_drift(feature: str, psi: float, status: str) -> None:
    """Registra métricas de drift para uma feature.

    Args:
        feature: Nome da feature.
        psi: Valor PSI calculado.
        status: 'OK', 'WARNING' ou 'RETRAIN'.
    """
    DRIFT_PSI.labels(feature=feature).set(psi)
    DRIFT_STATUS.labels(feature=feature).set(_STATUS_MAP.get(status, 0))


def record_security_event(event_type: str) -> None:
    """Registra evento de segurança.

    Args:
        event_type: Tipo de evento (prompt_injection, pii_detected, etc.)
    """
    SECURITY_EVENTS.labels(event_type=event_type).inc()
    logger.warning("Evento de segurança registrado: %s", event_type)


def update_business_metrics(
    recent_predictions: list[dict],
) -> None:
    """Atualiza métricas de negócio com base nas predições recentes.

    Args:
        recent_predictions: Lista de dicionários com 'risk_label' e 'probability'.
    """
    if not recent_predictions:
        return

    n = len(recent_predictions)
    n_low = sum(1 for p in recent_predictions if p.get("risk_label") == "LOW")
    n_high = sum(1 for p in recent_predictions if p.get("risk_label") == "HIGH")
    avg_prob = sum(p.get("probability", 0) for p in recent_predictions) / n

    APPROVAL_RATE.set(n_low / n)
    HIGH_RISK_RATE.set(n_high / n)
    AVERAGE_DEFAULT_PROBABILITY.set(avg_prob)


def track_latency(endpoint: str) -> Callable:
    """Decorator para medir latência de funções.

    Args:
        endpoint: Nome do endpoint para label da métrica.

    Returns:
        Decorator que registra latência.
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                REQUEST_TOTAL.labels(
                    method="call", endpoint=endpoint, status_code="200"
                ).inc()
                return result
            except Exception:
                REQUEST_TOTAL.labels(
                    method="call", endpoint=endpoint, status_code="500"
                ).inc()
                raise
            finally:
                REQUEST_DURATION.labels(endpoint=endpoint).observe(
                    time.perf_counter() - start
                )

        return wrapper

    return decorator

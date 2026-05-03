import random
import time
from concurrent.futures import ThreadPoolExecutor

import requests

# URL da sua API rodando no Docker
URL_PREDICT = "http://localhost:8000/predict"


def gerar_payload():
    """Gera perfis aleatórios (30% ruins, 70% bons) para movimentar os gráficos"""
    is_bad_payer = random.random() < 0.30

    if is_bad_payer:
        return {
            "borrower_income": round(random.uniform(20000, 45000), 2),
            "debt_to_income": round(random.uniform(0.5, 0.85), 2),
            "num_of_accounts": random.randint(8, 15),
            "derogatory_marks": random.randint(1, 4),
        }
    else:
        return {
            "borrower_income": round(random.uniform(50000, 120000), 2),
            "debt_to_income": round(random.uniform(0.1, 0.4), 2),
            "num_of_accounts": random.randint(2, 6),
            "derogatory_marks": random.randint(0, 1),
        }


def fazer_requisicao(i):
    payload = gerar_payload()
    try:
        resp = requests.post(URL_PREDICT, json=payload, timeout=10)
        if resp.status_code == 200:
            print(f"[{i}] ✅ Sucesso | Risco: {resp.json().get('label')}")
        else:
            print(f"[{i}] ⚠️ Erro {resp.status_code} | Detalhe: {resp.text}")
    except Exception as e:
        print(f"[{i}] ❌ Falha de conexão: {str(e)}")


def run_stress_test(total_requests=300, concurrent_workers=10):
    print(f"🚀 Iniciando teste de estresse com {total_requests} requisições...")
    with ThreadPoolExecutor(max_workers=concurrent_workers) as executor:
        executor.map(fazer_requisicao, range(total_requests))
    print(
        "\n🏁 Teste finalizado! Vá olhar seu Grafana (filtre para os Últimos 5 minutos)."
    )


if __name__ == "__main__":
    # Roda 300 requisições, 10 ao mesmo tempo
    run_stress_test(total_requests=300, concurrent_workers=1)
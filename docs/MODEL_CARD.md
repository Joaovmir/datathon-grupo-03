# Model Card — Credit Risk MLP

## Overview
Modelo de classificação binária para risco de crédito.

## Features
- borrower_income
- debt_to_income
- num_of_accounts
- derogatory_marks

## Model
- Tipo: MLP (PyTorch)
- Loss: BCEWithLogitsLoss
- Balanceamento: pos_weight

## Metrics
(Logadas via MLflow)

## Monitoring
- Prometheus:
  - Latência
  - Requests
  - Distribuição de predição

- Drift:
  - Evidently
  - Threshold: 30% colunas com drift

## Limitations
- Dados sintéticos podem não refletir produção
- Possível viés socioeconômico

## Retraining
- Trigger: drift_detected = True

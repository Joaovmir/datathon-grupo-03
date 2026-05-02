"""
conftest.py — Fixtures compartilhadas entre toda a suíte de testes.

Organização por domínio:
  - Dados tabulares      : sample_X, sample_y, sample_data, sample_payload
  - Modelos e artefatos  : trained_model, fitted_scaler, scaled_X
  - Mocks do agente      : dummy_model, dummy_scaler, dummy_vectorstore,
                           initialized_tools
  - Dados de monitoramento: reference_data, current_data_stable, current_data_drifted

Convenções:
  - Fixtures sem escopo explícito usam scope="function" (padrão pytest).
  - Fixtures computacionalmente caras (trained_model) usam scope="session"
    para serem criadas apenas uma vez por execução de testes.
  - Todos os dados sintéticos usam seeds fixas para garantir reproducibilidade.
"""

from __future__ import annotations

from typing import Generator
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler


@pytest.fixture()
def sample_X() -> pd.DataFrame:
    """
    DataFrame mínimo com as 4 features do modelo de crédito.

    Quatro linhas são suficientes para exercitar batching, scaling e
    inferência sem custo computacional relevante. Valores escolhidos para
    cobrir diferentes faixas de renda e DTI.

    Returns:
        pd.DataFrame: Features cruas (não escaladas) do solicitante.
    """
    return pd.DataFrame(
        {
            "borrower_income": [50_000.0, 60_000.0, 40_000.0, 70_000.0],
            "debt_to_income": [0.3, 0.5, 0.2, 0.4],
            "num_of_accounts": [3, 5, 2, 4],
            "derogatory_marks": [0, 1, 0, 1],
        }
    )


@pytest.fixture()
def sample_y() -> pd.Series:
    """
    Série de labels binários balanceados (2 positivos, 2 negativos).

    O balanceamento garante que pos_weight seja calculável em
    train_mlp_model sem divisão por zero.

    Returns:
        pd.Series: Target binário (0 = baixo risco, 1 = alto risco).
    """
    return pd.Series([0, 1, 0, 1])


@pytest.fixture()
def sample_data() -> pd.DataFrame:
    """
    Dataset sintético maior com todas as colunas do pipeline raw de features.

    Usado em test_features.py para exercitar o pipeline completo de
    feature engineering (select, split, scale, save).

    Returns:
        pd.DataFrame: 50 linhas com features e coluna loan_status.
    """
    np.random.seed(42)
    return pd.DataFrame(
        {
            "loan_size": np.random.randint(5_000, 20_000, 50),
            "interest_rate": np.random.randint(5, 12, 50),
            "borrower_income": np.random.randint(30_000, 100_000, 50),
            "debt_to_income": np.random.rand(50),
            "num_of_accounts": np.random.randint(1, 15, 50),
            "derogatory_marks": np.random.randint(0, 5, 50),
            "total_debt": np.random.randint(5_000, 60_000, 50),
            "loan_status": np.random.choice([0, 1], 50, p=[0.8, 0.2]),
        }
    )


@pytest.fixture()
def sample_payload() -> dict:
    """
    Payload JSON válido para o endpoint POST /predict da API.

    Representa um solicitante com perfil financeiro moderado, sem
    marcas negativas e renda compatível com a faixa "médio".

    Returns:
        dict: Dicionário com os 4 campos obrigatórios do schema de entrada.
    """
    return {
        "borrower_income": 50_000,
        "debt_to_income": 0.3,
        "num_of_accounts": 5,
        "derogatory_marks": 1,
    }


@pytest.fixture(scope="session")
def trained_model(
    sample_X: pd.DataFrame,  # type: ignore[override]
    sample_y: pd.Series,  # type: ignore[override]
):
    """
    MLPClassifier treinado por 2 épocas com mock de mlflow.log_metric.

    Escopo session: o treino ocorre uma única vez por execução,
    economizando tempo em suítes com muitos testes que precisam de um
    modelo com pesos ajustados.

    Note:
        Como o escopo é session, esta fixture depende de versões session-scoped
        de sample_X e sample_y — pytest resolve automaticamente pela hierarquia
        de escopo.

    Returns:
        MLPClassifier: Modelo em eval mode com pesos parcialmente ajustados.
    """
    from src.models.baseline import mlp_params
    from src.models.train import train_mlp_model

    with patch("mlflow.log_metric"):
        return train_mlp_model(sample_X, sample_y, {**mlp_params, "epochs": 2})


@pytest.fixture(scope="session")
def fitted_scaler(
    sample_X: pd.DataFrame,  # type: ignore[override]
) -> StandardScaler:
    """
    StandardScaler ajustado sobre sample_X.

    Escopo session: o fitting ocorre uma única vez e o objeto é
    compartilhado entre todos os testes que precisam de scaling
    consistente sem acesso ao artefato em disco.

    Returns:
        StandardScaler: Scaler com mean_ e scale_ calculados sobre sample_X.
    """
    scaler = StandardScaler()
    scaler.fit(sample_X)
    return scaler


@pytest.fixture(scope="session")
def scaled_X(
    sample_X: pd.DataFrame,  # type: ignore[override]
    fitted_scaler: StandardScaler,
) -> pd.DataFrame:
    """
    DataFrame com features de sample_X normalizadas pelo fitted_scaler.

    Evita repetir a chamada a scaler.transform em vários testes de
    evaluate_mlp — especialmente útil em TestEvalPredict e TestLoadModel.

    Returns:
        pd.DataFrame: Features escaladas com as mesmas colunas de sample_X.
    """
    scaled = fitted_scaler.transform(sample_X)
    return pd.DataFrame(scaled, columns=sample_X.columns)


@pytest.fixture()
def dummy_model() -> MagicMock:
    """
    Mock do MLPClassifier para testes do agente.

    Configurado com eval() retornando a si mesmo para permitir encadeamento
    fluente (model.eval().predict()), compatível com o pipeline de inferência.

    Returns:
        MagicMock: Substituto do MLPClassifier sem dependência de pesos reais.
    """
    model = MagicMock()
    model.eval.return_value = model
    return model


@pytest.fixture()
def dummy_scaler() -> MagicMock:
    """
    Mock do StandardScaler para testes do agente.

    transform devolve o array de entrada sem modificação, simulando
    um scaler identidade — suficiente para testes que verificam o fluxo
    de dados sem precisar de normalização real.

    Returns:
        MagicMock: Substituto do StandardScaler com transform identidade.
    """
    scaler = MagicMock()
    scaler.transform.side_effect = lambda x: np.array(x)
    return scaler


@pytest.fixture()
def dummy_vectorstore() -> MagicMock:
    """
    Mock do Chroma vectorstore para testes do agente.

    similarity_search retorna um documento fictício com conteúdo fixo,
    permitindo testar o fluxo RAG sem acesso ao ChromaDB em disco.

    Returns:
        MagicMock: Substituto do Chroma com resultado de busca pré-definido.
    """
    doc = MagicMock()
    doc.page_content = "Política de crédito: comunicar a decisão com empatia."
    vectorstore = MagicMock()
    vectorstore.similarity_search.return_value = [doc]
    return vectorstore


@pytest.fixture()
def initialized_tools(
    dummy_model: MagicMock,
    dummy_scaler: MagicMock,
    dummy_vectorstore: MagicMock,
) -> Generator[None, None, None]:
    """
    Inicializa o módulo tools com mocks e garante limpeza após cada teste.

    Injeta as dependências globais (_model, _scaler, _vectorstore) antes
    do teste e as redefine como None no teardown, evitando vazamento de
    estado entre testes que dependem de initialize_tools.

    Yields:
        None: Controle retorna ao teste com o módulo tools inicializado.
    """
    from src.agent import tools

    tools.initialize_tools(dummy_model, dummy_scaler, dummy_vectorstore)
    yield
    tools._model = None
    tools._scaler = None
    tools._vectorstore = None


@pytest.fixture()
def reference_data() -> pd.DataFrame:
    """
    Dataset de referência para testes de detecção de drift (PSI/Evidently).

    1000 amostras com seed fixa para garantir que os limiares de PSI
    se comportem previsivelmente nos testes de WARNING e RETRAIN.

    Returns:
        pd.DataFrame: Distribuição normal estável com 2 features.
    """
    np.random.seed(42)
    return pd.DataFrame(
        {
            "feature_1": np.random.normal(0, 1, 1_000),
            "feature_2": np.random.normal(5, 2, 1_000),
        }
    )


@pytest.fixture()
def current_data_stable(reference_data: pd.DataFrame) -> pd.DataFrame:
    """
    Dataset atual idêntico ao de referência — cenário sem drift.

    Usado para validar que compute_drift_report retorna status "OK"
    quando as distribuições são idênticas (PSI ≈ 0).

    Returns:
        pd.DataFrame: Cópia exata de reference_data.
    """
    return reference_data.copy()


@pytest.fixture()
def current_data_drifted() -> pd.DataFrame:
    """
    Dataset atual com drift severo em relação ao de referência.

    feature_1 e feature_2 têm médias deslocadas em 3 e 5 desvios-padrão
    respectivamente, garantindo PSI > 0.25 (limiar RETRAIN).

    Returns:
        pd.DataFrame: Distribuição deslocada que dispara alerta RETRAIN.
    """
    np.random.seed(42)
    return pd.DataFrame(
        {
            "feature_1": np.random.normal(3, 1, 1_000),
            "feature_2": np.random.normal(10, 2, 1_000),
        }
    )

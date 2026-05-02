from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.preprocessing import StandardScaler

from src.models.baseline import MLPClassifier, build_mlp, mlp_params
from src.models.evaluate_mlp import compute_metrics, load_model
from src.models.evaluate_mlp import predict as eval_predict
from src.models.evaluate_mlp import save_classification_report
from src.models.inference import predict as inference_predict
from src.models.train import save_reference_data, train_mlp_model

# ---------------------------------------------------------------------------
# Fixtures compartilhadas
# ---------------------------------------------------------------------------


@pytest.fixture()
def trained_model(sample_X: pd.DataFrame, sample_y: pd.Series) -> MLPClassifier:
    """
    MLPClassifier treinado por 2 épocas com mock de mlflow.log_metric.

    Usado em testes que precisam de um modelo com pesos ajustados,
    sem dependência de dados reais ou experimento MLflow ativo.
    """
    with patch("mlflow.log_metric"):
        return train_mlp_model(sample_X, sample_y, {**mlp_params, "epochs": 2})


@pytest.fixture()
def fitted_scaler(sample_X: pd.DataFrame) -> StandardScaler:
    """
    StandardScaler ajustado sobre sample_X.

    Permite testar pipelines de inferência com scaling consistente
    sem acesso ao artefato em disco.
    """
    scaler = StandardScaler()
    scaler.fit(sample_X)
    return scaler


@pytest.fixture()
def scaled_X(sample_X: pd.DataFrame, fitted_scaler: StandardScaler) -> pd.DataFrame:
    """
    DataFrame com features normalizadas prontas para entrada no modelo.

    Evita repetir a chamada a scaler.transform em vários testes de evaluate_mlp.
    """
    scaled = fitted_scaler.transform(sample_X)
    return pd.DataFrame(scaled, columns=sample_X.columns)


# ---------------------------------------------------------------------------
# baseline.py — MLPClassifier / build_mlp / mlp_params
# ---------------------------------------------------------------------------


class TestMLPClassifier:
    """Testes da arquitetura do MLP e da função build_mlp."""

    def test_build_mlp__retorna_instancia_mlp_classifier(self) -> None:
        """build_mlp deve retornar um MLPClassifier sem argumentos adicionais."""
        model = build_mlp()
        assert isinstance(model, MLPClassifier)

    def test_forward__batch_de_8__shape_correto(self) -> None:
        """A saída do forward para um batch de 8 amostras deve ter shape (8,)."""
        model = build_mlp()
        x = torch.randn(8, mlp_params["input_dim"])
        out = model(x)
        assert out.shape == (8,)

    def test_forward__amostra_unica__shape_correto(self) -> None:
        """A saída para uma única amostra deve ter shape (1,), não escalar."""
        model = build_mlp()
        x = torch.randn(1, mlp_params["input_dim"])
        out = model(x)
        assert out.shape == (1,)

    def test_mlp_params__contem_todas_as_chaves_obrigatorias(self) -> None:
        """mlp_params deve expor todas as chaves esperadas pelo pipeline de treino."""
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

    def test_forward__input_dim_errado__lanca_excecao(self) -> None:
        """
        Passar tensores com dimensão diferente de input_dim deve lançar
        RuntimeError antes de qualquer cálculo de gradiente.
        """
        model = build_mlp()
        wrong_input = torch.randn(4, mlp_params["input_dim"] + 5)
        with pytest.raises(RuntimeError):
            model(wrong_input)

    def test_forward__modo_treino__ativa_dropout(self) -> None:
        """
        Em modo de treino, passagens repetidas pelo mesmo input devem
        produzir logits diferentes devido ao Dropout estocástico.

        Falha esperada se dropout=0 — portanto validamos que mlp_params
        usa dropout > 0.
        """
        assert mlp_params["dropout"] > 0, "Este teste requer dropout > 0"

        model = build_mlp()
        model.train()
        x = torch.ones(16, mlp_params["input_dim"])

        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)

        assert not torch.equal(
            out1, out2
        ), "Em modo train com dropout > 0, as saídas devem diferir entre passagens"

    def test_forward__modo_eval__desativa_dropout(self) -> None:
        """
        Em modo de avaliação, Dropout é desativado e passagens repetidas
        devem produzir logits idênticos para o mesmo input.
        """
        model = build_mlp()
        model.eval()
        x = torch.ones(8, mlp_params["input_dim"])

        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)

        assert torch.equal(out1, out2)

    def test_mlp_params__input_dim_positivo(self) -> None:
        """input_dim deve ser inteiro positivo para a primeira camada
        linear ser válida."""
        assert isinstance(mlp_params["input_dim"], int)
        assert mlp_params["input_dim"] > 0

    def test_mlp_params__hidden_dims_e_lista_nao_vazia(self) -> None:
        """hidden_dims deve ser uma lista com pelo menos uma camada oculta."""
        assert isinstance(mlp_params["hidden_dims"], list)
        assert len(mlp_params["hidden_dims"]) >= 1

    def test_mlp_params__dropout_entre_0_e_1(self) -> None:
        """Dropout fora do intervalo [0, 1) causaria erro no PyTorch."""
        assert 0.0 <= mlp_params["dropout"] < 1.0


# ---------------------------------------------------------------------------
# train.py — train_mlp_model / save_reference_data
# ---------------------------------------------------------------------------


class TestTrainMlpModel:
    """Testes do loop de treinamento do MLP com PyTorch."""

    def test_retorna_mlp_classifier(
        self, sample_X: pd.DataFrame, sample_y: pd.Series
    ) -> None:
        """train_mlp_model deve retornar uma instância de MLPClassifier."""
        with patch("mlflow.log_metric"):
            model = train_mlp_model(sample_X, sample_y, {**mlp_params, "epochs": 2})
        assert isinstance(model, MLPClassifier)

    def test_modelo_em_modo_eval_apos_treino(
        self, trained_model: MLPClassifier
    ) -> None:
        """O modelo retornado deve estar em eval mode (model.training == False)."""
        assert not trained_model.training

    def test_output_shape_correto(
        self, trained_model: MLPClassifier, sample_X: pd.DataFrame
    ) -> None:
        """A saída do modelo treinado deve ter shape (n_amostras,)."""
        x = torch.tensor(sample_X.values, dtype=torch.float32)
        with torch.no_grad():
            out = trained_model(x)
        assert out.shape == (len(sample_X),)

    def test_log_metric_chamado_por_epoca(
        self, sample_X: pd.DataFrame, sample_y: pd.Series
    ) -> None:
        """
        mlflow.log_metric deve ser chamado exatamente uma vez por época,
        com o nome 'train_loss' e o step correspondente.
        """
        n_epochs = 3
        with patch("mlflow.log_metric") as mock_log:
            train_mlp_model(sample_X, sample_y, {**mlp_params, "epochs": n_epochs})

        assert mock_log.call_count == n_epochs
        calls = mock_log.call_args_list
        for i, call in enumerate(calls):
            assert call.args[0] == "train_loss"
            assert call.kwargs.get("step") == i + 1 or call.args[2] == i + 1

    def test_seed_garante_reproducibilidade(
        self, sample_X: pd.DataFrame, sample_y: pd.Series
    ) -> None:
        """
        Dois treinos com o mesmo random_state devem produzir pesos idênticos,
        garantindo reproducibilidade do pipeline.
        """
        params = {**mlp_params, "epochs": 2}

        with patch("mlflow.log_metric"):
            model_a = train_mlp_model(sample_X, sample_y, params)
            model_b = train_mlp_model(sample_X, sample_y, params)

        for (name_a, p_a), (_, p_b) in zip(
            model_a.named_parameters(), model_b.named_parameters()
        ):
            assert torch.equal(p_a, p_b), f"Pesos divergem em '{name_a}'"

    def test_pesos_mudam_apos_treino(
        self, sample_X: pd.DataFrame, sample_y: pd.Series
    ) -> None:
        """
        Os pesos do modelo devem ser diferentes dos valores inicializados,
        confirmando que o loop de otimização de fato atualizou os parâmetros.
        """
        model_before = build_mlp()
        initial_weights = {
            name: param.clone() for name, param in model_before.named_parameters()
        }

        with patch("mlflow.log_metric"):
            model_after = train_mlp_model(
                sample_X, sample_y, {**mlp_params, "epochs": 5}
            )

        any_changed = any(
            not torch.equal(initial_weights[name], param)
            for name, param in model_after.named_parameters()
            if name in initial_weights
        )
        assert any_changed, "Nenhum peso foi atualizado durante o treino"

    def test_pos_weight_calculado_sem_excecao(self, sample_X: pd.DataFrame) -> None:
        """
        Quando todas as amostras são da mesma classe (apenas 0s ou apenas 1s),
        o cálculo de pos_weight deve ser robusto — não necessariamente correto,
        mas não deve lançar exceção inesperada ou divisão por zero silenciosa.

        Nota: o comportamento real com divisão por zero em tensores PyTorch
        resulta em inf, não em exceção — verificamos apenas que o treino completa.
        """
        y_only_zeros = pd.Series([0, 0, 0, 0])
        with patch("mlflow.log_metric"):
            model = train_mlp_model(sample_X, y_only_zeros, {**mlp_params, "epochs": 1})
        assert isinstance(model, MLPClassifier)


class TestSaveReferenceData:
    """Testes da persistência dos dados de referência para detecção de drift."""

    def test_salva_csv_no_caminho_correto(self, sample_X: pd.DataFrame) -> None:
        """
        save_reference_data deve criar um arquivo CSV em REFERENCE_PATH
        com o mesmo número de linhas do DataFrame de entrada.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_path = Path(tmpdir) / "reference" / "train.csv"
            with patch("src.models.train.REFERENCE_PATH", fake_path):
                save_reference_data(sample_X)

            assert fake_path.exists()
            saved = pd.read_csv(fake_path)
            assert len(saved) == len(sample_X)

    def test_cria_diretorio_pai_se_nao_existir(self, sample_X: pd.DataFrame) -> None:
        """
        O diretório pai de REFERENCE_PATH deve ser criado automaticamente
        se ainda não existir, sem necessidade de mkdir manual externo.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_path = Path(tmpdir) / "a" / "b" / "c" / "reference.csv"
            with patch("src.models.train.REFERENCE_PATH", nested_path):
                save_reference_data(sample_X)

            assert nested_path.exists()

    def test_colunas_preservadas_no_csv(self, sample_X: pd.DataFrame) -> None:
        """As colunas do CSV salvo devem ser idênticas às do DataFrame original."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_path = Path(tmpdir) / "ref.csv"
            with patch("src.models.train.REFERENCE_PATH", fake_path):
                save_reference_data(sample_X)

            saved = pd.read_csv(fake_path)
            assert list(saved.columns) == list(sample_X.columns)


# ---------------------------------------------------------------------------
# inference.py — predict / load_artifacts / shap_explain
# ---------------------------------------------------------------------------


class TestInferencePredict:
    """Testes do pipeline de inferência em tempo real (inference.predict)."""

    def test_retorna_listas(
        self,
        trained_model: MLPClassifier,
        fitted_scaler: StandardScaler,
        sample_X: pd.DataFrame,
    ) -> None:
        """predict deve retornar um par (list, list) — predictions e probabilities."""
        preds, probs = inference_predict(sample_X, trained_model, fitted_scaler)
        assert isinstance(preds, list)
        assert isinstance(probs, list)

    def test_predicoes_sao_binarias(
        self,
        trained_model: MLPClassifier,
        fitted_scaler: StandardScaler,
        sample_X: pd.DataFrame,
    ) -> None:
        """Todos os valores em predictions devem ser 0 ou 1."""
        preds, _ = inference_predict(sample_X, trained_model, fitted_scaler)
        assert all(v in (0, 1) for v in preds)

    def test_comprimento_igual_ao_input(
        self,
        trained_model: MLPClassifier,
        fitted_scaler: StandardScaler,
        sample_X: pd.DataFrame,
    ) -> None:
        """O número de predições e probabilidades deve ser igual ao número de linhas."""
        preds, probs = inference_predict(sample_X, trained_model, fitted_scaler)
        assert len(preds) == len(sample_X)
        assert len(probs) == len(sample_X)

    def test_probabilidades_entre_0_e_1(
        self,
        trained_model: MLPClassifier,
        fitted_scaler: StandardScaler,
        sample_X: pd.DataFrame,
    ) -> None:
        """
        Todas as probabilidades devem estar no intervalo [0, 1],
        pois são resultado de torch.sigmoid.
        """
        _, probs = inference_predict(sample_X, trained_model, fitted_scaler)
        assert all(0.0 <= p <= 1.0 for p in probs)

    def test_amostra_unica(
        self,
        trained_model: MLPClassifier,
        fitted_scaler: StandardScaler,
        sample_X: pd.DataFrame,
    ) -> None:
        """predict deve funcionar com um DataFrame de linha única sem erros de shape."""
        single_row = sample_X.iloc[[0]]
        preds, probs = inference_predict(single_row, trained_model, fitted_scaler)
        assert len(preds) == 1
        assert len(probs) == 1

    def test_probabilidade_consistente_com_predicao(
        self,
        trained_model: MLPClassifier,
        fitted_scaler: StandardScaler,
        sample_X: pd.DataFrame,
    ) -> None:
        """
        Uma probabilidade ≥ 0.5 deve corresponder à predição 1,
        e < 0.5 deve corresponder à predição 0.
        """
        preds, probs = inference_predict(sample_X, trained_model, fitted_scaler)
        for pred, prob in zip(preds, probs):
            expected = 1 if prob >= 0.5 else 0
            assert pred == expected, f"Inconsistência: prob={prob:.4f} mas pred={pred}"


class TestLoadArtifacts:
    """Testes do carregamento de modelo e scaler a partir do disco."""

    def test_retorna_model_e_scaler(
        self, trained_model: MLPClassifier, fitted_scaler: StandardScaler
    ) -> None:
        """
        load_artifacts deve retornar um par (MLPClassifier, scaler)
        quando os artefatos existem em disco.
        """
        with (
            patch("src.models.inference.joblib.load", return_value=fitted_scaler),
            patch(
                "src.models.inference.torch.load",
                return_value=trained_model.state_dict(),
            ),
            patch.object(trained_model, "load_state_dict"),
            patch("src.models.inference.MLPClassifier", return_value=trained_model),
        ):
            from src.models.inference import load_artifacts

            model, scaler = load_artifacts()

        assert isinstance(scaler, StandardScaler)

    def test_modelo_em_eval_mode_apos_load(
        self, trained_model: MLPClassifier, fitted_scaler: StandardScaler
    ) -> None:
        """
        O modelo retornado por load_artifacts deve estar em eval mode,
        garantindo que Dropout e BatchNorm se comportem corretamente em produção.
        """
        with (
            patch("src.models.inference.joblib.load", return_value=fitted_scaler),
            patch(
                "src.models.inference.torch.load",
                return_value=trained_model.state_dict(),
            ),
            patch("src.models.inference.MLPClassifier", return_value=trained_model),
        ):
            from src.models.inference import load_artifacts

            model, _ = load_artifacts()

        assert not model.training


class TestShapExplain:
    """Testes da explicação SHAP por KernelExplainer."""

    def test_retorna_lista_com_4_elementos(
        self,
        trained_model: MLPClassifier,
        fitted_scaler: StandardScaler,
        sample_X: pd.DataFrame,
    ) -> None:
        """
        shap_explain deve retornar exatamente 4 dicionários — um por feature —
        independentemente dos valores de entrada.
        """
        from src.models.inference import shap_explain

        result = shap_explain(sample_X.iloc[[0]], trained_model, fitted_scaler)
        assert isinstance(result, list)
        assert len(result) == 4

    def test_chaves_obrigatorias_em_cada_item(
        self,
        trained_model: MLPClassifier,
        fitted_scaler: StandardScaler,
        sample_X: pd.DataFrame,
    ) -> None:
        """Cada dicionário deve conter exatamente as chaves 'feature',
        'shap_value' e 'direction'."""
        from src.models.inference import shap_explain

        result = shap_explain(sample_X.iloc[[0]], trained_model, fitted_scaler)
        for item in result:
            assert set(item.keys()) == {"feature", "shap_value", "direction"}

    def test_ordenado_por_valor_absoluto_decrescente(
        self,
        trained_model: MLPClassifier,
        fitted_scaler: StandardScaler,
        sample_X: pd.DataFrame,
    ) -> None:
        """
        Os itens devem estar ordenados do maior para o menor valor absoluto de SHAP,
        facilitando a identificação dos fatores de maior impacto.
        """
        from src.models.inference import shap_explain

        result = shap_explain(sample_X.iloc[[0]], trained_model, fitted_scaler)
        abs_values = [abs(item["shap_value"]) for item in result]
        assert abs_values == sorted(abs_values, reverse=True)

    def test_direction_consistente_com_shap_value(
        self,
        trained_model: MLPClassifier,
        fitted_scaler: StandardScaler,
        sample_X: pd.DataFrame,
    ) -> None:
        """
        'direction' deve ser 'aumenta risco' para shap_value > 0
        e 'reduz risco' para shap_value <= 0.
        """
        from src.models.inference import shap_explain

        result = shap_explain(sample_X.iloc[[0]], trained_model, fitted_scaler)
        for item in result:
            if item["shap_value"] > 0:
                assert item["direction"] == "aumenta risco"
            else:
                assert item["direction"] == "reduz risco"

    def test_features_correspondem_as_colunas_esperadas(
        self,
        trained_model: MLPClassifier,
        fitted_scaler: StandardScaler,
        sample_X: pd.DataFrame,
    ) -> None:
        """
        Os nomes de feature no resultado devem ser exatamente os definidos
        em FEATURE_COLS de inference.py, garantindo rastreabilidade.
        """
        from src.models.inference import FEATURE_COLS, shap_explain

        result = shap_explain(sample_X.iloc[[0]], trained_model, fitted_scaler)
        result_features = {item["feature"] for item in result}
        assert result_features == set(FEATURE_COLS)

    def test_shap_value_e_float_arredondado(
        self,
        trained_model: MLPClassifier,
        fitted_scaler: StandardScaler,
        sample_X: pd.DataFrame,
    ) -> None:
        """
        shap_value deve ser um float com no máximo 4 casas decimais,
        conforme round(..., 4) aplicado em inference.py.
        """
        from src.models.inference import shap_explain

        result = shap_explain(sample_X.iloc[[0]], trained_model, fitted_scaler)
        for item in result:
            assert isinstance(item["shap_value"], float)
            # Verifica que não há mais de 4 casas decimais
            rounded = round(item["shap_value"], 4)
            assert item["shap_value"] == rounded


# ---------------------------------------------------------------------------
# evaluate_mlp.py — predict / compute_metrics / load_model / save_classification_report
# ---------------------------------------------------------------------------


class TestEvalPredict:
    """Testes da função predict de evaluate_mlp (inferência sobre dados escalados)."""

    def test_retorna_ndarray(
        self, trained_model: MLPClassifier, scaled_X: pd.DataFrame
    ) -> None:
        """eval_predict deve retornar um np.ndarray."""
        result = eval_predict(trained_model, scaled_X)
        assert isinstance(result, np.ndarray)

    def test_valores_binarios(
        self, trained_model: MLPClassifier, scaled_X: pd.DataFrame
    ) -> None:
        """Todos os elementos do array devem ser 0 ou 1."""
        result = eval_predict(trained_model, scaled_X)
        assert set(result.flatten()).issubset({0, 1})

    def test_comprimento_igual_ao_input(
        self, trained_model: MLPClassifier, scaled_X: pd.DataFrame
    ) -> None:
        """O comprimento da saída deve ser igual ao número de linhas do input."""
        result = eval_predict(trained_model, scaled_X)
        assert len(result) == len(scaled_X)

    def test_amostra_unica(
        self, trained_model: MLPClassifier, scaled_X: pd.DataFrame
    ) -> None:
        """eval_predict deve funcionar com uma única linha sem erros de shape."""
        result = eval_predict(trained_model, scaled_X.iloc[[0]])
        assert len(result) == 1


class TestComputeMetrics:
    """Testes do cálculo de métricas de classificação."""

    def test_retorna_chaves_obrigatorias(self) -> None:
        """compute_metrics deve retornar dict com as chaves auc,
        precision, recall e f1."""
        y_true = pd.Series([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        metrics = compute_metrics(y_true, y_pred)
        assert set(metrics.keys()) == {"auc", "precision", "recall", "f1"}

    def test_predicoes_perfeitas_dao_metrica_1(self) -> None:
        """Com predições perfeitas, f1 e recall devem ser exatamente 1.0."""
        y_true = pd.Series([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        metrics = compute_metrics(y_true, y_pred)
        assert metrics["f1"] == 1.0
        assert metrics["recall"] == 1.0

    def test_valores_entre_0_e_1(self) -> None:
        """Todas as métricas devem estar no intervalo [0, 1]."""
        y_true = pd.Series([0, 1, 1, 0])
        y_pred = np.array([0, 1, 0, 1])
        metrics = compute_metrics(y_true, y_pred)
        for v in metrics.values():
            assert 0.0 <= v <= 1.0

    def test_predicoes_todas_erradas__recall_zero(self) -> None:
        """
        Quando nenhum positivo é detectado, recall deve ser 0.0.
        zero_division=0 garante que não há exceção por divisão por zero.
        """
        y_true = pd.Series([0, 1, 0, 1])
        y_pred = np.array([1, 0, 1, 0])  # invertido
        metrics = compute_metrics(y_true, y_pred)
        assert metrics["recall"] == 0.0

    def test_auc_chance__predicoes_aleatorias_simetricas(self) -> None:
        """
        Para predições exatamente invertidas (pior caso), AUC deve ser 0.0,
        pois roc_auc_score mede separação e não acurácia.
        """
        y_true = pd.Series([0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0])
        metrics = compute_metrics(y_true, y_pred)
        assert metrics["auc"] == 0.0

    def test_retorna_floats(self) -> None:
        """Todos os valores do dicionário devem ser do tipo float."""
        y_true = pd.Series([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0])
        metrics = compute_metrics(y_true, y_pred)
        for v in metrics.values():
            assert isinstance(v, float)


class TestLoadModel:
    """Testes do carregamento do MLPClassifier a partir de um state dict em disco."""

    def test_retorna_mlp_classifier(self, trained_model: MLPClassifier) -> None:
        """load_model deve retornar uma instância de MLPClassifier."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.pt"
            torch.save(trained_model.state_dict(), model_path)
            loaded = load_model(model_path, mlp_params)

        assert isinstance(loaded, MLPClassifier)

    def test_modelo_em_eval_mode(self, trained_model: MLPClassifier) -> None:
        """O modelo carregado deve estar em eval mode (training == False)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.pt"
            torch.save(trained_model.state_dict(), model_path)
            loaded = load_model(model_path, mlp_params)

        assert not loaded.training

    def test_pesos_preservados_apos_save_load(
        self, trained_model: MLPClassifier
    ) -> None:
        """
        Os pesos carregados do disco devem ser numericamente idênticos
        aos do modelo original, garantindo a integridade da serialização.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.pt"
            torch.save(trained_model.state_dict(), model_path)
            loaded = load_model(model_path, mlp_params)

        for (name, p_orig), (_, p_loaded) in zip(
            trained_model.named_parameters(), loaded.named_parameters()
        ):
            assert torch.equal(
                p_orig, p_loaded
            ), f"Peso '{name}' diverge após save/load"

    def test_inferencia_identica_apos_load(
        self,
        trained_model: MLPClassifier,
        scaled_X: pd.DataFrame,
    ) -> None:
        """
        A saída do modelo original e do modelo recarregado deve ser
        bit-a-bit idêntica para o mesmo input, validando o round-trip completo.
        """
        x = torch.tensor(scaled_X.values, dtype=torch.float32)

        with torch.no_grad():
            out_original = trained_model(x)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.pt"
            torch.save(trained_model.state_dict(), model_path)
            loaded = load_model(model_path, mlp_params)

        with torch.no_grad():
            out_loaded = loaded(x)

        assert torch.equal(out_original, out_loaded)


class TestSaveClassificationReport:
    """Testes da persistência do relatório de classificação em disco."""

    def test_arquivo_criado(self) -> None:
        """save_classification_report deve criar o arquivo no caminho indicado."""
        y_true = pd.Series([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])

        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "report.txt"
            save_classification_report(y_true, y_pred, report_path)
            assert report_path.exists()

    def test_conteudo_nao_vazio(self) -> None:
        """O arquivo gerado deve conter texto (não estar vazio)."""
        y_true = pd.Series([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])

        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "report.txt"
            save_classification_report(y_true, y_pred, report_path)
            content = report_path.read_text()
            assert len(content.strip()) > 0

    def test_cria_diretorio_pai_se_nao_existir(self) -> None:
        """
        O diretório pai deve ser criado automaticamente via mkdir(parents=True),
        mesmo que seja um caminho aninhado inexistente.
        """
        y_true = pd.Series([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])

        with tempfile.TemporaryDirectory() as tmpdir:
            nested_path = Path(tmpdir) / "a" / "b" / "report.txt"
            save_classification_report(y_true, y_pred, nested_path)
            assert nested_path.exists()

    def test_relatorio_contem_precision_recall_f1(self) -> None:
        """
        O relatório de classificação gerado pelo sklearn deve mencionar
        'precision', 'recall' e 'f1-score' — as métricas centrais do modelo.
        """
        y_true = pd.Series([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])

        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "report.txt"
            save_classification_report(y_true, y_pred, report_path)
            content = report_path.read_text()

        assert "precision" in content
        assert "recall" in content
        assert "f1-score" in content

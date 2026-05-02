from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Testes: tools.initialize_tools
# ---------------------------------------------------------------------------


class TestInitializeTools:
    """Testes da função initialize_tools que injeta dependências globais."""

    def test_initialize_tools__injeta_model_scaler_vectorstore__variaveis_globais(
        self,
        dummy_model: MagicMock,
        dummy_scaler: MagicMock,
        dummy_vectorstore: MagicMock,
    ) -> None:
        """
        Verifica que initialize_tools popula corretamente as variáveis
        globais _model, _scaler e _vectorstore do módulo tools.
        """
        from src.agent import tools

        tools.initialize_tools(dummy_model, dummy_scaler, dummy_vectorstore)

        assert tools._model is dummy_model
        assert tools._scaler is dummy_scaler
        assert tools._vectorstore is dummy_vectorstore

    def test_initialize_tools__chamado_duas_vezes__substitui_valores_anteriores(
        self,
        dummy_model: MagicMock,
        dummy_scaler: MagicMock,
        dummy_vectorstore: MagicMock,
    ) -> None:
        """
        Garante que uma segunda chamada a initialize_tools substitui os
        valores previamente injetados, sem efeitos colaterais.
        """
        from src.agent import tools

        novo_model = MagicMock()
        tools.initialize_tools(dummy_model, dummy_scaler, dummy_vectorstore)
        tools.initialize_tools(novo_model, dummy_scaler, dummy_vectorstore)

        assert tools._model is novo_model


# ---------------------------------------------------------------------------
# Testes: tools.assess_client_situation
# ---------------------------------------------------------------------------


class TestAssessClientSituation:
    """Testes da ferramenta que diagnostica o perfil financeiro do solicitante."""

    @pytest.mark.parametrize(
        "borrower_income, expected_income_level",
        [
            (100_000.0, "alto"),
            (60_000.0, "médio"),
            (30_000.0, "baixo"),
            (10_000.0, "muito baixo"),
        ],
    )
    def test_assess_client_situation__faixas_de_renda__classifica_corretamente(
        self,
        initialized_tools: None,
        borrower_income: float,
        expected_income_level: str,
    ) -> None:
        """
        Verifica que cada faixa de renda anual é mapeada para o nível correto:
        alto (≥ 80k), médio (≥ 40k), baixo (≥ 20k) e muito baixo (< 20k).
        """
        from src.agent.tools import assess_client_situation

        result: dict[str, Any] = assess_client_situation.invoke(
            {
                "borrower_income": borrower_income,
                "debt_to_income": 0.2,
                "num_of_accounts": 3,
                "derogatory_marks": 0,
            }
        )

        assert result["income_level"] == expected_income_level

    @pytest.mark.parametrize(
        "debt_to_income, expected_severity",
        [
            (0.20, "saudável"),
            (0.40, "moderado"),
            (0.60, "elevado"),
            (0.80, "crítico"),
        ],
    )
    def test_assess_client_situation__niveis_dti__classifica_corretamente(
        self,
        initialized_tools: None,
        debt_to_income: float,
        expected_severity: str,
    ) -> None:
        """
        Verifica que o índice dívida/renda (DTI) é classificado corretamente
        nas quatro categorias: saudável (≤ 0.30), moderado (≤ 0.50),
        elevado (≤ 0.70) e crítico (> 0.70).
        """
        from src.agent.tools import assess_client_situation

        result: dict[str, Any] = assess_client_situation.invoke(
            {
                "borrower_income": 50_000.0,
                "debt_to_income": debt_to_income,
                "num_of_accounts": 3,
                "derogatory_marks": 0,
            }
        )

        assert result["dti_severity"] == expected_severity

    @pytest.mark.parametrize(
        "derogatory_marks, expected_severity",
        [
            (0, "nenhum"),
            (1, "leve"),
            (2, "moderado"),
            (4, "grave"),
        ],
    )
    def test_assess_client_situation__marcas_negativas__classifica_corretamente(
        self,
        initialized_tools: None,
        derogatory_marks: int,
        expected_severity: str,
    ) -> None:
        """
        Verifica a classificação de marcas negativas (derogatory marks):
        nenhum (0), leve (1), moderado (2) e grave (≥ 3).
        """
        from src.agent.tools import assess_client_situation

        result: dict[str, Any] = assess_client_situation.invoke(
            {
                "borrower_income": 50_000.0,
                "debt_to_income": 0.2,
                "num_of_accounts": 3,
                "derogatory_marks": derogatory_marks,
            }
        )

        assert result["derogatory_severity"] == expected_severity

    def test_assess_client_situation__perfil_critico__tom_empatico_e_construtivo(
        self,
        initialized_tools: None,
    ) -> None:
        """
        Para perfis com dois ou mais fatores críticos, o tom sugerido deve
        ser 'empático e construtivo', refletindo a necessidade de maior cuidado
        na comunicação com o cliente.
        """
        from src.agent.tools import assess_client_situation

        result: dict[str, Any] = assess_client_situation.invoke(
            {
                "borrower_income": 8_000.0,  # muito baixo → crítico
                "debt_to_income": 0.85,  # crítico
                "num_of_accounts": 15,
                "derogatory_marks": 3,  # grave
            }
        )

        assert result["suggested_tone"] == "empático e construtivo"

    def test_assess_client_situation__perfil_saudavel__tom_positivo_e_encorajador(
        self,
        initialized_tools: None,
    ) -> None:
        """
        Para perfis sem fatores críticos, o tom deve ser 'positivo e encorajador',
        indicando uma comunicação mais celebratória do resultado.
        """
        from src.agent.tools import assess_client_situation

        result: dict[str, Any] = assess_client_situation.invoke(
            {
                "borrower_income": 120_000.0,
                "debt_to_income": 0.10,
                "num_of_accounts": 2,
                "derogatory_marks": 0,
            }
        )

        assert result["suggested_tone"] == "positivo e encorajador"

    def test_assess_client_situation__retorna_todas_as_chaves_esperadas(
        self,
        initialized_tools: None,
    ) -> None:
        """
        O dicionário retornado deve conter exatamente as chaves documentadas
        na assinatura da ferramenta, garantindo compatibilidade com o agente.
        """
        from src.agent.tools import assess_client_situation

        result: dict[str, Any] = assess_client_situation.invoke(
            {
                "borrower_income": 50_000.0,
                "debt_to_income": 0.3,
                "num_of_accounts": 4,
                "derogatory_marks": 1,
            }
        )

        expected_keys = {
            "income_level",
            "dti_severity",
            "accounts_status",
            "derogatory_severity",
            "primary_concern",
            "suggested_tone",
            "context",
        }
        assert expected_keys == set(result.keys())


# ---------------------------------------------------------------------------
# Testes: tools.get_model_explanation
# ---------------------------------------------------------------------------


class TestGetModelExplanation:
    """Testes da ferramenta que calcula explicações SHAP para o modelo MLP."""

    def test_get_model_explanation__retorna_lista_de_dicts(
        self,
        initialized_tools: None,
        dummy_model: MagicMock,
        dummy_scaler: MagicMock,
    ) -> None:
        """
        Verifica que get_model_explanation retorna uma lista de dicionários
        com as chaves 'feature', 'shap_value' e 'direction', conforme
        documentado na docstring da ferramenta.
        """
        shap_output = [
            {"feature": "debt_to_income", "shap_value": 0.45, "direction": "positive"},
            {
                "feature": "derogatory_marks",
                "shap_value": 0.30,
                "direction": "positive",
            },
            {
                "feature": "borrower_income",
                "shap_value": -0.20,
                "direction": "negative",
            },
            {"feature": "num_of_accounts", "shap_value": 0.05, "direction": "positive"},
        ]

        with patch("src.agent.tools.shap_explain", return_value=shap_output):
            from src.agent.tools import get_model_explanation

            result: list[dict[str, Any]] = get_model_explanation.invoke(
                {
                    "borrower_income": 45_000.0,
                    "debt_to_income": 0.65,
                    "num_of_accounts": 6,
                    "derogatory_marks": 2,
                }
            )

        assert isinstance(result, list)
        assert len(result) > 0
        for item in result:
            assert "feature" in item
            assert "shap_value" in item
            assert "direction" in item

    def test_get_model_explanation__chama_shap_explain_com_dataframe_correto(
        self,
        initialized_tools: None,
    ) -> None:
        """
        Garante que shap_explain é chamado com um DataFrame contendo exatamente
        as quatro colunas esperadas e uma única linha com os valores do solicitante.
        """
        import pandas as pd

        captured: dict[str, Any] = {}

        def capture_shap(data: pd.DataFrame, model: Any, scaler: Any) -> list:
            captured["data"] = data
            return []

        with patch("src.agent.tools.shap_explain", side_effect=capture_shap):
            from src.agent.tools import get_model_explanation

            get_model_explanation.invoke(
                {
                    "borrower_income": 55_000.0,
                    "debt_to_income": 0.40,
                    "num_of_accounts": 5,
                    "derogatory_marks": 1,
                }
            )

        df = captured["data"]
        assert list(df.columns) == [
            "borrower_income",
            "debt_to_income",
            "num_of_accounts",
            "derogatory_marks",
        ]
        assert len(df) == 1
        assert df["borrower_income"].iloc[0] == 55_000.0


# ---------------------------------------------------------------------------
# Testes: tools.get_loan_policy_rules
# ---------------------------------------------------------------------------


class TestGetLoanPolicyRules:
    """Testes da ferramenta de busca em documentos de política de crédito."""

    def test_get_loan_policy_rules__query_valida__retorna_string_nao_vazia(
        self,
        initialized_tools: None,
        dummy_vectorstore: MagicMock,
    ) -> None:
        """
        Verifica que uma consulta válida retorna uma string não vazia
        com os trechos mais relevantes da política de crédito.
        """
        from src.agent.tools import get_loan_policy_rules

        result: str = get_loan_policy_rules.invoke(
            {"query": "como comunicar negativa de crédito"}
        )

        assert isinstance(result, str)
        assert len(result) > 0

    def test_get_loan_policy_rules__chama_vectorstore_com_query_correta(
        self,
        initialized_tools: None,
        dummy_vectorstore: MagicMock,
    ) -> None:
        """
        Verifica que similarity_search é chamado com a query exata fornecida,
        garantindo que a ferramenta repassa o argumento sem modificação.
        """
        from src.agent.tools import get_loan_policy_rules

        query = "tom para aprovação de crédito"
        get_loan_policy_rules.invoke({"query": query})

        dummy_vectorstore.similarity_search.assert_called_once()
        call_args = dummy_vectorstore.similarity_search.call_args
        assert call_args[0][0] == query


# ---------------------------------------------------------------------------
# Testes: rag_pipeline
# ---------------------------------------------------------------------------


class TestRagPipeline:
    """Testes das funções de indexação e busca do pipeline RAG com ChromaDB."""

    def test_search__concatena_conteudo_dos_documentos(self) -> None:
        """
        Verifica que search concatena corretamente o page_content dos
        documentos retornados pelo vectorstore, separados por linha dupla.
        """
        from src.agent.rag_pipeline import search

        doc1 = MagicMock()
        doc1.page_content = "Trecho 1 da política."
        doc2 = MagicMock()
        doc2.page_content = "Trecho 2 da política."

        mock_vectorstore = MagicMock()
        mock_vectorstore.similarity_search.return_value = [doc1, doc2]

        result: str = search("comunicação de crédito", mock_vectorstore, k=2)

        assert "Trecho 1 da política." in result
        assert "Trecho 2 da política." in result
        assert "\n\n" in result

    def test_search__parametro_k_e_repassado_ao_vectorstore(self) -> None:
        """
        Garante que o parâmetro k é corretamente repassado ao método
        similarity_search do vectorstore, controlando a quantidade de
        documentos recuperados.
        """
        from src.agent.rag_pipeline import search

        mock_vectorstore = MagicMock()
        mock_vectorstore.similarity_search.return_value = []

        search("query teste", mock_vectorstore, k=5)

        mock_vectorstore.similarity_search.assert_called_once_with("query teste", k=5)

    def test_search__vectorstore_vazio__retorna_string_vazia(self) -> None:
        """
        Quando o vectorstore não retorna documentos relevantes,
        search deve retornar uma string vazia sem lançar exceção.
        """
        from src.agent.rag_pipeline import search

        mock_vectorstore = MagicMock()
        mock_vectorstore.similarity_search.return_value = []

        result: str = search("query sem resultado", mock_vectorstore)

        assert result == ""

    def test_load_index__cria_indice_quando_diretorio_nao_existe(self) -> None:
        """
        Quando CHROMA_DIR não existe, load_index deve chamar build_index
        para construir o índice do zero, evitando FileNotFoundError.
        """
        with (
            patch("src.agent.rag_pipeline.CHROMA_DIR") as mock_dir,
            patch("src.agent.rag_pipeline.build_index") as mock_build,
        ):
            mock_dir.exists.return_value = False
            mock_build.return_value = MagicMock()

            from src.agent.rag_pipeline import load_index

            load_index()

            mock_build.assert_called_once()

    def test_load_index__carrega_indice_existente_sem_reconstruir(self) -> None:
        """
        Quando CHROMA_DIR já existe, load_index deve carregar o índice
        existente sem chamar build_index, preservando o estado persistido.
        """
        with (
            patch("src.agent.rag_pipeline.CHROMA_DIR") as mock_dir,
            patch("src.agent.rag_pipeline.build_index") as mock_build,
            patch("src.agent.rag_pipeline.Chroma") as mock_chroma,
            patch("src.agent.rag_pipeline._get_embeddings"),
        ):
            mock_dir.exists.return_value = True
            mock_dir.__str__ = lambda self: "/fake/chroma"
            mock_chroma.return_value = MagicMock()

            from src.agent.rag_pipeline import load_index

            load_index()

            mock_build.assert_not_called()
            mock_chroma.assert_called_once()


# ---------------------------------------------------------------------------
# Testes: llm.get_llm
# ---------------------------------------------------------------------------


class TestGetLlm:
    """Testes da função de fábrica do cliente LLM (Groq/ChatGroq)."""

    def test_get_llm__retorna_instancia_chatgroq(self) -> None:
        """
        Verifica que get_llm retorna um objeto ChatGroq configurado
        com o modelo e temperatura corretos.
        """
        with (
            patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}),
            patch("src.agent.llm.ChatGroq") as mock_groq,
        ):
            mock_instance = MagicMock()
            mock_groq.return_value = mock_instance

            from src.agent.llm import GROQ_MODEL, get_llm

            result = get_llm(temperature=0.5)

            mock_groq.assert_called_once_with(
                model=GROQ_MODEL,
                api_key="test-key",
                temperature=0.5,
            )
            assert result is mock_instance

    def test_get_llm__temperatura_padrao_e_0_1(self) -> None:
        """
        Confirma que a temperatura padrão é 0.1, garantindo respostas
        mais determinísticas por padrão.
        """
        with (
            patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}),
            patch("src.agent.llm.ChatGroq") as mock_groq,
        ):
            from src.agent.llm import get_llm

            get_llm()

            _, kwargs = mock_groq.call_args
            assert kwargs["temperature"] == 0.1


# ---------------------------------------------------------------------------
# Testes: react_agent.build_agent
# ---------------------------------------------------------------------------


class TestBuildAgent:
    """Testes da função que instancia o cliente Groq para o agente ReAct."""

    def test_build_agent__retorna_instancia_groq(self) -> None:
        """
        Verifica que build_agent retorna um cliente Groq corretamente
        inicializado com a chave de API lida do ambiente.
        """
        with (
            patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}),
            patch("src.agent.react_agent.Groq") as mock_groq,
        ):
            mock_instance = MagicMock()
            mock_groq.return_value = mock_instance

            from src.agent.react_agent import build_agent

            result = build_agent()

            mock_groq.assert_called_once_with(api_key="test-key")
            assert result is mock_instance


# ---------------------------------------------------------------------------
# Testes: react_agent.run_agent
# ---------------------------------------------------------------------------


class TestRunAgent:
    """Testes do loop ReAct que orquestra as ferramentas e gera a mensagem final."""

    def _make_final_response(self, content: str) -> MagicMock:
        """
        Constrói um mock de resposta da API Groq sem chamadas de ferramentas,
        simulando a mensagem final gerada pelo LLM.

        Args:
            content: Texto da mensagem a ser retornada pelo modelo.

        Returns:
            MagicMock representando um objeto de resposta de chat completo.
        """
        message = MagicMock()
        message.tool_calls = None
        message.content = content

        usage = MagicMock()
        usage.total_tokens = 150

        choice = MagicMock()
        choice.message = message

        response = MagicMock()
        response.choices = [choice]
        response.usage = usage

        return response

    def _make_tool_call_response(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        tool_call_id: str = "call_abc123",
    ) -> MagicMock:
        """
        Constrói um mock de resposta da API Groq com uma chamada de ferramenta.

        Args:
            tool_name: Nome da ferramenta a ser invocada.
            tool_args: Argumentos a serem passados para a ferramenta.
            tool_call_id: Identificador único da chamada de ferramenta.

        Returns:
            MagicMock representando uma resposta com tool_calls.
        """
        tool_call = MagicMock()
        tool_call.id = tool_call_id
        tool_call.function.name = tool_name
        tool_call.function.arguments = json.dumps(tool_args)

        message = MagicMock()
        message.tool_calls = [tool_call]
        message.content = None

        usage = MagicMock()
        usage.total_tokens = 80

        choice = MagicMock()
        choice.message = message

        response = MagicMock()
        response.choices = [choice]
        response.usage = usage

        return response

    def test_run_agent__resposta_direta__retorna_mensagem_e_tokens(
        self, initialized_tools: None
    ) -> None:
        """
        Quando o LLM responde diretamente sem chamar ferramentas,
        run_agent deve retornar a mensagem e a contagem total de tokens.
        """
        agent = MagicMock()
        agent.chat.completions.create.return_value = self._make_final_response(
            "Prezado cliente, seu crédito foi aprovado."
        )

        with patch("src.agent.react_agent._TOOL_MAP", {}):
            from src.agent.react_agent import run_agent

            message, tokens = run_agent(
                agent=agent,
                borrower_income=60_000.0,
                debt_to_income=0.25,
                num_of_accounts=3,
                derogatory_marks=0,
                prediction=0,
                probability=0.12,
            )

        assert message == "Prezado cliente, seu crédito foi aprovado."
        assert tokens == 150

    def test_run_agent__aprovacao__mensagem_inicial_contem_aprovado(
        self, initialized_tools: None
    ) -> None:
        """
        Para prediction=0 (baixo risco), o texto enviado ao LLM deve
        conter a palavra 'APROVADO', garantindo que a decisão correta
        é comunicada ao modelo.
        """
        agent = MagicMock()
        agent.chat.completions.create.return_value = self._make_final_response("ok")

        captured_messages: list[list[dict]] = []

        fixed_response = self._make_final_response("ok")

        def capture_and_return(*args, **kwargs):
            captured_messages.append(kwargs.get("messages", []))
            return fixed_response

        agent.chat.completions.create.side_effect = capture_and_return

        from src.agent.react_agent import run_agent

        run_agent(
            agent=agent,
            borrower_income=80_000.0,
            debt_to_income=0.20,
            num_of_accounts=2,
            derogatory_marks=0,
            prediction=0,
            probability=0.08,
        )

        user_content = captured_messages[0][1]["content"]
        assert "APROVADO" in user_content
        assert "NÃO APROVADO" not in user_content

    def test_run_agent__negacao__mensagem_inicial_contem_nao_aprovado(
        self, initialized_tools: None
    ) -> None:
        """
        Para prediction=1 (alto risco), o texto enviado ao LLM deve
        conter a string 'NÃO APROVADO', refletindo a decisão do modelo.
        """
        agent = MagicMock()
        agent.chat.completions.create.return_value = self._make_final_response("ok")

        captured_messages: list[list[dict]] = []

        def capture_and_return(*args: Any, **kwargs: Any) -> Any:
            captured_messages.append(kwargs.get("messages", []))
            return agent.chat.completions.create.return_value

        agent.chat.completions.create.side_effect = capture_and_return

        from src.agent.react_agent import run_agent

        run_agent(
            agent=agent,
            borrower_income=15_000.0,
            debt_to_income=0.90,
            num_of_accounts=12,
            derogatory_marks=3,
            prediction=1,
            probability=0.87,
        )

        user_content = captured_messages[0][1]["content"]
        assert "NÃO APROVADO" in user_content

    def test_run_agent__com_tool_call__executa_ferramenta_e_continua_loop(
        self, initialized_tools: None
    ) -> None:
        """
        Quando o LLM solicita uma chamada de ferramenta, run_agent deve:
        1. Executar a ferramenta com os argumentos fornecidos.
        2. Adicionar o resultado ao histórico de mensagens.
        3. Continuar o loop até obter uma resposta sem tool_calls.

        Este teste simula um ciclo completo: tool_call → resultado → resposta final.
        """
        args = {
            "borrower_income": 40_000.0,
            "debt_to_income": 0.5,
            "num_of_accounts": 4,
            "derogatory_marks": 1,
        }
        tool_response = self._make_tool_call_response("assess_client_situation", args)
        final_response = self._make_final_response("Mensagem final ao cliente.")

        agent = MagicMock()
        agent.chat.completions.create.side_effect = [tool_response, final_response]

        mock_tool = MagicMock()
        mock_tool.invoke.return_value = {
            "income_level": "médio",
            "suggested_tone": "positivo e encorajador",
        }

        with patch(
            "src.agent.react_agent._TOOL_MAP",
            {"assess_client_situation": mock_tool},
        ):
            from src.agent.react_agent import run_agent

            message, tokens = run_agent(
                agent=agent,
                borrower_income=40_000.0,
                debt_to_income=0.5,
                num_of_accounts=4,
                derogatory_marks=1,
                prediction=0,
                probability=0.30,
            )

        mock_tool.invoke.assert_called_once_with(args)
        assert message == "Mensagem final ao cliente."
        assert tokens == 80 + 150  # soma das duas iterações

    def test_run_agent__acumula_tokens_entre_iteracoes(
        self, initialized_tools: None
    ) -> None:
        """
        Verifica que run_agent acumula corretamente os tokens de todas
        as iterações do loop, retornando o total exato ao final.
        """
        args = {
            "borrower_income": 50_000.0,
            "debt_to_income": 0.3,
            "num_of_accounts": 3,
            "derogatory_marks": 0,
        }

        resp1 = self._make_tool_call_response("assess_client_situation", args, "c1")
        resp1.usage.total_tokens = 100

        resp2 = self._make_tool_call_response("get_model_explanation", args, "c2")
        resp2.usage.total_tokens = 120

        resp3 = self._make_final_response("Mensagem gerada.")
        resp3.usage.total_tokens = 200

        agent = MagicMock()
        agent.chat.completions.create.side_effect = [resp1, resp2, resp3]

        mock_tool = MagicMock()
        mock_tool.invoke.return_value = {}

        with patch(
            "src.agent.react_agent._TOOL_MAP",
            {
                "assess_client_situation": mock_tool,
                "get_model_explanation": mock_tool,
            },
        ):
            from src.agent.react_agent import run_agent

            _, tokens = run_agent(
                agent=agent,
                borrower_income=50_000.0,
                debt_to_income=0.3,
                num_of_accounts=3,
                derogatory_marks=0,
                prediction=0,
                probability=0.15,
            )

        assert tokens == 420  # 100 + 120 + 200

    def test_run_agent__maximo_de_iteracoes__nao_lanca_excecao(
        self, initialized_tools: None
    ) -> None:
        """
        Quando o LLM continua chamando ferramentas por 6 iterações consecutivas
        (limite máximo), run_agent deve retornar normalmente sem lançar exceção,
        retornando o último conteúdo disponível.
        """
        args = {
            "borrower_income": 30_000.0,
            "debt_to_income": 0.6,
            "num_of_accounts": 8,
            "derogatory_marks": 2,
        }

        tool_response = self._make_tool_call_response("assess_client_situation", args)
        tool_response.message = tool_response.choices[0].message
        tool_response.choices[0].message.content = "conteúdo parcial"

        agent = MagicMock()
        agent.chat.completions.create.return_value = tool_response

        mock_tool = MagicMock()
        mock_tool.invoke.return_value = {}

        with patch(
            "src.agent.react_agent._TOOL_MAP",
            {"assess_client_situation": mock_tool},
        ):
            from src.agent.react_agent import run_agent

            result = run_agent(
                agent=agent,
                borrower_income=30_000.0,
                debt_to_income=0.6,
                num_of_accounts=8,
                derogatory_marks=2,
                prediction=1,
                probability=0.75,
            )

        assert isinstance(result, tuple)
        assert len(result) == 2

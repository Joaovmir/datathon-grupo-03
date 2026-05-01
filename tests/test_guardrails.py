import pytest

from src.agent.guardrails import (
    InputGuardrailError,
    OutputGuardrailError,
    validate_input,
    validate_output,
)

# ---------------------------------------------------------------------------
# Inputs válidos: helper para não repetir o caso feliz em cada teste
# ---------------------------------------------------------------------------
VALID_INPUT = dict(
    borrower_income=50000.0,
    debt_to_income=0.35,
    num_of_accounts=3,
    derogatory_marks=0,
)


def test_valid_input_passes():
    validate_input(**VALID_INPUT)


# ---------------------------------------------------------------------------
# Cenário 1: Campo nulo
# ---------------------------------------------------------------------------
def test_input_none_borrower_income():
    with pytest.raises(InputGuardrailError, match="borrower_income"):
        validate_input(
            borrower_income=None,
            debt_to_income=0.35,
            num_of_accounts=3,
            derogatory_marks=0,
        )


def test_input_none_debt_to_income():
    with pytest.raises(InputGuardrailError, match="debt_to_income"):
        validate_input(
            borrower_income=50000.0,
            debt_to_income=None,
            num_of_accounts=3,
            derogatory_marks=0,
        )


# ---------------------------------------------------------------------------
# Cenário 2: Renda negativa
# ---------------------------------------------------------------------------
def test_input_negative_income():
    with pytest.raises(InputGuardrailError, match="borrower_income"):
        validate_input(
            borrower_income=-5000.0,
            debt_to_income=0.35,
            num_of_accounts=3,
            derogatory_marks=0,
        )


# ---------------------------------------------------------------------------
# Cenário 3; DTI fora do intervalo [0, 1]
# ---------------------------------------------------------------------------
def test_input_dti_above_one():
    with pytest.raises(InputGuardrailError, match="debt_to_income"):
        validate_input(
            borrower_income=50000.0,
            debt_to_income=1.5,
            num_of_accounts=3,
            derogatory_marks=0,
        )


def test_input_dti_negative():
    with pytest.raises(InputGuardrailError, match="debt_to_income"):
        validate_input(
            borrower_income=50000.0,
            debt_to_income=-0.1,
            num_of_accounts=3,
            derogatory_marks=0,
        )


# ---------------------------------------------------------------------------
# Cenário 4: Palavra proibida na saída
# ---------------------------------------------------------------------------
def test_output_forbidden_word_risco():
    with pytest.raises(OutputGuardrailError, match="risco"):
        validate_output(
            "Prezado cliente, seu crédito foi negado pois nosso modelo "
            "identificou que você é um risco para a instituição."
        )


def test_output_forbidden_word_modelo():
    with pytest.raises(OutputGuardrailError, match="modelo"):
        validate_output("A decisão foi tomada pelo modelo preditivo.")


def test_output_forbidden_multiple_words():
    with pytest.raises(OutputGuardrailError) as exc_info:
        validate_output("O algoritmo avaliou seu risco de crédito.")
    detail = str(exc_info.value)
    assert "algoritmo" in detail
    assert "risco" in detail


def test_output_forbidden_case_insensitive():
    with pytest.raises(OutputGuardrailError, match="inadimplente"):
        validate_output("O cliente foi classificado como INADIMPLENTE.")


# ---------------------------------------------------------------------------
# Cenário 5: Resposta vazia do agente
# ---------------------------------------------------------------------------
def test_output_empty_string():
    with pytest.raises(OutputGuardrailError, match="vazia"):
        validate_output("")


def test_output_whitespace_only():
    with pytest.raises(OutputGuardrailError, match="vazia"):
        validate_output("   ")


# ---------------------------------------------------------------------------
# Output válido: sem palavras proibidas
# ---------------------------------------------------------------------------
def test_output_valid_message_passes():
    validate_output(
        "Prezado cliente, sua solicitação foi analisada"
        " com base no seu perfil financeiro. "
        "Identificamos que sua capacidade de pagamento"
        " e histórico de contas demonstram "
        "uma boa saúde financeira. Estamos felizes em "
        "informar que sua solicitação foi aprovada."
    )

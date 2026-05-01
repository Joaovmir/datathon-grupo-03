"""
Guardrails de input e output para o endpoint /explain.

Input  — valida os campos antes de chamar o agente.
Output — verifica palavras proibidas na resposta gerada.
"""

PALAVRAS_PROIBIDAS = ["inadimplente", "risco", "algoritmo", "modelo"]


class InputGuardrailError(ValueError):
    pass


class OutputGuardrailError(ValueError):
    pass


def validate_input(
    borrower_income: float,
    debt_to_income: float,
    num_of_accounts: int,
    derogatory_marks: int,
) -> None:
    """Valida que todos os campos são não-nulos e positivos.

    Raises:
        InputGuardrailError: se algum campo for nulo ou inválido.
    """
    fields = {
        "borrower_income": borrower_income,
        "debt_to_income": debt_to_income,
        "num_of_accounts": num_of_accounts,
        "derogatory_marks": derogatory_marks,
    }

    for name, value in fields.items():
        if value is None:
            raise InputGuardrailError(f"Campo '{name}' não pode ser nulo.")

    if borrower_income < 0:
        raise InputGuardrailError("'borrower_income' não pode ser negativo.")
    if (debt_to_income < 0) or (debt_to_income > 1):
        raise InputGuardrailError(
            "'debt_to_income' não pode ser negativo ou maior que 1."
        )
    if num_of_accounts < 0:
        raise InputGuardrailError("'num_of_accounts' não pode ser negativo.")
    # derogatory_marks = 0 é válido (nenhuma marca negativa)
    if derogatory_marks < 0:
        raise InputGuardrailError("'derogatory_marks' não pode ser negativo.")


def validate_output(message: str) -> None:
    """Verifica que a resposta não está vazia e não contém palavras proibidas.

    Raises:
        OutputGuardrailError: se a resposta for vazia ou contiver termo proibido.
    """
    if not message or not message.strip():
        raise OutputGuardrailError("Resposta do agente está vazia.")

    message_lower = message.lower()
    found = [w for w in PALAVRAS_PROIBIDAS if w in message_lower]
    if found:
        raise OutputGuardrailError(
            f"Resposta contém palavra(s) proibida(s): {', '.join(found)}"
        )

import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain_core.tools import tool

from src.agent.rag_pipeline import search
from src.models.baseline import MLPClassifier
from src.models.inference import shap_explain

_model: MLPClassifier | None = None
_scaler = None
_vectorstore: Chroma | None = None


def initialize_tools(model: MLPClassifier, scaler, vectorstore: Chroma) -> None:
    """
    Inject model, scaler and vectorstore into the tools module.

    Must be called once at application startup before any tool is invoked.

    Args:
        model: Trained MLPClassifier in eval mode.
        scaler: Fitted StandardScaler.
        vectorstore: Loaded Chroma vectorstore with credit policy documents.
    """
    global _model, _scaler, _vectorstore
    _model = model
    _scaler = scaler
    _vectorstore = vectorstore


@tool
def get_model_explanation(
    borrower_income: float,
    debt_to_income: float,
    num_of_accounts: int,
    derogatory_marks: int,
) -> list[dict]:
    """
    Explain which features most influenced the credit risk prediction
    for a specific applicant, using SHAP values from the trained MLP model.

    Use this tool to understand the real contribution of each factor
    to the model's decision — ranked from most to least impactful.

    Args:
        borrower_income: Annual income of the applicant in BRL.
        debt_to_income: Ratio of total debt to annual income (0.0 to 1.0).
        num_of_accounts: Number of active credit accounts.
        derogatory_marks: Number of negative marks in credit history.

    Returns:
        List of dicts sorted by absolute SHAP value (most impactful first):
        [{"feature": str, "shap_value": float, "direction": str}, ...]
    """
    data = pd.DataFrame(
        [
            {
                "borrower_income": borrower_income,
                "debt_to_income": debt_to_income,
                "num_of_accounts": num_of_accounts,
                "derogatory_marks": derogatory_marks,
            }
        ]
    )
    return shap_explain(data, _model, _scaler)


@tool
def assess_client_situation(
    borrower_income: float,
    debt_to_income: float,
    num_of_accounts: int,
    derogatory_marks: int,
) -> dict:
    """
    Analyse the applicant's financial profile and return a contextualised
    diagnosis to guide the tone and content of the communication message.

    This tool does NOT decide approval or denial — that is the ML model's role.
    It provides a human-readable reading of the profile so the agent can
    calibrate its language and tailor its orientations accordingly.

    Args:
        borrower_income: Annual income of the applicant in BRL.
        debt_to_income: Ratio of total debt to annual income (0.0 to 1.0).
        num_of_accounts: Number of active credit accounts.
        derogatory_marks: Number of negative marks in credit history.

    Returns:
        dict with severity levels, primary concern, suggested tone and context.
    """
    if borrower_income >= 80000:
        income_level = "alto"
    elif borrower_income >= 40000:
        income_level = "médio"
    elif borrower_income >= 20000:
        income_level = "baixo"
    else:
        income_level = "muito baixo"

    if debt_to_income <= 0.30:
        dti_severity = "saudável"
    elif debt_to_income <= 0.50:
        dti_severity = "moderado"
    elif debt_to_income <= 0.70:
        dti_severity = "elevado"
    else:
        dti_severity = "crítico"

    if num_of_accounts <= 5:
        accounts_status = "saudável"
    elif num_of_accounts <= 10:
        accounts_status = "elevado"
    else:
        accounts_status = "muito elevado"

    if derogatory_marks == 0:
        derogatory_severity = "nenhum"
    elif derogatory_marks == 1:
        derogatory_severity = "leve"
    elif derogatory_marks == 2:
        derogatory_severity = "moderado"
    else:
        derogatory_severity = "grave"

    concerns = {
        "debt_to_income": {"crítico": 3, "elevado": 2, "moderado": 1, "saudável": 0}[
            dti_severity
        ],
        "derogatory_marks": {"grave": 3, "moderado": 2, "leve": 1, "nenhum": 0}[
            derogatory_severity
        ],
        "num_of_accounts": {"muito elevado": 2, "elevado": 1, "saudável": 0}[
            accounts_status
        ],
        "borrower_income": {"muito baixo": 2, "baixo": 1, "médio": 0, "alto": 0}[
            income_level
        ],
    }
    primary_concern = max(concerns, key=lambda k: concerns[k])

    critical_count = sum(1 for v in concerns.values() if v >= 2)
    if critical_count >= 2:
        suggested_tone = "empático e construtivo"
    elif critical_count == 1:
        suggested_tone = "empático mas direto"
    else:
        suggested_tone = "positivo e encorajador"

    context_parts = []
    if dti_severity in ("elevado", "crítico"):
        context_parts.append(
            f"comprometimento financeiro {dti_severity} ({debt_to_income:.0%} da renda)"
        )
    if derogatory_severity in ("moderado", "grave"):
        context_parts.append(f"{derogatory_marks} registro(s) negativo(s) no histórico")
    if accounts_status != "saudável":
        context_parts.append(f"{num_of_accounts} contas ativas")
    if income_level in ("baixo", "muito baixo"):
        context_parts.append(f"renda de R$ {borrower_income:,.0f}")

    context = (
        f"Cliente com renda {income_level}. "
        + (
            ", ".join(context_parts) + ". "
            if context_parts
            else "Perfil sem pontos críticos. "
        )
        + (
            "Orientar redução do comprometimento financeiro antes de nova tentativa."
            if primary_concern == "debt_to_income"
            and dti_severity in ("elevado", "crítico")
            else (
                "Orientar regularização do histórico de crédito."
                if primary_concern == "derogatory_marks"
                and derogatory_severity in ("moderado", "grave")
                else "Perfil dentro de parâmetros razoáveis."
            )
        )
    )

    return {
        "income_level": income_level,
        "dti_severity": dti_severity,
        "accounts_status": accounts_status,
        "derogatory_severity": derogatory_severity,
        "primary_concern": primary_concern,
        "suggested_tone": suggested_tone,
        "context": context,
    }


@tool
def get_loan_policy_rules(query: str) -> str:
    """
    Search the bank's credit communication policy documents for guidelines
    on how to explain a credit decision to the applicant.

    Use this tool to find the appropriate tone, language and structure
    for communicating an approval or denial to the client.

    Args:
        query: Natural language question about how to communicate the decision,
               e.g. "como comunicar negativa de crédito" or "tom para aprovação".

    Returns:
        Relevant excerpts from the official communication policy document.
    """
    return search(query, _vectorstore)

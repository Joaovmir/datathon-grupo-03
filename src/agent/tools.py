import pandas as pd
from langchain_core.tools import tool

from src.models.baseline import MLPClassifier
from src.models.inference import shap_explain

_model: MLPClassifier | None = None
_scaler = None


def initialize_tools(model: MLPClassifier, scaler) -> None:
    """
    Inject model and scaler into the tools module.

    Must be called once at application startup before any tool is invoked.

    Args:
        model: Trained MLPClassifier in eval mode.
        scaler: Fitted StandardScaler.
    """
    global _model, _scaler
    _model = model
    _scaler = scaler


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

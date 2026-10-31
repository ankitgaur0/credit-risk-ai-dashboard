"""Default-model endpoints: metrics, account scores, what-if prediction."""

from fastapi import APIRouter

from src.agent.tools.portfolio_tools import (
    get_account_risk,
    get_model_performance,
    get_top_risk_accounts,
    predict_default_probability,
)
from src.api.errors import handle_errors, raise_if_tool_error
from src.schemas.api import ApplicantRequest

router = APIRouter(prefix="/model", tags=["model"])


@router.get("/metrics")
@handle_errors("GET /model/metrics")
def metrics():
    return raise_if_tool_error(get_model_performance.invoke({}), 500)


@router.get("/accounts/top")
@handle_errors("GET /model/accounts/top")
def top_accounts(n: int = 10, loan_type: str | None = None):
    filters = {"loan_type": loan_type} if loan_type else None
    return raise_if_tool_error(get_top_risk_accounts.invoke({"n": n, "filters": filters}))


@router.get("/accounts/{account_id}")
@handle_errors("GET /model/accounts/{id}")
def account(account_id: str):
    out = get_account_risk.invoke({"account_id": account_id})
    status = 404 if "not found" in str(out.get("error", "")) else 400
    return raise_if_tool_error(out, status)


@router.post("/predict")
@handle_errors("POST /model/predict")
def predict(req: ApplicantRequest):
    return raise_if_tool_error(predict_default_probability.invoke(req.model_dump(exclude_none=True)))

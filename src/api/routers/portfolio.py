"""Portfolio analytics endpoints."""

from fastapi import APIRouter

from src.agent.tools.portfolio_tools import (
    Dimension,
    Metric,
    get_cross_tab,
    get_portfolio_overview,
    get_segment_breakdown,
)
from src.api.errors import handle_errors, raise_if_tool_error

router = APIRouter(prefix="/portfolio", tags=["portfolio"])


@router.get("/kpis")
@handle_errors("GET /portfolio/kpis")
def kpis():
    return raise_if_tool_error(get_portfolio_overview.invoke({}), 500)


@router.get("/segments/{dimension}")
@handle_errors("GET /portfolio/segments")
def segments(dimension: Dimension, min_accounts: int = 30):
    return raise_if_tool_error(get_segment_breakdown.invoke(
        {"dimension": dimension, "min_accounts": min_accounts}))


@router.get("/crosstab")
@handle_errors("GET /portfolio/crosstab")
def crosstab(rows: Dimension, columns: Dimension, metric: Metric = "default_rate"):
    return raise_if_tool_error(get_cross_tab.invoke(
        {"row_dimension": rows, "column_dimension": columns, "metric": metric}))

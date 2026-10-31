"""LangChain tools for the "Ask the Portfolio" agent.

Every tool computes its figures with pandas / the trained model and returns a
JSON-serialisable dict that includes a `source` string. The LLM only chooses
which tool to call and phrases the result; it never calculates numbers itself.
Only aggregates (or one requested account) are returned, never raw customer tables.
"""

from __future__ import annotations

import functools
import time
from typing import Literal

import pandas as pd
from langchain_core.tools import tool

from src import data_store as ds
from src.logging_config import get_logger
from src.model import FEATURES, RISK_ACTIONS, risk_band, top_drivers
from src.portfolio import portfolio_kpis, segment_summary

log = get_logger("agent.tools")

Dimension = Literal[
    "loan_type", "employment_type", "region", "income_band", "age_band", "city_tier",
    "housing_type", "marital_status", "education_level", "loan_purpose",
    "disbursal_channel", "gender", "risk_band", "orig_year",
]
Metric = Literal["default_rate", "accounts", "exposure"]


def logged(fn):
    """Log every tool call (args, duration, result size) and turn exceptions into an
    {"error": ...} result so the agent can explain the failure instead of crashing."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        log.info("TOOL %s called args=%s", fn.__name__, kwargs or args)
        try:
            out = fn(*args, **kwargs)
        except Exception as e:
            log.exception("TOOL %s FAILED after %.2fs: %s", fn.__name__, time.perf_counter() - start, e)
            return {"error": f"{type(e).__name__}: {e}", "tool": fn.__name__,
                    "hint": "Check the arguments (dimension / filter names and values) and try again."}
        dt = time.perf_counter() - start
        if isinstance(out, dict) and "error" in out:
            log.warning("TOOL %s returned error in %.3fs -> %s", fn.__name__, dt, out["error"])
        else:
            log.info("TOOL %s ok in %.3fs -> %d chars", fn.__name__, dt, len(str(out)))
        return out
    return wrapper


def _r(x, nd: int = 4):
    return round(float(x), nd)


def _records(frame: pd.DataFrame, nd: int = 4) -> list[dict]:
    frame = frame.reset_index()
    for c in frame.select_dtypes("number").columns:
        frame[c] = frame[c].round(nd)
    return frame.astype(object).where(frame.notna(), None).to_dict("records")


def _apply_filters(df: pd.DataFrame, filters: dict | None) -> tuple[pd.DataFrame, str]:
    if not filters:
        return df, "all accounts"
    for col, val in filters.items():
        if col not in df.columns:
            raise ValueError(f"Unknown filter column '{col}'")
        vals = [str(v) for v in (val if isinstance(val, list) else [val])]
        valid = set(df[col].astype(str).unique())
        unknown = [v for v in vals if v not in valid]
        if unknown:
            raise ValueError(f"Unknown value(s) {unknown} for '{col}'. Valid: {sorted(valid)[:15]}")
        df = df[df[col].astype(str).isin(vals)]
    return df, ", ".join(f"{k}={v}" for k, v in filters.items())


@tool
@logged
def get_portfolio_overview() -> dict:
    """Headline portfolio KPIs: number of accounts and customers, overall default rate,
    total exposure (outstanding balance) and defaulted exposure."""
    k = portfolio_kpis(ds.get_dataset())
    return {
        "accounts": int(k["accounts"]), "customers": int(k["customers"]),
        "default_rate": _r(k["default_rate"]), "total_exposure_usd": _r(k["total_exposure"], 2),
        "defaulted_exposure_usd": _r(k["defaulted_exposure"], 2),
        "source": "loan_level dataset (all 14,685 loan accounts); exposure = latest outstanding balance",
    }


@tool
@logged
def get_segment_breakdown(dimension: Dimension, filters: dict | None = None, min_accounts: int = 30) -> dict:
    """Default rate, account count, exposure, exposure share and risk lift for each value of ONE
    dimension (e.g. loan_type, region, income_band, age_band, employment_type), sorted by default rate.
    Optional `filters` restricts the population first, e.g. {"region": "South"} or {"loan_type": ["Mortgage"]}.
    Use this for questions like 'which loan type has the highest default rate?'."""
    df, scope = _apply_filters(ds.get_dataset(), filters)
    s = segment_summary(df, dimension, min_accounts=min_accounts)
    cols = ["accounts", "defaults", "default_rate", "risk_lift", "exposure", "exposure_share"]
    return {"dimension": dimension, "population": scope, "rows": _records(s[cols]),
            "source": f"loan_level grouped by {dimension} ({scope}, n={len(df):,}); segments with < {min_accounts} accounts hidden"}


@tool
@logged
def get_cross_tab(row_dimension: Dimension, column_dimension: Dimension, metric: Metric = "default_rate",
                  filters: dict | None = None) -> dict:
    """Two-dimensional table of a metric (default_rate, accounts or exposure) for row_dimension x
    column_dimension, e.g. employment_type x income_band. Use for cross-dimensional questions."""
    df, scope = _apply_filters(ds.get_dataset(), filters)
    agg = {"default_rate": ("target_default", "mean"), "accounts": ("target_default", "size"),
           "exposure": ("exposure", "sum")}[metric]
    t = df.pivot_table(index=row_dimension, columns=column_dimension, values=agg[0], aggfunc=agg[1], observed=True)
    t.columns = t.columns.astype(str)
    return {"metric": metric, "rows": row_dimension, "columns": column_dimension, "table": _records(t),
            "source": f"loan_level pivot {row_dimension} x {column_dimension} ({scope}, n={len(df):,})"}


@tool
@logged
def get_account_risk(account_id: str) -> dict:
    """Model risk score for ONE loan account (e.g. 'ACC200001'): probability of default, risk band,
    top 5 SHAP drivers, plus key account facts. Use for account-level questions."""
    df = ds.get_dataset()
    row = df[df["account_id"] == account_id.strip().upper()]
    if row.empty:
        return {"error": f"Account {account_id} not found", "source": "loan_level"}
    r = row.iloc[0]
    drivers = top_drivers(ds.get_model(), row[FEATURES], n=5)[0]
    return {
        "account_id": r["account_id"], "customer_id": r["customer_id"],
        "pd_score": _r(r["pd_score"]), "risk_band": str(r["risk_band"]),
        "actual_default_flag": int(r["target_default"]),
        "loan_type": r["loan_type"], "loan_amount": _r(r["loan_amount"], 2), "exposure": _r(r["exposure"], 2),
        "annual_income": _r(r["annual_income"], 2), "employment_type": r["employment_type"],
        "missed_payments": int(r["rep_missed_cnt"]), "max_dpd": int(r["rep_max_dpd"]),
        "bureau_writeoffs": int(r["bureau_writeoff_cnt"]),
        "top_drivers": [{"feature": f, "shap_contribution": _r(v)} for f, v in drivers],
        "source": "XGBoost default model (models/xgb_default_model.joblib) + SHAP; positive SHAP raises risk",
    }


@tool
@logged
def get_top_risk_accounts(n: int = 10, filters: dict | None = None) -> dict:
    """The n accounts with the highest model probability of default, optionally filtered,
    e.g. {"loan_type": "Mortgage", "region": "South"}."""
    df, scope = _apply_filters(ds.get_dataset(), filters)
    top = df.nlargest(min(n, 50), "pd_score")[["account_id", "loan_type", "region", "exposure", "pd_score", "risk_band"]]
    top["risk_band"] = top["risk_band"].astype(str)
    return {"population": scope, "rows": _records(top.set_index("account_id")),
            "source": f"XGBoost pd_score ranking ({scope}, n={len(df):,})"}


@tool
@logged
def predict_default_probability(
    annual_income: float, loan_amount: float, loan_type: str, employment_type: str,
    age: int | None = None, interest_rate_pct: float | None = None, tenure_months: int | None = None,
    missed_payments: int | None = None, bureau_writeoffs: int | None = None, region: str | None = None,
) -> dict:
    """Score a HYPOTHETICAL applicant / loan with the trained default model. Fields not given are
    filled with portfolio medians (numeric) or most common values (categorical); these assumptions
    are listed in the output. loan_type: Personal Loan, Auto Loan, Credit Card, Mortgage, Business Loan.
    employment_type: Salaried, Self-Employed, Business Owner, Contract, Unemployed."""
    df = ds.get_dataset()
    base = {c: (df[c].median() if pd.api.types.is_numeric_dtype(df[c]) else df[c].mode().iat[0]) for c in FEATURES}
    given = {"annual_income": annual_income, "loan_amount": loan_amount, "loan_type": loan_type,
             "employment_type": employment_type, "age": age, "interest_rate_pct": interest_rate_pct,
             "tenure_months": tenure_months, "rep_missed_cnt": missed_payments,
             "bureau_writeoff_cnt": bureau_writeoffs, "region": region}
    given = {k: v for k, v in given.items() if v is not None}
    base.update(given)
    base["loan_to_income"] = base["loan_amount"] / base["annual_income"]
    base["collateral_flag"] = int(base["loan_type"] in ("Mortgage", "Auto Loan"))
    if "bureau_writeoff_cnt" in given:
        base["bureau_has_writeoff"] = int(given["bureau_writeoff_cnt"] > 0)
    if "rep_missed_cnt" in given:
        base["rep_missed_rate"] = given["rep_missed_cnt"] / 12  # 12 observed months in the data
    derived = {"loan_to_income", "collateral_flag", "bureau_has_writeoff", "rep_missed_rate"}
    X = pd.DataFrame([base])[FEATURES]
    p = float(ds.get_model().predict_proba(X)[0, 1])
    band = str(risk_band(pd.Series([p])).iat[0])
    # PD is calibrated: 0.25 means about 1 in 4 similar loans default
    return {"pd_score": _r(p), "risk_band": band,
            "risk_vs_portfolio": _r(p / df["target_default"].mean(), 2),
            "recommended_action": RISK_ACTIONS[band],
            "inputs_used": {k: (_r(v) if isinstance(v, (int, float)) else v) for k, v in given.items()},
            "assumed_from_portfolio": sorted(set(FEATURES) - set(given) - derived),
            "portfolio_default_rate": _r(df["target_default"].mean()),
            "source": "XGBoost default model; unspecified features set to portfolio median/mode"}


@tool
@logged
def get_model_performance() -> dict:
    """Default-model validation metrics (AUC, Gini, KS on train and test) for logistic regression
    and XGBoost, plus the top global feature importances (mean |SHAP|)."""
    imp = ds.get_global_importance().head(10)
    return {"metrics": {m: {s: {k: _r(v) for k, v in d.items()} for s, d in v.items()} for m, v in ds.get_metrics().items()},
            "top_features_mean_abs_shap": {k: _r(v) for k, v in imp.items()},
            "source": "models/metrics.json (30% customer-grouped hold-out) + SHAP on 2,000-account sample"}


@tool
@logged
def get_anomaly_summary(n: int = 10, flag_type: Literal["Own history", "Peer segment"] | None = None) -> dict:
    """Summary of accounts flagged by anomaly detection (unusual repayment or bureau behaviour vs own
    history or peer segment), with counts and the top-n flagged accounts and their reasons."""
    a = ds.get_anomalies()
    f = a[a["flagged"]]
    if flag_type:
        f = f[f["flag_type"] == flag_type]
    top = f.head(min(n, 25))
    return {"flagged_accounts": int(len(f)), "flag_rate": _r(len(f) / len(a)),
            "by_type": f["flag_type"].value_counts().to_dict(),
            "default_rate_flagged": _r(f["target_default"].mean()) if len(f) else None,
            "top": [{"account_id": r.account_id, "loan_type": r.loan_type, "flag_type": r.flag_type,
                     "reasons": r.reasons} for r in top.itertuples()],
            "source": "src/anomaly.py rules (own-history DPD jump / fresh misses, peer robust z-score) + IsolationForest ranking"}


PORTFOLIO_TOOLS = [
    get_portfolio_overview, get_segment_breakdown, get_cross_tab, get_account_risk,
    get_top_risk_accounts, predict_default_probability, get_model_performance, get_anomaly_summary,
]

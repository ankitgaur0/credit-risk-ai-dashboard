"""Portfolio profiling helpers shared by the analytics notebook and the dashboard."""

from __future__ import annotations

import pandas as pd

# Dimensions the brief asks the portfolio to be profiled across
PROFILE_DIMENSIONS = {
    "Income": "income_band",
    "Employment type": "employment_type",
    "Region": "region",
    "Age": "age_band",
    "Loan type": "loan_type",
}


def segment_summary(df: pd.DataFrame, dim: str | list[str], min_accounts: int = 0) -> pd.DataFrame:
    """Accounts, default rate, exposure and exposure share for one or more dimensions."""
    g = df.groupby(dim, observed=True)
    out = pd.DataFrame({
        "accounts": g.size(),
        "defaults": g["target_default"].sum(),
        "default_rate": g["target_default"].mean(),
        "exposure": g["exposure"].sum(),
        "avg_loan_amount": g["loan_amount"].mean(),
        "avg_income": g["annual_income"].mean(),
    })
    out["exposure_share"] = out["exposure"] / df["exposure"].sum()
    out["defaulted_exposure"] = df[df["target_default"] == 1].groupby(dim, observed=True)["exposure"].sum()
    out["defaulted_exposure"] = out["defaulted_exposure"].fillna(0)
    # Lift vs the portfolio: 1.5 = 50% riskier than average
    out["risk_lift"] = out["default_rate"] / df["target_default"].mean()
    return out[out["accounts"] >= min_accounts].sort_values("default_rate", ascending=False)


def cross_tab(df: pd.DataFrame, rows: str, cols: str, value: str = "default_rate") -> pd.DataFrame:
    """Two-dimensional pivot of default rate (or accounts / exposure)."""
    agg = {"default_rate": ("target_default", "mean"),
           "accounts": ("target_default", "size"),
           "exposure": ("exposure", "sum")}[value]
    return df.pivot_table(index=rows, columns=cols, values=agg[0], aggfunc=agg[1], observed=True)


def portfolio_kpis(df: pd.DataFrame) -> dict:
    return {
        "accounts": len(df),
        "customers": df["customer_id"].nunique(),
        "default_rate": df["target_default"].mean(),
        "total_exposure": df["exposure"].sum(),
        "defaulted_exposure": df.loc[df["target_default"] == 1, "exposure"].sum(),
    }


def default_rate_by(df: pd.DataFrame, feature: str, bins: int = 5) -> pd.DataFrame:
    """Default rate across quantile bins of a numeric feature (or levels of a categorical)."""
    key = df[feature]
    if pd.api.types.is_numeric_dtype(key) and key.nunique() > 10:
        key = pd.qcut(key.rank(method="first"), bins, labels=[f"Q{i}" for i in range(1, bins + 1)])
    return df.groupby(key, observed=True)["target_default"].agg(accounts="size", default_rate="mean")

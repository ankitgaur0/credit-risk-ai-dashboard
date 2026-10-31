"""Load, clean and join the 5 raw tables into one loan-level dataset.

Join logic (output grain = 1 row per loan account, ~14,685 rows):

    loan_accounts  (fact, 1 row / account)
      ├─ LEFT JOIN customers          ON customer_id   (many-to-one)
      ├─ LEFT JOIN bureau aggregates  ON customer_id   (bureau is 1-to-many → aggregated first)
      ├─ LEFT JOIN repayment aggs     ON account_id    (1-to-many monthly → aggregated first)
      └─ LEFT JOIN collections aggs   ON account_id    (sparse → aggregated, zero-filled)

Child tables are always aggregated to the join key *before* joining, so the
row count of loan_accounts is preserved (asserted below). Raw rows are never
dropped silently; every fix is logged in the data-quality report.

Run:  uv run python -m src.ingestion
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import ANALYTICS_DATASET, DQ_REPORT, PROCESSED_DIR, RAW_DIR, RAW_FILES

DATE_COLS = {
    "customers": ["customer_since"],
    "loans": ["origination_date"],
    "bureau": ["date_opened", "date_closed"],
    "collections": ["event_date"],
}

ID_COLS = {
    "customers": ["customer_id"],
    "loans": ["account_id", "customer_id"],
    "bureau": ["bureau_id", "customer_id"],
    "repayments": ["account_id"],
    "collections": ["collection_event_id", "account_id"],
}

PRIMARY_KEYS = {
    "customers": ["customer_id"],
    "loans": ["account_id"],
    "bureau": ["bureau_id"],
    "repayments": ["account_id", "payment_month"],
    "collections": ["collection_event_id"],
}


class DQLog:
    """Collects every data-quality decision so it can be exported and shown in the app."""

    def __init__(self) -> None:
        self.rows: list[dict] = []

    def add(self, table: str, column: str, issue: str, rows_affected: int, action: str) -> None:
        self.rows.append(
            {"table": table, "column": column, "issue": issue,
             "rows_affected": int(rows_affected), "action": action}
        )

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


# --------------------------------------------------------------------------- load

def load_raw(raw_dir=RAW_DIR) -> dict[str, pd.DataFrame]:
    """Read each CSV with explicit dtypes: IDs as strings, dates parsed."""
    tables = {}
    for name, file in RAW_FILES.items():
        df = pd.read_csv(
            raw_dir / file,
            dtype={c: "string" for c in ID_COLS[name]},
            parse_dates=DATE_COLS.get(name, []),
        )
        # trim stray whitespace in all text columns
        for col in df.select_dtypes(include=["object", "string"]).columns:
            df[col] = df[col].str.strip()
        tables[name] = df
    return tables


def validate_keys(t: dict[str, pd.DataFrame], log: DQLog) -> None:
    """Primary-key uniqueness and referential integrity checks."""
    for name, pk in PRIMARY_KEYS.items():
        dups = t[name].duplicated(pk).sum()
        if dups:
            t[name] = t[name].drop_duplicates(pk, keep="first")
            log.add(name, ",".join(pk), "duplicate primary key", dups, "kept first occurrence")

    checks = [
        ("loans", "customer_id", "customers"),
        ("bureau", "customer_id", "customers"),
        ("repayments", "account_id", "loans"),
        ("collections", "account_id", "loans"),
    ]
    for child, key, parent in checks:
        orphans = (~t[child][key].isin(t[parent][key])).sum()
        if orphans:
            t[child] = t[child][t[child][key].isin(t[parent][key])]
            log.add(child, key, f"orphan key (not in {parent})", orphans, "excluded from join")
        else:
            log.add(child, key, f"referential integrity vs {parent}", 0, "passed")


# --------------------------------------------------------------------------- clean

def clean_customers(df: pd.DataFrame, log: DQLog) -> pd.DataFrame:
    df = df.copy()

    short = df["gender"].isin(["M", "F"]).sum()
    df["gender"] = df["gender"].replace({"M": "Male", "F": "Female"})
    log.add("customers", "gender", "inconsistent labels 'M'/'F'", short, "mapped to Male/Female")

    # Impute with segment medians (not global) and keep a flag: missingness may itself be predictive
    for col, by in [("years_employed", "employment_type"),
                    ("num_dependents", "marital_status"),
                    ("years_at_residence", "housing_type")]:
        n = df[col].isna().sum()
        df[f"{col}_missing"] = df[col].isna().astype("int8")
        df[col] = df[col].fillna(df.groupby(by)[col].transform("median")).fillna(df[col].median())
        log.add("customers", col, "missing values", n,
                f"imputed median by {by}; flag column {col}_missing added")

    df["num_dependents"] = df["num_dependents"].round().astype("int64")

    df["age_band"] = pd.cut(df["age"], [20, 30, 40, 50, 60, 70],
                            labels=["21-30", "31-40", "41-50", "51-60", "61-70"])
    df["income_band"] = pd.qcut(df["annual_income"], 5,
                                labels=["Q1 (lowest)", "Q2", "Q3", "Q4", "Q5 (highest)"])
    return df


def clean_loans(df: pd.DataFrame, log: DQLog) -> pd.DataFrame:
    df = df.copy()
    n = df["interest_rate_pct"].isna().sum()
    df["interest_rate_missing"] = df["interest_rate_pct"].isna().astype("int8")
    df["interest_rate_pct"] = df["interest_rate_pct"].fillna(
        df.groupby("loan_type")["interest_rate_pct"].transform("median"))
    log.add("loans", "interest_rate_pct", "missing values", n,
            "imputed median by loan_type; flag interest_rate_missing added")

    bad_target = (~df["target_default"].isin([0, 1])).sum()
    log.add("loans", "target_default", "target not in {0,1}", bad_target,
            "none needed" if bad_target == 0 else "review")
    return df


def clean_bureau(df: pd.DataFrame, log: DQLog) -> pd.DataFrame:
    df = df.copy()
    n = df["days_past_due"].isna().sum()
    # Missing DPD is treated as "not reported", not as 0: keep flag, fill 0 only for aggregation
    df["dpd_missing"] = df["days_past_due"].isna().astype("int8")
    df["days_past_due"] = df["days_past_due"].fillna(0)
    log.add("bureau", "days_past_due", "missing values", n,
            "filled 0 for aggregation; count of missing kept as feature bureau_dpd_missing_cnt")

    log.add("bureau", "date_closed", "null for active lines", df["date_closed"].isna().sum(),
            "expected (still active) - left as NaT")

    over = (df["outstanding_amount"] > df["sanctioned_amount"]).sum()
    log.add("bureau", "outstanding_amount", "outstanding > sanctioned", over, "kept (over-limit is a risk signal)")
    return df


# --------------------------------------------------------------------------- aggregate

def aggregate_bureau(b: pd.DataFrame) -> pd.DataFrame:
    """Customer-level bureau features."""
    b = b.assign(
        is_active=b["account_status"].eq("Active"),
        is_writeoff=b["account_status"].eq("Written-Off"),
        active_outstanding=np.where(b["account_status"].eq("Active"), b["outstanding_amount"], 0.0),
        active_sanctioned=np.where(b["account_status"].eq("Active"), b["sanctioned_amount"], 0.0),
    )
    g = b.groupby("customer_id")
    out = pd.DataFrame({
        "bureau_lines_cnt": g.size(),
        "bureau_active_cnt": g["is_active"].sum(),
        "bureau_writeoff_cnt": g["is_writeoff"].sum(),
        "bureau_total_sanctioned": g["sanctioned_amount"].sum(),
        "bureau_total_outstanding": g["outstanding_amount"].sum(),
        "bureau_max_dpd": g["days_past_due"].max(),
        "bureau_mean_dpd": g["days_past_due"].mean(),
        "bureau_dpd_missing_cnt": g["dpd_missing"].sum(),
        "bureau_lender_types": g["lender_category"].nunique(),
        "bureau_oldest_line": g["date_opened"].min(),
        "_active_out": g["active_outstanding"].sum(),
        "_active_sanc": g["active_sanctioned"].sum(),
    })
    out["bureau_utilisation"] = (out["_active_out"] / out["_active_sanc"].replace(0, np.nan)).fillna(0)
    out["bureau_has_writeoff"] = (out["bureau_writeoff_cnt"] > 0).astype("int8")
    return out.drop(columns=["_active_out", "_active_sanc"]).reset_index()


def aggregate_repayments(r: pd.DataFrame) -> pd.DataFrame:
    """Account-level repayment behaviour features (incl. recent-vs-history trend)."""
    r = r.sort_values(["account_id", "payment_month"])
    last_m = r.groupby("account_id")["payment_month"].transform("max")
    r = r.assign(
        missed=r["payment_status"].eq("Missed"),
        late=r["payment_status"].eq("Late"),
        recent=r["payment_month"] > last_m - 3,  # last 3 months
    )
    g = r.groupby("account_id")
    out = pd.DataFrame({
        "rep_months_observed": g.size(),
        "rep_missed_cnt": g["missed"].sum(),
        "rep_late_cnt": g["late"].sum(),
        "rep_max_dpd": g["days_past_due"].max(),
        "rep_mean_dpd": g["days_past_due"].mean(),
        "rep_total_due": g["emi_due"].sum(),
        "rep_total_paid": g["emi_paid"].sum(),
        "rep_last_balance": g["outstanding_balance"].last(),
        "rep_dpd30_cnt": g["days_past_due"].apply(lambda s: (s >= 30).sum()),
    })
    recent = r[r["recent"]].groupby("account_id")["days_past_due"].mean().rename("rep_recent3_mean_dpd")
    out = out.join(recent)
    out["rep_pay_ratio"] = out["rep_total_paid"] / out["rep_total_due"].replace(0, np.nan)
    out["rep_missed_rate"] = out["rep_missed_cnt"] / out["rep_months_observed"]
    out["rep_dpd_trend"] = out["rep_recent3_mean_dpd"] - out["rep_mean_dpd"]  # >0 = worsening
    return out.reset_index()


def aggregate_collections(c: pd.DataFrame) -> pd.DataFrame:
    """Account-level collections summary.

    NOTE: collections usually happen *after* delinquency, so these columns are
    for analytics/dashboard only and are excluded from the default model to
    avoid target leakage.
    """
    g = c.groupby("account_id")
    out = pd.DataFrame({
        "coll_events_cnt": g.size(),
        "coll_amount_recovered": g["amount_recovered"].sum(),
        "coll_legal_or_writeoff": g["action_type"].apply(
            lambda s: s.isin(["Legal Notice", "Written-Off Referral"]).sum()),
        "coll_last_event_date": g["event_date"].max(),
    })
    return out.reset_index()


# --------------------------------------------------------------------------- build

def build_dataset(raw_dir=RAW_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    log = DQLog()
    t = load_raw(raw_dir)
    for name, df in t.items():
        log.add(name, "*", "rows loaded", len(df), "-")
    validate_keys(t, log)

    customers = clean_customers(t["customers"], log)
    loans = clean_loans(t["loans"], log)
    bureau = clean_bureau(t["bureau"], log)

    bureau_agg = aggregate_bureau(bureau)
    rep_agg = aggregate_repayments(t["repayments"])
    coll_agg = aggregate_collections(t["collections"])

    df = (
        loans
        .merge(customers, on="customer_id", how="left", validate="many_to_one")
        .merge(bureau_agg, on="customer_id", how="left", validate="many_to_one")
        .merge(rep_agg, on="account_id", how="left", validate="one_to_one")
        .merge(coll_agg, on="account_id", how="left", validate="one_to_one")
    )
    assert len(df) == len(loans), "join changed the loan-level row count"

    # Customers with no bureau record / accounts with no collections: absence = 0, not unknown
    no_bureau = df["bureau_lines_cnt"].isna().sum()
    zero_cols = [c for c in df.columns
                 if c.startswith(("bureau_", "coll_")) and c not in ("bureau_oldest_line", "coll_last_event_date")]
    df[zero_cols] = df[zero_cols].fillna(0)
    df["has_bureau_record"] = df["bureau_oldest_line"].notna().astype("int8")
    log.add("joined", "bureau_*", "customer has no bureau record", no_bureau, "filled 0; has_bureau_record flag")
    log.add("joined", "coll_*", "account has no collections events",
            (df["coll_events_cnt"] == 0).sum(), "filled 0 (no collections activity)")

    # Derived business features
    df["loan_to_income"] = df["loan_amount"] / df["annual_income"]
    df["monthly_emi"] = df["rep_total_due"] / df["rep_months_observed"]
    df["emi_to_monthly_income"] = df["monthly_emi"] / (df["annual_income"] / 12)
    df["exposure"] = df["rep_last_balance"].fillna(df["loan_amount"])
    df["orig_year"] = df["origination_date"].dt.year
    df["customer_tenure_yrs"] = (df["origination_date"] - df["customer_since"]).dt.days / 365.25
    df["customer_loan_cnt"] = df.groupby("customer_id")["account_id"].transform("count")

    log.add("joined", "*", "final analysis-ready rows", len(df), "1 row per loan account")
    return df, log.to_frame()


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df, dq = build_dataset()
    df.to_parquet(ANALYTICS_DATASET, index=False)
    dq.to_csv(DQ_REPORT, index=False)
    print(dq.to_string(index=False))
    print(f"\nSaved {df.shape[0]:,} rows x {df.shape[1]} cols -> {ANALYTICS_DATASET}")
    print(f"Default rate: {df['target_default'].mean():.2%}")


if __name__ == "__main__":
    main()

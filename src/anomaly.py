"""Anomaly detection on repayment and bureau behaviour.

Two lenses, combined:
  1. Own history  - the account's last 3 months vs its own earlier months
                    (sudden DPD jump, fresh missed payments after a clean record).
  2. Peer segment - robust z-score (median / MAD) against accounts of the same
                    loan_type x income_band.
An IsolationForest over the same features gives a single ranking score.

An account is flagged when at least one explainable rule fires; every flag
carries human-readable reasons so the analyst can confirm or dismiss it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.config import RAW_DIR

PEER_KEYS = ["loan_type", "income_band"]
PEER_FEATURES = {
    "rep_mean_dpd": "average repayment DPD",
    "rep_missed_rate": "missed-payment rate",
    "bureau_max_dpd": "max bureau DPD",
    "bureau_utilisation": "bureau utilisation",
    "bureau_writeoff_cnt": "bureau write-offs",
}
PEER_Z_THRESHOLD = 4.0       # robust z-score vs peer segment
DPD_JUMP_THRESHOLD = 20      # days: recent-3m mean DPD minus earlier mean DPD
NEW_MISSES_THRESHOLD = 2     # missed EMIs in last 3 months after none before
IFOREST_CONTAMINATION = 0.03

TRIGGER_TEXT = {
    "trigger": (f"Runs on every rebuild of the data. Fires when an account's recent 3-month DPD is "
                f">= {DPD_JUMP_THRESHOLD} days above its own earlier average, when it misses "
                f">= {NEW_MISSES_THRESHOLD} EMIs in the last 3 months after a clean history, or when a repayment/bureau metric "
                f"is >= {PEER_Z_THRESHOLD} robust z-scores above its peer segment (loan type x income band)."),
    "false_positive": ("A flagged account that is actually fine, e.g. a one-off late payment caused by a bank holiday "
                       "or a payment-mandate change, or a high-utilisation customer who always repays."),
    "false_negative": ("A deteriorating account that is missed, e.g. slow drift that never crosses a threshold, or "
                       "stress that shows up only outside this data (new loans at another lender)."),
    "human_in_loop": ("An analyst reviews each flag and its reasons, then records Confirm (escalate to collections "
                      "or early-warning watchlist) or Dismiss (with a note). Decisions are logged with timestamp and "
                      "reviewer, and can be used to re-tune thresholds."),
}


def own_history_features(raw_dir=RAW_DIR) -> pd.DataFrame:
    """Recent (last 3 months) vs earlier behaviour, per account."""
    r = pd.read_csv(raw_dir / "repayment_history.csv", dtype={"account_id": "string"})
    last = r.groupby("account_id")["payment_month"].transform("max")
    r["period"] = np.where(r["payment_month"] > last - 3, "recent", "earlier")
    r["missed"] = r["payment_status"].eq("Missed")
    p = r.pivot_table(index="account_id", columns="period",
                      values=["days_past_due", "missed"], aggfunc={"days_past_due": "mean", "missed": "sum"})
    p.columns = [f"{v}_{k}" for v, k in p.columns]
    return pd.DataFrame({
        "dpd_recent": p["days_past_due_recent"],
        "dpd_earlier": p["days_past_due_earlier"],
        "dpd_jump": p["days_past_due_recent"] - p["days_past_due_earlier"],
        "missed_recent": p["missed_recent"],
        "missed_earlier": p["missed_earlier"],
    }).reset_index()


def _robust_z(s: pd.Series) -> pd.Series:
    med = s.median()
    mad = (s - med).abs().median()
    scale = 1.4826 * mad if mad > 0 else s.std() or 1.0
    return (s - med) / scale


def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per account with anomaly score, flag and reasons."""
    d = df.merge(own_history_features(), on="account_id", how="left")

    peer_stats = {}
    for col in PEER_FEATURES:
        d[f"z_{col}"] = d.groupby(PEER_KEYS, observed=True)[col].transform(_robust_z)
        peer_stats[col] = d.groupby(PEER_KEYS, observed=True)[col].transform("median")

    iso_cols = list(PEER_FEATURES) + ["dpd_jump", "missed_recent", "loan_to_income"]
    X = d[iso_cols].fillna(0)
    iso = IsolationForest(n_estimators=300, contamination=IFOREST_CONTAMINATION, random_state=42).fit(X)
    d["anomaly_score"] = -iso.score_samples(X)  # higher = more unusual
    d["iforest_outlier"] = iso.predict(X) == -1

    reasons = [[] for _ in range(len(d))]
    jump = d["dpd_jump"] >= DPD_JUMP_THRESHOLD
    for i in np.flatnonzero(jump):
        reasons[i].append(f"Own history: avg DPD last 3 months {d['dpd_recent'].iat[i]:.0f} days vs "
                          f"{d['dpd_earlier'].iat[i]:.0f} days before (+{d['dpd_jump'].iat[i]:.0f})")
    fresh = (d["missed_recent"] >= NEW_MISSES_THRESHOLD) & (d["missed_earlier"] == 0)
    for i in np.flatnonzero(fresh):
        reasons[i].append(f"Own history: {d['missed_recent'].iat[i]:.0f} missed EMIs in last 3 months after a clean record")
    for col, label in PEER_FEATURES.items():
        hit = d[f"z_{col}"] >= PEER_Z_THRESHOLD
        for i in np.flatnonzero(hit):
            reasons[i].append(f"Peer segment: {label} {d[col].iat[i]:.2f} vs peer median {peer_stats[col].iat[i]:.2f} "
                              f"({d['loan_type'].iat[i]}, income {d['income_band'].iat[i]}; z={d[f'z_{col}'].iat[i]:.1f})")

    d["reasons"] = reasons
    d["n_reasons"] = d["reasons"].str.len()
    d["flagged"] = d["n_reasons"] > 0
    d["flag_type"] = np.select(
        [jump | fresh, d["n_reasons"] > 0], ["Own history", "Peer segment"], default="")

    keep = ["account_id", "customer_id", "loan_type", "income_band", "region", "employment_type",
            "exposure", "pd_score", "risk_band", "target_default", "anomaly_score", "iforest_outlier",
            "flagged", "flag_type", "n_reasons", "reasons", "dpd_recent", "dpd_earlier",
            "missed_recent", "missed_earlier"]
    return d[keep].sort_values(["flagged", "n_reasons", "anomaly_score"], ascending=False).reset_index(drop=True)

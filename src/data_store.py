"""Single cached access point for data, models and scores.

Used by the Streamlit app, the FastAPI routers and the agent tools, so every
surface reads exactly the same numbers. Builds the dataset and models on
first use if they are missing (e.g. a fresh Streamlit Cloud container).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from src.config import ANALYTICS_DATASET, DQ_REPORT, MODELS_DIR, PROCESSED_DIR
from src.logging_config import get_logger, timed

log = get_logger("data_store")

XGB_PATH = MODELS_DIR / "xgb_default_model.joblib"
LOGREG_PATH = MODELS_DIR / "logreg_default_model.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"
SCORES_PATH = MODELS_DIR / "account_scores.parquet"
DECISIONS_PATH = PROCESSED_DIR / "anomaly_decisions.csv"


def ensure_built() -> None:
    from src import ingestion, model

    if not ANALYTICS_DATASET.exists():
        log.warning("Processed dataset missing -> running ingestion")
        with timed(log, "ingestion build"):
            ingestion.main()
    if not all(p.exists() for p in (XGB_PATH, LOGREG_PATH, METRICS_PATH, SCORES_PATH)):
        log.warning("Model artefacts missing -> training models")
        with timed(log, "model training"):
            model.main()


@lru_cache(maxsize=1)
def get_dataset() -> pd.DataFrame:
    """Loan-level dataset joined with model scores (pd_score, risk_band)."""
    ensure_built()
    with timed(log, "load dataset + scores"):
        df = pd.read_parquet(ANALYTICS_DATASET)
        df = df.merge(pd.read_parquet(SCORES_PATH), on="account_id", how="left", validate="one_to_one")
        df["risk_band"] = pd.Categorical(df["risk_band"], ["Low", "Medium", "High", "Very High"], ordered=True)
    log.info("Dataset ready: %d accounts x %d columns", *df.shape)
    return df


@lru_cache(maxsize=1)
def get_dq_report() -> pd.DataFrame:
    ensure_built()
    return pd.read_csv(DQ_REPORT)


@lru_cache(maxsize=1)
def get_model():
    ensure_built()
    with timed(log, f"load model {XGB_PATH.name}"):
        return joblib.load(XGB_PATH)


@lru_cache(maxsize=1)
def get_metrics() -> dict:
    ensure_built()
    return json.loads(METRICS_PATH.read_text())


@lru_cache(maxsize=1)
def get_global_importance(sample: int = 2000) -> pd.Series:
    """Mean |SHAP| per feature for the XGBoost model on a fixed sample."""
    from src.model import FEATURES, shap_values

    df = get_dataset()
    with timed(log, f"global SHAP importance on {sample} accounts"):
        values, Xt = shap_values(get_model(), df[FEATURES].sample(min(sample, len(df)), random_state=42))
    return pd.Series(np.abs(values).mean(0), index=Xt.columns).sort_values(ascending=False)


@lru_cache(maxsize=1)
def get_anomalies() -> pd.DataFrame:
    from src.anomaly import detect_anomalies

    with timed(log, "anomaly detection"):
        a = detect_anomalies(get_dataset())
    log.info("Anomalies: %d of %d accounts flagged", int(a["flagged"].sum()), len(a))
    return a


# --------------------------------------------------------------------------- analyst decisions

def load_decisions() -> pd.DataFrame:
    if DECISIONS_PATH.exists():
        return pd.read_csv(DECISIONS_PATH)
    return pd.DataFrame(columns=["account_id", "decision", "note", "reviewer", "decided_at"])


def save_decision(account_id: str, decision: str, note: str = "", reviewer: str = "analyst") -> dict:
    """Append a confirm/dismiss decision; the latest decision per account wins."""
    row = {"account_id": account_id, "decision": decision, "note": note, "reviewer": reviewer,
           "decided_at": datetime.now(UTC).isoformat(timespec="seconds")}
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(DECISIONS_PATH, mode="a", header=not DECISIONS_PATH.exists(), index=False)
    log.info("Anomaly decision saved: %s -> %s by %s", account_id, decision, reviewer)
    return row


def latest_decisions() -> pd.DataFrame:
    return load_decisions().drop_duplicates("account_id", keep="last")

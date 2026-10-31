"""Central paths and constants shared across the pipeline."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"

ANALYTICS_DATASET = PROCESSED_DIR / "loan_level.parquet"
DQ_REPORT = PROCESSED_DIR / "data_quality_report.csv"

RAW_FILES = {
    "customers": "customers.csv",
    "loans": "loan_accounts.csv",
    "bureau": "bureau_credit.csv",
    "repayments": "repayment_history.csv",
    "collections": "collections_events.csv",
}

# Gemini free-tier model used by the "Ask the Portfolio" feature (override with GEMINI_MODEL env var)
GEMINI_MODEL = os.getenv("GEMINI_MODEL") or "gemini-3.5-flash-lite"

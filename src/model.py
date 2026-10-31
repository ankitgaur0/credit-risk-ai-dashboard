"""Default model: logistic-regression baseline vs XGBoost.

Features come from customer profile, loan terms, bureau aggregates and repayment
aggregates. Collections columns (coll_*) are excluded on purpose: collections
start after delinquency, so they would leak the target.

Run:  uv run python -m src.model
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src.config import ANALYTICS_DATASET, MODELS_DIR

TARGET = "target_default"
RANDOM_STATE = 42

NUMERIC_FEATURES = [
    # customer
    "age", "annual_income", "years_employed", "num_dependents", "years_at_residence",
    "years_employed_missing", "num_dependents_missing", "years_at_residence_missing",
    # loan
    "loan_amount", "interest_rate_pct", "interest_rate_missing", "tenure_months", "collateral_flag",
    "loan_to_income", "emi_to_monthly_income", "customer_tenure_yrs", "customer_loan_cnt",
    # bureau
    "has_bureau_record", "bureau_lines_cnt", "bureau_active_cnt", "bureau_writeoff_cnt",
    "bureau_has_writeoff", "bureau_total_outstanding", "bureau_utilisation", "bureau_max_dpd",
    "bureau_mean_dpd", "bureau_dpd_missing_cnt", "bureau_lender_types",
    # repayment
    "rep_missed_cnt", "rep_late_cnt", "rep_missed_rate", "rep_max_dpd", "rep_mean_dpd",
    "rep_dpd30_cnt", "rep_recent3_mean_dpd", "rep_dpd_trend", "rep_pay_ratio",
]
CATEGORICAL_FEATURES = [
    "employment_type", "loan_type", "region", "city_tier", "housing_type",
    "marital_status", "education_level", "disbursal_channel", "loan_purpose",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# --------------------------------------------------------------------------- metrics

def ks_statistic(y_true, y_score) -> float:
    """Max separation between the score distributions of defaulters and non-defaulters."""
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    return ks_2samp(y_score[y_true == 1], y_score[y_true == 0]).statistic


def evaluate(y_true, y_score) -> dict:
    auc = float(roc_auc_score(y_true, y_score))
    return {"AUC": auc, "Gini": 2 * auc - 1, "KS": float(ks_statistic(y_true, y_score))}


def roc_points(y_true, y_score) -> pd.DataFrame:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return pd.DataFrame({"fpr": fpr, "tpr": tpr})


# --------------------------------------------------------------------------- models

def _preprocessor(scale: bool) -> ColumnTransformer:
    num = StandardScaler() if scale else "passthrough"
    return ColumnTransformer([
        ("num", num, NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
    ], verbose_feature_names_out=False)


def build_logreg() -> Pipeline:
    return Pipeline([
        ("prep", _preprocessor(scale=True)),
        # strong L2 penalty: C=0.01 was best in 5-fold customer-grouped CV over [0.01, 0.1, 0.5, 1, 10]
        ("clf", LogisticRegression(max_iter=2000, C=0.01, class_weight="balanced")),
    ])


def build_xgb(scale_pos_weight: float = 1.0) -> Pipeline:
    return Pipeline([
        ("prep", _preprocessor(scale=False)),
        ("clf", XGBClassifier(
            # shallow + strongly regularised: chosen by 5-fold customer-grouped CV (deeper trees overfit)
            n_estimators=600, learning_rate=0.02, max_depth=2, subsample=0.8,
            colsample_bytree=0.8, min_child_weight=50, reg_lambda=1.0,
            scale_pos_weight=scale_pos_weight, eval_metric="auc", random_state=RANDOM_STATE,
        )),
    ])


def split(df: pd.DataFrame, test_size: float = 0.3):
    # Split by customer so the same customer's loans never sit in both train and test
    customers = np.asarray(df["customer_id"].unique(), dtype=object)
    train_c, test_c = train_test_split(customers, test_size=test_size, random_state=RANDOM_STATE)
    train, test = df[df["customer_id"].isin(train_c)], df[df["customer_id"].isin(test_c)]
    return train[FEATURES], test[FEATURES], train[TARGET], test[TARGET]


def train_models(df: pd.DataFrame) -> dict:
    X_tr, X_te, y_tr, y_te = split(df)
    # No class reweighting for XGBoost: reweighting leaves AUC unchanged but inflates every
    # probability ~3x (mean PD 45% vs 16% actual). Unweighted, the PD is calibrated and can be
    # read as a real probability, which the bands, the agent and the dashboard rely on.
    models = {"Logistic Regression": build_logreg(), "XGBoost": build_xgb()}

    results = {}
    for name, m in models.items():
        m.fit(X_tr, y_tr)
        results[name] = {
            "model": m,
            "train": evaluate(y_tr, m.predict_proba(X_tr)[:, 1]),
            "test": evaluate(y_te, m.predict_proba(X_te)[:, 1]),
            "test_scores": m.predict_proba(X_te)[:, 1],
        }
    return {"results": results, "X_test": X_te, "y_test": y_te}


# --------------------------------------------------------------------------- explainability

def logreg_coefficients(pipe: Pipeline) -> pd.DataFrame:
    names = pipe.named_steps["prep"].get_feature_names_out()
    coef = pipe.named_steps["clf"].coef_[0]
    return (pd.DataFrame({"feature": names, "coefficient": coef, "odds_ratio": np.exp(coef)})
            .assign(abs_coef=lambda d: d["coefficient"].abs())
            .sort_values("abs_coef", ascending=False).drop(columns="abs_coef"))


def shap_values(pipe: Pipeline, X: pd.DataFrame):
    """SHAP values for the XGBoost pipeline -> (values, transformed X with feature names)."""
    import shap

    Xt = pd.DataFrame(pipe.named_steps["prep"].transform(X),
                      columns=pipe.named_steps["prep"].get_feature_names_out(), index=X.index)
    explainer = shap.TreeExplainer(pipe.named_steps["clf"])
    return explainer.shap_values(Xt), Xt


def top_drivers(pipe: Pipeline, X: pd.DataFrame, n: int = 5) -> list[list[tuple[str, float]]]:
    """Per-account top-n SHAP drivers (feature, contribution); positive = pushes risk up."""
    values, Xt = shap_values(pipe, X)
    cols = np.array(Xt.columns)
    order = np.argsort(-np.abs(values), axis=1)[:, :n]
    return [[(cols[j], float(values[i, j])) for j in row] for i, row in enumerate(order)]


# Band edges around the 16% portfolio default rate: Low < half the average, Medium below average,
# High up to ~2x average, Very High above. Checked on the hold-out: predicted vs actual default
# rate per band = 6.3/5.5%, 11.7/11.6%, 21.5/21.7%, 39.4/33.0%.
RISK_BANDS = [-0.01, 0.08, 0.16, 0.30, 1.0]
RISK_ACTIONS = {"Low": "Approve", "Medium": "Approve", "High": "Refer to underwriter",
                "Very High": "Decline or require mitigation (collateral / guarantor / lower amount)"}


def risk_band(pd_score: pd.Series) -> pd.Series:
    return pd.cut(pd_score, RISK_BANDS, labels=["Low", "Medium", "High", "Very High"])


# --------------------------------------------------------------------------- persist

def main() -> None:
    df = pd.read_parquet(ANALYTICS_DATASET)
    out = train_models(df)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    metrics = {}
    for name, r in out["results"].items():
        metrics[name] = {"train": r["train"], "test": r["test"]}
        print(f"{name:20s} test AUC={r['test']['AUC']:.3f}  Gini={r['test']['Gini']:.3f}  "
              f"KS={r['test']['KS']:.3f}  (train AUC={r['train']['AUC']:.3f})")

    best = out["results"]["XGBoost"]["model"]
    joblib.dump(best, MODELS_DIR / "xgb_default_model.joblib")
    joblib.dump(out["results"]["Logistic Regression"]["model"], MODELS_DIR / "logreg_default_model.joblib")
    (MODELS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # Score the whole book for the dashboard
    scored = df[["account_id"]].copy()
    scored["pd_score"] = best.predict_proba(df[FEATURES])[:, 1]
    scored["risk_band"] = risk_band(scored["pd_score"]).astype(str)
    scored.to_parquet(MODELS_DIR / "account_scores.parquet", index=False)
    print(f"Saved models, metrics and {len(scored):,} account scores -> {MODELS_DIR}")


if __name__ == "__main__":
    main()

"""Credit Risk Portfolio Monitor: Streamlit dashboard.

Run:  uv run streamlit run app/app.py
Layout: sidebar filters | main views (tabs) | "Ask the Portfolio" chat panel on the right.
"""

import itertools
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")
try:  # Streamlit Cloud secrets -> env, so the agent finds the key the same way everywhere
    if "GEMINI_API_KEY" in st.secrets and not os.getenv("GEMINI_API_KEY"):
        os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
except Exception:
    pass

from src import data_store as ds  # noqa: E402
from src.agent.agent_model import portfolio_agent as agent  # noqa: E402
from src.anomaly import TRIGGER_TEXT as ANOMALY_TEXT  # noqa: E402
from src.config import GEMINI_MODEL, RAW_DIR  # noqa: E402
from src.logging_config import get_logger, new_request_id  # noqa: E402
from src.model import FEATURES, top_drivers  # noqa: E402
from src.portfolio import PROFILE_DIMENSIONS, cross_tab, portfolio_kpis, segment_summary  # noqa: E402

log = get_logger("streamlit")

st.set_page_config(page_title="Credit Risk Portfolio Monitor", layout="wide")
st.markdown("""<style>
.block-container {padding-top: 1.2rem;}
div[data-testid="stMetricValue"] {font-size: 1.5rem;}
</style>""", unsafe_allow_html=True)

BAND_COLORS = {"Low": "#2E7D32", "Medium": "#F9A825", "High": "#EF6C00", "Very High": "#C62828"}
EXTRA_DIMS = {"City tier": "city_tier", "Housing": "housing_type", "Risk band": "risk_band",
              "Origination year": "orig_year", "Channel": "disbursal_channel"}
ALL_DIMS = {**PROFILE_DIMENSIONS, **EXTRA_DIMS}


@st.cache_data(show_spinner="Building dataset and models (first run only)...")
def load():
    return ds.get_dataset(), ds.get_anomalies(), ds.get_dq_report(), ds.get_metrics()


@st.cache_data
def repayment_history() -> pd.DataFrame:
    return pd.read_csv(RAW_DIR / "repayment_history.csv", dtype={"account_id": "string"})


@st.cache_data
def global_importance() -> pd.Series:
    return ds.get_global_importance()


def _chain(first: str, rest):
    """Prepend the first chunk and re-emit word by word: Gemini sends a few large chunks,
    so splitting them gives a smooth typing effect."""
    for chunk in itertools.chain([first] if first else [], rest):
        for word in re.split(r"(\s+)", chunk):
            yield word
            time.sleep(0.012)


df, anomalies, dq, metrics = load()

# --------------------------------------------------------------------------- sidebar filters
st.sidebar.header("Filters")
filters = {}
for label, col in [("Loan type", "loan_type"), ("Region", "region"), ("Employment type", "employment_type"),
                   ("Income band", "income_band"), ("Risk band", "risk_band")]:
    opts = [str(v) for v in df[col].dropna().unique()]
    opts = sorted(opts) if col not in ("income_band", "risk_band") else [str(v) for v in df[col].cat.categories]
    sel = st.sidebar.multiselect(label, opts)
    if sel:
        filters[col] = sel
fdf = df
for col, vals in filters.items():
    fdf = fdf[fdf[col].astype(str).isin(vals)]
st.sidebar.caption(f"{len(fdf):,} of {len(df):,} accounts selected")
if fdf.empty:
    st.warning("No accounts match these filters.")
    st.stop()

# --------------------------------------------------------------------------- layout
st.title("Credit Risk Portfolio Monitor")
main, chat = st.columns([2.6, 1], gap="large")

# =========================================================================== main views
with main:
    k = portfolio_kpis(fdf)
    c = st.columns(5)
    c[0].metric("Accounts", f"{k['accounts']:,}")
    c[1].metric("Default rate", f"{k['default_rate']:.1%}",
                delta=f"{(k['default_rate'] - df['target_default'].mean()) * 100:+.1f} pts vs book", delta_color="inverse")
    c[2].metric("Exposure", f"${k['total_exposure'] / 1e6:,.1f}M")
    c[3].metric("Defaulted exposure", f"${k['defaulted_exposure'] / 1e6:,.1f}M")
    c[4].metric("Flagged anomalies", f"{anomalies['flagged'].loc[anomalies['account_id'].isin(fdf['account_id'])].sum():,}")

    t_port, t_acc, t_anom, t_model, t_dq = st.tabs(
        ["Portfolio", "Account", "Anomaly review", "Model", "Data quality"])

    # ---------------------------------------------------------------- portfolio view
    with t_port:
        left, right = st.columns(2)
        with left:
            bands = fdf.groupby("risk_band", observed=False).agg(accounts=("account_id", "size"),
                                                                exposure=("exposure", "sum")).reset_index()
            fig = px.bar(bands, x="risk_band", y="exposure", color="risk_band", text="accounts",
                         color_discrete_map=BAND_COLORS, title="Exposure by model risk segment (labels = accounts)")
            fig.update_layout(showlegend=False, height=330, yaxis_title="Exposure ($)", xaxis_title=None)
            st.plotly_chart(fig, width="stretch")
        with right:
            vint = fdf.groupby("orig_year").agg(default_rate=("target_default", "mean"),
                                                exposure=("exposure", "sum")).reset_index()
            fig = px.line(vint, x="orig_year", y="default_rate", markers=True, title="Default rate by origination year")
            fig.update_layout(height=330, yaxis_tickformat=".0%", xaxis_title=None)
            st.plotly_chart(fig, width="stretch")

        st.subheader("Profile by dimension")
        dim_label = st.selectbox("Dimension", list(ALL_DIMS), key="dim")
        dim = ALL_DIMS[dim_label]
        seg = segment_summary(fdf, dim).reset_index()
        seg[dim] = seg[dim].astype(str)
        l2, r2 = st.columns(2)
        fig = px.bar(seg.sort_values(dim), x=dim, y="default_rate", title=f"Default rate by {dim_label.lower()}",
                     text=seg.sort_values(dim)["default_rate"].map("{:.1%}".format))
        fig.add_hline(y=df["target_default"].mean(), line_dash="dot", annotation_text="book average")
        fig.update_layout(height=340, yaxis_tickformat=".0%", xaxis_title=None)
        l2.plotly_chart(fig, width="stretch")
        fig = px.pie(seg, names=dim, values="exposure", hole=.5, title=f"Exposure share by {dim_label.lower()}")
        fig.update_layout(height=340)
        r2.plotly_chart(fig, width="stretch")

        st.subheader("Cross-dimensional view")
        a, b = st.columns(2)
        rdim = ALL_DIMS[a.selectbox("Rows", list(ALL_DIMS), index=1)]
        cdim = ALL_DIMS[b.selectbox("Columns", list(ALL_DIMS), index=0)]
        if rdim != cdim:
            ct = cross_tab(fdf, rdim, cdim)
            fig = px.imshow(ct, text_auto=".1%", color_continuous_scale="Reds", aspect="auto")
            fig.update_layout(height=360, coloraxis_colorbar_tickformat=".0%")
            st.plotly_chart(fig, width="stretch")

        st.subheader("Drill-down")
        pick = st.selectbox(f"Show accounts in {dim_label.lower()} segment", seg[dim].tolist())
        drill = fdf[fdf[dim].astype(str) == pick].sort_values("pd_score", ascending=False)
        st.caption(f"{len(drill):,} accounts | default rate {drill['target_default'].mean():.1%} | "
                   f"exposure ${drill['exposure'].sum() / 1e6:,.1f}M. Pick one in the Account tab for details.")
        st.dataframe(drill[["account_id", "loan_type", "region", "employment_type", "annual_income", "exposure",
                            "pd_score", "risk_band", "rep_missed_cnt", "target_default"]].head(200),
                     width="stretch", hide_index=True,
                     column_config={"pd_score": st.column_config.ProgressColumn("PD", min_value=0, max_value=1, format="%.2f")})

    # ---------------------------------------------------------------- account view
    with t_acc:
        top_ids = fdf.sort_values("pd_score", ascending=False)["account_id"].tolist()
        acc_id = st.selectbox("Account (sorted by risk, highest first)", top_ids, key="acc")
        row = df[df["account_id"] == acc_id].iloc[0]
        m = st.columns(4)
        m[0].metric("Probability of default", f"{row['pd_score']:.1%}")
        m[1].metric("Risk band", str(row["risk_band"]))
        m[2].metric("Exposure", f"${row['exposure']:,.0f}")
        m[3].metric("Actual outcome", "Defaulted" if row["target_default"] else "Performing")

        l3, r3 = st.columns([1.2, 1])
        with l3:
            drv = pd.DataFrame(top_drivers(ds.get_model(), df.loc[[row.name], FEATURES], n=8)[0],
                               columns=["feature", "contribution"])
            drv["direction"] = drv["contribution"].map(lambda v: "raises risk" if v > 0 else "lowers risk")
            fig = px.bar(drv.iloc[::-1], x="contribution", y="feature", orientation="h", color="direction",
                         color_discrete_map={"raises risk": "#C62828", "lowers risk": "#2E7D32"},
                         title="Key risk drivers (SHAP, log-odds)")
            fig.update_layout(height=360, yaxis_title=None, legend_title=None)
            st.plotly_chart(fig, width="stretch")
        with r3:
            facts = {"Customer": row["customer_id"], "Loan type": row["loan_type"], "Purpose": row["loan_purpose"],
                     "Loan amount": f"${row['loan_amount']:,.0f}", "Interest rate": f"{row['interest_rate_pct']:.2f}%",
                     "Annual income": f"${row['annual_income']:,.0f}", "Employment": row["employment_type"],
                     "Age": int(row["age"]), "Region": row["region"], "Missed EMIs": int(row["rep_missed_cnt"]),
                     "Max DPD": int(row["rep_max_dpd"]), "Bureau write-offs": int(row["bureau_writeoff_cnt"]),
                     "Bureau utilisation": f"{row['bureau_utilisation']:.0%}", "Collections events": int(row["coll_events_cnt"])}
            st.dataframe(pd.DataFrame(facts.items(), columns=["Field", "Value"]).astype(str),
                         hide_index=True, width="stretch", height=360)
        hist = repayment_history()
        h = hist[hist["account_id"] == acc_id]
        fig = px.bar(h, x="payment_month", y="days_past_due", color="payment_status",
                     color_discrete_map={"On-Time": "#2E7D32", "Late": "#F9A825", "Missed": "#C62828"},
                     title="Repayment history: days past due by month")
        fig.update_layout(height=280)
        st.plotly_chart(fig, width="stretch")
        flag = anomalies[anomalies["account_id"] == acc_id].iloc[0]
        if flag["flagged"]:
            st.warning("**Anomaly flag:** " + "; ".join(flag["reasons"]))

    # ---------------------------------------------------------------- anomaly review
    with t_anom:
        with st.expander("How anomaly detection works: trigger, false positives/negatives, human review"):
            for k_, v in ANOMALY_TEXT.items():
                st.markdown(f"**{k_.replace('_', ' ').title()}:** {v}")
        flagged = anomalies[anomalies["flagged"] & anomalies["account_id"].isin(fdf["account_id"])]
        flagged = flagged.merge(ds.latest_decisions()[["account_id", "decision"]], on="account_id", how="left")
        flagged["decision"] = flagged["decision"].fillna("Pending")
        s = st.columns(4)
        s[0].metric("Flagged accounts", f"{len(flagged):,}")
        s[1].metric("Own-history flags", f"{(flagged['flag_type'] == 'Own history').sum():,}")
        s[2].metric("Peer-segment flags", f"{(flagged['flag_type'] == 'Peer segment').sum():,}")
        s[3].metric("Pending review", f"{(flagged['decision'] == 'Pending').sum():,}")

        ftype = st.radio("Show", ["All", "Own history", "Peer segment", "Pending only"], horizontal=True)
        view = flagged
        if ftype in ("Own history", "Peer segment"):
            view = view[view["flag_type"] == ftype]
        elif ftype == "Pending only":
            view = view[view["decision"] == "Pending"]
        show = view.assign(why=view["reasons"].map(lambda r: r[0] if r else ""), extra=view["n_reasons"] - 1)
        st.dataframe(show[["account_id", "loan_type", "flag_type", "why", "extra", "anomaly_score", "pd_score",
                           "exposure", "decision"]].head(300),
                     hide_index=True, width="stretch",
                     column_config={"why": st.column_config.TextColumn("Main reason", width="large"),
                                    "extra": "More reasons", "anomaly_score": st.column_config.NumberColumn(format="%.3f"),
                                    "pd_score": st.column_config.NumberColumn("PD", format="%.2f")})
        if not view.empty:
            st.markdown("#### Review an account")
            sel = st.selectbox("Flagged account", view["account_id"].head(300).tolist(), key="anom_sel")
            rec = view[view["account_id"] == sel].iloc[0]
            for r in rec["reasons"]:
                st.markdown(f"- {r}")
            with st.form("decision", clear_on_submit=True):
                note = st.text_input("Note (optional)")
                b1, b2 = st.columns(2)
                confirm = b1.form_submit_button("Confirm anomaly", width="stretch")
                dismiss = b2.form_submit_button("Dismiss (false positive)", width="stretch")
                if confirm or dismiss:
                    try:
                        ds.save_decision(sel, "Confirm" if confirm else "Dismiss", note)
                    except OSError as e:
                        log.exception("Could not save decision for %s", sel)
                        st.error(f"Could not save the decision: {e}")
                        st.stop()
                    st.success(f"Recorded {'Confirm' if confirm else 'Dismiss'} for {sel}")
                    st.rerun()
        decision_log = ds.load_decisions()
        if not decision_log.empty:
            st.markdown("#### Decision log")
            st.dataframe(decision_log.iloc[::-1], hide_index=True, width="stretch")

    # ---------------------------------------------------------------- model view
    with t_model:
        mt = pd.DataFrame({(m_, s_): v for m_, d in metrics.items() for s_, v in d.items()}).T
        st.markdown("**Validation (30% hold-out, split by customer)**")
        st.dataframe(mt.style.format("{:.3f}"), width="stretch")
        imp = global_importance().head(15).reset_index()
        imp.columns = ["feature", "mean_abs_shap"]
        fig = px.bar(imp.iloc[::-1], x="mean_abs_shap", y="feature", orientation="h",
                     title="Global feature importance: XGBoost mean |SHAP|")
        fig.update_layout(height=450, yaxis_title=None)
        st.plotly_chart(fig, width="stretch")
        st.markdown(
            "- **Champion:** logistic regression (C=0.01, class-balanced). It has the same AUC as XGBoost and is fully explainable.\n"
            "- **Challenger:** XGBoost (depth 2, heavily regularised); its SHAP values drive the account-level key drivers.\n"
            "- **Excluded:** collections features (they occur after delinquency, so they would leak the target).\n"
            "- **Monitoring:** monthly AUC/KS on matured accounts, PSI on score and top features, calibration by band, "
            "input missing-rate checks, override tracking.")

    with t_dq:
        st.markdown("Every data-quality decision taken during ingestion. **No rows were dropped.**")
        st.dataframe(dq, hide_index=True, width="stretch")

# =========================================================================== Ask the Portfolio
with chat:
    st.subheader("Ask the Portfolio")
    with st.expander("How it works"):
        st.markdown("Gemini picks a whitelisted analytics tool; **pandas computes the figures**; Gemini phrases them. "
                    "Every number is cross-checked against the tool output. Only aggregates are sent to the LLM.")
        for k_, v in agent.AI_FEATURE_TEXT.items():
            st.markdown(f"**{k_.replace('_', ' ').title()}:** {v}")

    if not agent.get_api_key():
        st.info("Set `GEMINI_API_KEY` (in `.env` locally or in Streamlit secrets) to enable the assistant. "
                "All other views work without it.")
    else:
        st.session_state.setdefault("messages", [])
        box = st.container(height=560)
        with box:
            if not st.session_state["messages"]:
                st.caption("Try one of these:")
                for q in ["Which loan type has the highest default rate?",
                          "Default rate by employment type for the lowest income band?",
                          "What is the risk score and main drivers of ACC200001?",
                          "How good is the default model?",
                          "What will interest rates be next year?"]:
                    if st.button(q, width="stretch"):
                        st.session_state["pending"] = q
            for msg in st.session_state["messages"]:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    if msg.get("warning"):
                        st.caption(msg["warning"])

        question = st.chat_input("Ask about the portfolio") or st.session_state.pop("pending", None)
        if question:
            history = [{"role": m["role"], "content": m["content"]} for m in st.session_state["messages"]]
            st.session_state["messages"].append({"role": "user", "content": question})
            rid = new_request_id()
            log.info("Chat question received (model=%s)", GEMINI_MODEL)
            with box:
                with st.chat_message("user"):
                    st.markdown(question)
                with st.chat_message("assistant"):
                    result: dict = {}
                    reply = {"role": "assistant"}
                    try:
                        # only LLM-generated text is streamed; tool calls/outputs stay in the logs
                        with st.spinner("Thinking..."):
                            stream = agent.stream_answer(question, history, result)
                            first = next(stream, "")  # wait (with spinner) until the first token arrives
                        streamed = st.write_stream(_chain(first, stream))
                        reply["content"] = result.get("answer") or streamed
                        if result and not result["declined"] and not result["grounded"]:
                            reply["warning"] = ("Note: some numbers could not be verified against the data: "
                                                f"{result['unverified_numbers']}. Please double-check them.")
                            st.caption(reply["warning"])
                    except agent.AgentError as e:  # classified + logged inside stream_answer()
                        reply["content"] = f"**Couldn't answer:** {e.message}\n\n`{e.code}` | trace id `{rid}`"
                        st.markdown(reply["content"])
                    except Exception as e:  # should not happen; log full traceback
                        log.exception("Chat failed unexpectedly: %s", e)
                        reply["content"] = f"Unexpected error: {type(e).__name__}. Trace id `{rid}`"
                        st.markdown(reply["content"])
            st.session_state["messages"].append(reply)
            st.rerun()
        if st.session_state["messages"] and st.button("Clear chat"):
            st.session_state["messages"] = []
            st.rerun()

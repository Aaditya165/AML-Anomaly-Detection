from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
import io 
import networkx as nx

from src.config import AppConfig
from src.data import load_transactions
from src.explain import explain_account, human_readable_reasons, shap_explain_account, human_readable_shap
from src.features import aggregate_labels_to_account, compute_account_features
from src.models import evaluate_if_labels_available, load_artifacts, score_accounts, train_models

#cachine
@st.cache_data(show_spinner=False)
def _load_and_featurize(csv_bytes: bytes, temporal_window_days: int):
    """Cached: re-runs only when path or temporal_window_days changes."""
    tx = load_transactions(io.BytesIO(csv_bytes))
    feat = compute_account_features(
        tx, 
        temporal_window_days=temporal_window_days
    )
    labels = aggregate_labels_to_account(tx)
    return tx, feat, labels

@st.cache_resource #cache_resource used instead of cache_data because model objetc are considered resources, not data
def load_default_model():
    return load_artifacts(
        str(Path.cwd()/ "artifacts"/ "aml_model.pkl")
    )

st.set_page_config(page_title="AML Transaction Graph Intelligence Dashboard", layout="wide")

st.title("AML Transaction Graph Intelligence Dashboard")
st.caption("Prototype for transaction-graph risk scoring and compliance review")

cfg = AppConfig()

with st.sidebar:
    st.header("Data")

    uploaded_csv = st.file_uploader("Upload transaction CSV", type=["csv"])
    temporal_window_days = st.slider("Temporal window for velocity", 1, 30, cfg.temporal_window_days)
    max_graph_nodes = st.slider("Graph node cap", 50, 300, cfg.max_graph_nodes, 10)
    max_alerts = st.slider("Alert rows", 25, 500, cfg.top_alerts, 25)


run_disabled = ( uploaded_csv is None )
run = st.button("Run analysis", type="primary", disabled=run_disabled)

if uploaded_csv is None:
    st.info( "Upload a transaction CSV file" )
    st.stop()

if run or "result" not in st.session_state:
    with st.spinner("Processing... (first run may take a minute; subsequent runs on the same file are instant)"):
        
        csv_bytes = uploaded_csv.getvalue()
        tx, feat, labels = _load_and_featurize(
            csv_bytes, 
            temporal_window_days
        )

        artifacts = load_default_model()

        scored = score_accounts(feat, artifacts).sort_values("risk_score", ascending=False).reset_index(drop=True)
        metrics = evaluate_if_labels_available(scored, labels if len(labels) else None)

        st.session_state["result"] = {
            "transactions": tx,
            "features": feat,
            "labels": labels,
            "scored": scored,
            "metrics": metrics,
            "artifacts": artifacts,
        }

result = st.session_state["result"]
tx = result["transactions"]
feat = result["features"]
labels = result["labels"]
scored = result["scored"]
metrics = result["metrics"]

left, right = st.columns([1.2, 1])

with left:
    st.subheader("Portfolio summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transactions", f"{len(tx):,}")
    c2.metric("Accounts", f"{len(feat):,}")
    c3.metric("Flagged high", f"{int((scored['risk_tier'] == 'High').sum()):,}")
    c4.metric("Flagged medium", f"{int((scored['risk_tier'] == 'Medium').sum()):,}")

with right:
    if metrics:
        st.subheader("Model Metrics")

        mc1, mc2, mc3 = st.columns(3)

        mc1.metric(
            "Precision",
            f"{metrics.get('precision', 0):.3f}"
        )

        mc2.metric(
            "Recall",
            f"{metrics.get('recall', 0):.3f}"
        )

        mc3.metric(
            "PR-AUC",
            f"{metrics.get('pr_auc', 0):.3f}"
        )

        metric_df = pd.DataFrame({
            "Metric": [
                "Precision",
                "Recall",
                "F1 Score",
                "ROC AUC",
                "PR AUC",
            ],
            "Value": [
                metrics.get("precision", 0),
                metrics.get("recall", 0),
                metrics.get("f1", 0),
                metrics.get("roc_auc", 0),
                metrics.get("pr_auc", 0),
            ]
        })

        st.dataframe(
            metric_df,
            use_container_width=True,
            hide_index=True,
        )

        if "confusion_matrix" in metrics:

            cm = metrics["confusion_matrix"]

            cm_df = pd.DataFrame(
                cm,
                index=[
                    "Actual Non-Laundering",
                    "Actual Laundering",
                ],
                columns=[
                    "Predicted Non-Laundering",
                    "Predicted Laundering",
                ],
            )

            st.subheader("Confusion Matrix")

            st.dataframe(
                cm_df,
                use_container_width=True,
            )
    else:
        st.info("No usable laundering labels were found in this file, so only anomaly scoring is available.")


tab2, tab3 = st.tabs(["Alert Feed", "Portfolio Analytics Dashboard"])

with tab2:
    st.subheader("Alert feed")
    min_tier = st.selectbox("Tier filter", ["All", "High", "Medium", "Low"])
    alerts = scored.copy()
    if min_tier != "All":
        order = {"Low": 0, "Medium": 1, "High": 2}
        alerts = alerts[alerts["risk_tier"].map(order) >= order[min_tier]]
    alerts = alerts.sort_values("risk_score", ascending=False).head(max_alerts)

    st.dataframe(
        alerts[["account", "risk_score", "risk_tier", "supervised_probability", "anomaly_score"]],
        use_container_width=True,
        hide_index=True,
    )

    selected = st.selectbox("Inspect account", [""] + alerts["account"].astype(str).tolist())

    if selected:
        st.markdown(f"### Explanation for {selected}")

        shap_reasons = shap_explain_account(
            feat,
            selected,
            result["artifacts"],
            top_k=7,
        )

        if len(shap_reasons):
            st.subheader("SHAP Model Explanation")
            st.write(human_readable_shap(shap_reasons))

            st.dataframe(pd.DataFrame(shap_reasons), use_container_width=True,)

        else:
            reasons = explain_account(feat, selected, top_k=7,)
            st.subheader("Feature Deviation Explanation")
            st.write(human_readable_reasons(reasons))

            st.dataframe(pd.DataFrame(reasons), use_container_width=True,)

with tab3:
    st.subheader("Portfolio analytics")
    st.write("Risk tier distribution")
    tier_counts = scored["risk_tier"].value_counts().reindex(["High", "Medium", "Low"]).fillna(0)
    st.bar_chart(tier_counts)

    st.write("Risk score distribution")
    st.line_chart(scored["risk_score"].sort_values().reset_index(drop=True))

    st.write("Transaction volume over time")
    tx_time = tx.copy()
    tx_time["date"] = pd.to_datetime(tx_time["timestamp"]).dt.date
    vol = tx_time.groupby("date")["amount_received"].sum().reset_index()
    st.line_chart(vol.set_index("date"))


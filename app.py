from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import AppConfig
from src.data import load_transactions
from src.explain import explain_account, human_readable_reasons
from src.features import aggregate_labels_to_account, compute_account_features
from src.graph_viz import build_risk_subgraph, plot_network
from src.models import evaluate_if_labels_available, load_artifacts, save_artifacts, score_accounts, train_models
from src.utils import find_transaction_files

st.set_page_config(page_title="AML Transaction Graph Intelligence Dashboard", layout="wide")

st.title("AML Transaction Graph Intelligence Dashboard")
st.caption("Prototype for transaction-graph risk scoring and compliance review")

cfg = AppConfig()

with st.sidebar:
    st.header("Data")
    mode = st.radio("Mode", ["Train and save pickle", "Load saved pickle"], index=0)

    data_path = st.text_input("Transaction CSV or folder", value=str(Path.cwd()))
    temporal_window_days = st.slider("Temporal window for velocity", 1, 30, cfg.temporal_window_days)
    max_graph_nodes = st.slider("Graph node cap", 50, 300, cfg.max_graph_nodes, 10)
    max_alerts = st.slider("Alert rows", 25, 500, cfg.top_alerts, 25)

    files = find_transaction_files(data_path)
    choice = st.selectbox("Transaction file", [str(f) for f in files]) if files else None

    if mode == "Train and save pickle":
        model_path = st.text_input("Save trained pickle as", value=str(Path.cwd() / "artifacts" / "aml_model.pkl"))
        uploaded_model = None
    else:
        uploaded_model = st.file_uploader("Upload pickle file", type=["pkl", "pickle"])
        model_path = None

run_disabled = choice is None or (mode == "Load saved pickle" and uploaded_model is None)
run = st.button("Run analysis", type="primary", disabled=run_disabled)

if not choice:
    st.info("Point the app at a folder containing the IBM AML `*_trans.csv` file.")
    st.stop()

if run or "result" not in st.session_state:
    with st.spinner("Processing..."):
        tx = load_transactions(choice)
        feat = compute_account_features(tx, temporal_window_days=temporal_window_days)
        labels = aggregate_labels_to_account(tx)

        if mode == "Train and save pickle":
            artifacts = train_models(feat, labels if len(labels) else None, random_state=cfg.random_state)
            if model_path:
                try:
                    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
                    save_artifacts(artifacts, model_path)
                    st.sidebar.success(f"Saved pickle model to {model_path}")
                except Exception as e:
                    st.sidebar.warning(f"Could not save pickle model: {e}")
        else:
            artifacts = load_artifacts(uploaded_model)

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
        st.subheader("Model metrics")
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Precision", f"{metrics.get('precision', 0):.3f}")
        mc2.metric("Recall", f"{metrics.get('recall', 0):.3f}")
        mc3.metric("PR-AUC", f"{metrics.get('pr_auc', 0):.3f}")
        st.json(metrics)
    else:
        st.info("No usable laundering labels were found in this file, so only anomaly scoring is available.")

tab1, tab2, tab3 = st.tabs(["High-Risk Network View", "Alert Feed", "Portfolio Analytics Dashboard"])

with tab1:
    st.subheader("High-risk network view")
    g = build_risk_subgraph(tx, scored, max_nodes=max_graph_nodes, hops=1)
    fig = plot_network(g, scored)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(
        scored.head(max_alerts)[["account", "risk_score", "risk_tier", "supervised_probability", "anomaly_score"]],
        use_container_width=True,
        hide_index=True,
    )

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
        reasons = explain_account(feat, selected, top_k=7)
        st.markdown(f"### Explanation for {selected}")
        st.write(human_readable_reasons(reasons))
        st.table(pd.DataFrame(reasons))

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

    st.write("Top risky accounts")
    st.dataframe(
        scored.head(20)[["account", "risk_score", "risk_tier", "supervised_probability", "anomaly_score"]],
        use_container_width=True,
        hide_index=True,
    )

st.caption("This prototype uses the transactions CSV only, as requested. It is intended for internal investigation support, not autonomous AML decisioning.")

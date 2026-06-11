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
from src.graph_viz import build_risk_subgraph, plot_network
from src.models import evaluate_if_labels_available, load_artifacts, save_artifacts, score_accounts, train_models
from src.utils import find_transaction_files

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


@st.cache_data(show_spinner=False)
def _cached_train(csv_bytes: bytes, temporal_window_days: int, random_state: int):
    """Cached: re-runs only when path, window, or seed changes."""
    _, feat, labels = _load_and_featurize(
        csv_bytes, 
        temporal_window_days
    )
    return train_models(feat, labels if len(labels) else None, random_state=random_state)

@st.cache_data(show_spinner=False)
def _load_cached_artifacts(model_bytes: bytes):
    return load_artifacts(model_bytes)

@st.cache_data
def build_cached_subgraph(tx, scored, max_nodes):
    return build_risk_subgraph(
        tx,
        scored,
        max_nodes=max_nodes,
        hops=1
    )

st.set_page_config(page_title="AML Transaction Graph Intelligence Dashboard", layout="wide")

st.title("AML Transaction Graph Intelligence Dashboard")
st.caption("Prototype for transaction-graph risk scoring and compliance review")

cfg = AppConfig()

with st.sidebar:
    st.header("Data")
    mode = st.radio("Mode", ["Train and save pickle", "Load saved pickle"], index=1) #paila 0 thiyo

    uploaded_csv = st.file_uploader("Upload transaction CSV", type=["csv"])
    temporal_window_days = st.slider("Temporal window for velocity", 1, 30, cfg.temporal_window_days)
    max_graph_nodes = st.slider("Graph node cap", 50, 300, cfg.max_graph_nodes, 10)
    max_alerts = st.slider("Alert rows", 25, 500, cfg.top_alerts, 25)

    if mode == "Train and save pickle":
        model_path = st.text_input("Save trained pickle as", value=str(Path.cwd() / "artifacts" / "aml_model.pkl"))
        uploaded_model = None
    else:
        uploaded_model = st.file_uploader("Upload pickle file", type=["pkl", "pickle"])
        model_path = None

run_disabled = ( uploaded_csv is None or ( mode == "Load saved pickle" and uploaded_model is None ) )
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

        if mode == "Train and save pickle":
            artifacts = _cached_train(csv_bytes, temporal_window_days, cfg.random_state)
            if model_path:
                try:
                    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
                    save_artifacts(artifacts, model_path)
                    st.sidebar.success(f"Saved pickle model to {model_path}")
                except Exception as e:
                    st.sidebar.warning(f"Could not save pickle model: {e}")
        else:
            artifacts = _load_cached_artifacts(uploaded_model.getvalue())

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

tab1, tab2, tab3 = st.tabs(["High-Risk Network View", "Alert Feed", "Portfolio Analytics Dashboard"])

with tab1:
    st.subheader("High-risk network view")
    g = build_cached_subgraph(tx, scored, scored, max_graph_nodes)
    
    st.write("Nodes:", g.number_of_nodes())
    st.write("Edges:", g.number_of_edges())

    st.write(
        "Connected components:",
        nx.number_connected_components(g.to_undirected)
    )

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

    st.write("Top risky accounts")
    st.dataframe(
        scored.head(20)[["account", "risk_score", "risk_tier", "supervised_probability", "anomaly_score"]],
        use_container_width=True,
        hide_index=True,
    )

st.caption("This prototype uses the transactions CSV only, as requested. It is intended for internal investigation support, not autonomous AML decisioning.")

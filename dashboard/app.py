"""
dashboard.py
------------
PRD Section 16: Dashboard Requirements (Transaction View, Account View).
Plus a Model Insights tab that the GNN version didn't have -- gradient-
boosted trees expose native feature importance, which is exactly the
kind of "why was this flagged" answer AML investigators/regulators need
and a GNN's learned embeddings don't give you for free.

Run with:
    streamlit run dashboard.py

Reads the CSV artifacts produced at the end of the training notebook:
    transaction_view.csv     (Transaction ID, Risk Score, Sender, Receiver, Amount, Timestamp)
    account_view.csv         (Account ID, Associated Risk Score, Number of Flagged
                               Transactions, Counterparty Summary)
    feature_importance.csv   (feature, xgboost, lightgbm, random_forest, average)

If those files aren't found next to this script, the sidebar lets you
point at different paths (e.g. if you copied them elsewhere).
"""

import os

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="AML Transaction-Graph Dashboard", layout="wide")

st.title("AML Detection Dashboard - XGBoost / LightGBM / Random Forest Ensemble")
st.caption("Transaction-level ensemble risk scores, account roll-ups, and model drivers.")

with st.sidebar:
    st.header("Data sources")
    txn_path = st.text_input("Transaction view CSV", "transaction_view.csv")
    acct_path = st.text_input("Account view CSV", "account_view.csv")
    importance_path = st.text_input("Feature importance CSV", "feature_importance.csv")
    st.divider()
    risk_threshold = st.slider("Risk score threshold", 0.0, 1.0, 0.5, 0.01)


def _load_csv(path, label):
    if not os.path.exists(path):
        st.warning(f"Couldn't find **{label}** at `{path}`. Update the path in the sidebar, "
                    f"or run the training notebook's final 'export for dashboard' cell first.")
        return None
    return pd.read_csv(path)


tab_txn, tab_acct, tab_model = st.tabs(["Transaction View", "Account View", "Model Insights"])

# --------------------------------------------------------------------------
# Transaction View (PRD 16): Transaction ID, Risk Score, Sender, Receiver,
# Amount, Timestamp.
# --------------------------------------------------------------------------
with tab_txn:
    df_txn = _load_csv(txn_path, "transaction view")
    if df_txn is not None:
        flagged = df_txn[df_txn["Risk Score"] >= risk_threshold]
        c1, c2, c3 = st.columns(3)
        c1.metric("Transactions shown", f"{len(df_txn):,}")
        c2.metric("Flagged at this threshold", f"{len(flagged):,}")
        c3.metric("Flag rate", f"{len(flagged) / max(len(df_txn), 1):.2%}")

        fig = px.histogram(
            df_txn, x="Risk Score", nbins=50,
            title="Distribution of transaction risk scores",
        )
        fig.add_vline(x=risk_threshold, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Highest-risk transactions")
        st.dataframe(
            df_txn.sort_values("Risk Score", ascending=False).head(200),
            use_container_width=True, hide_index=True,
        )

# --------------------------------------------------------------------------
# Account View (PRD 16): Account ID, Associated Risk Score, Number of
# Flagged Transactions, Counterparty Summary.
# --------------------------------------------------------------------------
with tab_acct:
    df_acct = _load_csv(acct_path, "account view")
    if df_acct is not None:
        alerted = df_acct[df_acct["Account Alert"]] if "Account Alert" in df_acct.columns else df_acct.iloc[0:0]
        c1, c2, c3 = st.columns(3)
        c1.metric("Accounts shown", f"{len(df_acct):,}")
        c2.metric("Accounts with an alert", f"{len(alerted):,}")
        c3.metric("Alert rate", f"{len(alerted) / max(len(df_acct), 1):.2%}")

        st.subheader("Highest-risk accounts")
        st.dataframe(
            df_acct.sort_values("Associated Risk Score", ascending=False).head(200),
            use_container_width=True, hide_index=True,
        )

        if "Number of Flagged Transactions" in df_acct.columns:
            top20 = df_acct.sort_values("Associated Risk Score", ascending=False).head(20)
            fig = px.bar(
                top20, x="Account ID", y="Number of Flagged Transactions",
                title="Flagged transaction count - top 20 riskiest accounts",
            )
            st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------------------
# Model Insights (new): which features are actually driving the ensemble's
# score, per base model and on average. Trees give this for free at
# inference time; the GNN's learned embeddings didn't expose an
# equivalent without extra post-hoc explainability work.
# --------------------------------------------------------------------------
with tab_model:
    df_imp = _load_csv(importance_path, "feature importance")
    if df_imp is not None:
        base_model_cols = [c for c in ("xgboost", "lightgbm", "random_forest") if c in df_imp.columns]

        st.subheader("Top feature drivers (gain-based, normalized 0-1)")
        if base_model_cols:
            st.caption(
                "Averaged across the three base learners so a feature that only one "
                "model likes doesn't dominate the ranking. High-cardinality account "
                "identity is deliberately excluded from the model -- see feature_columns.py."
            )
        else:
            st.caption(
                "From a single ExSTraQt model (see exstraqt/model.py) -- there's no "
                "per-base-learner breakdown to average over here, this is that one "
                "model's own gain-based importance. If one feature (e.g. a raw "
                "payment-format code) still dominates this chart by a wide margin, "
                "see README_EXSTRAQT.md's note on that."
            )
        fig = px.bar(
            df_imp.sort_values("average", ascending=True),
            x="average", y="feature", orientation="h",
            title="Feature importance" + (" -- average across base models" if base_model_cols else ""),
        )
        st.plotly_chart(fig, use_container_width=True)

        if base_model_cols:
            st.subheader("Per-model breakdown")
            melted = df_imp.melt(
                id_vars="feature", value_vars=base_model_cols,
                var_name="model", value_name="importance",
            )
            fig2 = px.bar(
                melted, x="feature", y="importance", color="model", barmode="group",
                title="Feature importance by base model",
            )
            fig2.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig2, use_container_width=True)

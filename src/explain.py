from __future__ import annotations

from typing import List, Tuple
import numpy as np
import pandas as pd

try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

def explain_account(
    feature_df: pd.DataFrame,
    account: str,
    top_k: int = 5,
):
    row = feature_df.loc[feature_df["account"] == account]
    if row.empty:
        return []
    row = row.iloc[0]
    numeric_cols = [c for c in feature_df.columns if c != "account"]
    values = row[numeric_cols].astype(float)

    # Heuristic explanation when SHAP is unavailable.
    # Returns the strongest signals by deviation from portfolio median.
    med = feature_df[numeric_cols].median(numeric_only=True)
    mad = (feature_df[numeric_cols].sub(med).abs()).median(numeric_only=True).replace(0, 1e-9)
    z = (values - med) / mad
    ranked = z.abs().sort_values(ascending=False).head(top_k)
    reasons = []
    for feat, score in ranked.items():
        direction = "high" if values[feat] >= med[feat] else "low"
        reasons.append(
            {
                "feature": feat,
                "direction": direction,
                "value": float(values[feat]),
                "median": float(med[feat]),
                "deviation": float(score),
            }
        )
    return reasons

def human_readable_reasons(reasons: list[dict]) -> list[str]:
    texts = []
    mapping = {
        "supervised_probability": "high supervised laundering probability",
        "anomaly_score": "unusual behavior relative to peers",
        "betweenness_centrality": "broker-like position in the network",
        "velocity_7d": "high transaction velocity",
        "relay_score": "relay-like timing pattern",
        "short_cycle_score": "short-cycle participation",
        "incoming_concentration": "incoming amount concentration",
        "outgoing_concentration": "outgoing amount concentration",
        "amount_balance": "large imbalance between inflows and outflows",
        "degree_balance": "imbalance between incoming and outgoing counterparties",
        "max_gap_days": "long inactivity gap followed by activity",
        "local_density": "dense neighborhood structure",
    }
    for r in reasons:
        feat = r["feature"]
        base = mapping.get(feat, feat.replace("_", " "))
        texts.append(f"{base} ({r['direction']} than the portfolio median)")
    return texts


def shap_explain_account(
    feature_df: pd.DataFrame,
    account: str,
    artifacts,
    top_k: int = 7,
):
    if not HAS_SHAP:
        return []

    if artifacts.supervised_model is None:
        return []

    row = feature_df.loc[
        feature_df["account"].astype(str) == str(account)
    ]

    if row.empty:
        return []

    X = row[artifacts.feature_columns]

    try:
        model = artifacts.supervised_model.named_steps["model"]

        explainer = shap.TreeExplainer(model)

        shap_values = explainer.shap_values(X)

        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        vals = shap_values[0]

        importance = pd.DataFrame({
            "feature": artifacts.feature_columns,
            "shap_value": vals,
        })

        importance["abs_shap"] = (
            importance["shap_value"]
            .abs()
        )

        importance = (
            importance
            .sort_values(
                "abs_shap",
                ascending=False,
            )
            .head(top_k)
        )

        return importance.to_dict("records")

    except Exception as e:
        print("SHAP error:", e)
        return []


def human_readable_shap(
    reasons: list[dict]
):
    texts = []

    for r in reasons:

        direction = (
            "increased"
            if r["shap_value"] > 0
            else "decreased"
        )

        texts.append(
            f"{r['feature']} {direction} the model risk score"
        )

    return texts
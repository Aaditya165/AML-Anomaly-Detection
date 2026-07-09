"""
explain.py
----------
Everything SHAP. Nothing related to training lives here.

For AML specifically, SHAP is worth the extra step over
`model.feature_importance()`: an investigator opening a flagged
transaction wants "why was THIS one flagged" (a per-transaction waterfall),
not "which features matter on average across the whole file" (global
importance). Both are provided here -- `summary()` for the global view,
`explain_transaction()` for the per-row view, `dependence()` for "how does
this one feature's effect change across its range, and does it interact
with another feature".

Uses `shap.TreeExplainer`, which reads an xgboost.Booster directly (no
need to re-wrap Model -- pass `model.booster_`).
"""

from typing import List, Optional

import numpy as np
import pandas as pd
import shap
import xgboost as xgb


def build_explainer(booster: xgb.Booster) -> xgb.Booster:
    """
    Native XGBoost TreeSHAP.
    The booster itself is all we need.
    """
    return booster


def shap_values(
    booster: xgb.Booster,
    X: pd.DataFrame,
) -> np.ndarray:
    """
    Native TreeSHAP from XGBoost.

    Returns
    -------
    ndarray of shape (n_rows, n_features)
    """

    dmatrix = xgb.DMatrix(
        X,
        enable_categorical=True,
        feature_names=list(X.columns),
    )

    contribs = booster.predict(
        dmatrix,
        pred_contribs=True,
    )

    # Last column is the bias term.
    return contribs[:, :-1]


def summary(
    booster,
    X: pd.DataFrame,
    max_display: int = 20,
):
    values = shap_values(booster, X)

    importance = (
        np.abs(values)
        .mean(axis=0)
    )

    out = pd.DataFrame({
        "feature": X.columns,
        "mean_abs_shap": importance,
    })

    out = out.sort_values(
        "mean_abs_shap",
        ascending=False,
    ).reset_index(drop=True)

    if max_display:
        out = out.head(max_display)

    return out


def explain_transaction(
    booster,
    X: pd.DataFrame,
    row_index,
    top_n: int = 15,
):
    if row_index in X.index:
        row = X.loc[[row_index]]
    else:
        row = X.iloc[[row_index]]

    values = shap_values(booster, row)[0]

    out = pd.DataFrame({
        "feature": X.columns,
        "feature_value": row.iloc[0].values,
        "shap_value": values,
    })

    out["abs_shap"] = out["shap_value"].abs()

    out = (
        out.sort_values(
            "abs_shap",
            ascending=False,
        )
        .drop(columns="abs_shap")
    )

    if top_n:
        out = out.head(top_n)

    return out


def waterfall_plot_data(explainer: shap.TreeExplainer, X: pd.DataFrame, row_index) -> shap.Explanation:
    """Same row-level explanation as `explain_transaction`, but as a
    `shap.Explanation` object ready for `shap.plots.waterfall(...)` or
    `shap.plots.force(...)` if you want the actual plot rather than a
    DataFrame (e.g. embedding in dashboard/app.py)."""
    row = X.loc[[row_index]] if row_index in X.index else X.iloc[[row_index]]
    values = shap_values(explainer, row)[0]
    base_value = explainer.expected_value
    return shap.Explanation(
        values=values, base_values=base_value, data=row.iloc[0].to_numpy(), feature_names=list(X.columns),
    )


def dependence(explainer: shap.TreeExplainer, X: pd.DataFrame, feature: str) -> pd.DataFrame:
    """How one feature's SHAP contribution varies across its own value
    range -- e.g. call with feature="payment_format_idx" to see directly
    whether specific codes are driving large positive contributions on
    their own, independent of everything else (useful for exactly the
    kind of single-feature-dominance question raised in README.md)."""
    values = shap_values(explainer, X)
    col_idx = list(X.columns).index(feature)
    return pd.DataFrame({
        "feature_value": X[feature].to_numpy(),
        "shap_value": values[:, col_idx],
    }).sort_values("feature_value").reset_index(drop=True)

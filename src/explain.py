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


def build_explainer(booster: xgb.Booster) -> shap.TreeExplainer:
    """One explainer, reusable across summary/transaction/dependence calls
    -- building it is the relatively expensive part (walks the trees
    once); computing SHAP values for a given X is comparatively cheap."""
    return shap.TreeExplainer(booster)


def shap_values(explainer: shap.TreeExplainer, X: pd.DataFrame) -> np.ndarray:
    """Raw per-row, per-feature SHAP values, shape (n_rows, n_features).
    Every other function in this module is a view over this array --
    compute it once per X and reuse, rather than recomputing per call."""
    return explainer.shap_values(X)


def summary(explainer: shap.TreeExplainer, X: pd.DataFrame, max_display: int = 20) -> pd.DataFrame:
    """Global view: mean |SHAP value| per feature, sorted descending --
    the SHAP analogue of model.feature_importance(), but derived from
    actual per-prediction attributions rather than the booster's internal
    gain accounting (the two usually agree on the big picture, but SHAP's
    ranking is the one that's actually consistent with what
    explain_transaction() shows for any individual row)."""
    values = shap_values(explainer, X)
    mean_abs = np.abs(values).mean(axis=0)
    out = pd.DataFrame({"feature": X.columns, "mean_abs_shap": mean_abs})
    out = out.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    return out.head(max_display) if max_display else out


def explain_transaction(
    explainer: shap.TreeExplainer, X: pd.DataFrame, row_index, top_n: int = 15
) -> pd.DataFrame:
    """Per-transaction explanation ('waterfall' data, not the plot itself
    -- see `waterfall_plot_data` if you want to hand this straight to
    `shap.plots.waterfall`): for one row, the top features pushing the
    score up or down, with their SHAP value and the row's own feature
    value alongside it (so "relay_timing_score = 0.9, contributing +0.31"
    reads the way an investigator would want it to).
    """
    row = X.loc[[row_index]] if row_index in X.index else X.iloc[[row_index]]
    values = shap_values(explainer, row)[0]
    out = pd.DataFrame({
        "feature": X.columns, "feature_value": row.iloc[0].to_numpy(), "shap_value": values,
    })
    out["abs_shap_value"] = out["shap_value"].abs()
    out = out.sort_values("abs_shap_value", ascending=False).drop(columns="abs_shap_value")
    return out.head(top_n) if top_n else out


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

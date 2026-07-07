"""
train.py
--------
train() / predict() / evaluate() / save_model() / load_model(). No feature
engineering lives here -- that's feature_merge.py's job. This file's only
responsibility is: given an already-merged feature frame, fit/score a
Model and report metrics.

Anti-leakage note: `train()` builds ExSTraQt node features ONLY from the
training split (see feature_merge.build_node_feature_table), then reuses
that same table to score validation -- validation accounts are scored
using only what the account graph looked like through the end of
training, never anything from their own period. `evaluate()` documents
the same pattern for a genuine held-out file.
"""

from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    average_precision_score, confusion_matrix, precision_recall_curve,
)

import feature_merge as fm
from model import Model


def chronological_split(df: pd.DataFrame, val_frac: float = 0.15):
    """`df` must already be time-sorted (data_processing.load_and_clean does this)."""
    split_at = int(len(df) * (1 - val_frac))
    return df.iloc[:split_at].reset_index(drop=True), df.iloc[split_at:].reset_index(drop=True)


def find_optimal_threshold(y_true: np.ndarray, probs: np.ndarray) -> tuple:
    precisions, recalls, thresholds = precision_recall_curve(y_true, probs)
    if len(thresholds) == 0:
        return 0.5, 0.0
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = int(np.argmax(f1_scores[:-1]))
    return float(thresholds[best_idx]), float(f1_scores[best_idx])


def evaluate(y_true: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (probs >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "pr_auc": average_precision_score(y_true, probs) if y_true.sum() > 0 else 0.0,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]),
        "threshold": threshold,
    }


def train(
    df: pd.DataFrame,
    val_frac: float = 0.15,
    restrict_to_accounts=None,
    base_numeric_columns: Optional[List[str]] = None,
    base_categorical_columns: Optional[List[str]] = None,
    source_col: str = "Sender Account",
    target_col: str = "Receiver Account",
    amount_col: str = "Amount Paid",
    timestamp_col: str = "Timestamp",
    label_col: str = "label",
    use_cache: bool = True,
) -> dict:
    """
    `df`: output of data_processing.load_and_clean (optionally already run
    through feature_engineering.engineer_all_features).
    `base_numeric_columns`/`base_categorical_columns`: pass
    base_feature_columns.NUMERIC_FEATURE_COLUMNS / CATEGORICAL_FEATURE_COLUMNS
    to train on BASE + ExSTraQt features together (recommended). Leave as
    None for an ExSTraQt-only ablation.

    Returns: model, node_features, threshold, metrics, val_probs,
    df_val_joined, feature_columns.
    """
    df_train, df_val = chronological_split(df, val_frac=val_frac)

    node_features = fm.build_node_feature_table(
        df_train, source_col, target_col, amount_col, timestamp_col,
        restrict_to_accounts=restrict_to_accounts, cache_key="train", use_cache=use_cache,
    )

    train_joined = fm.join_node_features_to_transactions(df_train, node_features, source_col, target_col)
    val_joined = fm.join_node_features_to_transactions(df_val, node_features, source_col, target_col)

    numeric_cols, categorical_cols = fm.all_feature_columns(
        list(node_features.columns), base_numeric_columns or [], base_categorical_columns or [],
    )
    X_train = fm.prepare_feature_frame(train_joined, numeric_cols, categorical_cols)
    X_val = fm.prepare_feature_frame(val_joined, numeric_cols, categorical_cols)
    y_train, y_val = train_joined[label_col].to_numpy(), val_joined[label_col].to_numpy()

    model = Model().fit(X_train, y_train, X_val, y_val)
    val_probs = model.predict_proba(X_val)
    threshold, _ = find_optimal_threshold(y_val, val_probs)
    metrics = evaluate(y_val, val_probs, threshold=threshold)

    return {
        "model": model, "node_features": node_features, "threshold": threshold, "metrics": metrics,
        "val_probs": val_probs, "df_val_joined": val_joined,
        "feature_columns": {"numeric": numeric_cols, "categorical": categorical_cols},
    }


def predict(model: Model, df_joined: pd.DataFrame, feature_columns: dict) -> np.ndarray:
    X = fm.prepare_feature_frame(df_joined, feature_columns["numeric"], feature_columns["categorical"])
    return model.predict_proba(X)


def score_holdout(
    model: Model, df_train_plus_val: pd.DataFrame, df_holdout: pd.DataFrame,
    feature_columns: dict, threshold: float, restrict_to_accounts=None,
    source_col: str = "Sender Account", target_col: str = "Receiver Account",
    amount_col: str = "Amount Paid", timestamp_col: str = "Timestamp", label_col: str = "label",
    use_cache: bool = True,
) -> dict:
    """Scores a genuine held-out file, rebuilding node features from
    `df_train_plus_val` first so the held-out transactions never
    influence their own community/flow features (same pattern `train()`
    uses for validation, just one split further out)."""
    node_features = fm.build_node_feature_table(
        df_train_plus_val, source_col, target_col, amount_col, timestamp_col,
        restrict_to_accounts=restrict_to_accounts, cache_key="train_plus_val", use_cache=use_cache,
    )
    holdout_joined = fm.join_node_features_to_transactions(df_holdout, node_features, source_col, target_col)
    probs = predict(model, holdout_joined, feature_columns)
    metrics = evaluate(holdout_joined[label_col].to_numpy(), probs, threshold=threshold)
    return {"node_features": node_features, "probs": probs, "metrics": metrics, "df_holdout_joined": holdout_joined}


def save_model(model: Model, path: str):
    model.save(path)


def load_model(path: str) -> Model:
    return Model.load(path)

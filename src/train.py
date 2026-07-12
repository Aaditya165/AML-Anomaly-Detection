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

import gc
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    average_precision_score, confusion_matrix, precision_recall_curve,
)
from pathlib import Path
from . import feature_merge as fm
from .model import Model


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
    graph_cache_dir:Path,
    community_cache_dir:Path,
    feature_cache_dir:Path,
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
    return_val_joined: bool = False,
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
        graph_cache_dir, community_cache_dir, feature_cache_dir,
        df_train, source_col, target_col, amount_col, timestamp_col,
        restrict_to_accounts=restrict_to_accounts, cache_key="train", use_cache=use_cache,
    )

    base_numeric = base_numeric_columns or []
    base_categorical = base_categorical_columns or []

    # Build X_train DIRECTLY -- no fat train_joined intermediate (see
    # build_model_matrix). y from df_train before it's needed elsewhere.
    X_train = fm.build_model_matrix(df_train, node_features, base_numeric, base_categorical,
                                    source_col, target_col)
    y_train = df_train[label_col].to_numpy()

    X_val = fm.build_model_matrix(df_val, node_features, base_numeric, base_categorical,
                                  source_col, target_col)
    y_val = df_val[label_col].to_numpy()

    numeric_cols, categorical_cols = fm.all_feature_columns(
        list(node_features.columns), base_numeric, base_categorical,
    )

    # df_train / df_val are dead once the matrices + labels exist -- free
    # them BEFORE fit() (which makes its own DMatrix copies of X_train/
    # X_val). If df_val_joined is requested later, it's rebuilt from a
    # kept slice; otherwise df_val goes too. df_train is never needed again.
    keep_df_val = df_val if return_val_joined else None
    del df_train, df_val
    gc.collect()

    model = Model().fit(X_train, y_train, X_val, y_val)
    del X_train
    gc.collect()

    val_probs = model.predict_proba(X_val)
    del X_val
    gc.collect()

    threshold, _ = find_optimal_threshold(y_val, val_probs)
    metrics = evaluate(y_val, val_probs, threshold=threshold)

    if return_val_joined:
        df_val_joined = fm.join_node_features_to_transactions(keep_df_val, node_features, source_col, target_col)
    else:
        df_val_joined = None

    return {
        "model": model, "node_features": node_features, "threshold": threshold, "metrics": metrics,
        "val_probs": val_probs, "df_val_joined": df_val_joined,
        "feature_columns": {
            "numeric": numeric_cols, "categorical": categorical_cols,
            # RAW base-only lists (pre exq-expansion) -- score_holdout needs
            # these to call build_model_matrix directly instead of the fat
            # join+prepare path. "numeric"/"categorical" above stay as the
            # combined lists prepare_feature_frame expects, for anything
            # still using that path.
            "base_numeric": base_numeric, "base_categorical": base_categorical,
        },
    }


def predict(model: Model, df_joined: pd.DataFrame, feature_columns: dict) -> np.ndarray:
    X = fm.prepare_feature_frame(df_joined, feature_columns["numeric"], feature_columns["categorical"])
    return model.predict_proba(X)


def score_holdout(
    graph_cache_dir: Path,
    community_cache_dir: Path,
    feature_cache_dir: Path,
    model: Model, df_train_plus_val: pd.DataFrame, df_holdout: pd.DataFrame,
    feature_columns: dict, threshold: float, restrict_to_accounts=None,
    source_col: str = "Sender Account", target_col: str = "Receiver Account",
    amount_col: str = "Amount Paid", timestamp_col: str = "Timestamp", label_col: str = "label",
    use_cache: bool = True,
    build_features_from_holdout: bool = False,
    score_batch_size: Optional[int] = 500_000,
) -> dict:
    """Scores a genuine held-out file.

    Two regimes, controlled by `build_features_from_holdout`:

    * `False` (default): node features are built from `df_train_plus_val`
      and joined onto `df_holdout`. Correct when the holdout is a LATER
      TIME SLICE OF THE SAME ACCOUNT POPULATION (accounts overlap heavily),
      because it prevents the holdout's own transactions from leaking into
      their features.

    * `True`: node features are built from `df_holdout` ITSELF. Correct when
      the holdout is a DIFFERENT ACCOUNT POPULATION (e.g. IBM LI scored by a
      model trained on HI -- the two files share ~1% of accounts). In that
      case there is no leakage risk (disjoint accounts), and building from
      train+val instead would leave the vast majority of holdout
      transactions with all their ExSTraQt features zero-filled, since
      their accounts never appear in the train+val node table.
      `df_train_plus_val` is ignored in this regime.

    If you're unsure which applies, check account overlap between the two
    files first (Jaccard on the union of sender+receiver accounts). Low
    overlap -> use `build_features_from_holdout=True`.

    MEMORY NOTE: builds the model matrix X DIRECTLY via
    `feature_merge.build_model_matrix`, the same way `train()` does --
    never materializes the fat "full df_holdout + ~270 exq columns" frame
    that `join_node_features_to_transactions` + `prepare_feature_frame`
    would. Only requires `feature_columns["base_numeric"]` /
    `["base_categorical"]` (added to what `train()` returns) -- if you're
    passing an older `feature_columns` dict that only has "numeric"/
    "categorical", this falls back to the old (fatter, slower) path
    automatically, with a printed warning.

    Returns `df_holdout_view` instead of a fat `df_holdout_joined`: just
    the 5 columns aggregation.py's `build_transaction_view`/
    `aggregate_accounts` actually read (txn_id, Sender Account,
    Receiver Account, Amount Paid, Timestamp) -- not the full engineered
    frame. `df_holdout_joined` is kept as an alias to the same slim frame
    for backward compatibility with existing notebook cells.
    """
    feature_source_df = df_holdout if build_features_from_holdout else df_train_plus_val
    node_cache_key = "holdout_selfbuilt" if build_features_from_holdout else "train_plus_val"
    node_features = fm.build_node_feature_table(
        graph_cache_dir, community_cache_dir, feature_cache_dir,
        feature_source_df, source_col, target_col, amount_col, timestamp_col,
        restrict_to_accounts=restrict_to_accounts, cache_key=node_cache_key, use_cache=use_cache,
    )

    if "base_numeric" in feature_columns and "base_categorical" in feature_columns:
        # CHUNKED SCORING. Building X for the whole holdout at once needs the
        # full matrix AND XGBoost's internal DMatrix copy of it alive at the
        # same time -- at LI scale that's ~8.4 GB + ~8.4 GB on top of df_holdout,
        # which is the OOM. Scoring in row-chunks means only ONE chunk's matrix
        # (+ its DMatrix) exists at a time; we keep just the float32 probability
        # array, which is ~28 MB for 7M rows. Predictions are row-independent,
        # so this is numerically identical to scoring in one shot -- purely a
        # memory optimization. `score_batch_size=None` restores the old
        # all-at-once behaviour.
        n = len(df_holdout)
        if score_batch_size is None or score_batch_size >= n:
            X_holdout = fm.build_model_matrix(
                df_holdout, node_features,
                feature_columns["base_numeric"], feature_columns["base_categorical"],
                source_col, target_col,
            )
            probs = model.predict_proba(X_holdout)
            del X_holdout
            gc.collect()
        else:
            probs = np.empty(n, dtype=np.float32)
            for start in range(0, n, score_batch_size):
                end = min(start + score_batch_size, n)
                chunk = df_holdout.iloc[start:end]
                X_chunk = fm.build_model_matrix(
                    chunk, node_features,
                    feature_columns["base_numeric"], feature_columns["base_categorical"],
                    source_col, target_col,
                )
                probs[start:end] = model.predict_proba(X_chunk)
                del X_chunk, chunk
                gc.collect()
                print(f"  scored {end:,}/{n:,} rows", end="\r")
            print()
    else:
        print("[score_holdout] feature_columns missing 'base_numeric'/'base_categorical' "
              "(from an older train() run) -- falling back to the fatter join+prepare path. "
              "Re-run train() to get the leaner one.")
        holdout_joined = fm.join_node_features_to_transactions(df_holdout, node_features, source_col, target_col)
        probs = predict(model, holdout_joined, feature_columns)
        del holdout_joined
        gc.collect()

    metrics = evaluate(df_holdout[label_col].to_numpy(), probs, threshold=threshold)

    # Slim view for aggregation.py -- only what build_transaction_view /
    # aggregate_accounts actually read. NOT the fat exq-joined frame.
    view_cols = [c for c in ["txn_id", source_col, target_col, amount_col, timestamp_col] if c in df_holdout.columns]
    df_holdout_view = df_holdout.loc[:, view_cols].copy()

    return {
        "node_features": node_features, "probs": probs, "metrics": metrics,
        "df_holdout_view": df_holdout_view,
        "df_holdout_joined": df_holdout_view,  # alias, same slim frame -- see docstring
    }


def save_model(model: Model, path: str):
    model.save(path)


def load_model(path: str) -> Model:
    return Model.load(path)


def save_run_artifacts(model_cache_dir, model: Model, threshold: float, feature_columns: dict):
    """Persist the THREE small things the holdout / SHAP cells need but that
    a kernel restart (e.g. after an OOM) would otherwise wipe: the model,
    the tuned decision threshold, and the feature-column lists.

    These are tiny (a few MB total) -- this is about surviving a restart,
    NOT about reducing peak memory. Do NOT cache the fat frames
    (df_val_joined etc.) here: those should be `del`'d after the SHAP cell,
    not reloaded, since reloading them during the holdout step would put
    the memory spike right back. See the holdout-cell notes.
    """
    from pathlib import Path
    import json
    model_cache_dir = Path(model_cache_dir)
    model_cache_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(model_cache_dir / "exstraqt_model.json"))
    with open(model_cache_dir / "run_meta.json", "w") as fl:
        json.dump({"threshold": float(threshold), "feature_columns": feature_columns}, fl)


def load_run_artifacts(model_cache_dir):
    """Reload what save_run_artifacts wrote. Returns (model, threshold,
    feature_columns) -- lets the holdout / SHAP cells run in a fresh kernel
    without re-running training."""
    from pathlib import Path
    import json
    model_cache_dir = Path(model_cache_dir)
    model = Model.load(str(model_cache_dir / "exstraqt_model.json"))
    with open(model_cache_dir / "run_meta.json") as fl:
        meta = json.load(fl)
    return model, meta["threshold"], meta["feature_columns"]
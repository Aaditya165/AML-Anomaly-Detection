"""
train.py
--------
Tabular training pipeline replacing the neighbor-sampled GNN loop.

    05. Model Training (total)
    06. Model Inference
    07. Account Risk Aggregation   (unchanged, still in aggregation.py)

No epochs, no mini-batches, no ReduceLROnPlateau: XGBoost/LightGBM do
their own internal boosting-round early stopping against a validation
set, and Random Forest is a single non-iterative fit. What used to be
"training a model" (many GPU/CPU epochs over neighbor-sampled subgraphs)
is now "fit three tree ensembles once" -- this is the biggest single
source of the speedup, not just the architecture change.
"""

import time

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    average_precision_score, confusion_matrix, precision_recall_curve,
)

from .model import AMLEnsemble


def chronological_split(df: pd.DataFrame, val_frac: float = 0.15):
    """`df` must already be time-sorted (it is -- data_processing.load_and_clean
    sorts by Timestamp and assigns txn_id in that order). The last
    `val_frac` becomes validation, same anti-leakage reasoning as the
    GNN version: a random split on a temporal money-flow dataset lets the
    model implicitly "see the future" through account-history features
    computed relative to nearby rows.
    """
    split_at = int(len(df) * (1 - val_frac))
    df_train = df.iloc[:split_at].reset_index(drop=True)
    df_val = df.iloc[split_at:].reset_index(drop=True)
    return df_train, df_val


def subsample_negatives(df: pd.DataFrame, neg_per_pos: float = None,
                         label_col: str = "label", seed: int = 0) -> pd.DataFrame:
    """
    Optional. Trees scale to millions of rows far better than
    neighbor-sampled GNN batches do, so unlike the old pipeline this is
    OFF by default (`neg_per_pos=None` = use every row). Kept available
    for constrained-compute environments or extremely large files: keeps
    every positive (laundering) row plus a random `neg_per_pos` negatives
    per positive.
    """
    if neg_per_pos is None:
        return df
    rng = np.random.default_rng(seed)
    pos = df[df[label_col] == 1]
    neg = df[df[label_col] == 0]
    n_keep_neg = min(len(neg), int(len(pos) * neg_per_pos))
    keep_neg = neg.sample(n=n_keep_neg, random_state=seed)
    out = pd.concat([pos, keep_neg]).sample(frac=1.0, random_state=seed)
    return out.reset_index(drop=True)


def find_optimal_threshold(y_true: np.ndarray, probs: np.ndarray) -> tuple:
    """Threshold that maximizes F1 on the given (val) set. `thresholds` from
    sklearn's precision_recall_curve has one fewer element than
    precisions/recalls -- slice consistently to avoid an index mismatch."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, probs)
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = int(np.argmax(f1_scores[:-1])) if len(thresholds) else 0
    if len(thresholds) == 0:
        return 0.5, 0.0
    return float(thresholds[best_idx]), float(f1_scores[best_idx])


def compute_classification_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (probs >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "pr_auc": average_precision_score(y_true, probs) if y_true.sum() > 0 else 0.0,
        "confusion_matrix": cm,
        "threshold": threshold,
    }


def print_metrics_report(metrics: dict, title: str = "TEST SET METRICS"):
    print("\n" + "=" * 60)
    print(title.center(60))
    print("=" * 60)
    for k in ["accuracy", "precision", "recall", "f1", "pr_auc"]:
        print(f"{k:<12s}: {metrics[k]:.4f}")
    print("Confusion matrix [rows=true, cols=pred] (0=legit, 1=laundering):")
    print(metrics["confusion_matrix"])
    print("=" * 60 + "\n")


def train_pipeline(
    df_train_full: pd.DataFrame,
    val_frac: float = 0.15,
    neg_per_pos: float = None,
    checkpoint_path=None,
):
    """
    End-to-end: chronological split -> (optional) negative subsampling ->
    fit ensemble (with internal early stopping against val) -> pick the
    F1-optimal decision threshold on val -> save.

    Returns (ensemble, optimal_threshold, df_val, val_probs) -- the val
    set + val probs are returned too so the caller can inspect the
    val-set PR curve, exactly like the original notebook's threshold cell
    intended (that cell referenced an undefined `val_probs`; this fixes it
    by construction instead of requiring a second, easy-to-forget
    inference pass on val).
    """
    t0 = time.time()

    df_train, df_val = chronological_split(df_train_full, val_frac=val_frac)

    if neg_per_pos is not None:
        n_before = len(df_train)
        df_train = subsample_negatives(df_train, neg_per_pos=neg_per_pos)
        print(f"neg_per_pos={neg_per_pos}: training rows {n_before:,} -> {len(df_train):,} "
              f"({n_before / max(len(df_train), 1):.1f}x fewer rows)")

    ensemble = AMLEnsemble()
    ensemble, val_probs = ensemble.fit(df_train, df_val)

    optimal_threshold, best_f1 = find_optimal_threshold(df_val["label"].values, val_probs)
    print(f"Optimal threshold (from val set): {optimal_threshold:.4f}   val F1: {best_f1:.4f}")

    if checkpoint_path is not None:
        ensemble.save(checkpoint_path)
        print(f"Saved ensemble to {checkpoint_path}")

    print("05. Model Training (total): ", time.time() - t0)
    return ensemble, optimal_threshold, df_val, val_probs


def run_inference(ensemble: AMLEnsemble, df: pd.DataFrame, batch_size: int = None) -> np.ndarray:
    """
    Full-dataset scoring pass (PRD Section 14). No graph/neighbor sampling
    needed -- `batch_size` is purely an optional memory-bounding knob for
    very large files, not a correctness requirement like it was for the
    GNN's mini-batches.
    """
    t0 = time.time()
    if batch_size is None:
        probs = ensemble.predict_proba(df)
    else:
        probs = np.empty(len(df), dtype=np.float32)
        for start in range(0, len(df), batch_size):
            end = min(start + batch_size, len(df))
            probs[start:end] = ensemble.predict_proba(df.iloc[start:end])
    print("06. Model Inference: ", time.time() - t0)
    return probs


if __name__ == "__main__":
    # Smoke test with the same synthetic shape used in model.py's self-test.
    rng = np.random.default_rng(0)
    from feature_columns import NUMERIC_FEATURE_COLUMNS

    n = 30_000
    df = pd.DataFrame({c: rng.normal(size=n) for c in NUMERIC_FEATURE_COLUMNS})
    df["from_bank_idx"] = rng.integers(0, 50, size=n)
    df["to_bank_idx"] = rng.integers(0, 50, size=n)
    df["payment_format_idx"] = rng.integers(0, 6, size=n)
    df["payment_currency_idx"] = rng.integers(0, 10, size=n)
    df["receiving_currency_idx"] = rng.integers(0, 10, size=n)
    signal = df["relay_timing_score"] + df["short_cycle_participation"] * 2 - df["sender_entropy"]
    prob = 1 / (1 + np.exp(-(signal - signal.mean())))
    df["label"] = (rng.random(n) < (prob * 0.05)).astype(int)

    ens, thresh, df_val, val_probs = train_pipeline(df, checkpoint_path="/tmp/train_smoketest.joblib")
    test_probs = run_inference(ens, df_val)
    metrics = compute_classification_metrics(df_val["label"].values, test_probs, threshold=thresh)
    print_metrics_report(metrics, title="SMOKETEST VAL METRICS")
    print("OK")

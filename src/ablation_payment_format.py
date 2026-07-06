"""
ablation_payment_format.py
---------------------------
Standalone check for the feature-importance chart showing payment_format_idx
as the single dominant feature (driven by ACH's ~0.84% laundering rate vs.
0.02-0.05% for every other channel -- confirmed via
`pd.crosstab(df_train["Payment Format"], df_train["label"], normalize="index")`).

Question this answers: how much of the ensemble's PR-AUC actually comes
from that one feature, vs. the other 37 engineered features standing on
their own?

Does NOT modify feature_columns.py or model.py on disk -- it monkey-patches
the module-level feature-list globals that model.py's AMLEnsemble.fit()
reads at call time, for the duration of this script only. Your main
pipeline files are untouched; this is purely a side experiment.

Usage (run after df_train has been engineered, e.g. at the bottom of the
notebook's feature-engineering cell, or as a separate script):

    from ablation_payment_format import run_ablation
    results = run_ablation(df_train)
"""

import numpy as np
import pandas as pd

from . import feature_columns
from . import model as model_module
from .model import AMLEnsemble
from .train import chronological_split, find_optimal_threshold, compute_classification_metrics, print_metrics_report


def _patch_feature_columns(exclude: list):
    """Point model.py's globals at a feature list with `exclude` removed,
    without touching feature_columns.py or model.py on disk."""
    numeric = [c for c in feature_columns.NUMERIC_FEATURE_COLUMNS if c not in exclude]
    categorical = [c for c in feature_columns.CATEGORICAL_FEATURE_COLUMNS if c not in exclude]
    all_cols = numeric + categorical

    def _prepare(df):
        X = df[all_cols].copy()
        for col in categorical:
            X[col] = X[col].astype("category")
        return X

    model_module.NUMERIC_FEATURE_COLUMNS = numeric
    model_module.CATEGORICAL_FEATURE_COLUMNS = categorical
    model_module.ALL_FEATURE_COLUMNS = all_cols
    model_module.prepare_feature_frame = _prepare


def run_ablation(df_train_full: pd.DataFrame, val_frac: float = 0.15) -> dict:
    df_train, df_val = chronological_split(df_train_full, val_frac=val_frac)

    results = {}
    for label, exclude in [
        ("full_feature_set", []),
        ("without_payment_format", ["payment_format_idx"]),
    ]:
        print(f"\n{'=' * 60}\n{label}\n{'=' * 60}")
        _patch_feature_columns(exclude)

        ensemble = AMLEnsemble()
        ensemble, val_probs = ensemble.fit(df_train, df_val)

        threshold, best_f1 = find_optimal_threshold(df_val["label"].values, val_probs)
        metrics = compute_classification_metrics(df_val["label"].values, val_probs, threshold=threshold)
        print_metrics_report(metrics, title=label.upper())
        results[label] = metrics

    print("\n" + "=" * 60)
    print("ABLATION SUMMARY")
    print("=" * 60)
    for label, m in results.items():
        print(f"{label:<25s}  PR-AUC={m['pr_auc']:.4f}  F1={m['f1']:.4f}  "
              f"precision={m['precision']:.4f}  recall={m['recall']:.4f}")

    full_pr_auc = results["full_feature_set"]["pr_auc"]
    without_pr_auc = results["without_payment_format"]["pr_auc"]
    drop = full_pr_auc - without_pr_auc
    pct = (drop / full_pr_auc) if full_pr_auc > 0 else float("nan")
    print(f"\nPR-AUC drop from removing payment_format_idx: {drop:.4f} "
          f"({pct:.1%} of full-model PR-AUC)")
    print(
        "Interpretation: a small drop means the other 37 features carry the "
        "model on their own and payment_format_idx is a helpful but non-load-"
        "bearing signal. A large drop means the model leans heavily on "
        "payment-channel selection specifically -- worth knowing before "
        "trusting this model against real-world laundering behavior that may "
        "not be as channel-concentrated as this dataset's ACH pattern."
    )
    return results


if __name__ == "__main__":
    # Smoke test with synthetic data shaped like the real pipeline's engineered
    # output, deliberately injecting an ACH-like format-correlated signal so
    # the ablation logic itself can be verified end to end.
    rng = np.random.default_rng(0)
    n = 40_000
    df = pd.DataFrame({c: rng.normal(size=n) for c in feature_columns.NUMERIC_FEATURE_COLUMNS})
    df["from_bank_idx"] = rng.integers(0, 50, size=n)
    df["to_bank_idx"] = rng.integers(0, 50, size=n)
    df["payment_format_idx"] = rng.integers(0, 5, size=n)  # pretend 0 == "ACH"
    df["payment_currency_idx"] = rng.integers(0, 10, size=n)
    df["receiving_currency_idx"] = rng.integers(0, 10, size=n)

    base_rate = 0.001
    format_multiplier = np.where(df["payment_format_idx"] == 0, 20.0, 1.0)
    prob = base_rate * format_multiplier
    df["label"] = (rng.random(n) < prob).astype(int)

    print("overall laundering rate:", df["label"].mean())
    print(pd.crosstab(df["payment_format_idx"], df["label"], normalize="index"))

    run_ablation(df)
    print("OK")

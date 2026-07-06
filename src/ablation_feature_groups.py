"""
ablation_feature_groups.py
----------------------------
Follow-up to ablation_payment_format.py. That script showed payment_format_idx
alone accounts for ~43% of ensemble PR-AUC. This script asks the next
question: of the OTHER 37 features, which engineered groups are actually
doing real work, vs. along for the ride?

Groups tested (each removed one at a time, full feature set otherwise):

    topology            : predecessor_count, successor_count, fan_in, fan_out,
                           relay_timing_score, continues_existing_chain,
                           sender_recently_receiver, short_cycle_participation
                           -- the pure graph-STRUCTURE features (require the
                           full NetworKit graph build to compute).

    pair_history         : time_since_prev_outgoing_sender/incoming_receiver,
                           time_since_last_pair_transfer, recency_pair_score,
                           pair_prior_txn_count, pair_total/mean/std/max_amount_prior,
                           pair_repeated_exact_amount_count -- sender/receiver
                           PAIR-level history (also needs the graph to look up
                           prior transactions between the same two accounts,
                           but describes a relationship, not topology).

    account_behavioral   : sender_/receiver_ hist_outflow_total, hist_inflow_total,
                           unique_counterparties, entropy, concentration,
                           degree_balance, velocity_1d/10d -- account-level
                           behavioral aggregates.

    basic_attributes     : log_amount_paid, log_amount_received, same_bank_flag,
                           hour_of_day, day_of_week -- doesn't need the graph
                           at all, pure per-transaction attributes. Included as
                           a reference point: if removing this "cheap" group
                           hurts MORE than removing an "expensive" graph group,
                           that's a direct signal the graph construction cost
                           isn't earning its keep relative to trivial features.

    payment_format       : payment_format_idx alone, included again here as a
                           reference row so all groups are visible in one table
                           alongside the original ablation's result.

Does NOT modify feature_columns.py or model.py on disk -- same monkey-patch
approach as ablation_payment_format.py.

Usage:
    from ablation_feature_groups import run_grouped_ablation
    results = run_grouped_ablation(df_train)

Each group's result is saved to `ablation_results/<group>.json` as soon as
that group finishes training, so a Kaggle disconnect only costs you the
group in progress, not the whole run. Re-running `run_grouped_ablation`
skips any group that already has a saved result file unless you pass
`force_rerun=True`.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import feature_columns
from . import model as model_module
from .model import AMLEnsemble
from .train import chronological_split, find_optimal_threshold, compute_classification_metrics, print_metrics_report

RESULTS_DIR = Path("ablation_results")

FEATURE_GROUPS = {
    "topology": [
        "predecessor_count", "successor_count", "fan_in", "fan_out",
        "relay_timing_score", "continues_existing_chain",
        "sender_recently_receiver", "short_cycle_participation",
    ],
    "pair_history": [
        "time_since_prev_outgoing_sender", "time_since_prev_incoming_receiver",
        "time_since_last_pair_transfer", "recency_pair_score",
        "pair_prior_txn_count", "pair_total_amount_prior", "pair_mean_amount_prior",
        "pair_std_amount_prior", "pair_max_amount_prior", "pair_repeated_exact_amount_count",
    ],
    "account_behavioral": [
        "sender_hist_outflow_total", "sender_hist_inflow_total",
        "sender_unique_counterparties", "sender_entropy", "sender_concentration",
        "sender_degree_balance", "sender_velocity_1d", "sender_velocity_10d",
        "receiver_hist_outflow_total", "receiver_hist_inflow_total",
        "receiver_unique_counterparties", "receiver_entropy", "receiver_concentration",
        "receiver_degree_balance", "receiver_velocity_1d", "receiver_velocity_10d",
    ],
    "basic_attributes": [
        "log_amount_paid", "log_amount_received", "same_bank_flag",
        "hour_of_day", "day_of_week",
    ],
    "payment_format": [
        "payment_format_idx",
    ],
}


def _patch_feature_columns(exclude: list):
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


def _result_path(label: str) -> Path:
    return RESULTS_DIR / f"{label}.json"


def _save_result(label: str, metrics: dict) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    serializable = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in metrics.items()}
    with open(_result_path(label), "w") as f:
        json.dump(serializable, f, indent=2)


def _load_result(label: str) -> dict:
    with open(_result_path(label)) as f:
        return json.load(f)


def run_grouped_ablation(df_train_full: pd.DataFrame, val_frac: float = 0.15,
                          force_rerun: bool = False) -> dict:
    df_train, df_val = chronological_split(df_train_full, val_frac=val_frac)

    configs = [("full_feature_set", [])] + list(FEATURE_GROUPS.items())

    results = {}
    for label, exclude in configs:
        if not force_rerun and _result_path(label).exists():
            print(f"[skip] {label}: found cached result at {_result_path(label)}")
            results[label] = _load_result(label)
            continue

        print(f"\n{'=' * 60}\n{label}  (excluding: {exclude if exclude else 'nothing'})\n{'=' * 60}")
        _patch_feature_columns(exclude)

        ensemble = AMLEnsemble()
        ensemble, val_probs = ensemble.fit(df_train, df_val)

        threshold, best_f1 = find_optimal_threshold(df_val["label"].values, val_probs)
        metrics = compute_classification_metrics(df_val["label"].values, val_probs, threshold=threshold)
        print_metrics_report(metrics, title=label.upper())

        _save_result(label, metrics)
        results[label] = metrics

    full_pr_auc = results["full_feature_set"]["pr_auc"]

    print("\n" + "=" * 60)
    print("GROUPED ABLATION SUMMARY")
    print("=" * 60)
    rows = []
    for label, m in results.items():
        if label == "full_feature_set":
            continue
        drop = full_pr_auc - m["pr_auc"]
        pct = (drop / full_pr_auc) if full_pr_auc > 0 else float("nan")
        rows.append((label, m["pr_auc"], m["f1"], m["precision"], m["recall"], drop, pct))

    rows.sort(key=lambda r: r[5], reverse=True)  # biggest drop (most load-bearing) first

    print(f"{'group':<22s} {'PR-AUC':>8s} {'F1':>8s} {'precision':>10s} {'recall':>8s} {'PR-AUC drop':>12s} {'% of full':>10s}")
    print(f"{'(full_feature_set)':<22s} {full_pr_auc:>8.4f}")
    for label, pr_auc, f1, prec, rec, drop, pct in rows:
        print(f"{label:<22s} {pr_auc:>8.4f} {f1:>8.4f} {prec:>10.4f} {rec:>8.4f} {drop:>12.4f} {pct:>9.1%}")

    print(
        "\nInterpretation: groups with the LARGEST PR-AUC drop when removed are "
        "the most load-bearing -- the model actually depends on them. Groups "
        "with a small/near-zero drop are present but not doing much real work; "
        "removing them costs little, meaning the model was finding equivalent "
        "signal elsewhere (or they were never carrying much signal to begin "
        "with). Compare 'topology' specifically against 'basic_attributes' -- "
        "if the cheap, graph-free basic_attributes group matters MORE than the "
        "expensive graph-topology group, that's a direct sign the NetworKit "
        "graph construction isn't earning its computational cost relative to "
        "trivial per-transaction fields."
    )
    return results


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 40_000
    df = pd.DataFrame({c: rng.normal(size=n) for c in feature_columns.NUMERIC_FEATURE_COLUMNS})
    df["from_bank_idx"] = rng.integers(0, 50, size=n)
    df["to_bank_idx"] = rng.integers(0, 50, size=n)
    df["payment_format_idx"] = rng.integers(0, 5, size=n)
    df["payment_currency_idx"] = rng.integers(0, 10, size=n)
    df["receiving_currency_idx"] = rng.integers(0, 10, size=n)

    # inject signal concentrated in the "topology" group specifically, to
    # verify the ablation correctly attributes a drop to the right group
    signal = df["short_cycle_participation"] * 3 + df["relay_timing_score"] * 2
    prob = 0.001 * (1 + np.clip(signal, 0, None))
    df["label"] = (rng.random(n) < prob).astype(int)
    print("laundering rate:", df["label"].mean())

    run_grouped_ablation(df, force_rerun=True)
    print("OK")

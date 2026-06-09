from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from .data import load_transactions, train_val_test_split_by_time
from .features import aggregate_labels_to_account, compute_account_features
from .models import ModelArtifacts, evaluate_if_labels_available, score_accounts, train_models

def run_pipeline(
    transaction_path: str | Path,
    temporal_window_days: int = 7,
    random_state: int = 42,
):
    tx = load_transactions(transaction_path)
    train_tx, val_tx, test_tx = train_val_test_split_by_time(tx)

    feat = compute_account_features(tx, temporal_window_days=temporal_window_days)
    labels = aggregate_labels_to_account(tx)

    artifacts = train_models(feat, labels if len(labels) else None, random_state=random_state)
    scored = score_accounts(feat, artifacts)
    metrics = evaluate_if_labels_available(scored, labels if len(labels) else None)

    return {
        "transactions": tx,
        "features": feat,
        "labels": labels,
        "scored": scored.sort_values("risk_score", ascending=False).reset_index(drop=True),
        "metrics": metrics,
        "artifacts": artifacts,
        "splits": {"train": train_tx, "val": val_tx, "test": test_tx},
    }

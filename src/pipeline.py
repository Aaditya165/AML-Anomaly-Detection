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

    train_feat = compute_account_features(
        train_tx, 
        temporal_window_days=temporal_window_days,
    )
    train_labels = aggregate_labels_to_account(train_tx)

    test_feat = compute_account_features(
        test_tx,
        temporal_window_days=temporal_window_days,
    )
    test_labels = aggregate_labels_to_account(test_tx)

    artifacts = train_models(
        train_feat,
        train_labels if len(train_labels) else None,
        random_state=random_state,
    )

    scored = score_accounts(test_feat, artifacts)

    metrics = evaluate_if_labels_available(
        scored, 
        test_labels if len(test_labels) else None,
    )

    return {
        "transactions": tx,
        "features": test_feat,
        "labels": test_labels,
        "scored": scored.sort_values("risk_score", ascending=False).reset_index(drop=True),
        "metrics": metrics,
        "artifacts": artifacts,
        "splits": {"train": train_tx, "val": val_tx, "test": test_tx},
    }

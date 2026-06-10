from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
import numpy as np

from .utils import normalize_colname

EXPECTED_CANONICAL = {
    "Timestamp": "timestamp",
    "From Bank": "from_bank",
    "Account": "from_account",
    "To Bank": "to_bank",
    "Account.1": "to_account",
    "Amount Received": "amount_received",
    "Receiving Currency": "receiving_currency",
    "Amount Paid": "amount_paid",
    "Payment Currency": "payment_currency",
    "Payment Format": "payment_format",
    "Is Laundering": "is_laundering",
}

ALT_EXPECTED = {
    "from bank": "from_bank",
    "from_bank": "from_bank",
    "to bank": "to_bank",
    "to_bank": "to_bank",
    "timestamp": "timestamp",
    "amount received": "amount_received",
    "amount_received": "amount_received",
    "amount paid": "amount_paid",
    "amount_paid": "amount_paid",
    "receiving currency": "receiving_currency",
    "receiving_currency": "receiving_currency",
    "payment currency": "payment_currency",
    "payment_currency": "payment_currency",
    "payment format": "payment_format",
    "payment_format": "payment_format",
    "is laundering": "is_laundering",
    "is_laundering": "is_laundering",
}

def _canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = list(df.columns)
    renamed = {}
    account_seen = 0
    for c in cols:
        norm = normalize_colname(c)
        low = norm.lower()
        if low in ["account", "Account.1"]:
            account_seen += 1
            renamed[c] = "from_account" if account_seen == 1 else "to_account"
        elif low in ALT_EXPECTED:
            renamed[c] = ALT_EXPECTED[low]
        else:
            renamed[c] = low.replace(" ", "_")
    out = df.rename(columns=renamed)
    # If pandas created Account.1 and our rename did not catch it, preserve a standard name.
    if "account_1" in out.columns and "to_account" not in out.columns:
        out = out.rename(columns={"account_1": "to_account"})
    return out

def load_transactions(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = _canonicalize_columns(df)

    if "timestamp" not in df.columns:
        raise ValueError(f"Timestamp column missing in {path}")
    if "from_account" not in df.columns or "to_account" not in df.columns:
        raise ValueError(
            "Could not locate both account columns. Expected duplicate Account columns in raw IBM AML file."
        )

    # normalize timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=False)
    df = df.dropna(subset=["timestamp"]).copy()

    # normalize numeric amounts
    for col in ["amount_received", "amount_paid"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["amount_received"] = df.get("amount_received", pd.Series(index=df.index, dtype=float)).fillna(0.0)
    df["amount_paid"] = df.get("amount_paid", pd.Series(index=df.index, dtype=float)).fillna(df["amount_received"]).fillna(0.0)

    # normalize strings
    for col in ["from_bank", "to_bank", "receiving_currency", "payment_currency", "payment_format", "from_account", "to_account"]:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna("").str.strip()

    if "is_laundering" in df.columns:
        df["is_laundering"] = pd.to_numeric(df["is_laundering"], errors="coerce").fillna(0).astype(int)
    else:
        df["is_laundering"] = np.nan

    df = df.sort_values("timestamp").reset_index(drop=True)
    return df

def train_val_test_split_by_time(df: pd.DataFrame, train_ratio=0.6, val_ratio=0.2):
    n = len(df)
    if n == 0:
        return df, df, df
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()
    return train, val, test

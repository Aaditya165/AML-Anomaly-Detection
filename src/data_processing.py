"""
data_processing.py
-------------------
PRD Section 3/4: "Data Cleaning & Normalization"

Loads the raw IBM-AML-format CSV and produces a clean, typed DataFrame
ready for graph construction. Wrapped in Timer() so load+clean cost is
visible in the final timing report.

Categorical vocabularies (account / bank / payment format / currency) are
fit ONCE on the training file and reused on the test file, with an
explicit "unknown" bucket for any account/bank/etc. that only appears in
test data -- otherwise train and test embedding indices wouldn't even
refer to the same accounts, and a held-out account would crash an
nn.Embedding lookup instead of degrading gracefully.
"""

import numpy as np
import pandas as pd
import time
import re

EXPECTED_CANONICAL = {
    "timestamp": "Timestamp",
    "from bank": "From Bank",
    "account": "Account",
    "to bank": "To Bank",
    "account.1": "Account.1",
    "amount received": "Amount Received",
    "receiving currency": "Receiving Currency",
    "amount paid": "Amount Paid",
    "payment currency": "Payment Currency",
    "payment format": "Payment Format",
    "is laundering": "Is Laundering",
}

# group_name -> (df_column, output_index_column)
CATEGORICAL_GROUPS = {
    "account": [("Sender Account", "sender_idx"), ("Receiver Account", "receiver_idx")],
    "bank": [("From Bank", "from_bank_idx"), ("To Bank", "to_bank_idx")],
    "payment_format": [("Payment Format", "payment_format_idx")],
    "currency": [("Payment Currency", "payment_currency_idx"),
                 ("Receiving Currency", "receiving_currency_idx")],
}

def canonicalize_columns(df):
    renamed = {}

    account_seen = 0

    for c in df.columns:
        norm = normalize_colname(c).lower()

        if norm in ["account", "account.1"]:
            account_seen += 1

            if account_seen == 1:
                renamed[c] = "Account"
            else:
                renamed[c] = "Account.1"

        elif norm in EXPECTED_CANONICAL:
            renamed[c] = EXPECTED_CANONICAL[norm]

    return df.rename(columns=renamed)

def normalize_colname(col):
    col = str(col).strip()
    col = re.sub(r"\s+", " ", col)
    col = col.replace("\ufeff", "")
    return col

def build_vocab(series_list) -> dict:
    """value -> contiguous int code, built from the union of one-or-more columns."""
    values = pd.unique(pd.concat(series_list, ignore_index=True))
    return {v: i for i, v in enumerate(sorted(values, key=str))}


def apply_vocab(series: pd.Series, vocab: dict) -> np.ndarray:
    """Map values to codes; anything unseen at fit-time -> UNK bucket (= len(vocab))."""
    unk = len(vocab)
    return series.map(vocab).fillna(unk).astype(np.int64).values


def load_and_clean(csv_path: str, vocabs: dict = None):
    """
    Read raw CSV -> typed, sorted, de-duplicated DataFrame with a stable
    `txn_id` (row position after sort = node id used everywhere downstream).

    Pass `vocabs=None` for the TRAIN file (fits & returns fresh vocabs).
    Pass the training `vocabs` dict back in for the TEST / inference file
    so categorical codes line up with the trained embedding tables.

    Returns (df, vocabs).
    """
    t0 = time.time()

    df = pd.read_csv(csv_path)
    df = canonicalize_columns(df)

    missing = set(EXPECTED_CANONICAL.values()) - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected IBM-AML columns: {missing}")

    df = df.rename(columns={
        "Account": "Sender Account",
        "Account.1": "Receiver Account",
    })

    df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    # Drop exact duplicate rows and rows with nulls in critical fields.
    df = df.drop_duplicates()
    df = df.dropna(subset=[
        "Timestamp", "Sender Account", "Receiver Account",
        "Amount Paid", "Amount Received",
    ])

    # Basic sanity filters: non-negative amounts, no self-loops.
    df = df[(df["Amount Paid"] > 0) & (df["Amount Received"] > 0)]
    df = df[df["Sender Account"] != df["Receiver Account"]]

    # Stable chronological ordering -> this row position becomes the
    # transaction-node id used by the graph & PyG Data object.
    df = df.sort_values("Timestamp", kind="mergesort").reset_index(drop=True)
    df["txn_id"] = df.index.values

    # Derived numeric fields used heavily downstream.
    df["log_amount_paid"] = np.log1p(df["Amount Paid"])
    df["log_amount_received"] = np.log1p(df["Amount Received"])
    df["same_bank_flag"] = (df["From Bank"] == df["To Bank"]).astype(np.int8)
    df["hour_of_day"] = df["Timestamp"].dt.hour.astype(np.int8)
    df["day_of_week"] = df["Timestamp"].dt.dayofweek.astype(np.int8)

    # Integer-coded categoricals for embedding layers (section 6),
    # sharing one vocab per group (e.g. sender & receiver share the
    # "account" vocab) and fit only on the training file.
    fit_mode = vocabs is None
    if fit_mode:
        vocabs = {}
    for group, cols in CATEGORICAL_GROUPS.items():
        if fit_mode:
            vocabs[group] = build_vocab([df[c] for c, _ in cols])
        for raw_col, out_col in cols:
            df[out_col] = apply_vocab(df[raw_col], vocabs[group])

    df["label"] = df["Is Laundering"].astype(np.int64)

    print("01. Data Loading and Normalization: ", time.time() - t0)
    
    return df, vocabs


def vocab_sizes(vocabs: dict) -> dict:
    """Embedding-table sizes needed by the model (+1 per group for the UNK bucket)."""
    return {
        "n_accounts": len(vocabs["account"]) + 1,
        "n_banks": len(vocabs["bank"]) + 1,
        "n_payment_formats": len(vocabs["payment_format"]) + 1,
        "n_currencies": len(vocabs["currency"]) + 1,
    }


if __name__ == "__main__":
    from synthetic_data import generate_synthetic_transactions
    raw = generate_synthetic_transactions()
    raw.to_csv("HI-Small_Trans_SYNTHETIC.csv", index=False)
    clean, vocabs = load_and_clean("HI-Small_Trans_SYNTHETIC.csv")
    print(clean.shape)
    print(vocab_sizes(vocabs))

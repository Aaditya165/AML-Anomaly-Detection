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



REQUIRED_COLUMNS = [
    "Timestamp", "From Bank", "Account", "To Bank", "Account.1",
    "Amount Received", "Receiving Currency", "Amount Paid",
    "Payment Currency", "Payment Format", "Is Laundering",
]

# group_name -> (df_column, output_index_column)
CATEGORICAL_GROUPS = {
    "account": [("Sender Account", "sender_idx"), ("Receiver Account", "receiver_idx")],
    "bank": [("From Bank", "from_bank_idx"), ("To Bank", "to_bank_idx")],
    "payment_format": [("Payment Format", "payment_format_idx")],
    "currency": [("Payment Currency", "payment_currency_idx"),
                 ("Receiving Currency", "receiving_currency_idx")],
}


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

    DTYPE NOTE: right after reading, repeated-string columns are cast to
    `category` dtype. On the real IBM-AML files this matters a lot more
    than it looks: `Payment Format`/`Payment Currency`/`Receiving Currency`
    each have ~6-16 *unique* values but were costing ~65-70MB per column
    at 4.5M rows storing that handful of values as a full Python string
    object on every single row -- `category` dtype stores the small set of
    uniques once plus a compact int8/int16 code per row instead. `Sender
    Account`/`Receiver Account` get the same treatment (~422k uniques /
    4.5M rows, so ~90% of the repetition is redundant the same way). This
    is on top of (not a replacement for) the CSR adjacency fix in
    `graph_construction.py` -- both were independently large enough to be
    the dominant cost at this dataset's scale.
    """
    t0 = time.time()
        
    # Parse straight into efficient dtypes where the CSV's original
    # column names allow it (low-cardinality columns as `category`,
    # amounts as float32, the label as int8) -- this avoids ever
    # materializing the wasteful default dtype and then converting,
    # which would hold both copies in memory at once during the
    # conversion. `Account`/`Account.1` are handled after rename,
    # below, since they need a CategoricalDtype *shared* across both
    # columns (see note there), which read_csv's per-column dtype=
    # can't express on its own.
    read_dtypes = {
        "Payment Format": "category",
        "Payment Currency": "category",
        "Receiving Currency": "category",
        "Amount Received": "float32",
        "Amount Paid": "float32",
        "Is Laundering": "int8",
    }
    df = pd.read_csv(csv_path, dtype=read_dtypes)

    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected IBM-AML columns: {missing}")

    df = df.rename(columns={
        "Account": "Sender Account",
        "Account.1": "Receiver Account",
    })

    df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    # Sender/Receiver Account share ONE CategoricalDtype: they're the
    # same kind of value, so a shared category list avoids storing it
    # twice, AND pandas categoricals can only be compared with
    # `!=`/`==` if their `.categories` are identical, which the
    # cleaning filter below relies on for Sender Account != Receiver
    # Account. (These two still go through a one-time object-dtype
    # read since read_csv's per-column dtype= can't express "shared
    # categories across two columns" directly.)
    account_categories = pd.unique(
        pd.concat([df["Sender Account"], df["Receiver Account"]], ignore_index=True)
    )
    account_dtype = pd.CategoricalDtype(categories=account_categories)
    df["Sender Account"] = df["Sender Account"].astype(account_dtype)
    df["Receiver Account"] = df["Receiver Account"].astype(account_dtype)

    # Same idea for the two currency columns -- already individually
    # `category` from read_dtypes above, just unify their categories.
    currency_categories = pd.unique(
        pd.concat([df["Payment Currency"], df["Receiving Currency"]], ignore_index=True)
    )
    currency_dtype = pd.CategoricalDtype(categories=currency_categories)
    df["Payment Currency"] = df["Payment Currency"].astype(currency_dtype)
    df["Receiving Currency"] = df["Receiving Currency"].astype(currency_dtype)

    df["From Bank"] = pd.to_numeric(df["From Bank"], downcast="integer")
    df["To Bank"] = pd.to_numeric(df["To Bank"], downcast="integer")

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
    df["txn_id"] = df.index.values.astype(np.int64)

    # Derived numeric fields used heavily downstream.
    df["log_amount_paid"] = np.log1p(df["Amount Paid"]).astype(np.float32)
    df["log_amount_received"] = np.log1p(df["Amount Received"]).astype(np.float32)
    df["same_bank_flag"] = (df["From Bank"] == df["To Bank"]).astype(np.int8)
    df["hour_of_day"] = df["Timestamp"].dt.hour.astype(np.int8)
    df["day_of_week"] = df["Timestamp"].dt.dayofweek.astype(np.int8)

    # Integer-coded categoricals for embedding layers (section 6),
    # sharing one vocab per group (e.g. sender & receiver share the
    # "account" vocab) and fit only on the training file. `.map()` on
    # an already-`category`-dtype column only has to map the small set
    # of unique categories, not every row -- another free speedup from
    # the dtype change above.
    fit_mode = vocabs is None
    if fit_mode:
        vocabs = {}
    for group, cols in CATEGORICAL_GROUPS.items():
        if fit_mode:
            vocabs[group] = build_vocab([df[c] for c, _ in cols])
        for raw_col, out_col in cols:
            df[out_col] = apply_vocab(df[raw_col], vocabs[group])
            df[out_col] = pd.to_numeric(df[out_col], downcast="integer")

    df["txn_id"] = df["txn_id"].astype(np.int64)  # NetworKit's addEdges() requires int64
    df["label"] = df["Is Laundering"].astype(np.int8)

    print("01. Data Loading & Normalization", time.time() - t0)

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

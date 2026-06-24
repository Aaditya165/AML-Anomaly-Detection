"""
aggregation.py
--------------
PRD Section 14 ("High-Risk Transaction Extraction -> Account Aggregation"),
Section 15 (Alert Generation), Section 16 (Dashboard Requirements).
"""

import numpy as np
import pandas as pd
import time



def build_transaction_view(df: pd.DataFrame, probs: np.ndarray, threshold: float = 0.5) -> pd.DataFrame:
    """Section 16 'Transaction View' table."""
    view = pd.DataFrame({
        "Transaction ID": df["txn_id"].values,
        "Risk Score": probs,
        "Sender": df["Sender Account"].values,
        "Receiver": df["Receiver Account"].values,
        "Amount": df["Amount Paid"].values,
        "Timestamp": df["Timestamp"].values,
        "Flagged": probs >= threshold,
    })
    return view.sort_values("Risk Score", ascending=False).reset_index(drop=True)


def aggregate_accounts(df: pd.DataFrame, probs: np.ndarray, threshold: float = 0.5) -> pd.DataFrame:
    """
    Section 14 'Account Risk Aggregation' + Section 15 'Account Alert'.

    An account's risk score = the MAX risk score among any transaction it
    participated in (as sender or receiver) -- a single high-risk hop is
    enough to warrant investigator attention, even if most of an account's
    other activity looks ordinary.

    Account Alert fires when the account has >=1 flagged transaction
    (Section 15's "Flagged transactions" criterion; relay-timing /
    short-cycle / chain features already feed into the model's score, so
    a flagged transaction on an account that is mid-layering-chain or
    round-tripping is, by construction, also how "high-risk chains" /
    "repeated suspicious patterns" surface here).
    """
    t0 = time.time()
    flagged = probs >= threshold

    sender_side = pd.DataFrame({
        "Account ID": df["Sender Account"].values,
        "Risk Score": probs,
        "Flagged": flagged,
        "Counterparty": df["Receiver Account"].values,
    })
    receiver_side = pd.DataFrame({
        "Account ID": df["Receiver Account"].values,
        "Risk Score": probs,
        "Flagged": flagged,
        "Counterparty": df["Sender Account"].values,
    })
    long_df = pd.concat([sender_side, receiver_side], ignore_index=True)

    agg = long_df.groupby("Account ID").agg(
        **{
            "Associated Risk Score": ("Risk Score", "max"),
            "Mean Transaction Risk": ("Risk Score", "mean"),
            "Number of Flagged Transactions": ("Flagged", "sum"),
            "Total Transactions": ("Flagged", "size"),
        }
    ).reset_index()

    counterparty_counts = long_df.groupby("Account ID")["Counterparty"].nunique()
    agg["Counterparty Summary"] = agg["Account ID"].map(counterparty_counts)

    agg["Account Alert"] = agg["Number of Flagged Transactions"] > 0
    agg = agg.sort_values("Associated Risk Score", ascending=False).reset_index(drop=True)

    print("07. Account Risk Aggregation: ", time.time() - t0)

    return agg


if __name__ == "__main__":
    from data_processing import load_and_clean
    from graph_construction import build_transaction_graph
    from feature_engineering import engineer_all_features

    df, vocabs = load_and_clean("HI-Small_Trans_SYNTHETIC.csv")
    g, src, dst, preds, succs = build_transaction_graph(df)
    df = engineer_all_features(df, preds, succs, src, dst)

    # fake "probs" for a standalone smoke test of the aggregation logic
    rng = np.random.default_rng(0)
    fake_probs = rng.random(len(df))

    txn_view = build_transaction_view(df, fake_probs)
    acct_view = aggregate_accounts(df, fake_probs)
    print(txn_view.head())
    print(acct_view.head())

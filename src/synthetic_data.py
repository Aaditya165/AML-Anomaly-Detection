"""
synthetic_data.py
------------------
The real IBM AML dataset (HI-Small_Trans.csv / LI-Small_Trans.csv) is not
present in this environment, so this module generates a small synthetic
dataset with the SAME COLUMN SCHEMA so the rest of the pipeline can be
built, run, and timed end-to-end. Swap in the real CSV paths later --
nothing else needs to change.

Real IBM-AML schema:
    Timestamp, From Bank, Account, To Bank, Account.1,
    Amount Received, Receiving Currency, Amount Paid, Payment Currency,
    Payment Format, Is Laundering

(pandas auto-renames the 2nd "Account" column header to "Account.1")
"""

import numpy as np
import pandas as pd


CURRENCIES = ["US Dollar", "Euro", "UK Pound", "Yen", "Bitcoin", "Saudi Riyal"]
PAYMENT_FORMATS = ["Credit Card", "ACH", "Cheque", "Wire", "Cash", "Reinvestment"]


def generate_synthetic_transactions(
    n_accounts: int = 1500,
    n_transactions: int = 40000,
    laundering_rate: float = 0.015,
    n_banks: int = 60,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Build a synthetic transaction ledger that mimics the IBM AML schema.

    Includes a mix of:
      - ordinary random transactions (legitimate)
      - injected short "layering" chains: A->B->C->D within a few hours,
        each hop labelled Is Laundering = 1 (mimics smurfing/pass-through)
      - injected short cycles: A->B->A (round-tripping), also labelled 1

    This is purely for pipeline validation/timing, not for real AML research.
    """
    rng = np.random.default_rng(seed)

    accounts = np.array([f"ACC{100000 + i}" for i in range(n_accounts)])
    account_bank = rng.integers(0, n_banks, size=n_accounts)

    base_time = pd.Timestamp("2022-09-01 00:00:00")

    rows = []

    # ---- 1) Bulk of ordinary, mostly-legitimate transactions ----
    n_normal = int(n_transactions * (1 - laundering_rate * 3))  # leave room for injected chains
    senders_idx = rng.integers(0, n_accounts, size=n_normal)
    receivers_idx = rng.integers(0, n_accounts, size=n_normal)
    same = senders_idx == receivers_idx
    receivers_idx[same] = (receivers_idx[same] + 1) % n_accounts

    offsets_minutes = np.sort(rng.exponential(scale=240, size=n_normal).cumsum())
    offsets_minutes = offsets_minutes / offsets_minutes[-1] * (60 * 24 * 60)  # spread over ~60 days

    amounts = np.round(rng.lognormal(mean=6.5, sigma=1.3, size=n_normal), 2)

    for i in range(n_normal):
        s, r = senders_idx[i], receivers_idx[i]
        ts = base_time + pd.Timedelta(minutes=float(offsets_minutes[i]))
        pay_ccy = rng.choice(CURRENCIES)
        rec_ccy = pay_ccy if rng.random() < 0.85 else rng.choice(CURRENCIES)
        rows.append({
            "Timestamp": ts,
            "From Bank": int(account_bank[s]),
            "Account": accounts[s],
            "To Bank": int(account_bank[r]),
            "Account.1": accounts[r],
            "Amount Received": amounts[i],
            "Receiving Currency": rec_ccy,
            "Amount Paid": amounts[i],
            "Payment Currency": pay_ccy,
            "Payment Format": rng.choice(PAYMENT_FORMATS),
            "Is Laundering": 0,
        })

    # ---- 2) Injected layering chains (A->B->C->D, fast relay) ----
    n_chains = int((n_transactions - n_normal) * 0.6 / 3)
    for _ in range(n_chains):
        chain_accounts = rng.choice(n_accounts, size=4, replace=False)
        start_ts = base_time + pd.Timedelta(
            minutes=float(rng.uniform(0, 60 * 24 * 60))
        )
        amt = float(np.round(rng.uniform(5000, 9999), 2))  # just-under-reporting-threshold amounts
        t = start_ts
        for hop in range(3):
            s, r = chain_accounts[hop], chain_accounts[hop + 1]
            t = t + pd.Timedelta(minutes=float(rng.uniform(2, 45)))  # fast relay
            pay_ccy = rng.choice(CURRENCIES)
            rows.append({
                "Timestamp": t,
                "From Bank": int(account_bank[s]),
                "Account": accounts[s],
                "To Bank": int(account_bank[r]),
                "Account.1": accounts[r],
                "Amount Received": amt,
                "Receiving Currency": pay_ccy,
                "Amount Paid": amt,
                "Payment Currency": pay_ccy,
                "Payment Format": rng.choice(["Wire", "ACH"]),
                "Is Laundering": 1,
            })

    # ---- 3) Injected short cycles (A->B->A, round-tripping) ----
    n_cycles = int((n_transactions - n_normal) * 0.4 / 2)
    for _ in range(n_cycles):
        a, b = rng.choice(n_accounts, size=2, replace=False)
        start_ts = base_time + pd.Timedelta(
            minutes=float(rng.uniform(0, 60 * 24 * 60))
        )
        amt = float(np.round(rng.uniform(2000, 8000), 2))
        for (s, r) in [(a, b), (b, a)]:
            start_ts = start_ts + pd.Timedelta(minutes=float(rng.uniform(5, 60)))
            pay_ccy = rng.choice(CURRENCIES)
            rows.append({
                "Timestamp": start_ts,
                "From Bank": int(account_bank[s]),
                "Account": accounts[s],
                "To Bank": int(account_bank[r]),
                "Account.1": accounts[r],
                "Amount Received": amt,
                "Receiving Currency": pay_ccy,
                "Amount Paid": amt,
                "Payment Currency": pay_ccy,
                "Payment Format": rng.choice(["Wire", "Cash"]),
                "Is Laundering": 1,
            })

    df = pd.DataFrame(rows)
    df = df.sort_values("Timestamp").reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = generate_synthetic_transactions()
    print(df.shape)
    print(df["Is Laundering"].value_counts(normalize=True))
    df.to_csv("HI-Small_Trans_SYNTHETIC.csv", index=False)

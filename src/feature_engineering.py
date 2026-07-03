"""
feature_engineering.py
-----------------------
PRD Sections 7-9: transaction node features.

Broken into 4 independently-timed sub-stages so the final timing report
shows exactly which *kind* of feature engineering is expensive:

    03a. Temporal Features
    03b. Pair History Features
    03c. Transaction Flow Features   (uses the NetworKit graph)
    03d. Account Context Features    (sender/receiver behavioral profiles)

All "historical" / "prior" statistics are computed causally (only using
transactions strictly before the current one in time) to avoid label
leakage -- this is what makes 03d a single chronological streaming pass
rather than a vectorized groupby, and is typically the most expensive
stage on real data.
"""

import math
from collections import defaultdict, deque
import time
import numpy as np
import pandas as pd


ONE_DAY_NS = 24 * 3600 * 1_000_000_000
TEN_DAYS_NS = 10 * ONE_DAY_NS
RELAY_DECAY_HOURS = 24.0           # relay-timing exponential decay constant
RECENCY_DECAY_DAYS = 10.0          # pair-recency exponential decay constant
PASS_THROUGH_WINDOW_NS = 2 * ONE_DAY_NS   # "sender recently receiver" window
CYCLE_WINDOW_NS = TEN_DAYS_NS              # short-cycle lookback window


# ---------------------------------------------------------------------------
# 03a. Temporal Features
# ---------------------------------------------------------------------------
def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    t0 = time.time()

    df = df.copy()

    df["time_since_prev_outgoing_sender"] = (
        df.groupby("sender_idx")["Timestamp"].diff().dt.total_seconds()
    )
    df["time_since_prev_incoming_receiver"] = (
        df.groupby("receiver_idx")["Timestamp"].diff().dt.total_seconds()
    )

    pair_diff = df.groupby(["sender_idx", "receiver_idx"])["Timestamp"].diff()
    df["time_since_last_pair_transfer"] = pair_diff.dt.total_seconds()

    days_since_pair = df["time_since_last_pair_transfer"] / 86400.0
    df["recency_pair_score"] = np.exp(-days_since_pair / RECENCY_DECAY_DAYS)

    fill_cols = [
        "time_since_prev_outgoing_sender",
        "time_since_prev_incoming_receiver",
        "time_since_last_pair_transfer",
    ]
    df[fill_cols] = df[fill_cols].fillna(-1)        # -1 = "no prior event"
    df["recency_pair_score"] = df["recency_pair_score"].fillna(0.0)

    print("03a. Feature Engineering - Temporal Features: ", time.time() - t0)

    return df


# ---------------------------------------------------------------------------
# 03b. Pair History Features
# ---------------------------------------------------------------------------
def add_pair_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    All stats below are computed with cumulative sums / cumulative max
    (vectorized, O(n log n) via groupby) rather than per-group
    `.expanding().std()/.max()` Python callbacks. On 40k rows the naive
    `groupby(...).transform(lambda s: s.shift(1).expanding().std())`
    version took ~48s (thousands of tiny groups, each paying Python-loop
    overhead); the cumsum/cummax version below does the same computation
    in well under a second and scales to millions of rows.
    """
    t0 = time.time()

    df = df.copy()
    pair_key = ["sender_idx", "receiver_idx"]
    amt = df["Amount Paid"]

    df["pair_prior_txn_count"] = df.groupby(pair_key).cumcount()
    prior_count_safe = df["pair_prior_txn_count"].replace(0, np.nan)

    # ---- prior sum / mean, fully vectorized via cumsum ----
    cumsum_incl = df.groupby(pair_key)["Amount Paid"].cumsum()
    df["pair_total_amount_prior"] = cumsum_incl - amt
    df["pair_mean_amount_prior"] = (
        df["pair_total_amount_prior"] / prior_count_safe
    ).fillna(0.0)

    # ---- prior std, via cumsum of squares (E[x^2] - E[x]^2) ----
    amt_sq = amt.pow(2)
    cumsumsq_incl = amt_sq.groupby([df["sender_idx"], df["receiver_idx"]]).cumsum()
    prior_sumsq = cumsumsq_incl - amt_sq
    prior_mean_raw = df["pair_total_amount_prior"] / prior_count_safe
    prior_var = (prior_sumsq / prior_count_safe) - prior_mean_raw.pow(2)
    df["pair_std_amount_prior"] = np.sqrt(prior_var.clip(lower=0)).fillna(0.0)

    # ---- prior max, via shift(1) + grouped cummax (NaN-safe) ----
    shifted = df.groupby(pair_key)["Amount Paid"].shift(1)
    running_max_prior = shifted.groupby(
        [df["sender_idx"], df["receiver_idx"]]
    ).cummax()
    df["pair_max_amount_prior"] = running_max_prior.fillna(0.0)

    # ---- repeated exact-amount count on this pair (vectorized cumcount) ----
    df["pair_repeated_exact_amount_count"] = df.groupby(
        pair_key + ["Amount Paid"]
    ).cumcount()

    print("03b. Feature Engineering - Pair History Features: ", time.time() - t0)

    return df


# ---------------------------------------------------------------------------
# 03c. Transaction Flow Features (graph-topology driven)
# ---------------------------------------------------------------------------
def add_flow_features(
    df: pd.DataFrame,
    predecessors,
    successors,
    edge_src: np.ndarray,
    edge_dst: np.ndarray,
) -> pd.DataFrame:
    t0 = time.time()

    df = df.copy()
    n = len(df)
    ts_ns = df["Timestamp"].values.astype("datetime64[ns]").astype(np.int64)
    sender_idx = df["sender_idx"].values
    receiver_idx = df["receiver_idx"].values

    # Predecessor / Successor counts = raw graph in/out-degree.
    pred_count = predecessors.degrees().astype(np.int32)
    succ_count = successors.degrees().astype(np.int32)
    df["predecessor_count"] = pred_count
    df["successor_count"] = succ_count

    # Fan-In / Fan-Out = *distinct counterparties* among those edges
    # (captures aggregation/dispersal behavior, not just raw degree).
    fan_in = np.zeros(n, dtype=np.int32)
    fan_out = np.zeros(n, dtype=np.int32)
    
    for t in range(n):

        if pred_count[t] > 0:

            preds = predecessors[t]

            fan_in[t] = len(
                np.unique(sender_idx[preds])
            )

        if succ_count[t] > 0:

            succs = successors[t]

            fan_out[t] = len(
                np.unique(receiver_idx[succs])
            )
    
    df["fan_in"] = fan_in
    df["fan_out"] = fan_out

    # Relay timing score: mean exp(-Δt_hours / RELAY_DECAY_HOURS) over
    # this node's outgoing edges -> high score = funds relayed quickly.
    if len(edge_src) > 0:
        dt_hours = (ts_ns[edge_dst] - ts_ns[edge_src]) / 3.6e12
        edge_score = np.exp(-dt_hours / RELAY_DECAY_HOURS)
        relay_sum = np.bincount(edge_src, weights=edge_score, minlength=n)
        relay_score = np.divide(
            relay_sum, np.maximum(succ_count, 1),
            out=np.zeros(n), where=succ_count > 0,
        )
    else:
        relay_score = np.zeros(n)
    df["relay_timing_score"] = relay_score

    # Continues Existing Chain: this node extends a path already in the graph.
    df["continues_existing_chain"] = (pred_count > 0).astype(np.int8)

    # Sender Recently Receiver (pass-through behavior), independent of
    # the formal top-15-successor-capped graph: did the SENDER receive
    # *any* funds within the last PASS_THROUGH_WINDOW before this txn?
    order = np.argsort(receiver_idx, kind="mergesort")
    sorted_recv = receiver_idx[order]
    sorted_ts_by_recv = ts_ns[order]
    uniq_recv, start_pos = np.unique(sorted_recv, return_index=True)
    end_pos = np.append(start_pos[1:], len(sorted_recv))
    recv_block = {a: (s, e) for a, s, e in zip(uniq_recv, start_pos, end_pos)}

    sender_recently_receiver = np.zeros(n, dtype=np.int8)
    for t in range(n):
        block = recv_block.get(sender_idx[t])
        if block is None:
            continue
        s, e = block
        block_ts = sorted_ts_by_recv[s:e]
        lo = np.searchsorted(block_ts, ts_ns[t] - PASS_THROUGH_WINDOW_NS, side="left")
        hi = np.searchsorted(block_ts, ts_ns[t], side="left")  # strictly before t
        if hi > lo:
            sender_recently_receiver[t] = 1
    df["sender_recently_receiver"] = sender_recently_receiver

    # Short Cycle Participation: hash-based lookup (no DFS/BFS).
    # Maintain {(sender,receiver): sorted timestamps[ns]} once, then for
    # each txn (S,R,t) binary-search the *reverse* key (R,S) for any
    # prior timestamp within CYCLE_WINDOW_NS  -> O(log n) per row.
    pair_df = pd.DataFrame({
        "s": sender_idx, "r": receiver_idx, "ts": ts_ns,
    }).sort_values("ts", kind="mergesort")
    pair_index = defaultdict(list)
    for s, r, t in zip(pair_df["s"].values, pair_df["r"].values, pair_df["ts"].values):
        pair_index[(s, r)].append(t)
    pair_index = {k: np.asarray(v, dtype=np.int64) for k, v in pair_index.items()}

    short_cycle = np.zeros(n, dtype=np.int8)
    for t in range(n):
        reverse_ts = pair_index.get((receiver_idx[t], sender_idx[t]))
        if reverse_ts is None:
            continue
        lo = np.searchsorted(reverse_ts, ts_ns[t] - CYCLE_WINDOW_NS, side="left")
        hi = np.searchsorted(reverse_ts, ts_ns[t], side="left")
        if hi > lo:
            short_cycle[t] = 1
    df["short_cycle_participation"] = short_cycle

    print("03c. Feature Engineering - Transaction Flow Features: ", time.time() - t0)

    return df


# ---------------------------------------------------------------------------
# 03d. Account Context Features (single chronological streaming pass)
# ---------------------------------------------------------------------------
def _safe_clogc(c: int) -> float:
    return 0.0 if c <= 0 else c * math.log(c)


class _AccountState:
    """Running, causal (prior-only) profile for a single account.

    Summary stats (entropy, concentration, unique-counterparty count) are
    maintained incrementally in O(1) amortized per update, rather than
    recomputed from the full counterparty-count history on every snapshot.
    The original version rebuilt these from scratch on every transaction,
    making per-account cost O(degree^2) instead of O(degree) -- fine for
    typical accounts, but catastrophic for high-degree hub/mule accounts.
    """
    __slots__ = (
        "outflow_total", "inflow_total",
        "out_counterparty_counts", "in_counterparty_counts",
        "out_count", "in_count",
        "activity_times",
        "unique_counterparties_set",
        "sum_clogc",
        "max_category_count",
    )

    def __init__(self):
        self.outflow_total = 0.0
        self.inflow_total = 0.0
        self.out_counterparty_counts = {}
        self.in_counterparty_counts = {}
        self.out_count = 0
        self.in_count = 0
        self.activity_times = deque()
        self.unique_counterparties_set = set()
        self.sum_clogc = 0.0
        self.max_category_count = 0

    def _bump(self, counts: dict, counterparty) -> None:
        old = counts.get(counterparty, 0)
        new = old + 1
        counts[counterparty] = new
        self.sum_clogc += _safe_clogc(new) - _safe_clogc(old)
        if new > self.max_category_count:
            self.max_category_count = new
        self.unique_counterparties_set.add(counterparty)

    def record_outgoing(self, counterparty, amt: float, now: int) -> None:
        self.outflow_total += amt
        self._bump(self.out_counterparty_counts, counterparty)
        self.out_count += 1
        self.activity_times.append(now)

    def record_incoming(self, counterparty, amt: float, now: int) -> None:
        self.inflow_total += amt
        self._bump(self.in_counterparty_counts, counterparty)
        self.in_count += 1
        self.activity_times.append(now)


def add_account_context_features(df: pd.DataFrame) -> pd.DataFrame:
    t0 = time.time()
    df = df.copy()
    n = len(df)
    ts_ns = df["Timestamp"].values.astype("datetime64[ns]").astype(np.int64)
    sender_idx = df["sender_idx"].values
    receiver_idx = df["receiver_idx"].values
    amount = df["Amount Paid"].values

    states: dict = defaultdict(_AccountState)

    out_cols = {
        "hist_outflow_total": np.zeros(n), "hist_inflow_total": np.zeros(n),
        "unique_counterparties": np.zeros(n, dtype=np.int32),
        "entropy": np.zeros(n), "concentration": np.zeros(n),
        "degree_balance": np.zeros(n), "velocity_1d": np.zeros(n, dtype=np.int32),
        "velocity_10d": np.zeros(n, dtype=np.int32),
    }
    sender_feats = {k: v.copy() for k, v in out_cols.items()}
    receiver_feats = {k: v.copy() for k, v in out_cols.items()}

    def _velocity(times: deque, now: int, window_ns: int) -> int:
        while times and times[0] < now - window_ns:
            times.popleft()
        return len(times)

    def _snapshot(state: "_AccountState", now: int) -> dict:
        out_count, in_count = state.out_count, state.in_count
        total = out_count + in_count
        unique_cp = len(state.unique_counterparties_set)
        entropy = (
            math.log(total) - state.sum_clogc / total
            if total > 0 else 0.0
        )
        concentration = state.max_category_count / total if total > 0 else 0.0
        degree_balance = (in_count - out_count) / total if total > 0 else 0.0
        v1 = _velocity(state.activity_times, now, ONE_DAY_NS)
        v10 = _velocity(state.activity_times, now, TEN_DAYS_NS)
        return {
            "hist_outflow_total": state.outflow_total,
            "hist_inflow_total": state.inflow_total,
            "unique_counterparties": unique_cp,
            "entropy": entropy,
            "concentration": concentration,
            "degree_balance": degree_balance,
            "velocity_1d": v1,
            "velocity_10d": v10,
        }

    for t in range(n):
        now = ts_ns[t]
        s, r, amt = sender_idx[t], receiver_idx[t], amount[t]

        s_state = states[s]
        r_state = states[r]

        # ---- snapshot BEFORE updating with the current transaction ----
        snap_s = _snapshot(s_state, now)
        snap_r = _snapshot(r_state, now)
        for k, v in snap_s.items():
            sender_feats[k][t] = v
        for k, v in snap_r.items():
            receiver_feats[k][t] = v

        # ---- now apply this transaction's effect for future rows ----
        s_state.record_outgoing(r, amt, now)
        r_state.record_incoming(s, amt, now)

    for k, v in sender_feats.items():
        df[f"sender_{k}"] = v
    for k, v in receiver_feats.items():
        df[f"receiver_{k}"] = v

    print("03d. Feature Engineering - Account Context Features: ", time.time() - t0)

    return df


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def engineer_all_features(
    df: pd.DataFrame,
    predecessors,
    successors,
    edge_src: np.ndarray,
    edge_dst: np.ndarray,
) -> pd.DataFrame:
    df = add_temporal_features(df)
    df = add_pair_history_features(df)
    df = add_flow_features(df, predecessors, successors, edge_src, edge_dst)
    df = add_account_context_features(df)
    return df


if __name__ == "__main__":
    from data_processing import load_and_clean
    from graph_construction import build_transaction_graph

    df, vocabs = load_and_clean("data/HI-Small_Trans.csv")
    g, src, dst, preds, succs = build_transaction_graph(df)
    df = engineer_all_features(df, preds, succs, src, dst)
    print(df.shape)
    print(df.filter(like="sender_").head())
    print(df.filter(like="short_cycle").sum())

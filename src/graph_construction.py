"""
graph_construction.py
----------------------
PRD Section 5: "Transaction Graph Construction (NetworKit)"

Each transaction is a node. A directed edge T1 -> T2 exists iff:
    receiver(T1) == sender(T2)
    AND timestamp(T2) > timestamp(T1)
    AND (timestamp(T2) - timestamp(T1)) <= 10 days
Each T1 keeps at most its 15 chronologically-nearest successors
(Constraint 3), to stop hub accounts from blowing up graph density.

Implementation note on efficiency:
    For every account, we pre-sort that account's *outgoing* transactions
    by timestamp once. For a given T1, candidate successors are just the
    outgoing transactions of T1's receiver account -- so we binary-search
    that one sorted array for the 10-day window and take the first 15
    entries (already nearest-in-time because the array is sorted and we
    only keep timestamps > T1). This is O(n log n) overall instead of an
    O(n^2) pairwise scan.
"""

import numpy as np
import pandas as pd
import networkit as nk
import time

TIME_WINDOW_NS = 10 * 24 * 3600 * 1_000_000_000  # 10 days, in ns
MAX_SUCCESSORS = 15


def build_transaction_graph(df: pd.DataFrame):
    """
    Returns:
        g            : networkit.Graph (directed), one node per transaction
        src, dst     : np.ndarray edge endpoint arrays (txn_id space), handy
                       for direct export to PyTorch Geometric later
        predecessors : list[np.ndarray] predecessors[t] = txn_ids flowing into t
        successors   : list[np.ndarray] successors[t]   = txn_ids flowing out of t
    """
    n = len(df)
    ts_ns = df["Timestamp"].values.astype("datetime64[ns]").astype(np.int64)
    receiver_idx = df["receiver_idx"].values
    sender_idx = df["sender_idx"].values
    txn_id = df["txn_id"].values  # == np.arange(n), but keep explicit

    #with Timer("02. Transaction Graph Construction (NetworKit)"):
    t0 = time.time()
    # ---- Step A: per-account sorted outgoing-transaction index ----
    # account -> (sorted_timestamps[ns], corresponding txn_ids)
    order = np.argsort(sender_idx, kind="mergesort")
    sorted_sender = sender_idx[order]
    sorted_ts_by_sender = ts_ns[order]
    sorted_txn_by_sender = txn_id[order]

    # boundaries of each account's block within the sender-sorted arrays
    unique_accs, start_pos = np.unique(sorted_sender, return_index=True)
    end_pos = np.append(start_pos[1:], len(sorted_sender))
    acc_to_block = {acc: (s, e) for acc, s, e in zip(unique_accs, start_pos, end_pos)}

    g = nk.Graph(n, weighted=False, directed=True)

    edge_src, edge_dst = [], []

    # ---- Step B: for each T1, binary-search receiver's outgoing block ----
    for t1 in range(n):
        recv = receiver_idx[t1]
        block = acc_to_block.get(recv)
        if block is None:
            continue
        s, e = block
        block_ts = sorted_ts_by_sender[s:e]
        block_txn = sorted_txn_by_sender[s:e]

        lo = np.searchsorted(block_ts, ts_ns[t1], side="right")
        hi = np.searchsorted(block_ts, ts_ns[t1] + TIME_WINDOW_NS, side="right")
        if hi <= lo:
            continue

        cand_txn = block_txn[lo:hi]
        # already time-sorted ascending => first MAX_SUCCESSORS are nearest
        cand_txn = cand_txn[:MAX_SUCCESSORS]
        # a node cannot be its own successor (guards a pathological case
        # where a transaction would point back to itself if duplicated)
        cand_txn = cand_txn[cand_txn != t1]

        for t2 in cand_txn:
            edge_src.append(t1)
            edge_dst.append(int(t2))

    edge_src = np.asarray(edge_src, dtype=np.int64)
    edge_dst = np.asarray(edge_dst, dtype=np.int64)

    g.addEdges((edge_src, edge_dst), checkMultiEdge=False)

    # ---- Step C: adjacency lists for downstream feature engineering ----
    predecessors = [[] for _ in range(n)]
    successors = [[] for _ in range(n)]
    for s, d in zip(edge_src, edge_dst):
        successors[s].append(d)
        predecessors[d].append(s)
    predecessors = [np.asarray(p, dtype=np.int64) for p in predecessors]
    successors = [np.asarray(s, dtype=np.int64) for s in successors]

    print("02. Transaction Graph Construction: ", time.time() - t0)

    return g, edge_src, edge_dst, predecessors, successors


if __name__ == "__main__":
    from data_processing import load_and_clean
    df, vocabs = load_and_clean("HI-Small_Trans_SYNTHETIC.csv")
    g, src, dst, preds, succs = build_transaction_graph(df)
    print("nodes:", g.numberOfNodes(), "edges:", g.numberOfEdges())
    print("avg out-degree:", src.shape[0] / g.numberOfNodes())
    print("max successor fan-out actually used:", max(len(s) for s in succs))

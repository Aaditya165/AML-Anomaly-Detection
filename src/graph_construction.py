"""
graph_construction.py
----------------------
Builds BOTH graph representations this project uses. No feature
engineering happens here -- that's feature_engineering.py (transaction
graph) and flow_features.py / community_detection.py (account graph).

  * `build_transaction_graph` -- PRD Section 5. One node PER TRANSACTION;
    a directed edge T1 -> T2 exists iff receiver(T1) == sender(T2) and T2
    happens within 10 days of T1. Excellent for temporal propagation
    (relay timing, short cycles) -- unchanged from the original pipeline.

  * `build_account_graph` -- ExSTraQt paper Section 3.1.2 / Eq. 1. One
    node PER ACCOUNT; repeated transfers between the same two accounts
    collapse into a single weighted edge (total amount, plus the
    sender/receiver-balanced weight from Eq. 1). This is the graph
    community_detection.py and flow_features.py operate on.

PRD Section 5 continued (transaction graph):

Each transaction is a node. A directed edge T1 -> T2 exists iff:
    receiver(T1) == sender(T2)
    AND timestamp(T2) > timestamp(T1)
    AND (timestamp(T2) - timestamp(T1)) <= 10 days
Each T1 keeps at most its 15 chronologically-nearest successors
(Constraint 3), to stop hub accounts from blowing up graph density.

PERFORMANCE / MEMORY NOTE (this version replaces an earlier one that OOM'd
on the real ~4.5M-6M row IBM-AML files):

  - Edge construction is vectorized PER RECEIVER-ACCOUNT GROUP instead of
    looping row-by-row in Python. Each group does one vectorized
    `np.searchsorted` over ALL of that group's transactions at once, so the
    outer Python loop runs ~(number of unique accounts) times instead of
    (number of transactions) times -- e.g. ~420k iterations instead of
    4.5M, each iteration doing real work on dozens of rows via numpy
    instead of one row at a time. No Python ints/lists are used to
    accumulate edges; everything stays in numpy arrays via a repeat+offset
    trick (see `_vectorized_topk_join`) until one final `np.concatenate`.

  - Predecessor/successor adjacency is stored as a single CSR structure
    (`CSRAdjacency`: one `indices` array for every edge in the graph + one
    `indptr` array of length N+1) instead of N separate Python lists that
    get converted into N separate numpy arrays. The old approach created
    ~2*N individual array objects (predecessors + successors), each
    carrying its own ~100+ byte object-header overhead *on top of* the
    data -- at N in the millions, that overhead alone was multiple GB,
    which is what was actually exhausting Colab's RAM (NetworKit's own
    C++ graph object was never the problem). CSR uses exactly 2 flat
    arrays total, sized O(E) and O(N), with zero per-node object overhead.
    `adj[node]` still works (returns a zero-copy view into `indices`), so
    every downstream consumer (`feature_engineering.py`,
    `neighbor_sampling.py`) needed no changes.
"""

import numpy as np
import pandas as pd
import networkit as nk
import time


TIME_WINDOW_NS = 10 * 24 * 3600 * 1_000_000_000  # 10 days, in ns
MAX_SUCCESSORS = 60

class CSRAdjacency:
    """
    Memory-efficient variable-length-per-node adjacency list.

    One flat `indices` array (every edge endpoint in the whole graph) plus
    one `indptr` array of length N+1 (indptr[i]:indptr[i+1] marks node i's
    slice). `adj[node]` returns a zero-copy view, same as indexing the old
    "list of N numpy arrays" -- so existing call sites (`predecessors[t]`,
    `len(predecessors[t])`, `for p in predecessors`) work unchanged.
    """
    __slots__ = ("indices", "indptr")

    def __init__(self, indices: np.ndarray, indptr: np.ndarray):
        self.indices = indices
        self.indptr = indptr

    def __getitem__(self, node: int) -> np.ndarray:
        return self.indices[self.indptr[node]:self.indptr[node + 1]]

    def __len__(self) -> int:
        return len(self.indptr) - 1

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def degrees(self) -> np.ndarray:
        """Vectorized per-node degree (no Python loop) -- diff of indptr."""
        return np.diff(self.indptr)


def _build_csr(group_keys: np.ndarray, group_vals: np.ndarray, n: int) -> CSRAdjacency:
    """
    Build a CSRAdjacency mapping `group_keys[i] -> group_vals[i]` for all i,
    fully vectorized (argsort + bincount + cumsum, no Python loop).
    e.g. group_keys=edge_dst, group_vals=edge_src  ->  predecessors CSR
         group_keys=edge_src, group_vals=edge_dst  ->  successors CSR
    """
    order = np.argsort(group_keys, kind="stable")
    indices = group_vals[order]
    counts = np.bincount(group_keys, minlength=n) if len(group_keys) else np.zeros(n, dtype=np.int64)
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    return CSRAdjacency(indices.astype(np.int64, copy=False), indptr)


def _vectorized_topk_join(t1_ids, t1_ts, block_ts, block_txn, max_k, window_ns):
    """
    For a single account's worth of T1 rows (`t1_ids`/`t1_ts`, already in
    chronological order) and that SAME account's own sorted outgoing
    transactions (`block_ts`/`block_txn`), find -- for every T1 at once --
    up to `max_k` chronologically-nearest successors within `window_ns`.

    Returns (edge_src, edge_dst) numpy arrays (possibly empty).
    Pure numpy: one vectorized searchsorted call handles every row in the
    group simultaneously; a repeat+cumsum-offset trick (the standard way
    to build a ragged/jagged result in numpy) avoids any Python-level
    per-row loop or per-edge Python int.
    """
    lo = np.searchsorted(block_ts, t1_ts, side="right")
    hi = np.searchsorted(block_ts, t1_ts + window_ns, side="right")
    counts = np.clip(hi - lo, 0, max_k)

    total = int(counts.sum())
    if total == 0:
        return (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64))

    src_rep = np.repeat(t1_ids, counts)
    starts_rep = np.repeat(lo, counts)
    group_start = np.cumsum(counts) - counts
    local_offset = np.arange(total) - np.repeat(group_start, counts)
    flat_positions = starts_rep + local_offset
    dst_rep = block_txn[flat_positions]

    return src_rep, dst_rep


def build_transaction_graph(df: pd.DataFrame):
    """
    Returns:
        g            : networkit.Graph (directed), one node per transaction
        src, dst     : np.ndarray edge endpoint arrays (txn_id space)
        predecessors : CSRAdjacency, predecessors[t] = txn_ids flowing into t
        successors   : CSRAdjacency, successors[t]   = txn_ids flowing out of t
    """
    n = len(df)
    ts_ns = df["Timestamp"].values.astype("datetime64[ns]").astype(np.int64)
    receiver_idx = df["receiver_idx"].values
    sender_idx = df["sender_idx"].values
    txn_id = df["txn_id"].values  # == np.arange(n)

    t0 = time.time()
    # ---- Step A: per-account sorted outgoing-transaction index ----
    order_by_sender = np.argsort(sender_idx, kind="mergesort")
    sorted_sender = sender_idx[order_by_sender]
    sorted_ts_by_sender = ts_ns[order_by_sender]
    sorted_txn_by_sender = txn_id[order_by_sender]

    uniq_senders, sender_start = np.unique(sorted_sender, return_index=True)
    sender_end = np.append(sender_start[1:], len(sorted_sender))
    acc_to_block = {acc: (s, e) for acc, s, e in zip(uniq_senders, sender_start, sender_end)}

    # ---- Step B: group T1's by RECEIVER, vectorize within each group ----
    # (stable sort by receiver preserves each group's original, already
    # chronological, row order -- so t1_ts within a group stays ascending)
    order_by_receiver = np.argsort(receiver_idx, kind="stable")
    sorted_receiver = receiver_idx[order_by_receiver]
    sorted_t1_ts = ts_ns[order_by_receiver]
    sorted_t1_id = txn_id[order_by_receiver]

    uniq_receivers, recv_start = np.unique(sorted_receiver, return_index=True)
    recv_end = np.append(recv_start[1:], len(sorted_receiver))

    edge_src_parts, edge_dst_parts = [], []
    for acc, rs, re in zip(uniq_receivers, recv_start, recv_end):
        block = acc_to_block.get(acc)
        if block is None:
            continue
        s, e = block
        block_ts = sorted_ts_by_sender[s:e]
        block_txn = sorted_txn_by_sender[s:e]

        src_part, dst_part = _vectorized_topk_join(
            sorted_t1_id[rs:re], sorted_t1_ts[rs:re],
            block_ts, block_txn, MAX_SUCCESSORS, TIME_WINDOW_NS,
        )
        if len(src_part):
            edge_src_parts.append(src_part)
            edge_dst_parts.append(dst_part)

    if edge_src_parts:
        edge_src = np.concatenate(edge_src_parts)
        edge_dst = np.concatenate(edge_dst_parts)
    else:
        edge_src = np.empty(0, dtype=np.int64)
        edge_dst = np.empty(0, dtype=np.int64)

    # defensive: drop any accidental self-loop (shouldn't occur given
    # upstream cleaning already forbids sender==receiver on a single row)
    self_loop = edge_src == edge_dst
    if self_loop.any():
        edge_src = edge_src[~self_loop]
        edge_dst = edge_dst[~self_loop]

    g = nk.Graph(n, weighted=False, directed=True)
    g.addEdges((edge_src, edge_dst), checkMultiEdge=False)

    # ---- Step C: CSR adjacency (2 flat arrays total, not 2*n) ----
    predecessors = _build_csr(edge_dst, edge_src, n)
    successors = _build_csr(edge_src, edge_dst, n)

    print("02. Transaction Graph Construction (NetworKit): ", time.time() - t0)

    return g, edge_src, edge_dst, predecessors, successors


# ---------------------------------------------------------------------------
# Account graph (ExSTraQt paper Section 3.1.2, Equation 1)
# ---------------------------------------------------------------------------
# One node per ACCOUNT rather than per transaction. Repeated transfers
# between the same two accounts collapse into a single weighted edge:
#
#     A --$500--> B
#     A --$100--> B          =>      A --(amount=$650, weight=W)--> B
#     A --$50-->  B
#
# The weight isn't just the summed amount -- Eq. 1 combines the sender's
# and receiver's OWN shares of that edge:
#
#     W(s->t) = amount(s->t)/total_sent(s) + amount(s->t)/total_received(t)
#
# so a launderer can't hide a high-volume relationship by routing through
# one big intermediary account: doing that only shrinks ONE of the two
# terms, not both. This weighted graph is what community_detection.py
# (Leiden + random-walk) and flow_features.py trace over.

import igraph as ig


def build_aggregated_edges(
    df: pd.DataFrame,
    source_col: str = "Sender Account",
    target_col: str = "Receiver Account",
    amount_col: str = "Amount Paid",
) -> pd.DataFrame:
    """Collapse transaction-level rows into one row per unique
    (source, target) pair with the total amount transferred.
    Returns columns: source, target, amount."""
    edges = (
        df.groupby([source_col, target_col], observed=True)[amount_col]
        .sum()
        .reset_index()
        .rename(columns={source_col: "source", target_col: "target", amount_col: "amount"})
    )
    edges["amount"] = edges["amount"].astype(np.float64)
    return edges


def compute_edge_weights(edges_agg: pd.DataFrame) -> pd.DataFrame:
    """Equation 1. Adds `weight` and `amount_weighted` (= amount rescaled
    by weight/weight.max(), what gets fed into Leiden -- see
    community_detection.py) to an aggregated edge table."""
    totals_sent = edges_agg.groupby("source")["amount"].sum()
    totals_received = edges_agg.groupby("target")["amount"].sum()

    out = edges_agg.copy()
    s_total = out["source"].map(totals_sent).astype(np.float64)
    r_total = out["target"].map(totals_received).astype(np.float64)

    with np.errstate(divide="ignore", invalid="ignore"):
        weight = np.where(s_total > 0, out["amount"] / s_total, 0.0) + \
                 np.where(r_total > 0, out["amount"] / r_total, 0.0)
    out["weight"] = weight
    max_weight = out["weight"].max()
    out["amount_weighted"] = out["amount"] * (out["weight"] / max_weight if max_weight > 0 else 0.0)
    return out


def node_totals(edges_agg: pd.DataFrame) -> tuple:
    """(totals_sent, totals_received) dicts, node -> total amount. Used by
    flow_features.py to cap a traced flow at a node's own volume."""
    totals_sent = edges_agg.groupby("source")["amount"].sum().to_dict()
    totals_received = edges_agg.groupby("target")["amount"].sum().to_dict()
    return totals_sent, totals_received


def build_account_graph(
    df: pd.DataFrame,
    source_col: str = "Sender Account",
    target_col: str = "Receiver Account",
    amount_col: str = "Amount Paid",
) -> tuple:
    """
    Returns (graph, edges_agg):
        graph     : directed igraph.Graph, one vertex per account
                    (vertex attribute "name" = account id), edge attribute
                    "weight" = Eq. 1's amount_weighted.
        edges_agg : the underlying aggregated+weighted edge DataFrame
                    (source, target, amount, weight, amount_weighted) --
                    community_detection.py and flow_features.py both need
                    this directly, not just the igraph object.
    """
    edges_agg = build_aggregated_edges(df, source_col, target_col, amount_col)
    edges_weighted = compute_edge_weights(edges_agg)

    edge_df = edges_weighted.loc[:, ["source", "target"]]
    graph = ig.Graph.DataFrame(edge_df, directed=True, use_vids=False)
    graph.es["weight"] = edges_weighted["amount_weighted"].to_numpy()

    return graph, edges_weighted


if __name__ == "__main__":
    from data_processing import load_and_clean
    df, vocabs = load_and_clean("data/HI-Small_Trans.csv")
    g, src, dst, preds, succs = build_transaction_graph(df)
    print("nodes:", g.numberOfNodes(), "edges:", g.numberOfEdges())
    print("avg out-degree:", src.shape[0] / g.numberOfNodes())
    print("max successor fan-out actually used:", succs.degrees().max())

    account_graph, edges_agg = build_account_graph(df)
    print("account graph:", account_graph.vcount(), "accounts,", account_graph.ecount(), "unique edges")

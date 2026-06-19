from __future__ import annotations

import time
import numpy as np
import pandas as pd
import networkit as nk


def _entropy(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[values > 0]
    if len(values) == 0:
        return 0.0
    p = values / values.sum()
    return float(-(p * np.log2(p)).sum())


def _relay_score(in_times: list[pd.Timestamp], out_times: list[pd.Timestamp]) -> float:
    if not in_times or not out_times:
        return 0.0
    in_ns = np.sort(pd.to_datetime(in_times).astype(np.int64))
    out_ns = np.sort(pd.to_datetime(out_times).astype(np.int64))

    idxs = np.searchsorted(out_ns, in_ns)
    valid = idxs < len(out_ns)
    if not valid.any():
        return 0.0
    deltas_days = (out_ns[idxs[valid]] - in_ns[valid]) / (86400.0 * 1e9)
    return float(np.exp(-deltas_days.mean() / 3.0))


def build_transaction_graph_nk(df: pd.DataFrame):
    """
    Builds a NetworKit directed graph from transactions.

    NetworKit nodes are plain integers (0..n-1), not arbitrary hashable
    labels like networkx, so we keep our own account <-> index map.

    Self-loops (from_account == to_account) ARE kept in the directed graph
    so in/out-degree semantics match the old networkx version exactly.
    They're stripped only from the undirected copy used for clustering /
    eigenvector / community detection, since those algorithms don't
    support self-loops.
    """
    all_accounts = pd.concat([
        df["from_account"].astype(str),
        df["to_account"].astype(str),
    ]).unique()
    acc_to_idx = {a: i for i, a in enumerate(all_accounts)}
    idx_to_acc = {i: a for a, i in acc_to_idx.items()}
    n = len(all_accounts)

    df_e = df.copy()
    df_e["from_idx"] = df_e["from_account"].astype(str).map(acc_to_idx)
    df_e["to_idx"] = df_e["to_account"].astype(str).map(acc_to_idx)
    df_e["amount_received"] = df_e["amount_received"].fillna(0.0)

    edge_agg = df_e.groupby(["from_idx", "to_idx"]).agg(
        amount=("amount_received", "sum"),
        count=("amount_received", "size"),
    ).reset_index()

    g = nk.Graph(n, weighted=False, directed=True)
    for u, v in zip(edge_agg["from_idx"].to_numpy(), edge_agg["to_idx"].to_numpy()):
        g.addEdge(int(u), int(v))

    return g, acc_to_idx, idx_to_acc


def compute_account_features(
    df: pd.DataFrame,
    temporal_window_days: int = 7,
    max_exact_betweenness_nodes: int = 1000,
    betweenness_samples: int = 200,
) -> pd.DataFrame:
    t0 = time.time()
    g, acc_to_idx, idx_to_acc = build_transaction_graph_nk(df)
    print("Graph building: ", time.time() - t0)

    n = g.numberOfNodes()
    if n == 0:
        return pd.DataFrame()

    node_idxs = np.arange(n)

    # Undirected, self-loop-free view for clustering / eigenvector / community
    t0 = time.time()
    undirected = nk.graphtools.toUndirected(g)
    undirected.removeSelfLoops()
    print("to undirected: ", time.time() - t0)

    t0 = time.time()
    lcc = nk.centrality.LocalClusteringCoefficient(undirected)
    lcc.run()
    clustering_arr = np.array(lcc.scores())
    print("clustering coefficient: ", time.time() - t0)

    try:
        t0 = time.time()
        ec = nk.centrality.EigenvectorCentrality(undirected)
        ec.run()
        eigenvector_arr = np.array(ec.scores())
        print("eigenvector centrality: ", time.time() - t0)
    except Exception:
        eigenvector_arr = np.zeros(n)

    try:
        t0 = time.time()
        plp = nk.community.PLP(undirected)
        plp.run()
        partition = plp.getPartition()
        comm_ids = np.array(partition.getVector())
        size_map = partition.subsetSizeMap()
        community_size_arr = np.array([size_map[c] for c in comm_ids], dtype=float)
        print("communities: ", time.time() - t0)
    except Exception:
        community_size_arr = np.ones(n)

    t0 = time.time()
    if n <= max_exact_betweenness_nodes:
        bc = nk.centrality.Betweenness(g, normalized=True)
        bc.run()
        betweenness_arr = np.array(bc.scores())
        print("betweenness centrality (exact): ", time.time() - t0)
    else:
        eb = nk.centrality.EstimateBetweenness(
            g, nSamples=betweenness_samples, normalized=True, parallel=True
        )
        eb.run()
        betweenness_arr = np.array(eb.scores())
        print("betweenness centrality (sampled estimate): ", time.time() - t0)

    t0 = time.time()
    scc = nk.components.StronglyConnectedComponents(g)
    scc.run()
    scc_ids = np.array(scc.getPartition().getVector())
    scc_size_map = scc.getComponentSizes()
    scc_size_arr = np.array([scc_size_map[c] for c in scc_ids], dtype=float)
    print("SCC:", time.time() - t0)

    t0 = time.time()
    in_degree_arr = np.array([g.degreeIn(i) for i in node_idxs])
    out_degree_arr = np.array([g.degreeOut(i) for i in node_idxs])
    print("degrees:", time.time() - t0)

    # ----- transaction-attribute aggregates (unchanged, pandas-based, not the bottleneck) -----
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["from_account"] = df["from_account"].astype(str)
    df["to_account"] = df["to_account"].astype(str)
    df["amount_received"] = df["amount_received"].fillna(0.0)

    global_span_days = max(
        (df["timestamp"].max() - df["timestamp"].min()).total_seconds() / 86400.0,
        1.0,
    )

    out_agg = df.groupby("from_account").agg(
        outgoing_amount_total=("amount_received", "sum"),
        outgoing_amount_mean=("amount_received", "mean"),
        outgoing_amount_std=("amount_received", "std"),
        _out_tx_count=("amount_received", "count"),
        _out_max=("amount_received", "max"),
    ).rename_axis("account")
    out_agg["unique_out_counterparties"] = df.groupby("from_account")["to_account"].nunique()
    out_agg["outgoing_concentration"] = out_agg["_out_max"] / (out_agg["outgoing_amount_total"] + 1e-9)
    out_agg["outgoing_entropy"] = df.groupby("from_account")["amount_received"].apply(
        lambda x: _entropy(x.values)
    )

    in_agg = df.groupby("to_account").agg(
        incoming_amount_total=("amount_received", "sum"),
        incoming_amount_mean=("amount_received", "mean"),
        incoming_amount_std=("amount_received", "std"),
        _in_tx_count=("amount_received", "count"),
        _in_max=("amount_received", "max"),
    ).rename_axis("account")
    in_agg["unique_in_counterparties"] = df.groupby("to_account")["from_account"].nunique()
    in_agg["incoming_concentration"] = in_agg["_in_max"] / (in_agg["incoming_amount_total"] + 1e-9)
    in_agg["incoming_entropy"] = df.groupby("to_account")["amount_received"].apply(
        lambda x: _entropy(x.values)
    )

    from_ts = df[["from_account", "timestamp"]].rename(columns={"from_account": "account"})
    to_ts = df[["to_account", "timestamp"]].rename(columns={"to_account": "account"})
    all_ts = pd.concat([from_ts, to_ts], ignore_index=True).sort_values(["account", "timestamp"])
    all_ts["gap_days"] = (
        all_ts["timestamp"] - all_ts.groupby("account")["timestamp"].shift(1)
    ).dt.total_seconds() / 86400.0
    gap_agg = (
        all_ts.groupby("account")["gap_days"]
        .agg(max_gap_days="max", mean_gap_days="mean")
        .fillna(0.0)
    )

    out_times_dict = df.groupby("from_account")["timestamp"].apply(list).to_dict()
    in_times_dict = df.groupby("to_account")["timestamp"].apply(list).to_dict()

    # ----- assemble per-account features -----
    # Only relay_score still needs a per-account Python loop (variable-length
    # timestamp comparisons). Everything graph-related is now a precomputed
    # numpy array indexed straight from NetworKit — no per-node traversal.
    loop_records = []
    for i in node_idxs:
        node = idx_to_acc[i]
        relay = _relay_score(
            in_times_dict.get(node, []),
            out_times_dict.get(node, []),
        )
        loop_records.append({
            "account": node,
            "in_degree": int(in_degree_arr[i]),
            "out_degree": int(out_degree_arr[i]),
            "betweenness_centrality": float(betweenness_arr[i]),
            "eigenvector_centrality": float(eigenvector_arr[i]),
            "clustering_coefficient": float(clustering_arr[i]),
            "relay_score": relay,
            # NOTE: previously this fell back to clustering_coefficient anyway
            # for any node above DENSITY_DEGREE_CUTOFF, and the per-node
            # subgraph-density computation for the rest was itself a real cost.
            # LocalClusteringCoefficient is cheap at any graph size in
            # NetworKit, so we use it uniformly here instead.
            "local_density": float(clustering_arr[i]),
            "short_cycle_score": float(max(0, scc_size_arr[i] - 1)),
            "community_size": int(community_size_arr[i]),
            "is_isolated": int(in_degree_arr[i] == 0 and out_degree_arr[i] == 0),
        })

    feat = pd.DataFrame(loop_records).set_index("account")
    feat = feat.join(out_agg.drop(columns=["_out_max"]), how="left")
    feat = feat.join(in_agg.drop(columns=["_in_max"]), how="left")
    feat = feat.join(gap_agg, how="left")

    feat["tx_count"] = (
        feat["_out_tx_count"].fillna(0) + feat["_in_tx_count"].fillna(0)
    ).astype(int)
    feat["velocity_7d"] = feat["tx_count"] / max(global_span_days / temporal_window_days, 1.0)

    feat["weighted_in_degree"] = feat["incoming_amount_total"].fillna(0)
    feat["weighted_out_degree"] = feat["outgoing_amount_total"].fillna(0)
    feat["weighted_degree"] = feat["weighted_in_degree"] + feat["weighted_out_degree"]
    feat["total_unique_counterparties"] = (
        feat["unique_in_counterparties"].fillna(0) + feat["unique_out_counterparties"].fillna(0)
    ).astype(int)

    feat = feat.drop(columns=["_out_tx_count", "_in_tx_count"], errors="ignore")
    feat = feat.fillna(0.0).reset_index()

    feat["in_out_degree_ratio"] = feat["in_degree"] / (feat["out_degree"] + 1.0)
    feat["in_out_amount_ratio"] = feat["incoming_amount_total"] / (feat["outgoing_amount_total"] + 1.0)
    feat["cashflow_ratio"] = feat["incoming_amount_total"] / (feat["outgoing_amount_total"] + 1.0)
    feat["fanout_ratio"] = feat["unique_out_counterparties"] / (feat["unique_in_counterparties"] + 1.0)
    feat["degree_balance"] = (feat["in_degree"] - feat["out_degree"]).abs()
    feat["amount_balance"] = (feat["incoming_amount_total"] - feat["outgoing_amount_total"]).abs()

    return feat


def aggregate_labels_to_account(df: pd.DataFrame) -> pd.DataFrame:
    if "is_laundering" not in df.columns or df["is_laundering"].isna().all():
        return pd.DataFrame(columns=["account", "label"])
    positives = set(df.loc[df["is_laundering"] == 1, "from_account"].astype(str)) | set(
        df.loc[df["is_laundering"] == 1, "to_account"].astype(str)
    )
    all_accounts = set(df["from_account"].astype(str)) | set(df["to_account"].astype(str))
    return pd.DataFrame({
        "account": sorted(all_accounts),
        "label": [1 if a in positives else 0 for a in sorted(all_accounts)],
    })
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from math import log2
from typing import Dict, Iterable, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from networkx.algorithms.community import greedy_modularity_communities, label_propagation_communities

def _entropy(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[values > 0]
    if len(values) == 0:
        return 0.0
    p = values / values.sum()
    return float(-(p * np.log2(p)).sum())

def build_transaction_graph(df: pd.DataFrame) -> nx.DiGraph:
    g = nx.DiGraph()

    #adding all unique nodes in one shot
    all_accounts = pd.concat([
        df["from_account"].astype(str),
        df["to_account"].astype(str),
    ]).unique()
    g.add_nodes_from(all_accounts)

    #pre-cast once
    df_e = df.copy()
    df_e["from_account"] = df_e["from_account"].astype(str)
    df_e["to_account"] = df_e["to_account"].astype(str)
    df_e["amount_received"] = df_e["amount_received"].fillna(0.0)

    #iterate over unique (from, to) pairs
    for (u,v), grp in df_e.groupby(["from_account", "to_account"]):
        g.add_edge(
            u, v,
            amount=float(grp["amount_received"].sum()),
            count=len(grp),
            amounts=grp["amount_received"].tolist(),
            timestamps=grp["timestamp"].tolist(),
            payment_formats=grp["payment_format"].astype(str).tolist() if "payment_format" in df.columns else [],
            receiving_currencies=grp["receiving_currency"].astype(str).tolist() if "receiving_currency" in df.columns else [],
            payment_currencies=grp["payment_currency"].astype(str).tolist() if "payment_currency" in df.columns else [],
        )
    return g

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


def compute_account_features(df: pd.DataFrame, temporal_window_days: int = 7) -> pd.DataFrame:
    g = build_transaction_graph(df)
    all_nodes = list(g.nodes())

    if not all_nodes:
        return pd.DataFrame()
    
    undirected = g.to_undirected(as_view=False)
    clustering = nx.clustering(undirected)

    try:
        eigenvector = nx.eigenvector_centrality(undirected, max_iter=200, tol=1e-4)
    except Exception:
        eigenvector = {n: 0.0 for n in g.nodes()}
    
    try:
        community_size_map = {}
        for community in label_propagation_communities(undirected):
            size = len(community)
            for node in community:
                community_size_map[node] = size
    except Exception:
        community_size_map = {n: 1 for n in g.nodes()}

    if len(g) <= 1000:
        betweenness = nx.betweenness_centrality(g, normalized=True)
    else:
        sample_k = min(200, len(g))
        betweenness = nx.betweenness_centrality(g, k=sample_k, seed=42, normalized=True)
    
    scc_map = {}
    for component in nx.strongly_connected_components(g):
        size = len(component)
        for node in component:
            scc_map[node] = size 
    
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["from_account"] = df["from_account"].astype(str)
    df["to_account"] = df["to_account"].astype(str)
    df["amount_received"] = df["amount_received"].fillna(0.0)

    global_span_days = max(
        (df["timestamp"].max() - df["timestamp"].min()).total_seconds() / 86400.0,
        1.0,
    )
    # Outgoing stats
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

    # Incoming stats
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

    # Gap stats - vectorised sort+diff across combined in+out timeline per account
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

    # Pre-build time-lists for relay score (dict lookup is much faster than repeated get_group)
    out_times_dict = df.groupby("from_account")["timestamp"].apply(list).to_dict()
    in_times_dict = df.groupby("to_account")["timestamp"].apply(list).to_dict()

    # --- Per-node loop — now only relay_score and local_density remain ---
    DENSITY_DEGREE_CUTOFF = 100  # skip expensive subgraph for very high-degree nodes
    loop_records = []
    for node in all_nodes:
        in_deg = g.in_degree(node)
        out_deg = g.out_degree(node)

        relay = _relay_score(
            in_times_dict.get(node, []),
            out_times_dict.get(node, []),
        )

        nbrs = set(g.predecessors(node)) | set(g.successors(node))
        if 1 < len(nbrs) <= DENSITY_DEGREE_CUTOFF:
            sub = undirected.subgraph(nbrs | {node})
            local_density = sub.number_of_edges() / max(len(sub) * (len(sub) - 1) / 2, 1)
        else:
            # clustering coefficient is equivalent for high-degree nodes
            local_density = float(clustering.get(node, 0.0))

        loop_records.append({
            "account": node,
            "in_degree": in_deg,
            "out_degree": out_deg,
            "betweenness_centrality": float(betweenness.get(node, 0.0)),
            "eigenvector_centrality": float(eigenvector.get(node, 0.0)),
            "clustering_coefficient": float(clustering.get(node, 0.0)),
            "relay_score": relay,
            "local_density": local_density,
            "short_cycle_score": float(max(0, scc_map.get(node, 1) - 1)),
            "community_size": int(community_size_map.get(node, 1)),
            "is_isolated": int(in_deg == 0 and out_deg == 0),
        })    

    # --- Assemble final DataFrame ---
    feat = pd.DataFrame(loop_records).set_index("account")
    feat = feat.join(out_agg.drop(columns=["_out_max"]), how="left")
    feat = feat.join(in_agg.drop(columns=["_in_max"]), how="left")
    feat = feat.join(gap_agg, how="left")

    # tx_count and velocity from the vectorised counts
    feat["tx_count"] = (
        feat["_out_tx_count"].fillna(0) + feat["_in_tx_count"].fillna(0)
    ).astype(int)
    feat["velocity_7d"] = feat["tx_count"] / max(global_span_days / temporal_window_days, 1.0)

    # Aliases expected by models.py and explain.py
    feat["weighted_in_degree"] = feat["incoming_amount_total"].fillna(0)
    feat["weighted_out_degree"] = feat["outgoing_amount_total"].fillna(0)
    feat["weighted_degree"] = feat["weighted_in_degree"] + feat["weighted_out_degree"]
    feat["total_unique_counterparties"] = (
        feat["unique_in_counterparties"].fillna(0) + feat["unique_out_counterparties"].fillna(0)
    ).astype(int)

    feat = feat.drop(columns=["_out_tx_count", "_in_tx_count"], errors="ignore")
    feat = feat.fillna(0.0).reset_index()

    # Derived ratios — unchanged from original
    feat["in_out_degree_ratio"] = feat["in_degree"] / (feat["out_degree"] + 1.0)
    feat["in_out_amount_ratio"] = feat["incoming_amount_total"] / (feat["outgoing_amount_total"] + 1.0)
    feat["cashflow_ratio"] = feat["incoming_amount_total"] / (feat["outgoing_amount_total"] + 1.0)
    feat["fanout_ratio"] = feat["unique_out_counterparties"] / (feat["unique_in_counterparties"] + 1.0)
    feat["degree_balance"] = (feat["in_degree"] - feat["out_degree"]).abs()
    feat["amount_balance"] = (feat["incoming_amount_total"] - feat["outgoing_amount_total"]).abs()

    return feat


def aggregate_labels_to_account(df: pd.DataFrame) -> pd.DataFrame:
    # account is positive if it participates in any laundering transaction
    if "is_laundering" not in df.columns or df["is_laundering"].isna().all():
        return pd.DataFrame(columns=["account", "label"])
    positives = set(df.loc[df["is_laundering"] == 1, "from_account"].astype(str)) | set(df.loc[df["is_laundering"] == 1, "to_account"].astype(str))
    all_accounts = set(df["from_account"].astype(str)) | set(df["to_account"].astype(str))
    return pd.DataFrame({"account": sorted(all_accounts), "label": [1 if a in positives else 0 for a in sorted(all_accounts)]})

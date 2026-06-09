from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from math import log2
from typing import Dict, Iterable, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd

def _entropy(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[values > 0]
    if len(values) == 0:
        return 0.0
    p = values / values.sum()
    return float(-(p * np.log2(p)).sum())

def build_transaction_graph(df: pd.DataFrame) -> nx.DiGraph:
    g = nx.DiGraph()
    for _, row in df.iterrows():
        u = str(row["from_account"])
        v = str(row["to_account"])
        amount = float(row.get("amount_received", 0.0) or 0.0)
        ts = row["timestamp"]
        g.add_node(u)
        g.add_node(v)
        if g.has_edge(u, v):
            g[u][v]["amounts"].append(amount)
            g[u][v]["timestamps"].append(ts)
            g[u][v]["count"] += 1
        else:
            g.add_edge(
                u,
                v,
                amount=float(amount),
                count=1,
                amounts=[amount],
                timestamps=[ts],
                payment_formats=[str(row.get("payment_format", ""))],
                receiving_currencies=[str(row.get("receiving_currency", ""))],
                payment_currencies=[str(row.get("payment_currency", ""))],
            )
    return g

def _relay_score(in_times: list[pd.Timestamp], out_times: list[pd.Timestamp]) -> float:
    if not in_times or not out_times:
        return 0.0
    in_times = sorted(pd.to_datetime(in_times).tolist())
    out_times = sorted(pd.to_datetime(out_times).tolist())
    # if outgoing activity typically follows incoming activity within a few days, score rises
    deltas = []
    for tin in in_times:
        later = [tout for tout in out_times if tout >= tin]
        if later:
            deltas.append((later[0] - tin).total_seconds() / 86400.0)
    if not deltas:
        return 0.0
    mean_delta = float(np.mean(deltas))
    return float(np.exp(-mean_delta / 3.0))

def _cycle_participation(g: nx.DiGraph, node: str, max_depth: int = 4) -> float:
    # bounded DFS for short cycles; cheap enough for prototype use
    try:
        count = 0
        stack = [(node, [node])]
        while stack:
            cur, path = stack.pop()
            if len(path) > max_depth:
                continue
            for nbr in g.successors(cur):
                if nbr == node and len(path) >= 2:
                    count += 1
                elif nbr not in path:
                    stack.append((nbr, path + [nbr]))
        return float(min(count, 10))
    except Exception:
        return 0.0

def compute_account_features(df: pd.DataFrame, temporal_window_days: int = 7) -> pd.DataFrame:
    g = build_transaction_graph(df)

    # Basic node sets
    all_nodes = list(g.nodes())
    records = []

    # precompute global centralities where feasible
    undirected = g.to_undirected(as_view=False)
    clustering = nx.clustering(undirected)
    if len(g) <= 1000:
        betweenness = nx.betweenness_centrality(g, normalized=True)
    else:
        sample_k = min(200, len(g))
        betweenness = nx.betweenness_centrality(g, k=sample_k, seed=42, normalized=True)

    # temporal grouping
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    grouped_out = df.groupby("from_account")
    grouped_in = df.groupby("to_account")

    # time span for velocity
    global_span_days = max((df["timestamp"].max() - df["timestamp"].min()).total_seconds() / 86400.0, 1.0)

    for node in all_nodes:
        out_rows = grouped_out.get_group(node) if node in grouped_out.groups else pd.DataFrame(columns=df.columns)
        in_rows = grouped_in.get_group(node) if node in grouped_in.groups else pd.DataFrame(columns=df.columns)

        out_amt = out_rows["amount_received"].fillna(0).astype(float) if len(out_rows) else pd.Series(dtype=float)
        in_amt = in_rows["amount_received"].fillna(0).astype(float) if len(in_rows) else pd.Series(dtype=float)

        out_times = out_rows["timestamp"].tolist() if len(out_rows) else []
        in_times = in_rows["timestamp"].tolist() if len(in_rows) else []

        # velocity: transactions per window
        tx_count = len(out_rows) + len(in_rows)
        velocity = tx_count / max(global_span_days / temporal_window_days, 1.0)

        # dormancy/reactivation: max gap in outgoing or incoming activity
        combined_times = sorted(out_times + in_times)
        if len(combined_times) >= 2:
            gaps = np.diff(pd.Series(combined_times).astype("int64")) / 1e9 / 86400.0
            max_gap = float(np.max(gaps))
            mean_gap = float(np.mean(gaps))
        else:
            max_gap = 0.0
            mean_gap = 0.0

        out_counterparties = set(out_rows["to_account"].astype(str).tolist()) if len(out_rows) else set()
        in_counterparties = set(in_rows["from_account"].astype(str).tolist()) if len(in_rows) else set()

        out_amounts = out_amt.to_numpy() if len(out_amt) else np.array([])
        in_amounts = in_amt.to_numpy() if len(in_amt) else np.array([])

        # amount concentration / entropy
        out_entropy = _entropy(out_amounts)
        in_entropy = _entropy(in_amounts)
        out_concentration = float(out_amounts.max() / (out_amounts.sum() + 1e-9)) if len(out_amounts) else 0.0
        in_concentration = float(in_amounts.max() / (in_amounts.sum() + 1e-9)) if len(in_amounts) else 0.0

        relay = _relay_score(in_times, out_times)
        cycle_part = _cycle_participation(g, node)

        # community-ish density proxy using local neighborhood
        nbrs = set(g.predecessors(node)) | set(g.successors(node))
        local_density = 0.0
        if len(nbrs) > 1:
            sub = undirected.subgraph(nbrs | {node})
            possible = max(len(sub) * (len(sub) - 1) / 2, 1)
            local_density = sub.number_of_edges() / possible

        records.append(
            {
                "account": node,
                "in_degree": g.in_degree(node),
                "out_degree": g.out_degree(node),
                "weighted_in_degree": sum(in_amt) if len(in_amt) else 0.0,
                "weighted_out_degree": sum(out_amt) if len(out_amt) else 0.0,
                "weighted_degree": (sum(in_amt) if len(in_amt) else 0.0) + (sum(out_amt) if len(out_amt) else 0.0),
                "betweenness_centrality": float(betweenness.get(node, 0.0)),
                "clustering_coefficient": float(clustering.get(node, 0.0)),
                "tx_count": tx_count,
                "velocity_7d": velocity,
                "incoming_amount_total": float(sum(in_amt) if len(in_amt) else 0.0),
                "outgoing_amount_total": float(sum(out_amt) if len(out_amt) else 0.0),
                "incoming_amount_mean": float(np.mean(in_amt) if len(in_amt) else 0.0),
                "outgoing_amount_mean": float(np.mean(out_amt) if len(out_amt) else 0.0),
                "incoming_amount_std": float(np.std(in_amt) if len(in_amt) else 0.0),
                "outgoing_amount_std": float(np.std(out_amt) if len(out_amt) else 0.0),
                "incoming_entropy": in_entropy,
                "outgoing_entropy": out_entropy,
                "incoming_concentration": in_concentration,
                "outgoing_concentration": out_concentration,
                "unique_in_counterparties": len(in_counterparties),
                "unique_out_counterparties": len(out_counterparties),
                "total_unique_counterparties": len(in_counterparties | out_counterparties),
                "max_gap_days": max_gap,
                "mean_gap_days": mean_gap,
                "relay_score": relay,
                "short_cycle_score": cycle_part,
                "local_density": local_density,
                "is_isolated": int(g.in_degree(node) == 0 and g.out_degree(node) == 0),
            }
        )

    feat = pd.DataFrame(records).fillna(0.0)

    # Add robust ratios
    feat["in_out_degree_ratio"] = feat["in_degree"] / (feat["out_degree"] + 1.0)
    feat["in_out_amount_ratio"] = feat["incoming_amount_total"] / (feat["outgoing_amount_total"] + 1.0)
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

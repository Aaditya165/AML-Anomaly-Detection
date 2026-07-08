"""
community_detection.py
------------------------
Consumes the ACCOUNT graph (graph_construction.build_account_graph).
Produces Leiden communities, random-walk communities, and per-community
statistics. Nothing else -- flow_features.py is a separate concern even
though it runs over the same graph.

Section 4.1 "Community Detection" -- two complementary views:

  * Top-down (Leiden / modularity): one GLOBAL, non-overlapping community
    per account -- "what context does this account operate in, seen from
    the whole network".
  * Bottom-up (random-walk / personalized PageRank): one LOCAL, overlapping
    community PER ACCOUNT, built from an up-to-`2*RANDOM_WALK_N_HOPS`-hop
    directed neighborhood. Overlapping matters here: a laundering agent
    can sit at the center of several such neighborhoods at once, which a
    non-overlapping partition can't represent.

Section 4.4 "Communities (or subgraphs) Features" -- for each community
(from either view), computes membership stats (how many dispense-only /
sink-only / passthrough accounts it contains), network properties (degree
distribution, density, diameter, assortativity, biconnected components,
articulation points), turnover volumes, and Eq. 2's amount-weighted
temporal stats.
"""

from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import igraph as ig
import leidenalg as la
from joblib import Parallel, delayed

from . import config


# ---------------------------------------------------------------------------
# Leiden (top-down, non-overlapping)
# ---------------------------------------------------------------------------

def leiden_communities(graph: ig.Graph, seed: int = config.SEED) -> Dict[str, str]:
    """Returns {account_id: community_id}. Two accounts share a community
    iff they map to the same (opaque, string) community_id."""
    weights = graph.es["weight"] if "weight" in graph.es.attributes() else None
    partition = la.find_partition(
        graph, la.ModularityVertexPartition,
        n_iterations=config.LEIDEN_N_ITERATIONS, weights=weights, seed=seed,
    )
    node_names = graph.vs["name"]
    return {node_names[i]: f"leiden_{comm_idx}" for i, comm_idx in enumerate(partition.membership)}



# ---------------------------------------------------------------------------
# Section 4.4: per-community statistics (shared by both community views)
# ---------------------------------------------------------------------------

def _classify_node_types(source_series: pd.Series, target_series: pd.Series) -> Tuple[set, set, set]:
    """Section 3.1.3 / Figure 1: dispense-only (source, never target),
    sink-only (target, never source), passthrough (both)."""
    sources, targets = set(source_series.unique()), set(target_series.unique())
    return sources - targets, targets - sources, sources & targets


def _build_member_row_index(df: pd.DataFrame, source_col: str, target_col: str) -> Dict[str, np.ndarray]:
    """account -> array of row positions where it appears as sender or
    receiver (df must already have a default 0..n-1 RangeIndex)."""
    combined: Dict[str, list] = {}
    for mapping in (df.groupby(source_col, observed=True).indices, df.groupby(target_col, observed=True).indices):
        for key, positions in mapping.items():
            combined.setdefault(key, []).append(positions)
    return {key: np.unique(np.concatenate(parts)) for key, parts in combined.items()}


def _network_properties(sub_edges: pd.DataFrame) -> dict:
    if sub_edges.empty:
        return {}
    edge_df = sub_edges[["source", "target"]].copy()
    edge_df["source"] = edge_df["source"].astype(str)
    edge_df["target"] = edge_df["target"].astype(str)

    graph = ig.Graph.DataFrame(
        edge_df,
        directed=True,
        use_vids=False,
    )
    props = {"num_nodes": graph.vcount(), "num_edges": graph.ecount(), "density": graph.density(loops=False)}
    degrees_all = graph.degree(mode="all")
    props["max_degree"] = max(degrees_all) if degrees_all else 0
    props["max_degree_in"] = max(graph.degree(mode="in")) if graph.vcount() else 0
    props["max_degree_out"] = max(graph.degree(mode="out")) if graph.vcount() else 0
    for key, fn in [
        ("assortativity_degree", lambda: graph.assortativity_degree(directed=True)),
        ("assortativity_degree_ud", lambda: graph.assortativity_degree(directed=False)),
        ("diameter", lambda: graph.diameter(directed=True, unconn=True)),
        ("diameter_ud", lambda: graph.diameter(directed=False, unconn=True)),
    ]:
        try:
            props[key] = fn()
        except Exception:
            props[key] = np.nan
    try:
        biconn, articulation = graph.biconnected_components(return_articulation_points=True)
        props["num_biconn_components"] = len(biconn)
        props["num_articulation_points"] = len(articulation)
    except Exception:
        props["num_biconn_components"], props["num_articulation_points"] = 0, 0
    return props


def _turnover_stats(sub_edges: pd.DataFrame) -> dict:
    if sub_edges.empty:
        return {}
    credit = sub_edges.groupby("target")["amount"].sum()
    debit = sub_edges.groupby("source")["amount"].sum()
    balance = pd.concat([credit.rename("credit"), debit.rename("debit")], axis=1).fillna(0.0)
    turnover = float((balance["credit"] - balance["debit"]).abs().sum())
    edge_totals = sub_edges.groupby(["source", "target"])["amount"].sum()
    return {
        "turnover": turnover,
        "total_amount": float(sub_edges["amount"].sum()),
        "mean_edge_amount": float(edge_totals.mean()),
        "median_edge_amount": float(edge_totals.median()),
        "max_edge_amount": float(edge_totals.max()),
        "std_edge_amount": float(edge_totals.std() or 0.0),
    }


def _weighted_time_stats(sub_edges: pd.DataFrame) -> dict:
    """Equation 2 (+ weighted std/median in the same spirit)."""
    if sub_edges.empty:
        return {}
    ts = sub_edges["timestamp"].values.astype("datetime64[ns]").astype(np.int64).astype(np.float64)
    trend = ts - ts.min()
    weights = sub_edges["amount"].to_numpy().astype(np.float64)
    if weights.sum() <= 0:
        weights = np.ones_like(weights)
    mean = float(np.average(trend, weights=weights))
    variance = float(np.average((trend - mean) ** 2, weights=weights))
    std = float(np.sqrt(max(variance, 0.0)))
    order = np.argsort(trend)
    sorted_w, sorted_t = weights[order], trend[order]
    cum = sorted_w.cumsum()
    median = float(np.interp(0.5, (cum - sorted_w / 2) / cum[-1], sorted_t)) if cum[-1] > 0 else float(trend.mean())
    return {"weighted_time_mean": mean, "weighted_time_std": std, "weighted_time_median": median}


def _one_community(key, members: Iterable[str], df_reset: pd.DataFrame, member_rows: Dict[str, np.ndarray],
                    dispense_only: set, sink_only: set, passthrough_set: set) -> Tuple[object, dict]:
    members = set(members)
    if not members:
        return key, {}
    position_arrays = [member_rows[m] for m in members if m in member_rows]
    if not position_arrays:
        return key, {}
    # Vectorized (numpy-only) union of row positions -- NOT
    # `set().union(*(arr.tolist() for arr in position_arrays))`. The
    # Python-level version has to materialize every member's position
    # array as a boxed-int Python list simultaneously (via *args
    # unpacking) before union even starts, which is fine for a
    # 2-member community and a genuine OOM risk for a 30,000-member one
    # built around a hub account touching a large fraction of the file.
    candidate_positions = np.unique(np.concatenate(position_arrays))
    sub = df_reset.iloc[candidate_positions]
    mask = sub["source"].isin(members).to_numpy() & sub["target"].isin(members).to_numpy()
    sub = sub.loc[mask]

    row = {
        "community_size": len(members),
        "num_dispense_members": len(members & dispense_only),
        "num_sink_members": len(members & sink_only),
        "num_passthrough_members": len(members & passthrough_set),
    }
    if len(members) <= config.MAX_GRAPH_METRICS_COMMUNITY_SIZE:
        row.update(_network_properties(sub))
    else:
        row.update({
            "num_nodes": np.nan,
            "num_edges": np.nan,
            "density": np.nan,
            "max_degree": np.nan,
            "max_degree_in": np.nan,
            "max_degree_out": np.nan,
            "assortativity_degree": np.nan,
            "assortativity_degree_ud": np.nan,
            "diameter": np.nan,
            "diameter_ud": np.nan,
            "num_biconn_components": np.nan,
            "num_articulation_points": np.nan,
        })

    # Same size gate as graph metrics -- `sub` for an oversized community
    # can itself be a large fraction of the whole transaction table (a
    # hub-dominated 30k-account community plausibly touches millions of
    # rows), and groupby/sort/argsort over that, repeated across several
    # oversized communities in the same run, adds up fast. Unlike graph
    # metrics these are cheap to *approximate* instead of skip: sample
    # down to a bounded number of rows rather than NaN them out entirely.
    if len(sub) > config.MAX_TURNOVER_STATS_ROWS:
        sub_for_stats = sub.sample(n=config.MAX_TURNOVER_STATS_ROWS, random_state=config.SEED)
    else:
        sub_for_stats = sub

    row.update(_turnover_stats(sub_for_stats))
    row.update(_weighted_time_stats(sub_for_stats))
    return key, row


def compute_community_statistics(
    df: pd.DataFrame,
    communities: Dict[object, Iterable[str]],
    source_col: str = "Sender Account",
    target_col: str = "Receiver Account",
    amount_col: str = "Amount Paid",
    timestamp_col: str = "Timestamp",
    n_jobs: int = 1,
    prefix: str = "community",
) -> pd.DataFrame:
    """
    Calculates statistics for communities. Automatically routes non-overlapping
    communities (Leiden) to a memory-efficient global Pandas groupby, bypassing
    the row-level concatenation that causes OOMs on massive hub communities.
    """
    df_reset = df[[source_col, target_col, amount_col, timestamp_col]].reset_index(drop=True)
    df_reset = df_reset.rename(
        columns={source_col: "source", target_col: "target", amount_col: "amount", timestamp_col: "timestamp"}
    )
    
    # Coerce to string so dict lookups match the graph vertex names exactly
    df_reset["source"] = df_reset["source"].astype(str)
    df_reset["target"] = df_reset["target"].astype(str)
    
    dispense_only, sink_only, passthrough_set = _classify_node_types(df_reset["source"], df_reset["target"])

    # Check if communities are non-overlapping (Leiden) to use the Fast Path
    node_to_comm = {}
    is_overlapping = False
    for c_id, members in communities.items():
        for m in members:
            if m in node_to_comm:
                is_overlapping = True
                break
            node_to_comm[m] = c_id
        if is_overlapping:
            break
            
    if not is_overlapping:
        # =================================================================
        # FAST PATH: Non-overlapping (Leiden)
        # Eliminates OOM by globally identifying internal edges in pandas
        # =================================================================
        print("Executing memory-optimized Fast Path for non-overlapping communities...")
        df_reset["source_comm"] = df_reset["source"].map(node_to_comm)
        df_reset["target_comm"] = df_reset["target"].map(node_to_comm)
        
        # Keep only edges where source and target are in the same community
        internal_edges = df_reset[
            (df_reset["source_comm"] == df_reset["target_comm"]) & 
            (df_reset["source_comm"].notna())
        ]
        grouped_internal = internal_edges.groupby("source_comm")
        
        def _fast_one(key, members):
            members = set(members)
            if not members:
                return key, {}
                
            try:
                sub = grouped_internal.get_group(key)
            except KeyError:
                sub = pd.DataFrame(columns=["source", "target", "amount", "timestamp"])
                
            row = {
                "community_size": len(members),
                "num_dispense_members": len(members & dispense_only),
                "num_sink_members": len(members & sink_only),
                "num_passthrough_members": len(members & passthrough_set),
            }
            
            if len(members) <= config.MAX_GRAPH_METRICS_COMMUNITY_SIZE:
                row.update(_network_properties(sub))
            else:
                row.update({
                    "num_nodes": np.nan, "num_edges": np.nan, "density": np.nan,
                    "max_degree": np.nan, "max_degree_in": np.nan, "max_degree_out": np.nan,
                    "assortativity_degree": np.nan, "assortativity_degree_ud": np.nan,
                    "diameter": np.nan, "diameter_ud": np.nan,
                    "num_biconn_components": np.nan, "num_articulation_points": np.nan,
                })

            if len(sub) > config.MAX_TURNOVER_STATS_ROWS:
                sub_for_stats = sub.sample(n=config.MAX_TURNOVER_STATS_ROWS, random_state=config.SEED)
            else:
                sub_for_stats = sub

            row.update(_turnover_stats(sub_for_stats))
            row.update(_weighted_time_stats(sub_for_stats))
            return key, row

        results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_fast_one)(key, members) for key, members in communities.items()
        )
        
    else:
        # =================================================================
        # SLOW PATH: Overlapping (Random Walks)
        # Uses original row-index intersection logic
        # =================================================================
        print("Executing standard extraction for overlapping communities...")
        member_rows = _build_member_row_index(df_reset, "source", "target")
        results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_one_community)(key, members, df_reset, member_rows, dispense_only, sink_only, passthrough_set)
            for key, members in communities.items()
        )
        
    rows = {key: feats for key, feats in results if feats}
    out = pd.DataFrame.from_dict(rows, orient="index")
    if not out.empty:
        out = out.add_prefix(f"{prefix}_")
    out.index.name = "key"
    return out


def leiden_groups_from_membership(leiden_membership: Dict[str, str]) -> Dict[str, Set[str]]:
    """Invert {account: community_id} -> {community_id: {members}}."""
    groups: Dict[str, Set[str]] = defaultdict(set)
    for account, community_id in leiden_membership.items():
        groups[community_id].add(account)
    return groups


def broadcast_leiden_features(leiden_membership: Dict[str, str], leiden_stats: pd.DataFrame) -> pd.DataFrame:
    """Each account gets its own community's feature row (rather than one
    row per community) -- this is what makes Leiden's output joinable
    onto the account-indexed table alongside random-walk's already-1:1
    output. See feature_merge.py."""
    table = pd.Series(leiden_membership, name="_leiden_community_id").to_frame()
    table = table.join(leiden_stats, on="_leiden_community_id")
    del table["_leiden_community_id"]
    return table

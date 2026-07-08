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
# Random-walk (bottom-up, overlapping)
# ---------------------------------------------------------------------------

def _build_candidate_neighborhoods(edges_agg: pd.DataFrame, top_n: int) -> Dict[str, Set[str]]:
    """For every account, its 2-hop *candidate* neighborhood: the union of
    its own top-N (by undirected weight) neighbors' top-N neighbors. This
    bounded pool is what personalized PageRank then runs over, so a hub
    account can't blow up every other account's induced subgraph.

    This is the ORIGINAL candidate-construction strategy: purely local,
    doesn't need Leiden's output, and needs `target_accounts` (in
    `random_walk_communities`) to bound total cost since every account's
    candidate pool is built independently regardless of global structure.
    See `_build_candidate_neighborhoods_from_leiden` for an alternative
    that nests inside the already-computed Leiden partition instead --
    that one bounds cost from the community-size side rather than needing
    you to restrict which accounts get processed at all.
    """
    fwd = edges_agg.loc[:, ["source", "target", "amount"]]
    rev = fwd.rename(columns={"source": "target", "target": "source"})
    undirected = pd.concat([fwd, rev], ignore_index=True)
    undirected = undirected.groupby(["source", "target"], observed=True)["amount"].sum().reset_index()
    undirected = undirected.sort_values("amount", ascending=False)
    neighbor_map = (
        undirected.groupby("source", sort=False).head(top_n).groupby("source")["target"].apply(set).to_dict()
    )

    neighborhoods: Dict[str, Set[str]] = {}
    for node, neighbors in neighbor_map.items():
        candidates = {node} | neighbors
        for nb in neighbors:
            candidates |= neighbor_map.get(nb, set())
        neighborhoods[node] = candidates
    return neighborhoods


def _build_candidate_neighborhoods_from_leiden(
    edges_agg: pd.DataFrame,
    leiden_membership: Dict[str, str],
    include_bridge_neighbors: bool = True,
    max_candidates: int = 500,
) -> Dict[str, Set[str]]:
    """
    Alternative to `_build_candidate_neighborhoods`: candidate pool = the
    account's own Leiden community, optionally unioned with the Leiden
    communities of its DIRECT neighbors (so cross-community bridging --
    the entire reason bottom-up detection exists ALONGSIDE Leiden, per
    Section 4.1 -- isn't silently defeated by nesting inside Leiden's
    partition; an account whose direct neighbor sits in a different
    community can still reach that neighbor's community).

    Every account gets a candidate pool this way regardless of
    `target_accounts` -- Leiden's own partition does the cost-bounding
    instead of you needing to restrict which accounts get processed. This
    means you can run `random_walk_communities` for every account without
    losing coverage the way a blind `target_accounts` restriction would.

    `max_candidates`: hard safety cap. Leiden is NOT guaranteed to produce
    evenly-sized communities on every graph -- a sufficiently dominant hub
    (or a small number of them) can still pull a large fraction of the
    graph into one community (measured as high as ~11% of all accounts in
    one community on a hub-heavy synthetic test). When a pool exceeds this
    cap, the account's OWN community is kept in full (that's the primary
    context) and bridge-community members are added, sorted by the
    bridging edge's own weight, only until the cap is reached -- rather
    than silently processing an enormous subgraph for the unlucky
    accounts that happen to share a community with a hub.
    """
    community_members = leiden_groups_from_membership(leiden_membership)

    direct_neighbors: Dict[str, list] = {}
    if include_bridge_neighbors:
        fwd = edges_agg.loc[:, ["source", "target", "amount"]]
        rev = fwd.rename(columns={"source": "target", "target": "source"})
        undirected = pd.concat([fwd, rev], ignore_index=True)
        undirected = undirected.groupby(["source", "target"], observed=True)["amount"].sum().reset_index()
        undirected = undirected.sort_values("amount", ascending=False)
        for source, group in undirected.groupby("source", sort=False):
            direct_neighbors[source] = list(group["target"])  # already sorted by weight desc

    neighborhoods: Dict[str, Set[str]] = {}
    for node, community_id in leiden_membership.items():
        own_community = community_members[community_id]

        if len(own_community) >= max_candidates:
            # own community alone already fills the budget -- no room for
            # bridging regardless; truncate deterministically.
            candidates = set(list(own_community)[:max_candidates])
        else:
            candidates = set(own_community)
            if include_bridge_neighbors:
                seen_bridge_communities = set()
                for neighbor in direct_neighbors.get(node, []):
                    remaining_budget = max_candidates - len(candidates)
                    if remaining_budget <= 0:
                        break
                    neighbor_community_id = leiden_membership.get(neighbor)
                    if neighbor_community_id is None or neighbor_community_id == community_id:
                        continue
                    if neighbor_community_id in seen_bridge_communities:
                        continue
                    seen_bridge_communities.add(neighbor_community_id)
                    bridge_members = community_members[neighbor_community_id]
                    if len(bridge_members) > remaining_budget:
                        # partial add -- fill the remaining budget only,
                        # rather than either dropping this whole bridge
                        # community or blowing past the cap
                        candidates |= set(list(bridge_members)[:remaining_budget])
                        break
                    candidates |= set(bridge_members)

        candidates.add(node)
        neighborhoods[node] = candidates
    return neighborhoods


def _personalized_pagerank_community(
    node: str, candidates: Set[str], graph: ig.Graph, node_index: Dict[str, int],
    damping: float, threshold: float,
) -> Tuple[str, Set[str]]:
    vids = [node_index[c] for c in candidates if c in node_index]
    if len(vids) <= 1:
        return node, {node}

    sub = graph.induced_subgraph(vids)
    sub_names = sub.vs["name"]
    name_to_subidx = {n: i for i, n in enumerate(sub_names)}
    if node not in name_to_subidx:
        return node, {node}

    weights = sub.es["weight"] if "weight" in sub.es.attributes() else None
    scores = np.asarray(sub.personalized_pagerank(
        reset_vertices=[name_to_subidx[node]], damping=damping, weights=weights, directed=True,
    ))
    max_score = scores.max() if scores.max() > 0 else 1.0
    keep = {sub_names[i] for i, sc in enumerate(scores) if (sc / max_score) >= threshold}
    keep.add(node)
    return node, keep


def _personalized_pagerank_batch(
    accounts: List[str], neighborhoods: Dict[str, Set[str]], graph: ig.Graph, node_index: Dict[str, int],
    damping: float, threshold: float,
) -> List[Tuple[str, Set[str]]]:
    """Same computation as calling `_personalized_pagerank_community` once
    per account, just grouped into one dispatched task -- at hundreds of
    thousands of accounts, one joblib task per account means hundreds of
    thousands of tiny dispatches, each paying its own scheduling overhead
    regardless of backend. Batching amortizes that overhead; it changes
    nothing about the result."""
    return [
        _personalized_pagerank_community(node, neighborhoods[node], graph, node_index, damping, threshold)
        for node in accounts
    ]


def random_walk_communities(
    edges_agg: pd.DataFrame,
    graph: ig.Graph,
    candidate_top_n: int = config.RANDOM_WALK_CANDIDATE_TOP_N,
    damping: float = config.RANDOM_WALK_PPR_DAMPING,
    threshold: float = config.RANDOM_WALK_MEMBERSHIP_THRESHOLD,
    n_jobs: int = config.RANDOM_WALK_N_JOBS,
    target_accounts: Optional[Iterable[str]] = None,
    batch_size: int = 200,
    leiden_membership: Optional[Dict[str, str]] = None,
    max_candidates_from_leiden: int = config.RANDOM_WALK_MAX_CANDIDATES_FROM_LEIDEN,
) -> Dict[str, Set[str]]:
    """
    One OVERLAPPING community per account in `target_accounts` (default:
    every account in the graph). Returns {account_id: {its local community
    members}}.

    Two ways to build each account's candidate pool (what personalized
    PageRank actually runs over):

      * `leiden_membership=None` (default) -- the original strategy: each
        account's own top-N-by-weight neighbors' top-N neighbors, built
        independently of any community structure. Needs `target_accounts`
        to bound total cost, since nothing here stops a hub-adjacent
        account's pool from ballooning.
      * `leiden_membership=<the dict from leiden_communities(...)>` --
        nests inside the ALREADY-COMPUTED Leiden partition instead: each
        account's pool is its own Leiden community (+ bridging neighbors'
        communities, capped at `max_candidates_from_leiden`). This lets
        you run every account without `target_accounts` at all, since
        Leiden's own partition (plus the hard cap) bounds the per-account
        subgraph size instead. See
        `_build_candidate_neighborhoods_from_leiden`'s docstring for the
        cross-community-bridging reasoning and the cap's safety role.

    Cost note, in order of expected impact:
      1. `target_accounts` (if not using `leiden_membership`) -- restrict
         to the accounts you actually need. Dwarfs everything else.
      2. `candidate_top_n` / `max_candidates_from_leiden` -- the dominant
         driver of PER-CALL cost.
      3. `batch_size` -- accounts are grouped into chunks of this size per
         dispatched joblib task, cutting per-task scheduling overhead
         without changing the result.
      `threshold` does NOT affect this stage's runtime -- PageRank runs
      over the full induced neighborhood regardless of threshold; it only
      changes which nodes survive into the returned community afterward.
    """
    if leiden_membership is not None:
        neighborhoods = _build_candidate_neighborhoods_from_leiden(
            edges_agg, leiden_membership, max_candidates=max_candidates_from_leiden,
        )
    else:
        neighborhoods = _build_candidate_neighborhoods(edges_agg, candidate_top_n)
    node_index = {name: i for i, name in enumerate(graph.vs["name"])}

    accounts_to_process = list(target_accounts) if target_accounts is not None else list(neighborhoods.keys())
    accounts_to_process = [a for a in accounts_to_process if a in neighborhoods]

    batches = [accounts_to_process[i:i + batch_size] for i in range(0, len(accounts_to_process), batch_size)]
    batch_results = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(_personalized_pagerank_batch)(batch, neighborhoods, graph, node_index, damping, threshold)
        for batch in batches
    )
    return dict(pair for batch in batch_results for pair in batch)


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
    graph = ig.Graph.DataFrame(sub_edges[["source", "target"]], directed=True, use_vids=False)
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
    row.update(_network_properties(sub))
    row.update(_turnover_stats(sub))
    row.update(_weighted_time_stats(sub))
    return key, row


def compute_community_statistics(
    df: pd.DataFrame,
    communities: Dict[object, Iterable[str]],
    source_col: str = "Sender Account",
    target_col: str = "Receiver Account",
    amount_col: str = "Amount Paid",
    timestamp_col: str = "Timestamp",
    n_jobs: int = config.RANDOM_WALK_N_JOBS,
    prefix: str = "community",
) -> pd.DataFrame:
    """
    `communities`: {key: iterable(member_account_ids)}. Works for BOTH
    community views:
      * Leiden      -> invert {account: community_id} into
                       {community_id: {members}} first (this module's
                       `leiden_communities` + a groupby), then broadcast
                       each community's row back onto its members
                       (feature_merge.py does this broadcast).
      * Random-walk -> already IS {account: {its own local members}}, so
                       the returned table is already account-indexed.

    Returns a DataFrame indexed by `key`, columns prefixed `{prefix}_`.
    """
    df_reset = df[[source_col, target_col, amount_col, timestamp_col]].reset_index(drop=True)
    df_reset = df_reset.rename(
        columns={source_col: "source", target_col: "target", amount_col: "amount", timestamp_col: "timestamp"}
    )
    member_rows = _build_member_row_index(df_reset, "source", "target")
    dispense_only, sink_only, passthrough_set = _classify_node_types(df_reset["source"], df_reset["target"])

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

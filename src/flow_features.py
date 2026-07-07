"""
flow_features.py
------------------
The heart of ExSTraQt. Consumes the account graph (and, for the temporal
version, the raw transaction stream) + community_detection.py's output is
NOT required here -- flow tracing and community detection are independent
views of the same account graph, not a pipeline stage that depends on the
other.

Section 4.2 + Algorithm 1 + Figure 3 ("dispense flow calculation"): once a
node sends (or receives) an amount, how much of it can still be traced
flowing forward through the network, hop by hop, and to how many distinct
downstream accounts? At each hop, the carried-forward amount is capped by
MIN(previous chain amount, the next edge's own amount), and only the
`FLOW_TOP_N` highest-amount continuations survive per hop.

Three account "profiles" (Section 3.1.3 / Figure 1), all built with the
same underlying tracing routine, pointed at a different (direction,
capping-total) pair:

    compute_dispense_features()     forward,  capped by the account's own SENT total     (placement)
    compute_sink_features()         backward, capped by the account's own RECEIVED total  (integration)
    compute_passthrough_features()  forward,  capped by the account's own RECEIVED total  (layering)

Section 4.3 "Flow-based Temporal Features" adds the chronological
constraint back in on the raw transaction stream (a dispense->passthrough
transaction and a passthrough->sink transaction only form a valid relay if
the second happened at or after the first):

    compute_temporal_flow_features()

The paper keeps these as separate stages; they're combined into one module
here since they're closely related and operate on the same underlying
graphs -- reduces duplication without losing the functional separation
(each is still its own function).
"""

from typing import Dict

import numpy as np
import pandas as pd

from . import config


# ---------------------------------------------------------------------------
# Section 4.2, Algorithm 1: aggregated-graph flow tracing
# ---------------------------------------------------------------------------

def _build_capped_base_table(edges_agg: pd.DataFrame, totals: Dict[str, float], top_n: int) -> pd.DataFrame:
    """For every account, its top-N highest-amount outgoing edges, capped
    at `totals[account]` (that account's own sent/received total). Reused
    as the join target at EVERY hop -- a node that shows up as an
    intermediate hop for one origin already has its own row here (it was
    computed for every account, not just explicit origins), so hop 2+
    never needs to touch the raw edge table again."""
    base = edges_agg.loc[edges_agg["source"] != edges_agg["target"], ["source", "target", "amount"]].copy()
    own_total = base["source"].map(totals).fillna(0.0).to_numpy()
    base["amount"] = np.minimum(base["amount"].to_numpy(), own_total)
    base = base[base["amount"] > 0]
    base = base.sort_values("amount", ascending=False)
    return base.groupby("source", sort=False, observed=True).head(top_n).reset_index(drop=True)


def _hop_stats(chain: pd.DataFrame, totals: Dict[str, float], hop: int) -> pd.DataFrame:
    grouped = chain.groupby("origin")["amount"]
    out = grouped.agg(total="sum", number_of_nodes="count", std_amounts="std", max_amounts="max")
    out["std_amounts"] = out["std_amounts"].fillna(0.0)
    own_total = out.index.to_series().map(totals).replace(0, np.nan)
    out["rel_transferred"] = (out["total"] / own_total).fillna(0.0)
    return out.add_prefix(f"flow_hop_{hop}_")


def _trace_flow(
    edges_agg: pd.DataFrame, totals: Dict[str, float],
    top_n: int = config.FLOW_TOP_N, num_hops: int = config.FLOW_NUM_HOPS, origin_accounts=None,
) -> pd.DataFrame:
    """Core Algorithm 1 implementation. `edges_agg` must already be
    oriented in the direction to trace forward through (pass as-is for
    dispense/passthrough; pass with source/target swapped for sink)."""
    base = _build_capped_base_table(edges_agg, totals, top_n)

    if origin_accounts is not None:
        chain = base[base["source"].isin(set(origin_accounts))].rename(
            columns={"source": "origin", "target": "frontier"})
    else:
        chain = base.rename(columns={"source": "origin", "target": "frontier"})

    stats = [_hop_stats(chain, totals, 1)]
    for hop in range(2, num_hops + 1):
        if chain.empty:
            break
        nxt = chain.merge(base, left_on="frontier", right_on="source", how="inner")
        nxt["amount"] = np.minimum(nxt["amount_x"].to_numpy(), nxt["amount_y"].to_numpy())
        nxt = nxt.loc[nxt["origin"] != nxt["target"], ["origin", "target", "amount"]].rename(
            columns={"target": "frontier"})
        nxt = nxt.sort_values("amount", ascending=False)
        chain = nxt.groupby("origin", sort=False, observed=True).head(top_n).reset_index(drop=True)
        stats.append(_hop_stats(chain, totals, hop))

    result = pd.concat(stats, axis=1)
    result.index.name = "key"
    return result


def compute_dispense_features(edges_agg: pd.DataFrame, origin_accounts=None,
                                top_n: int = config.FLOW_TOP_N, num_hops: int = config.FLOW_NUM_HOPS) -> pd.DataFrame:
    """Placement profile: forward from an account's own SENT total."""
    totals_sent, _ = _totals(edges_agg)
    return _trace_flow(edges_agg, totals_sent, top_n, num_hops, origin_accounts).add_prefix("dispense_")


def compute_sink_features(edges_agg: pd.DataFrame, origin_accounts=None,
                            top_n: int = config.FLOW_TOP_N, num_hops: int = config.FLOW_NUM_HOPS) -> pd.DataFrame:
    """Integration profile: backward from an account's own RECEIVED total
    (implemented as forward-tracing on the direction-reversed graph)."""
    _, totals_received = _totals(edges_agg)
    reversed_edges = edges_agg.rename(columns={"source": "target", "target": "source"})[["source", "target", "amount"]]
    return _trace_flow(reversed_edges, totals_received, top_n, num_hops, origin_accounts).add_prefix("sink_")


def compute_passthrough_features(edges_agg: pd.DataFrame, origin_accounts=None,
                                    top_n: int = config.FLOW_TOP_N, num_hops: int = config.FLOW_NUM_HOPS) -> pd.DataFrame:
    """Layering profile: forward, but capped by what the account RECEIVED
    rather than what it sent -- "of what I took in, how much do I pass
    along"."""
    _, totals_received = _totals(edges_agg)
    return _trace_flow(edges_agg, totals_received, top_n, num_hops, origin_accounts).add_prefix("passthrough_")


def _totals(edges_agg: pd.DataFrame) -> tuple:
    return (edges_agg.groupby("source")["amount"].sum().to_dict(),
            edges_agg.groupby("target")["amount"].sum().to_dict())


# ---------------------------------------------------------------------------
# Section 4.3: temporal flow features (chronological dispense->passthrough->sink)
# ---------------------------------------------------------------------------

def _restrict_top_counterparties(txns: pd.DataFrame, group_col: str, other_col: str, top_n: int) -> pd.DataFrame:
    """Keep only the `top_n` most frequent (group_col, other_col) edges (by
    transaction count) per value of group_col -- bounds fan-in/out breadth
    before the expensive chronological join."""
    edge_counts = (
        txns.groupby([group_col, other_col], observed=True).size().reset_index(name="n").sort_values("n", ascending=False)
    )
    return edge_counts.groupby(group_col, sort=False, observed=True).head(top_n)[[group_col, other_col]]


def _chronological_triples(left: pd.DataFrame, right: pd.DataFrame, max_matches: int) -> pd.DataFrame:
    """
    left  : [dispense, passthrough, amount_in, ts_in]   (dispense -> passthrough txns)
    right : [passthrough, sink, amount_out, ts_out]     (passthrough -> sink txns)

    For every passthrough account, joins incoming against outgoing txns
    with ts_out >= ts_in (vectorized per-account `np.searchsorted`, the
    same repeat+cumsum-offset idiom graph_construction.py's
    `_vectorized_topk_join` uses), keeping at most `max_matches`
    earliest-qualifying outgoing txns per incoming txn.
    """
    if left.empty or right.empty:
        return pd.DataFrame(columns=["dispense", "passthrough", "sink", "amount"])

    left = left.sort_values(["passthrough", "ts_in"]).reset_index(drop=True)
    right = right.sort_values(["passthrough", "ts_out"]).reset_index(drop=True)
    right_groups = dict(tuple(right.groupby("passthrough", sort=False, observed=True)))

    chunks = []
    for p, lg in left.groupby("passthrough", sort=False, observed=True):
        rg = right_groups.get(p)
        if rg is None:
            continue
        r_ts, l_ts = rg["ts_out"].to_numpy(), lg["ts_in"].to_numpy()
        lo = np.searchsorted(r_ts, l_ts, side="left")
        counts = np.clip(len(r_ts) - lo, 0, max_matches)
        total = int(counts.sum())
        if total == 0:
            continue
        group_start = np.cumsum(counts) - counts
        local_offset = np.arange(total) - np.repeat(group_start, counts)
        flat_positions = np.repeat(lo, counts) + local_offset
        left_rep = np.repeat(np.arange(len(lg)), counts)

        chunks.append(pd.DataFrame({
            "dispense": lg["dispense"].to_numpy()[left_rep],
            "passthrough": p,
            "sink": rg["sink"].to_numpy()[flat_positions],
            "amount": np.minimum(lg["amount_in"].to_numpy()[left_rep], rg["amount_out"].to_numpy()[flat_positions]),
        }))

    if not chunks:
        return pd.DataFrame(columns=["dispense", "passthrough", "sink", "amount"])
    return pd.concat(chunks, ignore_index=True)


def _aggregate_triples(triples: pd.DataFrame, key_col: str) -> pd.DataFrame:
    """Per-account amount stats + distinct dispense/passthrough/sink
    counts, plus the same restricted to `dispense == sink` (money that
    round-trips to its origin -- a classic layering/cycling tell),
    suffixed `_cycle`."""
    def _stats(frame: pd.DataFrame) -> pd.DataFrame:
        g = frame.groupby(key_col)["amount"]
        stats = g.agg(amount_sum="sum", amount_mean="mean", amount_median="median", amount_max="max", amount_std="std")
        stats["amount_std"] = stats["amount_std"].fillna(0.0)
        counts = frame.groupby(key_col).agg(
            dispense_count=("dispense", "nunique"), passthrough_count=("passthrough", "nunique"),
            sink_count=("sink", "nunique"),
        )
        return stats.join(counts)

    out = _stats(triples)
    cyclic = triples[triples["dispense"] == triples["sink"]]
    if not cyclic.empty:
        out = out.join(_stats(cyclic).add_suffix("_cycle"), how="left")
    out.index.name = "key"
    return out


def compute_temporal_flow_features(
    df: pd.DataFrame,
    source_col: str = "Sender Account", target_col: str = "Receiver Account",
    amount_col: str = "Amount Paid", timestamp_col: str = "Timestamp",
    top_n: int = config.TEMPORAL_FLOW_TOP_N,
) -> Dict[str, pd.DataFrame]:
    """Returns {"dispense": df, "passthrough": df, "sink": df}, each
    indexed by account id, columns prefixed `temporal_flow_{role}_`."""
    txns = df[[source_col, target_col, amount_col, timestamp_col]].rename(
        columns={source_col: "source", target_col: "target", amount_col: "amount", timestamp_col: "timestamp"}
    ).copy()
    txns["ts"] = txns["timestamp"].values.astype("datetime64[ns]").astype(np.int64)

    incoming_keep = _restrict_top_counterparties(txns, "target", "source", top_n)
    outgoing_keep = _restrict_top_counterparties(txns, "source", "target", top_n)

    left = txns.merge(incoming_keep, on=["source", "target"], how="inner").rename(
        columns={"source": "dispense", "target": "passthrough", "amount": "amount_in", "ts": "ts_in"}
    )[["dispense", "passthrough", "amount_in", "ts_in"]]
    right = txns.merge(outgoing_keep, on=["source", "target"], how="inner").rename(
        columns={"source": "passthrough", "target": "sink", "amount": "amount_out", "ts": "ts_out"}
    )[["passthrough", "sink", "amount_out", "ts_out"]]

    triples = _chronological_triples(left, right, max_matches=top_n)
    if triples.empty:
        empty = pd.DataFrame()
        return {"dispense": empty, "passthrough": empty, "sink": empty}

    return {
        "dispense": _aggregate_triples(triples, "dispense").add_prefix("temporal_flow_dispense_"),
        "passthrough": _aggregate_triples(triples, "passthrough").add_prefix("temporal_flow_passthrough_"),
        "sink": _aggregate_triples(triples, "sink").add_prefix("temporal_flow_sink_"),
    }

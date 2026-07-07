"""
feature_merge.py
-----------------
    behavior features  +  flow features  +  community features
                            |
                            v
                     final dataframe

This is the only module that touches ALL of graph_construction.py,
community_detection.py, and flow_features.py at once -- each of those
stays independent (neither community_detection.py nor flow_features.py
imports the other), and this module is where their outputs actually meet.

Also owns:
  * Section 4.5's anomaly score (small enough not to warrant its own file
    -- it's one IsolationForest fit on the already-merged node table).
  * the account-graph -> community/flow/anomaly -> account-indexed
    "node feature table" pipeline (`build_node_feature_table`), with a
    cache checkpoint at every expensive stage (see utils.py).
  * joining that node-indexed table onto each transaction, twice --
    Section 4.6: "we join the resulting table with our main transaction
    table; once with the source account and then with the target
    account."
  * the combined BASE + ExSTraQt feature-column registry that model.py /
    train.py actually train on.
"""

from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

import config
import utils
from graph_construction import build_account_graph
import community_detection as cd
import flow_features as ff


# ---------------------------------------------------------------------------
# Section 4.5: anomaly score (small enough to live here rather than its own file)
# ---------------------------------------------------------------------------

def _fit_anomaly_scores(node_features: pd.DataFrame) -> pd.Series:
    """IsolationForest over every numeric column of the node feature table
    (NaNs -> 0). Higher = more anomalous (sklearn's decision_function is
    negated so the sign matches every other risk feature's convention)."""
    numeric = node_features.select_dtypes(include=[np.number]).fillna(0.0)
    if numeric.empty:
        return pd.Series(0.0, index=node_features.index, name="anomaly_score")
    model = IsolationForest(
        n_estimators=config.ISOLATION_FOREST_N_ESTIMATORS,
        max_samples=config.ISOLATION_FOREST_MAX_SAMPLES,
        random_state=config.SEED, n_jobs=-1,
    )
    model.fit(numeric)
    return pd.Series(-model.decision_function(numeric), index=node_features.index, name="anomaly_score")


# ---------------------------------------------------------------------------
# Node feature table: account graph -> communities + flow + anomaly
# ---------------------------------------------------------------------------

def build_node_feature_table(
    df: pd.DataFrame,
    source_col: str = "Sender Account",
    target_col: str = "Receiver Account",
    amount_col: str = "Amount Paid",
    timestamp_col: str = "Timestamp",
    restrict_to_accounts: Optional[Iterable[str]] = None,
    cache_key: Optional[str] = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Returns a DataFrame indexed by account id with every ExSTraQt
    node-level feature group as columns.

    `cache_key`: a short string identifying THIS run's inputs (e.g.
    "train", "train_val") -- used as a filename prefix under cache/. Two
    calls with the same cache_key reuse each other's intermediate results;
    use a different key whenever the underlying transactions or
    restrict_to_accounts set changes (see utils.py's docstring -- cache
    invalidation here is explicit, not automatic).
    `restrict_to_accounts`: see config.RESTRICT_TO_ACCOUNTS's docstring --
    applied to the two expensive per-account stages (random-walk
    communities, flow tracing); Leiden and the aggregated-edge
    construction still see the whole graph regardless.
    """
    key = cache_key or "default"
    target_accounts = list(restrict_to_accounts) if restrict_to_accounts is not None else None

    def _load(path, compute_fn, kind="pickle"):
        if not use_cache:
            return compute_fn()
        return (utils.cache_parquet if kind == "parquet" else utils.cache_pickle)(path, compute_fn)

    # --- graph ---
    edges_agg = _load(
        config.GRAPH_CACHE_DIR / f"account_graph_edges_{key}.parquet",
        lambda: build_account_graph(df, source_col, target_col, amount_col)[1],
        kind="parquet",
    )
    graph = _load(
        config.GRAPH_CACHE_DIR / f"account_graph_{key}.pkl",
        lambda: build_account_graph(df, source_col, target_col, amount_col)[0],
    )

    # --- communities ---
    leiden_membership = _load(config.COMMUNITY_CACHE_DIR / f"leiden_{key}.pkl", lambda: cd.leiden_communities(graph))
    bottom_up_membership = _load(
        config.COMMUNITY_CACHE_DIR / f"random_walk_{key}.pkl",
        lambda: cd.random_walk_communities(edges_agg, graph, target_accounts=target_accounts),
    )

    # --- community statistics ---
    leiden_groups = cd.leiden_groups_from_membership(leiden_membership)
    leiden_stats = _load(
        config.FEATURE_CACHE_DIR / f"community_stats_leiden_{key}.parquet",
        lambda: cd.compute_community_statistics(df, leiden_groups, source_col, target_col, amount_col, timestamp_col, prefix="leiden"),
        kind="parquet",
    )
    local_stats = _load(
        config.FEATURE_CACHE_DIR / f"community_stats_local_{key}.parquet",
        lambda: cd.compute_community_statistics(df, bottom_up_membership, source_col, target_col, amount_col, timestamp_col, prefix="local"),
        kind="parquet",
    )
    leiden_node_table = cd.broadcast_leiden_features(leiden_membership, leiden_stats)

    # --- flow features ---
    dispense = _load(config.FEATURE_CACHE_DIR / f"flow_dispense_{key}.parquet",
                      lambda: ff.compute_dispense_features(edges_agg, origin_accounts=target_accounts), kind="parquet")
    sink = _load(config.FEATURE_CACHE_DIR / f"flow_sink_{key}.parquet",
                 lambda: ff.compute_sink_features(edges_agg, origin_accounts=target_accounts), kind="parquet")
    passthrough = _load(config.FEATURE_CACHE_DIR / f"flow_passthrough_{key}.parquet",
                         lambda: ff.compute_passthrough_features(edges_agg, origin_accounts=target_accounts), kind="parquet")
    temporal = _load(config.FEATURE_CACHE_DIR / f"flow_temporal_{key}.pkl",
                      lambda: ff.compute_temporal_flow_features(df, source_col, target_col, amount_col, timestamp_col))
    # temporal is a dict of 3 DataFrames -- cache_pickle (not cache_parquet) handles that natively

    tables = [leiden_node_table, local_stats, dispense, sink, passthrough]
    tables += [t for t in temporal.values() if not t.empty]
    node_table = pd.concat(tables, axis=1, join="outer")

    if target_accounts is not None:
        node_table = node_table.reindex(target_accounts)

    anomaly = _fit_anomaly_scores(node_table)
    node_table = node_table.join(anomaly)
    node_table.index.name = "account"
    return node_table


# ---------------------------------------------------------------------------
# Section 4.6: join node features onto each transaction
# ---------------------------------------------------------------------------

def join_node_features_to_transactions(
    df: pd.DataFrame,
    node_features: pd.DataFrame,
    source_col: str = "Sender Account",
    target_col: str = "Receiver Account",
    source_prefix: str = "src_exq_",
    target_prefix: str = "dst_exq_",
) -> pd.DataFrame:
    """Joins `node_features` onto `df` twice -- once keyed on `source_col`
    (prefixed `source_prefix`), once on `target_col` (prefixed
    `target_prefix`). Missing accounts get 0 rather than NaN."""
    src_feats, dst_feats = node_features.add_prefix(source_prefix), node_features.add_prefix(target_prefix)
    out = df.merge(src_feats, how="left", left_on=source_col, right_index=True)
    out = out.merge(dst_feats, how="left", left_on=target_col, right_index=True)

    new_cols = list(src_feats.columns) + list(dst_feats.columns)
    out[new_cols] = out[new_cols].fillna(0.0)

    if "anomaly_score" in node_features.columns:
        src_a, dst_a = out[f"{source_prefix}anomaly_score"].to_numpy(), out[f"{target_prefix}anomaly_score"].to_numpy()
        interactions = pd.DataFrame({
            "exq_anomaly_score_diff": src_a - dst_a,
            "exq_anomaly_score_min": np.minimum(src_a, dst_a),
            "exq_anomaly_score_max": np.maximum(src_a, dst_a),
            "exq_anomaly_score_mean": (src_a + dst_a) / 2.0,
        }, index=out.index)
        out = pd.concat([out, interactions], axis=1)

    return out.copy()  # de-fragment after the merges/concat above


# ---------------------------------------------------------------------------
# Combined BASE + ExSTraQt feature-column registry
# ---------------------------------------------------------------------------

def all_feature_columns(
    node_feature_columns: List[str], base_numeric: List[str], base_categorical: List[str],
    source_prefix: str = "src_exq_", target_prefix: str = "dst_exq_",
) -> Tuple[List[str], List[str]]:
    """Returns (numeric_columns, categorical_columns). All ExSTraQt-derived
    columns are numeric -- only the BASE categorical columns (payment
    format, currency, bank, ...) stay categorical."""
    exq_numeric = [f"{source_prefix}{c}" for c in node_feature_columns]
    exq_numeric += [f"{target_prefix}{c}" for c in node_feature_columns]
    if "anomaly_score" in node_feature_columns:
        exq_numeric += ["exq_anomaly_score_diff", "exq_anomaly_score_min",
                         "exq_anomaly_score_max", "exq_anomaly_score_mean"]
    return list(base_numeric) + exq_numeric, list(base_categorical)


def prepare_feature_frame(df: pd.DataFrame, numeric_cols: List[str], categorical_cols: List[str]) -> pd.DataFrame:
    """Slice + type the model-input columns: categoricals -> pandas
    'category' dtype (XGBoost enable_categorical=True reads these
    natively); numeric columns coerced to float with NaN -> 0."""
    X = df.loc[:, numeric_cols + categorical_cols].copy()
    for col in categorical_cols:
        X[col] = X[col].astype("category")
    for col in numeric_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0.0)
    return X

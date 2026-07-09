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

import gc
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from pathlib import Path
from . import config
from . import utils
from .graph_construction import build_account_graph
from . import community_detection as cd
from . import flow_features as ff


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
    graph_cache_dir:Path,
    community_cache_dir:Path,
    feature_cache_dir:Path,
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
        graph_cache_dir / f"account_graph_edges_{key}.parquet",
        lambda: build_account_graph(df, source_col, target_col, amount_col)[1],
        kind="parquet",
    )
    graph = _load(
        graph_cache_dir / f"account_graph_{key}.pkl",
        lambda: build_account_graph(df, source_col, target_col, amount_col)[0],
    )

    # --- communities ---
    leiden_membership = _load(community_cache_dir / f"leiden_{key}.pkl", lambda: cd.leiden_communities(graph))
    # `graph` (an igraph object over all ~417k accounts) is only needed by
    # Leiden. If it was just computed (cache miss) it's live now; free it
    # before the flow stage rather than holding it to function end.
    del graph
    gc.collect()

    # --- community statistics ---
    leiden_groups = cd.leiden_groups_from_membership(leiden_membership)
    leiden_stats = _load(
        feature_cache_dir / f"community_stats_leiden_{key}.parquet",
        lambda: cd.compute_community_statistics(df, leiden_groups, source_col, target_col, amount_col, timestamp_col, prefix="leiden"),
        kind="parquet",
    )
    leiden_node_table = cd.broadcast_leiden_features(leiden_membership, leiden_stats)
    del leiden_groups, leiden_stats, leiden_membership

    # --- flow features ---
    dispense = _load(feature_cache_dir / f"flow_dispense_{key}.parquet",
                      lambda: ff.compute_dispense_features(edges_agg, origin_accounts=target_accounts), kind="parquet")
    sink = _load(feature_cache_dir / f"flow_sink_{key}.parquet",
                 lambda: ff.compute_sink_features(edges_agg, origin_accounts=target_accounts), kind="parquet")
    passthrough = _load(feature_cache_dir / f"flow_passthrough_{key}.parquet",
                         lambda: ff.compute_passthrough_features(edges_agg, origin_accounts=target_accounts), kind="parquet")
    temporal = _load(feature_cache_dir / f"flow_temporal_{key}.pkl",
                      lambda: ff.compute_temporal_flow_features(df, source_col, target_col, amount_col, timestamp_col))
    # temporal is a dict of 3 DataFrames -- cache_pickle (not cache_parquet) handles that natively
    del edges_agg  # dead after the three flow features above
    gc.collect()

    tables = [leiden_node_table, dispense, sink, passthrough]
    tables += [t for t in temporal.values() if not t.empty]
    node_table = pd.concat(tables, axis=1, join="outer")
    del tables, leiden_node_table, dispense, sink, passthrough, temporal

    # float32 at the account level: 417k rows x ~134 cols halves from
    # ~450MB to ~225MB here, and -- more importantly -- everything built
    # FROM this table downstream (the transaction-level join, X_train,
    # the anomaly fit input) inherits the smaller dtype.
    node_table = node_table.astype(np.float32)

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
    `target_prefix`). Missing accounts get 0 rather than NaN.

    MEMORY NOTE: this fans ~270 account-level columns out onto millions of
    transaction rows -- at HI-Small scale (~4.3M train rows) it is the
    single largest allocation in the whole pipeline, and the previous
    merge-based implementation OOM'd Kaggle at exactly this point. Three
    changes keep the peak roughly half of what it was (verified ~1.9x on
    a 300k-row benchmark, identical output to float32 precision):

      1. float32, applied at the ACCOUNT level (417k rows -- cheap) so
         every transaction-level block is BORN float32 instead of being
         built float64 and (never) downcast. Halves every allocation.
      2. `.take()` on the node-feature array instead of two `df.merge`
         calls followed by `out[new_cols].fillna(0.0)` -- the fillna on a
         ~270-column block was a full transient copy of the block, and
         each merge builds its own intermediate frame. The take-based
         path fills missing accounts and NaN feature values in place.
      3. ONE final `pd.concat` producing an already-contiguous frame --
         which also removes the need for the old `return out.copy()`
         de-fragmentation step (a full copy of everything, base columns
         included) that pandas' fragmentation warning had pushed us into.
    """
    # float32 once, at account level; also hoist the ndarray out of the
    # per-block closure so it's materialized once, not once per side.
    nf = node_features.astype(np.float32)
    nf_array = nf.to_numpy()
    position = pd.Series(np.arange(len(nf), dtype=np.int64), index=nf.index)

    def _block(accounts: pd.Series, prefix: str) -> pd.DataFrame:
        idx = accounts.map(position)              # NaN where account unseen
        missing = idx.isna().to_numpy()
        idx_filled = idx.fillna(0).to_numpy(dtype=np.int64)
        block = nf_array[idx_filled]              # float32 row-take
        block[missing] = 0.0                      # unseen accounts -> 0, in place
        np.nan_to_num(block, copy=False)          # NaN feature values -> 0, in place
        return pd.DataFrame(block, index=df.index,
                            columns=[f"{prefix}{c}" for c in nf.columns])

    src_block = _block(df[source_col], source_prefix)
    dst_block = _block(df[target_col], target_prefix)

    parts = [df, src_block, dst_block]
    if "anomaly_score" in nf.columns:
        src_a = src_block[f"{source_prefix}anomaly_score"].to_numpy()
        dst_a = dst_block[f"{target_prefix}anomaly_score"].to_numpy()
        parts.append(pd.DataFrame({
            "exq_anomaly_score_diff": src_a - dst_a,
            "exq_anomaly_score_min": np.minimum(src_a, dst_a),
            "exq_anomaly_score_max": np.maximum(src_a, dst_a),
            "exq_anomaly_score_mean": (src_a + dst_a) / 2.0,
        }, index=df.index))

    return pd.concat(parts, axis=1)


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
    natively); numeric columns coerced to float32 with NaN -> 0.

    MEMORY NOTE: float32, not float64 -- XGBoost converts to float32
    internally anyway, so float64 here only doubles this frame's size
    (~5GB extra at HI-Small scale) for zero precision benefit. Columns
    that are ALREADY float32 (every src_exq_/dst_exq_/exq_ column, after
    the join rewrite above) are left untouched instead of being pushed
    through a redundant to_numeric copy -- at ~270 such columns, that
    skipped pass is most of this function's former cost."""
    X = df.loc[:, numeric_cols + categorical_cols].copy()
    for col in categorical_cols:
        X[col] = X[col].astype("category")
    for col in numeric_cols:
        if X[col].dtype == np.float32:
            continue
        X[col] = pd.to_numeric(X[col], errors="coerce").astype(np.float32).fillna(0.0)
    return X
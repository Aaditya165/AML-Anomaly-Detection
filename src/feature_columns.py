"""
feature_columns.py
-------------------
Replaces pyg_export.py. There's no graph tensor to build anymore -- the
model consumes a plain feature matrix -- but every downstream module
(model.py, train.py, dashboard.py) needs one shared, explicit definition
of which engineered columns go into the model and how, so it lives here
instead of being redefined in each place.

NUMERIC_FEATURE_COLUMNS
    Unchanged from the GNN version. This is still the bulk of the signal:
    temporal, pair-history, graph-topology (predecessor/successor counts,
    fan-in/out, relay timing, short-cycle) and account-behavioral
    (entropy, concentration, velocity) features from feature_engineering.py.
    Trees consume these as plain floats -- no scaling required (unlike the
    GNN's StandardScaler, which existed only because gradient descent needs
    normalized inputs; tree splits are invariant to monotonic rescaling).

CATEGORICAL_FEATURE_COLUMNS
    Low/medium-cardinality categorical codes that XGBoost/LightGBM can
    split on natively. Deliberately EXCLUDES `sender_idx` / `receiver_idx`:
    those are raw account identities with tens/hundreds of thousands of
    unique values. The GNN used them to index a learned embedding table,
    which generalizes because embeddings are optimized jointly with the
    task. A raw high-cardinality integer code fed to a tree instead just
    invites the tree to memorize specific accounts (overfits, doesn't
    generalize to accounts unseen in training, and is a leakage risk).
    Every generalizable signal that account identity carries is already
    captured by the `sender_*` / `receiver_*` behavioral aggregates in
    NUMERIC_FEATURE_COLUMNS (historical flow, entropy, concentration,
    velocity, unique counterparties) -- those describe *behavior*, not
    *identity*, which is exactly what should transfer to a new account.
"""

NUMERIC_FEATURE_COLUMNS = [
    "log_amount_paid", "log_amount_received", "same_bank_flag",
    "hour_of_day", "day_of_week",
    "time_since_prev_outgoing_sender", "time_since_prev_incoming_receiver",
    "time_since_last_pair_transfer", "recency_pair_score",
    "pair_prior_txn_count", "pair_total_amount_prior", "pair_mean_amount_prior",
    "pair_std_amount_prior", "pair_max_amount_prior", "pair_repeated_exact_amount_count",
    "predecessor_count", "successor_count", "fan_in", "fan_out",
    "relay_timing_score", "continues_existing_chain", "sender_recently_receiver",
    "short_cycle_participation",
    "sender_hist_outflow_total", "sender_hist_inflow_total",
    "sender_unique_counterparties", "sender_entropy", "sender_concentration",
    "sender_degree_balance", "sender_velocity_1d", "sender_velocity_10d",
    "receiver_hist_outflow_total", "receiver_hist_inflow_total",
    "receiver_unique_counterparties", "receiver_entropy", "receiver_concentration",
    "receiver_degree_balance", "receiver_velocity_1d", "receiver_velocity_10d",
]

CATEGORICAL_FEATURE_COLUMNS = [
    "from_bank_idx",
    "to_bank_idx",
    "payment_format_idx",
    "payment_currency_idx",
    "receiving_currency_idx",
]

ALL_FEATURE_COLUMNS = NUMERIC_FEATURE_COLUMNS + CATEGORICAL_FEATURE_COLUMNS


def prepare_feature_frame(df):
    """
    Slice + type the model-input columns out of an engineered DataFrame
    (the output of feature_engineering.engineer_all_features).

    Categorical columns are cast to pandas 'category' dtype, which
    XGBoost (enable_categorical=True) and LightGBM (auto-detected) both
    read natively -- no one-hot explosion, no separate encoder to fit/
    save/version alongside the model.
    """
    X = df[ALL_FEATURE_COLUMNS].copy()
    for col in CATEGORICAL_FEATURE_COLUMNS:
        X[col] = X[col].astype("category")
    return X

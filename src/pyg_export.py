"""
pyg_export.py
--------------
PRD Section 4: "Graph Export -> PyTorch Geometric"

Converts the engineered transaction DataFrame + NetworKit-derived edge
list into a `torch_geometric.data.Data` object:

    data.x            : [N, F] standardized numeric transaction features
    data.edge_index   : [2, E] money-flow edges (txn -> txn)
    data.y            : [N]    Is-Laundering label
    data.<cat>_idx    : [N]    long-tensor categorical codes, one per
                                embedding table from Section 6 (account,
                                bank, payment format, currency)

A single fitted StandardScaler is reused for train/val/test so feature
scaling is consistent across splits (fit on train only).
"""

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
import time

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

CATEGORICAL_COLUMNS = {
    "sender_idx": "sender_idx",
    "receiver_idx": "receiver_idx",
    "from_bank_idx": "from_bank_idx",
    "to_bank_idx": "to_bank_idx",
    "payment_format_idx": "payment_format_idx",
    "payment_currency_idx": "payment_currency_idx",
    "receiving_currency_idx": "receiving_currency_idx",
}


def build_pyg_data(df, edge_src, edge_dst, scaler: StandardScaler = None, fit_scaler: bool = False):
    """
    Returns (data, scaler). Pass the *same* scaler object back in
    (fit_scaler=False) when exporting validation/test/inference graphs so
    feature scaling matches what the model was trained on.
    """
    t0 = time.time()
    numeric = df[NUMERIC_FEATURE_COLUMNS].values.astype(np.float32)

    if fit_scaler or scaler is None:
        scaler = StandardScaler()
        numeric = scaler.fit_transform(numeric).astype(np.float32)
    else:
        numeric = scaler.transform(numeric).astype(np.float32)

    x = torch.from_numpy(numeric)
    edge_index = torch.tensor(np.vstack([edge_src, edge_dst]), dtype=torch.long)
    y = torch.tensor(df["label"].values, dtype=torch.float32)

    data = Data(x=x, edge_index=edge_index, y=y, num_nodes=len(df))
    for attr_name, col in CATEGORICAL_COLUMNS.items():
        setattr(data, attr_name, torch.tensor(df[col].values, dtype=torch.long))

    print("04. Graph Export to PyTorch Geometric: ", time.time()-t0)

    return data, scaler


if __name__ == "__main__":
    from data_processing import load_and_clean
    from graph_construction import build_transaction_graph
    from feature_engineering import engineer_all_features

    df, vocabs = load_and_clean("HI-Small_Trans_SYNTHETIC.csv")
    g, src, dst, preds, succs = build_transaction_graph(df)
    df = engineer_all_features(df, preds, succs, src, dst)
    data, scaler = build_pyg_data(df, src, dst, fit_scaler=True)
    print(data)
    print("x dtype/shape:", data.x.dtype, data.x.shape)
    print("sender_idx dtype/shape:", data.sender_idx.dtype, data.sender_idx.shape)

    # sanity-check that NeighborLoader correctly slices our custom
    # categorical node attributes along with x/y/edge_index.
    from torch_geometric.loader import NeighborLoader
    loader = NeighborLoader(
        data, num_neighbors=[10, 10], batch_size=512,
        input_nodes=torch.arange(data.num_nodes),
        shuffle=True,
    )
    batch = next(iter(loader))
    print(batch)
    assert batch.sender_idx.shape[0] == batch.x.shape[0]
    print("NeighborLoader correctly carries custom categorical attrs. OK.")

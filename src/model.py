"""
model.py
--------
PRD Section 6 (Embedding Strategy) + Section 10 (Graph Learning Architecture).

`LineMVGNN` is not a published reference architecture with a fixed spec in
the PRD beyond "learn transaction relationships and money-flow structures"
plus the embedding strategy in Section 6, so it's implemented here as a
**Multi-View** GNN operating on the transaction *line graph* described in
Section 5, where each "view" corresponds to a distinct way of looking at a
transaction node:

    View A - Structural view  : GATConv stack over the money-flow graph
                                 (learns which predecessor transactions
                                 matter most -- attention over flow).
    View B - Account-context  : GCNConv stack over the same graph, applied
                                 to a representation dominated by the
                                 account/bank/currency embeddings (Section 6)
                                 -- smooths account-behavioral signal across
                                 connected transactions (pass-through,
                                 layering chains).
    View C - Attribute view   : graph-free MLP over the instantaneous
                                 transaction attributes (amount, time,
                                 payment format, ...) -- captures signal
                                 that doesn't depend on graph context.

The three views are fused (concat + linear) into a single "Transaction
Embedding" which then feeds the Section 10 classifier head:

    Transaction Embedding -> Linear -> ReLU -> Dropout -> Linear -> Sigmoid

(The final Sigmoid is applied outside the module via
`torch.sigmoid(logits)` / `BCEWithLogitsLoss`, the numerically-stable
standard equivalent -- the produced probability is identical.)

If you have a specific published LineMVGNN paper/architecture in mind,
swap this module out; everything downstream (training loop, NeighborLoader
sampling, evaluation) is architecture-agnostic and only depends on the
`forward(...) -> logits[N]` contract below.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv


class LineMVGNN(nn.Module):
    def __init__(
        self,
        numeric_dim: int,
        n_accounts: int,
        n_banks: int,
        n_payment_formats: int,
        n_currencies: int,
        emb_dim: int = 16,
        hidden_dim: int = 64,
        num_layers: int = 2,
        gat_heads: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        half = max(emb_dim // 2, 4)

        # ---- Section 6: learnable categorical embeddings ----
        self.account_emb = nn.Embedding(n_accounts, emb_dim)        # sender & receiver share this table
        self.bank_emb = nn.Embedding(n_banks, half)                 # from & to bank share this table
        self.payfmt_emb = nn.Embedding(n_payment_formats, half)
        self.currency_emb = nn.Embedding(n_currencies, half)        # payment & receiving currency share this table

        embed_total_dim = emb_dim * 2 + half * 2 + half + half * 2
        self.input_proj = nn.Linear(numeric_dim + embed_total_dim, hidden_dim)

        # ---- View A: structural / topology (attention over money flow) ----
        self.gat_layers = nn.ModuleList([
            GATConv(hidden_dim, hidden_dim // gat_heads, heads=gat_heads,
                    concat=True, dropout=dropout)
            for _ in range(num_layers)
        ])

        # ---- View B: account-context (smoothing over the flow graph) ----
        self.gcn_layers = nn.ModuleList([
            GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])

        # ---- View C: transaction-attribute view (graph-free) ----
        self.attr_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.fusion = nn.Linear(hidden_dim * 3, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # ---- Section 10: MLP classifier head ----
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def embed_categoricals(self, batch) -> torch.Tensor:
        return torch.cat([
            self.account_emb(batch.sender_idx),
            self.account_emb(batch.receiver_idx),
            self.bank_emb(batch.from_bank_idx),
            self.bank_emb(batch.to_bank_idx),
            self.payfmt_emb(batch.payment_format_idx),
            self.currency_emb(batch.payment_currency_idx),
            self.currency_emb(batch.receiving_currency_idx),
        ], dim=-1)

    def forward(self, batch) -> torch.Tensor:
        """`batch` is a (sub)graph with .x, .edge_index, .<cat>_idx attrs.
        Returns raw logits of shape [N] (apply torch.sigmoid for probability)."""
        cat_emb = self.embed_categoricals(batch)
        h0 = F.relu(self.input_proj(torch.cat([batch.x, cat_emb], dim=-1)))

        h_struct = h0
        for layer in self.gat_layers:
            h_struct = F.relu(layer(h_struct, batch.edge_index))
            h_struct = self.dropout(h_struct)

        h_ctx = h0
        for layer in self.gcn_layers:
            h_ctx = F.relu(layer(h_ctx, batch.edge_index))
            h_ctx = self.dropout(h_ctx)

        h_attr = self.attr_mlp(h0)

        fused = F.relu(self.fusion(torch.cat([h_struct, h_ctx, h_attr], dim=-1)))
        fused = self.dropout(fused)

        logits = self.classifier(fused).squeeze(-1)
        return logits


if __name__ == "__main__":
    from data_processing import load_and_clean, vocab_sizes
    from graph_construction import build_transaction_graph
    from feature_engineering import engineer_all_features
    from pyg_export import build_pyg_data, NUMERIC_FEATURE_COLUMNS
    from neighbor_sampling import SimpleNeighborLoader

    df, vocabs = load_and_clean("HI-Small_Trans_SYNTHETIC.csv")
    g, src, dst, preds, succs = build_transaction_graph(df)
    df = engineer_all_features(df, preds, succs, src, dst)
    data, scaler = build_pyg_data(df, src, dst, fit_scaler=True)
    vs = vocab_sizes(vocabs)

    model = LineMVGNN(
        numeric_dim=len(NUMERIC_FEATURE_COLUMNS),
        n_accounts=vs["n_accounts"], n_banks=vs["n_banks"],
        n_payment_formats=vs["n_payment_formats"], n_currencies=vs["n_currencies"],
    )
    print(model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total trainable parameters: {n_params:,}")

    loader = SimpleNeighborLoader(
        data, preds,
        num_neighbors=[15,10],
        input_nodes=torch.arange(data.num_nodes),
        batch_size=256,
    )
    batch = next(iter(loader))
    logits = model(batch)
    print("logits shape:", logits.shape, "seed_size:", batch.seed_size)
    probs = torch.sigmoid(logits[:batch.seed_size])
    print("sample probabilities:", probs[:5])

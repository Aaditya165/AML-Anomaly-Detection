"""
neighbor_sampling.py
---------------------
PRD Section 11: "Sampling - PyTorch Geometric NeighborLoader... Enable
training on large graphs, avoid loading full graph into memory."

`torch_geometric.loader.NeighborLoader` needs the compiled `pyg-lib` or
`torch-sparse` extensions. On Colab, `pyg-lib` currently has no published
wheel for some torch/CUDA/Python combinations ("No matching distribution
found"), and `torch-sparse`'s source build can fail to compile against a
newer torch than it was written against ("error: subprocess-exited-with-
error" / "Failed building wheel"). Both are version-matching problems,
not something a retried `pip install` fixes.

`SimpleNeighborLoader` below reimplements the same *idea* in plain
Python/PyTorch using the predecessor adjacency lists we already built in
`graph_construction.py`: for a batch of seed (target) transaction nodes,
it samples up to `num_neighbors[hop]` predecessors per hop, for as many
hops as the model has GNN layers, and returns a small induced subgraph
(local node ids, local edge_index, sliced features/labels). Seed nodes
are always placed first, exactly like PyG's NeighborLoader convention,
so `batch.y[:batch.seed_size]` / `out[:batch.seed_size]` are what you
compute loss/metrics on.

If your environment CAN install `pyg-lib`/`torch-sparse` (e.g. by pinning
torch to a version PyG has a published wheel for, then installing from
https://data.pyg.org/whl/), you can swap this for the real
`torch_geometric.loader.NeighborLoader` against the exact same `Data`
object with no other code changes -- that's why this module mirrors its
batch/seed conventions.
"""

import numpy as np
import torch
from torch_geometric.data import Data


class SimpleNeighborLoader:
    def __init__(
        self,
        data: Data,
        predecessors: list,
        input_nodes: torch.Tensor,
        num_neighbors: list = (15, 15),
        batch_size: int = 512,
        shuffle: bool = True,
        seed: int = 0,
    ):
        self.data = data
        self.predecessors = predecessors
        self.input_nodes = input_nodes.numpy() if torch.is_tensor(input_nodes) else np.asarray(input_nodes)
        self.num_neighbors = list(num_neighbors)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)

        self._node_attr_names = [
            k for k in data.keys()
            if torch.is_tensor(data[k]) and data[k].dim() >= 1
            and data[k].shape[0] == data.num_nodes
        ]

    def __len__(self):
        return int(np.ceil(len(self.input_nodes) / self.batch_size))

    def __iter__(self):
        order = self.input_nodes.copy()
        if self.shuffle:
            self.rng.shuffle(order)
        for start in range(0, len(order), self.batch_size):
            seeds = order[start:start + self.batch_size]
            yield self._sample(seeds)

    def _sample(self, seeds: np.ndarray) -> Data:
        nodes = list(seeds)
        local_id = {int(g): i for i, g in enumerate(nodes)}
        edges_local_src, edges_local_dst = [], []

        frontier = list(seeds)
        for hop_fanout in self.num_neighbors:
            new_frontier = []
            for node in frontier:
                preds = self.predecessors[node]
                if len(preds) == 0:
                    continue
                if len(preds) > hop_fanout:
                    preds = self.rng.choice(preds, size=hop_fanout, replace=False)
                for p in preds:
                    p = int(p)
                    if p not in local_id:
                        local_id[p] = len(nodes)
                        nodes.append(p)
                        new_frontier.append(p)
                    edges_local_src.append(local_id[p])
                    edges_local_dst.append(local_id[int(node)])
            frontier = new_frontier
            if not frontier:
                break

        node_idx = torch.tensor(nodes, dtype=torch.long)
        sub = Data()
        for attr in self._node_attr_names:
            sub[attr] = self.data[attr][node_idx]
        if edges_local_src:
            sub.edge_index = torch.tensor(
                [edges_local_src, edges_local_dst], dtype=torch.long
            )
        else:
            sub.edge_index = torch.empty((2, 0), dtype=torch.long)
        sub.num_nodes = len(nodes)
        sub.seed_size = len(seeds)
        return sub


if __name__ == "__main__":
    import time
    from data_processing import load_and_clean
    from graph_construction import build_transaction_graph
    from feature_engineering import engineer_all_features
    from pyg_export import build_pyg_data

    df, vocabs = load_and_clean("HI-Small_Trans_SYNTHETIC.csv")
    g, src, dst, preds, succs = build_transaction_graph(df)
    df = engineer_all_features(df, preds, succs, src, dst)
    data, scaler = build_pyg_data(df, src, dst, fit_scaler=True)

    loader = SimpleNeighborLoader(
        data, preds, input_nodes=torch.arange(data.num_nodes),
        num_neighbors=[15, 10], batch_size=512,
    )
    t0 = time.time()
    batch = next(iter(loader))
    print("one batch sampled in", time.time() - t0, "s")
    print(batch)
    print("seed_size:", batch.seed_size, "total nodes pulled in:", batch.num_nodes)
    assert batch.x.shape[0] == batch.num_nodes
    assert batch.sender_idx.shape[0] == batch.num_nodes
    print("OK - custom neighbor sampler produces a consistent mini-batch subgraph.")

# Graph-Based AML Detection with LineMVGNN

Implementation of the PRD: transaction-level graph construction (NetworKit)
→ feature engineering → PyTorch Geometric export → LineMVGNN training →
evaluation → model persistence → inference → account-level risk
aggregation → Streamlit dashboard — with **every heavy-compute section
wrapped in a timer**, so you can see exactly where a run's time goes.

## Quick start

```bash
pip install torch torch_geometric networkit pandas numpy scikit-learn plotly streamlit
jupyter notebook aml_gnn_pipeline.ipynb     # Run All
streamlit run dashboard.py                  # after the notebook has run once
```

The notebook already ships **with its outputs populated** (run on a
synthetic stand-in dataset, see below), so you can read through results
without re-running anything — or hit Run All to regenerate them yourself.

## What's timed, and what we actually saw

| Stage | Typical share of total time (40k-txn demo run) |
|---|---|
| **Model Training (total)** | ~48% |
| &nbsp;&nbsp;↳ Forward/backward pass | ~42% |
| &nbsp;&nbsp;↳ Neighbor sampling | ~5% |
| &nbsp;&nbsp;↳ Validation pass | ~1.5% |
| Account-Context Feature Engineering | ~1-2.5% |
| Transaction Flow Feature Engineering | ~0.5-1% |
| Graph Construction (NetworKit) | ~0.5-1% |
| Model Inference | ~0.3% |
| Data Loading & Normalization | ~0.2% |
| Pair-History / Temporal Features | <0.1% each |
| Graph Export to PyTorch Geometric | <0.1% |
| Account Risk Aggregation | <0.1% |

**Training's forward/backward pass dominates everything else**, by a wide
margin, once the model has more than a couple of epochs to run — at this
demo scale, *not* graph construction or feature engineering. The full
breakdown (numbers + a sorted bar chart) prints at the end of the
notebook (Section 11) and is re-orderable per-run from the timing log it
exports. The biggest, least obvious win was vectorizing the pair-history
statistics (running mean/std/max per sender→receiver pair): a naive
per-group `.expanding().std()` pandas implementation took **48 seconds**
on 40k rows; replacing it with cumulative-sum/cummax algebra (see the
docstring in `feature_engineering.py`) brought that down to **0.07
seconds** — a ~650x improvement, and the kind of thing this timer
instrumentation is meant to surface before it becomes a problem at "millions
of transactions" scale.

## Two things this implementation had to decide that the PRD doesn't fully pin down

1. **`LineMVGNN` architecture.** The PRD names it as "the primary GNN
   architecture" and specifies its *purpose* and the embedding strategy
   (Section 6), but not its internals. `model.py` implements it as a
   **Multi-View GNN**: a structural/topology view (GAT attention over the
   money-flow graph), an account-context view (GCN), and a graph-free
   transaction-attribute view (MLP) — fused into the Section 10 classifier
   head. If you have a specific paper or architecture in mind, swap out
   `model.py`; the training loop, sampler, and evaluation code are
   architecture-agnostic (they only depend on `model(batch) -> logits[N]`).

2. **`NeighborLoader` substitution.** `torch_geometric.loader.NeighborLoader`
   needs the compiled `pyg-lib` or `torch-sparse` extensions, which are
   distributed as wheels from a separate package index (`data.pyg.org`)
   that wasn't reachable while building this. `neighbor_sampling.py`
   re-implements the same mini-batch-sampling idea in plain
   Python/PyTorch using the predecessor adjacency lists from graph
   construction. If your own environment can install `pyg-lib`/
   `torch-sparse`, swap in the real `NeighborLoader` against the exact
   same `Data` object — no other code changes needed.

## No real data was available here

`HI-Small_Trans.csv` / `LI-Small_Trans.csv` (the IBM AML dataset) weren't
uploaded, so `synthetic_data.py` generates a small stand-in dataset with
the **same column schema** (plus a few injected layering chains / cycles
so the model has something real to learn) so the full pipeline could be
built, run, and timed end-to-end. **To use the real data**, just change
the two file paths in Section 1 of the notebook — nothing else needs to
change. Expect very different absolute timings at full scale (millions of
rows) — re-run Section 11 on your own data to get numbers that matter for
your environment.

## File guide

```
aml_gnn_pipeline.ipynb     Main notebook -- run this first
dashboard.py                streamlit run dashboard.py (PRD Section 16)

timing_utils.py             The Timer()/profiling harness everything else uses
synthetic_data.py           Synthetic stand-in dataset generator (same schema as IBM-AML)
data_processing.py          Section 3/4: cleaning, normalization, shared train/test vocabularies
graph_construction.py       Section 5: NetworKit transaction graph (10-day window, top-15 successors)
feature_engineering.py      Sections 7-9: temporal / pair-history / flow / account-context features
pyg_export.py               Section 4: DataFrame + edges -> torch_geometric.data.Data
neighbor_sampling.py        Section 11: mini-batch neighbor sampler (NeighborLoader substitute)
model.py                    Sections 6 & 10: LineMVGNN (multi-view GNN) + MLP classifier head
train.py                    Section 11-12: training loop, evaluation metrics
aggregation.py               Sections 14-16: transaction/account risk views, alerting

test_pipeline.py            Standalone smoke-test script (not required to run the notebook)
requirements.txt
```

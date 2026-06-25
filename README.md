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

## Three real bugs found and fixed at IBM-AML scale (4.5M-6M rows)

The first version of this worked fine on a small synthetic dataset but
**crashed Colab with an OOM kill** on the real `HI-Small_Trans.csv` /
`LI-Small_Trans.csv` files. Each of the three fixes below was confirmed
with a direct before/after measurement at the real dataset's scale
(4.48M train rows / 422k accounts, matching the user's actual numbers),
not guessed at:

### 1. Graph adjacency: N small objects → one flat CSR structure
`graph_construction.py` was storing each transaction's predecessors/
successors as its own little Python list, later converted to its own
little numpy array — at 4.5M+6.1M nodes that's ~21 million individual
objects, each paying ~100+ bytes of pure object-header overhead before a
single byte of real data. **Not NetworKit's fault** — `nk.Graph` itself
is fine; this was the Python glue code around it. Fixed with a CSR
(compressed-sparse-row) layout: one flat `indices` array for every edge
in the whole graph + one `indptr` array of length N+1, with a thin
`CSRAdjacency` wrapper so `adj[node]` still works everywhere it's used.
Edge construction itself was also rewritten to be vectorized per
receiver-account group instead of a 4.5M-iteration Python loop.

  **Measured at train-set scale (4.49M rows):** old approach peaked at
  3.83GB and took 53s; new approach peaked at 2.87GB (-25%) and took
  16s (-70%). Verified byte-for-byte identical output (same edge set,
  same adjacency) against the old implementation first.

### 2. DataFrame dtypes: object/float64 columns repeating a handful of values millions of times
Independent of the graph, just *loading and cleaning* both CSVs was
hitting ~3.7GB peak in a small (3.9GB) test environment — `Payment
Format`/`Payment Currency`/`Receiving Currency` each have **6-16 unique
values** but were costing ~65-70MB per column at 4.5M rows storing that
handful of strings as a full Python object on every row. Fixed by
parsing straight into `category` dtype in `read_csv` (not converting
after the fact, which would briefly hold both representations at once),
`float32` for amounts, and downcasting embedding-index columns to the
smallest safe int type. `Sender Account`/`Receiver Account` (and the two
currency columns) needed a **shared** `CategoricalDtype` across both
columns, not independently-built ones — pandas categoricals can only be
compared with `==`/`!=` if their `.categories` match, which the
sender≠receiver cleaning filter relies on.

  **Measured:** cleaned train DataFrame dropped from 0.98GB → 0.52GB
  deep memory (-46%), test from 1.33GB → 0.68GB (-49%).

### 3. Workflow: don't hold train's graph hostage while building test's
Even with both fixes above, building train's graph then test's graph
**in the same process** (exactly what the original notebook did, back to
back, before training even started) still ran out of memory in testing —
because nothing was ever freed, so test's graph got built *on top of*
train's still-fully-resident graph, features, and DataFrame. The
notebook is restructured into two phases: fully process **and train** on
TRAIN first, explicitly `del` everything train-specific + `gc.collect()`
once weights are saved (Section 7.5), *then* load TEST for the first
time. Only the trained model, the fitted scaler, and the vocabularies
cross from phase 1 to phase 2 — not one row of train data.

  **Measured:** with all three fixes, the full train→test sequence that
  previously OOM-killed now completes, peaking at 3.83GB in the same
  constrained test environment that couldn't even finish *loading* both
  files at 3.74GB before fix #2.

**If you still hit memory pressure on the real files:** try a high-RAM
runtime, or run Phase 1 (through Section 7.5, which saves
`line_mvgnn_weights.pth` + `preprocessing_state.pkl` to disk) and Phase 2
(Section 8 onward, loading those files back) as two **separate** Colab
sessions, so each starts from a fully clean memory state.

## What's timed, and what we actually saw (small synthetic run)

| Stage | Share of total time |
|---|---|
| **Model Training (total)** | ~48% |
| &nbsp;&nbsp;↳ Forward/backward pass | ~42% |
| &nbsp;&nbsp;↳ Neighbor sampling | ~5% |
| &nbsp;&nbsp;↳ Validation pass | ~2% |
| Account-Context Feature Engineering | ~1-2.5% |
| Transaction Flow Feature Engineering | ~0.5-1% |
| Graph Construction (NetworKit) | ~0.1-0.5% |
| Model Inference | ~0.3-0.7% |
| Data Loading & Normalization | ~0.2% |
| Pair-History / Temporal Features | <0.1% each |
| Graph Export to PyTorch Geometric | <0.1% |
| Account Risk Aggregation | <0.1% |

Training's forward/backward pass dominates at this demo scale. Re-run
Section 12 on the real files for numbers that reflect your actual data —
at 10M+ rows the balance between graph/feature work and training will
shift (graph construction and feature engineering are O(n) or O(n log n)
and now genuinely fast; training cost scales with epochs × batches
regardless of dataset size in between).

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
change.

## File guide

```
aml_gnn_pipeline.ipynb     Main notebook -- run this first (two-phase: train fully, free, then test)
dashboard.py                streamlit run dashboard.py (PRD Section 16)

timing_utils.py             The Timer()/profiling harness everything else uses
synthetic_data.py           Synthetic stand-in dataset generator (same schema as IBM-AML)
data_processing.py          Section 3/4: cleaning, normalization, dtype optimization, shared train/test vocabularies
graph_construction.py       Section 5: NetworKit transaction graph, vectorized edges, CSR adjacency
feature_engineering.py      Sections 7-9: temporal / pair-history / flow / account-context features
pyg_export.py               Section 4: DataFrame + edges -> torch_geometric.data.Data
neighbor_sampling.py        Section 11: mini-batch neighbor sampler (NeighborLoader substitute)
model.py                    Sections 6 & 10: LineMVGNN (multi-view GNN) + MLP classifier head
train.py                    Section 11-12: training loop, evaluation metrics
aggregation.py               Sections 14-16: transaction/account risk views, alerting

test_pipeline.py            Standalone smoke-test script, mirrors the notebook's two-phase flow
requirements.txt
```

# AML Detection: GNN -> Tree Ensemble Migration

## What changed

| File | Status | Why |
|---|---|---|
| `data_processing.py` | **Unchanged** | Cleaning/typing/vocab logic doesn't depend on the model choice. |
| `graph_construction.py` | **Unchanged** | Still builds the transaction graph — but now purely as an input to feature engineering, not to a GNN. |
| `feature_engineering.py` | **Unchanged** | All 38 engineered numeric features (temporal, pair-history, graph-topology, account-behavioral) carry over directly as tabular model input. |
| `aggregation.py` | **Unchanged** | Only needs a `probs` array + the cleaned/engineered DataFrame — doesn't care how `probs` was produced. |
| `feature_columns.py` | **New** | Replaces `pyg_export.py`. Defines the feature contract (which columns, numeric vs. categorical) without building a PyG `Data` object. |
| `model.py` | **Rewritten** | `LineMVGNN` → `AMLEnsemble` (XGBoost + LightGBM + Random Forest, stacked with logistic regression). |
| `train.py` | **Rewritten** | Neighbor-sampled epoch loop → single-pass `fit()` per base learner with internal early stopping. Same chronological split / F1-threshold-on-val philosophy as before, and fixes a bug in the original notebook where the threshold-selection cell referenced an undefined `val_probs`. |
| `dashboard.py` | **Updated** | Same Transaction/Account views, plus a new **Model Insights** tab (feature importance) since trees expose this natively and the GNN didn't without extra work. |
| `pyg_export.py` | **Removed** | No PyG `Data` object needed. |
| `neighbor_sampling.py` | **Removed** | No mini-batch graph sampling needed — trees train on the full tabular matrix directly. |
| `requirements.txt` | **Updated** | Dropped `torch` / `torch-geometric`; added `xgboost`, `lightgbm`, `joblib`. |

## Why the graph construction stayed

The GNN and the tree ensemble both need the **same graph-derived features** —
`predecessor_count`, `fan_in`/`fan_out`, `relay_timing_score`,
`short_cycle_participation`, etc. The GNN used the graph two ways: (1) to
compute those features, and (2) to message-pass over it at both train and
inference time via `neighbor_sampling.py`. The ensemble only needs (1) — the
graph is walked once during feature engineering and never touched again.
That's most of where the speedup comes from: no per-batch neighbor sampling,
no repeated subgraph construction, no epochs.

## Why raw account/bank IDs are excluded from the model

The GNN indexed `sender_idx`/`receiver_idx` into a learned `nn.Embedding`
table, which can generalize because the embedding is optimized jointly with
the task. Feeding that same high-cardinality integer straight into a tree
just invites memorization of specific accounts and won't generalize to
accounts unseen in training. See the comment block at the top of
`feature_columns.py` — every generalizable signal account identity carries
is already captured by the `sender_*`/`receiver_*` behavioral aggregates
(historical flow, entropy, concentration, velocity, unique counterparties).
Bank/payment-format/currency codes stay in, as native categorical splits in
XGBoost/LightGBM.

## Running it

```bash
pip install -r requirements.txt
jupyter notebook aml_pipeline_xgboost.ipynb
```

The notebook mirrors the original's structure: load/clean → build graph →
engineer features → **train ensemble** → score test set → export
`transaction_view.csv` / `account_view.csv` / `feature_importance.csv` for
the dashboard.

```bash
streamlit run dashboard.py
```

## Extending the ensemble

`model.py`'s `AMLEnsemble` is deliberately a flat, swappable list of base
learners plus a stacker — to add a fourth base model (e.g. CatBoost, which
has particularly strong native categorical handling and is worth trying if
`from_bank_idx`/`to_bank_idx` cardinality is large in your real data), add a
model to `fit()`/`_base_probs()`/`feature_importance()` and widen the
stacker's input.

## A note on model quality, not just speed

This migration is presented as a systems/production tradeoff (a GNN is
heavier to serve, harder to version, and slower to iterate on), not as a
strict quality upgrade. Tree ensembles on rich engineered features are a
strong, standard, and often very competitive baseline for tabular fraud/AML
detection — but a GNN can in principle capture indirect, multi-hop
laundering patterns that no hand-engineered feature fully summarizes. If
recall on the hardest, most indirect laundering patterns regresses
noticeably on your real data, that's the tradeoff to weigh against the
production benefits here.

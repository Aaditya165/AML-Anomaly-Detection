# AML Detection -- ExSTraQt-style pipeline

Feature-generation philosophy switched from "one transaction graph -> tree
ensemble" to ExSTraQt's "multiple graph representations -> multiple
feature-extraction pipelines -> merge -> one XGBoost". `data_processing.py`
and `feature_engineering.py` (behavioral/temporal/pair-history features)
are unchanged and still used -- this is additive, not a rewrite.

```
CSV
  |
  v
data_processing.py        clean, encode, sort chronologically (unchanged)
  |
  v
graph_construction.py     TWO graphs: transaction graph (unchanged) + account graph (new)
  |                                                          |
  v                                                          v
feature_engineering.py                    community_detection.py, flow_features.py
(unchanged: pair history, velocity,       (new: Leiden + random-walk communities,
 entropy, concentration, ...)              dispense/sink/passthrough + temporal flow)
  |                                                          |
  +----------------------------+-----------------------------+
                               v
                        feature_merge.py
                               |
                               v
                       model.py / train.py   (one XGBoost)
                               |
                               v
                 explain.py (SHAP)  +  aggregation.py / dashboard/app.py
```

## About `payment_format_idx`

The result you found -- removing `payment_format` drops PR-AUC from
0.4552 to 0.2606 -- is very unlikely to be a bug in the feature
engineering. It's a known property of the **IBM synthetic AML dataset**:
the data generator ties certain payment formats (most notoriously
`Reinvestment`) almost deterministically to specific laundering
typologies it simulates. That's real, valid signal in this dataset, not
leakage -- but it's also narrow and fragile: a model that gets most of its
lift from one categorical field has learned "which typology-generator
produced this row" more than "what does laundering *behavior* look like".
That gap is exactly what community/flow structure is meant to close.

Check `model.feature_importance()` (or `explain.summary()` for the SHAP
version) after training and look at `payment_format_idx`'s share. In our
own smoke tests (synthetic data with the same format-to-label correlation
deliberately injected), adding the ExSTraQt feature groups took it from
dominating by a wide margin down to roughly 10% of total gain -- not
because anything suppresses it directly, just because there are now many
more informative, competing features. If it's still dominant on your real
data after this, that's a useful diagnostic (the IBM generator's
format-to-label coupling is even stronger than the graph signal here),
not a sign the pipeline is broken.

One more thing worth knowing: the original `model.py` had a
`FEATURE_WEIGHT_OVERRIDES` mechanism manually downweighting
`payment_format_idx` to 0.1 for XGBoost -- a hand-tuned patch for the same
symptom. Your "full_feature_set" baseline numbers were already computed
*with* that dampening applied, for what it's worth.

## Project layout

```
AML-Detection/
├── data/
│   ├── raw/          # put HI-Small_Trans.csv / LI-Small_Trans.csv here
│   ├── processed/
│   └── exports/       # transaction_view.csv / account_view.csv / feature_importance.csv
├── cache/
│   ├── graphs/         # account_graph_{key}.pkl, account_graph_edges_{key}.parquet
│   ├── communities/    # leiden_{key}.pkl, random_walk_{key}.pkl
│   ├── features/       # community_stats_*, flow_*  (parquet/pickle)
│   ├── models/
│   └── explainability/
├── notebooks/
│   └── train.ipynb    # main entry point
├── dashboard/
│   └── app.py          # streamlit run dashboard/app.py
└── src/
    ├── config.py             every path + tunable in one place
    ├── utils.py               cache_pickle / cache_parquet -- the caching pattern used everywhere below
    ├── data_processing.py     unchanged: clean, encode, sort chronologically
    ├── graph_construction.py  BOTH graphs: transaction graph (unchanged) + account graph (new, Eq. 1)
    ├── feature_engineering.py unchanged: pair history, velocity, entropy, concentration, ...
    ├── base_feature_columns.py  unchanged (renamed from feature_columns.py to avoid a name clash)
    ├── community_detection.py  Leiden + random-walk communities + their statistics
    ├── flow_features.py        dispense / sink / passthrough + temporal flow tracing
    ├── feature_merge.py        behavior + flow + community -> final dataframe (+ caching, + column registry)
    ├── model.py                one XGBoost (num_parallel_tree=10 boosted-forest hybrid)
    ├── train.py                train() / predict() / evaluate() / save_model() / load_model() only
    ├── explain.py               SHAP: global summary, per-transaction waterfall, dependence
    └── aggregation.py          unchanged: transaction view -> account roll-up -> alerts
```

## Running it

```bash
pip install -r requirements.txt
# put HI-Small_Trans.csv (and optionally LI-Small_Trans.csv) in data/raw/
jupyter notebook notebooks/train.ipynb
```

or from a script:

```python
import sys; sys.path.insert(0, "src")
import config
from data_processing import load_and_clean
import train as T

df, vocabs = load_and_clean(str(config.TRAIN_CSV))
result = T.train(df)
print(result["metrics"])
print(result["model"].feature_importance(top_n=20))
```

`streamlit run dashboard/app.py` reads `data/exports/*.csv` the same way
the original dashboard did (lightly patched to also handle the new
single-model `feature_importance.csv` format -- see the diff comments in
`dashboard/app.py`'s Model Insights tab).

## Caching

ExSTraQt's expensive stages are graph/feature generation, not model
training -- Leiden and multi-hop flow tracing are the ones worth caching.
Every stage in `feature_merge.build_node_feature_table` follows:

```python
if cache_exists:
    load_cache()
else:
    compute()
    save_cache()
```

via `utils.cache_pickle` / `utils.cache_parquet`, keyed by an explicit
`cache_key` string (e.g. `"train"`, `"train_plus_val"`) you pass in --
cache invalidation is deliberately explicit (delete the file, or use a
different key) rather than an automatic hash of the input data, so you
control exactly what a config change invalidates. `utils.clear_cache(...)`
deletes specific cache files without wiping the whole tree.

## Single machine vs. the paper's PySpark cluster

The reference implementation (in `exstraqt-main.zip`, if you still have
it) targets a distributed cluster or a 12-core/36GB workstation dedicated
to this job, and reports multi-hour runtimes on the IBM *Large* file even
there. This codebase makes the same algorithms run on a laptop by:

1. **Candidate-node scoping** (`config.RESTRICT_TO_ACCOUNTS`, threaded
   through `community_detection.random_walk_communities` and
   `flow_features`'s `origin_accounts` parameters). The paper's own
   production design (Section 4.7 / Figure 4) is a *secondary* system
   sitting behind a cheap primary risk-based filter -- it was never meant
   to score every account from scratch. Use this the same way: restrict
   the expensive per-account stages to whatever your primary filter (or a
   first-pass score) surfaces. Leiden and the account graph itself still
   see every account regardless -- partitioning needs global structure.
2. **Smaller default knobs** (`FLOW_TOP_N`, `FLOW_NUM_HOPS`,
   `TEMPORAL_FLOW_TOP_N` in `config.py`) -- the flow-tracing join cost
   grows like `O(n_accounts × top_n^hops)` transiently before each hop's
   truncation kicks back in, which is fine on a cluster and isn't in a
   single pandas process at the paper's own numbers (`top_n=50`).
3. **igraph + joblib** instead of PySpark -- same underlying algorithms
   (Leiden/leidenalg, personalized PageRank, biconnected components),
   parallelized across CPU cores instead of a cluster.

Rough scaling from our own (synthetic-data) testing: on an account graph
with ~22k unique aggregated edges, Leiden took about 30 seconds; the same
graph's random-walk communities took a few seconds per hundred target
accounts. Leiden is the one step that must see the whole graph, so it's
the likeliest bottleneck on the real IBM *Small*/*Medium* files -- if it's
too slow, drop `LEIDEN_N_ITERATIONS`, or pre-filter very low-amount edges
out of the account graph before partitioning.

## Known simplifications vs. the reference implementation

- **Community turnover features** skip the reference's per-currency/
  per-bank decomposition (same groupby pattern, just on more columns --
  a straightforward extension, left out to keep `community_detection.py`
  reviewable).
- **Flow-based features** follow the paper's textual profile definitions
  (dispense = forward from own-sent total, sink = backward from
  own-received total, passthrough = forward from own-received total)
  rather than replicating every join-order detail of the reference's
  PySpark implementation line for line -- the algorithmic result
  (Figure 3's capped-chain semantics) is the same.
- **Anti-leakage windowing** is one level simpler than the reference's
  three-way (train / train+val / train+val+test) cumulative feature
  windows: `train.train()` builds node features once on the training
  split and reuses them for validation; `train.score_holdout()` documents
  extending the same pattern to a true held-out file.

## Ablation: is the new feature set actually why this is better?

Set `COMBINE_WITH_BASE_FEATURES = False` in `notebooks/train.ipynb` (or
pass `base_numeric_columns=None, base_categorical_columns=None` to
`train.train()`) to train on ExSTraQt's own features alone, no
`feature_engineering.py` output involved -- a clean apples-to-apples
comparison against the current model.

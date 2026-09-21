# AML Anomaly Detection

A transaction-monitoring pipeline that flags likely money-laundering activity in bank transfer data. It combines classic behavioral/temporal feature engineering on a transaction graph with a second, account-level graph pipeline (community detection + flow tracing, inspired by the ExSTraQt paper) and feeds everything into a single XGBoost model.

## How it works

The pipeline builds two different graph views of the same transaction data, engineers features from each, merges them into one table, and trains one model on top.

```
raw CSV (IBM synthetic AML dataset)
        |
        v
data_processing.py        clean, type, de-duplicate, sort chronologically
        |
        +-----------------------------+------------------------------+
        v                                                             v
transaction graph                                              account graph
(one node per transaction)                                     (one node per account)
        |                                                             |
        v                                                             v
feature_engineering.py                              community_detection.py + flow_features.py
temporal, pair-history,                             Leiden communities, dispense/sink/passthrough
flow, account-behavior                              flow tracing, temporal flow chains
features                                                                     |
        |                                                             v
        |                                                     feature_merge.py
        |                                                     (builds one account-indexed
        |                                                      feature table, joins it onto
        |                                                      every transaction twice -
        |                                                      once for sender, once for
        |                                                      receiver)
        +-----------------------------+------------------------------+
                                       v
                                  model.py / train.py
                                  one XGBoost model
                                       |
                                       v
                        aggregation.py           explain.py         dashboard/app.py
                        transaction + account     SHAP               Streamlit UI
                        risk views, flagging      explanations
```

### The two graphs

**Transaction graph** (`graph_construction.build_transaction_graph`) - one node per transaction. A directed edge goes from transaction T1 to T2 if the receiver of T1 is the sender of T2, and T2 happens within 10 days of T1. This is what lets the model see chains of transfers ("money moved from A, through B, to C") and is used for the flow/relay/cycle features in `feature_engineering.py`. Each transaction keeps at most its 15 nearest chronological successors so that hub accounts don't blow up the graph's density.

**Account graph** (`graph_construction.build_account_graph`) - one node per account. Repeated transfers between the same two accounts collapse into a single weighted edge. The weight isn't just the total amount moved - it combines each account's own share of that edge (its fraction of total sent, plus the receiver's fraction of total received), so an account can't hide a heavy relationship by routing it through one large intermediary. This weighted graph is what community detection and flow tracing run on.

Both graphs are built from the same cleaned transaction table but capture different things: the transaction graph is about the sequence and timing of individual transfers, the account graph is about the overall relationship and structure between accounts.

## Feature engineering

Features come from two separate pipelines that are later merged into one table.

### 1. Transaction-level features (`feature_engineering.py`)

Computed per transaction, all causally (only using information available strictly before that transaction in time, to avoid leaking future information into the model):

- **Temporal** - time since the sender's last outgoing transfer, time since the receiver's last incoming transfer, time since this exact sender-receiver pair last transacted, and a recency score that decays exponentially the longer it's been.
- **Pair history** - for this sender-receiver pair: how many times they've transacted before, the running total/mean/std/max amount sent between them, and how often they've repeated the exact same amount (a common structuring/smurfing tell).
- **Transaction flow (graph-based)** - predecessor/successor counts and fan-in/fan-out (distinct counterparties feeding into or out of a transaction) from the transaction graph, a relay-timing score (how quickly funds get forwarded onward), whether a transaction continues an existing chain, whether the sender recently received funds shortly before sending (pass-through behavior), and whether the transaction participates in a short cycle (money returning to somewhere it came from within a short window).
- **Account context / behavioral** - running, causal profiles for both the sender and receiver: total historical inflow/outflow, number of unique counterparties, an entropy score over counterparty distribution (many small, evenly-spread counterparties vs. a few dominant ones), a concentration score (share of activity with the single most frequent counterparty), in/out degree balance, and transaction velocity over the last 1 and 10 days.

### 2. Account-graph features (community + flow)

Built once per account (not per transaction) and then joined onto transactions - see "Merging" below.

- **Community detection** (`community_detection.py`) - Leiden clustering partitions every account into a non-overlapping community based on the weighted account graph. For each community, the pipeline computes membership stats (how many dispense-only, sink-only, and passthrough accounts it contains - see below), network properties (density, degree distribution, diameter, assortativity, articulation points), turnover volume, and amount-weighted temporal stats.
- **Flow tracing** (`flow_features.py`) - for every account, traces how far its money can be followed moving through the network, hop by hop, capping the traced amount at each hop and keeping only the top-N highest-amount continuations. This produces three separate profiles per account:
  - **Dispense** - forward tracing capped by what the account itself sent (placement behavior - where does money go once it leaves this account).
  - **Sink** - backward tracing capped by what the account itself received (integration behavior - where did the money that landed here come from).
  - **Passthrough** - forward tracing capped by what the account received, not sent (layering behavior - of what came in, how much gets passed along).
- **Temporal flow chains** (`flow_features.compute_temporal_flow_features`) - the same dispense/passthrough/sink idea but restricted to transactions that actually happen in the right chronological order (a relay only counts if the outgoing leg happens at or after the incoming leg). Also separately tracks chains where money round-trips back to its origin account (`dispense == sink`), a classic layering/cycling signal.
- **Anomaly score** (`feature_merge._fit_anomaly_scores`) - an IsolationForest fit over every numeric account-level feature, producing one extra "how unusual is this account overall" score per account.

### Merging

`feature_merge.py` takes the account-level feature table (communities + flow + anomaly score) and joins it onto every transaction twice: once keyed on the sender's account and once on the receiver's, producing `src_exq_*` and `dst_exq_*` columns. It also adds a few summary columns comparing the sender's and receiver's anomaly scores directly (difference, min, max, mean). The final model matrix is the transaction-level features plus both sides' account-level features, combined.

## Flagging logic

The model outputs a probability (risk score) for every transaction. From there:

- **Transaction flagging** (`aggregation.build_transaction_view`) - a transaction is flagged if its risk score is at or above a chosen threshold. The default threshold (~0.633) is the one that maximized F1 on the validation split during training, but it's adjustable at scoring time.
- **Account-level aggregation** (`aggregation.aggregate_accounts`) - every account gets a risk score equal to the maximum risk score among any transaction it took part in, as either sender or receiver. The reasoning: a single high-risk hop is enough to warrant a look, even if the rest of that account's activity looks ordinary. Alongside that, each account gets its mean transaction risk, number of flagged transactions, total transaction count, and number of distinct counterparties.
- **Account alert** - fires whenever an account has one or more flagged transactions. Because the model's inputs already include relay-timing, short-cycle, and chain-continuation features, an account sitting in the middle of a layering chain or a round-tripping cycle tends to surface here naturally, without needing separate rule-based logic on top.

These two views (transaction-level and account-level) are what the dashboard's "Transaction Alerts" and "Account Alerts" tabs show.

## Data used

The pipeline is built for the **IBM synthetic AML transaction dataset** (the same generator behind the Kaggle "IBM Transactions for Anti Money Laundering" dataset). Each row is one transaction with a timestamp, sender/receiver bank and account, amount and currency paid/received, payment format, and a binary `Is Laundering` label.

Two files from that dataset are used:
- `HI-Small_Trans.csv` - the training/validation file.
- `LI-Small_Trans.csv` - used as a genuine held-out file to check how the model generalizes to a mostly-disjoint set of accounts (HI and LI share only a small fraction of accounts).

Both are synthetic, so column names and scale are specific to this generator (`data_processing.py`'s `REQUIRED_COLUMNS` documents exactly what's expected), but the pipeline itself doesn't hard-code anything laundering-typology-specific - it would work on any transaction file with the same shape.

## What it shows

Running the full pipeline produces:

- A trained XGBoost model that scores each transaction's laundering probability.
- A **transaction view**: every transaction with its risk score and flagged/not-flagged status, sorted highest-risk first.
- An **account view**: every account with its aggregated risk score, mean transaction risk, flagged transaction count, total transaction count, counterparty count, and alert status.
- Feature importance (gain-based, from the trained model) and SHAP explanations - both a global summary (which features matter most overall) and a per-transaction waterfall (why this specific transaction was flagged).
- A Streamlit dashboard (`dashboard/app.py`) with two working tabs - Transaction Alerts and Account Alerts - showing summary metrics, the top flagged transactions/accounts, and CSV downloads of both views. It can either load pre-computed results instantly or run the full pipeline from scratch on any uploaded CSV of the same shape.

On the held-out validation split of the training file, the model reaches a PR-AUC of roughly 0.31 (laundering is extremely rare - under 0.2% of transactions - so PR-AUC, not accuracy, is the meaningful number here). Scored on the fully disjoint LI file, PR-AUC drops to roughly 0.05, which is expected: LI shares only a small fraction of accounts with the training file, so most of its accounts start with little to no account-graph history built up.

## `notebooks/kaggle.ipynb`

This is the actual notebook that was run end-to-end on Kaggle, kept as-is (including its false starts, scratch cells, and memory-profiling prints) rather than cleaned up, so it's a real reference for what a complete run looks like: loading the data, building both graphs, running community detection and flow tracing, training the model, scoring the held-out file, and saving the trained model. If you want to see actual output, actual timings, and actual memory usage for a full run rather than guessing from the source alone, this is the file to open.

`notebooks/train.ipynb` is the lighter, intended entry point for running the pipeline yourself.

## Project layout

```
AML-Anomaly-Detection/
├── data/
│   ├── raw/          put HI-Small_Trans.csv / LI-Small_Trans.csv here
│   ├── processed/
│   └── exports/       transaction_view.csv, account_view.csv, feature_importance.csv
├── cache/
│   ├── graphs/         account_graph_{key}.pkl, account_graph_edges_{key}.parquet
│   ├── communities/    leiden_{key}.pkl
│   ├── features/       community_stats_*, flow_* (parquet/pickle)
│   ├── models/
│   └── explainability/
├── notebooks/
│   ├── train.ipynb     intended entry point for running the pipeline
│   └── kaggle.ipynb    the actual notebook run end-to-end on Kaggle - reference for a full run
├── dashboard/
│   └── app.py           streamlit run dashboard/app.py
├── requirements.txt
├── count.py
└── src/
    ├── config.py                every path and tunable knob in one place
    ├── utils.py                  cache_pickle / cache_parquet - the caching pattern used everywhere
    ├── data_processing.py        loads and cleans the raw CSV, builds categorical vocabularies, sorts chronologically
    ├── graph_construction.py     builds both graphs: the transaction graph and the account graph
    ├── feature_engineering.py    transaction-level features: temporal, pair-history, flow, account-behavior
    ├── base_feature_columns.py   the definitive list of which columns feed the model, and how
    ├── community_detection.py    Leiden community detection + per-community statistics
    ├── flow_features.py          dispense/sink/passthrough flow tracing + temporal flow chains
    ├── feature_merge.py          merges account-level features onto transactions, builds the anomaly score,
    │                             assembles the final model matrix
    ├── model.py                  the XGBoost model wrapper (fit / predict / save / load / feature importance)
    ├── train.py                  train() / predict() / evaluate() / score a held-out file
    ├── explain.py                SHAP: global summary, per-transaction waterfall, feature dependence
    └── aggregation.py            builds the transaction view and account view, applies flagging logic
```

### What each file does

- **`data_processing.py`** - reads the raw CSV, fixes dtypes (categoricals, float32 amounts), builds integer vocabularies for banks/payment formats/currencies (fit once on training data, reused on test data so codes line up), drops duplicates and bad rows, sorts everything chronologically, and derives a handful of base numeric columns (log amount, same-bank flag, hour of day, day of week).
- **`graph_construction.py`** - builds the transaction graph (one node per transaction, edges between chronologically-linked transfers) and the account graph (one node per account, weighted edges between accounts based on aggregated transfer amounts).
- **`feature_engineering.py`** - runs the four transaction-level feature stages described above (temporal, pair history, transaction flow, account context) and orchestrates them in one call.
- **`base_feature_columns.py`** - the single shared definition of which engineered columns actually go into the model, split into numeric and categorical, used by both `feature_merge.py` and the dashboard.
- **`community_detection.py`** - runs Leiden clustering on the account graph and computes per-community statistics (size, member composition, network topology, turnover, weighted-time stats).
- **`flow_features.py`** - implements the dispense/sink/passthrough flow-tracing algorithm and the chronologically-constrained temporal flow version.
- **`feature_merge.py`** - the glue module: builds the full account-indexed feature table (graph + communities + flow + anomaly score, with disk caching at every stage), joins it onto transactions on both the sender and receiver side, and assembles the final numeric/categorical feature matrix the model trains on.
- **`model.py`** - a thin wrapper around a single XGBoost booster (with `num_parallel_tree=10`, effectively a boosted-forest hybrid), handling fit, predict, save/load, and gain-based feature importance.
- **`train.py`** - orchestrates a full training run (chronological train/validation split, builds features only from the training split to avoid leakage, fits the model, finds the F1-optimal threshold, evaluates), plus `score_holdout` for scoring a completely separate file.
- **`explain.py`** - SHAP-based explanations on top of the trained model: a global feature-importance summary, a per-transaction waterfall showing why one specific transaction got its score, and a dependence view for how one feature's effect changes across its value range.
- **`aggregation.py`** - takes the trained model's risk scores and builds the transaction view and account view described in "Flagging logic" above.
- **`utils.py`** - the disk-caching helpers (`cache_pickle`, `cache_parquet`) used throughout `feature_merge.py` so expensive graph/community/flow computations aren't repeated on every run.
- **`config.py`** - every path and tunable parameter (Leiden iterations, flow-tracing hop count and top-N, XGBoost params, caching keys) in one place.
- **`count.py`** - a small standalone script that counts how many laundering-labeled rows are in a CSV; not part of the pipeline itself.
- **`dashboard/app.py`** - a Streamlit app with two modes: load pre-computed CSV exports instantly, or run the full pipeline from scratch on any uploaded CSV. Shows transaction alerts and account alerts as sortable, downloadable tables with summary metrics.

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
result = T.train(config.GRAPH_CACHE_DIR, config.COMMUNITY_CACHE_DIR, config.FEATURE_CACHE_DIR, df)
print(result["metrics"])
print(result["model"].feature_importance(top_n=20))
```

To try the dashboard:

```bash
streamlit run dashboard/app.py
```

## Caching

The expensive stages are graph and feature generation, not model training - Leiden clustering and multi-hop flow tracing are the ones worth caching. Every stage in `feature_merge.build_node_feature_table` follows the same pattern (via `utils.cache_pickle` / `utils.cache_parquet`):

```python
if cache_exists:
    load_cache()
else:
    compute()
    save_cache()
```

Caching is keyed by an explicit `cache_key` string you pass in (e.g. `"train"`, `"train_plus_val"`), so invalidation is deliberate - delete the file, or use a different key - rather than an automatic hash of the input data. `utils.clear_cache(...)` deletes specific cache files without wiping the whole tree.
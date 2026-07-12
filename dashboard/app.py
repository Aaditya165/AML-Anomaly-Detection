"""
app.py -- AML Detection Dashboard (ExSTraQt pipeline)
=====================================================
PRD Section 16: Transaction View, Account View, plus Model Insights.

Run with:
    streamlit run dashboard/app.py

Upload size: Streamlit's default cap is 200 MB, which the IBM AML files
blow past. `.streamlit/config.toml` (next to this project's root) raises
it -- see that file. You can also override on the command line:
    streamlit run dashboard/app.py --server.maxUploadSize=2048

TWO MODES
---------
1. "Instant (pre-computed)" -- loads exports/transaction_view.csv and
   exports/account_view.csv that the notebook already wrote. Sub-second.
   Use this if you need the demo to be immediate and you're showing
   results, not the machinery.

2. "Compute from uploaded CSV" -- runs the real pipeline on whatever you
   upload, reusing the disk caches under cache/ so the genuinely expensive
   stages (Leiden community detection ~72 min, community stats + flow
   features ~10 min) are skipped entirely.

   HONEST TIMING: even with every cache hit, this is NOT instant. These
   stages are NOT cached and run on the uploaded file every time:
       load_and_clean            ~35 s
       build_transaction_graph   ~19 s
       engineer_all_features    ~165 s   <-- the bulk
       chunked scoring (6M rows) ~60-90 s
   Budget ~4-5 minutes on an IBM *-Small file. The caches turn ~85 minutes
   into ~5, which is the difference that matters -- but plan the demo
   around 5 minutes, not 5 seconds.

CACHE / UPLOAD CONSISTENCY (important)
--------------------------------------
The cached artifacts are keyed by CACHE_KEY (sidebar). Each key's caches
were built from ONE specific file's account graph:
    "train_plus_val"     -> built from HI-Small_Trans.csv
    "holdout_selfbuilt"  -> built from LI-Small_Trans.csv
You must upload the file that MATCHES the selected key. Upload HI with the
LI caches (or vice versa) and the account graph won't correspond to the
uploaded transactions -- accounts won't be found in the node table, their
ExSTraQt features silently become zeros, and the scores will be garbage.
The app checks this by comparing the uploaded file's accounts against the
cached node table and warns loudly if the overlap is low.
"""

import gc
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# --- make ../src importable regardless of where streamlit is launched from ---
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))

from src.data_processing import load_and_clean
from src.graph_construction import build_transaction_graph
from src.feature_engineering import engineer_all_features
from src.base_feature_columns import NUMERIC_FEATURE_COLUMNS, CATEGORICAL_FEATURE_COLUMNS
from src import feature_merge as fm
from src.model import Model
from src.aggregation import build_transaction_view, aggregate_accounts


# ===========================================================================
# Hardcoded paths (per the brief -- swap to sidebar inputs later if needed)
# ===========================================================================
CACHE_DIR = _ROOT / "cache"
GRAPH_CACHE_DIR = CACHE_DIR / "graphs"
COMMUNITY_CACHE_DIR = CACHE_DIR / "communities"
FEATURE_CACHE_DIR = CACHE_DIR / "features"
MODEL_CACHE_DIR = CACHE_DIR / "models"
EXPORT_DIR = _ROOT / "exports"

MODEL_PATH = MODEL_CACHE_DIR / "exstraqt_model.json"
DEFAULT_THRESHOLD = 0.6334          # from the HI validation run
SCORE_BATCH_SIZE = 500_000          # chunked scoring -- see train.score_holdout

st.set_page_config(page_title="AML Detection Dashboard", layout="wide")
st.title("AML Detection Dashboard")
st.caption("ExSTraQt-style graph features + XGBoost. Transaction risk scores, account roll-ups, model drivers.")


# ===========================================================================
# Sidebar
# ===========================================================================
with st.sidebar:
    st.header("Mode")
    mode = st.radio(
        "How should results be produced?",
        ["Instant (pre-computed exports)", "Compute from uploaded CSV"],
        help="Instant loads the CSVs the notebook already wrote. Compute runs the "
             "real pipeline on your upload, reusing disk caches.",
    )

    st.divider()
    st.header("Scoring")
    risk_threshold = st.slider(
        "Risk score threshold", 0.0, 1.0, float(DEFAULT_THRESHOLD), 0.01,
        help="Transactions at or above this score are flagged. The tuned value from "
             "the HI validation split was 0.6334.",
    )

    if mode == "Compute from uploaded CSV":
        st.divider()
        st.header("Cache")
        cache_key = st.selectbox(
            "Cached account graph to reuse",
            ["train_plus_val", "holdout_selfbuilt"],
            help="train_plus_val = built from HI-Small_Trans.csv. "
                 "holdout_selfbuilt = built from LI-Small_Trans.csv. "
                 "MUST match the file you upload.",
        )
        expected_file = "HI-Small_Trans.csv" if cache_key == "train_plus_val" else "LI-Small_Trans.csv"
        st.info(f"Upload **{expected_file}** to match this cache key.")


# ===========================================================================
# Loaders
# ===========================================================================
@st.cache_resource(show_spinner=False)
def load_model():
    if not MODEL_PATH.exists():
        return None
    return Model.load(str(MODEL_PATH))


@st.cache_data(show_spinner=False)
def load_exported_views():
    txn_p, acct_p = EXPORT_DIR / "transaction_view.csv", EXPORT_DIR / "account_view.csv"
    if not txn_p.exists() or not acct_p.exists():
        return None, None
    return pd.read_csv(txn_p), pd.read_csv(acct_p)


def compute_from_upload(uploaded_file, cache_key: str, threshold: float):
    """Full pipeline on an uploaded CSV, reusing the disk caches."""
    tmp_csv = Path(tempfile.gettempdir()) / uploaded_file.name
    with open(tmp_csv, "wb") as fh:
        fh.write(uploaded_file.getbuffer())

    model = load_model()
    if model is None:
        st.error(f"No trained model at `{MODEL_PATH}`. Run the notebook's save-model cell first.")
        return None, None

    status = st.status("Running the ExSTraQt pipeline...", expanded=True)

    with status:
        st.write("**1/5** Cleaning + typing the raw CSV...")
        df, _ = load_and_clean(str(tmp_csv))
        tmp_csv.unlink(missing_ok=True)  # done with it -- don't leave uploads piling up in temp
        st.write(f"&nbsp;&nbsp;&nbsp;&nbsp;{len(df):,} transactions after cleaning.")

        st.write("**2/5** Building the transaction graph + engineering base features...")
        _, s, d, preds, succs = build_transaction_graph(df)
        df = engineer_all_features(df, preds, succs, s, d)
        del s, d, preds, succs
        gc.collect()

        st.write("**3/5** Loading cached ExSTraQt node features (account graph, "
                 "Leiden communities, flow tracing)...")
        node_features = fm.build_node_feature_table(
            GRAPH_CACHE_DIR, COMMUNITY_CACHE_DIR, FEATURE_CACHE_DIR,
            df, cache_key=cache_key, use_cache=True,
        )

        # --- consistency guard: do the uploaded file's accounts actually exist
        #     in the cached node table? If not, the caches belong to a different
        #     file and every ExSTraQt feature would silently be zero-filled. ---
        uploaded_accounts = pd.unique(pd.concat(
            [df["Sender Account"].astype(str), df["Receiver Account"].astype(str)],
            ignore_index=True,
        ))
        cached_accounts = set(node_features.index.astype(str))
        overlap = np.mean([a in cached_accounts for a in uploaded_accounts[:20_000]])
        if overlap < 0.5:
            st.error(
                f"**Cache / upload mismatch.** Only {overlap:.1%} of the uploaded file's "
                f"accounts appear in the `{cache_key}` cached node table. These caches were "
                f"built from a different file, so the graph features would be almost entirely "
                f"zeros and the scores meaningless. Pick the matching cache key, or upload "
                f"the file these caches were built from."
            )
            status.update(label="Aborted -- cache/upload mismatch", state="error")
            return None, None
        st.write(f"&nbsp;&nbsp;&nbsp;&nbsp;Account overlap with cache: **{overlap:.1%}** — OK.")

        st.write(f"**4/5** Scoring {len(df):,} transactions (in {SCORE_BATCH_SIZE:,}-row chunks)...")
        probs = _score_in_chunks(model, df, node_features)

        st.write("**5/5** Building transaction view + rolling up to accounts...")
        view_cols = ["txn_id", "Sender Account", "Receiver Account", "Amount Paid", "Timestamp"]
        df_view = df.loc[:, view_cols].copy()
        del df, node_features
        gc.collect()

        txn_view = build_transaction_view(df_view, probs, threshold=threshold)
        acct_view = aggregate_accounts(df_view, probs, threshold=threshold)

    status.update(label="Pipeline complete", state="complete", expanded=False)
    return txn_view, acct_view


def _score_in_chunks(model, df, node_features):
    """Chunked scoring: the full model matrix AND XGBoost's internal DMatrix
    copy of it would otherwise both be live at once (~8 GB each on a 6M-row
    file). Row-independent, so identical to scoring in one shot."""
    n = len(df)
    probs = np.empty(n, dtype=np.float32)
    progress = st.progress(0.0)
    for start in range(0, n, SCORE_BATCH_SIZE):
        end = min(start + SCORE_BATCH_SIZE, n)
        X = fm.build_model_matrix(
            df.iloc[start:end], node_features,
            NUMERIC_FEATURE_COLUMNS, CATEGORICAL_FEATURE_COLUMNS,
        )
        probs[start:end] = model.predict_proba(X)
        del X
        gc.collect()
        progress.progress(end / n)
    progress.empty()
    return probs


# ===========================================================================
# Get the two views, per the selected mode
# ===========================================================================
txn_view = acct_view = None

if mode == "Instant (pre-computed exports)":
    txn_view, acct_view = load_exported_views()
    if txn_view is None:
        st.warning(
            f"No pre-computed exports found in `{EXPORT_DIR}`. Run the notebook's "
            f"export cell, or switch to **Compute from uploaded CSV**."
        )
    else:
        st.success(f"Loaded pre-computed results — {len(txn_view):,} transactions, "
                   f"{len(acct_view):,} accounts.")
else:
    uploaded = st.file_uploader(
        "Upload a transaction CSV (IBM AML format)",
        type=["csv"],
        help="Must have the raw IBM columns: Timestamp, From Bank, Account, To Bank, "
             "Account.1, Amount Received/Paid, currencies, Payment Format, Is Laundering.",
    )
    if uploaded is not None:
        txn_view, acct_view = compute_from_upload(uploaded, cache_key, risk_threshold)
    else:
        st.info("Upload a CSV to run the pipeline.")


# ===========================================================================
# Render
# ===========================================================================
if txn_view is not None and acct_view is not None:
    tab_txn, tab_acct, tab_model = st.tabs(
        ["Transaction Alerts", "Account Alerts", "Model Insights"]
    )

    # ---------------- Transaction View ----------------
    with tab_txn:
        flagged = txn_view[txn_view["Risk Score"] >= risk_threshold]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Transactions", f"{len(txn_view):,}")
        c2.metric("Flagged", f"{len(flagged):,}")
        c3.metric("Flag rate", f"{len(flagged)/max(len(txn_view),1):.3%}")
        c4.metric("Threshold", f"{risk_threshold:.4f}")

        # fig = px.histogram(
        #     txn_view, x="Risk Score", nbins=60, log_y=True,
        #     title="Distribution of transaction risk scores (log-scaled count)",
        # )
        # fig.add_vline(x=risk_threshold, line_dash="dash", line_color="red")
        # st.plotly_chart(fig, use_container_width=True)

        st.subheader("Top alert transactions")
        st.dataframe(
            txn_view.sort_values("Risk Score", ascending=False).head(200),
            use_container_width=True, hide_index=True,
        )
        st.download_button(
            "Download transaction_view.csv",
            txn_view.to_csv(index=False).encode(),
            "transaction_view.csv", "text/csv",
        )

    # ---------------- Account View ----------------
    with tab_acct:
        alerted = (acct_view[acct_view["Account Alert"]]
                   if "Account Alert" in acct_view.columns else acct_view.iloc[0:0])
        c1, c2, c3 = st.columns(3)
        c1.metric("Accounts", f"{len(acct_view):,}")
        c2.metric("Accounts alerted", f"{len(alerted):,}")
        c3.metric("Alert rate", f"{len(alerted)/max(len(acct_view),1):.2%}")

        st.subheader("Top alert accounts")
        st.caption(
            "An account's risk score is the MAX across any transaction it took part in — "
            "one high-risk hop warrants a look even if the rest of its activity is ordinary."
        )
        st.dataframe(
            acct_view.sort_values("Associated Risk Score", ascending=False).head(200),
            use_container_width=True, hide_index=True,
        )

        if "Number of Flagged Transactions" in acct_view.columns:
            top20 = acct_view.sort_values("Associated Risk Score", ascending=False).head(20)
            fig = px.bar(
                top20, x="Account ID", y="Number of Flagged Transactions",
                title="Flagged transaction count — top 20 riskiest accounts",
            )
            fig.update_layout(xaxis_type="category", xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

        st.download_button(
            "Download account_view.csv",
            acct_view.to_csv(index=False).encode(),
            "account_view.csv", "text/csv",
        )

    # ---------------- Model Insights ----------------
    with tab_model:
        model = load_model()
        if model is None:
            st.warning(f"No model at `{MODEL_PATH}` — feature importance unavailable.")
        else:
            st.subheader("What drives the model's risk scores")
            st.caption(
                "Gain-based importance from the single XGBoost model, normalized to sum to 1. "
                "Raw account identity and bank IDs are deliberately excluded from the feature "
                "set — they don't transfer across datasets (HI and LI share ~1% of accounts). "
                "The `src_exq_*` / `dst_exq_*` features are the ExSTraQt graph features for the "
                "sender and receiver respectively."
            )
            imp = model.feature_importance(top_n=25)
            fig = px.bar(
                imp.sort_values("importance"), x="importance", y="feature",
                orientation="h", height=700,
                title="Top 25 features by gain",
            )
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(imp, use_container_width=True, hide_index=True)
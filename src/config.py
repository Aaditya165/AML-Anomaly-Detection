"""
config.py
---------
Every path and tunable knob in one place.

PATHS
-----
Mirrors the project layout: raw CSVs under data/raw, the disk-cache tree
under cache/<stage>/, exports for the dashboard under data/exports.

EXSTRAQT TUNABLES
-----------------
See each docstring for the reasoning; the short version is that these are
deliberately smaller than the paper's own numbers because the paper's
numbers were produced on a PySpark cluster / a 12-core, 36GB workstation
dedicated to exactly this job, and this codebase runs on a single machine
(see README.md's "single machine vs. PySpark cluster" section).
"""

from pathlib import Path

# --- paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EXPORTS_DIR = DATA_DIR / "exports"

CACHE_DIR = PROJECT_ROOT / "cache"
GRAPH_CACHE_DIR = CACHE_DIR / "graphs"
COMMUNITY_CACHE_DIR = CACHE_DIR / "communities"
FEATURE_CACHE_DIR = CACHE_DIR / "features"
MODEL_CACHE_DIR = CACHE_DIR / "models"
EXPLAIN_CACHE_DIR = CACHE_DIR / "explainability"

for _dir in (RAW_DIR, PROCESSED_DIR, EXPORTS_DIR, GRAPH_CACHE_DIR,
             COMMUNITY_CACHE_DIR, FEATURE_CACHE_DIR, MODEL_CACHE_DIR, EXPLAIN_CACHE_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

TRAIN_CSV = RAW_DIR / "HI-Small_Trans.csv"
TEST_CSV = RAW_DIR / "LI-Small_Trans.csv"

SEED = 42

# --- Section 4.1 Community Detection ---
LEIDEN_N_ITERATIONS = 100
RANDOM_WALK_N_HOPS = 2
RANDOM_WALK_CANDIDATE_TOP_N = 100
RANDOM_WALK_PPR_DAMPING = 0.85
RANDOM_WALK_MEMBERSHIP_THRESHOLD = 0.01
RANDOM_WALK_N_JOBS = -1

# --- Section 4.2 Flow-based features (Algorithm 1) ---
# Memory scales like O(n_accounts * FLOW_TOP_N^FLOW_NUM_HOPS) transiently
# per hop before truncation kicks back in -- raise these only with RAM/
# cores to match. See flow_features.py's module docstring.
FLOW_TOP_N = 15
FLOW_NUM_HOPS = 4

# --- Section 4.3 Temporal flow features ---
TEMPORAL_FLOW_TOP_N = 20

# --- Section 4.5 Anomaly scoring ---
ISOLATION_FOREST_N_ESTIMATORS = 1_000
ISOLATION_FOREST_MAX_SAMPLES = "auto"

# --- Model ---
XGB_PARAMS = dict(
    seed=SEED,
    max_depth=6,
    eta=0.3,
    subsample=1.0,
    colsample_bytree=0.5,
    num_parallel_tree=10,     # the paper's "boosted random-forest" hybrid
    objective="binary:logistic",
    eval_metric="aucpr",
    disable_default_eval_metric=True,
    tree_method="hist",
)
XGB_NUM_BOOST_ROUND = 100
XGB_EARLY_STOPPING_ROUNDS = 10

# --- Candidate-node scoping (see graph_construction.py / flow_features.py /
# community_detection.py) ---
# None = compute full ExSTraQt node features for every account in the file.
# Set to an iterable of account ids (e.g. accounts touched by your existing
# primary risk-based filter) to keep the expensive per-account stages
# tractable on anything bigger than the IBM *Small* file.
RESTRICT_TO_ACCOUNTS = None

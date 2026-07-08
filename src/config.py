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


SEED = 42

MAX_GRAPH_METRICS_COMMUNITY_SIZE = 5000
#to prevent memory blowup when computing graph metrics on the largest communities

# --- Section 4.1 Community Detection ---
LEIDEN_N_ITERATIONS = 100

# Cap used when leiden_membership is passed to random_walk_communities (see
# its docstring) -- Leiden isn't guaranteed to produce evenly-sized
# communities on every graph, so this is the safety net against the
# unlucky accounts that land in one hub-dominated community.
RANDOM_WALK_MAX_CANDIDATES_FROM_LEIDEN = 500

# --- Section 4.2 Flow-based features (Algorithm 1) ---
# Memory scales like O(n_accounts * FLOW_TOP_N^FLOW_NUM_HOPS) transiently
# per hop before truncation kicks back in -- raise these only with RAM/
# cores to match. See flow_features.py's module docstring.
FLOW_TOP_N = 15
FLOW_NUM_HOPS = 4

# --- Section 4.3 Temporal flow features ---
TEMPORAL_FLOW_TOP_N = 20

# --- Section 4.5 Anomaly scoring ---
ISOLATION_FOREST_N_ESTIMATORS = 400
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

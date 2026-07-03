"""
model.py
--------
Replaces the LineMVGNN. Three tree-based base learners over the exact
same engineered feature set the GNN used (feature_columns.py), combined
by a small logistic-regression stacker:

    View A analogue - XGBoost        : gradient-boosted trees, the
                                        workhorse model; strong on the
                                        mixed numeric/categorical feature
                                        set, native categorical support,
                                        handles the extreme class
                                        imbalance via scale_pos_weight.
    View B analogue - LightGBM       : gradient-boosted trees with a
                                        different (leaf-wise) growth
                                        strategy and its own native
                                        categorical splitting -- adds a
                                        second, differently-biased
                                        boosting model to the ensemble
                                        rather than a second copy of
                                        XGBoost's biases.
    View C analogue - Random Forest  : bagged (not boosted) trees --
                                        decorrelated error structure vs.
                                        the two boosters, which is what
                                        actually makes an ensemble worth
                                        it rather than three votes for
                                        the same mistakes.

Fusion analogue - Stacking: a LogisticRegression meta-learner is fit on
the three base models' predicted probabilities on a held-out validation
slice (never on data the base models trained on), the same anti-leakage
principle behind every "prior-only" feature in feature_engineering.py.

Why this is production-appropriate where the GNN wasn't:
  - No graph batching / neighbor sampling at inference time -- scoring a
    transaction is 3 tree lookups + a 3-input logistic regression, all
    CPU, all sub-millisecond even at high volume.
  - No custom PyG plumbing to keep in sync with library versions
    (`neighbor_sampling.py` existed specifically because pyg-lib/
    torch-sparse wheels were unreliable -- that whole failure mode is
    gone).
  - Model artifacts are a handful of small files (`joblib.dump`), easy to
    version, diff, and roll back -- vs. a state_dict tied to an exact
    architecture definition.
  - Every base learner exposes native feature importances, which matters
    for AML specifically: investigators and regulators expect to know
    *why* a transaction was flagged, not just that a black-box embedding
    said so.
"""

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression

import lightgbm as lgb
import xgboost as xgb

from .feature_columns import (
    NUMERIC_FEATURE_COLUMNS,
    CATEGORICAL_FEATURE_COLUMNS,
    ALL_FEATURE_COLUMNS,
    prepare_feature_frame,
)

DEFAULT_XGB_PARAMS = dict(
    n_estimators=1200,
    max_depth=7,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=2,
    reg_lambda=1.0,
    max_delta_step=1,       # stabilizes logistic loss under extreme imbalance
    tree_method="hist",
    device="cuda",          # GPU — remove this line to fall back to CPU
    enable_categorical=True,
    eval_metric="aucpr",
    n_jobs=-1,
    random_state=0,
)

DEFAULT_LGB_PARAMS = dict(
    n_estimators=2000,
    num_leaves=31,
    learning_rate=0.02,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    min_child_samples=50,
    min_child_weight=1e-3,
    reg_lambda=5.0,
    reg_alpha=1.0,
    objective="binary",
    metric="average_precision",
    device="gpu",            # see caveat below before enabling
    n_jobs=-1,
    random_state=0,
    verbosity=-1,
)

LGB_MAX_SCALE_POS_WEIGHT = 50.0

"""
Caveat on device="gpu": 
unlike XGBoost, LightGBM's default PyPI wheel is often CPU-only — 
GPU support usually needs a build with --config-setting=cmake.define.USE_GPU=ON 
or a conda-forge GPU build. Test it first:
import lightgbm as lgb
lgb.LGBMClassifier(device="gpu").fit([[1,2],[3,4]], [0,1])
If that raises something like "GPU Tree Learner was not enabled in this build", 
delete the device="gpu" line and stay CPU for LightGBM 
— it's a much smaller cost than RF anyway.
"""

DEFAULT_RF_PARAMS = dict(
    n_estimators=600,
    max_depth=18,
    min_samples_leaf=3,
    max_features="sqrt",
    class_weight="balanced_subsample",
    n_jobs=-1,
    random_state=0,
)

DEFAULT_ISO_PARAMS = dict(
    n_estimators=200,
    max_samples="auto",
    contamination="auto",
    n_jobs=-1,
    random_state=0,
)


def _scale_pos_weight(y: np.ndarray) -> float:
    n_pos = float((y == 1).sum())
    n_neg = float((y == 0).sum())
    return max(n_neg, 1.0) / max(n_pos, 1.0)


@dataclass
class AMLEnsemble:
    """
    Usage:
        ens = AMLEnsemble()
        ens.fit(df_train, df_val)          # df_* need a "label" column +
                                            # every column in ALL_FEATURE_COLUMNS
        probs = ens.predict_proba(df_test)
        ens.save("cache/ensemble")
        ens2 = AMLEnsemble.load("cache/ensemble")
    """

    xgb_params: dict = field(default_factory=lambda: dict(DEFAULT_XGB_PARAMS))
    lgb_params: dict = field(default_factory=lambda: dict(DEFAULT_LGB_PARAMS))
    rf_params: dict = field(default_factory=lambda: dict(DEFAULT_RF_PARAMS))
    iso_params: dict = field(default_factory=lambda: dict(DEFAULT_ISO_PARAMS))
    early_stopping_rounds: int = 40

    def __post_init__(self):
        self.xgb_model_: xgb.XGBClassifier | None = None
        self.lgb_model_: lgb.LGBMClassifier | None = None
        self.rf_model_: RandomForestClassifier | None = None
        self.iso_model_: IsolationForest | None = None
        self.iso_score_min_: float | None = None
        self.iso_score_max_: float | None = None
        self.stacker_: LogisticRegression | None = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(self, df_train: pd.DataFrame, df_val: pd.DataFrame, label_col: str = "label"):
        y_train = df_train[label_col].values.astype(np.int32)
        y_val = df_val[label_col].values.astype(np.int32)

        X_train = prepare_feature_frame(df_train)
        X_val = prepare_feature_frame(df_val)

        spw = _scale_pos_weight(y_train)
        print(f"scale_pos_weight (train): {spw:.2f} "
              f"({int((y_train == 1).sum())} positive / {int((y_train == 0).sum())} negative)")

        # ---- XGBoost (native pandas categorical dtype support) ----
        self.xgb_model_ = xgb.XGBClassifier(
            **self.xgb_params,
            scale_pos_weight=spw,
            early_stopping_rounds=self.early_stopping_rounds,
        )
        self.xgb_model_.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        print(f"XGBoost: best_iteration={self.xgb_model_.best_iteration}, "
              f"best val PR-AUC={self.xgb_model_.best_score:.4f}")

        # ---- LightGBM (native categorical via pandas 'category' dtype) ----
        self.lgb_model_ = lgb.LGBMClassifier(**self.lgb_params, scale_pos_weight=spw)
        self.lgb_model_.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            categorical_feature=CATEGORICAL_FEATURE_COLUMNS,
            callbacks=[lgb.early_stopping(self.early_stopping_rounds, verbose=False)],
        )
        print(f"LightGBM: best_iteration={self.lgb_model_.best_iteration_}, "
              f"best val PR-AUC={self.lgb_model_.best_score_['valid_0']['average_precision']:.4f}")

        # ---- Random Forest (bagged, decorrelated w.r.t. the two boosters).
        # Categorical columns go in as their existing ordinal integer codes
        # -- sklearn's RandomForestClassifier has no native categorical
        # split type, and one-hot-ing bank IDs would blow up the column
        # count for a single ensemble member whose whole job is just to
        # be a differently-biased vote. Numeric features are identical to
        # the other two models. ----
        X_train_rf = X_train.copy()
        X_val_rf = X_val.copy()
        for c in CATEGORICAL_FEATURE_COLUMNS:
            X_train_rf[c] = X_train_rf[c].cat.codes
            X_val_rf[c] = X_val_rf[c].cat.codes
        self.rf_model_ = RandomForestClassifier(**self.rf_params)
        self.rf_model_.fit(X_train_rf, y_train)

        # ---- Isolation Forest (unsupervised anomaly score, 4th view) ----
        self.iso_model_ = IsolationForest(**self.iso_params)
        self.iso_model_.fit(X_train[NUMERIC_FEATURE_COLUMNS])

        # decision_function is HIGH for normal points, LOW for anomalies --
        # flip sign so higher = more anomalous. Bounds are fixed from TRAIN
        # here and reused at inference (not recomputed per batch) -- doing
        # it per-batch would make a single-row production score always
        # normalize to 0.
        iso_raw_train = -self.iso_model_.decision_function(X_train[NUMERIC_FEATURE_COLUMNS])
        self.iso_score_min_ = float(iso_raw_train.min())
        self.iso_score_max_ = float(iso_raw_train.max())

        # ---- Stack: fit a tiny logistic regression on VAL-set base
        # predictions only, so the meta-learner never sees predictions
        # made on rows the base models were themselves trained on. ----
        base_val_probs = self._base_probs(df_val, X_val=X_val, X_val_rf=X_val_rf)
        self.stacker_ = LogisticRegression(class_weight="balanced", max_iter=1000)
        self.stacker_.fit(base_val_probs, y_val)

        val_probs = self.stacker_.predict_proba(base_val_probs)[:, 1]
        print("Stacker coefficients [xgb, lgb, rf, iso_forest]:", np.round(self.stacker_.coef_[0], 3))
        return self, val_probs

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def _base_probs(self, df: pd.DataFrame, X_val=None, X_val_rf=None) -> np.ndarray:
        X = X_val if X_val is not None else prepare_feature_frame(df)
        if X_val_rf is not None:
            X_rf = X_val_rf
        else:
            X_rf = X.copy()
            for c in CATEGORICAL_FEATURE_COLUMNS:
                X_rf[c] = X_rf[c].cat.codes

        p_xgb = self.xgb_model_.predict_proba(X)[:, 1]
        p_lgb = self.lgb_model_.predict_proba(X)[:, 1]
        p_rf = self.rf_model_.predict_proba(X_rf)[:, 1]

        iso_raw = -self.iso_model_.decision_function(X[NUMERIC_FEATURE_COLUMNS])
        iso_clipped = np.clip(iso_raw, self.iso_score_min_, self.iso_score_max_)
        iso_score = (iso_clipped - self.iso_score_min_) / (self.iso_score_max_ - self.iso_score_min_ + 1e-9)

        return np.column_stack([p_xgb, p_lgb, p_rf, iso_score])

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Full ensemble probability, aligned to df's row order."""
        base_probs = self._base_probs(df)
        return self.stacker_.predict_proba(base_probs)[:, 1]

    def predict_proba_by_model(self, df: pd.DataFrame) -> pd.DataFrame:
        """Per-base-model probabilities, useful for the dashboard / debugging
        disagreement between the boosted and bagged views."""
        base_probs = self._base_probs(df)
        stacked = self.stacker_.predict_proba(base_probs)[:, 1]
        return pd.DataFrame({
            "xgboost": base_probs[:, 0],
            "lightgbm": base_probs[:, 1],
            "random_forest": base_probs[:, 2],
            "isolation_forest_anomaly": base_probs[:, 3],
            "ensemble": stacked,
        })

    # ------------------------------------------------------------------
    # Explainability
    # ------------------------------------------------------------------
    def feature_importance(self, top_n: int = 25) -> pd.DataFrame:
        """
        Gain-based importance from each booster (comparable across
        features, unlike raw split counts) plus RF's impurity-based
        importance, min-max normalized per model then averaged -- gives
        investigators a single ranked driver list instead of three
        disagreeing tables.
        """
        xgb_imp = pd.Series(
            self.xgb_model_.get_booster().get_score(importance_type="gain"),
        )
        # XGBoost keys importances by internal feature names (f0, f1, ...)
        # when a plain ndarray is used, but by real column names when a
        # DataFrame with categorical dtype is used (our case) -- reindex
        # defensively either way.
        xgb_imp = xgb_imp.reindex(ALL_FEATURE_COLUMNS).fillna(0.0)

        lgb_imp = pd.Series(
            self.lgb_model_.booster_.feature_importance(importance_type="gain"),
            index=self.lgb_model_.booster_.feature_name(),
        ).reindex(ALL_FEATURE_COLUMNS).fillna(0.0)

        rf_imp = pd.Series(
            self.rf_model_.feature_importances_,
            index=ALL_FEATURE_COLUMNS,
        )

        def _norm(s: pd.Series) -> pd.Series:
            rng = s.max() - s.min()
            return (s - s.min()) / rng if rng > 0 else s * 0.0

        combined = pd.DataFrame({
            "xgboost": _norm(xgb_imp),
            "lightgbm": _norm(lgb_imp),
            "random_forest": _norm(rf_imp),
        })
        combined["average"] = combined.mean(axis=1)
        combined = combined.sort_values("average", ascending=False)
        combined.index.name = "feature"
        return combined.head(top_n).reset_index()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "xgb_model": self.xgb_model_,
                "lgb_model": self.lgb_model_,
                "rf_model": self.rf_model_,
                "iso_model": self.iso_model_,
                "iso_score_min": self.iso_score_min_,
                "iso_score_max": self.iso_score_max_,
                "stacker": self.stacker_,
                "xgb_params": self.xgb_params,
                "lgb_params": self.lgb_params,
                "rf_params": self.rf_params,
                "iso_params": self.iso_params,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "AMLEnsemble":
        blob = joblib.load(path)
        ens = cls(
            xgb_params=blob["xgb_params"],
            lgb_params=blob["lgb_params"],
            rf_params=blob["rf_params"],
            iso_params=blob["iso_params"],
        )
        ens.xgb_model_ = blob["xgb_model"]
        ens.lgb_model_ = blob["lgb_model"]
        ens.rf_model_ = blob["rf_model"]
        ens.iso_model_ = blob["iso_model"]
        ens.iso_score_min_ = blob["iso_score_min"]
        ens.iso_score_max_ = blob["iso_score_max"]
        ens.stacker_ = blob["stacker"]
        return ens


if __name__ == "__main__":
    # Lightweight self-test with synthetic data -- doesn't require the
    # full data_processing/graph_construction/feature_engineering chain,
    # just something shaped like their output.
    rng = np.random.default_rng(0)
    n = 20_000
    df = pd.DataFrame({c: rng.normal(size=n) for c in NUMERIC_FEATURE_COLUMNS})
    df["from_bank_idx"] = rng.integers(0, 50, size=n)
    df["to_bank_idx"] = rng.integers(0, 50, size=n)
    df["payment_format_idx"] = rng.integers(0, 6, size=n)
    df["payment_currency_idx"] = rng.integers(0, 10, size=n)
    df["receiving_currency_idx"] = rng.integers(0, 10, size=n)
    signal = df["relay_timing_score"] + df["short_cycle_participation"] * 2 - df["sender_entropy"]
    prob = 1 / (1 + np.exp(-(signal - signal.mean())))
    df["label"] = (rng.random(n) < (prob * 0.05)).astype(int)

    split = int(n * 0.8)
    df_train, df_val = df.iloc[:split], df.iloc[split:]

    ens = AMLEnsemble()
    ens, val_probs = ens.fit(df_train, df_val)
    print("val probs range:", val_probs.min(), val_probs.max())
    print(ens.feature_importance(10))

    ens.save("/tmp/ensemble_smoketest.joblib")
    reloaded = AMLEnsemble.load("/tmp/ensemble_smoketest.joblib")
    reloaded_probs = reloaded.predict_proba(df_val)
    print("max abs diff after reload:", np.abs(val_probs - reloaded_probs).max())
    print("OK")

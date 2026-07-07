"""
model.py
--------
Only the model definition. XGBoost only, to start (per the paper's own
`ibm/training.ipynb` and Section 1's "sheer simplicity" as a stated
design goal, not a corner cut -- Table 4's ablation shows the lift comes
from the features, not from stacking learners on top of them). LightGBM /
CatBoost / an ensemble are a straightforward later addition if a genuine
apples-to-apples comparison shows they're worth the extra complexity --
see README.md's ablation instructions.

`num_parallel_tree=10` makes each boosting round grow 10 trees instead of
1 -- a "boosted random forest" hybrid -- which is where a good chunk of
variance reduction comes from without a separate Random Forest.

Expects categorical columns already cast to pandas `category` dtype
(feature_merge.prepare_feature_frame does this) -- native
`enable_categorical=True`, no one-hot, no separate encoder to version.
"""

import json
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb

from . import config


class Model:
    """Thin, save/load-able wrapper around one xgboost.Booster."""

    def __init__(
        self,
        params: Optional[dict] = None,
        num_boost_round: int = config.XGB_NUM_BOOST_ROUND,
        early_stopping_rounds: int = config.XGB_EARLY_STOPPING_ROUNDS,
        scale_pos_weight: Optional[float] = None,
    ):
        self.params = dict(config.XGB_PARAMS)
        if params:
            self.params.update(params)
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.scale_pos_weight = scale_pos_weight
        self.booster_: Optional[xgb.Booster] = None
        self.best_iteration_: Optional[int] = None
        self.feature_names_: Optional[list] = None

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray, X_val: pd.DataFrame, y_val: np.ndarray) -> "Model":
        params = dict(self.params)
        if self.scale_pos_weight is not None:
            params["scale_pos_weight"] = self.scale_pos_weight
        elif "scale_pos_weight" not in params:
            n_pos, n_neg = int(np.sum(y_train == 1)), int(np.sum(y_train == 0))
            params["scale_pos_weight"] = max(n_neg / max(n_pos, 1), 1.0)

        self.feature_names_ = list(X_train.columns)
        dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
        dval = xgb.DMatrix(X_val, label=y_val, enable_categorical=True)

        self.booster_ = xgb.train(
            params, dtrain, num_boost_round=self.num_boost_round,
            evals=[(dval, "validation")], early_stopping_rounds=self.early_stopping_rounds,
            verbose_eval=False,
        )
        self.best_iteration_ = self.booster_.best_iteration
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.booster_ is None:
            raise RuntimeError("Call .fit() (or .load()) before .predict_proba().")
        X = X.loc[:, self.feature_names_] if self.feature_names_ else X
        dmatrix = xgb.DMatrix(X, enable_categorical=True)
        iteration_range = (0, self.best_iteration_ + 1) if self.best_iteration_ is not None else (0, 0)
        return self.booster_.predict(dmatrix, iteration_range=iteration_range)

    def feature_importance(self, top_n: Optional[int] = 25, importance_type: str = "gain") -> pd.DataFrame:
        """Gain-based, normalized 0-1, sorted descending -- check this
        after training to see how much any single feature (e.g. a raw
        payment-format code) dominates. See README.md."""
        if self.booster_ is None:
            raise RuntimeError("Call .fit() (or .load()) before .feature_importance().")
        raw = self.booster_.get_score(importance_type=importance_type)
        importance = pd.Series(raw, name="importance").sort_values(ascending=False)
        if importance.sum() > 0:
            importance = importance / importance.sum()
        out = importance.reset_index().rename(columns={"index": "feature"})
        return out.head(top_n) if top_n else out

    def save(self, path: str):
        if self.booster_ is None:
            raise RuntimeError("Nothing to save -- call .fit() first.")
        self.booster_.save_model(path)
        with open(f"{path}.meta.json", "w") as fl:
            json.dump({"best_iteration": self.best_iteration_, "feature_names": self.feature_names_,
                       "params": self.params}, fl)

    @classmethod
    def load(cls, path: str) -> "Model":
        instance = cls()
        instance.booster_ = xgb.Booster()
        instance.booster_.load_model(path)
        try:
            with open(f"{path}.meta.json") as fl:
                meta = json.load(fl)
            instance.best_iteration_ = meta.get("best_iteration")
            instance.feature_names_ = meta.get("feature_names")
            instance.params = meta.get("params", instance.params)
        except FileNotFoundError:
            instance.best_iteration_ = getattr(instance.booster_, "best_iteration", None)
        return instance

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier  # type: ignore
    HAS_XGB = True
except Exception:
    HAS_XGB = False
    XGBClassifier = None  # type: ignore

try:
    import shap 
    HAS_SHAP = True
except Exception: 
    HAS_SHAP = False


@dataclass
class ModelArtifacts:
    supervised_model: object | None
    anomaly_model: object
    feature_columns: list[str]
    threshold_high: float
    threshold_medium: float
    metadata: dict


def _build_supervised_model(
    scale_pos_weight: float = 1.0, 
    random_state: int = 42
):
    if HAS_XGB:
        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=500,
            max_depth=6,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=2.0,
            tree_method="hist",
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            n_jobs=4,
        )
    return HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=6,
        max_iter=300,
        random_state=random_state,
    )


def train_models(
    feature_df: pd.DataFrame,
    labels_df: pd.DataFrame | None,
    random_state: int = 42,
):
    feature_df = feature_df.copy()
    feature_columns = [c for c in feature_df.columns if c != "account"]
    X = feature_df[feature_columns]

    supervised_model = None
    positive_count = 0
    negative_count = 0
    scale_pos_weight = 1.0

    if labels_df is not None and len(labels_df):
        merged = feature_df.merge(labels_df, on="account", how="inner")
        if "label" in merged.columns and merged["label"].nunique() >= 2 and len(merged) >= 10:
            y = merged["label"].astype(int)
            X_train = merged[feature_columns]
            
            positive_count = int(y.sum())
            
            negative_count = len(y) - positive_count 
            
            scale_pos_weight = max(
                negative_count / max(positive_count, 1),
                1.0,
            )
            supervised_model = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", _build_supervised_model(scale_pos_weight=scale_pos_weight, random_state=random_state,)),
                ]
            )
            supervised_model.fit(X_train, y)

    anomaly_model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", IsolationForest(
                n_estimators=300,
                contamination="auto",
                random_state=random_state,
                n_jobs=4,
            )),
        ]
    )
    anomaly_model.fit(X)

    if supervised_model is not None:
        sup_prob_train = _supervised_prob(supervised_model, X,)
    else:
        sup_prob_train = np.zeros(len(X), dtype=float,)
    
    X_imp = anomaly_model.named_steps["imputer"].transform(X)

    X_scaled = anomaly_model.named_steps["scaler"].transform(X_imp)

    iso = anomaly_model.named_steps["model"]

    anomaly_raw = -iso.score_samples(X_scaled)

    if (np.nanmax(anomaly_raw) - np.nanmin(anomaly_raw)) > 1e-9:
        anomaly_score_train = (anomaly_raw - np.nanmin(anomaly_raw)) / (np.nanmax(anomaly_raw) - np.nanmin(anomaly_raw))
    else:
        anomaly_score_train = np.zeros(len(X), dtype=float,)
    
    training_risk_scores = (0.80 * sup_prob_train + 0.20 * anomaly_score_train)

    threshold_medium = float(np.quantile(training_risk_scores, 0.95))

    threshold_high = float(np.quantile(training_risk_scores, 0.99))

    return ModelArtifacts(
        supervised_model=supervised_model,
        anomaly_model=anomaly_model,
        feature_columns=feature_columns,
        threshold_high=threshold_high,
        threshold_medium=threshold_medium,
        metadata = {
            "model_type": "XGBoost" if HAS_XGB else "HistGradientBoosting",
            "training_samples": int(len(feature_df)),
            "feature_count":int(len(feature_columns)),
            "positive_labels": int(positive_count) if labels_df is not None else 0,
            "negative_labels": int(negative_count) if labels_df is not None else 0,
            "scale_pos_weight": float(scale_pos_weight) if labels_df is not None else 1.0,
        },
    )


def _supervised_prob(model, X: pd.DataFrame) -> np.ndarray:
    if model is None:
        return np.zeros(len(X), dtype=float)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1].astype(float)
    preds = model.predict(X)
    return preds.astype(float)


def score_accounts(feature_df: pd.DataFrame, artifacts: ModelArtifacts) -> pd.DataFrame:
    X = feature_df[artifacts.feature_columns].copy()

    if artifacts.supervised_model is not None:
        sup_prob = _supervised_prob(artifacts.supervised_model, X)
    else:
        sup_prob = np.zeros(len(X), dtype=float)

    X_imp = artifacts.anomaly_model.named_steps["imputer"].transform(X)
    X_scaled = artifacts.anomaly_model.named_steps["scaler"].transform(X_imp)
    iso = artifacts.anomaly_model.named_steps["model"]
    anomaly_raw = -iso.score_samples(X_scaled)  # higher = more anomalous
    if np.nanmax(anomaly_raw) - np.nanmin(anomaly_raw) > 1e-9:
        anomaly_score = (anomaly_raw - np.nanmin(anomaly_raw)) / (np.nanmax(anomaly_raw) - np.nanmin(anomaly_raw))
    else:
        anomaly_score = np.zeros(len(X), dtype=float)

    risk_score = 0.80 * sup_prob + 0.20 * anomaly_score
    out = feature_df[["account"]].copy()
    out["supervised_probability"] = sup_prob
    out["anomaly_score"] = anomaly_score
    out["risk_score"] = risk_score
    out["risk_tier"] = pd.cut(
        out["risk_score"],
        bins=[-1, artifacts.threshold_medium, artifacts.threshold_high, 2],
        labels=["Low", "Medium", "High"],
    ).astype(str)
    out["risk_tier"] = out["risk_tier"].replace({"nan": "Low"})
    return out


def evaluate_if_labels_available(scored_accounts: pd.DataFrame, labels_df: pd.DataFrame | None) -> dict:
    if labels_df is None or len(labels_df) == 0 or "label" not in labels_df.columns:
        return {}
    merged = scored_accounts.merge(labels_df, on="account", how="inner")
    if merged["label"].nunique() < 2:
        return {}

    y_true = merged["label"].astype(int)
    y_prob = merged["risk_score"].astype(float)
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
    }
    cm = confusion_matrix(y_true, y_pred).tolist()
    metrics["confusion_matrix"] = cm
    return metrics


def save_artifacts(artifacts: ModelArtifacts, path: str):
    with open(path, "wb") as f:
        pickle.dump(artifacts, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_artifacts(source) -> ModelArtifacts:
    if hasattr(source, "read"):
        return pickle.load(source)
    if isinstance(source, (bytes, bytearray)):
        return pickle.loads(source)
    with open(source, "rb") as f:
        return pickle.load(f)

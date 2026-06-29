"""Modeling harness — the honest training/evaluation loop (CLAUDE.md rules 1-6).

One function does the whole rigorous pipeline for a given feature set:

    time_split (train=earlier, test=later; never shuffle)
        -> class-weighted XGBoost (ONE imbalance mechanism: scale_pos_weight; no SMOTE)
        -> early-stop on validation PR-AUC (model selection by PR-AUC, not F1@0.5)
        -> freeze the decision threshold tuned on validation
        -> evaluate on the future test period
        -> persist model + threshold + feature list

The ablation calls this twice (baseline cols, then baseline+topology cols) with an
identical split and config, so the only thing that differs is the feature set.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from modeling.metrics import evaluate, tune_threshold

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]


def time_split(ts: pd.Series, train_frac: float = 0.6, val_frac: float = 0.2,
               partition=None):
    """Chronological split into boolean masks (train, val, test). NO shuffling (rule 3).

    Default: a global day-aligned quantile split (earlier trains, later tests).

    ``partition`` (e.g. SAP's ``run_id``): split *within* each partition by time
    position — each run contributes its early rows to train and its late rows to
    test. Required when the global time axis is synthetic and not a real cross-run
    chronology (SAP's runs are ordered fraud-first/clean-middle, which would put a
    zero-fraud run in validation under a global split)."""
    ts = np.asarray(ts)
    n = len(ts)
    if partition is None:
        t1 = int(np.floor(np.quantile(ts, train_frac)))
        t2 = int(np.floor(np.quantile(ts, train_frac + val_frac)))
        if t2 <= t1:  # degenerate (coarse time axis) -> nudge the val boundary up
            t2 = t1 + 1
        return ts < t1, (ts >= t1) & (ts < t2), ts >= t2

    partition = np.asarray(partition)
    train = np.zeros(n, bool); val = np.zeros(n, bool); test = np.zeros(n, bool)
    for p in pd.unique(partition):
        idx = np.where(partition == p)[0]          # rows are globally time-sorted -> contiguous & ordered
        m = len(idx)
        c1, c2 = int(m * train_frac), int(m * (train_frac + val_frac))
        train[idx[:c1]] = True
        val[idx[c1:c2]] = True
        test[idx[c2:]] = True
    return train, val, test


def _fit_xgb(Xtr, ytr, Xval, yval, seed: int):
    """Class-weighted XGBoost; early-stop on validation PR-AUC."""
    pos = int(ytr.sum())
    neg = int(len(ytr) - pos)
    spw = (neg / pos) if pos else 1.0   # the single imbalance mechanism (class weight)
    model = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_lambda=1.0,
        scale_pos_weight=spw,
        eval_metric="aucpr",          # PR-AUC drives early stopping (rule 5)
        early_stopping_rounds=40,
        tree_method="hist",
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=False)
    return model, spw


def train_and_evaluate(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    label: str = "label",
    ts_col: str = "timestamp",
    seed: int = 42,
    persist_as: str | None = None,
    partition=None,
) -> dict:
    """Run the full honest pipeline for one feature set and return a result dict."""
    train, val, test = time_split(df[ts_col], partition=partition)
    X = df[feature_cols]
    y = df[label].astype(int).to_numpy()

    model, spw = _fit_xgb(X[train], y[train], X[val], y[val], seed)
    p_val = model.predict_proba(X[val])[:, 1]
    p_test = model.predict_proba(X[test])[:, 1]

    threshold = tune_threshold(y[val], p_val)            # tuned on val, then frozen
    test_metrics = evaluate(y[test], p_test, threshold)
    val_metrics = evaluate(y[val], p_val, threshold)

    importances = dict(sorted(
        zip(feature_cols, (float(v) for v in model.feature_importances_)),
        key=lambda kv: kv[1], reverse=True,
    ))
    result = {
        "n_features": len(feature_cols),
        "features": feature_cols,
        "split_sizes": {"train": int(train.sum()), "val": int(val.sum()), "test": int(test.sum())},
        "split_frauds": {"train": int(y[train].sum()), "val": int(y[val].sum()), "test": int(y[test].sum())},
        "scale_pos_weight": round(spw, 2),
        "best_iteration": int(getattr(model, "best_iteration", -1) or -1),
        "val": val_metrics,
        "test": test_metrics,
        "feature_importance": importances,
    }

    if persist_as:
        art_dir = REPO_ROOT / "artifacts" / persist_as
        art_dir.mkdir(parents=True, exist_ok=True)
        model.save_model(art_dir / "model.json")
        with open(art_dir / "bundle.json", "w", encoding="utf-8") as f:
            json.dump({"threshold": threshold, "features": feature_cols,
                       "scale_pos_weight": spw, "note": "trees need no scaler"}, f, indent=2)
        result["artifact_dir"] = str(art_dir)
    return result

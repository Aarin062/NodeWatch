"""Evaluation metrics for imbalanced fraud detection (CLAUDE.md rule 5).

PR-AUC (average precision) is PRIMARY and the model-selection metric. We also
report precision@k / recall@k — the operationally honest view, since an auditor
only reviews the top-N alerts — and ROC-AUC, which we report but never rely on
(it looks flatteringly high on imbalanced data).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def precision_recall_at_k(y_true: np.ndarray, scores: np.ndarray, ks) -> dict:
    """precision@k and recall@k for each k: rank by score desc, take the top k."""
    order = np.argsort(-scores, kind="stable")
    y_sorted = np.asarray(y_true)[order]
    total_pos = int(y_sorted.sum())
    out = {}
    csum = np.cumsum(y_sorted)
    for k in ks:
        kk = min(k, len(y_sorted))
        hits = int(csum[kk - 1]) if kk > 0 else 0
        out[f"precision@{k}"] = round(hits / kk, 4) if kk else 0.0
        out[f"recall@{k}"] = round(hits / total_pos, 4) if total_pos else 0.0
    return out


def tune_threshold(y_val: np.ndarray, p_val: np.ndarray) -> float:
    """Pick the decision threshold that maximises F1 on the VALIDATION set.

    This is the only place a threshold is chosen; it is then frozen for the test
    set (rule 2 — never report at a fixed 0.5)."""
    prec, rec, thr = precision_recall_curve(y_val, p_val)
    # prec/rec have len = len(thr)+1; align by dropping the last point.
    f1 = np.divide(2 * prec[:-1] * rec[:-1], prec[:-1] + rec[:-1],
                   out=np.zeros_like(prec[:-1]), where=(prec[:-1] + rec[:-1]) > 0)
    if len(thr) == 0:
        return 0.5
    return float(thr[int(np.argmax(f1))])


def evaluate(y_true: np.ndarray, scores: np.ndarray, threshold: float,
             ks=(50, 100, 500, 1000)) -> dict:
    """Full metric bundle on a held-out set at a FROZEN threshold."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = (scores >= threshold).astype(int)
    out = {
        "pr_auc": round(float(average_precision_score(y_true, scores)), 5),   # PRIMARY
        "roc_auc": round(float(roc_auc_score(y_true, scores)), 5),            # reported only
        "threshold": round(float(threshold), 6),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "n_pos": int(y_true.sum()),
        "n": int(len(y_true)),
    }
    out.update(precision_recall_at_k(y_true, scores, ks))
    return out

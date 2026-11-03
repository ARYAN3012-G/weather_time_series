"""
metrics.py — classification + early-warning metrics, severe-class threshold
tuning, and block bootstrap confidence intervals.

Early-warning terms for the Severe class (event = Severe):
  POD (probability of detection) = TP / (TP + FN)   (= severe recall)
  FAR (false alarm ratio)        = FP / (TP + FP)   (= 1 - severe precision)
  CSI (critical success index)   = TP / (TP + FN + FP)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_recall_fscore_support,
                             roc_auc_score)

CLASS_NAMES = ["healthy", "moderate", "severe"]
SEVERE = 2


def _safe(fn, *a, **k):
    try:
        return float(fn(*a, **k))
    except ValueError:
        return float("nan")


def event_scores(y_true, y_pred, event=SEVERE):
    t = y_true == event; p = y_pred == event
    tp = int((t & p).sum()); fn = int((t & ~p).sum()); fp = int((~t & p).sum())
    pod = tp / (tp + fn) if tp + fn else float("nan")
    far = fp / (tp + fp) if tp + fp else float("nan")
    csi = tp / (tp + fn + fp) if tp + fn + fp else float("nan")
    return {"severe_pod": pod, "severe_far": far, "severe_csi": csi}


def compute_metrics(y_true, prob, y_pred=None, k: int = 3) -> dict:
    y_true = np.asarray(y_true)
    prob = np.asarray(prob, dtype=float)
    if y_pred is None:
        y_pred = prob.argmax(1)
    labels = list(range(k))
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    yb = np.eye(k)[y_true]
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_precision": float(prec.mean()),
        "macro_recall": float(rec.mean()),
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "roc_auc_macro_ovr": _safe(roc_auc_score, yb, prob, average="macro") if k > 2
        else _safe(roc_auc_score, y_true, prob[:, 1]),
        "pr_auc_macro": float(np.nanmean([_safe(average_precision_score, yb[:, c], prob[:, c]) for c in labels])),
        "brier": float(np.mean(np.sum((prob - yb) ** 2, axis=1))),
    }
    names = CLASS_NAMES if k == 3 else ["no_drought", "drought"]
    for c in labels:
        out[f"{names[c]}_precision"] = float(prec[c])
        out[f"{names[c]}_recall"] = float(rec[c])
        out[f"{names[c]}_f1"] = float(f1[c])
    if k == 3:
        out["severe_roc_auc"] = _safe(roc_auc_score, yb[:, SEVERE], prob[:, SEVERE])
        out["severe_pr_auc"] = _safe(average_precision_score, yb[:, SEVERE], prob[:, SEVERE])
        out.update(event_scores(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    for i in labels:
        for j in labels:
            out[f"cm_{i}{j}"] = int(cm[i, j])
    out["n"] = int(len(y_true))
    return {k_: (round(v, 4) if isinstance(v, float) else v) for k_, v in out.items()}


# ---------------------------------------------------------------------------
# Severe-class threshold (tuned on VALIDATION only)
# ---------------------------------------------------------------------------
def predict_with_severe_threshold(prob, tau):
    pred = prob[:, :SEVERE].argmax(1)
    pred[prob[:, SEVERE] >= tau] = SEVERE
    return pred


def tune_severe_threshold(y_val, p_val, grid=None) -> tuple[float, float]:
    """Return (tau, val_severe_f1) maximising Severe F1 on validation. Ties -> tau closest to argmax rule."""
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2) if grid is None else grid
    best_tau, best = None, -1.0
    for tau in grid:
        f = f1_score(y_val, predict_with_severe_threshold(p_val, tau), labels=[SEVERE],
                     average="macro", zero_division=0)
        if f > best + 1e-9:
            best_tau, best = float(tau), f
    return best_tau, round(float(best), 4)


# ---------------------------------------------------------------------------
# Block bootstrap
# ---------------------------------------------------------------------------
def _fast_f1(y, p, k=3):
    cm = np.bincount(y * k + p, minlength=k * k).reshape(k, k)
    tp = np.diag(cm); fp = cm.sum(0) - tp; fn = cm.sum(1) - tp
    denom = 2 * tp + fp + fn
    f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)
    return f1.mean(), f1[-1]


def block_bootstrap(y, pred, blocks, n_boot=1000, seed=12345, pred_ref=None, k=3) -> dict:
    """
    Resample whole blocks (e.g. target year-month, all sites together) with replacement.
    Returns 95% CIs for macro-F1 and severe-F1; if pred_ref is given, also the CI of
    the paired difference (model - reference) on the same resamples.
    """
    y = np.asarray(y); pred = np.asarray(pred)
    codes, uniq = pd.factorize(pd.Series(blocks))
    order = np.argsort(codes, kind="stable")
    bounds = np.searchsorted(codes[order], np.arange(len(uniq) + 1))
    groups = [order[bounds[i]:bounds[i + 1]] for i in range(len(uniq))]
    rng = np.random.default_rng(seed)
    stats = {"macro_f1": [], "severe_f1": [], "d_macro_f1": [], "d_severe_f1": []}
    for _ in range(n_boot):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        m, s = _fast_f1(y[idx], pred[idx], k)
        stats["macro_f1"].append(m); stats["severe_f1"].append(s)
        if pred_ref is not None:
            mr, sr = _fast_f1(y[idx], np.asarray(pred_ref)[idx], k)
            stats["d_macro_f1"].append(m - mr); stats["d_severe_f1"].append(s - sr)
    out = {}
    for key, v in stats.items():
        if v:
            out[f"{key}_ci_low"] = round(float(np.percentile(v, 2.5)), 4)
            out[f"{key}_ci_high"] = round(float(np.percentile(v, 97.5)), 4)
    out["n_blocks"] = len(groups)
    return out

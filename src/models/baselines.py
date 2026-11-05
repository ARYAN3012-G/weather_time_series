"""
baselines.py — reference forecasts every model must beat.

All baselines are fitted on TRAIN samples only and return class probabilities.

  persistence : class on the last input day t is the forecast for t+h (one-hot)
  markov      : P(class at t+h | class at t), transition matrix from TRAIN
                (probabilistic persistence; gives meaningful ROC/PR-AUC)
  climatology : P(class | calendar day of target), from TRAIN labels,
                smoothed with a +/-15-day circular window
  majority    : constant TRAIN class frequencies
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-6


def _one_hot(labels, k):
    p = np.full((len(labels), k), EPS)
    p[np.arange(len(labels)), labels] = 1.0
    return p / p.sum(1, keepdims=True)


def persistence(meta_te: pd.DataFrame, k: int = 3) -> np.ndarray:
    return _one_hot(meta_te["label_t"].to_numpy(int), k)


def markov(meta_tr, y_tr, meta_te, k: int = 3) -> np.ndarray:
    T = np.ones((k, k))                                       # Laplace smoothing
    np.add.at(T, (meta_tr["label_t"].to_numpy(int), y_tr), 1)
    T = T / T.sum(1, keepdims=True)
    return T[meta_te["label_t"].to_numpy(int)]


def climatology(meta_tr, y_tr, meta_te, k: int = 3, half_window: int = 15) -> np.ndarray:
    doy_tr = np.minimum(pd.to_datetime(meta_tr["target_date"]).dt.day_of_year.to_numpy(), 365) - 1
    counts = np.ones((365, k))
    np.add.at(counts, (doy_tr, y_tr), 1)
    smooth = np.zeros_like(counts)
    for s in range(-half_window, half_window + 1):
        smooth += np.roll(counts, s, axis=0)
    probs = smooth / smooth.sum(1, keepdims=True)
    doy_te = np.minimum(pd.to_datetime(meta_te["target_date"]).dt.day_of_year.to_numpy(), 365) - 1
    return probs[doy_te]


def majority(y_tr, n_te: int, k: int = 3) -> np.ndarray:
    freq = np.bincount(y_tr, minlength=k) / len(y_tr)
    return np.tile(freq, (n_te, 1))


def all_baselines(meta_tr, y_tr, meta_val, meta_te, k: int = 3) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """{name: (p_val, p_test)}"""
    return {
        "persistence": (persistence(meta_val, k), persistence(meta_te, k)),
        "markov":      (markov(meta_tr, y_tr, meta_val, k), markov(meta_tr, y_tr, meta_te, k)),
        "climatology": (climatology(meta_tr, y_tr, meta_val, k), climatology(meta_tr, y_tr, meta_te, k)),
        "majority":    (majority(y_tr, len(meta_val), k), majority(y_tr, len(meta_te), k)),
    }

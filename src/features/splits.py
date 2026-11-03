"""
splits.py — purged chronological splits and spatial hold-out folds.

Temporal periods (by year):  TRAIN <= 2011 | VAL 2012–2018 | TEST >= 2019

Purging: a sample spans [input_start, target_date]. It is assigned to a period
only if BOTH ends fall in that period; samples straddling a boundary are dropped.
This creates a gap of (seq_len + horizon - 1) days at each boundary, so no
day is used by samples of two different periods.

Spatial modes:
  temporal : all sites in every period (pooled, site-agnostic model)
  loco     : leave-one-cluster-out — train/val on one cluster, test on the other
  loso     : leave-one-site-out — train/val on 6 sites, test on the held-out site
In spatial modes the held-out sites are evaluated on TEST years only, so the
test set is unseen in both space and time.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


def period_of(dates: pd.Series) -> np.ndarray:
    y = pd.to_datetime(dates).dt.year.to_numpy()
    return np.where(y <= config.TRAIN_END_YEAR, "train",
                    np.where(y <= config.VAL_END_YEAR, "val", "test"))


def purged_periods(meta: pd.DataFrame) -> np.ndarray:
    """Period label per sample, or 'purged' if its span crosses a boundary."""
    p_start = period_of(meta["input_start"])
    p_end = period_of(meta["target_date"])
    return np.where(p_start == p_end, p_end, "purged")


def spatial_folds(mode: str) -> list[tuple[str, list[str], list[str]]]:
    """[(fold_name, train_fields, test_fields)]"""
    if mode == "temporal":
        return [("all_sites", list(config.FIELDS), list(config.FIELDS))]
    if mode == "loco":
        return [(f"test_{c}", [f for f in config.FIELDS if f not in fs], list(fs))
                for c, fs in config.CLUSTERS.items()]
    if mode == "loso":
        return [(f"test_{s}", [f for f in config.FIELDS if f != s], [s]) for s in config.FIELDS]
    raise ValueError(f"unknown split mode {mode}")


def make_masks(meta: pd.DataFrame, train_fields: list[str], test_fields: list[str]) -> dict[str, np.ndarray]:
    per = purged_periods(meta)
    in_train_sites = meta["field"].isin(train_fields).to_numpy()
    in_test_sites = meta["field"].isin(test_fields).to_numpy()
    masks = {
        "train": (per == "train") & in_train_sites,
        "val":   (per == "val") & in_train_sites,
        "test":  (per == "test") & in_test_sites,
    }
    if set(train_fields) != set(test_fields):
        assert not (set(train_fields) & set(test_fields)), "spatial folds must be disjoint"
    return masks


def describe_masks(meta: pd.DataFrame, y: np.ndarray, masks: dict) -> pd.DataFrame:
    rows = []
    for k, m in masks.items():
        c = np.bincount(y[m], minlength=3)
        rows.append({"split": k, "n": int(m.sum()),
                     "first_input": meta.loc[m, "input_start"].min(),
                     "last_target": meta.loc[m, "target_date"].max(),
                     **{config.INVERSE_LABEL_MAP[i]: f"{c[i]} ({100 * c[i] / max(m.sum(), 1):.1f}%)"
                        for i in range(3)}})
    rows.append({"split": "purged", "n": int((purged_periods(meta) == "purged").sum())})
    return pd.DataFrame(rows)

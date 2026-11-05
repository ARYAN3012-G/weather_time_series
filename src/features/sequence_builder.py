"""
sequence_builder.py — sliding-window samples for genuine future forecasting.

For a sample with last input day t:
    input  X : feature rows for days [t - seq_len + 1, ..., t]
    target y : drought class on day t + h          (h = forecast horizon, >= 1)

Guarantees (asserted here and in tests/test_pipeline.py):
  * every input day is <= t < target day;
  * windows never cross a gap in the daily record;
  * historical SPEI (feature set 'weather_spei' / 'spei_only') is taken only
    from input days, never from the target day.

meta columns: field, cluster, input_start, last_input_date, target_date, horizon,
              label_t (class on day t, used by the persistence baseline), spei_t
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from utils.logger import get_logger

logger = get_logger(__name__)


def load_prepared(fields: list[str] | None = None) -> dict[str, pd.DataFrame]:
    fields = fields or config.FIELDS
    frames = {}
    for f in fields:
        path = config.PREPARED_DIR / f"{f}.csv"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing — run `python src/dataprep/prepare_data.py` first")
        frames[f] = pd.read_csv(path, parse_dates=[config.DATE_COLUMN])
    return frames


def add_doy_features(df: pd.DataFrame) -> pd.DataFrame:
    """sin/cos of day-of-year from each row's own date. A window therefore only
    carries the dates of its input days, never the target date."""
    df = df.copy()
    angle = 2 * np.pi * (df[config.DATE_COLUMN].dt.day_of_year - 1) / 365.25
    df["DOY_sin"] = np.sin(angle)
    df["DOY_cos"] = np.cos(angle)
    return df


def build_sequences_for_field(df: pd.DataFrame, feature_cols: list[str],
                              sequence_length: int, horizon: int):
    if horizon < 1:
        raise ValueError("horizon must be >= 1 (genuine future forecasting)")
    df = df.sort_values(config.DATE_COLUMN).reset_index(drop=True)
    if any(c in config.DOY_FEATURES for c in feature_cols):
        df = add_doy_features(df)
    dates = df[config.DATE_COLUMN].to_numpy("datetime64[D]")
    feats = df[feature_cols].to_numpy(np.float32)
    labels = df["label"].to_numpy(np.int64)
    spei = df["SPEI"].to_numpy(np.float32)

    n = len(df)
    n_seq = n - sequence_length - horizon + 1
    if n_seq <= 0:
        return np.empty((0, sequence_length, len(feature_cols)), np.float32), np.empty(0, np.int64), pd.DataFrame()

    first = np.arange(n_seq)
    last = first + sequence_length - 1
    tgt = last + horizon

    # keep only windows on a contiguous daily record and with finite inputs
    span_days = (dates[tgt] - dates[first]).astype(int)
    ok = span_days == (sequence_length - 1 + horizon)
    row_finite = np.isfinite(feats).all(axis=1)
    win_finite = sliding_window_view(row_finite, sequence_length)[:n_seq].all(axis=1)
    ok &= win_finite
    first, last, tgt = first[ok], last[ok], tgt[ok]

    windows = sliding_window_view(feats, sequence_length, axis=0)       # (n-L+1, F, L)
    X = np.ascontiguousarray(windows[first].transpose(0, 2, 1))          # (n, L, F)
    y = labels[tgt]

    meta = pd.DataFrame({
        "field": df["field"].iloc[0],
        "cluster": config.FIELD_TO_CLUSTER[df["field"].iloc[0]],
        "input_start": pd.to_datetime(dates[first]),
        "last_input_date": pd.to_datetime(dates[last]),
        "target_date": pd.to_datetime(dates[tgt]),
        "horizon": horizon,
        "label_t": labels[last],
        "spei_t": spei[last],
        # target-day SPEI: used ONLY to build the binary Marusov-protocol label, never as input
        "spei_target": spei[tgt],
    })
    assert ((meta.target_date - meta.last_input_date).dt.days == horizon).all()
    assert (meta.last_input_date < meta.target_date).all()
    return X, y, meta


def build_sequences_all_fields(frames: dict[str, pd.DataFrame], feature_set: str,
                               horizon: int, sequence_length: int = config.SEQUENCE_LENGTH_DAYS):
    feature_cols = config.FEATURE_SETS[feature_set]
    Xs, ys, metas = [], [], []
    for name in sorted(frames):
        X, y, meta = build_sequences_for_field(frames[name], feature_cols, sequence_length, horizon)
        if len(y):
            Xs.append(X); ys.append(y); metas.append(meta)
    if not Xs:
        raise RuntimeError("No sequences built")
    X = np.concatenate(Xs); y = np.concatenate(ys)
    meta = pd.concat(metas, ignore_index=True)
    logger.info(f"Sequences [{feature_set}, h={horizon}]: X={X.shape} from {meta.field.nunique()} sites")
    return X, y, meta

"""
sequence_builder.py

Builds sliding-window sequences for TWO tasks:
  Task A (current):  label at t+1   (next day)
  Task B (forecast): label at t+15  (15 days ahead)

Input:  per-field processed CSV with columns:
        date, 9 weather features, Water_Balance, WB30, SPEI, label, field
Output: X (n_seq, seq_len, 9), y (n_seq,), meta DataFrame
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from utils.logger import get_logger

logger = get_logger(__name__)


def build_sequences_for_field(
    field_df: pd.DataFrame,
    sequence_length: int = config.SEQUENCE_LENGTH_DAYS,
    forecast_horizon: int = 1,
) -> tuple:
    """
    Build sequences for ONE field.
    Returns (X, y, meta):
      X    : (n_seq, seq_len, n_features) float32
      y    : (n_seq,) int64
      meta : DataFrame with 'field', 'target_date', 'horizon'

    A sequence i uses rows [i .. i+seq_len-1] as input
    and row i+seq_len-1+forecast_horizon as the label target.
    No leakage: target day's weather is never in the input window.
    """
    assert not set(config.MODEL_INPUT_FEATURES) & set(config.LEAKAGE_COLUMNS)

    date_col = config.DATE_COLUMN
    field_df = field_df.sort_values(date_col).reset_index(drop=True)

    field_name = field_df['field'].iloc[0] if 'field' in field_df.columns else 'unknown'

    feature_mat = field_df[config.MODEL_INPUT_FEATURES].values.astype(np.float32)
    labels = field_df['label'].values.astype(np.int64)
    dates = field_df[date_col].values

    n_rows = len(field_df)
    # Need seq_len input days + forecast_horizon days to reach target
    n_seq = n_rows - sequence_length - forecast_horizon + 1

    if n_seq <= 0:
        logger.warning(f'[{field_name}] Not enough rows ({n_rows}) for seq_len={sequence_length} + horizon={forecast_horizon}. Skipping.')
        return (
            np.empty((0, sequence_length, len(config.MODEL_INPUT_FEATURES)), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            pd.DataFrame(columns=['field', 'target_date', 'horizon'])
        )

    X = np.empty((n_seq, sequence_length, len(config.MODEL_INPUT_FEATURES)), dtype=np.float32)
    y = np.empty((n_seq,), dtype=np.int64)
    target_dates = []

    for i in range(n_seq):
        X[i] = feature_mat[i: i + sequence_length]
        target_idx = i + sequence_length - 1 + forecast_horizon
        y[i] = labels[target_idx]
        target_dates.append(dates[target_idx])

    meta = pd.DataFrame({
        'field': field_name,
        'target_date': target_dates,
        'horizon': forecast_horizon,
    })

    logger.info(f'[{field_name}] horizon={forecast_horizon:2d}d -> {n_seq} sequences ({sequence_length}-day windows).')
    return X, y, meta


def build_sequences_all_fields(
    labeled_frames: dict,
    sequence_length: int = config.SEQUENCE_LENGTH_DAYS,
    forecast_horizon: int = 1,
) -> tuple:
    """Build and pool sequences for all fields."""
    X_list, y_list, meta_list = [], [], []
    for field_name, df in labeled_frames.items():
        X, y, meta = build_sequences_for_field(df, sequence_length, forecast_horizon)
        if len(y) > 0:
            X_list.append(X)
            y_list.append(y)
            meta_list.append(meta)

    if not X_list:
        raise RuntimeError('No sequences built for any field.')

    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    meta_all = pd.concat(meta_list, ignore_index=True)

    logger.info(f'Pooled: X={X_all.shape}, y={y_all.shape}, fields={meta_all["field"].nunique()}')
    return X_all, y_all, meta_all

"""
train.py - Trains ALL models for BOTH drought prediction tasks.

  Task A: Next-Day Drought Forecast   (horizon =  1 day)
  Task B: 15-Day-Ahead Forecast       (horizon = 15 days)

Models: Random Forest, XGBoost, LSTM, Transformer, Ensemble

Training approach:
  - POOLED SITE-AGNOSTIC MODEL: One model trained on data from all 8 sites.
  - Site identity is NOT an input feature — the model learns generalizable
    weather-drought relationships across all sites.
  - Optional future experiment: Leave-One-Site-Out spatial generalization.

Splitting:
  - Strict chronological by target_date year:
    TRAIN: 1979–2011 | VALIDATION: 2012–2018 | TEST: 2019–2025
  - StandardScaler fitted on TRAIN only, applied to VAL and TEST.

Ensemble:
  - Weights proportional to each model's VALIDATION Macro-F1.
  - Weights are frozen before any test-set evaluation.

Model selection:
  - Best model selected by VALIDATION Macro-F1 (not test).

Usage:
    python src/training/train.py

Results saved to:
    results/taskA_nextday_forecast.csv
    results/taskB_15day_forecast.csv
    results/summary.json
    results/confusion_matrix_taskA.csv
    results/confusion_matrix_taskB.csv
    models/  (saved model files)
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    confusion_matrix, f1_score, roc_auc_score,
)
from sklearn.preprocessing import label_binarize, StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import config
from features.sequence_builder import build_sequences_all_fields
from utils.logger import get_logger

logger = get_logger(__name__)

torch.manual_seed(config.RANDOM_SEED)
np.random.seed(config.RANDOM_SEED)

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_GPU_XGB = torch.cuda.is_available()
logger.info(f"Device: {DEVICE}  |  XGBoost GPU: {USE_GPU_XGB}")


# =============================================================================
# Neural Network Models
# =============================================================================

class DroughtLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout):
        super().__init__()
        self.lstm    = nn.LSTM(input_size, hidden_size, num_layers,
                               batch_first=True,
                               dropout=dropout if num_layers > 1 else 0.0)
        self.norm    = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (batch, seq_len=30, features=9)
        _, (hn, _) = self.lstm(x)
        return self.fc(self.dropout(self.norm(hn[-1])))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe           = torch.zeros(max_len, d_model)
        position     = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term     = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class DroughtTransformer(nn.Module):
    """
    Architecture: 30×9 → Linear(9, d_model) → PosEnc → TransformerEncoder
                  → MeanPool → LayerNorm → Linear(d_model, 3)
    """
    def __init__(self, input_size, num_classes,
                 d_model=config.TF_D_MODEL,
                 nhead=config.TF_NHEAD,
                 num_layers=config.TF_NUM_LAYERS,
                 dropout=config.TF_DROPOUT,
                 dim_feedforward=config.TF_DIM_FEEDFORWARD):
        super().__init__()
        self.input_proj  = nn.Linear(input_size, d_model)
        self.pos_enc     = PositionalEncoding(d_model, dropout=dropout)
        enc_layer        = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm        = nn.LayerNorm(d_model)
        self.fc          = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: (batch, seq_len=30, features=9)
        x = self.input_proj(x)       # -> (batch, 30, d_model)
        x = self.pos_enc(x)
        x = self.transformer(x)      # -> (batch, 30, d_model)
        x = x.mean(dim=1)            # Mean pooling over sequence dimension
        x = self.norm(x)             # -> (batch, d_model)
        return self.fc(x)            # -> (batch, 3)


# =============================================================================
# Utilities
# =============================================================================

def load_data() -> dict:
    """Load all 8 processed field CSVs."""
    frames = {}
    for csv_path in sorted(config.PROCESSED_WEATHER_DIR.glob("*_with_spei.csv")):
        name = csv_path.stem.replace("_with_spei", "")
        if name == "pooled":
            continue
        df = pd.read_csv(csv_path, parse_dates=[config.DATE_COLUMN])
        if "field" not in df.columns:
            df["field"] = name
        frames[name] = df
    logger.info(f"Loaded {len(frames)} fields: {sorted(frames.keys())}")
    return frames


def temporal_split(X, y, meta):
    """
    Strict chronological split by target_date year.

        TRAIN:       target_date.year <= 2011
        VALIDATION:  2012 <= target_date.year <= 2018
        TEST:        target_date.year >= 2019

    Uses the meta DataFrame (with 'target_date') from the sequence builder.
    This replaces the old fraction-based chrono_split.
    """
    dates = pd.to_datetime(meta["target_date"])
    years = dates.dt.year

    train_mask = years <= config.TRAIN_END_YEAR
    val_mask   = (years > config.TRAIN_END_YEAR) & (years <= config.VAL_END_YEAR)
    test_mask  = years > config.VAL_END_YEAR

    return (
        X[train_mask.values], y[train_mask.values],
        X[val_mask.values],   y[val_mask.values],
        X[test_mask.values],  y[test_mask.values],
    )


def dist_str(arr):
    """Format class distribution as a readable string with counts and percentages."""
    counts = np.bincount(arr, minlength=3)
    total  = max(len(arr), 1)
    return {config.INVERSE_LABEL_MAP[i]: f"{counts[i]}({counts[i]*100//total}%)"
            for i in range(3)}


def log_split_distribution(y_tr, y_val, y_te):
    """Report actual class distribution for train / validation / test splits."""
    for name, arr in [("TRAIN", y_tr), ("VALIDATION", y_val), ("TEST", y_te)]:
        counts = np.bincount(arr, minlength=3)
        total  = len(arr)
        pcts   = counts * 100.0 / max(total, 1)
        logger.info(
            f"  {name:10s}: {total:>7d} samples | "
            f"Healthy={counts[0]:>6d} ({pcts[0]:5.1f}%) | "
            f"Moderate={counts[1]:>6d} ({pcts[1]:5.1f}%) | "
            f"Severe={counts[2]:>6d} ({pcts[2]:5.1f}%)"
        )


def compute_metrics(y_true, y_pred, y_prob=None, label=""):
    """
    Compute classification metrics.
    ROC-AUC: One-vs-Rest (OvR), Macro average.
    """
    acc   = accuracy_score(y_true, y_pred)
    bacc  = balanced_accuracy_score(y_true, y_pred)
    mf1   = f1_score(y_true, y_pred, average="macro",   zero_division=0)
    hf1   = f1_score(y_true, y_pred, labels=[0], average="macro", zero_division=0)
    modf1 = f1_score(y_true, y_pred, labels=[1], average="macro", zero_division=0)
    sevf1 = f1_score(y_true, y_pred, labels=[2], average="macro", zero_division=0)
    roc   = 0.0
    if y_prob is not None:
        try:
            yb  = label_binarize(y_true, classes=[0, 1, 2])
            # Explicitly: One-vs-Rest (OvR), Macro average
            roc = roc_auc_score(yb, y_prob, multi_class="ovr", average="macro")
        except Exception:
            pass
    return {
        "model":             label,
        "accuracy":          round(acc,   4),
        "balanced_accuracy": round(bacc,  4),
        "macro_f1":          round(mf1,   4),
        "roc_auc":           round(roc,   4),
        "healthy_f1":        round(hf1,   4),
        "moderate_f1":       round(modf1, 4),
        "severe_f1":         round(sevf1, 4),
    }


def flatten(X):
    """Flatten 3-D (n, seq_len=30, features=9) -> 2-D (n, 270) for RF/XGBoost."""
    return X.reshape(len(X), -1)


def class_weights_tensor(y_tr):
    """Inverse-frequency class weights for CrossEntropyLoss (LSTM/Transformer)."""
    counts  = np.bincount(y_tr, minlength=3).astype(float)
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * 3
    return torch.tensor(weights, dtype=torch.float32).to(DEVICE)


def scale_sequences(scaler: StandardScaler, X: np.ndarray) -> np.ndarray:
    """Apply per-feature z-score to a 3-D (n, t, f) sequence array."""
    n, t, f = X.shape
    return scaler.transform(X.reshape(-1, f)).reshape(n, t, f).astype(np.float32)


def sample_weights(y_tr: np.ndarray) -> np.ndarray:
    """Per-sample inverse-frequency weights for XGBoost/RF fit()."""
    counts = np.bincount(y_tr, minlength=3).astype(float)
    sw_map = len(y_tr) / (3.0 * np.maximum(counts, 1))
    return sw_map[y_tr]


def save_confusion_matrix(y_true, y_pred, tag):
    """Save confusion matrix to results/ as CSV."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    cm_df = pd.DataFrame(
        cm,
        index=[f"True_{c}" for c in config.DROUGHT_CLASSES],
        columns=[f"Pred_{c}" for c in config.DROUGHT_CLASSES],
    )
    path = config.RESULTS_DIR / f"confusion_matrix_{tag}.csv"
    cm_df.to_csv(path)
    logger.info(f"  Confusion matrix saved -> {path}")


# =============================================================================
# Model trainers
# =============================================================================

def train_rf(X_tr, y_tr, X_val, y_val, X_te, y_te, tag):
    """
    Train Random Forest on FLATTENED input.
    Input shape: (n, 30, 9) -> flatten -> (n, 270 features).
    Returns: (test_results_dict, val_macro_f1, model).
    """
    t0  = time.time()
    clf = RandomForestClassifier(
        n_estimators=300, class_weight="balanced",
        n_jobs=-1, random_state=config.RANDOM_SEED
    )
    # RF/XGBoost require flattened 2-D input: (n, 30×9=270)
    clf.fit(flatten(X_tr), y_tr)

    # Validation metrics (for ensemble weights & model selection)
    val_pred = clf.predict(flatten(X_val))
    val_prob = clf.predict_proba(flatten(X_val))
    val_res  = compute_metrics(y_val, val_pred, val_prob, "Random Forest")
    val_f1   = val_res["macro_f1"]

    # Test metrics (final evaluation only)
    pred = clf.predict(flatten(X_te))
    prob = clf.predict_proba(flatten(X_te))
    res  = compute_metrics(y_te, pred, prob, "Random Forest")
    res["val_macro_f1"]  = val_f1
    res["train_time_s"]  = round(time.time() - t0, 1)

    joblib.dump(clf, config.MODELS_DIR / f"rf_{tag}.pkl")
    return res, val_f1, clf


def train_xgb(X_tr, y_tr, X_val, y_val, X_te, y_te, tag):
    """
    Train XGBoost on FLATTENED input.
    Input shape: (n, 30, 9) -> flatten -> (n, 270 features).
    Returns: (test_results_dict, val_macro_f1, model).
    """
    t0     = time.time()
    device = "cuda" if USE_GPU_XGB else "cpu"
    sw     = sample_weights(y_tr)
    clf    = XGBClassifier(
        n_estimators=1000, learning_rate=0.05,
        max_depth=7, subsample=0.8, colsample_bytree=0.8,
        min_child_weight=3, gamma=0.1,
        eval_metric="mlogloss", device=device,
        early_stopping_rounds=50,
        random_state=config.RANDOM_SEED, verbosity=0,
    )
    # XGBoost requires flattened 2-D input: (n, 30×9=270)
    clf.fit(
        flatten(X_tr), y_tr,
        sample_weight=sw,
        eval_set=[(flatten(X_val), y_val)],
        verbose=False,
    )

    # Validation metrics (for ensemble weights & model selection)
    val_pred = clf.predict(flatten(X_val))
    val_prob = clf.predict_proba(flatten(X_val))
    val_res  = compute_metrics(y_val, val_pred, val_prob, "XGBoost")
    val_f1   = val_res["macro_f1"]

    # Test metrics (final evaluation only)
    pred = clf.predict(flatten(X_te))
    prob = clf.predict_proba(flatten(X_te))
    res  = compute_metrics(y_te, pred, prob, "XGBoost")
    res["val_macro_f1"]  = val_f1
    res["train_time_s"]  = round(time.time() - t0, 1)

    joblib.dump(clf, config.MODELS_DIR / f"xgb_{tag}.pkl")
    return res, val_f1, clf


def train_nn(model_cls, model_kwargs, X_tr, y_tr, X_val, y_val, X_te, y_te, tag, name):
    """
    Train LSTM or Transformer on SCALED 3-D input.
    Input shape: (n, seq_len=30, features=9) — no flattening.

    Early stopping:
      - Monitor: Validation Macro-F1
      - Patience: config.EARLY_STOPPING_PATIENCE (15 epochs)
      - Restore: best validation checkpoint weights

    Returns: (test_results_dict, best_val_f1, model_with_best_weights).
    """
    t0        = time.time()
    model     = model_cls(**model_kwargs).to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, min_lr=1e-5
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor(y_tr))

    Xtr_t = torch.tensor(X_tr,  dtype=torch.float32)
    ytr_t = torch.tensor(y_tr,  dtype=torch.long)
    Xv_t  = torch.tensor(X_val, dtype=torch.float32)

    loader = DataLoader(TensorDataset(Xtr_t, ytr_t),
                        batch_size=config.BATCH_SIZE, shuffle=True)

    best_val_f1, patience_cnt, best_state = -1.0, 0, None

    for epoch in range(config.NUM_EPOCHS):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # --- Validation evaluation (NEVER uses test data) ---
        model.eval()
        with torch.no_grad():
            val_pred = model(Xv_t.to(DEVICE)).argmax(1).cpu().numpy()
        val_f1 = f1_score(y_val, val_pred, average="macro", zero_division=0)
        scheduler.step(-val_f1)

        if val_f1 > best_val_f1:
            best_val_f1  = val_f1
            patience_cnt = 0
            # Restore best weights: save a deep copy of the best state
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
            if patience_cnt >= config.EARLY_STOPPING_PATIENCE:
                logger.info(f"    Early stop at epoch {epoch+1} "
                            f"(best val_f1={best_val_f1:.4f})")
                break

    # Restore best validation checkpoint
    model.load_state_dict(best_state)
    model.eval()

    # Test evaluation (final, after best weights are restored)
    Xte_t = torch.tensor(X_te, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        logits = model(Xte_t)
        pred   = logits.argmax(1).cpu().numpy()
        prob   = torch.softmax(logits, dim=1).cpu().numpy()

    res = compute_metrics(y_te, pred, prob, name)
    res["val_macro_f1"]  = round(best_val_f1, 4)
    res["train_time_s"]  = round(time.time() - t0, 1)
    model_path = config.MODELS_DIR / f"{name.lower().replace(' ','_')}_{tag}.pt"
    torch.save(model.state_dict(), model_path)

    return res, best_val_f1, model


def ensemble_predict(models: dict, X_te_raw, X_te_sc, val_f1s: dict):
    """
    Ensemble prediction with FROZEN validation-based weights.

    Weights are proportional to each model's VALIDATION Macro-F1.
    These weights are computed and frozen BEFORE any test-set evaluation.

    Args:
        models:    {"rf": clf, "xgb": clf, "lstm": nn.Module, "tf": nn.Module}
        X_te_raw:  unscaled test sequences (n, 30, 9) — for RF/XGBoost (flattened)
        X_te_sc:   scaled test sequences (n, 30, 9) — for LSTM/Transformer
        val_f1s:   {"rf": float, "xgb": float, "lstm": float, "tf": float}
                   Validation Macro-F1 scores — NOT test scores.

    Returns:
        (predictions, probabilities, frozen_weights_dict)
    """
    w = np.array([
        val_f1s["tf"], val_f1s["xgb"], val_f1s["lstm"], val_f1s["rf"],
    ], dtype=float)
    # Weights directly proportional to validation Macro-F1.
    # No arbitrary minimum — a model with 0 val F1 gets 0 weight.
    # Fallback to uniform weights only if ALL models have 0 F1.
    if w.sum() <= 0:
        w = np.ones_like(w) / len(w)
    else:
        w /= w.sum()

    frozen_weights = {"Transformer": w[0], "XGBoost": w[1], "LSTM": w[2], "RF": w[3]}
    logger.info(
        f"    Ensemble weights (from VALIDATION F1, frozen) -> "
        f"TF={w[0]:.3f}  XGB={w[1]:.3f}  LSTM={w[2]:.3f}  RF={w[3]:.3f}"
    )

    # RF/XGBoost: flattened input (n, 270)
    Xf    = flatten(X_te_raw)
    # LSTM/Transformer: scaled 3-D input (n, 30, 9)
    Xte_t = torch.tensor(X_te_sc, dtype=torch.float32).to(DEVICE)

    xgb_p = models["xgb"].predict_proba(Xf)
    rf_p  = models["rf"].predict_proba(Xf)
    with torch.no_grad():
        lstm_p = torch.softmax(models["lstm"](Xte_t), 1).cpu().numpy()
        tf_p   = torch.softmax(models["tf"](Xte_t),   1).cpu().numpy()

    ens_p = w[0]*tf_p + w[1]*xgb_p + w[2]*lstm_p + w[3]*rf_p
    return ens_p.argmax(1), ens_p, frozen_weights


# =============================================================================
# Per-task pipeline
# =============================================================================

def run_task(frames: dict, horizon: int, task_label: str) -> list:
    """
    Train all 5 models (RF, XGBoost, LSTM, Transformer, Ensemble) for one task.

    All model selection and ensemble weighting uses VALIDATION metrics only.
    TEST metrics are computed once at the end for final reporting.
    """
    task_desc = "Next-Day Forecast" if horizon == 1 else f"{horizon}-Day-Ahead Forecast"
    logger.info("")
    logger.info(f"{'='*65}")
    logger.info(f"  {task_label.upper()}: horizon={horizon} day(s)  ({task_desc})")
    logger.info(f"{'='*65}")

    # Build sequences for all 8 sites (pooled, site-agnostic)
    X, y, meta = build_sequences_all_fields(
        frames,
        sequence_length  = config.SEQUENCE_LENGTH_DAYS,
        forecast_horizon = horizon,
    )
    logger.info(f"Sequences: X={X.shape}  y={y.shape}")

    # Year-based temporal split using target_date from meta
    X_tr, y_tr, X_val, y_val, X_te, y_te = temporal_split(X, y, meta)

    # Report actual class distribution per split
    logger.info("  Class distribution per split:")
    log_split_distribution(y_tr, y_val, y_te)

    # Feature normalization — fit StandardScaler on TRAIN only
    _, t, f = X_tr.shape
    scaler = StandardScaler()
    scaler.fit(X_tr.reshape(-1, f))
    X_tr_sc  = scale_sequences(scaler, X_tr)
    X_val_sc = scale_sequences(scaler, X_val)
    X_te_sc  = scale_sequences(scaler, X_te)
    joblib.dump(scaler, config.MODELS_DIR / f"scaler_{task_label}.pkl")
    logger.info(f"  Scaler fitted on {X_tr.shape[0]} train sequences.")

    results = []
    val_f1s = {}    # Validation Macro-F1 for ensemble weights
    models  = {}    # Trained model objects
    tag     = task_label

    # --- Random Forest (raw X, flattened to 270 features) ---
    logger.info(f"  Training Random Forest ({tag}) ...")
    res_rf, val_f1_rf, rf_clf = train_rf(X_tr, y_tr, X_val, y_val, X_te, y_te, tag)
    results.append(res_rf)
    val_f1s["rf"] = val_f1_rf
    models["rf"]  = rf_clf
    logger.info(f"  [RF]          ValF1={val_f1_rf:.4f}  TestAcc={res_rf['accuracy']:.4f}  "
                f"TestF1={res_rf['macro_f1']:.4f}  SevereF1={res_rf['severe_f1']:.4f}")

    # --- XGBoost (raw X, flattened to 270 features + sample weights) ---
    logger.info(f"  Training XGBoost ({tag}) ...")
    res_xgb, val_f1_xgb, xgb_clf = train_xgb(X_tr, y_tr, X_val, y_val, X_te, y_te, tag)
    results.append(res_xgb)
    val_f1s["xgb"] = val_f1_xgb
    models["xgb"]  = xgb_clf
    logger.info(f"  [XGB]         ValF1={val_f1_xgb:.4f}  TestAcc={res_xgb['accuracy']:.4f}  "
                f"TestF1={res_xgb['macro_f1']:.4f}  SevereF1={res_xgb['severe_f1']:.4f}")

    # --- LSTM (scaled X, 3-D input: n × 30 × 9) ---
    logger.info(f"  Training LSTM ({tag}) ...")
    lstm_kwargs = dict(
        input_size=len(config.MODEL_INPUT_FEATURES),
        hidden_size=config.LSTM_HIDDEN_SIZE,
        num_layers=config.LSTM_NUM_LAYERS,
        num_classes=3,
        dropout=config.LSTM_DROPOUT,
    )
    res_lstm, val_f1_lstm, lstm_m = train_nn(
        DroughtLSTM, lstm_kwargs,
        X_tr_sc, y_tr, X_val_sc, y_val, X_te_sc, y_te, tag, "LSTM"
    )
    results.append(res_lstm)
    val_f1s["lstm"] = val_f1_lstm
    models["lstm"]  = lstm_m
    logger.info(f"  [LSTM]        ValF1={val_f1_lstm:.4f}  TestAcc={res_lstm['accuracy']:.4f}  "
                f"TestF1={res_lstm['macro_f1']:.4f}  SevereF1={res_lstm['severe_f1']:.4f}")

    # --- Transformer (scaled X, 3-D input: n × 30 × 9, mean pooling) ---
    logger.info(f"  Training Transformer ({tag}) ...")
    tf_kwargs = dict(
        input_size=len(config.MODEL_INPUT_FEATURES),
        num_classes=3,
        d_model=config.TF_D_MODEL,
        nhead=config.TF_NHEAD,
        num_layers=config.TF_NUM_LAYERS,
        dropout=config.TF_DROPOUT,
        dim_feedforward=config.TF_DIM_FEEDFORWARD,
    )
    res_tf, val_f1_tf, tf_m = train_nn(
        DroughtTransformer, tf_kwargs,
        X_tr_sc, y_tr, X_val_sc, y_val, X_te_sc, y_te, tag, "Transformer"
    )
    results.append(res_tf)
    val_f1s["tf"] = val_f1_tf
    models["tf"]  = tf_m
    logger.info(f"  [Transformer] ValF1={val_f1_tf:.4f}  TestAcc={res_tf['accuracy']:.4f}  "
                f"TestF1={res_tf['macro_f1']:.4f}  SevereF1={res_tf['severe_f1']:.4f}")

    # --- Ensemble (weights from VALIDATION F1, frozen before test eval) ---
    logger.info(f"  Ensemble (weights from VALIDATION Macro-F1, frozen) ...")
    ens_pred, ens_prob, frozen_weights = ensemble_predict(
        models, X_te, X_te_sc, val_f1s
    )
    res_ens = compute_metrics(y_te, ens_pred, ens_prob, "Ensemble")
    res_ens["val_macro_f1"] = float("nan")  # ensemble has no single val F1
    res_ens["train_time_s"] = float("nan")
    results.append(res_ens)
    logger.info(f"  [Ensemble]    TestAcc={res_ens['accuracy']:.4f}  "
                f"TestF1={res_ens['macro_f1']:.4f}  SevereF1={res_ens['severe_f1']:.4f}")

    # Save confusion matrix for the best individual model (by val F1)
    best_individual = max(val_f1s, key=val_f1s.get)
    # Get best model's test predictions for confusion matrix
    if best_individual in ("rf", "xgb"):
        best_pred = models[best_individual].predict(flatten(X_te))
    else:
        Xte_t = torch.tensor(X_te_sc, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            best_pred = models[best_individual](Xte_t).argmax(1).cpu().numpy()
    save_confusion_matrix(y_te, best_pred, tag)

    # Also save ensemble confusion matrix
    save_confusion_matrix(y_te, ens_pred, f"{tag}_ensemble")

    return results


# =============================================================================
# Main
# =============================================================================

def main():
    logger.info("=" * 70)
    logger.info("  DROUGHT DETECTION -- WEATHER TIME SERIES")
    logger.info("  Task A: Next-Day Drought Forecast        (horizon =  1 day)")
    logger.info(f"  Task B: {config.FORECAST_HORIZON}-Day-Ahead Forecast"
                f"         (horizon = {config.FORECAST_HORIZON} days)")
    logger.info(f"  Window: {config.SEQUENCE_LENGTH_DAYS} days | "
                f"Features: {len(config.MODEL_INPUT_FEATURES)} | "
                f"Sites: {len(config.FIELDS)}")
    logger.info(f"  Split: TRAIN <=  {config.TRAIN_END_YEAR} | "
                f"VAL {config.TRAIN_END_YEAR+1}-{config.VAL_END_YEAR} | "
                f"TEST >= {config.VAL_END_YEAR+1}")
    logger.info(f"  Pooled site-agnostic model (site identity NOT an input feature)")
    logger.info("=" * 70)

    frames = load_data()

    # Task A: Next-Day Forecast (horizon = 1)
    results_A = run_task(frames, horizon=1, task_label="taskA")
    df_A = pd.DataFrame(results_A)
    df_A.to_csv(config.RESULTS_DIR / "taskA_nextday_forecast.csv", index=False)

    # Task B: 15-Day-Ahead Forecast (horizon = 15)
    results_B = run_task(frames, horizon=config.FORECAST_HORIZON, task_label="taskB")
    df_B = pd.DataFrame(results_B)
    df_B.to_csv(config.RESULTS_DIR / "taskB_15day_forecast.csv", index=False)

    # Summary — best model selected by VALIDATION Macro-F1 (not test)
    # Filter out Ensemble for best individual model selection
    df_A_individual = df_A[df_A["model"] != "Ensemble"]
    df_B_individual = df_B[df_B["model"] != "Ensemble"]

    best_A = df_A_individual.sort_values("val_macro_f1", ascending=False).iloc[0]
    best_B = df_B_individual.sort_values("val_macro_f1", ascending=False).iloc[0]

    summary = {
        "Task_A_NextDay_Forecast": {
            "best_model":    best_A["model"],
            "selected_by":   "validation_macro_f1",
            "val_macro_f1":  float(best_A["val_macro_f1"]),
            "test_accuracy": float(best_A["accuracy"]),
            "test_macro_f1": float(best_A["macro_f1"]),
            "test_severe_f1": float(best_A["severe_f1"]),
            "test_roc_auc":  float(best_A["roc_auc"]),
        },
        "Task_B_15Day_Forecast": {
            "best_model":    best_B["model"],
            "selected_by":   "validation_macro_f1",
            "val_macro_f1":  float(best_B["val_macro_f1"]),
            "test_accuracy": float(best_B["accuracy"]),
            "test_macro_f1": float(best_B["macro_f1"]),
            "test_severe_f1": float(best_B["severe_f1"]),
            "test_roc_auc":  float(best_B["roc_auc"]),
        },
        "split": {
            "train": f"<= {config.TRAIN_END_YEAR}",
            "validation": f"{config.TRAIN_END_YEAR+1}-{config.VAL_END_YEAR}",
            "test": f">= {config.VAL_END_YEAR+1}",
        },
    }
    with open(config.RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("")
    logger.info("=" * 70)
    logger.info("  RESULTS SUMMARY (best individual model by VALIDATION Macro-F1)")
    logger.info("=" * 70)
    logger.info(f"  Task A (Next-Day)        -> Best: {best_A['model']:15s}  "
                f"ValF1={best_A['val_macro_f1']:.4f}  "
                f"TestF1={best_A['macro_f1']:.4f}  "
                f"SevereF1={best_A['severe_f1']:.4f}")
    logger.info(f"  Task B (15-Day Forecast) -> Best: {best_B['model']:15s}  "
                f"ValF1={best_B['val_macro_f1']:.4f}  "
                f"TestF1={best_B['macro_f1']:.4f}  "
                f"SevereF1={best_B['severe_f1']:.4f}")
    logger.info("")
    logger.info(f"  Results -> {config.RESULTS_DIR}")
    logger.info(f"  Models  -> {config.MODELS_DIR}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()

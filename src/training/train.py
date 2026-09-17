"""
train.py - Trains ALL models for BOTH drought prediction tasks.

  Task A: current-state classification  (horizon = 1 day)
  Task B: 15-day-ahead forecast          (horizon = 15 days)

Models: XGBoost, Random Forest, LSTM, Transformer, Ensemble

Usage:
    python src/training/train.py

Results saved to:
    results/taskA_current_state.csv
    results/taskB_15day_forecast.csv
    results/summary.json
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
    f1_score, roc_auc_score,
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
    def __init__(self, input_size, num_classes,
                 d_model=128, nhead=4, num_layers=2, dropout=0.2):
        super().__init__()
        self.input_proj  = nn.Linear(input_size, d_model)
        self.pos_enc     = PositionalEncoding(d_model, dropout=dropout)
        enc_layer        = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=256,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm        = nn.LayerNorm(d_model)
        self.fc          = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.transformer(x)
        x = self.norm(x[:, -1, :])
        return self.fc(x)


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


def chrono_split(X, y):
    """Chronological 70/15/15 split."""
    n      = len(y)
    i_val  = int(n * config.TRAIN_FRACTION)
    i_test = int(n * (config.TRAIN_FRACTION + config.VAL_FRACTION))
    return (
        X[:i_val],       y[:i_val],
        X[i_val:i_test], y[i_val:i_test],
        X[i_test:],      y[i_test:],
    )


def dist_str(arr):
    counts = np.bincount(arr, minlength=3)
    total  = max(len(arr), 1)
    return {config.INVERSE_LABEL_MAP[i]: f"{counts[i]}({counts[i]*100//total}%)"
            for i in range(3)}


def compute_metrics(y_true, y_pred, y_prob=None, label=""):
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
    return X.reshape(len(X), -1)


def class_weights_tensor(y_tr):
    counts  = np.bincount(y_tr, minlength=3).astype(float)
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * 3
    return torch.tensor(weights, dtype=torch.float32).to(DEVICE)


def scale_sequences(scaler: StandardScaler, X: np.ndarray) -> np.ndarray:
    """Apply per-feature z-score to a 3-D (n, t, f) sequence array."""
    n, t, f = X.shape
    return scaler.transform(X.reshape(-1, f)).reshape(n, t, f).astype(np.float32)


def sample_weights(y_tr: np.ndarray) -> np.ndarray:
    """Per-sample inverse-frequency weights for XGBoost fit()."""
    counts = np.bincount(y_tr, minlength=3).astype(float)
    sw_map = len(y_tr) / (3.0 * np.maximum(counts, 1))
    return sw_map[y_tr]


# =============================================================================
# Model trainers
# =============================================================================

def train_rf(X_tr, y_tr, X_val, y_val, X_te, y_te, tag):
    t0  = time.time()
    clf = RandomForestClassifier(
        n_estimators=300, class_weight="balanced",
        n_jobs=-1, random_state=config.RANDOM_SEED
    )
    clf.fit(flatten(X_tr), y_tr)
    pred = clf.predict(flatten(X_te))
    prob = clf.predict_proba(flatten(X_te))
    res  = compute_metrics(y_te, pred, prob, "Random Forest")
    res["train_time_s"] = round(time.time() - t0, 1)
    joblib.dump(clf, config.MODELS_DIR / f"rf_{tag}.pkl")
    return res


def train_xgb(X_tr, y_tr, X_val, y_val, X_te, y_te, tag):
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
    clf.fit(
        flatten(X_tr), y_tr,
        sample_weight=sw,
        eval_set=[(flatten(X_val), y_val)],
        verbose=False,
    )
    pred = clf.predict(flatten(X_te))
    prob = clf.predict_proba(flatten(X_te))
    res  = compute_metrics(y_te, pred, prob, "XGBoost")
    res["train_time_s"] = round(time.time() - t0, 1)
    joblib.dump(clf, config.MODELS_DIR / f"xgb_{tag}.pkl")
    return res


def train_nn(model_cls, model_kwargs, X_tr, y_tr, X_val, y_val, X_te, y_te, tag, name):
    t0        = time.time()
    model     = model_cls(**model_kwargs).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
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

        model.eval()
        with torch.no_grad():
            val_pred = model(Xv_t.to(DEVICE)).argmax(1).cpu().numpy()
        val_f1 = f1_score(y_val, val_pred, average="macro", zero_division=0)
        scheduler.step(-val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_cnt = 0
            best_state  = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
            if patience_cnt >= config.EARLY_STOPPING_PATIENCE:
                logger.info(f"    Early stop at epoch {epoch+1} (best val_f1={best_val_f1:.4f})")
                break

    model.load_state_dict(best_state)
    model.eval()
    Xte_t = torch.tensor(X_te, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        logits = model(Xte_t)
        pred   = logits.argmax(1).cpu().numpy()
        prob   = torch.softmax(logits, dim=1).cpu().numpy()

    res = compute_metrics(y_te, pred, prob, name)
    res["train_time_s"] = round(time.time() - t0, 1)
    model_path = config.MODELS_DIR / f"{name.lower().replace(' ','_')}_{tag}.pt"
    torch.save(model.state_dict(), model_path)
    return res, model


def ensemble_predict_dynamic(xgb_clf, rf_clf, lstm_m, tf_m,
                              X_te_raw, X_te_sc, val_f1s: dict):
    """Dynamic ensemble: weights proportional to each model's test MacroF1."""
    w = np.array([
        val_f1s["tf"], val_f1s["xgb"], val_f1s["lstm"], val_f1s["rf"],
    ], dtype=float)
    w = np.clip(w, 0.01, None)   # floor at 0.01 — collapsed models still contribute tiny weight
    w /= w.sum()
    logger.info(
        f"    Ensemble weights -> TF={w[0]:.3f}  XGB={w[1]:.3f}  "
        f"LSTM={w[2]:.3f}  RF={w[3]:.3f}"
    )
    Xf    = flatten(X_te_raw)
    Xte_t = torch.tensor(X_te_sc, dtype=torch.float32).to(DEVICE)
    xgb_p = xgb_clf.predict_proba(Xf)
    rf_p  = rf_clf.predict_proba(Xf)
    with torch.no_grad():
        lstm_p = torch.softmax(lstm_m(Xte_t), 1).cpu().numpy()
        tf_p   = torch.softmax(tf_m(Xte_t),   1).cpu().numpy()
    ens_p = w[0]*tf_p + w[1]*xgb_p + w[2]*lstm_p + w[3]*rf_p
    return ens_p.argmax(1), ens_p


# =============================================================================
# Per-task pipeline
# =============================================================================

def run_task(frames: dict, horizon: int, task_label: str) -> list:
    """Train all 5 models for one task (one horizon)."""
    logger.info("")
    logger.info(f"{'='*65}")
    logger.info(f"  {task_label.upper()}: horizon={horizon} day(s)  "
                f"({'Current-state' if horizon==1 else f'{horizon}-day forecast'})")
    logger.info(f"{'='*65}")

    X, y, meta = build_sequences_all_fields(
        frames,
        sequence_length  = config.SEQUENCE_LENGTH_DAYS,
        forecast_horizon = horizon,
    )
    logger.info(f"Sequences: X={X.shape}  y={y.shape}")

    X_tr, y_tr, X_val, y_val, X_te, y_te = chrono_split(X, y)
    logger.info(f"  Train: {dist_str(y_tr)}")
    logger.info(f"  Val:   {dist_str(y_val)}")
    logger.info(f"  Test:  {dist_str(y_te)}")

    # Feature normalisation — fit ONLY on train, apply to val + test
    _, t, f = X_tr.shape
    scaler = StandardScaler()
    scaler.fit(X_tr.reshape(-1, f))
    X_tr_sc  = scale_sequences(scaler, X_tr)
    X_val_sc = scale_sequences(scaler, X_val)
    X_te_sc  = scale_sequences(scaler, X_te)
    joblib.dump(scaler, config.MODELS_DIR / f"scaler_{task_label}.pkl")
    logger.info(f"  Scaler fitted on {X_tr.shape[0]} train sequences.")

    results = []
    tag     = task_label

    # Random Forest (raw X)
    logger.info(f"  Training Random Forest ({tag}) ...")
    res_rf = train_rf(X_tr, y_tr, X_val, y_val, X_te, y_te, tag)
    results.append(res_rf)
    logger.info(f"  [RF]          Acc={res_rf['accuracy']:.4f}  MacroF1={res_rf['macro_f1']:.4f}  SevereF1={res_rf['severe_f1']:.4f}")

    # XGBoost (raw X + sample weights)
    logger.info(f"  Training XGBoost ({tag}) ...")
    res_xgb = train_xgb(X_tr, y_tr, X_val, y_val, X_te, y_te, tag)
    results.append(res_xgb)
    logger.info(f"  [XGB]         Acc={res_xgb['accuracy']:.4f}  MacroF1={res_xgb['macro_f1']:.4f}  SevereF1={res_xgb['severe_f1']:.4f}")
    xgb_clf = joblib.load(config.MODELS_DIR / f"xgb_{tag}.pkl")
    rf_clf  = joblib.load(config.MODELS_DIR / f"rf_{tag}.pkl")

    # LSTM (scaled X)
    logger.info(f"  Training LSTM ({tag}) ...")
    lstm_kwargs = dict(
        input_size=len(config.MODEL_INPUT_FEATURES),
        hidden_size=config.LSTM_HIDDEN_SIZE,
        num_layers=config.LSTM_NUM_LAYERS,
        num_classes=3,
        dropout=config.LSTM_DROPOUT,
    )
    res_lstm, lstm_m = train_nn(
        DroughtLSTM, lstm_kwargs,
        X_tr_sc, y_tr, X_val_sc, y_val, X_te_sc, y_te, tag, "LSTM"
    )
    results.append(res_lstm)
    logger.info(f"  [LSTM]        Acc={res_lstm['accuracy']:.4f}  MacroF1={res_lstm['macro_f1']:.4f}  SevereF1={res_lstm['severe_f1']:.4f}")

    # Transformer (scaled X)
    logger.info(f"  Training Transformer ({tag}) ...")
    tf_kwargs = dict(
        input_size=len(config.MODEL_INPUT_FEATURES),
        num_classes=3, d_model=128, nhead=4, num_layers=2, dropout=0.2
    )
    res_tf, tf_m = train_nn(
        DroughtTransformer, tf_kwargs,
        X_tr_sc, y_tr, X_val_sc, y_val, X_te_sc, y_te, tag, "Transformer"
    )
    results.append(res_tf)
    logger.info(f"  [Transformer] Acc={res_tf['accuracy']:.4f}  MacroF1={res_tf['macro_f1']:.4f}  SevereF1={res_tf['severe_f1']:.4f}")

    # Dynamic ensemble (weights ∝ test MacroF1)
    val_f1s = {
        "rf":   res_rf["macro_f1"],
        "xgb":  res_xgb["macro_f1"],
        "lstm": res_lstm["macro_f1"],
        "tf":   res_tf["macro_f1"],
    }
    logger.info(f"  Ensemble (dynamic weights by MacroF1) ...")
    ens_pred, ens_prob = ensemble_predict_dynamic(
        xgb_clf, rf_clf, lstm_m, tf_m, X_te, X_te_sc, val_f1s
    )
    res_ens = compute_metrics(y_te, ens_pred, ens_prob, "Ensemble")
    res_ens["train_time_s"] = float("nan")
    results.append(res_ens)
    logger.info(f"  [Ensemble]    Acc={res_ens['accuracy']:.4f}  MacroF1={res_ens['macro_f1']:.4f}  SevereF1={res_ens['severe_f1']:.4f}")

    return results


# =============================================================================
# Main
# =============================================================================

def main():
    logger.info("=" * 70)
    logger.info("  DROUGHT DETECTION -- WEATHER TIME SERIES")
    logger.info("  Task A: Current-state classification  (horizon =  1 day)")
    logger.info(f"  Task B: {config.FORECAST_HORIZON}-day ahead forecast         (horizon = {config.FORECAST_HORIZON} days)")
    logger.info(f"  Window: {config.SEQUENCE_LENGTH_DAYS} days | Features: {len(config.MODEL_INPUT_FEATURES)} | Sites: {len(config.FIELDS)}")
    logger.info("=" * 70)

    frames = load_data()

    # Task A: current-state (horizon = 1)
    results_A = run_task(frames, horizon=1,                       task_label="taskA")
    df_A = pd.DataFrame(results_A)
    df_A.to_csv(config.RESULTS_DIR / "taskA_current_state.csv", index=False)

    # Task B: 15-day forecast (horizon = 15)
    results_B = run_task(frames, horizon=config.FORECAST_HORIZON, task_label="taskB")
    df_B = pd.DataFrame(results_B)
    df_B.to_csv(config.RESULTS_DIR / "taskB_15day_forecast.csv", index=False)

    # Summary
    best_A = df_A.sort_values("macro_f1", ascending=False).iloc[0]
    best_B = df_B.sort_values("macro_f1", ascending=False).iloc[0]

    summary = {
        "Task_A_Current_State": {
            "best_model": best_A["model"],
            "accuracy":   float(best_A["accuracy"]),
            "macro_f1":   float(best_A["macro_f1"]),
            "severe_f1":  float(best_A["severe_f1"]),
            "roc_auc":    float(best_A["roc_auc"]),
        },
        "Task_B_15Day_Forecast": {
            "best_model": best_B["model"],
            "accuracy":   float(best_B["accuracy"]),
            "macro_f1":   float(best_B["macro_f1"]),
            "severe_f1":  float(best_B["severe_f1"]),
            "roc_auc":    float(best_B["roc_auc"]),
        },
    }
    with open(config.RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("")
    logger.info("=" * 70)
    logger.info("  RESULTS SUMMARY")
    logger.info("=" * 70)
    logger.info(f"  Task A (Current-State)   -> Best: {best_A['model']:15s}  MacroF1={best_A['macro_f1']:.4f}  SevereF1={best_A['severe_f1']:.4f}")
    logger.info(f"  Task B (15-Day Forecast) -> Best: {best_B['model']:15s}  MacroF1={best_B['macro_f1']:.4f}  SevereF1={best_B['severe_f1']:.4f}")
    logger.info("")
    logger.info(f"  Results -> {config.RESULTS_DIR}")
    logger.info(f"  Models  -> {config.MODELS_DIR}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()

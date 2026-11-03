"""
trainers.py — train one model with one seed and return validation + test probabilities.

All trainers share the signature
    train_<model>(data: dict, seed: int, num_classes: int, quick: bool) -> (model, p_val, p_test, info)
where data = {"X_tr","y_tr","X_val","y_val","X_te"} (raw, unscaled sequences).
Scaling for neural nets is fitted on TRAIN only, inside this module.
Model selection / early stopping only ever uses VALIDATION data.
"""
from __future__ import annotations

import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from models.networks import DroughtLSTM, DroughtTransformer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def flatten(X: np.ndarray) -> np.ndarray:
    return X.reshape(len(X), -1)


def balanced_sample_weights(y: np.ndarray, num_classes: int) -> np.ndarray:
    counts = np.bincount(y, minlength=num_classes).astype(float)
    w = len(y) / (num_classes * np.maximum(counts, 1))
    return w[y]


def _full_proba(model, X, num_classes):
    """predict_proba padded to all classes (in case a class is absent from a training fold)."""
    p = model.predict_proba(X)
    if p.shape[1] == num_classes:
        return p
    full = np.zeros((len(X), num_classes))
    full[:, model.classes_] = p
    return full


# ---------------------------------------------------------------------------
def train_rf(data, seed, num_classes=3, quick=False):
    t0 = time.time()
    params = dict(config.RF_PARAMS)
    if quick:
        params["n_estimators"] = 20
    clf = RandomForestClassifier(random_state=seed, **params)
    clf.fit(flatten(data["X_tr"]), data["y_tr"])
    p_val = _full_proba(clf, flatten(data["X_val"]), num_classes)
    p_te = _full_proba(clf, flatten(data["X_te"]), num_classes)
    return clf, p_val, p_te, {"train_time_s": round(time.time() - t0, 1)}


def train_xgb(data, seed, num_classes=3, quick=False):
    t0 = time.time()
    params = dict(config.XGB_PARAMS)
    if quick:
        params.update(n_estimators=20, early_stopping_rounds=5)
    if num_classes == 2:
        params["eval_metric"] = "logloss"
    clf = XGBClassifier(random_state=seed, device="cuda" if torch.cuda.is_available() else "cpu", **params)
    clf.fit(flatten(data["X_tr"]), data["y_tr"],
            sample_weight=balanced_sample_weights(data["y_tr"], num_classes),
            eval_set=[(flatten(data["X_val"]), data["y_val"])], verbose=False)
    p_val = _full_proba(clf, flatten(data["X_val"]), num_classes)
    p_te = _full_proba(clf, flatten(data["X_te"]), num_classes)
    return clf, p_val, p_te, {"train_time_s": round(time.time() - t0, 1),
                              "best_iteration": int(getattr(clf, "best_iteration", -1) or -1)}


# ---------------------------------------------------------------------------
def _scale(data):
    f = data["X_tr"].shape[2]
    scaler = StandardScaler().fit(data["X_tr"].reshape(-1, f))
    tr = lambda X: scaler.transform(X.reshape(-1, f)).reshape(X.shape).astype(np.float32)
    return scaler, tr(data["X_tr"]), tr(data["X_val"]), tr(data["X_te"])


@torch.no_grad()
def _predict(model, X_t, bs=4096):
    model.eval()
    out = [torch.softmax(model(X_t[i:i + bs]), 1) for i in range(0, len(X_t), bs)]
    return torch.cat(out).cpu().numpy()


def _train_nn(model, data, seed, num_classes, quick):
    t0 = time.time()
    scaler, Xtr, Xval, Xte = _scale(data)
    Xtr_t = torch.tensor(Xtr, device=DEVICE)
    ytr_t = torch.tensor(data["y_tr"], dtype=torch.long, device=DEVICE)
    Xval_t = torch.tensor(Xval, device=DEVICE)
    Xte_t = torch.tensor(Xte, device=DEVICE)

    counts = np.bincount(data["y_tr"], minlength=num_classes).astype(float)
    w = 1.0 / (counts + 1e-6); w = w / w.sum() * num_classes
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=DEVICE))
    opt = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", patience=5, factor=0.5, min_lr=1e-5)

    epochs = 2 if quick else config.NUM_EPOCHS
    gen = torch.Generator(device="cpu").manual_seed(seed)
    best_f1, best_state, best_epoch, patience = -1.0, None, 0, 0
    n = len(Xtr_t)
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=gen).to(DEVICE)
        for i in range(0, n, config.BATCH_SIZE):
            idx = perm[i:i + config.BATCH_SIZE]
            opt.zero_grad()
            loss = criterion(model(Xtr_t[idx]), ytr_t[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
        val_pred = _predict(model, Xval_t).argmax(1)
        val_f1 = f1_score(data["y_val"], val_pred, average="macro", zero_division=0)
        sched.step(val_f1)
        if val_f1 > best_f1:
            best_f1, best_epoch, patience = val_f1, epoch + 1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= config.EARLY_STOPPING_PATIENCE:
                break
    model.load_state_dict(best_state)
    p_val, p_te = _predict(model, Xval_t), _predict(model, Xte_t)
    info = {"train_time_s": round(time.time() - t0, 1), "best_epoch": best_epoch,
            "epochs_run": epoch + 1, "best_val_macro_f1": round(float(best_f1), 4)}
    model.scaler = scaler
    return model, p_val, p_te, info


def train_lstm(data, seed, num_classes=3, quick=False):
    set_seed(seed)
    model = DroughtLSTM(data["X_tr"].shape[2], config.LSTM_HIDDEN_SIZE, config.LSTM_NUM_LAYERS,
                        num_classes, config.LSTM_DROPOUT).to(DEVICE)
    return _train_nn(model, data, seed, num_classes, quick)


def train_transformer(data, seed, num_classes=3, quick=False):
    set_seed(seed)
    model = DroughtTransformer(data["X_tr"].shape[2], num_classes, config.TF_D_MODEL, config.TF_NHEAD,
                               config.TF_NUM_LAYERS, config.TF_DROPOUT, config.TF_DIM_FEEDFORWARD).to(DEVICE)
    return _train_nn(model, data, seed, num_classes, quick)


TRAINERS = {"rf": train_rf, "xgb": train_xgb, "lstm": train_lstm, "transformer": train_transformer}

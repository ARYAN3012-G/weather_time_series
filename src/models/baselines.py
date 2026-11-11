"""
baselines.py — reference forecasts every model must beat.

All baselines are fitted on TRAIN samples only and return class probabilities.

  persistence : class on the last input day t is the forecast for t+h (one-hot)
  markov      : P(class at t+h | class at t), transition matrix from TRAIN
                (probabilistic persistence; gives meaningful ROC/PR-AUC)
  climatology : P(class | calendar day of target), from TRAIN labels,
                smoothed with a +/-15-day circular window
  majority    : constant TRAIN class frequencies
  wb_ensemble : physical baseline - observed part of the SPEI-30 window + each training year's
                water balance for the unknown days, frozen SPEI calibration (see wb_ensemble)
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


def all_baselines(meta_tr, y_tr, meta_val, meta_te, k: int = 3, frames: dict | None = None
                  ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """{name: (p_val, p_test)}; frames (prepared daily data) enables the water-balance ensemble."""
    extra = {} if frames is None else {
        "wb_ensemble": (wb_ensemble(meta_val.reset_index(drop=True), frames, k),
                        wb_ensemble(meta_te.reset_index(drop=True), frames, k))}
    return extra | {
        "persistence": (persistence(meta_val, k), persistence(meta_te, k)),
        "markov":      (markov(meta_tr, y_tr, meta_val, k), markov(meta_tr, y_tr, meta_te, k)),
        "climatology": (climatology(meta_tr, y_tr, meta_val, k), climatology(meta_tr, y_tr, meta_te, k)),
        "majority":    (majority(y_tr, len(meta_val), k), majority(y_tr, len(meta_te), k)),
    }


# ---------------------------------------------------------------------------
# Water-balance climatological ensemble (strong physical baseline)
# ---------------------------------------------------------------------------
def wb_ensemble(meta: pd.DataFrame, frames: dict, k: int = 3) -> np.ndarray:
    """P(class at t+h) from the SPEI-30 accumulation itself, no learning:
    target-day accumulation = water balance already observed in the 30-day window (days up to t)
    + the unknown days (t, t+h] taken from each TRAINING year 1979-2011 in turn (same calendar
    days). Each year gives one SPEI value with the site's frozen 1979-2011 calibration; class
    probabilities = shares of years (Laplace-smoothed). Uses only data up to t plus the training
    climatology (an ensemble-streamflow-prediction style forecast). Ensemble years that contain
    the sample's own unknown days are left out, so training samples never see their future."""
    import config
    from dataprep.spei import SPEICalibrator

    out = np.full((len(meta), k), np.nan)
    years = np.arange(config.SPEI_CALIB_START, config.SPEI_CALIB_END + 1)
    for field, idx in meta.groupby("field").groups.items():
        df = frames[field].sort_values(config.DATE_COLUMN).reset_index(drop=True)
        d = pd.to_datetime(df[config.DATE_COLUMN])
        wb = (df["Precipitation_Flux"] - df["ReferenceET_PenmanMonteith_FAO56"]).to_numpy(float)
        cal = SPEICalibrator(config.SPEI_SCALE_DAYS, config.SPEI_CALIB_START, config.SPEI_CALIB_END,
                             config.SPEI_CLIP).fit(d, pd.Series(wb).rolling(config.SPEI_SCALE_DAYS).sum())
        # climatology matrix: training year x calendar day (365, Feb 29 dropped), tiled twice
        noleap = ~((d.dt.month == 2) & (d.dt.day == 29))
        cday = (pd.to_datetime(dict(year=2001, month=d.dt.month, day=d.dt.day.where(noleap, 28)))
                .dt.dayofyear.to_numpy() - 1)
        clim = np.full((len(years), 365), np.nan)
        for i, y in enumerate(years):
            m = (d.dt.year == y).to_numpy() & noleap.to_numpy()
            clim[i, cday[m]] = wb[m]
        clim = np.where(np.isnan(clim), np.nanmean(clim, axis=0), clim)
        cs = np.concatenate([np.zeros((len(years), 1)), np.cumsum(np.tile(clim, 2), axis=1)], axis=1)
        pos = pd.Series(np.arange(len(d)), index=d)
        cw = np.concatenate([[0.0], np.cumsum(wb)])

        m = meta.loc[idx]
        t = pd.to_datetime(m["last_input_date"])
        h = m["horizon"].to_numpy(int)
        L = config.SPEI_SCALE_DAYS
        n_unknown = np.minimum(h, L)
        n_known = L - n_unknown
        p_t = pos.reindex(t).to_numpy()
        known = cw[p_t + 1] - cw[p_t + 1 - n_known]                       # days t-n_known+1 .. t
        first_unknown = t + pd.to_timedelta(h - n_unknown + 1, "D")
        s = (pd.to_datetime(dict(year=2001, month=first_unknown.dt.month,
                                 day=first_unknown.dt.day.where(~((first_unknown.dt.month == 2)
                                                                   & (first_unknown.dt.day == 29)), 28)))
             .dt.dayofyear.to_numpy() - 1)
        unknown = cs[:, s + n_unknown] - cs[:, s]                          # (years, samples)
        acc = known[None, :] + unknown
        tgt = pd.to_datetime(m["target_date"]).to_numpy()
        spei = cal.transform(pd.Series(np.tile(tgt, len(years))),
                             pd.Series(acc.ravel())).to_numpy().reshape(acc.shape)
        cls = np.where(spei < config.SPEI_THRESHOLDS["Moderate"], 2,
                       np.where(spei < config.SPEI_THRESHOLDS["Healthy"], 1, 0))
        # never use the sample's own (actual) future: drop ensemble years that ARE the unknown days
        # (only possible for samples inside 1979-2011, i.e. training samples)
        own = ((years[:, None] == first_unknown.dt.year.to_numpy()[None, :])
               | (years[:, None] == pd.to_datetime(m["target_date"]).dt.year.to_numpy()[None, :]))
        probs = np.stack([((cls == c) & ~own).sum(0) for c in range(k)], axis=1) + 1.0
        out[meta.index.get_indexer(idx)] = probs / probs.sum(1, keepdims=True)
    return out

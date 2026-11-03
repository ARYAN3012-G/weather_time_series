"""
spei.py — SPEI with a frozen, training-period-only calibration.

Method (Vicente-Serrano et al., 2010, J. Climate):
  1. Climatic water balance  D = P - PET  (daily)
  2. Accumulate over SPEI_SCALE_DAYS (rolling sum, uses full history, so no
     observations are discarded at split boundaries).
  3. For each calendar month, fit a 3-parameter log-logistic distribution by
     L-moments (probability-weighted moments) using ONLY reference-period values.
  4. SPEI = Phi^-1( F(D_acc) ) for every day, with the frozen parameters.

If a month's log-logistic fit is invalid (beta <= 1 or non-finite), that month
falls back to a z-score with reference-period mean/std. Fallbacks are recorded.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import gamma as gamma_fn


def water_balance(df: pd.DataFrame) -> pd.Series:
    return df["Precipitation_Flux"] - df["ReferenceET_PenmanMonteith_FAO56"]


def accumulate(wb: pd.Series, scale: int) -> pd.Series:
    return wb.rolling(scale, min_periods=scale).sum()


def fit_loglogistic_lmom(x: np.ndarray) -> dict | None:
    """3-parameter log-logistic via unbiased PWMs (Vicente-Serrano 2010, eqs. 4–8)."""
    x = np.sort(np.asarray(x, dtype=float))
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 30:
        return None
    i = np.arange(1, n + 1)
    F = (i - 0.35) / n
    w0 = x.mean()
    w1 = np.mean(x * (1 - F))
    w2 = np.mean(x * (1 - F) ** 2)
    denom = 6 * w1 - w0 - 6 * w2
    if denom == 0:
        return None
    beta = (2 * w1 - w0) / denom
    if not np.isfinite(beta) or beta <= 1.0:
        return None
    g = gamma_fn(1 + 1 / beta) * gamma_fn(1 - 1 / beta)
    alpha = (w0 - 2 * w1) * beta / g
    gam = w0 - alpha * g
    if not (np.isfinite(alpha) and np.isfinite(gam)) or alpha <= 0:
        return None
    return {"alpha": float(alpha), "beta": float(beta), "gamma": float(gam)}


def loglogistic_cdf(x: np.ndarray, alpha: float, beta: float, gamma: float) -> np.ndarray:
    r = np.clip(np.asarray(x, dtype=float) - gamma, 1e-12, None)
    return 1.0 / (1.0 + (alpha / r) ** beta)


class SPEICalibrator:
    """Fit per-calendar-month distributions on a reference period, then transform any period."""

    def __init__(self, scale: int, calib_start: int, calib_end: int, clip: float = 4.0):
        self.scale = scale
        self.calib_start = calib_start
        self.calib_end = calib_end
        self.clip = clip
        self.params_: dict[int, dict] = {}

    def fit(self, dates: pd.Series, d_acc: pd.Series) -> "SPEICalibrator":
        dates = pd.to_datetime(dates)
        ref = (dates.dt.year >= self.calib_start) & (dates.dt.year <= self.calib_end) & d_acc.notna()
        for m in range(1, 13):
            vals = d_acc[ref & (dates.dt.month == m)].to_numpy()
            p = fit_loglogistic_lmom(vals)
            if p is not None:
                p["method"] = "loglogistic"
            else:
                p = {"method": "zscore", "mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1))}
            p["n_ref"] = int(len(vals))
            self.params_[m] = p
        return self

    def transform(self, dates: pd.Series, d_acc: pd.Series) -> pd.Series:
        if not self.params_:
            raise RuntimeError("SPEICalibrator.transform called before fit")
        dates = pd.to_datetime(dates)
        out = np.full(len(d_acc), np.nan)
        x = d_acc.to_numpy(dtype=float)
        months = dates.dt.month.to_numpy()
        for m, p in self.params_.items():
            idx = (months == m) & np.isfinite(x)
            if not idx.any():
                continue
            if p["method"] == "loglogistic":
                cdf = loglogistic_cdf(x[idx], p["alpha"], p["beta"], p["gamma"])
                lo, hi = stats.norm.cdf(-self.clip), stats.norm.cdf(self.clip)
                out[idx] = stats.norm.ppf(np.clip(cdf, lo, hi))
            else:
                out[idx] = (x[idx] - p["mean"]) / p["std"]
        out = np.clip(out, -self.clip, self.clip)
        return pd.Series(out, index=d_acc.index, name="SPEI")

    def to_dict(self) -> dict:
        return {"scale_days": self.scale, "calib_start": self.calib_start,
                "calib_end": self.calib_end, "clip": self.clip,
                "params": {str(k): v for k, v in self.params_.items()}}

    @classmethod
    def from_dict(cls, d: dict) -> "SPEICalibrator":
        c = cls(d["scale_days"], d["calib_start"], d["calib_end"], d["clip"])
        c.params_ = {int(k): v for k, v in d["params"].items()}
        return c

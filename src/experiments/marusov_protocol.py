"""
marusov_protocol.py — adapt the evaluation methodology of

  Marusov, Grabar, Maximov, Sotiriadi, Bulkin, Zaytsev. "Long-term drought prediction
  using deep neural networks based on geospatial weather data."
  Environmental Modelling & Software (arXiv:2309.06212).

Their setup -> our adaptation
  binary drought, PDSI < -2              -> SPEI(t+h) < config.MARUSOV_DROUGHT_THRESHOLD (-1.0)
  input = index history only             -> feature set 'spei_only'  (their setting)
                                            feature set 'weather_spei' (our approach)
  majority-class baseline                -> majority (+ persistence, which they did not report)
  logistic regression on the latest value-> logreg_last (SPEI on day t)
  XGBoost                                -> xgb on the full 30-day history window
  ROC-AUC / PR-AUC / F1 per grid cell,   -> per site, then MEDIAN across the 7 sites
  median across cells
  horizons 1–12 months, chronological    -> 7 / 15 / 30 days, purged chronological
  70/30 split, no separate validation       train / val / test (stricter)

ConvLSTM / EarthFormer need a dense spatial grid; with 7 point sites they are not
applicable, which is stated as a limitation. This is a METHODOLOGY comparison:
numbers are not directly comparable with the paper's monthly PDSI results.

Usage:
  python src/experiments/marusov_protocol.py                  # all horizons, seeds from config
  python src/experiments/marusov_protocol.py --models xgb --seeds 0 --quick
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from features.sequence_builder import build_sequences_all_fields, load_prepared
from features.splits import make_masks
from models.trainers import TRAINERS
from utils.logger import get_logger
from utils.report import to_md

logger = get_logger("marusov_protocol")


def per_site_scores(meta_te, y, p):
    rows = []
    for site, idx in meta_te.groupby("field").indices.items():
        yt, pt = y[idx], p[idx]
        ok = len(np.unique(yt)) == 2
        rows.append({"site": site,
                     "roc_auc": roc_auc_score(yt, pt) if ok else np.nan,
                     "pr_auc": average_precision_score(yt, pt) if ok else np.nan,
                     "f1": f1_score(yt, (pt >= 0.5).astype(int), zero_division=0),
                     "prevalence": yt.mean()})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", type=int, nargs="+", default=config.HORIZONS)
    ap.add_argument("--models", nargs="+", default=["xgb", "lstm"], choices=["xgb", "lstm", "transformer", "rf"])
    ap.add_argument("--seeds", type=int, nargs="+", default=config.SEEDS)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    out_dir = config.MARUSOV_DIR / ("_quick" if args.quick else "")
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = load_prepared()
    rows = []
    for h in args.horizons:
        for fs in ["spei_only", "weather_spei"]:
            X, _, meta = build_sequences_all_fields(frames, fs, h)
            yb = (meta["spei_target"].to_numpy() < config.MARUSOV_DROUGHT_THRESHOLD).astype(np.int64)
            masks = make_masks(meta, config.FIELDS, config.FIELDS)
            m_te = meta[masks["test"]].reset_index(drop=True)
            y_te = yb[masks["test"]]
            tr_idx = np.flatnonzero(masks["train"])
            if args.quick and len(tr_idx) > 5000:
                tr_idx = np.random.default_rng(0).choice(tr_idx, 5000, replace=False)

            preds = {}
            if fs == "spei_only":   # reference methods (feature-set independent / index-only)
                p_tr = yb[tr_idx].mean()
                preds["majority"] = [np.full(len(y_te), p_tr)]
                preds["persistence"] = [(m_te["spei_t"].to_numpy() < config.MARUSOV_DROUGHT_THRESHOLD).astype(float)]
                sc = StandardScaler().fit(meta.loc[tr_idx, ["spei_t"]])
                lr = LogisticRegression(class_weight="balanced", max_iter=1000)
                lr.fit(sc.transform(meta.loc[tr_idx, ["spei_t"]]), yb[tr_idx])
                preds["logreg_last"] = [lr.predict_proba(sc.transform(m_te[["spei_t"]]))[:, 1]]
            data = {"X_tr": X[tr_idx], "y_tr": yb[tr_idx], "X_val": X[masks["val"]],
                    "y_val": yb[masks["val"]], "X_te": X[masks["test"]]}
            for mname in args.models:
                preds[mname] = []
                for seed in args.seeds:
                    _, _, p_te, info = TRAINERS[mname](data, seed, 2, args.quick)
                    preds[mname].append(p_te[:, 1])
                    logger.info(f"h={h} {fs} {mname} seed={seed} done ({info['train_time_s']}s)")

            for mname, plist in preds.items():
                seed_scores = [per_site_scores(m_te, y_te, p) for p in plist]
                for si, sc_df in enumerate(seed_scores):
                    for _, r in sc_df.iterrows():
                        rows.append({"horizon": h, "input": fs, "model": mname, "seed": si,
                                     **r.to_dict()})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "per_site_scores.csv", index=False)

    med = (df.groupby(["horizon", "input", "model", "seed"])[["roc_auc", "pr_auc", "f1"]].median()
             .groupby(["horizon", "input", "model"]).agg(["mean", "std"]))
    med.columns = [f"median_{a}_{b}" for a, b in med.columns]
    med = med.reset_index()
    med.to_csv(out_dir / "median_scores.csv", index=False)

    lines = ["# Marusov et al. protocol on our data\n",
             f"Binary drought = SPEI(t+h) < {config.MARUSOV_DROUGHT_THRESHOLD}. Scores computed per site on the "
             "TEST period (2019+) and summarised by the median across the 7 sites (as in the paper); "
             "mean ± std across seeds. `spei_only` = their input (index history); "
             "`weather_spei` = our input. Methodology comparison only — not comparable to the "
             "paper's monthly PDSI numbers.\n"]
    for h in sorted(med.horizon.unique()):
        s = med[med.horizon == h]
        t = pd.DataFrame({"input": s["input"], "model": s["model"]})
        for k in ["roc_auc", "pr_auc", "f1"]:
            t[f"median {k}"] = [f"{a:.3f}" + (f" ± {b:.3f}" if b > 0 else "")
                                for a, b in zip(s[f"median_{k}_mean"], s[f"median_{k}_std"].fillna(0))]
        lines.append(f"\n## Horizon {h} days\n\n" + to_md(t, index=False))
    (out_dir / "MARUSOV_PROTOCOL.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info(f"Wrote {out_dir / 'MARUSOV_PROTOCOL.md'}")


if __name__ == "__main__":
    main()
    # All results are written at this point. On Windows, the GPU libraries can abort
    # during process shutdown (0xc0000409 in ucrtbase.dll) after long runs, giving a
    # non-zero exit code that stops run_all.bat although nothing failed. Flush the logs,
    # then end the process directly so that shutdown code never runs.
    # (Errors inside main() still raise before this point and exit with code 1.)
    import logging, os
    logging.shutdown()
    sys.stdout.flush(); sys.stderr.flush()
    if os.name == "nt":
        import ctypes
        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        k32.TerminateProcess(ctypes.c_void_p(k32.GetCurrentProcess()), 0)
    os._exit(0)

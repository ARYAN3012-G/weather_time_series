"""
aggregate.py — turn saved predictions into the final tables and report (no training).

For every split / fold / horizon:
  * metrics for baselines and every (feature_set, model, seed)
  * validation-F1-weighted ensemble per (feature_set, seed) when rf/xgb/lstm/transformer all exist
  * severe-class threshold tuned on VALIDATION, applied once to TEST (POD / FAR / CSI)
  * Brier skill score vs climatology
  * block bootstrap 95% CIs (blocks = target year-month, all sites together) on the
    seed-averaged prediction, incl. paired difference vs the strongest baseline
    (chosen by VALIDATION macro-F1)

Outputs (results/v3/tables/):
  all_runs_<split>.csv     one row per run (seed level)
  summary_<split>.csv      mean / std over seeds + bootstrap CIs
  confusion_<split>_<fold>_h<h>.csv   best model (by val F1) seed-mean confusion matrix
  REPORT_<split>.md        readable tables for the paper / professors

Usage:
  python src/experiments/aggregate.py                # all splits found
  python src/experiments/aggregate.py --split temporal --n-boot 1000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from evaluation.metrics import (block_bootstrap, compute_metrics, predict_with_severe_threshold,
                                tune_severe_threshold)
from utils.logger import get_logger
from utils.report import to_md

logger = get_logger("aggregate")
ENSEMBLE_MEMBERS = ["rf", "xgb", "lstm", "transformer"]
BASELINES = ["persistence", "markov", "climatology", "majority"]
# Their probabilities are (nearly) constant over time, so a tuned severe threshold
# gives meaningless POD/FAR (e.g. POD 1.0 by flagging every day). Not reported.
NO_SEVERE_WARNING = ["climatology", "majority"]
KEY = ["macro_f1", "severe_f1", "balanced_accuracy", "roc_auc_macro_ovr", "pr_auc_macro",
       "severe_pr_auc", "brier", "bss_vs_climatology",
       "tuned_severe_f1", "tuned_severe_pod", "tuned_severe_far", "tuned_severe_csi", "val_macro_f1"]


def evaluate(y_val, y_te, p_val, p_te, brier_clim) -> dict:
    m = compute_metrics(y_te, p_te)
    m["val_macro_f1"] = round(f1_score(y_val, p_val.argmax(1), average="macro", zero_division=0), 4)
    tau, val_sev = tune_severe_threshold(y_val, p_val)
    tuned = compute_metrics(y_te, p_te, predict_with_severe_threshold(p_te, tau))
    m.update({"severe_tau": tau, "val_severe_f1_tuned": val_sev,
              "tuned_severe_f1": tuned["severe_f1"], "tuned_severe_pod": tuned["severe_pod"],
              "tuned_severe_far": tuned["severe_far"], "tuned_severe_csi": tuned["severe_csi"],
              "tuned_macro_f1": tuned["macro_f1"]})
    m["bss_vs_climatology"] = round(1 - m["brier"] / brier_clim, 4) if brier_clim > 0 else np.nan
    return m


def load_preds(path: Path):
    d = np.load(path)
    return d["p_val"].astype(float), d["p_test"].astype(float)


def aggregate_base(split: str, fold: str, base: Path, n_boot: int):
    tg = np.load(base / "targets.npz")
    y_val, y_te = tg["y_val"], tg["y_test"]
    tmeta = pd.read_csv(base / "test_meta.csv", parse_dates=["target_date"])
    blocks = tmeta["target_date"].dt.strftime("%Y-%m").to_numpy()
    h = int(base.name[1:])
    ctx = {"split": split, "fold": fold, "horizon": h}

    bpreds = {b: load_preds(base / "baselines" / b / "preds.npz")
              for b in BASELINES if (base / "baselines" / b / "preds.npz").exists()}
    brier_clim = compute_metrics(y_te, bpreds["climatology"][1])["brier"]

    rows, seed_mean = [], {}
    for b, (pv, pt) in bpreds.items():
        rows.append({**ctx, "feature_set": "-", "model": b, "seed": -1, "kind": "baseline",
                     **evaluate(y_val, y_te, pv, pt, brier_clim)})
        seed_mean[("-", b)] = (pv, pt)

    for fs_dir in sorted(p for p in base.iterdir() if p.is_dir() and p.name in config.FEATURE_SETS):
        fs = fs_dir.name
        per_model = {}
        for m_dir in sorted(p for p in fs_dir.iterdir() if p.is_dir()):
            for s_dir in sorted(m_dir.glob("seed*")):
                if not (s_dir / "preds.npz").exists():
                    continue
                seed = int(s_dir.name[4:])
                pv, pt = load_preds(s_dir / "preds.npz")
                per_model.setdefault(m_dir.name, {})[seed] = (pv, pt)
                rows.append({**ctx, "feature_set": fs, "model": m_dir.name, "seed": seed, "kind": "model",
                             **evaluate(y_val, y_te, pv, pt, brier_clim)})
        # ensemble per seed (weights = validation macro-F1, frozen before test)
        if all(m in per_model for m in ENSEMBLE_MEMBERS):
            common = set.intersection(*(set(per_model[m]) for m in ENSEMBLE_MEMBERS))
            for seed in sorted(common):
                w = np.array([f1_score(y_val, per_model[m][seed][0].argmax(1), average="macro")
                              for m in ENSEMBLE_MEMBERS])
                w = w / w.sum() if w.sum() > 0 else np.full(len(w), 1 / len(w))
                pv = sum(wi * per_model[m][seed][0] for wi, m in zip(w, ENSEMBLE_MEMBERS))
                pt = sum(wi * per_model[m][seed][1] for wi, m in zip(w, ENSEMBLE_MEMBERS))
                per_model.setdefault("ensemble", {})[seed] = (pv, pt)
                m_ens = evaluate(y_val, y_te, pv, pt, brier_clim)
                # weights were fitted on validation, so the ensemble's validation score is
                # in-sample: report test only, and never pick the ensemble "by validation"
                m_ens["val_macro_f1"] = np.nan
                rows.append({**ctx, "feature_set": fs, "model": "ensemble", "seed": seed, "kind": "model",
                             "ensemble_weights": json.dumps(dict(zip(ENSEMBLE_MEMBERS, np.round(w, 4).tolist()))),
                             **m_ens})
        for m, seeds in per_model.items():
            seed_mean[(fs, m)] = (np.mean([v[0] for v in seeds.values()], 0),
                                  np.mean([v[1] for v in seeds.values()], 0))

    runs = pd.DataFrame(rows)

    # strongest baseline by VALIDATION macro-F1 = reference for paired bootstrap
    bl = runs[runs.kind == "baseline"]
    ref_name = bl.sort_values("val_macro_f1", ascending=False).iloc[0]["model"]
    ref_pred = seed_mean[("-", ref_name)][1].argmax(1)

    boot_rows = []
    for (fs, m), (pv, pt) in seed_mean.items():
        b = block_bootstrap(y_te, pt.argmax(1), blocks, n_boot=n_boot, seed=config.BOOTSTRAP_SEED,
                            pred_ref=None if m == ref_name else ref_pred)
        sm = compute_metrics(y_te, pt)
        boot_rows.append({**ctx, "feature_set": fs, "model": m, "reference_baseline": ref_name,
                          "seedmean_macro_f1": sm["macro_f1"], "seedmean_severe_f1": sm["severe_f1"], **b})
        cm = np.array([[sm[f"cm_{i}{j}"] for j in range(3)] for i in range(3)])
        seed_mean[(fs, m)] = (pv, pt, cm)
    return runs, pd.DataFrame(boot_rows), seed_mean


def summarise(runs: pd.DataFrame, boots: pd.DataFrame) -> pd.DataFrame:
    g = runs.groupby(["split", "fold", "horizon", "feature_set", "model", "kind"])
    agg = g[[k for k in KEY if k in runs]].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg["n_seeds"] = g.size()
    agg = agg.reset_index()
    return agg.merge(boots, on=["split", "fold", "horizon", "feature_set", "model"], how="left")


def fmt(mean, std, lo=None, hi=None):
    if pd.isna(mean):
        return ""
    s = f"{mean:.3f}"
    if not pd.isna(std) and std > 0:
        s += f" ± {std:.3f}"
    if lo is not None and not pd.isna(lo):
        s += f" [{lo:.3f}, {hi:.3f}]"
    return s


def write_report(split: str, summ: pd.DataFrame, confusions: dict, path: Path):
    lines = [f"# Results — split: `{split}`\n",
             "Metrics on the untouched TEST period (2019+). Models: mean ± std over seeds; "
             "[95% block-bootstrap CI of the seed-averaged prediction; blocks = target month]. "
             "Δ vs ref = paired bootstrap CI of (model − strongest baseline by validation macro-F1); "
             "an interval entirely above 0 means the model significantly beats that baseline. "
             "Severe POD/FAR use a probability threshold tuned on VALIDATION only; not shown for "
             "climatology/majority (near-constant probabilities make a tuned threshold meaningless). "
             "Ensemble: test scores only (its weights are fitted on validation, so its validation "
             "score would be in-sample; it is never chosen as 'best by validation').\n"]
    for fold in summ.fold.unique():
        for h in sorted(summ.horizon.unique()):
            s = summ[(summ.fold == fold) & (summ.horizon == h)].copy()
            if s.empty:
                continue
            s["order"] = s.kind.map({"baseline": 0, "model": 1})
            s = s.sort_values(["order", "feature_set", "model"])
            ref = s.reference_baseline.dropna().iloc[0] if s.reference_baseline.notna().any() else "-"
            tbl = pd.DataFrame({
                "feature set": s.feature_set, "model": s.model,
                "macro-F1": [fmt(r.macro_f1_mean, r.macro_f1_std, r.get("macro_f1_ci_low"), r.get("macro_f1_ci_high"))
                             for _, r in s.iterrows()],
                f"Δ macro-F1 vs {ref}": [fmt(np.nan, np.nan) if pd.isna(r.get("d_macro_f1_ci_low"))
                                         else f"[{r.d_macro_f1_ci_low:+.3f}, {r.d_macro_f1_ci_high:+.3f}]"
                                         for _, r in s.iterrows()],
                "severe-F1": [fmt(r.severe_f1_mean, r.severe_f1_std) for _, r in s.iterrows()],
                "bal. acc": [fmt(r.balanced_accuracy_mean, r.balanced_accuracy_std) for _, r in s.iterrows()],
                "ROC-AUC": [fmt(r.roc_auc_macro_ovr_mean, r.roc_auc_macro_ovr_std) for _, r in s.iterrows()],
                "BSS vs clim": [fmt(r.bss_vs_climatology_mean, r.bss_vs_climatology_std) for _, r in s.iterrows()],
                "severe POD (tuned)": ["" if r.model in NO_SEVERE_WARNING else
                                       fmt(r.tuned_severe_pod_mean, r.tuned_severe_pod_std) for _, r in s.iterrows()],
                "severe FAR (tuned)": ["" if r.model in NO_SEVERE_WARNING else
                                       fmt(r.tuned_severe_far_mean, r.tuned_severe_far_std) for _, r in s.iterrows()],
                "val macro-F1": [fmt(r.val_macro_f1_mean, r.val_macro_f1_std) for _, r in s.iterrows()],
                "seeds": s.n_seeds,
            })
            lines.append(f"\n## {fold} — horizon {h} days\n")
            lines.append(to_md(tbl, index=False))
            models = s[s.kind == "model"]
            if not models.empty:
                best = models.sort_values("val_macro_f1_mean", ascending=False, na_position="last").iloc[0]
                lines.append(f"\nBest model by **validation** macro-F1: `{best.feature_set}/{best.model}` "
                             f"(test macro-F1 {best.macro_f1_mean:.3f}).")
                cm = confusions.get((fold, h))
                if cm is not None:
                    sev = cm[2]
                    lines.append(f"Severe-class analysis (seed-averaged argmax): of {sev.sum()} severe days, "
                                 f"{sev[2]} detected, {sev[1]} predicted Moderate, {sev[0]} predicted Healthy; "
                                 f"{cm[0, 2] + cm[1, 2]} false severe alarms.")
    # ablation view
    models = summ[summ.kind == "model"]
    if not models.empty and models.feature_set.nunique() > 1:
        lines.append("\n## Feature-set ablation (test macro-F1, mean over seeds)\n")
        for fold in models.fold.unique():
            piv = models[models.fold == fold].pivot_table(index=["horizon", "model"], columns="feature_set",
                                                          values="macro_f1_mean")
            lines.append(f"\n### {fold}\n\n" + to_md(piv.round(3)))
    if summ.fold.nunique() > 1 and not models.empty:
        lines.append("\n## Mean over spatial folds (test macro-F1)\n")
        piv = models.groupby(["horizon", "feature_set", "model"])[["macro_f1_mean", "severe_f1_mean"]].mean()
        lines.append(to_md(piv.round(3)))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", nargs="*", default=None)
    ap.add_argument("--n-boot", type=int, default=config.BOOTSTRAP_N)
    ap.add_argument("--quick", action="store_true", help="aggregate runs/_quick")
    args = ap.parse_args()

    runs_root = config.RUNS_DIR / "_quick" if args.quick else config.RUNS_DIR
    out_dir = config.TABLES_DIR / "_quick" if args.quick else config.TABLES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = args.split or [p.name for p in runs_root.iterdir() if p.is_dir() and not p.name.startswith("_")]

    for split in splits:
        all_runs, all_boot, confusions = [], [], {}
        for fold_dir in sorted(p for p in (runs_root / split).iterdir() if p.is_dir()):
            for base in sorted(fold_dir.glob("h*")):
                if not (base / "targets.npz").exists():
                    continue
                logger.info(f"Aggregating {split}/{fold_dir.name}/{base.name}")
                runs, boots, seed_mean = aggregate_base(split, fold_dir.name, base, args.n_boot)
                all_runs.append(runs); all_boot.append(boots)
                models = runs[runs.kind == "model"]
                if not models.empty:
                    best = models.groupby(["feature_set", "model"]).val_macro_f1.mean().idxmax()
                    cm = seed_mean[best][2]
                    confusions[(fold_dir.name, int(base.name[1:]))] = cm
                    pd.DataFrame(cm, index=[f"true_{c}" for c in config.DROUGHT_CLASSES],
                                 columns=[f"pred_{c}" for c in config.DROUGHT_CLASSES]) \
                        .to_csv(out_dir / f"confusion_{split}_{fold_dir.name}_{base.name}_{best[0]}_{best[1]}.csv")
        if not all_runs:
            logger.warning(f"No runs found for split {split}")
            continue
        runs = pd.concat(all_runs, ignore_index=True)
        boots = pd.concat(all_boot, ignore_index=True)
        runs.to_csv(out_dir / f"all_runs_{split}.csv", index=False)
        summ = summarise(runs, boots)
        summ.to_csv(out_dir / f"summary_{split}.csv", index=False)
        write_report(split, summ, confusions, out_dir / f"REPORT_{split}.md")
        logger.info(f"[{split}] wrote {out_dir / f'REPORT_{split}.md'}")


if __name__ == "__main__":
    main()

"""
run_experiments.py — train/evaluate models for the v3 drought forecasting study.

Covers plan steps 4–8:
  * genuine future forecasting at 7 / 15 / 30 days
  * persistence / markov / climatology / majority baselines
  * feature-set ablation: weather | weather_spei | spei_only
  * multiple seeds (predictions saved for bootstrap + ensembling later)
  * split modes: temporal (pooled) | loco (leave-one-cluster-out) | loso

Every run writes   runs/<split>/<fold>/h<h>/<feature_set>/<model>/seed<s>/{preds.npz, run.json}
Every (split, fold, horizon) writes targets.npz, test_meta.csv, split_summary.csv, baselines/
A config snapshot (config values + CLI args + git commit) is written per invocation.
Finished runs are skipped unless --overwrite, so an interrupted job can be resumed.

Examples:
  python src/experiments/run_experiments.py --quick                      # smoke test (minutes)
  python src/experiments/run_experiments.py --split temporal             # main study
  python src/experiments/run_experiments.py --split loco --feature-sets weather weather_spei
Then:
  python src/experiments/aggregate.py
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from features.sequence_builder import build_sequences_all_fields, load_prepared
from features.splits import describe_masks, make_masks, spatial_folds
from models.baselines import all_baselines
from models.trainers import TRAINERS, DEVICE
from utils.logger import get_logger

logger = get_logger("run_experiments")


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=config.PROJECT_ROOT,
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="temporal", choices=["temporal", "loco", "loso"])
    ap.add_argument("--horizons", type=int, nargs="+", default=config.HORIZONS)
    ap.add_argument("--feature-sets", nargs="+", default=config.DEFAULT_FEATURE_SETS,
                    choices=list(config.FEATURE_SETS))
    ap.add_argument("--models", nargs="+", default=config.MODELS, choices=list(TRAINERS))
    ap.add_argument("--seeds", type=int, nargs="+", default=config.SEEDS)
    ap.add_argument("--quick", action="store_true",
                    help="smoke test: 5k train samples, tiny models, 2 epochs, writes to runs/_quick")
    ap.add_argument("--overwrite", action="store_true", help="re-run finished runs")
    ap.add_argument("--save-models", action="store_true", help="save fitted models for the first seed")
    return ap.parse_args()


def main():
    args = parse_args()
    runs_root = config.RUNS_DIR / "_quick" if args.quick else config.RUNS_DIR
    split_root = runs_root / args.split
    split_root.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fh = logging.FileHandler(config.LOGS_DIR / f"run_experiments_{args.split}_{ts}.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s"))
    logger.addHandler(fh)

    snap = {"timestamp": ts, "args": vars(args), "git_commit": git_commit(), "device": str(DEVICE),
            "torch": torch.__version__, "cuda": torch.cuda.is_available(), "config": config.snapshot()}
    with open(split_root / f"config_snapshot_{ts}.json", "w", encoding="utf-8") as f:
        json.dump(snap, f, indent=2, default=str)
    logger.info(f"Split={args.split} horizons={args.horizons} feature_sets={args.feature_sets} "
                f"models={args.models} seeds={args.seeds} device={DEVICE} quick={args.quick}")

    frames = load_prepared()
    folds = spatial_folds(args.split)
    n_total = len(args.horizons) * len(args.feature_sets) * len(folds) * len(args.models) * len(args.seeds)
    done = 0
    t_start = time.time()

    for h in args.horizons:
        for fs in args.feature_sets:
            X, y, meta = build_sequences_all_fields(frames, fs, h)
            for fold, tr_fields, te_fields in folds:
                masks = make_masks(meta, tr_fields, te_fields)
                base = split_root / fold / f"h{h}"
                base.mkdir(parents=True, exist_ok=True)

                # targets + test meta: identical across feature sets; write once, verify after
                tgt_path = base / "targets.npz"
                if tgt_path.exists():
                    old = np.load(tgt_path)
                    if not (np.array_equal(old["y_val"], y[masks["val"]]) and
                            np.array_equal(old["y_test"], y[masks["test"]])):
                        raise RuntimeError(f"{base}: targets differ between feature sets / runs. "
                                           f"Delete {base} (data changed?) and re-run.")
                else:
                    np.savez_compressed(tgt_path, y_val=y[masks["val"]], y_test=y[masks["test"]])
                    meta.loc[masks["test"], ["field", "cluster", "last_input_date", "target_date", "label_t"]] \
                        .to_csv(base / "test_meta.csv", index=False)
                    describe_masks(meta, y, masks).to_csv(base / "split_summary.csv", index=False)
                    with open(base / "fold.json", "w") as f:
                        json.dump({"fold": fold, "train_fields": tr_fields, "test_fields": te_fields}, f, indent=2)

                # baselines (feature-set independent)
                bdir = base / "baselines"
                if not (bdir / "persistence" / "preds.npz").exists() or args.overwrite:
                    bl = all_baselines(meta[masks["train"]], y[masks["train"]],
                                       meta[masks["val"]], meta[masks["test"]])
                    for name, (pv, pt) in bl.items():
                        (bdir / name).mkdir(parents=True, exist_ok=True)
                        np.savez_compressed(bdir / name / "preds.npz", p_val=pv, p_test=pt)
                    logger.info(f"  [{fold} h={h}] baselines test macro-F1: " + ", ".join(
                        f"{n}={f1_score(y[masks['test']], pt.argmax(1), average='macro'):.3f}"
                        for n, (_, pt) in bl.items()))

                tr_idx = np.flatnonzero(masks["train"])
                if args.quick and len(tr_idx) > 5000:
                    tr_idx = np.random.default_rng(0).choice(tr_idx, 5000, replace=False)
                data = {"X_tr": X[tr_idx], "y_tr": y[tr_idx],
                        "X_val": X[masks["val"]], "y_val": y[masks["val"]],
                        "X_te": X[masks["test"]]}
                y_te = y[masks["test"]]

                for model_name in args.models:
                    for seed in args.seeds:
                        done += 1
                        run_dir = base / fs / model_name / f"seed{seed}"
                        if (run_dir / "run.json").exists() and not args.overwrite:
                            logger.info(f"  skip (done) {run_dir.relative_to(runs_root)}")
                            continue
                        run_dir.mkdir(parents=True, exist_ok=True)
                        model, p_val, p_te, info = TRAINERS[model_name](data, seed, 3, args.quick)
                        np.savez_compressed(run_dir / "preds.npz", p_val=p_val.astype(np.float32),
                                            p_test=p_te.astype(np.float32))
                        val_f1 = f1_score(data["y_val"], p_val.argmax(1), average="macro", zero_division=0)
                        te_f1 = f1_score(y_te, p_te.argmax(1), average="macro", zero_division=0)
                        te_sev = f1_score(y_te, p_te.argmax(1), labels=[2], average="macro", zero_division=0)
                        rec = {"split": args.split, "fold": fold, "horizon": h, "feature_set": fs,
                               "model": model_name, "seed": seed, "features": config.FEATURE_SETS[fs],
                               "n_train": int(len(tr_idx)), "n_val": int(masks["val"].sum()),
                               "n_test": int(masks["test"].sum()), "train_fields": tr_fields,
                               "test_fields": te_fields, "val_macro_f1": round(val_f1, 4),
                               "test_macro_f1_argmax": round(te_f1, 4), "test_severe_f1_argmax": round(te_sev, 4),
                               "quick": args.quick, "finished": datetime.now().isoformat(timespec="seconds"),
                               **info}
                        with open(run_dir / "run.json", "w") as f:
                            json.dump(rec, f, indent=2)
                        if args.save_models and seed == args.seeds[0]:
                            mdir = config.MODELS_DIR / args.split / fold / f"h{h}" / fs
                            mdir.mkdir(parents=True, exist_ok=True)
                            if model_name in ("rf", "xgb"):
                                joblib.dump(model, mdir / f"{model_name}.pkl")
                            else:
                                torch.save(model.state_dict(), mdir / f"{model_name}.pt")
                                joblib.dump(model.scaler, mdir / f"{model_name}_scaler.pkl")
                        el = time.time() - t_start
                        logger.info(f"  [{done}/{n_total}] {fold} h={h} {fs:12s} {model_name:11s} seed={seed} "
                                    f"valF1={val_f1:.3f} testF1={te_f1:.3f} sevF1={te_sev:.3f} "
                                    f"({info['train_time_s']}s, elapsed {el / 60:.1f} min)")
    logger.info(f"Finished. Now run: python src/experiments/aggregate.py{' --quick' if args.quick else ''}")


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

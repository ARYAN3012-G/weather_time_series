"""
prepare_data.py — Step 1 + Step 2 of the v3 pipeline (no model training).

  1. Site audit: coordinates, ERA5-Land grid cell (0.1°), pairwise identity of
     weather features -> results/v3/audit/
  2. Remove duplicate site(s) listed in config.DUPLICATE_SITES (verified by the audit).
  3. Recompute SPEI with frozen calibration fitted on 1979–2011 only
     -> data/prepared/<site>.csv  and  results/v3/spei_calibration.json
  4. Old-vs-corrected SPEI / label comparison report -> results/v3/spei_report/
  5. Archive v2 result files -> results/archive/v2_pre_correction/

Usage:
    python src/dataprep/prepare_data.py
"""
from __future__ import annotations

import itertools
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from dataprep.spei import SPEICalibrator, accumulate, water_balance
from utils.logger import get_logger
from utils.report import to_md

logger = get_logger("prepare_data")

ERA5_LAND_RES = 0.1


def period_of_year(year):
    year = np.asarray(year)
    return np.where(year <= config.TRAIN_END_YEAR, "train",
                    np.where(year <= config.VAL_END_YEAR, "val", "test"))


# ---------------------------------------------------------------------------
# 1. Load + audit
# ---------------------------------------------------------------------------
def load_raw() -> dict[str, pd.DataFrame]:
    frames = {}
    for name in config.ALL_RAW_FIELDS:
        path = config.RAW_WEATHER_DIR / f"{name}_with_spei.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing raw file: {path}")
        df = pd.read_csv(path, parse_dates=[config.DATE_COLUMN])
        df = df.sort_values(config.DATE_COLUMN).drop_duplicates(config.DATE_COLUMN).reset_index(drop=True)
        df["field"] = name
        frames[name] = df
    logger.info(f"Loaded {len(frames)} raw sites from {config.RAW_WEATHER_DIR}")
    return frames


def audit_sites(frames: dict) -> pd.DataFrame:
    rows = []
    for n, df in frames.items():
        lat, lon = float(df.latitude.iloc[0]), float(df.longitude.iloc[0])
        gaps = int((df[config.DATE_COLUMN].diff().dt.days.dropna() != 1).sum())
        rows.append({"field": n, "latitude": round(lat, 4), "longitude": round(lon, 4),
                     "era5_land_cell": f"({round(lat / ERA5_LAND_RES) * ERA5_LAND_RES:.1f}, "
                                       f"{round(lon / ERA5_LAND_RES) * ERA5_LAND_RES:.1f})",
                     "rows": len(df), "start": df[config.DATE_COLUMN].min().date(),
                     "end": df[config.DATE_COLUMN].max().date(), "date_gaps": gaps,
                     "missing_weather_values": int(df[config.WEATHER_FEATURES].isna().sum().sum())})
    sites = pd.DataFrame(rows)
    sites.to_csv(config.AUDIT_DIR / "site_audit.csv", index=False)

    pair_rows = []
    for a, b in itertools.combinations(frames, 2):
        A = frames[a].set_index(config.DATE_COLUMN)
        B = frames[b].set_index(config.DATE_COLUMN)
        idx = A.index.intersection(B.index)
        rec = {"site_a": a, "site_b": b, "overlap_days": len(idx)}
        same_feat = []
        for c in config.WEATHER_FEATURES:
            eq = np.isclose(A.loc[idx, c].to_numpy(), B.loc[idx, c].to_numpy(), rtol=1e-6, atol=1e-9)
            rec[f"identical_{c}"] = round(float(eq.mean()), 4)
            same_feat.append(eq.mean())
        rec["n_features_identical_99pct"] = int(sum(s >= 0.99 for s in same_feat))
        rec["precip_corr"] = round(float(A.loc[idx, "Precipitation_Flux"].corr(B.loc[idx, "Precipitation_Flux"])), 4)
        old = A.loc[idx, "label"].notna() & B.loc[idx, "label"].notna()   # v2 labels (absent for days added later)
        rec["label_agreement"] = round(float((A.loc[idx, "label"][old] == B.loc[idx, "label"][old]).mean()), 4)
        rec["same_era5_cell"] = (sites.set_index("field").loc[a, "era5_land_cell"]
                                 == sites.set_index("field").loc[b, "era5_land_cell"])
        pair_rows.append(rec)
    pairs = pd.DataFrame(pair_rows)
    pairs.to_csv(config.AUDIT_DIR / "site_pairs.csv", index=False)

    detected = pairs[(pairs.n_features_identical_99pct >= len(config.WEATHER_FEATURES) - 1)]
    detected_set = {tuple(sorted((r.site_a, r.site_b))) for r in detected.itertuples()}
    configured = {tuple(sorted((d, c))) for d, c in config.DUPLICATE_SITES.items()}
    logger.info(f"Duplicate pairs detected by audit: {sorted(detected_set)}")
    if detected_set != configured:
        raise RuntimeError(f"Audit duplicates {detected_set} != config.DUPLICATE_SITES {configured}. "
                           f"Update config before continuing.")

    with open(config.AUDIT_DIR / "AUDIT.md", "w", encoding="utf-8") as f:
        f.write("# Site audit\n\n")
        f.write(to_md(sites, index=False) + "\n\n")
        f.write("## Duplicate weather locations\n\n")
        for d, c in config.DUPLICATE_SITES.items():
            r = pairs[((pairs.site_a == d) & (pairs.site_b == c)) | ((pairs.site_a == c) & (pairs.site_b == d))].iloc[0]
            f.write(f"- **{d}** and **{c}** map to the same ERA5-Land grid cell "
                    f"(same_era5_cell={r.same_era5_cell}); {r.n_features_identical_99pct}/9 weather features "
                    f"identical on >=99% of days; label agreement {r.label_agreement:.4f}. "
                    f"`{d}` is removed as an independent weather location; `{c}` is kept "
                    f"(field metadata records that it also represents `{d}`).\n")
        f.write("\nThe two agricultural fields may be physically distinct; they are not distinct "
                "in the ERA5-Land weather data used here.\n\n")
        f.write("## Pairwise summary\n\n")
        f.write(to_md(pairs[["site_a", "site_b", "same_era5_cell", "n_features_identical_99pct",
                             "precip_corr", "label_agreement"]], index=False) + "\n")
    logger.info(f"Audit written -> {config.AUDIT_DIR}")
    return pairs


# ---------------------------------------------------------------------------
# 2–3. Recompute SPEI with frozen training-period calibration
# ---------------------------------------------------------------------------
def recompute_spei(df: pd.DataFrame) -> tuple[pd.DataFrame, SPEICalibrator, dict]:
    df = df.copy()
    wb = water_balance(df)
    d_acc = accumulate(wb, config.SPEI_SCALE_DAYS)
    # The raw files start mid-record; their first rows' WB30 already includes earlier
    # history. Keep that history instead of discarding it: verify agreement where
    # both exist, then fill the first (scale-1) rows from the provided WB30.
    both = d_acc.notna() & df["WB30"].notna()
    max_diff = float((d_acc[both] - df.loc[both, "WB30"]).abs().max())
    if max_diff > 1e-3 * max(1.0, float(df["WB30"].abs().max())):
        raise RuntimeError(f"[{df.field.iloc[0]}] recomputed WB30 disagrees with provided WB30 (max diff {max_diff})")
    d_acc = d_acc.fillna(df["WB30"])

    cal = SPEICalibrator(config.SPEI_SCALE_DAYS, config.SPEI_CALIB_START,
                         config.SPEI_CALIB_END, config.SPEI_CLIP)
    cal.fit(df[config.DATE_COLUMN], d_acc)
    spei_new = cal.transform(df[config.DATE_COLUMN], d_acc)

    out = df[[config.DATE_COLUMN, "field", "latitude", "longitude"] + config.WEATHER_FEATURES].copy()
    out["cluster"] = config.FIELD_TO_CLUSTER[df.field.iloc[0]]
    out["Water_Balance"] = wb
    out["WB30"] = d_acc
    out["SPEI"] = spei_new
    out["label"] = config.classify_spei(spei_new)
    out["SPEI_old"] = df["SPEI"]
    out["label_old"] = df["label"].astype("Int64")   # empty for days appended by tools/update_agera5.py
    out["period"] = period_of_year(out[config.DATE_COLUMN].dt.year)
    info = {"wb30_check_max_abs_diff": max_diff,
            "fallback_months": [m for m, p in cal.params_.items() if p["method"] != "loglogistic"]}
    return out, cal, info


# ---------------------------------------------------------------------------
# 4. Old vs corrected report
# ---------------------------------------------------------------------------
def spei_report(prepared: dict[str, pd.DataFrame]) -> None:
    allp = pd.concat(prepared.values(), ignore_index=True)
    allp = allp[allp.label_old.notna()].copy()   # old-vs-new comparison only where a v2 label exists
    allp["label_old"] = allp["label_old"].astype(int)

    # per site / period SPEI moments
    rows = []
    for (f, p), g in allp.groupby(["field", "period"]):
        rows.append({"field": f, "period": p, "days": len(g),
                     "spei_old_mean": g.SPEI_old.mean(), "spei_old_std": g.SPEI_old.std(),
                     "spei_new_mean": g.SPEI.mean(), "spei_new_std": g.SPEI.std(),
                     "corr_old_new": g.SPEI.corr(g.SPEI_old),
                     "label_agreement": (g.label == g.label_old).mean()})
    moments = pd.DataFrame(rows).round(4)
    moments.to_csv(config.SPEI_REPORT_DIR / "spei_moments_by_site_period.csv", index=False)

    # class distribution by period (days, pooled over the 7 sites)
    dist_rows = []
    for p in ["train", "val", "test"]:
        g = allp[allp.period == p]
        for ver, col in [("old", "label_old"), ("new", "label")]:
            c = np.bincount(g[col], minlength=3)
            dist_rows.append({"period": p, "version": ver, "days": len(g),
                              **{f"{config.INVERSE_LABEL_MAP[i]}_pct": round(100 * c[i] / len(g), 2)
                                 for i in range(3)}})
    dist = pd.DataFrame(dist_rows)
    dist.to_csv(config.SPEI_REPORT_DIR / "class_distribution_old_vs_new.csv", index=False)

    # transition matrix old label -> new label (test period, where leakage mattered most)
    trans = {}
    for p in ["train", "val", "test"]:
        g = allp[allp.period == p]
        ct = pd.crosstab(g.label_old.map(config.INVERSE_LABEL_MAP), g.label.map(config.INVERSE_LABEL_MAP),
                         rownames=["old"], colnames=["new"])
        ct.to_csv(config.SPEI_REPORT_DIR / f"label_transition_{p}.csv")
        trans[p] = ct

    with open(config.SPEI_REPORT_DIR / "SPEI_REPORT.md", "w", encoding="utf-8") as f:
        f.write("# SPEI recalibration report (v2 -> v3)\n\n")
        f.write(f"- Accumulation: {config.SPEI_SCALE_DAYS}-day rolling sum of P - PET (FAO-56)\n")
        f.write(f"- Distribution: 3-parameter log-logistic, L-moments, per site and calendar month\n")
        f.write(f"- Reference (calibration) period: {config.SPEI_CALIB_START}–{config.SPEI_CALIB_END} "
                f"(= TRAIN years). Parameters frozen in `results/v3/spei_calibration.json` and applied "
                f"unchanged to validation (2012–2018) and test (2019+).\n")
        f.write("- v2 SPEI: per-calendar-month standardisation fitted on the full 1979–2026 record "
                "(inferred: original preprocessing script is not in the repository). Mean v2 SPEI is ~0 "
                "over the whole record, i.e. test years influenced the calibration.\n\n")
        f.write("## Class distribution (% of days, 7 sites pooled)\n\n")
        f.write(to_md(dist, index=False) + "\n\n")
        f.write("## Mean SPEI by period (7 sites pooled)\n\n")
        f.write(to_md(allp.groupby("period")[["SPEI_old", "SPEI"]].agg(["mean", "std"]).round(3)
                      .reindex(["train", "val", "test"])) + "\n\n")
        f.write("## Label agreement old vs new\n\n")
        agree = (allp.label == allp.label_old).groupby(allp.period).mean()
        f.write(to_md(agree.reindex(["train", "val", "test"]).rename("agreement")) + "\n\n")
        for p, ct in trans.items():
            f.write(f"## Label transitions old -> new ({p})\n\n{to_md(ct)}\n\n")
        f.write("## Per site and period\n\n" + to_md(moments, index=False) + "\n")
    logger.info(f"SPEI report written -> {config.SPEI_REPORT_DIR}")
    logger.info("\n" + dist.to_string(index=False))


# ---------------------------------------------------------------------------
# 5. Archive v2 results
# ---------------------------------------------------------------------------
def archive_v2_results() -> None:
    results_root = config.PROJECT_ROOT / "results"
    old_files = [p for p in results_root.iterdir() if p.is_file()]
    if not old_files:
        return
    config.ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    for p in old_files:
        shutil.move(str(p), str(config.ARCHIVE_DIR / p.name))
    with open(config.ARCHIVE_DIR / "README.md", "w", encoding="utf-8") as f:
        f.write("# v2 results (pre-correction) — kept for reproducibility only\n\n"
                "Produced before the v3 corrections (duplicate site, SPEI calibrated on all years incl. "
                "test, 1-day task dominated by persistence, no purged split). Do not report these as "
                "final results; they must be re-evaluated with the v3 pipeline.\n")
    logger.info(f"Archived {len(old_files)} v2 result files -> {config.ARCHIVE_DIR}")


def main():
    archive_v2_results()
    frames = load_raw()
    audit_sites(frames)

    prepared, calib, infos = {}, {}, {}
    for name in config.FIELDS:
        out, cal, info = recompute_spei(frames[name])
        out.to_csv(config.PREPARED_DIR / f"{name}.csv", index=False)
        prepared[name], calib[name], infos[name] = out, cal.to_dict(), info
        logger.info(f"[{name}] SPEI recalibrated | fallback months: {info['fallback_months'] or 'none'} "
                    f"| label agreement with v2: {(out.label == out.label_old).mean():.3f}")

    # remove stale prepared files (e.g. a duplicate site written by an older run)
    for p in config.PREPARED_DIR.glob("*.csv"):
        if p.stem not in config.FIELDS:
            p.unlink()

    with open(config.SPEI_CALIBRATION_FILE, "w", encoding="utf-8") as f:
        json.dump({"method": config.SPEI_METHOD, "grouping": config.SPEI_GROUPING,
                   "sites": calib, "checks": infos,
                   "removed_duplicates": config.DUPLICATE_SITES}, f, indent=2, default=str)
    logger.info(f"Frozen calibration parameters -> {config.SPEI_CALIBRATION_FILE}")

    spei_report(prepared)
    logger.info("Data preparation complete.")


if __name__ == "__main__":
    main()

"""
Replace a site's raw weather file with the full AgERA5 record of the 0.1-degree cell that the
site's field centre lies in (used when the original file came from a different cell).

    python tools/set_site_cell.py <site> <lat> <lon> [--end 2026-11-02]

Downloads 1979-01-01 .. end from the CDS dataset "sis-agrometeorological-indicators-timeseries"
(same dataset, variables and column names as tools/update_agera5.py) and writes
data/processed/weather/<site>_with_spei.csv in the raw format. The previous file is moved to
data/processed/weather/archive_wrong_cell/. The old v2 columns (SPEI, drought_class, label) are
left empty: v3 recomputes SPEI with the frozen 1979-2011 calibration (prepare_data.py).
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import update_agera5 as U  # noqa: E402

RAW = U.RAW


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("site")
    ap.add_argument("lat", type=float)
    ap.add_argument("lon", type=float)
    ap.add_argument("--start", default="1979-01-01")
    ap.add_argument("--end", default="2026-11-02")
    a = ap.parse_args()

    path = RAW / f"{a.site}_with_spei.csv"
    old = pd.read_csv(path, nrows=5)
    cache = U.ROOT / "data" / "downloads" / "agera5"
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / f"cell_{a.lat}_{a.lon}_{a.start}_{a.end}.download"
    d = U.read_download(target) if target.exists() else U.download(a.lat, a.lon, a.start, a.end, target)
    d = d.rename(columns=U.VARS).sort_values("valid_time").reset_index(drop=True)
    if len(pd.date_range(d.valid_time.min(), d.valid_time.max())) != len(d) or d[U.CORE].isna().any().any():
        sys.exit("download has gaps or missing values - nothing written")
    d["latitude"], d["longitude"], d["field"] = d["latitude"].iloc[0], d["longitude"].iloc[0], a.site
    d["Water_Balance"] = d["Precipitation_Flux"] - d["ReferenceET_PenmanMonteith_FAO56"]
    d["WB30"] = d["Water_Balance"].rolling(30, min_periods=30).sum()
    d = d[d["WB30"].notna()].reset_index(drop=True)                       # same start rule as the originals
    for c in old.columns:
        if c not in d:
            d[c] = np.nan
    d = d[list(old.columns)]
    d["valid_time"] = d["valid_time"].dt.strftime("%Y-%m-%d")

    arch = RAW / "archive_wrong_cell"
    arch.mkdir(exist_ok=True)
    o = pd.read_csv(path, usecols=["latitude", "longitude"], nrows=1)
    shutil.move(str(path), str(arch / f"{a.site}_cell_{o.latitude.iloc[0]:.1f}_{o.longitude.iloc[0]:.1f}.csv"))
    d.to_csv(path, index=False)
    print(f"{a.site}: cell ({o.latitude.iloc[0]:.1f}, {o.longitude.iloc[0]:.1f}) -> ({a.lat}, {a.lon}); "
          f"{d.valid_time.iloc[0]} .. {d.valid_time.iloc[-1]}, {len(d)} days. Next: python src/dataprep/prepare_data.py")


if __name__ == "__main__":
    main()

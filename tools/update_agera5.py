"""
Extend the raw weather files with the latest AgERA5 days from the Copernicus Climate Data Store.

    python tools/update_agera5.py [--start 2026-06-01] [--end 2026-11-02] [--dry-run]

Source: CDS dataset "sis-agrometeorological-indicators-timeseries" (AgERA5 v2.0, 0.1 deg,
daily) - the same dataset the original files came from. Needs a free CDS account, the
dataset licence accepted, and the API key in ~/.cdsapirc (url + key), plus `pip install cdsapi`.

For each site in data/processed/weather/<site>_with_spei.csv:
  1. downloads all 13 variables for the file's own grid point, from --start to --end;
  2. checks that the days already in the file are reproduced (otherwise stops: the source changed);
  3. appends only the new days, with Water_Balance = P - ET0 and WB30 = 30-day sum.
The old v2 columns SPEI / drought_class / label are left empty for new days: v3 never uses them
(prepare_data.py recomputes SPEI with the frozen 1979-2011 calibration).
A backup of each original file is written to data/processed/weather/backup_<date>/.
"""
from __future__ import annotations

import argparse
import io
import shutil
import sys
import zipfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "processed" / "weather"
DATASET = "sis-agrometeorological-indicators-timeseries"

# CDS variable name -> column name in the raw files
VARS = {
    "10m_wind_speed_24_hour_mean": "Wind_Speed_10m_Mean_24h",
    "2m_temperature_24_hour_maximum": "Temperature_Air_2m_Max_24h",
    "2m_temperature_24_hour_mean": "Temperature_Air_2m_Mean_24h",
    "2m_temperature_24_hour_minimum": "Temperature_Air_2m_Min_24h",
    "derived_2m_relative_humidity_24_hour_maximum": "Derived_Relative_Humidity_2m_Max_24h",
    "derived_2m_relative_humidity_24_hour_minimum": "Derived_Relative_Humidity_2m_Min_24h",
    "precipitation_flux": "Precipitation_Flux",
    "reference_evapotranspiration": "ReferenceET_PenmanMonteith_FAO56",
    "solar_radiation_flux": "Solar_Radiation_Flux",
    "vapour_pressure_24_hour_mean": "Vapour_Pressure_Mean_24h",
    "vapour_pressure_deficit_at_maximum_temperature": "Vapour_Pressure_Deficit_at_Maximum_Temperature",
    "cloud_cover_24_hour_mean": "Cloud_Cover_Mean_24h",
    "2m_dewpoint_temperature_24_hour_mean": "Dew_Point_Temperature_2m_Mean_24h",
}
CORE = [c for c in VARS.values() if c not in ("Cloud_Cover_Mean_24h", "Dew_Point_Temperature_2m_Mean_24h")]


def read_download(path: Path) -> pd.DataFrame:
    """CDS returns one CSV, or a zip of CSVs (one per variable). Merge them on valid_time."""
    raw = path.read_bytes()
    tables = []
    if raw[:2] == b"PK":
        with zipfile.ZipFile(path) as z:
            for n in z.namelist():
                if n.endswith(".csv"):
                    tables.append(pd.read_csv(io.BytesIO(z.read(n))))
    else:
        tables.append(pd.read_csv(io.BytesIO(raw)))
    out = None
    for t in tables:
        t = t.drop(columns=[c for c in ("latitude", "longitude") if c in t and out is not None])
        out = t if out is None else out.merge(t, on="valid_time", how="outer")
    out["valid_time"] = pd.to_datetime(out["valid_time"])
    return out.sort_values("valid_time").reset_index(drop=True)


def download(lat: float, lon: float, start: str, end: str, target: Path) -> pd.DataFrame:
    import cdsapi
    req = {"variable": list(VARS), "location": {"latitude": round(lat, 2), "longitude": round(lon, 2)},
           "date": [f"{start}/{end}"], "data_format": "csv"}
    cdsapi.Client(quiet=True).retrieve(DATASET, req, str(target))
    return read_download(target)


def check_overlap(old: pd.DataFrame, new: pd.DataFrame, site: str) -> int:
    o = old.set_index("valid_time")
    n = new.set_index("valid_time")
    idx = o.index.intersection(n.index)
    worst = 0.0
    for c in CORE:
        a, b = o.loc[idx, c].to_numpy(float), n.loc[idx, c].to_numpy(float)
        worst = max(worst, float((np.abs(a - b) / np.maximum(np.abs(a), 1e-6))[np.abs(a - b) > 1e-3].max(initial=0)))
        # 0.5 %: some files store solar radiation with slightly different rounding (max seen 0.27 %)
        if not np.allclose(a, b, rtol=5e-3, atol=1e-3):
            bad = np.abs(a - b).max()
            sys.exit(f"[{site}] {c}: downloaded values differ from the existing file on overlapping "
                     f"days (max diff {bad:.4g}). The source changed - stopping, nothing written.")
    return len(idx), worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-06-01")
    ap.add_argument("--end", default="2026-11-02")
    ap.add_argument("--dry-run", action="store_true", help="download and check, but do not write")
    a = ap.parse_args()

    cache = ROOT / "data" / "downloads" / "agera5"
    cache.mkdir(parents=True, exist_ok=True)
    backup = RAW / f"backup_{date.today():%Y%m%d}"
    files = sorted(RAW.glob("*_with_spei.csv"))
    if not files:
        sys.exit(f"No raw files in {RAW}")
    for f in files:
        site = f.name.replace("_with_spei.csv", "")
        old = pd.read_csv(f, parse_dates=["valid_time"])
        lat, lon = float(old.latitude.iloc[0]), float(old.longitude.iloc[0])
        target = cache / f"{site}_{a.start}_{a.end}.download"
        new = read_download(target) if target.exists() else download(lat, lon, a.start, a.end, target)
        new = new.rename(columns=VARS)
        missing = [c for c in CORE if c not in new]
        if missing:
            sys.exit(f"[{site}] download is missing columns {missing}")
        n_overlap, worst = check_overlap(old, new, site)
        add = new[new.valid_time > old.valid_time.max()].copy()
        add = add.dropna(subset=CORE)
        days = pd.date_range(old.valid_time.max() + pd.Timedelta(days=1), add.valid_time.max()) \
            if len(add) else []
        if len(add) != len(days):
            sys.exit(f"[{site}] new days are not continuous ({len(add)} rows for {len(days)} days)")
        add["latitude"], add["longitude"], add["field"] = old.latitude.iloc[0], old.longitude.iloc[0], old.field.iloc[0]
        for c in old.columns:
            if c not in add:
                add[c] = np.nan
        merged = pd.concat([old, add[old.columns]], ignore_index=True)
        new_rows = merged.index >= len(old)
        merged.loc[new_rows, "Water_Balance"] = (merged["Precipitation_Flux"]
                                                 - merged["ReferenceET_PenmanMonteith_FAO56"])[new_rows]
        wb30 = merged["Water_Balance"].rolling(30, min_periods=30).sum()
        merged.loc[new_rows, "WB30"] = wb30[new_rows]
        print(f"{site:26s} overlap {n_overlap:3d} days match (max diff {100 * worst:.2f}%) | {old.valid_time.max().date()} -> "
              f"{merged.valid_time.max().date()} (+{len(add)} days) | "
              f"P Jun-Oct 2026 = {merged.loc[merged.valid_time.between('2026-06-01', '2026-10-31'), 'Precipitation_Flux'].sum():.0f} mm")
        if not a.dry_run and len(add):
            backup.mkdir(exist_ok=True)
            if not (backup / f.name).exists():
                shutil.copy2(f, backup / f.name)
            # append only the new rows, so existing lines stay byte-identical
            out = merged[new_rows].copy()
            out["valid_time"] = out["valid_time"].dt.strftime("%Y-%m-%d")
            text = f.read_bytes()
            nl = b"\r\n" if b"\r\n" in text[:5000] else b"\n"
            body = out.to_csv(index=False, header=False, lineterminator=nl.decode()).encode()
            f.write_bytes(text + (b"" if text.endswith(nl) else nl) + body)
    print("Dry run - nothing written." if a.dry_run else "Raw files updated. Next: python src/dataprep/prepare_data.py")


if __name__ == "__main__":
    main()

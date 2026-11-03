"""
config.py — Single source of truth for the v3 (corrected) drought forecasting pipeline.

What changed from v2 (see README "Corrections in v3"):
  1. Duplicate site removed: lepakshi falls in the same ERA5-Land grid cell (0.1°)
     as cholasamudram, so the two are not independent weather locations.
  2. SPEI recalibrated: log-logistic (L-moments) per site and calendar month,
     fitted on the TRAIN reference period 1979–2011 only, parameters frozen.
  3. Genuine future forecasting at 7 / 15 / 30 days (1-day dropped as a headline task).
  4. Purged chronological split: a sample whose span (input start .. target date)
     crosses a split boundary is dropped, so no two splits share days.
  5. Baselines, feature-set ablation, multi-seed runs, block bootstrap CIs,
     leave-one-cluster-out, Marusov et al. protocol, severe-class thresholds.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT    = Path(__file__).resolve().parent.parent
RAW_WEATHER_DIR = PROJECT_ROOT / "data" / "processed" / "weather"  # original v2 CSVs (read-only input)
PREPARED_DIR    = PROJECT_ROOT / "data" / "prepared"               # v3 corrected per-site CSVs
RESULTS_DIR     = PROJECT_ROOT / "results" / "v3"
RUNS_DIR        = RESULTS_DIR / "runs"                              # per-run predictions + configs
TABLES_DIR      = RESULTS_DIR / "tables"
AUDIT_DIR       = RESULTS_DIR / "audit"
SPEI_REPORT_DIR = RESULTS_DIR / "spei_report"
MARUSOV_DIR     = RESULTS_DIR / "marusov_protocol"
ARCHIVE_DIR     = PROJECT_ROOT / "results" / "archive" / "v2_pre_correction"
MODELS_DIR      = PROJECT_ROOT / "models" / "v3"
LOGS_DIR        = PROJECT_ROOT / "logs"
SPEI_CALIBRATION_FILE = RESULTS_DIR / "spei_calibration.json"

for _d in (PREPARED_DIR, RESULTS_DIR, RUNS_DIR, TABLES_DIR, AUDIT_DIR,
           SPEI_REPORT_DIR, MARUSOV_DIR, MODELS_DIR, LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

DATE_COLUMN = "valid_time"

# ---------------------------------------------------------------------------
# Sites
# ---------------------------------------------------------------------------
ALL_RAW_FIELDS = [
    "cholasamudram", "kurnool_balaji_villas", "kurnool_dinedevarapadu",
    "kurnool_pulliah", "lepakshi", "Peddakadubur", "sira", "Somandepalle",
]

# duplicate -> canonical site kept. Verified in prepare_data.py audit:
# same ERA5-Land grid cell (13.8N, 77.6E); 8/9 weather features identical on 100% of days.
DUPLICATE_SITES = {"lepakshi": "cholasamudram"}

FIELDS = [f for f in ALL_RAW_FIELDS if f not in DUPLICATE_SITES]   # 7 independent sites

FIELD_METADATA = {
    "cholasamudram":          {"district": "Sri Sathya Sai", "state": "Andhra Pradesh",
                               "also_represents": ["lepakshi"]},
    "kurnool_balaji_villas":  {"district": "Kurnool",        "state": "Andhra Pradesh"},
    "kurnool_dinedevarapadu": {"district": "Kurnool",        "state": "Andhra Pradesh"},
    "kurnool_pulliah":        {"district": "Kurnool",        "state": "Andhra Pradesh"},
    "Peddakadubur":           {"district": "Kurnool",        "state": "Andhra Pradesh"},
    "sira":                   {"district": "Tumkur",         "state": "Karnataka"},
    "Somandepalle":           {"district": "Sri Sathya Sai", "state": "Andhra Pradesh"},
}

# Spatial clusters for leave-one-cluster-out (from label/precip correlation audit)
CLUSTERS = {
    "kurnool": ["Peddakadubur", "kurnool_balaji_villas", "kurnool_dinedevarapadu", "kurnool_pulliah"],
    "south":   ["Somandepalle", "cholasamudram", "sira"],
}
FIELD_TO_CLUSTER = {f: c for c, fs in CLUSTERS.items() for f in fs}
assert sorted(FIELD_TO_CLUSTER) == sorted(FIELDS)

# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
WEATHER_FEATURES = [
    "Wind_Speed_10m_Mean_24h",
    "Temperature_Air_2m_Max_24h",
    "Temperature_Air_2m_Mean_24h",
    "Temperature_Air_2m_Min_24h",
    "Derived_Relative_Humidity_2m_Max_24h",
    "Derived_Relative_Humidity_2m_Min_24h",
    "Precipitation_Flux",
    "ReferenceET_PenmanMonteith_FAO56",
    "Solar_Radiation_Flux",
]

# Historical SPEI is allowed as input: it is only ever taken from days <= t
# (the last input day), which are known at forecast time. The target-day SPEI
# is never in the input window (enforced in sequence_builder + tests).
FEATURE_SETS = {
    "weather":      WEATHER_FEATURES,               # v2 setting
    "weather_spei": WEATHER_FEATURES + ["SPEI"],    # weather + historical SPEI
    "spei_only":    ["SPEI"],                       # index history only (Marusov-style)
}
MODEL_INPUT_FEATURES = WEATHER_FEATURES  # backwards-compatible alias

# ---------------------------------------------------------------------------
# SPEI calibration
# ---------------------------------------------------------------------------
SPEI_SCALE_DAYS   = 30            # accumulation window of climatic water balance (P - PET)
SPEI_CALIB_START  = 1979          # reference period = TRAIN years only
SPEI_CALIB_END    = 2011
SPEI_METHOD       = "loglogistic"  # log-logistic, L-moments (Vicente-Serrano et al. 2010)
SPEI_GROUPING     = "month"        # separate distribution per calendar month
SPEI_CLIP         = 4.0           # |SPEI| clipped to this value

# ---------------------------------------------------------------------------
# Drought classes
# ---------------------------------------------------------------------------
DROUGHT_CLASSES   = ["Healthy", "Moderate", "Severe"]
LABEL_MAP         = {c: i for i, c in enumerate(DROUGHT_CLASSES)}
INVERSE_LABEL_MAP = {i: c for c, i in LABEL_MAP.items()}
SEVERE            = LABEL_MAP["Severe"]
SPEI_THRESHOLDS   = {"Healthy": 0.0, "Moderate": -1.5}


def classify_spei(spei):
    """Vectorised: Healthy (>=0) | Moderate (-1.5 <= s < 0) | Severe (< -1.5)."""
    import numpy as np
    s = np.asarray(spei, dtype=float)
    out = np.where(s >= SPEI_THRESHOLDS["Healthy"], 0,
                   np.where(s >= SPEI_THRESHOLDS["Moderate"], 1, 2))
    return out.astype(int)


# ---------------------------------------------------------------------------
# Forecast setup
# ---------------------------------------------------------------------------
SEQUENCE_LENGTH_DAYS = 30
HORIZONS             = [7, 15, 30]   # days between last input day t and target day t+h

# Purged chronological split by year. A sample spans [input_start, target_date];
# it is kept only if the whole span lies inside one split period.
TRAIN_END_YEAR = 2011   # TRAIN 1979–2011
VAL_END_YEAR   = 2018   # VAL   2012–2018 ; TEST 2019–2026

# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------
SEEDS            = [0, 1, 2, 3, 4]
MODELS           = ["rf", "xgb", "lstm", "transformer"]
BOOTSTRAP_N      = 1000
BOOTSTRAP_SEED   = 12345
# Blocks = calendar month of the target date (all sites together), so the
# bootstrap respects both temporal autocorrelation and cross-site correlation.

# Marusov et al. (2024) protocol adaptation: binary drought, index history input.
# Their target is PDSI < -2; our SPEI analogue is SPEI < -1.0 ("moderate drought or worse").
MARUSOV_DROUGHT_THRESHOLD = -1.0

# ---------------------------------------------------------------------------
# Hyperparameters (unchanged from v2 so differences come from the corrections)
# ---------------------------------------------------------------------------
BATCH_SIZE              = 128
NUM_EPOCHS              = 100
EARLY_STOPPING_PATIENCE = 15
LEARNING_RATE           = 3e-4
WEIGHT_DECAY            = 0.01

RF_PARAMS = dict(n_estimators=300, class_weight="balanced", n_jobs=-1)
XGB_PARAMS = dict(n_estimators=1000, learning_rate=0.05, max_depth=7, subsample=0.8,
                  colsample_bytree=0.8, min_child_weight=3, gamma=0.1,
                  eval_metric="mlogloss", early_stopping_rounds=50, verbosity=0)

LSTM_HIDDEN_SIZE = 128
LSTM_NUM_LAYERS  = 2
LSTM_DROPOUT     = 0.35

TF_D_MODEL         = 128
TF_NHEAD           = 4
TF_NUM_LAYERS      = 2
TF_DROPOUT         = 0.2
TF_DIM_FEEDFORWARD = 256


def snapshot() -> dict:
    """All upper-case config values as a JSON-serialisable dict (saved with every run)."""
    out = {}
    for k, v in globals().items():
        if k.isupper():
            out[k] = str(v) if isinstance(v, Path) else v
    return out

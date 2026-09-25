"""
config.py — Single source of truth for the drought weather-time-series pipeline.

Two tasks:
  Task A (next-day):  predict drought label at day t+1  (1-day-ahead forecast)
  Task B (forecast):  predict drought label at day t+15 (15-day-ahead forecast)

Training approach:
  - Pooled site-agnostic model: data from all 8 sites is combined.
  - Site identity is NOT an input feature — models learn general
    weather-drought relationships, not site-specific patterns.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT          = Path(__file__).resolve().parent.parent
PROCESSED_WEATHER_DIR = PROJECT_ROOT / "data" / "processed" / "weather"
MODELS_DIR            = PROJECT_ROOT / "models"
RESULTS_DIR           = PROJECT_ROOT / "results"
LOGS_DIR              = PROJECT_ROOT / "logs"

for _d in (MODELS_DIR, RESULTS_DIR, LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 8 field sites (Puddur & Shilpa Township excluded — no processed data)
# ---------------------------------------------------------------------------
FIELDS = [
    "cholasamudram",
    "kurnool_balaji_villas",
    "kurnool_dinedevarapadu",
    "kurnool_pulliah",
    "lepakshi",
    "Peddakadubur",
    "sira",
    "Somandepalle",
]

FIELD_METADATA = {
    "cholasamudram":          {"district": "Sri Sathya Sai", "state": "Andhra Pradesh"},
    "kurnool_balaji_villas":  {"district": "Kurnool",        "state": "Andhra Pradesh"},
    "kurnool_dinedevarapadu": {"district": "Kurnool",        "state": "Andhra Pradesh"},
    "kurnool_pulliah":        {"district": "Kurnool",        "state": "Andhra Pradesh"},
    "lepakshi":               {"district": "Sri Sathya Sai", "state": "Andhra Pradesh"},
    "Peddakadubur":           {"district": "Kurnool",        "state": "Andhra Pradesh"},
    "sira":                   {"district": "Tumkur",         "state": "Karnataka"},
    "Somandepalle":           {"district": "Sri Sathya Sai", "state": "Andhra Pradesh"},
}

# ---------------------------------------------------------------------------
# Column names — 9 weather input features from ERA5-Land
# ---------------------------------------------------------------------------
DATE_COLUMN = "valid_time"

MODEL_INPUT_FEATURES = [
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

# ---------------------------------------------------------------------------
# Anti-leakage: these columns are used ONLY for label generation,
# NEVER as model inputs.
# ---------------------------------------------------------------------------
LEAKAGE_COLUMNS = ["Water_Balance", "WB30", "SPEI"]
assert not set(MODEL_INPUT_FEATURES) & set(LEAKAGE_COLUMNS)

# ---------------------------------------------------------------------------
# Drought classification (from pre-computed SPEI in CSVs)
# ---------------------------------------------------------------------------
DROUGHT_CLASSES   = ["Healthy", "Moderate", "Severe"]
LABEL_MAP         = {cls: i for i, cls in enumerate(DROUGHT_CLASSES)}
INVERSE_LABEL_MAP = {i: cls for cls, i in LABEL_MAP.items()}
SPEI_THRESHOLDS   = {"Healthy": 0.0, "Moderate": -1.5}

def classify_spei(spei_value: float) -> int:
    """Healthy (SPEI >= 0), Moderate (-1.5 <= SPEI < 0), Severe (SPEI < -1.5)."""
    if spei_value >= SPEI_THRESHOLDS["Healthy"]:
        return LABEL_MAP["Healthy"]
    elif spei_value >= SPEI_THRESHOLDS["Moderate"]:
        return LABEL_MAP["Moderate"]
    else:
        return LABEL_MAP["Severe"]

# ---------------------------------------------------------------------------
# Sequence / forecast settings (fixed — no sequence-length experiments)
# ---------------------------------------------------------------------------
SEQUENCE_LENGTH_DAYS = 30   # past 30 days as input window
FORECAST_HORIZON     = 15   # Task B: predict 15 days ahead

# ---------------------------------------------------------------------------
# Strict chronological train / validation / test split (by year)
#   TRAIN:       1979 – 2011  (33 years)
#   VALIDATION:  2012 – 2018  ( 7 years)
#   TEST:        2019 – 2025  ( 7 years)
# ---------------------------------------------------------------------------
TRAIN_END_YEAR = 2011
VAL_END_YEAR   = 2018

# ---------------------------------------------------------------------------
# Hyperparameters — General
# ---------------------------------------------------------------------------
RANDOM_SEED             = 42
BATCH_SIZE              = 128
NUM_EPOCHS              = 100
EARLY_STOPPING_PATIENCE = 15   # epochs without val Macro-F1 improvement
LEARNING_RATE           = 3e-4
WEIGHT_DECAY            = 0.01  # AdamW weight decay

# ---------------------------------------------------------------------------
# Hyperparameters — LSTM
# ---------------------------------------------------------------------------
LSTM_HIDDEN_SIZE = 128
LSTM_NUM_LAYERS  = 2
LSTM_DROPOUT     = 0.35

# ---------------------------------------------------------------------------
# Hyperparameters — Transformer
#   Architecture: 30×9 → Linear(9, d_model) → PosEnc → TransformerEncoder
#                 → MeanPool → LayerNorm → Linear(d_model, 3) → Softmax
# ---------------------------------------------------------------------------
TF_D_MODEL          = 128
TF_NHEAD            = 4
TF_NUM_LAYERS       = 2
TF_DROPOUT          = 0.2
TF_DIM_FEEDFORWARD  = 256

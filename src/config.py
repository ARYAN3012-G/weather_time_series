"""
config.py — Single source of truth for the drought weather-time-series pipeline.

Two tasks:
  Task A (current):  predict drought label at day t   (next day after window)
  Task B (forecast): predict drought label at day t+15 (15 days ahead)
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
# Column names
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

LEAKAGE_COLUMNS = ["Water_Balance", "WB30", "SPEI"]
assert not set(MODEL_INPUT_FEATURES) & set(LEAKAGE_COLUMNS)

# ---------------------------------------------------------------------------
# Drought classification
# ---------------------------------------------------------------------------
DROUGHT_CLASSES   = ["Healthy", "Moderate", "Severe"]
LABEL_MAP         = {cls: i for i, cls in enumerate(DROUGHT_CLASSES)}
INVERSE_LABEL_MAP = {i: cls for cls, i in LABEL_MAP.items()}
SPEI_THRESHOLDS   = {"Healthy": 0.0, "Moderate": -1.5}

def classify_spei(spei_value: float) -> int:
    if spei_value >= SPEI_THRESHOLDS["Healthy"]:
        return LABEL_MAP["Healthy"]
    elif spei_value >= SPEI_THRESHOLDS["Moderate"]:
        return LABEL_MAP["Moderate"]
    else:
        return LABEL_MAP["Severe"]

# ---------------------------------------------------------------------------
# Sequence / forecast settings
# ---------------------------------------------------------------------------
SEQUENCE_LENGTH_DAYS = 30   # past 30 days as input
FORECAST_HORIZON     = 15   # Task B: predict 15 days ahead

# ---------------------------------------------------------------------------
# Train/val/test split
# ---------------------------------------------------------------------------
TRAIN_FRACTION = 0.70
VAL_FRACTION   = 0.15
TEST_FRACTION  = 0.15

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
RANDOM_SEED             = 42
BATCH_SIZE              = 128
NUM_EPOCHS              = 100
EARLY_STOPPING_PATIENCE = 12
LEARNING_RATE           = 5e-4
LSTM_HIDDEN_SIZE        = 64
LSTM_NUM_LAYERS         = 2
LSTM_DROPOUT            = 0.45

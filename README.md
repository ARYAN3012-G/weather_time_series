# Weather Time Series — Drought Prediction

Hydrological drought prediction using ERA5-Land weather time series data across 8 agricultural field sites in the Rayalaseema region (Andhra Pradesh) and Karnataka.

## Overview

This module implements a **pooled site-agnostic** drought prediction pipeline. One model is trained on data from all 8 sites combined — site identity is **not** used as an input feature. The model learns generalizable weather-drought relationships across sites.

### Two Prediction Tasks

| Task | Description | Horizon |
|------|-------------|---------|
| **Task A** | Next-Day Drought Forecast | 1 day ahead |
| **Task B** | 15-Day-Ahead Drought Forecast | 15 days ahead |

Both tasks use the same 30-day input window of 9 weather features to predict a drought class at the target horizon.

### Three Drought Classes (from SPEI)

| Class | SPEI Range | Label |
|-------|-----------|-------|
| Healthy | SPEI ≥ 0.0 | 0 |
| Moderate | -1.5 ≤ SPEI < 0.0 | 1 |
| Severe | SPEI < -1.5 | 2 |

SPEI (Standardized Precipitation Evapotranspiration Index) is pre-computed in the source CSVs and used **only for label generation** — it is never an input feature (anti-leakage design).

---

## Input Features

9 ERA5-Land weather variables (daily):

| # | Feature | Description |
|---|---------|-------------|
| 1 | `Wind_Speed_10m_Mean_24h` | Mean wind speed at 10m |
| 2 | `Temperature_Air_2m_Max_24h` | Max air temperature at 2m |
| 3 | `Temperature_Air_2m_Mean_24h` | Mean air temperature at 2m |
| 4 | `Temperature_Air_2m_Min_24h` | Min air temperature at 2m |
| 5 | `Derived_Relative_Humidity_2m_Max_24h` | Max relative humidity at 2m |
| 6 | `Derived_Relative_Humidity_2m_Min_24h` | Min relative humidity at 2m |
| 7 | `Precipitation_Flux` | Daily precipitation |
| 8 | `ReferenceET_PenmanMonteith_FAO56` | Reference evapotranspiration (FAO-56) |
| 9 | `Solar_Radiation_Flux` | Incoming solar radiation |

### Anti-Leakage Columns (excluded from model input)

- `Water_Balance` — derived from precipitation and ET
- `WB30` — 30-day rolling water balance
- `SPEI` — used only for label generation

---

## Data

### 8 Field Sites

| Site | District | State |
|------|----------|-------|
| Cholasamudram | Sri Sathya Sai | Andhra Pradesh |
| Kurnool Balaji Villas | Kurnool | Andhra Pradesh |
| Kurnool Dinedevarapadu | Kurnool | Andhra Pradesh |
| Kurnool Pulliah | Kurnool | Andhra Pradesh |
| Lepakshi | Sri Sathya Sai | Andhra Pradesh |
| Peddakadubur | Kurnool | Andhra Pradesh |
| Sira | Tumkur | Karnataka |
| Somandepalle | Sri Sathya Sai | Andhra Pradesh |

### Temporal Coverage

1979–2026 (~47 years of daily data per site, ~17,300 rows each).

### Strict Chronological Train / Validation / Test Split

Splitting is based on the **target date year** (the date being predicted), not the input window dates.

| Split | Years | Samples | Percentage |
|-------|-------|---------|------------|
| **Train** | 1979 – 2011 | 95,952 | 69.4% |
| **Validation** | 2012 – 2018 | 20,456 | 14.8% |
| **Test** | 2019 – 2026 | 21,869 | 15.8% |

### Class Distribution per Split

| Split | Healthy | Moderate | Severe |
|-------|---------|----------|--------|
| Train | 46,965 (48%) | 41,027 (42%) | 7,960 (8%) |
| Validation | 9,643 (47%) | 9,140 (44%) | 1,673 (8%) |
| Test | 14,441 (66%) | 6,309 (28%) | 1,119 (5%) |

---

## Models

### 4 Individual Models + 1 Ensemble

| Model | Input Shape | Notes |
|-------|------------|-------|
| **Random Forest** | `(n, 270)` — flattened | `n_estimators=300`, `class_weight="balanced"` |
| **XGBoost** | `(n, 270)` — flattened | `n_estimators=1000`, early stopping on val, sample weights |
| **LSTM** | `(n, 30, 9)` — 3D sequence | 2-layer, hidden=128, dropout=0.35, class-weighted CE loss |
| **Transformer** | `(n, 30, 9)` — 3D sequence | d_model=128, 4 heads, 2 layers, mean pooling, class-weighted CE loss |
| **Ensemble** | — | Soft-voting, weights ∝ validation Macro-F1 |

- **RF / XGBoost**: Input is flattened from `(n, 30, 9)` to `(n, 270)` — 270 features.
- **LSTM / Transformer**: Input is the native 3D sequence `(n, 30, 9)`.

### Transformer Architecture

```
Input (30 × 9) → Linear(9, 128) → Positional Encoding → TransformerEncoder(2 layers, 4 heads)
→ Mean Pooling → LayerNorm → Linear(128, 3)
```

### Training Protocol

- **Feature scaling**: `StandardScaler` fitted on TRAIN only, applied to VAL and TEST.
- **Class imbalance**: Inverse-frequency weighted `CrossEntropyLoss` (LSTM/Transformer), `class_weight="balanced"` (RF), sample weights (XGBoost).
- **Early stopping**: Monitor validation Macro-F1, patience = 15 epochs, restore best checkpoint.
- **Optimizer**: AdamW, lr=3e-4, weight_decay=0.01.
- **No sequence-length experiments**: Fixed 30-day window only.

### Ensemble Methodology

Ensemble weights are proportional to each model's **validation Macro-F1**:

```
w = [val_f1_tf, val_f1_xgb, val_f1_lstm, val_f1_rf]
w /= w.sum()
ensemble_prob = w[0]*tf_prob + w[1]*xgb_prob + w[2]*lstm_prob + w[3]*rf_prob
```

- Weights are computed from validation performance and **frozen** before any test-set evaluation.
- No arbitrary minimum weight — a model with 0 validation F1 gets 0 ensemble weight.
- If all models have 0 F1, weights fall back to uniform (0.25 each).

### Model Selection

Best individual model is selected by **validation Macro-F1** (not test performance).

---

## Evaluation Metrics

- Accuracy
- Balanced Accuracy
- Macro Precision / Recall / F1
- Per-class F1 (Healthy, Moderate, Severe)
- ROC-AUC (One-vs-Rest, Macro average)
- Confusion Matrix (saved per task)

---

## Project Structure

```
weather_time_series/
├── data/
│   └── processed/
│       └── weather/          # 8 site CSVs with pre-computed SPEI
│           ├── cholasamudram_with_spei.csv
│           ├── kurnool_balaji_villas_with_spei.csv
│           ├── kurnool_dinedevarapadu_with_spei.csv
│           ├── kurnool_pulliah_with_spei.csv
│           ├── lepakshi_with_spei.csv
│           ├── Peddakadubur_with_spei.csv
│           ├── sira_with_spei.csv
│           └── Somandepalle_with_spei.csv
├── models/                   # Saved model checkpoints
├── results/                  # CSV results, confusion matrices, summary.json
├── src/
│   ├── config.py             # All constants, paths, hyperparameters
│   ├── features/
│   │   └── sequence_builder.py   # Sliding-window sequence generation
│   ├── training/
│   │   └── train.py          # Full training pipeline
│   └── utils/
│       └── logger.py         # Logging utility
├── requirements.txt
└── README.md
```

---

## Usage

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Training

```bash
python src/training/train.py
```

### Output Files

| File | Description |
|------|-------------|
| `results/taskA_nextday_forecast.csv` | Per-model metrics for Task A |
| `results/taskB_15day_forecast.csv` | Per-model metrics for Task B |
| `results/summary.json` | Best model summary (selected by val F1) |
| `results/confusion_matrix_taskA.csv` | Best model confusion matrix (Task A) |
| `results/confusion_matrix_taskB.csv` | Best model confusion matrix (Task B) |
| `results/confusion_matrix_taskA_ensemble.csv` | Ensemble confusion matrix (Task A) |
| `results/confusion_matrix_taskB_ensemble.csv` | Ensemble confusion matrix (Task B) |
| `models/*.pkl` | Saved RF / XGBoost / Scaler models |
| `models/*.pt` | Saved LSTM / Transformer weights |

---

## Requirements

```
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
xgboost>=2.0
torch>=2.0
joblib>=1.3
```

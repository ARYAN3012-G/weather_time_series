# Weather Time Series — Drought Prediction

Part of the **DeepDroughtUAV** project (IIITDM Kurnool × JNTUA × SSSIHL)

## Overview

This module predicts agricultural drought using ERA5-Land weather time series data for 8 sites in Rayalaseema, Andhra Pradesh.

### Two Tasks

| Task | Description | Horizon |
|---|---|---|
| **Task A** | Current-state drought classification | 1 day (next day) |
| **Task B** | 15-day ahead drought forecast | 15 days |

### 3 Drought Classes
- **Healthy**: SPEI ≥ 0.0
- **Moderate**: −1.5 ≤ SPEI < 0.0
- **Severe**: SPEI < −1.5

### 8 Field Sites

| Site | District | State |
|---|---|---|
| Cholasamudram | Sri Sathya Sai | Andhra Pradesh |
| Kurnool Balaji Villas | Kurnool | Andhra Pradesh |
| Kurnool Dinedevarapadu | Kurnool | Andhra Pradesh |
| Kurnool Pulliah | Kurnool | Andhra Pradesh |
| Lepakshi | Sri Sathya Sai | Andhra Pradesh |
| Peddakadubur | Kurnool | Andhra Pradesh |
| Sira | Tumkur | Karnataka |
| Somandepalle | Sri Sathya Sai | Andhra Pradesh |

### 9 Input Features (ERA5-Land)
- Wind Speed 10m Mean 24h
- Temperature Air 2m Max/Mean/Min 24h
- Derived Relative Humidity 2m Max/Min 24h
- Precipitation Flux
- Reference ET (Penman-Monteith FAO-56)
- Solar Radiation Flux

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python src/training/train.py
```

## Output

```
results/
  taskA_current_state.csv   ← All model results for Task A
  taskB_15day_forecast.csv  ← All model results for Task B
  summary.json              ← Best model per task
models/
  rf_taskA.pkl / rf_taskB.pkl
  xgb_taskA.pkl / xgb_taskB.pkl
  lstm_taskA.pt / lstm_taskB.pt
  transformer_taskA.pt / transformer_taskB.pt
```

## Architecture

```
30-day weather window (9 features)
         │
         ├── Task A (horizon=1)  → classify TODAY
         └── Task B (horizon=15) → forecast 15 days ahead
                   │
                   ├── Random Forest
                   ├── XGBoost
                   ├── LSTM
                   ├── Transformer
                   └── Ensemble (TF×40% + XGB×30% + LSTM×20% + RF×10%)
```

## Labels — Data Leakage Guard

`Water_Balance`, `WB30`, and `SPEI` are strictly **excluded** from model input features. They are only used to derive the target label.

## Integration with DeepDroughtUAV

This weather module acts as an **early warning system**:
- If Task B predicts Moderate/Severe drought at any site 15 days ahead
- → Satellite module confirms at regional scale
- → UAV drone deployed to that site for precise field-level verification

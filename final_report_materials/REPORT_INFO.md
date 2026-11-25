# Final-semester report: facts sheet (weather + satellite + rabi modules)

Everything needed to write the report, chapter by chapter. Every number comes from the frozen results in the three repositories, with the source file named. The UAV module will be added later (marked *[UAV]*).

**Repositories**
- Weather: github.com/ARYAN3012-G/weather_time_series. Results v4: `results/v4/`.
- Satellite (kharif): github.com/ARYAN3012-G/Sentinel2-Drought-Confirmation. Report PDF: `docs/Drought_Early_Warning_Final_Report.pdf`.
- Rabi: github.com/ARYAN3012-G/Rabi-Drought-Monitoring-. Frozen tag `rabi-forecast-v1-final`; results: `results/forecast_v1_A1/FORECAST_RESULTS.md`.

**Diagrams** (in `diagrams/`, PNG 300 dpi + SVG; regenerate with `python final_report_materials/make_diagrams.py`)

| File | Use in chapter |
|---|---|
| fig1_system_architecture | Ch 1 / 3: whole system |
| fig2_weather_module_architecture | Ch 4: weather module |
| fig3_satellite_module_architecture | Ch 5: satellite module |
| fig4_rabi_module_architecture | Ch 6: rabi module |
| fig5_operational_timing | Ch 3 or 7: data delays and horizons |
| fig6_data_splits | Ch 3: train / validation / test periods |
| fig7_rabi_forecast_rules | Ch 6: eligibility and target rules |

---

## Suggested chapters

1. Introduction
2. Literature survey (`LITERATURE_SURVEY.md`)
3. System design and study area
4. Weather module
5. Satellite module
6. Rabi module
7. *[UAV]* module
8. Results and discussion
9. Limitations
10. Conclusion and future work
11. References

---

## 1. Introduction: content

- **Problem.** Rayalaseema (Andhra Pradesh) and neighbouring Tumakuru (Karnataka) are among India's most drought-prone semi-arid regions.
  - Drought is declared by the state through the Manual for Drought Management (2016). The declaration comes late in the season: kharif orders in Oct–Nov, rabi orders in mid/late March.
  - National studies find drought detection weakest in Andhra Pradesh and Karnataka (Bhardwaj & Mishra 2021).
- **Idea: a three-stage early-warning chain.**
  1. **Weather predicts** meteorological drought 7–30 days ahead.
  2. **Satellite confirms** whether crops actually became stressed, and which fields.
  3. **UAV quantifies** the damaged area on the flagged fields *[UAV]*.
- **Objectives**
  1. Forecast the SPEI-30 drought class 7, 15 and 30 days ahead, and beat persistence and a physical baseline.
  2. Measure whether and when crop greenness follows weather drought; forecast crop condition; confirm alarms against government declarations in a realistic live replay.
  3. Rabi season: build a method suited to rabi (stored soil water, irrigation) and forecast crop greenness 15 days ahead.
  4. *[UAV]* Quantify the damaged area.
- **Principles across all modules:**
  - no information from the test years (frozen calibrations, purged splits, automated leakage tests);
  - strong simple baselines;
  - rules written before evaluation;
  - government orders used only as an independent comparison.

## 3. Study area and data

| Site | Village | Mandal (district) | Centre (°N, °E) | Weather cell |
|---|---|---|---|---|
| Somandepalle | Tungodu | Somandepalle (Sri Sathya Sai) | 13.9988, 77.6247 | 14.0, 77.6 |
| Lepakshi | Mamidimakalapalli | Lepakshi (Sri Sathya Sai) | 13.7998, 77.6194 | 13.8, 77.6* |
| Cholasamudram | Cholasamudram | Lepakshi (Sri Sathya Sai) | 13.8118, 77.5687 | 13.8, 77.6* |
| Sira | Gidaganahalli | Sira taluk (Tumakuru, KA) | 13.7975, 76.9750 | 13.8, 77.0 |
| Peddakadubur | Pedda Thumbalam | Adoni / Pedda Kadubur (Kurnool) | 15.7826, 77.2528 | 15.8, 77.3 |
| Kurnool – Dinnedevarapadu | Dinnedevarapadu | Kurnool Rural | 15.7715, 78.0471 | 15.8, 78.0 |
| Kurnool – Balaji Villas | Kurnool outskirts | Kurnool Urban | 15.7834, 78.0561 | 15.8, 78.1* |
| Kurnool – Pullaiah College | Venkayyapalle | Kurnool Urban | 15.7982, 78.0806 | 15.8, 78.1* |

\* = shared cell, so there are **6 independent weather cells**. Each site centre is a UAV survey field.

| Data | Content | Period | Use |
|---|---|---|---|
| AgERA5 v2.0 (Copernicus CDS) | daily rain, FAO-56 ET0, temperature, humidity, radiation, wind; 0.1° | 1979-01-30 – 2026-11-02 | weather inputs, SPEI |
| Sentinel-2 SR Harmonized (Earth Engine) | B2–B12, NDVI, NDMI, EVI; Cloud Score+ ≥ 0.6 and SCL masking | 2019 – Oct 2026 | crop condition |
| ESA WorldCover 2020/21, Dynamic World | 10 m land cover | 2019–2023 | finding farmland |
| geoBoundaries IND ADM3 (LGD 2018) | mandal polygons | 2018 | rabi mandals, boundary check |
| AP G.O.Ms 4/2023, 15/2024, 6/2025, 4 & 5/2026 (kharif); 7/2024, 3/2025, 1/2026 (rabi); Karnataka 2023 order | declared drought mandals/taluks, severe/moderate | 2023–2026 | independent comparison |
| UAV dataset (DeepDroughtUAV) | 10 surveyed fields, 2,254 images | Nov 2025 – Jan 2026 | field check *[UAV]* |

**Periods** (Fig. 6):

| Module | Train | Validation | Test |
|---|---|---|---|
| Weather | 1979–2011 | 2012–2018 | 2019–2026 |
| Satellite | 2019–2023 | 2024 | 2025–2026 |
| Rabi | seasons 2019-20 – 2021-22 | 2022-23 | 2023-24, 2024-25, 2025-26 |

**Tools:**
- Python (pandas, numpy, scikit-learn, XGBoost, PyTorch for LSTM/Transformer, matplotlib, reportlab), Google Earth Engine (JavaScript), git/GitHub.
- The heavy weather runs used a lab desktop with an RTX 2000 Ada GPU: about 7 h for the full study (main 2.6 h, leave-one-cluster-out 2.0 h, Marusov 0.6 h, date sets 1.9 h).
- Satellite and rabi models run on a laptop in minutes.

## 4. Weather module: method and results (Fig. 2)

**Method**
- **Target:** SPEI-30 class 7/15/30 days ahead:
  - Healthy ≥ 0;
  - Moderate −1.5 to 0;
  - Severe < −1.5.
- **SPEI:** log-logistic distribution fitted by L-moments, per site and calendar month, on 1979–2011 only, then frozen.
- **Input:** a 30-day window of features. The feature sets are:
  - `weather` (9 variables);
  - `weather_spei`;
  - `spei_only`;
  - `weather_spei_doy`;
  - `weather_spei_doy_wb`, the **hybrid**, which adds 3 water-balance-ensemble class probabilities.
- **Water-balance ensemble (physics baseline):** the observed part of the SPEI window, plus each 1979–2011 year's water balance for the unknown days; class probability = share of years.
- **Models:** RF, XGBoost, LSTM, Transformer, and a soft-voting ensemble weighted by validation macro-F1. 5 seeds each.
- **Baselines:** persistence, Markov, climatology, majority, water-balance ensemble.
- **Split:** purged chronological. Leave-one-cluster-out (Kurnool vs south). Marusov et al. protocol. Block-bootstrap 95% CIs. 22 leakage and integrity tests.

**Results (test 2019–2026, macro-F1)**

| Model | 7 days | 15 days | 30 days |
|---|---|---|---|
| Persistence | 0.662 | 0.515 | 0.424 |
| Water-balance ensemble (physics only) | 0.783 | 0.592 | 0.375 |
| Best model without physics (XGBoost) | 0.755 | 0.613 | 0.424 |
| **Hybrid, ensemble of 4 models** | **0.819** | **0.651** | 0.424 |
| Hybrid vs physics, Δ [95% CI] | [−0.007, +0.088] | [+0.010, +0.099] | [+0.023, +0.084] |
| Balanced accuracy, hybrid | 85.6% | 68.3% | 42.5% |
| ROC-AUC, hybrid | 0.97 | 0.90 | 0.68 |

- **7 days:** physics is hard to beat; the hybrid's gain over it is not significant. Both beat persistence widely.
- **15 days:** the hybrid significantly beats both physics and persistence.
- **30 days:** no model beats persistence.
- **Leave-one-cluster-out:** the hybrid holds up (0.811 / 0.633 / 0.381).
- **Marusov protocol** (binary SPEI < −1): weather inputs raise 15-day ROC-AUC from 0.81 to 0.89.
- **Corrections made during the project (v2 → v3/v4)**, a good "lessons" subsection:
  - a duplicate weather cell was counted as two sites;
  - SPEI had been standardised on all years (leakage);
  - the 1-day horizon was trivially solved by persistence;
  - samples crossed split boundaries;
  - two sites used the wrong weather cell (fixed in v4).

## 5. Satellite module: method and results (Fig. 3)

**Method**
- **Farmland** (Earth Engine script A, training years only):
  - WorldCover cropland, minus Dynamic World built-up / tree / water pixels;
  - a crop cycle in ≥ 2 years;
  - SNIC segments of 0.3–10 ha, 13,750 segments in total;
  - a seeded random sample of 20 rainfed + 10 irrigated per site = **240 fields**;
  - irrigated = green in Mar–May.
- **Series** (script B): every clear image, kept if ≥ 60% of the field is clear; 5-day grid. Three versions:
  - causal (model inputs);
  - interpolated (labels only);
  - reference (normals).
- **Label:** SVI = (NDVI − normal) / spread, per field and day of year. Normals come from images ≤ 31 Dec 2023, with the scored year removed. Classes use the SPEI thresholds.
- **Analyses:**
  - **Lag analysis:** SVI vs SPEI-k at lags 0–90 days.
  - **Crop-condition forecast:** SVI class 15/30 days ahead, RF and XGBoost, 4 input versions: satellite only, + past weather, + weather-module forecast, + actual future weather (upper bound).
  - **Operational replay:** issued every 5 days Jun–Nov, weather 7 d late, satellite 2 d late.
    - Alarm: SPEI-30 < −1 or a severe forecast.
    - Confirmed: median rainfed SVI < −1 from ≥ 5 clear fields.
- 18 leakage tests (42 across both modules).

**Results**
- **Crops follow weather:** rainfed SVI follows SPEI-30 with a **10-day lag**. ρ = 0.48 in training and 0.45 on unseen 2024–2026; irrigated: SPEI-60, 15-day lag, ρ = 0.24 / 0.29.
- **Alarms:** in 22 of 44 weather alarms, rainfed crop stress followed within 60 days.
- **Forecast, 15 days, test 2025–26, macro-F1:**

  | Version | Macro-F1 |
  |---|---|
  | Persistence | 0.648 |
  | Satellite history | 0.667 |
  | + past weather | 0.675 |
  | + weather forecast | 0.675 |
  | + future weather (upper bound) | 0.686 |

  - The weather versions beat persistence.
  - Their gain over satellite history alone is **borderline and not claimed**.
- **Onset warning:** fields that will fall below normal within 15 days are ranked at ROC-AUC **0.76**, vs 0.72 for a logistic baseline.
- **Operational replay vs government orders:**
  - Severe site-seasons: **9/9 confirmed** eventually, **5/9 before the order** (median lead 7 days).
  - Each decision counted once:
    - **6/6 severe kharif mandal-seasons**, 5 before the order, all on in-season images;
    - false alarms **1/11** site-seasons;
    - prospective 2024–26: severe 6/6, false alarms 0/10.
  - Both prospective moderate 2025 cases were missed.
- **UAV fields:** clear agreement at 3 of 10, 2 mixed. Flights were post-harvest, and 10 m pixels average out small dry patches.
- **Rabi (Dec–Mar) with the kharif method:** 0 of 4 severe rabi mandal-seasons confirmed on in-season images. That motivates the separate rabi module.

## 6. Rabi module: method and results (Figs. 4, 7)

**Why a separate module**
- Rabi crops grow on stored soil water and irrigation, not current rain, and they fail late (terminal drought).
- The kharif chain (weather alarm, then satellite confirmation) confirmed no rabi case.
- There are too few rabi declarations to train a classifier, so the module forecasts **crop greenness**, whose target comes from future satellite images.

**History (an honest development story)**
1. **Candidate rabi fields** (scripts C1/C1v2): green in Jan–Mar with a "dip then rise" curve; 233 frozen fields.
2. **Protocol v1** (registered 21 Nov 2026): establishment and condition rules. On the 2023–26 development check it flagged 0 of 4 severe mandal-seasons. Field-level rabi activity and the orders disagree; the selected fields are biased toward resilient fields.
3. **Exploratory early-drying signal:** severe 2/4, false flags 1/10.
4. **Whole-mandal measurement** (script C3, fixed cropland denominator): irrigated area established by March fell to 17% at Adoni (normal 44%) and 28% at Pedda Kadubur (normal 48%) in 2023-24.

**Final model** (rules `docs/FORECAST_v1.md`, registered in git before training; amendments A1/A1b after two independent reviews; tag `rabi-forecast-v1-final`)
- **Data:**
  - 1,350 stratified random cropland points: 9 mandals × (100 rainfed + 50 irrigated);
  - 355,043 point observations, 313,797 after cleaning;
  - 7 seasons, 2019-20 – 2025-26.
- **Forecast every 5 days.** A forecast is used only if:
  - the latest image is ≤ 12 days old (excluded otherwise: **54,260**);
  - a real image exists at t + 15 ± 3 days, with no interpolation (excluded otherwise: **55,620**).
  - That leaves **233,020 of 342,900 usable (68%)**. Missing data is never treated as "no crop".
- **Inputs:**
  - past NDVI (now, −15 d, −30 d), changes, ages, season minimum/maximum, observation count, day of season;
  - the point's training-season normal at t + 15;
  - AgERA5 weather available **8 days late**, with frozen SPEI.
- **Model:** XGBoost regression on the 15-day NDVI change, with fixed settings.
- **Baselines:** persistence, climatology, anomaly persistence.

**Results** (historical evaluation, test seasons never used for any choice)

| Held-out season | XGBoost MAE | Best baseline MAE | Lower error |
|---|---|---|---|
| 2023-24 | 0.046 | 0.054 (persistence) | 14% |
| 2024-25 | 0.051 | 0.061 (anomaly persistence) | 16% |
| 2025-26 | 0.056 | 0.068 (anomaly persistence) | 17% |
| **All three** | **0.051** | **0.062** | **18%** |

- **Approved wording:** "XGBoost forecast NDVI approximately 15 days ahead with about 18% lower mean absolute error overall than the best simple baseline. It outperformed all three baselines in each held-out season, with seasonal improvements of 14–17%. Adding the tested weather features did not improve overall accuracy. These are historical evaluation results, not a prospective test or validation of drought detection."
- **December–March only:** 17% lower error, with wins in each season.
- **Uncertainty:** 95% CI of the error difference, resampling sampled points: [−0.012, −0.010]; resampling the 5 weather cells: [−0.015, −0.008] (indicative).
- **Satellite-only XGBoost:** MAE 0.051, so weather adds nothing in this experiment.
- **Decline warnings** (established crops, decline ≥ 0.10 from the latest image to the target; span 15–30 days, median 20):
  - AUC 0.83, precision **62%**, recall **24%**: selective, and misses about three-quarters of declines.
  - A decline can be drought or harvest.

## 8. Discussion points
- **Physics + ML:** the physical water-balance ensemble is a strong baseline, and machine learning adds skill mainly at 15 days. Report this as a strength (honest baselines).
- **Crop stress follows meteorological drought** by about 10 days on rainfed land. This makes a weather alarm a useful "where to look" signal.
- **Weather adds little to 15–30-day crop forecasts** beyond satellite history, in both kharif and rabi. Crop condition is set mostly by water already lost.
- **Government declarations are administrative decisions.** Agreement is strong for severe kharif droughts and weak for moderate and rabi cases.
- **Rabi needs its own design:** stored water, establishment, terminal drought. Forecasting greenness avoids training on a handful of declarations.

## 9. Limitations
- **Short satellite record:** five years of normals. Few drought events, so agreement figures are case evidence.
- **Clouds:** monsoon clouds delay confirmation in Aug–Sep.
- **Index vs damage:** SVI classes are index categories, not measured damage. Irrigation is inferred from dry-season greenness.
- **Spatial mismatch** between mandal, 3 km area and field.
  - Mandal boundary check: 37% of Lepakshi fields lie in Chilamathur, 23% of Sira fields in Agali, and 70% of Peddakadubur fields in Pedda Kadubur.
  - The 2018 polygons cannot separate Kurnool Urban from Kurnool Rural.
- **AgERA5 delay:** about a week, so a 7-day weather forecast gives little real warning.
- **Rabi:**
  - evaluated on seasons known during design;
  - it forecasts NDVI, not drought;
  - points are sampled pixels, not verified fields;
  - only five weather cells and three test seasons.

## 10. Future work
- **Prospective test** on rabi 2026-27 and kharif 2027 with the frozen rules (the protocol and R1/R2 registrations are already in git).
- **Independent agricultural-drought validation:** crop-cutting or yield statistics, field surveys, UAV damage maps.
- **Soil-water balance** with soil-specific storage for rabi; SIF or soil-moisture products.
- **Longer records:** Sentinel-2 TOA from 2016, or Landsat, as a separate extension.
- *[UAV]* Integration: satellite-flagged fields lead to drone segmentation and area estimates.

## Novelty (table for the report)

| Aspect | Common practice | This work |
|---|---|---|
| Baselines | other ML models | persistence, Markov, climatology, physical water-balance ensemble |
| Calibration | full record | SPEI frozen on 1979–2011; SVI normals 2019–2023 with the scored year removed |
| Leakage | rarely tested | 42 automated tests, independent reviews, purged splits |
| Fields | hand-picked or district averages | automatic farmland mapping, random fields and points |
| Validation | index vs index | government orders over four seasons, live replay with data delays, UAV fields |
| Rabi | post-hoc coarse indices | 10 m greenness forecast, rules registered before training, missing-data rules |

## Corrected early-drying event audit (22 November 2026)

Source: `rabi_module/results/detect_v2/EVENT_REPORT.md`. Supersedes classifier v1 scores; keep the separate frozen NDVI forecasting result.

# Rabi early-drying risk: corrected historical evaluation and unique-event audit

Development correction v2, 22 November 2026. Supersedes detect_v1 for operational claims. Satellite-defined early drying is a proxy event; agricultural drought causation is not independently verified.

## Corrections and scope

- Removed retrospective completed-season status from inputs; all satellite features use only issue-time observations.
- Reference population, drying normals and NDVI curves use training seasons 2019-21 only (own season excluded).
- Weather uses an 8-day publication lag, including the early-December antecedent-rain feature.
- Forecasts target a new event within the horizon. Known/past events cannot earn advance-warning credit.
- Weights, thresholds and primary model are chosen on 2022-23 validation, not the historical evaluation seasons.
- v1 did not save thresholds/validation predictions, preventing an exact recount at its original operating point. Both are saved in v2. The cohort changed with the corrected reference; v1/v2 scores are not a same-cohort ablation.

## Coverage

| season | points | reference_ok | season_quality_ok | scorable | early_events | forecast_points | new_onset_training_or_evaluation_rows |
|---|---|---|---|---|---|---|---|
| 2019 | 1350 | 127 | 1350 | 127 | 27 | 92 | 1107 |
| 2020 | 1350 | 141 | 1350 | 141 | 9 | 96 | 1559 |
| 2021 | 1350 | 132 | 1350 | 132 | 16 | 103 | 1416 |
| 2022 | 1350 | 232 | 1345 | 232 | 18 | 149 | 2440 |
| 2023 | 1350 | 232 | 1350 | 232 | 59 | 148 | 1984 |
| 2024 | 1350 | 232 | 1350 | 232 | 33 | 161 | 2377 |
| 2025 | 1350 | 232 | 1350 | 232 | 17 | 145 | 2472 |

## New-onset point-date metrics (validation-selected model)

| horizon | model | n | positives | prevalence | roc_auc | average_precision | precision | recall |
|---|---|---|---|---|---|---|---|---|
| 7 | Random forest | 6833 | 144 | 0.021 | 0.925 | 0.175 | 0.075 | 0.958 |
| 15 | Random forest | 6833 | 292 | 0.043 | 0.914 | 0.312 | 0.162 | 0.897 |
| 30 | Random forest | 6833 | 527 | 0.077 | 0.905 | 0.372 | 0.242 | 0.920 |

## Unique point-season events (validation-selected model)

| horizon | model | monitored_events | events_with_opportunity | caught_events | recall_all_events | recall_opportunity | warned_point_seasons | warning_precision | median_lead |
|---|---|---|---|---|---|---|---|---|---|
| 7 | Random forest | 108 | 104 | 102 | 0.944 | 0.981 | 331 | 0.308 | 5.000 |
| 15 | Random forest | 108 | 106 | 101 | 0.935 | 0.953 | 305 | 0.331 | 12.000 |
| 30 | Random forest | 108 | 107 | 104 | 0.963 | 0.972 | 332 | 0.313 | 27.000 |

Each point-season contributes at most one event and one warned unit. Warning precision here is the fraction of warned point-seasons with at least one timely alert. It is not per-alert precision. Earlier false alerts are not erased from the saved point-date predictions. Lead time uses the first timely warning per caught event.

## Lead-time requirements

| horizon | model | recall_at_least_7_days | recall_at_least_14_days | lead_q25 | median_lead | lead_q75 |
|---|---|---|---|---|---|---|
| 7 | Random forest | 0.213 | 0.000 | 3.000 | 5.000 | 6.000 |
| 15 | Random forest | 0.843 | 0.120 | 11.000 | 12.000 | 13.000 |
| 30 | Random forest | 0.917 | 0.833 | 20.000 | 27.000 | 28.000 |

## Event uncertainty

| horizon | model | bootstrap_unit | clusters | recall_all_events_lo | recall_all_events_hi | recall_opportunity_lo | recall_opportunity_hi | warning_precision_lo | warning_precision_hi |
|---|---|---|---|---|---|---|---|---|---|
| 7 | Random forest | point_id | 232 | 0.897 | 0.981 | 0.951 | 1.000 | 0.253 | 0.361 |
| 7 | Random forest | cell | 5 | 0.846 | 0.974 | 0.944 | 1.000 | 0.252 | 0.365 |
| 15 | Random forest | point_id | 232 | 0.885 | 0.976 | 0.908 | 0.990 | 0.275 | 0.387 |
| 15 | Random forest | cell | 5 | 0.900 | 0.947 | 0.932 | 0.983 | 0.277 | 0.392 |
| 30 | Random forest | point_id | 232 | 0.922 | 0.992 | 0.938 | 1.000 | 0.259 | 0.370 |
| 30 | Random forest | cell | 5 | 0.944 | 0.984 | 0.944 | 1.000 | 0.267 | 0.372 |

Point resampling retains each point across seasons; weather-cell resampling is indicative with very few cells. Neither creates additional independent drought seasons.


## Rabi soil-drought report update (23 November 2026)

Added `latex/rabi_drought_section.tex` from saved drought_v1 metrics. The selected HGB models predict regional soil-moisture deficit, not verified field damage. Recent evaluation has 0 positive drought targets; the earlier post hoc new-onset comparison favors conditional transition. All three rabi tasks remain separate. Rebuild with `update_rabi_drought.py` after the earlier report generators. `RABI_FIGURE_AUDIT.md` records corrections and data sources.

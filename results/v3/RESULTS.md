# Weather module (stage 1) — v3 results and project log

Meteorological drought forecasting 7, 15 and 30 days ahead from daily ERA5-Land weather at
7 sites in Rayalaseema (Andhra Pradesh) and Tumkur (Karnataka). Stage 1 (*Predict*) of the
integrated drought assessment system: weather forecast → satellite confirmation → UAV quantification.

All numbers below come from the full v3 run of 4–5 November 2026 (commit `78e3185` + exit-code fix)
and can be regenerated with `run_all.bat`. Detailed tables:
[`tables/REPORT_temporal.md`](tables/REPORT_temporal.md), [`tables/REPORT_loco.md`](tables/REPORT_loco.md),
[`marusov_protocol/MARUSOV_PROTOCOL.md`](marusov_protocol/MARUSOV_PROTOCOL.md),
full numbers in `tables/summary_*.csv` and `tables/all_runs_*.csv`.

> v2 results (XGB 81.02 %, Transformer 82.18 %) are **invalid** (duplicate site, SPEI leakage,
> trivial 1-day task) and are not reported anywhere in this document.

---

## 1. Headline findings

1. **7 and 15 days: the models significantly beat persistence**, the strongest baseline.
   At 7 days, the validation-selected model (`weather_spei_doy` XGBoost) scores macro-F1 **0.768** vs
   persistence 0.634 (paired 95 % CI of the gain **+0.093 to +0.186**). At 15 days: **0.630** vs 0.494
   (CI **+0.071 to +0.202**). Climatology scores only 0.32.
2. **30 days: no model beats persistence.** Best 0.436 vs 0.410; the CI includes 0. The target
   SPEI-30 window then shares no days with the input, so this is effectively a monthly rainfall
   forecast from local weather. Reported as a limitation.
3. **Our input (weather + SPEI history) beats both weather alone and SPEI history alone**, for every
   model and horizon (ablation, Section 5). SPEI history alone is the input of Marusov et al.
4. **Marusov et al. protocol on our data:** our input improves median ROC-AUC over their index-only input
   at every horizon (7 d: 0.959 vs 0.936; 15 d: 0.889 vs 0.823; 30 d: 0.699 vs 0.657).
5. **Severe-drought early warning is usable at 7 days** (POD 0.74, FAR 0.52 for the `weather_spei`
   ensemble), weak at 15 days (FAR ≈ 0.7) and not usable at 30 days (FAR ≈ 0.9).
6. **Spatial transfer (leave-one-cluster-out):** with SPEI history the models stay level with
   persistence on an unseen cluster (significantly better only for Kurnool at 15 days); weather-only
   models do **not** transfer (significantly worse than persistence at 7 days).

---

## 2. What was done in this phase

### 2.1 Corrections from v2 to v3 (pipeline, commit `347a867`)

| Problem in v2 | Fix in v3 |
|---|---|
| `lepakshi` and `cholasamudram` share one ERA5-Land cell (13.8° N, 77.6° E) — identical data counted as two sites | `lepakshi` removed → **7 independent sites** (`audit/AUDIT.md`) |
| SPEI standardised with all years 1979–2026 → test information in the labels | Log-logistic SPEI (L-moments), per site and calendar month, **calibrated on 1979–2011 only**, parameters frozen (`spei_calibration.json`) |
| 1-day task: persistence scored macro-F1 0.92 | Horizons **7 / 15 / 30 days** with persistence, Markov, climatology and majority baselines |
| Samples near split boundaries shared days | **Purged chronological split**: TRAIN ≤ 2011, VAL 2012–2018, TEST ≥ 2019 |
| One seed, no uncertainty, no spatial test | 5 seeds, block-bootstrap 95 % CIs (blocks = target month), leave-one-cluster-out |

### 2.2 Review and additions in this session (commits `78e3185` and the results commit)

- **Code review before training.** Checked SPEI formulas against Vicente-Serrano et al. (2010), units
  (precipitation and FAO-56 PET both mm/day), sequence building, purged splits, baselines, metrics,
  aggregation and the Marusov protocol. No leakage or logic errors found. Data preparation reproduces
  the committed SPEI parameters exactly (differences only at floating-point noise level, ~1e-14).
- **GPU environment.** PyTorch was not installed, and `pip install -r requirements.txt` on Windows
  installs a CPU-only build. Added a project virtual environment (`.venv`, PyTorch 2.11 + CUDA 12.8,
  NVIDIA RTX 2000 Ada 16 GB); `run_all.bat` / `run_all.sh` now create and use it.
- **Date feature sets (extra experiment, fixed before any results were seen).** `weather_doy` and
  `weather_spei_doy` add sin/cos of day-of-year of **each input day**. Motivation: SPEI is
  standardised per calendar month, so season matters, and the climatology baseline already knows the
  date. Kept out of the default run (`config.DEFAULT_FEATURE_SETS`); existing sets unchanged.
- **New tests** (18 → 22, all passing): date features encode input-window dates only and leave the
  other channels unchanged; the date sets are not in the default run.
- **Report rules.** Tuned severe POD/FAR are not shown for climatology/majority (near-constant
  probabilities make a tuned threshold meaningless). The ensemble is reported on **test only**: its
  weights are fitted on validation, so its validation score is in-sample and it is never chosen as
  "best by validation".
- **Windows exit crash fixed.** After long GPU runs, Python aborted during shutdown
  (`0xc0000409` in `ucrtbase.dll`, confirmed in the Windows event log) *after* all results were saved;
  the non-zero exit code stopped `run_all.bat`. `run_experiments.py` and `marusov_protocol.py` now end
  the process directly once everything is written; real errors still exit with an error code.
- pandas 3 deprecation fixed (`dayofyear` → `day_of_year`).

### 2.3 Runs completed

| Step | Runs | Wall time (GPU) |
|---|---|---|
| Main study (temporal): 3 horizons × 3 feature sets × 4 models × 5 seeds | 180 | 2.6 h |
| Leave-one-cluster-out: 2 folds × 3 × 2 × 4 × 5 | 240 | 2.0 h |
| Marusov et al. protocol: 3 × 2 inputs × 2 models × 5 seeds (+ reference methods) | 60 | 0.6 h |
| Date feature sets (temporal): 3 × 2 × 4 × 5 | 120 | 1.9 h |
| Aggregation (tables, ensembles, 1000 bootstrap resamples) | — | 3 min |

---

## 3. Setup

- **Data:** daily ERA5-Land, 1979-01-30 to 2026-06-29, 7 sites (clusters: *kurnool* = Peddakadubur,
  Kurnool Balaji Villas, Kurnool Dinedevarapadu, Kurnool Pulliah; *south* = Somandepalle,
  Cholasamudram, Sira).
- **Labels (SPEI-30):** Healthy ≥ 0, Moderate −1.5 … 0, Severe < −1.5.
- **Task:** input = 30 days of features (t−29 … t); target = drought class on day t+h, h ∈ {7, 15, 30}.
- **Feature sets:** `weather` (9 ERA5 variables), `weather_spei` (+ historical SPEI, **our approach**),
  `spei_only` (index history, Marusov-style), `weather_doy`, `weather_spei_doy` (+ season).
- **Models:** Random Forest, XGBoost, LSTM, Transformer, validation-F1-weighted ensemble.
- **Baselines:** persistence (class on day t), Markov (train transition matrix), climatology
  (train class frequency for the target calendar day), majority.

Sample counts (h = 7, pooled over 7 sites):

| Split | Samples | Healthy | Moderate | Severe |
|---|---|---|---|---|
| Train 1979–2011 | 83,916 | 49.0 % | 44.9 % | 6.1 % |
| Validation 2012–2018 | 17,647 | 48.0 % | 44.8 % | 7.1 % |
| Test 2019–2026 | 18,886 | 66.6 % | 29.2 % | 4.2 % |
| Purged (cross a boundary) | 504 | | | |

The test period is markedly wetter than training (a real climate signal, not a calibration artefact).

---

## 4. Main results — temporal split (test 2019+)

Macro-F1, mean over 5 seeds (seed std ≤ 0.02 everywhere). Δ = paired block-bootstrap 95 % CI of
(model − persistence); an interval entirely above 0 is a significant improvement. **Bold** = significant.

### 4.1 Baselines

| Baseline | 7 d | 15 d | 30 d |
|---|---|---|---|
| Persistence | 0.634 | 0.494 | 0.410 |
| Markov | 0.490 | 0.417 | 0.368 |
| Climatology | 0.321 | 0.322 | 0.322 |
| Majority | 0.266 | 0.266 | 0.266 |

### 4.2 Models (selected rows; full table in `tables/REPORT_temporal.md`)

| Horizon | Feature set | Model | Macro-F1 | Δ vs persistence | Severe-F1 |
|---|---|---|---|---|---|
| 7 d | weather | ensemble | 0.652 | [−0.031, +0.059] | 0.431 |
| 7 d | spei_only | ensemble | 0.682 | **[+0.013, +0.090]** | 0.510 |
| 7 d | weather_spei | XGBoost | 0.721 | **[+0.052, +0.126]** | 0.555 |
| 7 d | weather_spei | ensemble | 0.740 | **[+0.075, +0.159]** | 0.595 |
| 7 d | weather_spei_doy | XGBoost ★ | **0.768** | **[+0.093, +0.186]** | 0.627 |
| 15 d | weather | ensemble | 0.565 | **[+0.037, +0.131]** | 0.319 |
| 15 d | spei_only | ensemble | 0.533 | [−0.008, +0.104] | 0.297 |
| 15 d | weather_spei | XGBoost | 0.593 | **[+0.040, +0.160]** | 0.351 |
| 15 d | weather_spei | ensemble | 0.609 | **[+0.064, +0.184]** | 0.395 |
| 15 d | weather_spei_doy | XGBoost ★ | **0.630** | **[+0.071, +0.202]** | 0.414 |
| 30 d | weather | ensemble | 0.374 | [−0.090, +0.018] | 0.033 |
| 30 d | spei_only | ensemble | 0.380 | [−0.070, +0.005] | 0.139 |
| 30 d | weather_spei | XGBoost | 0.412 | [−0.045, +0.054] | 0.115 |
| 30 d | weather_spei | ensemble | 0.406 | [−0.046, +0.052] | 0.081 |
| 30 d | weather_spei_doy | XGBoost ★ | 0.436 | [−0.025, +0.083] | 0.161 |

★ = best model by **validation** macro-F1 at that horizon (selection does not use test data).

Probabilistic skill: Brier skill score vs climatology for the `weather_spei` ensemble is +0.55 (7 d),
+0.35 (15 d) and +0.05 (30 d); persistence scores +0.09, −0.32 and −0.56.

---

## 5. Ablation — what does each input contribute?

Test macro-F1 of the ensemble (validation-weighted soft vote of RF, XGB, LSTM, Transformer):

| Horizon | weather | spei_only | weather_spei | weather_doy | weather_spei_doy |
|---|---|---|---|---|---|
| 7 d | 0.652 | 0.682 | **0.740** | 0.719 | **0.761** |
| 15 d | 0.565 | 0.533 | **0.609** | 0.610 | **0.622** |
| 30 d | 0.374 | 0.380 | **0.406** | 0.409 | **0.425** |

- `weather_spei` > `weather` and > `spei_only` for **every** model at every horizon: weather and index
  history carry complementary information.
- The date features help most when SPEI is absent (`weather` XGB 0.637 → `weather_doy` XGB 0.717 at
  7 d) and add a smaller gain on top of SPEI (0.721 → 0.768).
- Among single models, XGBoost is the best model for every feature set that includes weather, at
  every horizon; with `spei_only` the LSTM is best. The ensemble beats every single model for
  `weather_spei` at 7 and 15 days.

---

## 6. Severe-drought early warning

Severe = SPEI < −1.5. Probability threshold tuned on **validation** only, applied once to test.
POD = share of severe days detected; FAR = share of severe warnings that were false.

| Horizon | Model | Severe-F1 | POD (tuned) | FAR (tuned) |
|---|---|---|---|---|
| 7 d | persistence | 0.454 | 0.454 | 0.546 |
| 7 d | weather_spei ensemble | 0.595 | 0.743 | 0.524 |
| 7 d | weather_spei_doy XGB | 0.627 | 0.743 | 0.468 |
| 15 d | persistence | 0.279 | 0.279 | 0.721 |
| 15 d | weather_spei ensemble | 0.395 | 0.601 | 0.715 |
| 15 d | weather_spei_doy XGB | 0.414 | 0.619 | 0.688 |
| 30 d | persistence | 0.158 | 0.158 | 0.842 |
| 30 d | weather_spei_doy XGB | 0.161 | 0.517 | 0.876 |

Confusion matrix at 7 days (`weather_spei_doy` XGB, seed-averaged probabilities, argmax): of 792
severe test days, 499 were predicted Severe, 290 Moderate and only 3 Healthy; 290 non-severe days
were wrongly flagged Severe (`tables/confusion_temporal_all_sites_h7_weather_spei_doy_xgb.csv`).
Misses are almost always "one class too mild", not missed droughts.

---

## 7. Spatial generalisation — leave-one-cluster-out

Train/validate on one cluster, test on the other cluster's 2019+ data (unseen in space and time).
Macro-F1; Δ vs persistence on the held-out cluster.

| Test cluster | Horizon | Persistence | weather ensemble | weather_spei XGB | weather_spei ensemble |
|---|---|---|---|---|---|
| kurnool | 7 d | 0.624 | 0.526 [−0.146, −0.034] | 0.674 [−0.003, +0.106] | 0.669 [−0.001, +0.107] |
| kurnool | 15 d | 0.469 | 0.468 [−0.054, +0.072] | 0.545 **[+0.014, +0.143]** | 0.531 **[+0.013, +0.139]** |
| kurnool | 30 d | 0.427 | 0.365 [−0.117, +0.013] | 0.380 [−0.113, +0.026] | 0.361 [−0.128, +0.011] |
| south | 7 d | 0.646 | 0.511 [−0.214, −0.043] | 0.663 [−0.046, +0.093] | 0.666 [−0.044, +0.109] |
| south | 15 d | 0.522 | 0.484 [−0.115, +0.020] | 0.515 [−0.086, +0.086] | 0.521 [−0.078, +0.081] |
| south | 30 d | 0.388 | 0.311 [−0.142, −0.014] | 0.375 [−0.068, +0.053] | 0.357 [−0.074, +0.029] |

- With SPEI history, a model trained on one cluster stays level with persistence on the other
  (point estimates above persistence at 7 days in both clusters; significant for Kurnool at 15 days).
- Weather-only models do not transfer: significantly **worse** than persistence at 7 days in both
  clusters. Local weather–drought relationships differ between clusters; index history is what
  transfers.
- Limitation: only two clusters, ~200 km apart, so this is a modest test of spatial transfer.

---

## 8. Comparison with Marusov et al. (base paper)

Marusov et al., *Long-term drought prediction using deep neural networks based on geospatial weather
data*, Environmental Modelling & Software (arXiv:2309.06212): binary drought PDSI < −2, index-history
input, per-cell ROC-AUC / PR-AUC / F1 summarised by the median, 70/30 split without a validation set,
majority baseline only.

Our reproduction of their protocol on our data (binary drought = SPEI(t+h) < −1.0; scores per site on
the test period, median over the 7 sites; mean over 5 seeds):

| Horizon | Input | Model | Median ROC-AUC | Median PR-AUC | Median F1 |
|---|---|---|---|---|---|
| 7 d | their input (spei_only) | logistic regression (last value) | 0.906 | 0.549 | 0.527 |
| 7 d | their input (spei_only) | LSTM | 0.936 | 0.659 | 0.595 |
| 7 d | their input (spei_only) | XGBoost | 0.910 | 0.602 | 0.578 |
| 7 d | – | persistence | 0.745 | 0.335 | 0.546 |
| 7 d | **our input (weather_spei)** | **XGBoost** | **0.959** | **0.759** | **0.683** |
| 15 d | their input (spei_only) | LSTM | 0.823 | 0.376 | 0.407 |
| 15 d | – | persistence | 0.650 | 0.184 | 0.344 |
| 15 d | **our input (weather_spei)** | **XGBoost** | **0.889** | **0.437** | **0.450** |
| 30 d | their input (spei_only) | logistic regression (last value) | 0.657 | 0.174 | 0.252 |
| 30 d | – | persistence | 0.574 | 0.153 | 0.259 |
| 30 d | **our input (weather_spei)** | **XGBoost** | **0.699** | **0.212** | 0.227 |

What this shows:
- Adding weather to the index history improves ranking skill (ROC-AUC, PR-AUC) at every horizon and
  F1 at 7 and 15 days.
- Their protocol reports only a majority baseline; under it every model looks strong (majority
  ROC-AUC = 0.5). Persistence, which they did not report, is a much harder reference — at 30 days it
  has the best F1 of all methods. Our main evaluation therefore uses persistence as the reference.
- Methodological differences we add: separate validation set, purged chronological split, SPEI
  calibrated on training years only, multiple seeds, block-bootstrap CIs, spatial hold-out.
- This is a **methodology** comparison on our data; our numbers are not directly comparable to the
  paper's monthly PDSI results on a dense grid. ConvLSTM/EarthFormer need a dense spatial grid and are
  not applicable to 7 point sites.

Other related work cited: Dikshit, Pradhan & Huete (2021), LSTM SPEI forecasting, J. Environ. Manage.
283:111979; Tang et al. (2025), SQUARE-Mamba, arXiv:2502.20703.

---

## 9. Limitations

1. **No skill over persistence at 30 days** (macro-F1 and severe warning). Longer lead times would need
   a longer accumulation index (e.g. SPEI-90) or large-scale climate drivers.
2. **Wetter test period** (66.6 % Healthy vs 49.0 % in training); only 792 severe test days (4.2 %).
3. **Many configurations were compared.** We quote the model chosen by validation; test-best numbers
   would be optimistic.
4. **Limited spatial test:** 2 clusters, 7 sites, all in one semi-arid region.
5. **Meteorological drought only:** SPEI-30 is a climatic water-balance index, not crop stress;
   agricultural impact is assessed in the satellite and UAV stages.
6. Hyperparameters were not tuned (kept from v2 so that differences come from the corrections).

---

## 10. Reproducing

```bat
cd weather_time_series
run_all.bat
.venv\Scripts\python.exe src\experiments\run_experiments.py --split temporal --feature-sets weather_doy weather_spei_doy
.venv\Scripts\python.exe src\experiments\aggregate.py
```

Environment used: Windows 11, Python 3.14, PyTorch 2.11 + CUDA 12.8, XGBoost 3.4, scikit-learn 1.9,
pandas 3.1, NVIDIA RTX 2000 Ada (16 GB), 36 CPU cores, 64 GB RAM. 22/22 integrity tests passing.

## 11. Next steps

1. Satellite module (stage 2): confirm predicted meteorological drought with vegetation/moisture
   indices (e.g. NDVI, VCI, soil moisture) for the same sites and dates.
2. UAV module (stage 3): field-level damage segmentation and quantification.
3. Literature check of the integrated prediction → confirmation → quantification design (novelty claim).

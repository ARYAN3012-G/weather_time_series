# Modules and results (final numbers, as on the slides)

Project title: **A Multimodal Framework for Agricultural Drought Monitoring and Prediction using Weather Time Series, Satellite Imagery, and UAV Semantic Segmentation**.
Team: G. Aryan Rajesh (123CS0020), M. Rovan (123CS0036), M. Harsha Vardhan (123CS0044). Guide: Dr. M. Naresh Babu, IIITDM Kurnool.

**Idea in one line:** weather predicts where to look → satellite confirms which fields are stressed → drone measures the damage, in both seasons.
- Kharif chain: weather + satellite → UAV.
- Rabi chain: rabi module → UAV.

## 8 study sites
Somandepalle, Lepakshi, Cholasamudram (Sri Sathya Sai), Sira (Tumakuru, Karnataka), Peddakadubur, Dinnedevarapadu, near Balaji Villas, near Pullaiah College (Kurnool).
Each site is a UAV survey field. There are 6 independent weather cells.

## Data
| Source | Content | Period |
|---|---|---|
| AgERA5 | daily rain, ET0, temperature, humidity | 1979–2026 |
| Sentinel-2 (Earth Engine, `COPERNICUS/S2_SR_HARMONIZED`) | 10 m NDVI: 240 fields (kharif), 1,350 points (rabi) | 2019–2026 |
| ERA5-Land | root-zone soil moisture, 45 points | 1990–2023 |
| MODIS MOD13Q1 | 250 m NDVI on cropland | 2000–2026 |
| District crop statistics (APY, India Data Portal) | rabi yield of all crops | 1997–2022 |
| Drought G.O.s | declared mandals (comparison only) | 2023–2026 |
| UAV (DJI Mini 4 Pro), flown by Dr. M. Naresh Babu & team | 2,254 photos, 10 locations → 14 orthomosaics | Nov 2025 – Jan 2026 |

---

## 1. Weather module (kharif + year-round)
**Label:**
- SPEI-30 drought class: healthy ≥ 0, moderate −1.5 to 0, severe < −1.5.
- SPEI is computed from the 30-day water balance (rain − ET0), with a log-logistic fit per site and month on 1979–2011 only, then frozen.

**Split:** train 1979–2011, validation 2012–2018, test 2019–2026. The split is purged: no sample crosses a boundary.

**Inputs:**
- a 30-day window of 9 weather variables;
- SPEI history;
- day of year;
- water-balance ensemble probabilities (physics).

**Models:** RF, XGBoost, LSTM, Transformer (5 seeds each), combined by soft voting into the hybrid.

**Results (macro-F1, test 2019–2026):**

| | 7 d | 15 d | 30 d |
|---|---|---|---|
| Persistence | 0.662 | 0.515 | 0.424 |
| Water-balance ensemble (physics) | 0.783 | 0.592 | 0.375 |
| XGBoost | 0.814 | 0.631 | 0.418 |
| **Hybrid ensemble** | **0.819** | **0.651** | 0.424 |

- At 15 days the hybrid beats the physics model significantly (95% CI [+0.010, +0.099]).
- 80% of severe days are caught 7 days ahead (false-alarm ratio 0.34).
- There is no skill at 30 days.

## 2. Satellite module (kharif)
**Fields:** in Google Earth Engine, within 3 km of each site:
1. WorldCover cropland;
2. Dynamic World removes built-up land, trees and water;
3. keep only land with a crop cycle in ≥ 2 of 7 years;
4. remove edges;
5. split into SNIC segments (0.3–10 ha);
6. randomly sample 240 fields (20 rainfed + 10 irrigated per site).

**Label:** SVI = (NDVI − normal) / spread, per field and day of year.
- The normal comes from 2019–23, with the labelled year left out.
- Classes use the same cut-offs as SPEI: normal ≥ 0; below normal −1.5 to 0; strongly below < −1.5.
- VCI is kept only as an extra input.

**Lag analysis** (correlation ρ, chosen on 2019–23, checked on 2024–26):
- Rainfed: SPEI-30 with a 10-day lag, 0.48 → 0.45.
- Irrigated: SPEI-60 with a 15-day lag, 0.24 → 0.29.

Crops react about 10 days after weather drought.

**15-day crop-condition forecast (test 2025–26):**

| | Macro-F1 | Onset AUC |
|---|---|---|
| Persistence (rule: same as today) | 0.65 | 0.50 |
| Logistic (baseline, onset only) | – | 0.72 |
| **XGBoost (ours: satellite + weather-module forecast)** | **0.67** | **0.76** |

- Gain over persistence: 95% CI +0.012 to +0.040 (macro-F1).
- Gain over logistic: 95% CI +0.02 to +0.06 (onset).
- Random Forest on the same inputs scores 0.67 / 0.73.
- The deployed system (`satellite_module/src/system/run_system.py`) uses XGBoost.

**Replay against government drought orders** (kharif 2023–26, 8 sites, one site in one season = one case):
- Rules: weather 7 days late, satellite 2 days late. Alarm if SPEI-30 < −1. Confirmed if the median rainfed SVI < −1 over ≥ 5 clear fields.
- **Severe: 9 / 9 confirmed**, 5 / 9 before the order (up to 22 days before, in 2026).
- Moderate: 3 / 6. The 3 misses are Cholasamudram, Lepakshi and Somandepalle in 2025. Rain was near normal and crops stayed green; the weather alarm fired on 24 Sep, but the satellite saw no stress.
- The 4 severe cases confirmed late (3–18 days after the order) were delayed by monsoon clouds: too few clear images.
- Not declared: 1 false alarm in 11 (Kurnool Rural 2023, next to Kurnool Urban, which was declared severe). Removed from the slide by user choice; it is still in the report.

## 3. Rabi module (Oct–Apr)
**Why separate:**
1. Rabi crops grow on stored soil water; in-season rain is unrelated to yield (ρ = +0.01).
2. The kharif chain confirmed none of the 4 severe rabi cases.
3. There are too few rabi declarations to train on.
4. Harvest looks like drought (greenness falls at harvest too).

**Three parts:**
1. **DD (soil dryness):**
   - Data: ERA5-Land root-zone soil moisture (0–100 cm), 45 points, 1990–2023.
   - Method: compare each day with the same date in 1990–2004. A deficit day is one in the driest 20%. DD = share of 1 Nov – 28 Feb days in deficit.
   - This is a simple indicator, with no training.
2. **MODIS greenness (VPCT):** the Dec–Feb mean NDVI percentile against 2000–2009. It confirms crop stress; no training.
3. **Random forest (detect_v6):**
   - Data: 1,350 random Sentinel-2 points, AgERA5 weather, ERA5-Land soil, 39 features.
   - Event: an established crop dries ≥ 21 days before its normal drying date.
   - Warns 7 / 15 / 30 days ahead.
   - Chosen over logistic regression and XGBoost on validation 2022. Train 2019–21, test 2023–25.

**District results (Kurnool + Anantapur, 2000–2022, 46 district-seasons, 11 loss seasons; loss = yield ≥ 15% below the previous 5 seasons):**

| Indicator | AUC | ρ with yield |
|---|---|---|
| DD (soil), by 15 Dec | 0.80 | −0.58 |
| MODIS greenness (2010–22) | 0.83 | +0.70 |
| SPEI (rain) | 0.60 | +0.44 |

**Field results (30 days ahead, test 2023–25, 108 events):**

| Warnings from | Precision | Caught |
|---|---|---|
| Random guess | 0.24 | – |
| Random forest, all | 0.54 | 30 / 108 |
| Random forest, dry soil | 0.67 | 20 / 108 |

- Lead time is 16 days. Wet-soil warnings are only 38% correct; they are mostly harvest.
- 7-day precision is 0.25 and 15-day precision is 0.31 (15 days used the weighted ensemble and caught only 9 events). 30 days is the best horizon.
- Why the field numbers are low:
  - The threshold favours precision (F0.5 on 2022), so warnings are few but sure.
  - Harvest also drops NDVI.
  - Only 2023 was a dry test season.
- Code: `rabi_module_v2/src/detect/causal_events_v6.py`. Docs: `rabi_module_v2/docs/DETECT_v6.md`, `COMBINED_v1.md`, `MODIS_TEST_v1.md`.

## 4. UAV module
**Data:**
- DJI Mini 4 Pro, flown at 10–12 m, 1.2–1.5 m/s, gimbal −90°, 70–80% overlap, east–west lines.
- 14 orthomosaics made in Agisoft Metashape, from 3.8 million to 2.86 billion pixels.
- GSD 1.68–5.55 mm (9 fields), and 3.6–18.5 mm for the 5 Peddakadubur fields (built from low-resolution frames).

**Labels:**
- Pseudo-labels: 512 px window, stride 256, dense CRF; confidence below 0.55 → ignore.
- Corrected in our own annotation tool (Tkinter/Matplotlib/OpenCV).
- 279 clean tiles: 224 train, 55 validation.
- Each epoch: 500 clean + 500 pseudo-labelled + 1,000 synthetic samples, weighted 8 / 0.5 / 1.

**Model:**
- 9-channel SegFormer-B2: RGB plus vegetation indices (ExG, ExGR, CIVE, GLI, NGRDI, V).
- 7 classes: background, soil, crop, cotton boll, green weed, dry weed, drought.
- Loss = focal + 2.5·Dice + Lovász; drought class weight 10, cotton boll 8.
- AdamW, lr 2e-6, 10 epochs.
- Logit adjustment: drought −1.2, dry weed −0.4.

**Validation (55 tiles):**
- mIoU 65.1%, pixel accuracy 82.7%, drought F1 79.2%.
- Per-class IoU: soil 74, crop 73, cotton 53, green weed 71, dry weed 54, drought 65.

**Stitching:**
- 1024 px tiles, stride 768, Hann-weighted blending.
- A rolling band uses 1.5 GB instead of 80 GB of memory.

**Stress flag:** D > 0.15 or (N > 0.10 and C > 0.25).
- D = drought share of the crop area.
- N = the same share, outside harvest zones.
- C = share of drought in the largest drought region.

**Field result:** 3 fields flagged (Lepakshi 71.5%, Sira 54.3%, Somandepalle 25.5% drought). These are exactly the southern fields the survey team judged drought-hit; no Kurnool field is flagged.

## Limitations (as on the conclusion slide)
- There is no field-level drought ground truth; government orders and yields are indirect.
- There is no skill 30 days ahead (kharif), and few dry rabi test seasons.
- UAV results come from validation tiles only; RGB shows visual stress, not its cause.

## Future work
- Run the full kharif / rabi → UAV loop in rabi 2026-27.
- Finish the 127-tile independent UAV test, with field confirmation.
- Build a trend-robust greenness reference; use multispectral or thermal drone imagery.
- Correct the 2017–2018 Sentinel-2 images ourselves, to add 2 more seasons.

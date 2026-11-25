# Viva explanations (questions asked in the chat, with simple answers)

Each answer has a plain explanation and a **Viva line** (Telugu-English) to say out loud.

---

## Basics

### What is AUC?
- Pick one random bad (loss) season and one random normal season. AUC is how often the bad one gets the higher risk score.
- 0.5 = coin toss, 1.0 = perfect, 0.80 = good.
- It is **not** "80% accuracy". It only measures **ranking**: do bad seasons come above good ones?

**Viva line:** "Random ga oka bad season, oka good season teesukunte, 80% times bad season ke December 15 naatiki matti ekkuva dry ga undi."

### What is ρ (rho)?
- Correlation: do two numbers go up and down together? 0 = no link; 0.2–0.3 weak; 0.4–0.6 medium; 0.7+ strong.
- Negative ρ (DD vs yield −0.58): the more soil deficit, the lower the yield.

### What is Random Forest (RF)?
- Many decision trees (500). Each tree asks yes/no questions on the data, and all the trees vote.
- Like asking 500 doctors and taking the majority.
- It suits table-like data, works with small data, and shows which feature matters.

**Viva line:** "Random Forest ante chaala decision trees ni kalipi, vaati majority vote tho prediction ichhe ML model."

### Precision, recall, lead time
- **Precision:** of 100 warnings, how many were right.
- **Recall:** of all real events, how many were warned.
- **Lead time:** how many days before the event the warning came.

---

## Rabi module

### What does "DD by 15 December, AUC 0.80, direct ranking, not ML" mean?
- DD = share of rabi days with soil drier than normal, using data only up to 15 Dec.
- Seasons with high DD later had yield loss.
- **No model was trained.** Seasons were simply sorted by DD, so yields were never used to build it and the score is honest.

**Viva line:** "Training lekunda, December 15 naati soil deficit tho, nijamaina yield-loss season 80% times normal season kanna dry ga kanipistundi."

### The three parts and their order
- **DD:** is the district's soil dry? Early season risk (the cause).
- **MODIS:** did the crops really lose greenness? Confirmation (the effect).
- **Random forest:** which fields will dry early? A field list for the drone.
- They run **in parallel**, not as a strict filter: the RF runs on all fields, and each warning gets a soil-dry / soil-wet tag.
- DD and MODIS are **not** inputs to the RF. The RF uses Sentinel-2 NDVI, AgERA5 weather and raw ERA5-Land soil moisture.

**Viva line:** "District level lo DD season risk chupistundi, MODIS history tho validate chesam. Field level lo Random Forest early-drying warnings istundi, prathi warning ki soil dry/wet tag."

### Random forest inputs (39 features)
- Satellite: latest NDVI and its age, peak so far, drop from the peak, 15/30-day change, anomaly vs the field's normal curve.
- Field history: normal drying day, day of season, irrigated flag, crop-timing guess (early/mid/late peak).
- Weather (8-day delay): June–Nov rain vs normal, rain since 1 Oct and in the last 60 days, ET0, Tmax, SPEI.
- Soil (8-day delay): root-zone moisture, surface minus root-zone moisture.
- Output: the probability that the field dries early within 30 days. Above the threshold, it gives a warning.

### Why only RF? Why not other models?
- LR, RF, XGBoost and 2 ensembles were compared. RF won on validation 2022 at 7 and 30 days; at 15 days the weighted ensemble was slightly better.
- No LSTM or other deep model: only about 7 seasons and 108 test events. A deep model would memorise the data.

**Viva line:** "Logistic Regression, Random Forest, XGBoost compare chesam. Validation lo Random Forest best. Data takkuva unnanduku deep learning vaadaledu, overfitting avutundi."

### Why are the field numbers low? (precision 0.54, recall 0.28)
- **Random guess = 0.24** (only 108 of 453 field-seasons had an event), so 0.54 is about 2× chance. Dry-soil warnings are 0.67.
- **Low recall is by design.** The threshold maximises F0.5 on 2022 (favours precision), because each warning = a drone trip.
- **Harvest looks like drought.** Wet-soil warnings are only 38% right.
- **Only 2023 was dry** among the test seasons.
- Don't hide recall: precision alone can be faked by warning rarely.

**Viva line:** "Random ga 24% correct, mana model 54%, soil dry unte 67%. Drone trips waste kakunda takkuva warnings set chesam, anduke 108 lo 30 maatrame pattukundi."

### Why 30 days and not 15?
- 7 days: precision 0.25, 20 caught.
- 15 days: weighted ensemble, precision 0.31, only 9 of 108 caught.
- 30 days: precision 0.54, 30 caught, 16-day lead. It is best on every measure and gives time to plan flights.

---

## Satellite module

### The lag table (correlation of SVI with SPEI)
- SPEI = weather drought; SVI = crop condition from the satellite.
- **Rainfed (SPEI-30, lag 10 d): 0.48 → 0.45.** Last 30 days of weather drought shows in the crop 10 days later; the result is stable on new years.
- **Irrigated (SPEI-60, lag 15 d): 0.24 → 0.29.** The link is weak.

### Why 60 days and why low correlation for irrigated?
- **60 days:** borewells, tanks and canals carry crops through short dry spells. Only a long drought (about 2 months) dries the water sources. The 60-day window was chosen by the data. SPEI-60 and SPEI-90 scored almost the same.
- **Low ρ:**
  - The farmer's irrigation cuts the weather → crop link.
  - Groundwater levels are not in SPEI.
  - Electricity, labour and cost also matter.
  - The irrigation flag is not 100% correct.
- Even rainfed is only 0.48, because soil, crop type, sowing date and pests also matter.

**Viva line:** "Irrigated fields lo raithu neeru pedatharu, borewell level manaki teliyadu, kabatti weather–crop connection weak. Anduke satellite tho crop ni direct ga confirm chestham."

### Persistence, logistic, XGBoost (15-day forecast)
- **Persistence = a rule, no training:** "in 15 days the class = today's class". If there is no label today (clouds), it falls back to climatology.
- **Logistic = a simple trained baseline**, for onset only.
  - Trained on fields that are healthy today.
  - Inputs: today's SVI, the 15-day SVI change, season, image age.
  - It answers yes/no (will it turn stressed?), so it has **no macro-F1**.
  - It is not used in the final system; it exists only for comparison.
- **XGBoost = our model:** satellite history + the weather-module forecast.
- **Macro-F1:** how well all 3 classes are predicted. **Onset AUC:** how well a healthy field turning stressed is ranked.
- Results: persistence 0.65 / 0.50; logistic – / 0.72; XGBoost 0.67 / 0.76. Gains are real (95% CI above zero).

**Example (made up, for understanding):**
- 1 Sep: SVI −0.2 (normal), slowly falling, and the weather forecast says no rain (SPEI −1.3).
- Predictions:
  - Persistence says "normal".
  - Logistic gives 0.45.
  - XGBoost gives 0.80.
- 16 Sep: SVI −1.1, so the field is now below normal. Only XGBoost warned strongly.

**Viva line:** "Persistence 'ippudu unnatte untundi' ane rule; kotha drought ni eppudu cheppaledu (0.50). Mana XGBoost weather forecast kooda chustundi, kabatti onset ni 0.76 tho cheptundi."

### Replay results (why some cases differ)
- **4 example graphs:**
  - Balaji Villas 2023, severe, confirmed.
  - Lepakshi 2026, severe, confirmed before the order.
  - Dinnedevarapadu 2026, severe, 22 days before the order.
  - Somandepalle 2025, moderate, missed.
- The table counts all 8 sites, kharif 2023–26.
- **3 moderate missed (2025):** the weather alarm fired on 24 Sep, but crops stayed green. Both weather and satellite must agree.
- **4 severe late (3–18 days):** monsoon clouds; confirmation needs ≥ 5 clear fields. Pullaiah 2026 confirmed from only 1 date.
- **1 false alarm (Kurnool Rural 2023):** next to Kurnool Urban, which was declared severe that season, so the dryness was likely real. The government declares by mandal boundary.

### Why the plan changed from the mid-review
| Mid-review | Final | Why |
|---|---|---|
| 40 hand-drawn fields | 240 random fields | No selection bias |
| NDVI, NDMI, SAVI, VCI | One index: SVI (+ SPEI link) | VCI is unstable with few years; the indices repeat the same greenness signal |
| Transformer classifier | RF / XGBoost + baselines | Few years and droughts; labels came from the same indices (circular) |
| Test 2025 only | Replay 2023–26 vs G.O.s | One test year is not enough; G.O.s are independent |

### Why is there so little data for satellite and rabi?
- Sentinel-2A launched in 2015 and 2B in 2017. The atmosphere-corrected product we use (Level-2A, `S2_SR_HARMONIZED`) starts 28 Mar 2017, and its 2017–18 coverage is not global, so we use 2019 onward = 7 seasons. Level-1C (2015+) has no atmosphere correction, and mixing products would spoil the SVI normal.
- Drought does not come every year: 2023 and 2026 kharif; only 2023 was dry in the rabi test.
- Monsoon clouds: only 51% of kharif site-dates have clear images (rabi 96%).
- Rabi yields: one number per district per season. 2 districts × 23 seasons = 46 points, 11 loss seasons. APY data ends in 2022–23.
- There is no field drought survey, so labels come from the satellite; checks use G.O.s and yields.

**Viva line:** "Weather 1979 nundi undi, kaani field-level Sentinel-2 2019 nundi, 7 seasons. Drought seasons rendu moodu maatrame. Anduke simple models, chronological testing, confidence intervals."

### Field segments (Earth Engine)
WorldCover cropland → Dynamic World removes built-up, trees and water → crop cycle in ≥ 2 of 7 years → edges removed → SNIC segments (0.3–10 ha) → 240 random fields.
SNIC = a superpixel method that groups similar neighbouring pixels into field-like pieces. We used it because no surveyed field maps exist.

---

## Model-choice summary (why each module uses its model)
| Module | Compared | Chosen |
|---|---|---|
| Weather | RF, XGBoost, LSTM, Transformer, ensemble | Soft-voting hybrid ensemble |
| Satellite | RF, XGBoost + persistence / logistic / climatology baselines | XGBoost |
| Rabi field | LR, RF, XGBoost + 2 ensembles | Random forest (30 days) |
| UAV | DeepLabV3+, SegFormer-B2, SegFormer-B3 | SegFormer-B2 (V11) |

Rule in every module: choose on validation data; use test data only at the end.

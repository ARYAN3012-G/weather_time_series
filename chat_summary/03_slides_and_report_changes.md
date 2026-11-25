# Slides and report: what was done and why

## Final slide deck (29 slides, `latex/presentation_final.tex`)

**Theme "clean card":**
- Colours: white background, teal accent `#0F766E`, amber `#B45309` for warnings, green `#15803D` for key results.
- Light cards with a left stripe, using three boxes: `card`, `warncard`, `keycard`.
- A thin line above the footer; the footer reads "IIITDM Kurnool | Multimodal Agricultural Drought Monitoring and Prediction  n / N".
- Source Sans Pro font, 16:9.
- The older navy version is in `presentation_navy_backup.tex`.

**Order:**
1. Title
2. Introduction + 8 sites
3. Research gaps (4 cards)
4. Problem statement + objectives O1–O5
5. Data sources
6. Overall architecture (diagram)
7. 5.1 Literature – weather
8. 5.2 Weather method
9. 5.3 Weather architecture
10. 5.4 Weather results
11. 5.5 Literature – satellite
12. 5.6 Satellite architecture
13. 5.7 Satellite method and results
14. 5.8 Replay vs government orders
15. Why a separate rabi module
16. 6.1 Literature – rabi
17. 6.2 Rabi architecture: three parts
18. 6.3 Rabi results
19. 7.1 Literature – UAV
20. 7.2 UAV data and photo stitching
21. 7.3 UAV pipeline
22. 7.4 Labels and 9 channels
23. 7.5 Model and training
24. 7.6 Prediction stitching and stress decision
25. 7.7 UAV validation results
26. 7.8 UAV results on 14 fields
27. 8. Conclusion, limitations, future work
28. References (22)
29. Thank you

### Changes made in this chat (latest first)

**Rabi results (slide 18):**
- The field table now has one row per method: random guess 0.24 / RF all 0.54 (30/108) / RF dry soil 0.67 (20/108).
- "Honest note" was replaced by "Why field numbers are low" (3 reasons).

**Satellite method and results (slide 13):**
- The left card is now "Method": SVI, classes, lag, forecast, replay.
- The results table has one row per model: Persistence (same as today) / Logistic (baseline) / XGBoost (ours). Columns are Macro-F1 and Onset AUC.
- The correlation table says clearly that the values are correlations ρ with lags.

**Satellite architecture diagram (slide 12):**
- The farmland box shows that "Dynamic World removes built-up, trees, water".
- The segments box shows "crop cycle ≥ 2 of 7 yrs, edges removed, SNIC".
- Source: `make_satellite_slide.py`.

**Replay slide (slide 14):**
- The graph is bigger; the rules box is removed.
- The results card is "Results: 8 sites, kharif 2023–26", with Severe 9/9 (5/9 before the order) and Moderate 3/6.
- The "Not declared 1/11" row was removed by user choice.
- "Why some cases differ": 3 moderate missed (2025) and 4 of 9 severe confirmed late (clouds).

**Rabi section:**
- The architecture image slide was removed, and its details were merged into "6.2 Architecture: Three Parts" (DD, MODIS, random forest cards plus a "Why all three" box).
- The old method / results / field-warning slides were simplified into one results slide.

**Literature tables (slides 7, 11, 16, 19):** rewritten in simple words, with verified facts and recent papers. See `04_references_checked.md`.

**Satellite model name fixed:** the slide said "RF 0.68 / onset 0.76", which mixed two models. Corrected to XGBoost 0.67 / 0.76 (the deployed model); RF on the same inputs scores 0.67 / 0.73. Fixed in the report too.

**Earlier in the chat:**
- Removed slides: "What changed since mid-review" and Contents.
- Removed the APSCHE line and the "Hand-off files" line.
- Fixed footer line; aligned cards with `columns[T,onlytextwidth]`.
- References in 2 columns (11 + 11), with the list font set via `\setbeamerfont{itemize/enumerate body}`.
- UAV step images (stitching, labels, channels); the professor's deck images in `figures/prof/` (lawnmower path credited to Humboldt State Univ.).
- No phone numbers or emails anywhere.

## Final report (`latex/main_v2.tex`, 18 pages)

**Structure:**
1. Front page: "End-Semester Review Report".
2. Abstract.
3. Introduction.
4. Literature: Table 1 as 4 boxed sub-tables, (a) Weather, (b) Satellite, (c) Rabi, (d) UAV, with the headings outside the tables.
5. Gaps / problem / objectives.
6. Study area, data and architecture.
7. One section per module.
8. Conclusion.

**Style:**
- No table of contents.
- All tables boxed in a plain style.
- Formulas in an `align` block.
- `placeins` + float settings to avoid white gaps.
- Section 8 "Implementation and Work Status" removed (covered in future work).
- "Why a separate rabi module" has 4 points.

**UAV facts in the report:**
- Surveys by Dr. M. Naresh Babu and his team.
- Flights Nov 2025 – Jan 2026; Metashape processing 30 May – 1 June 2026.
- 9 orthomosaics from DJI FC8482 photos (8064×4536), GSD 1.68–5.55 mm.
- 5 Peddakadubur orthomosaics from 1280×720 frames (camera not recorded), GSD 3.6–18.5 mm.

**Spelling:** "Somandepalle" (not "Somandhapalle").

**Changes in this chat:**
- Literature rows updated (Prodhan 2021, Danodia 2021, Kulkarni 2020, Niu 2024, corrected Shahi 2023).
- Bibliography fixed.
- "Why separate rabi" now cites Kulkarni 2020 ("soil moisture is the largest contributor"); the unsupported "fail late" claim was removed.
- Satellite sentence now names XGBoost and RF on the same inputs (0.67 / 0.73).

## User preferences seen in the chat
- Simple English; Telugu-English explanations for viva practice.
- No congested slides, no white gaps, perfect alignment, nothing outside the boxes.
- Low text on slides but complete information; clear tables (one row = one thing).
- Honest results: keep weak numbers (e.g., recall) and explain them, instead of hiding them.
- Literature: recent papers (2020+), keeping older ones only where we really use their method or index.

# SPEI recalibration report (v2 -> v3)

- Accumulation: 30-day rolling sum of P - PET (FAO-56)
- Distribution: 3-parameter log-logistic, L-moments, per site and calendar month
- Reference (calibration) period: 1979–2011 (= TRAIN years). Parameters frozen in `results/v3/spei_calibration.json` and applied unchanged to validation (2012–2018) and test (2019+).
- v2 SPEI: per-calendar-month standardisation fitted on the full 1979–2026 record (inferred: original preprocessing script is not in the repository). Mean v2 SPEI is ~0 over the whole record, i.e. test years influenced the calibration.

## Class distribution (% of days, 7 sites pooled)

| period | version | days | Healthy_pct | Moderate_pct | Severe_pct |
|---|---|---|---|---|---|
| train | old | 60120 | 49.1600 | 42.6500 | 8.2000 |
| train | new | 60120 | 49.2100 | 44.7100 | 6.0800 |
| val | old | 12785 | 46.9900 | 44.7900 | 8.2100 |
| val | new | 12785 | 47.6600 | 45.2600 | 7.0900 |
| test | old | 13667 | 65.5700 | 29.1400 | 5.2900 |
| test | new | 13667 | 65.2700 | 30.2200 | 4.5100 |

## Mean SPEI by period (7 sites pooled)

| period | SPEI_old mean | SPEI_old std | SPEI mean | SPEI std |
|---|---|---|---|---|
| train | -0.0850 | 1.0020 | 0.0000 | 1.0080 |
| val | -0.1250 | 0.9940 | -0.0400 | 1.0370 |
| test | 0.2590 | 0.9750 | 0.3770 | 1.0740 |

## Label agreement old vs new

| period | agreement |
|---|---|
| train | 0.9472 |
| val | 0.9500 |
| test | 0.9629 |

## Label transitions old -> new (train)

| old | Healthy | Moderate | Severe |
|---|---|---|---|
| Healthy | 28706 | 847 | 0 |
| Moderate | 881 | 24673 | 86 |
| Severe | 0 | 1358 | 3569 |

## Label transitions old -> new (val)

| old | Healthy | Moderate | Severe |
|---|---|---|---|
| Healthy | 5855 | 153 | 0 |
| Moderate | 238 | 5437 | 52 |
| Severe | 0 | 196 | 854 |

## Label transitions old -> new (test)

| old | Healthy | Moderate | Severe |
|---|---|---|---|
| Healthy | 8775 | 186 | 0 |
| Moderate | 145 | 3803 | 35 |
| Severe | 0 | 141 | 582 |

## Per site and period

| field | period | days | spei_old_mean | spei_old_std | spei_new_mean | spei_new_std | corr_old_new | label_agreement |
|---|---|---|---|---|---|---|---|---|
| Somandepalle | test | 2731 | 0.2129 | 1.0284 | 0.3373 | 1.1200 | 0.9806 | 0.9590 |
| Somandepalle | train | 12024 | -0.0766 | 0.9840 | 0.0023 | 1.0008 | 0.9804 | 0.9447 |
| Somandepalle | val | 2557 | -0.1368 | 1.0086 | -0.0509 | 1.0447 | 0.9791 | 0.9445 |
| cholasamudram | test | 2737 | 0.2104 | 1.0305 | 0.3486 | 1.1163 | 0.9768 | 0.9620 |
| cholasamudram | train | 12024 | -0.0830 | 0.9854 | 0.0025 | 1.0012 | 0.9794 | 0.9406 |
| cholasamudram | val | 2557 | -0.1142 | 1.0034 | -0.0203 | 1.0528 | 0.9801 | 0.9562 |
| kurnool_balaji_villas | test | 2737 | 0.2492 | 0.9343 | 0.3113 | 1.0424 | 0.9799 | 0.9613 |
| kurnool_balaji_villas | train | 12024 | -0.0695 | 1.0127 | -0.0028 | 1.0184 | 0.9803 | 0.9518 |
| kurnool_balaji_villas | val | 2557 | -0.1451 | 1.0122 | -0.0934 | 1.0682 | 0.9851 | 0.9519 |
| kurnool_dinedevarapadu | test | 2731 | 0.3390 | 0.9253 | 0.4497 | 1.0068 | 0.9856 | 0.9674 |
| kurnool_dinedevarapadu | train | 12024 | -0.0948 | 1.0182 | -0.0021 | 1.0152 | 0.9842 | 0.9456 |
| kurnool_dinedevarapadu | val | 2557 | -0.1381 | 0.9573 | -0.0593 | 1.0022 | 0.9870 | 0.9488 |
| sira | test | 2731 | 0.2820 | 0.9454 | 0.4376 | 1.0725 | 0.9806 | 0.9648 |
| sira | train | 12024 | -0.1032 | 1.0068 | 0.0015 | 1.0040 | 0.9791 | 0.9534 |
| sira | val | 2557 | -0.0892 | 0.9854 | 0.0229 | 1.0143 | 0.9742 | 0.9488 |

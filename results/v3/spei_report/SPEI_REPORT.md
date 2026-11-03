# SPEI recalibration report (v2 -> v3)

- Accumulation: 30-day rolling sum of P - PET (FAO-56)
- Distribution: 3-parameter log-logistic, L-moments, per site and calendar month
- Reference (calibration) period: 1979–2011 (= TRAIN years). Parameters frozen in `results/v3/spei_calibration.json` and applied unchanged to validation (2012–2018) and test (2019+).
- v2 SPEI: per-calendar-month standardisation fitted on the full 1979–2026 record (inferred: original preprocessing script is not in the repository). Mean v2 SPEI is ~0 over the whole record, i.e. test years influenced the calibration.

## Class distribution (% of days, 7 sites pooled)

| period | version | days | Healthy_pct | Moderate_pct | Severe_pct |
|---|---|---|---|---|---|
| train | old | 84168 | 48.9200 | 42.7800 | 8.3000 |
| train | new | 84168 | 49.1300 | 44.8200 | 6.0600 |
| val | old | 17899 | 47.1600 | 44.6500 | 8.1900 |
| val | new | 17899 | 48.0000 | 44.9500 | 7.0600 |
| test | old | 19138 | 66.1900 | 28.9400 | 4.8600 |
| test | new | 19138 | 66.0300 | 29.6000 | 4.3700 |

## Mean SPEI by period (7 sites pooled)

| period | SPEI_old mean | SPEI_old std | SPEI mean | SPEI std |
|---|---|---|---|---|
| train | -0.0880 | 1.0070 | -0.0000 | 1.0090 |
| val | -0.1210 | 0.9820 | -0.0390 | 1.0270 |
| test | 0.2640 | 0.9640 | 0.3760 | 1.0630 |

## Label agreement old vs new

| period | agreement |
|---|---|
| train | 0.9455 |
| val | 0.9499 |
| test | 0.9623 |

## Label transitions old -> new (train)

| old | Healthy | Moderate | Severe |
|---|---|---|---|
| Healthy | 40059 | 1114 | 0 |
| Moderate | 1289 | 34574 | 148 |
| Severe | 0 | 2033 | 4951 |

## Label transitions old -> new (val)

| old | Healthy | Moderate | Severe |
|---|---|---|---|
| Healthy | 8240 | 201 | 0 |
| Moderate | 351 | 7570 | 71 |
| Severe | 0 | 274 | 1192 |

## Label transitions old -> new (test)

| old | Healthy | Moderate | Severe |
|---|---|---|---|
| Healthy | 12414 | 254 | 0 |
| Moderate | 222 | 5241 | 76 |
| Severe | 0 | 170 | 761 |

## Per site and period

| field | period | days | spei_old_mean | spei_old_std | spei_new_mean | spei_new_std | corr_old_new | label_agreement |
|---|---|---|---|---|---|---|---|---|
| Peddakadubur | test | 2734 | 0.4026 | 0.8456 | 0.5550 | 0.9414 | 0.9827 | 0.9693 |
| Peddakadubur | train | 12024 | -0.1308 | 1.0315 | -0.0004 | 1.0073 | 0.9827 | 0.9419 |
| Peddakadubur | val | 2557 | -0.0892 | 0.9329 | 0.0294 | 0.9779 | 0.9873 | 0.9515 |
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
| kurnool_pulliah | test | 2737 | 0.1520 | 1.0019 | 0.1959 | 1.0929 | 0.9868 | 0.9521 |
| kurnool_pulliah | train | 12024 | -0.0560 | 1.0109 | -0.0011 | 1.0134 | 0.9791 | 0.9406 |
| kurnool_pulliah | val | 2557 | -0.1370 | 0.9717 | -0.1040 | 1.0224 | 0.9832 | 0.9476 |
| sira | test | 2731 | 0.2820 | 0.9454 | 0.4376 | 1.0725 | 0.9806 | 0.9648 |
| sira | train | 12024 | -0.1032 | 1.0068 | 0.0015 | 1.0040 | 0.9791 | 0.9534 |
| sira | val | 2557 | -0.0892 | 0.9854 | 0.0229 | 1.0143 | 0.9742 | 0.9488 |

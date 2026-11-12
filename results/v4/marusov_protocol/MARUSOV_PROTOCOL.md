# Marusov et al. protocol on our data

Binary drought = SPEI(t+h) < -1.0. Scores computed per site on the TEST period (2019+) and summarised by the median across the 7 sites (as in the paper); mean ± std across seeds. `spei_only` = their input (index history); `weather_spei` = our input. Methodology comparison only — not comparable to the paper's monthly PDSI numbers.


## Horizon 7 days

| input | model | median roc_auc | median pr_auc | median f1 |
|---|---|---|---|---|
| spei_only | logreg_last | 0.918 | 0.639 | 0.576 |
| spei_only | lstm | 0.939 ± 0.006 | 0.676 ± 0.021 | 0.622 ± 0.026 |
| spei_only | majority | 0.500 | 0.117 | 0.000 |
| spei_only | persistence | 0.778 | 0.418 | 0.609 |
| spei_only | xgb | 0.919 ± 0.001 | 0.624 ± 0.003 | 0.614 ± 0.005 |
| weather_spei | lstm | 0.949 ± 0.004 | 0.719 ± 0.023 | 0.662 ± 0.018 |
| weather_spei | xgb | 0.960 ± 0.001 | 0.775 ± 0.002 | 0.703 ± 0.007 |

## Horizon 15 days

| input | model | median roc_auc | median pr_auc | median f1 |
|---|---|---|---|---|
| spei_only | logreg_last | 0.806 | 0.394 | 0.405 |
| spei_only | lstm | 0.809 ± 0.009 | 0.379 ± 0.024 | 0.415 ± 0.012 |
| spei_only | majority | 0.500 | 0.117 | 0.000 |
| spei_only | persistence | 0.687 | 0.277 | 0.456 |
| spei_only | xgb | 0.777 ± 0.005 | 0.347 ± 0.004 | 0.377 ± 0.004 |
| weather_spei | lstm | 0.873 ± 0.022 | 0.452 ± 0.029 | 0.484 ± 0.045 |
| weather_spei | xgb | 0.886 ± 0.002 | 0.487 ± 0.006 | 0.495 ± 0.004 |

## Horizon 30 days

| input | model | median roc_auc | median pr_auc | median f1 |
|---|---|---|---|---|
| spei_only | logreg_last | 0.701 | 0.239 | 0.324 |
| spei_only | lstm | 0.682 ± 0.018 | 0.228 ± 0.015 | 0.308 ± 0.019 |
| spei_only | majority | 0.500 | 0.117 | 0.000 |
| spei_only | persistence | 0.598 | 0.172 | 0.300 |
| spei_only | xgb | 0.587 ± 0.003 | 0.159 ± 0.003 | 0.218 ± 0.009 |
| weather_spei | lstm | 0.575 ± 0.047 | 0.153 ± 0.011 | 0.179 ± 0.027 |
| weather_spei | xgb | 0.706 ± 0.008 | 0.222 ± 0.006 | 0.217 ± 0.009 |

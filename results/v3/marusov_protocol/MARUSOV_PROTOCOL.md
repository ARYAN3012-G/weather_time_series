# Marusov et al. protocol on our data

Binary drought = SPEI(t+h) < -1.0. Scores computed per site on the TEST period (2019+) and summarised by the median across the 7 sites (as in the paper); mean ± std across seeds. `spei_only` = their input (index history); `weather_spei` = our input. Methodology comparison only — not comparable to the paper's monthly PDSI numbers.


## Horizon 7 days

| input | model | median roc_auc | median pr_auc | median f1 |
|---|---|---|---|---|
| spei_only | logreg_last | 0.906 | 0.549 | 0.527 |
| spei_only | lstm | 0.936 ± 0.005 | 0.659 ± 0.006 | 0.595 ± 0.012 |
| spei_only | majority | 0.500 | 0.101 | 0.000 |
| spei_only | persistence | 0.745 | 0.335 | 0.546 |
| spei_only | xgb | 0.910 ± 0.001 | 0.602 ± 0.003 | 0.578 ± 0.006 |
| weather_spei | lstm | 0.944 ± 0.003 | 0.650 ± 0.036 | 0.635 ± 0.011 |
| weather_spei | xgb | 0.959 ± 0.001 | 0.759 ± 0.004 | 0.683 ± 0.007 |

## Horizon 15 days

| input | model | median roc_auc | median pr_auc | median f1 |
|---|---|---|---|---|
| spei_only | logreg_last | 0.775 | 0.276 | 0.323 |
| spei_only | lstm | 0.823 ± 0.012 | 0.376 ± 0.027 | 0.407 ± 0.014 |
| spei_only | majority | 0.500 | 0.101 | 0.000 |
| spei_only | persistence | 0.650 | 0.184 | 0.344 |
| spei_only | xgb | 0.761 ± 0.002 | 0.313 ± 0.002 | 0.362 ± 0.008 |
| weather_spei | lstm | 0.868 ± 0.015 | 0.411 ± 0.019 | 0.445 ± 0.014 |
| weather_spei | xgb | 0.889 ± 0.003 | 0.437 ± 0.012 | 0.450 ± 0.019 |

## Horizon 30 days

| input | model | median roc_auc | median pr_auc | median f1 |
|---|---|---|---|---|
| spei_only | logreg_last | 0.657 | 0.174 | 0.252 |
| spei_only | lstm | 0.641 ± 0.019 | 0.178 ± 0.021 | 0.240 ± 0.015 |
| spei_only | majority | 0.500 | 0.101 | 0.000 |
| spei_only | persistence | 0.574 | 0.153 | 0.259 |
| spei_only | xgb | 0.612 ± 0.021 | 0.161 ± 0.011 | 0.238 ± 0.016 |
| weather_spei | lstm | 0.614 ± 0.032 | 0.144 ± 0.012 | 0.210 ± 0.026 |
| weather_spei | xgb | 0.699 ± 0.006 | 0.212 ± 0.008 | 0.227 ± 0.015 |

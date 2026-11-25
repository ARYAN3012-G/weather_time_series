# Rabi report figure audit — 23 November 2026

This edit changes reporting and figure generation, not fitted models or labels.

| Asset in latex/figures | Source / check | Action |
|---|---|---|
| fig1_system_architecture | Final drought_v1 workflow and earlier satellite tasks | Updated rabi box to distinguish regional soil-risk scores and satellite context; ERA5-Land input named explicitly. |
| fig4_rabi_module_architecture | forecast_v1_A1 results and forecast rules | Retained as the NDVI experiment, not the final drought architecture. Corrected target wording and validation-based tree-count selection. |
| fig5_operational_timing | Issue-date definition | Removed blanket horizon-minus-delay statement for rabi. Horizon begins at issue; caption distinguishes the soil experiment. |
| fig6_data_splits | NDVI chronological split | Retained its explicit NDVI scope. Caption directs readers to the distinct soil-model reference/training split. |
| fig7_rabi_forecast_rules | data/forecast/instances.csv.gz | Retained image-age/target rules and 233,020 / 342,900 eligibility. |
| res_r_month | forecast_v1_A1/predictions.csv.gz | Regenerated monthly MAE for the same historical evaluation rows. |
| res_r_scatter | forecast_v1_A1/predictions.csv.gz | Corrected axes from fixed “15-day change” to change from latest image; shared colour normalization across both panels. |
| res_r_examples | C3_points_timeseries.csv and saved predictions | Regenerated. Caption identifies sampling within preselected mandal/irrigation groups, not a representative random draw of all points. Predictions plotted at issue + 15 days. |
| res_r_importance | forecast_v1_A1/FORECAST_RESULTS.md | Regenerated. Corrected prose: temperature, 90-day rain and ET0 total 9.5% of gain, not water balance / 11%. Feature gain is not causal attribution. |
| fig8_rabi_early_drying_architecture | detect_v2 protocol and corrected event audit | Retained as a separate satellite-proxy model. Conclusion now includes always-warn comparison (107/108 vs 104/108 events; 453 vs 332 warned point-seasons). |
| fig9_rabi_soil_drought_architecture | drought_v1 model selections and operational workflow | Added: regional target, frozen reference, splits, t−8 cutoff, selected HGB variants, parallel satellite context and remaining validation limitation. |
| res_r_soil_new_onset | drought_v1/backtest/metrics.csv | Added selected-model vs conditional-transition precision/recall at 7/15/30 days. Explicitly post hoc; repeated cell-dates, not independent events. |
| res_r_soil_product_sensitivity | drought_v1/cross_product/by_season.csv | Added ERA5-Land 0/0/0 versus MERRA-2 15/2/0 dry cell-dates, out of 125 per season on 15-day targets. Neither is field truth. |

Saved NDVI predictions reproduce the rounded pooled MAE table: main 0.0511,
anomaly persistence 0.0623, persistence 0.0641 and climatology 0.0909.
The new soil tables and graphs are generated directly from saved CSV rows;
selected rows and source SHA-256 hashes accompany the Overleaf bundle in
`rabi_drought_audit/`.

The full PDF is rendered for layout review, with detailed checks of the rabi
figures, tables and captions. The scope of this update is rabi reporting;
unrelated weather, satellite and UAV results are not newly scientifically audited.

Reproduction order: regenerate rabi result figures with `make_results.rabi()`,
architecture figures with `make_diagrams` and `NO_TITLE=1`, then run
`update_rabi_drought.py` last. Compile `latex/main.tex`. The latter script
preserves the separate early-drying section and packages all referenced assets.

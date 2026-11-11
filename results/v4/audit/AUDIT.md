# Site audit

| field | latitude | longitude | era5_land_cell | rows | start | end | date_gaps | missing_weather_values |
|---|---|---|---|---|---|---|---|---|
| cholasamudram | 13.8000 | 77.6000 | (13.8, 77.6) | 17444 | 1979-01-30 | 2026-11-02 | 0 | 0 |
| kurnool_balaji_villas | 15.8000 | 78.1000 | (15.8, 78.1) | 17444 | 1979-01-30 | 2026-11-02 | 0 | 0 |
| kurnool_dinedevarapadu | 15.8000 | 78.0000 | (15.8, 78.0) | 17444 | 1979-01-30 | 2026-11-02 | 0 | 0 |
| kurnool_pulliah | 15.8000 | 78.1000 | (15.8, 78.1) | 17444 | 1979-01-30 | 2026-11-02 | 0 | 0 |
| lepakshi | 13.8000 | 77.6000 | (13.8, 77.6) | 17444 | 1979-01-30 | 2026-11-02 | 0 | 0 |
| Peddakadubur | 15.8000 | 77.3000 | (15.8, 77.3) | 17444 | 1979-01-30 | 2026-11-02 | 0 | 0 |
| sira | 13.8000 | 77.0000 | (13.8, 77.0) | 17444 | 1979-01-30 | 2026-11-02 | 0 | 0 |
| Somandepalle | 14.0000 | 77.6000 | (14.0, 77.6) | 17444 | 1979-01-30 | 2026-11-02 | 0 | 0 |

## Duplicate weather locations

- **lepakshi** and **cholasamudram** map to the same ERA5-Land grid cell (same_era5_cell=True); 8/9 weather features identical on >=99% of days; label agreement 0.9998. `lepakshi` is removed as an independent weather location; `cholasamudram` is kept (field metadata records that it also represents `lepakshi`).
- **kurnool_pulliah** and **kurnool_balaji_villas** map to the same ERA5-Land grid cell (same_era5_cell=True); 9/9 weather features identical on >=99% of days; label agreement nan. `kurnool_pulliah` is removed as an independent weather location; `kurnool_balaji_villas` is kept (field metadata records that it also represents `kurnool_pulliah`).

The two agricultural fields may be physically distinct; they are not distinct in the ERA5-Land weather data used here.

## Pairwise summary

| site_a | site_b | same_era5_cell | n_features_identical_99pct | precip_corr | label_agreement |
|---|---|---|---|---|---|
| cholasamudram | kurnool_balaji_villas | False | 0 | 0.3915 | 0.7127 |
| cholasamudram | kurnool_dinedevarapadu | False | 0 | 0.4493 | 0.7387 |
| cholasamudram | kurnool_pulliah | False | 0 | 0.3915 |  |
| cholasamudram | lepakshi | True | 8 | 1.0000 | 0.9998 |
| cholasamudram | Peddakadubur | False | 0 | 0.4894 |  |
| cholasamudram | sira | False | 0 | 0.8383 | 0.8755 |
| cholasamudram | Somandepalle | False | 0 | 0.9379 | 0.9218 |
| kurnool_balaji_villas | kurnool_dinedevarapadu | False | 0 | 0.9229 | 0.9068 |
| kurnool_balaji_villas | kurnool_pulliah | True | 9 | 1.0000 |  |
| kurnool_balaji_villas | lepakshi | False | 0 | 0.3915 | 0.7126 |
| kurnool_balaji_villas | Peddakadubur | False | 0 | 0.7930 |  |
| kurnool_balaji_villas | sira | False | 0 | 0.3881 | 0.6951 |
| kurnool_balaji_villas | Somandepalle | False | 0 | 0.4356 | 0.7415 |
| kurnool_dinedevarapadu | kurnool_pulliah | False | 0 | 0.9229 |  |
| kurnool_dinedevarapadu | lepakshi | False | 0 | 0.4493 | 0.7387 |
| kurnool_dinedevarapadu | Peddakadubur | False | 0 | 0.8732 |  |
| kurnool_dinedevarapadu | sira | False | 0 | 0.4451 | 0.7205 |
| kurnool_dinedevarapadu | Somandepalle | False | 0 | 0.5004 | 0.7688 |
| kurnool_pulliah | lepakshi | False | 0 | 0.3915 |  |
| kurnool_pulliah | Peddakadubur | False | 0 | 0.7930 |  |
| kurnool_pulliah | sira | False | 0 | 0.3881 |  |
| kurnool_pulliah | Somandepalle | False | 0 | 0.4356 |  |
| lepakshi | Peddakadubur | False | 0 | 0.4894 |  |
| lepakshi | sira | False | 0 | 0.8383 | 0.8756 |
| lepakshi | Somandepalle | False | 0 | 0.9379 | 0.9217 |
| Peddakadubur | sira | False | 0 | 0.4981 |  |
| Peddakadubur | Somandepalle | False | 0 | 0.5367 |  |
| sira | Somandepalle | False | 0 | 0.7967 | 0.8509 |

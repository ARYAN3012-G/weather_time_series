# Literature survey: weather, satellite and rabi modules

How to use this file:
- Each entry gives the reference, then **what they did**, then **how our work relates**. The "how it relates" lines can be turned straight into the literature-review paragraphs.
- ✔ = checked online on 22 Nov 2026 (title, authors, venue).
- ● = a standard, widely cited reference whose details are well established.
- **Before submitting**, open each DOI once and copy the exact page numbers into your reference style.

---

## A. Weather module: meteorological drought indices and forecasting (12 papers)

| # | Reference | What they did | How our work relates |
|---|---|---|---|
| A1 ● | McKee, T. B., Doesken, N. J., Kleist, J. (1993). The relationship of drought frequency and duration to time scales. *Proc. 8th Conf. on Applied Climatology*, AMS, 179–184. | Introduced the **Standardized Precipitation Index (SPI)**: rain over multiple time scales, standardised to a normal distribution. | SPI uses rain only. We use SPEI, which also counts evaporation demand, which matters in hot, semi-arid Rayalaseema. |
| A2 ● | Vicente-Serrano, S. M., Beguería, S., López-Moreno, J. I. (2010). A multiscalar drought index sensitive to global warming: the Standardized Precipitation Evapotranspiration Index. *Journal of Climate* 23(7), 1696–1718. | Introduced **SPEI**: climatic water balance (P – PET), fitted with a log-logistic distribution. | Our labels are SPEI-30. Unlike most applications, we calibrate on 1979–2011 only and freeze the parameters, so the test years cannot leak into the labels. |
| A3 ● | Mishra, A. K., Singh, V. P. (2010). A review of drought concepts. *Journal of Hydrology* 391(1–2), 202–216. | Classic review of drought types (meteorological, agricultural, hydrological, socio-economic) and indices. | Gives the framework for our chain: meteorological drought (Stage 1) leads to agricultural drought (Stages 2–3). |
| A4 ● | Allen, R. G., Pereira, L. S., Raes, D., Smith, M. (1998). *Crop evapotranspiration: guidelines for computing crop water requirements.* FAO Irrigation and Drainage Paper 56, Rome. | The FAO-56 Penman–Monteith reference evapotranspiration (ET0). | AgERA5 provides FAO-56 ET0, which we use for the water balance (P – ET0) behind SPEI. |
| A5 ● | Boogaard, H., Schubert, J., de Wit, A., et al. AgERA5: agrometeorological indicators from 1979 to present derived from reanalysis. Copernicus Climate Change Service (C3S) Climate Data Store. | Daily 0.1° agro-meteorological dataset derived from ERA5. | Our weather source (6 independent cells, 1979 – Nov 2026). Its ~8-day publication delay is built into all our operational tests. |
| A6 ● | Belayneh, A., Adamowski, J., Khalil, B., Ozga-Zielinski, B. (2014). Long-term SPI drought forecasting in the Awash River Basin in Ethiopia using wavelet neural network and wavelet support vector regression models. *Journal of Hydrology* 508, 418–429. | Machine-learning forecasting of SPI with wavelet-coupled neural networks and SVR. | An early ML drought-forecasting study. It evaluates against other ML models; we add persistence, Markov, climatology and a physical water-balance baseline. |
| A7 ● | Dikshit, A., Pradhan, B., Huete, A. (2021). An improved SPEI drought forecasting approach using the long short-term memory neural network. *Journal of Environmental Management* 283, 111979. | LSTM forecasting of SPEI in Australia. | We also test LSTM and Transformer models, but tree models with physics inputs (hybrid XGBoost) performed best on our data. |
| A8 ✔ | Marusov, A., Grabar, V., Maximov, Y., et al. (2024). Long-term drought prediction using deep neural networks based on geospatial weather data. *Environmental Modelling & Software* 179; arXiv:2309.06212. | Deep learning for multi-month drought (PDSI) prediction from weather grids. | We reproduce their protocol style (binary drought, per-site scores) as a comparison: weather inputs raise 15-day ROC-AUC from 0.81 to 0.89. |
| A9 ✔ | Prodhan, F. A., Zhang, J., Hasan, S. S., Pangali Sharma, T. P., Mohana, H. P. (2022). A review of machine learning methods for drought hazard monitoring and forecasting: current research trends, challenges, and future research directions. *Environmental Modelling & Software* 149. | Review of ML in drought forecasting: trends, challenges, future directions. | Supports our choice of RF/XGBoost/LSTM and motivates our focus on leakage control and honest baselines, which reviews flag as weak points. |
| A10 ✔ | Aadhar, S., Mishra, V. (2017). High-resolution near real-time drought monitoring in South Asia. *Scientific Data* 4, 170145. doi:10.1038/sdata.2017.145 | Built a 0.05° bias-corrected, near-real-time drought-monitoring dataset for South Asia. | Indian context for near-real-time drought monitoring. We work at site level with an explicit publication delay rather than a gridded monitor. |
| A11 ● | Chen, T., Guestrin, C. (2016). XGBoost: a scalable tree boosting system. *Proc. 22nd ACM SIGKDD*, 785–794. | The XGBoost gradient-boosting algorithm. | Main model in all three modules. |
| A12 ● | Breiman, L. (2001). Random forests. *Machine Learning* 45(1), 5–32. | The Random Forest algorithm. | Second model in the weather and satellite modules. |

**Gap our weather module addresses:**
- Many ML drought-forecasting studies standardise SPEI over the full record, include short horizons that persistence already solves, and compare only ML models against each other.
- We freeze the calibration on training years, purge the split boundaries, add persistence and a physical water-balance ensemble as baselines, and report bootstrap confidence intervals and leave-one-cluster-out tests.

---

## B. Satellite module: vegetation-based agricultural drought monitoring (12 papers)

| # | Reference | What they did | How our work relates |
|---|---|---|---|
| B1 ● | Tucker, C. J. (1979). Red and photographic infrared linear combinations for monitoring vegetation. *Remote Sensing of Environment* 8(2), 127–150. | Founded the use of red / near-infrared indices such as **NDVI** for vegetation monitoring. | NDVI from Sentinel-2 is our basic crop-condition signal. |
| B2 ● | Kogan, F. N. (1995). Application of vegetation index and brightness temperature for drought detection. *Advances in Space Research* 15(11), 91–100. | Introduced the **Vegetation Condition Index (VCI)**, NDVI scaled between its historical minimum and maximum. | We show VCI is unstable with only 4–5 reference years (~20% of dates fall to VCI ≈ 0 by chance), so we label with SVI and keep VCI as an input. |
| B3 ● | Peters, A. J., Walter-Shea, E. A., Ji, L., Viña, A., Hayes, M., Svoboda, M. D. (2002). Drought monitoring with NDVI-based Standardized Vegetation Index. *Photogrammetric Engineering & Remote Sensing* 68(1), 71–75. | Introduced the **Standardized Vegetation Index (SVI)**, a z-score of NDVI by date. | Our label. We compute it per field and day of year from 2019–2023 normals, with the scored year removed. |
| B4 ✔ | Rhee, J., Im, J., Carbone, G. J. (2010). Monitoring agricultural drought for arid and humid regions using multi-sensor remote sensing data. *Remote Sensing of Environment* 114(12), 2875–2887. doi:10.1016/j.rse.2010.07.005 | **Scaled Drought Condition Index** combining MODIS LST, NDVI and TRMM rain; better than NDVI and VHI. | Shows combining weather and vegetation helps. We link them in time instead (weather alarm, then satellite confirmation) and measure the lag (~10 days). |
| B5 ✔ | Park, S., Im, J., Jang, E., Rhee, J. (2016). Drought assessment and monitoring through blending of multi-sensor indices using machine learning approaches for different climate regions. *Agricultural and Forest Meteorology* 216, 157–169. doi:10.1016/j.agrformet.2015.10.011 | RF, boosted trees and Cubist on 16 MODIS/TRMM drought factors; RF best. | Same model family (RF, boosted trees). We work at 10 m field scale with Sentinel-2 rather than MODIS. |
| B6 ● | Brown, J. F., Wardlow, B. D., Tadesse, T., Hayes, M. J., Reed, B. C. (2008). The Vegetation Drought Response Index (VegDRI): a new integrated approach for monitoring drought stress in vegetation. *GIScience & Remote Sensing* 45(1), 16–46. | Operational US index combining satellite vegetation, climate drought indices and biophysical data. | An operational precedent for weather + satellite integration. Ours is evaluated against government declarations with real data delays. |
| B7 ✔ | Bhardwaj, K., Mishra, V. (2021). Drought detection and declaration in India. *Water Security* 14, 100104. doi:10.1016/j.wasec.2021.100104 | Compared detected and declared droughts for all Indian districts (2000–2017); lower detection in **Andhra Pradesh and Karnataka**; SIF better than NDVI. | Directly relevant: our study area is the region they flag as hard. We validate against AP/Karnataka orders at mandal level. |
| B8 ● | Gorelick, N., Hancher, M., Dixon, M., Ilyushchenko, S., Thau, D., Moore, R. (2017). Google Earth Engine: planetary-scale geospatial analysis for everyone. *Remote Sensing of Environment* 202, 18–27. | The Earth Engine platform. | All satellite processing (scripts A, B, C1–C3) runs in Earth Engine. |
| B9 ● | Drusch, M., Del Bello, U., Carlier, S., et al. (2012). Sentinel-2: ESA's optical high-resolution mission for GMES operational services. *Remote Sensing of Environment* 120, 25–36. | Describes the Sentinel-2 mission (10 m, ~5-day revisit). | Our satellite data source. |
| B10 ● | Pasquarella, V. J., Brown, C. F., Czerwinski, W., Rucklidge, W. J. (2023). Comprehensive quality assessment of optical satellite imagery using weakly supervised video learning. *CVPR Workshops* (Cloud Score+). | Cloud Score+ per-pixel quality scores for Sentinel-2. | Our cloud and shadow mask (threshold 0.6), together with the SCL layer. |
| B11 ● | Achanta, R., Süsstrunk, S. (2017). Superpixels and polygons using simple non-iterative clustering. *CVPR*, 4651–4660. | The SNIC superpixel algorithm. | Splits farmland into field-sized segments (13,750 segments) because no field boundaries exist. |
| B12 ● | Zanaga, D., et al. (2021, 2022). ESA WorldCover 10 m v100 / v200. · Brown, C. F., et al. (2022). Dynamic World, near real-time global 10 m land use land cover mapping. *Scientific Data* 9, 251. | Global 10 m land-cover maps. | Used to find cropland and remove built-up, tree and water pixels. |

**Gap our satellite module addresses:**
- Most studies validate one index against another, or use district averages.
- We use automatically mapped, randomly sampled fields; normals that never see the scored year; and a live replay with realistic data delays against four seasons of government orders and UAV fields.

---

## C. Rabi module: rabi-season drought and vegetation forecasting (10 papers)

| # | Reference | What they did | How our work relates |
|---|---|---|---|
| C1 ● | Government of India (2016, updated 2020). *Manual for Drought Management.* Ministry of Agriculture & Farmers Welfare. | Official process: rainfall/SPI/dry-spell triggers plus impact indicators (remote sensing, sown area, soil moisture, hydrology); separate rabi assessment. | Defines what "declared drought" means. The orders are an **administrative comparison**, never a training target. |
| C2 ✔ | Padhee, S. K., Nikam, B. R., Aggarwal, S. P., Garg, V. (2014). Integrating effective drought index (EDI) and remote sensing derived parameters for agricultural drought assessment and prediction in Bundelkhand region of India. *ISPRS Archives* XL-8, 89–100. doi:10.5194/isprsarchives-XL-8-89-2014 | Rabi (Oct–Apr) 2000–2009; MODIS 16-day NDVI anomaly linked to soil-moisture anomaly lagged by 16 days; linear NDVI prediction. | The closest Indian precedent. We forecast NDVI 15 days ahead at 10 m with XGBoost and test it on later held-out seasons. |
| C3 ✔ | Nikam, B. R., et al. (2020). Assessment of early season agricultural drought using remote sensing. *ISPRS Archives* XLIII-B3-2020, 1691–1695. doi:10.5194/isprs-archives-XLIII-B3-2020-1691-2020 | Early-rabi drought in Marathwada caused by low October rain; conventional NDVI indices struggle early in the season. | Supports our finding that rabi needs a different method from kharif, and our separate "establishment" view. |
| C4 ✔ | Asoka, A., Mishra, V. (2015). Prediction of vegetation anomalies to improve food security and water management in India. *Geophysical Research Letters* 42(13), 5290–5298. doi:10.1002/2015GL063991 | Forecast NDVI anomalies over India 1–3 months ahead using NDVI, soil moisture and sea-surface temperature. | Same idea (forecast vegetation, not declarations) at regional scale. Ours is field-point scale, 15 days, with chronological evaluation. |
| C5 ✔ | Descals, A., Torres, K., Verger, A., Peñuelas, J. (2025). Evaluating Sentinel-2 for monitoring drought-induced crop failure in winter cereals. *Remote Sensing* 17(2), 340. doi:10.3390/rs17020340 | Drought-hit winter cereals in Catalonia showed **earlier decline** of vegetation indices; RF with field observations. | Supports "early decline" as a winter-crop drought signal. Our decline warnings measure it but cannot yet separate drought from harvest. |
| C6 ✔ | Pang, J., et al. (2017). Pattern of water use and seed yield under terminal drought in chickpea genotypes. *Frontiers in Plant Science* 8, 1375. doi:10.3389/fpls.2017.01375 | Water withheld from early podding cut chickpea growth and yield; conservative water use helped. | Explains **terminal drought** in rabi pulses: crops grow on stored soil water and fail late. This is why rabi greenness can look normal until late in the season. |
| C7 ✔ | Kladny, Milanta, Mraz, Hufkens, Stocker (2024) [add initials from the DOI page]. Enhanced prediction of vegetation responses to extreme drought using deep learning and Earth observation data. *Ecological Informatics* 80, 102474. doi:10.1016/j.ecoinf.2024.102474 | ConvLSTM forecasting of Sentinel-2 greenness under drought, conditioned on weather (EarthNet framework). | The deep-learning counterpart of our forecast. We use a simpler, explainable XGBoost with strict baselines, and found weather features added no accuracy at 15 days. |
| C8 ✔ | Requena-Mesa, C., Benson, V., Reichstein, M., Runge, J., Denzler, J. (2021). EarthNet2021: a large-scale dataset and challenge for Earth surface forecasting as a guided video prediction task. *CVPR Workshops*, 1132–1142. | Benchmark for forecasting Sentinel-2 imagery from past imagery and weather. | Frames "satellite greenness forecasting" as a task. Our rabi module is a small, field-level instance with eligibility rules for cloud gaps. |
| C9 ✔ | Murthy, C. S., Sesha Sai, M. V. R., Bhanuja Kumari, V., Prakash, V. S., Roy, P. S. (2008). Study of crop condition and assessment of agricultural drought in rabi season using IRS-AWiFS images. (AGRIS record; journal to confirm) | Mapped rabi 2005-06 drought in Bagalkot (Karnataka) with AWiFS; rabi crops on residual soil moisture are as drought-prone as kharif. | Neighbouring-region rabi evidence. **Confirm the journal name before citing.** |
| C10 ● | Lobell, D. B., Thau, D., Seifert, C., Engle, E., Little, B. (2015). A scalable satellite-based crop yield mapper. *Remote Sensing of Environment* 164, 324–333. | Field-scale crop yield from satellite and weather using crop-model simulations. | Future work: link our greenness forecasts to measured yield, giving independent agricultural-drought validation. |

**Gap our rabi module addresses:**
- Indian rabi studies mostly assess drought after the fact with coarse MODIS indices.
- We forecast 10 m crop greenness 15 days ahead on random cropland points, with rules registered before training, explicit missing-data handling, and chronological testing against three baselines.

---

## D. Suggested literature-review structure (about 4–5 pages)
1. Drought concepts and indices (A1–A4, B1–B3).
2. Machine learning for drought forecasting (A6–A9, A11–A12).
3. Satellite monitoring of agricultural drought (B4–B7, C2, C4).
4. Drought in India and the declaration process (A10, B7, C1, C3, C9).
5. Rabi and terminal drought; vegetation forecasting (C5–C8).
6. Research gaps, which lead into our objectives (use the three "Gap" paragraphs above).
7. *(UAV-module papers to be added later.)*

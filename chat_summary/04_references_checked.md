# References: what was checked and fixed

Every paper on the literature slides was checked against its published abstract and against Crossref, the official DOI registry, in this chat.

## Wrong references found and fixed (slides + report)
| Was | Problem | Now |
|---|---|---|
| Shahi et al. 2023, "Classification of water stress in cotton using CNNs from UAV images", *Remote Sens.* 15(1) 35 | Does not exist; that DOI belongs to a different paper | **Niu, Landivar, Duffield (2024)**, "Classification of cotton water stress using convolutional neural networks and UAV-based RGB imagery", *Advances in Modern Agriculture* 5(1), doi 10.54517/ama.v5i1.2457 |
| Dang, Chen, Lu, Li 2023, "Deep learning-based weed detection using UAV images: a comparative study", *Drones* 7(1) 2 | Title is real, but the authors and issue were wrong | **Shahi, Dahal, Sitaula, Neupane, Guo (2023)**, *Drones* 7(10) 624, doi 10.3390/drones7100624 |

## Old papers replaced by recent (2020+) related papers
| Module | Old | New (verified) | Link to our work |
|---|---|---|---|
| Weather | Belayneh 2014 | **Tamrakar, Das, Sharma (2024)**, *Discover Geoscience* 2, doi 10.1007/s44288-024-00089-z | RF/SVM SPEI forecast in India; they compare only 2 models, we compare with persistence, climatology and physics |
| Satellite | Park 2016 | **Prodhan et al. (2021)**, *Remote Sensing* 13(9) 1715, doi 10.3390/rs13091715 | DL + RF + GBM for agricultural drought in South Asia; regional scale with a soil-index target, ours is 10 m crop condition |
| Rabi | Padhee 2014 | **Danodia, Kushwaha, Patel (2021)**, *Environ. Dev. Sustain.* 23(10) 15432–15449, doi 10.1007/s10668-021-01305-3 | Combined index for rabi pulses that matches yield; we use one soil indicator with AUC 0.80 by 15 Dec |
| Rabi | Pang 2017 | **Kulkarni et al. (2020)**, *Remote Sensing* 12(13) 2091, doi 10.3390/rs12132091 | Marathwada index checked on kharif and rabi yields; soil moisture weighs most (35%), which supports our soil approach |

## Facts corrected in existing rows
- **Park 2016** (before it was replaced): it was validated with crop yield as well as SPI, not "index vs index" only.
- **Padhee 2014** (before it was replaced): the abstract does not say "rabi".
- **Descals 2025:** "harvest not separated" is not supported. Now: "One season; needs field survey" (Spain, winter cereals, 2023).
- **Pang 2017** (before it was replaced): it is a glasshouse study.
- **Marusov 2024:**
  - Before: "compared mostly with ML", which was not verified.
  - Now: "Transformer / ConvLSTM forecast PDSI up to 12 months; deep models only".
- **Dikshit 2021:** LSTM forecasts monthly SPEI-1/3 in New South Wales and beats RF and ANN (CRU data 1901–2018).
- **Sharma 2025:** Sentinel-2 + ML (XGBoost best) in Jodhpur, Amravati and Thanjavur (rabi season). Labels come from government declarations and news, with one label per region and season.
- **Nikam 2020:** October drought in Aurangabad (Marathwada) from SPI + a MODIS index (SASI); crop stress appeared even without a rain deficit.

## Old papers kept on purpose (we really use their method or index)
| Paper | Why kept |
|---|---|
| Vicente-Serrano 2010 (SPEI) | SPEI is the weather module's label |
| Kogan 1995 (VCI) | VCI is computed as an extra input |
| Peters 2002 (SVI) | SVI is the satellite label |
| Achanta 2017 (SNIC) | Field segments |
| Berman 2018 (Lovász), Lin 2017 (focal) | UAV training losses |
| Breiman 2001 (RF), Chen 2016 (XGBoost), Allen 1998 (FAO-56 ET0), Drusch 2012 (Sentinel-2), Gorelick 2017 (Earth Engine) | Tools and models we use |

## Final literature rows on the slides
- **Weather:** SPEI 2010 · Dikshit 2021 · Marusov 2024 · Tamrakar 2024
- **Satellite:** Kogan 1995 · Peters 2002 · Prodhan 2021 · Sharma 2025
- **Rabi:** Danodia 2021 · Nikam 2020 · Descals 2025 · Kulkarni 2020
- **UAV:** Xie 2021 (SegFormer) · Niu 2024 · Shahi 2023 · Berman 2018 / Lin 2017

## Source links used for checking
- Sharma 2025: https://mdpi.com/2072-4292/17/18/3159
- Nikam 2020: https://isprs-archives.copernicus.org/articles/XLIII-B3-2020/1691/2020/
- Descals 2025: https://www.mdpi.com/2072-4292/17/2/340
- Prodhan 2021: https://mdpi.com/2072-4292/13/9/1715
- Kulkarni 2020: https://doi.org/10.3390/rs12132091
- Tamrakar 2024: https://link.springer.com/article/10.1007/s44288-024-00089-z
- Shahi 2023: https://mdpi.com/2504-446X/7/10/624
- Marusov 2024 (arXiv): https://arxiv.org/abs/2309.06212
- Earth Engine S2_SR_HARMONIZED (starts 28 Mar 2017; 2017–18 coverage not global): https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED

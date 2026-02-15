# Arctic Tundra Dynamics: One-Key LST and Vegetation Index Extraction (GEE)

## 1. Scientific Abstract

This module provides a “one-key” Google Earth Engine (GEE) workflow to extract daily land surface temperature (LST) and vegetation index (VI) time series for Arctic tundra regions, with a particular focus on the **2020 Siberia heatwave and drought**. The scientific objective is to quantify **structural overshoot** and ecosystem resilience in tundra vegetation following extreme climate anomalies. By combining MODIS LST, optical vegetation indices, and climate reanalysis data, the tool produces analysis-ready CSV time series for subsequent statistical and modelling work.

*Reference*: *Figure_Structural_Overshoot_Siberia2020_Yongyuan.pdf*.

---

## 2. Technical Implementation

- **Platform and datasets**
  - Implemented in the GEE JavaScript API.
  - Uses MODIS LST products for surface temperature.
  - Uses MODIS or other optical sensors to derive vegetation indices (e.g. NDVI, EVI).
  - Optionally integrates ERA5 or similar reanalysis data for additional climate variables.

- **ROI and mask handling**
  - Accepts user-defined regions of interest (ROI) representing Arctic tundra ecosystems.
  - Applies cloud and quality masks to ensure robust LST and VI estimates.

- **Time-series construction**
  - Aggregates multi-sensor observations into a consistent daily time axis.
  - Applies compositing or temporal interpolation to mitigate data gaps.
  - Produces harmonised LST and VI time series suitable for structural overshoot analysis.

- **Export for offline analysis**
  - Exports per-ROI or per-pixel time series as CSV files to Google Drive or Cloud Storage.
  - Designed for downstream use in Python/R for statistical modelling and figure generation.

---

## 3. Key Files Description

- `arctic_lst_vis_onekey.js`  
  One-key GEE script that:
  - Defines ROIs for Arctic tundra study areas.
  - Loads and filters MODIS LST and VI-related products.
  - Constructs daily time series of LST and vegetation indices.
  - Exports CSV files capturing the joint climate–vegetation dynamics for each ROI.

---

## 4. Usage / Example

1. Open the script in the GEE Code Editor:
   - Navigate to: https://code.earthengine.google.com/
   - Import `arctic_lst_vis_onekey.js` into your personal scripts.

2. Configure in-script parameters:
   - ROIs representing tundra regions of interest.
   - Temporal window covering the pre-event, event, and recovery phases (e.g. 2018–2021).
   - Output folder or asset destination for exported CSV files.

3. Run the script to:
   - Visualise LST and VI composites for specific dates.
   - Trigger CSV exports for subsequent analysis of structural overshoot and resilience trajectories.

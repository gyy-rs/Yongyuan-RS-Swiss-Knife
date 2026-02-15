# Large-Scale Crop Type Mapping on Google Earth Engine

## 1. Scientific Abstract

This module implements a cloud-native workflow for large-scale crop type mapping using the **Google Earth Engine (GEE)** platform. The scientific objective is to generate spatially explicit crop maps over large agricultural regions by exploiting the temporal richness of Sentinel-2 and Landsat surface reflectance archives. By combining time-series vegetation indices with a Random Forest classifier, the workflow provides robust crop-type predictions that can support yield estimation, policy analysis, and food-security assessments.

---

## 2. Technical Implementation

- **Platform and data sources**
  - Implemented in the GEE JavaScript API.
  - Uses multi-temporal Sentinel-2 and/or Landsat imagery (surface reflectance collections).
  - Masks clouds and shadows using built-in quality flags and custom logic where needed.

- **Feature engineering**
  - Derives vegetation indices such as NDVI and EVI over the growing season.
  - Aggregates spectral and index time series into phenological descriptors (e.g. seasonal maxima, integrated greenness).
  - Optionally includes ancillary features such as observation counts, spectral bands, and texture measures.

- **Supervised classification**
  - Uses a Random Forest classifier trained on user-supplied training samples (field survey points or polygons).
  - Supports probability-mode outputs for uncertainty-aware analyses.
  - Performs cross-validation and accuracy assessment within GEE (confusion matrix, overall accuracy).

- **Scalable deployment**
  - Designed to run at regional to national scales by leveraging GEE’s distributed computation.
  - Exports classification results and accuracy metrics to Google Drive, Cloud Storage, or Earth Engine Assets.

---

## 3. Key Files Description

- `rf_crop_mapping.js`  
  Standalone GEE script that:
  - Loads and pre-processes Sentinel-2/Landsat image collections.
  - Constructs a stack of spectral and temporal features for the target year and region.
  - Ingests training samples, trains a Random Forest classifier, and applies it to the feature stack.
  - Computes confusion matrices and accuracy statistics.
  - Exports final crop-type maps and (optionally) per-class probability layers.

---

## 4. Usage / Example

1. Open the script in the GEE Code Editor:
   - Navigate to: https://code.earthengine.google.com/
   - Import `rf_crop_mapping.js` into your personal scripts.

2. Configure the following within the script:
   - Region of interest (ROI) geometry.
   - Time window corresponding to the growing season.
   - Image collections (Sentinel-2 / Landsat) and preprocessing options.
   - Training sample asset (points or polygons with crop type labels).

3. Run the script to:
   - Visualise intermediate vegetation index composites.
   - Train the Random Forest and inspect accuracy metrics.
   - Export classification results as GeoTIFF or Earth Engine Assets for downstream analysis.

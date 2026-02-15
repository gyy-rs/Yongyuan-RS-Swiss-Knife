# Yongyuan-RS-Swiss-Knife

*A Multi-Functional Research Toolkit for Remote Sensing: From Physical Mechanisms to Big Data & AI.*

[![Python](https://img.shields.io/badge/Python-Scientific-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Spatial%20Data-336791?logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![GEE](https://img.shields.io/badge/Google%20Earth%20Engine-Cloud%20Remote%20Sensing-4285F4?logo=googlecloud&logoColor=white)](https://earthengine.google.com/)
[![CUDA](https://img.shields.io/badge/CUDA-GPU%20Acceleration-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-zone)

---

## 1. Introduction

This repository represents the **research "Swiss Army Knife"** for Remote Sensing and Earth System Science, developed by **Yongyuan Gao**.

Just as a Swiss knife combines specialized tools (a saw for heavy cutting, tweezers for precision work), this codebase assembles **specialized computational tools** for every major challenge in modern remote sensing:

- **Heavy-duty Data Handling**: Managing and processing ~32 TB of satellite data in PostgreSQL.
- **Precision Physical Modeling**: GPU-accelerated BRDF kernels for radiative transfer and SIF normalization.
- **Cloud-scale Analytics**: Google Earth Engine scripts for Arctic tundra and large-scale crop monitoring.
- **Advanced AI & KGML**: Physics-guided deep learning (M-Net) for crop yield prediction.

Together, these modules demonstrate a **full-stack capability** for Earth observation science: integrating **physical principles**, **big-data engineering**, and **machine learning** within one coherent toolkit.

*Note: This repository serves as a curated code portfolio for my Postdoctoral application at the University of Zurich (UZH).*

---

## 2. The Toolkit (Module Overview)

Each numbered folder corresponds to a specialized "blade" of the Swiss knife.

| Module | Key Technology | Scientific Application |
|--------|----------------|------------------------|
| [01_brdf_physical_modeling](./01_brdf_physical_modeling) | GPU-accelerated Ross–Li BRDF kernels (CuPy/CUDA) | Normalizing multi-angle SIF to detect early drought signals (North China Plain). |
| [02_big_data_engineering_postgresql](./02_big_data_engineering_postgresql) | PostgreSQL/PostGIS, Partitioning, ETL | Managing ~32 TB of TROPOMI SIF and MODIS data; generating directional SIF products. |
| [03_tropomi_sif_downscaling](./03_tropomi_sif_downscaling) | Random Forest Regression | Prototyping spatial downscaling of TROPOMI SIF (ESA–MOST Dragon project). |
| [04_retrieve_sif_from_hsi_uav_data](./04_retrieve_sif_from_hsi_uav_data) | UAV Hyperspectral Processing (FLD/3FLD) | Retrieving canopy SIF for SIF–GPP decoupling analysis in cotton experiments. |
| [05_gee_remote_sensing_crop_classification_mapping](./05_gee_remote_sensing_crop_classification_mapping) | GEE (JS API), Random Forest | Large-scale crop type mapping using multi-temporal Sentinel-2 / Landsat. |
| [06_gee_arctic_tundra_vegetation_indices_onekey_extraction](./06_gee_arctic_tundra_vegetation_indices_onekey_extraction) | GEE Time-series Extraction | Analyzing Arctic tundra LST/VI dynamics and "structural overshoot" (2020 Siberia heatwave). |
| [07_mnet_crop_yield_prediction](./07_mnet_crop_yield_prediction) | Physics-guided Deep Learning (PyTorch, KGML) | **M-Net**: A BRDF-aware deep neural network for robust crop yield prediction. |

---

## 3. Technical Stack Matrix

### 3.1 Physical Modeling
- Ross–Li BRDF kernels for multi-angle reflectance modeling and SIF normalization.
- GPU-accelerated radiative transfer computations using CuPy/CUDA.
- Physically informed feature engineering for downstream machine learning.

### 3.2 Data Engineering
- Design and operation of a **PostgreSQL/PostGIS** data warehouse for ~32 TB of satellite data.
- Declarative table partitioning by time and spatial tile for efficient time-series queries.
- Robust ETL pipelines: automated harvesting, checksum verification, and bulk loading.

### 3.3 Cloud Computing
- **Google Earth Engine (GEE)** for scalable remote sensing analytics.
- Cloud-based extraction of multi-year LST and vegetation index time series for Arctic tundra.
- Regional-to-continental crop classification workflows.

### 3.4 Deep Learning and KGML
- PyTorch-based sequence models (LSTM, GRU, Attention) for Earth observation.
- **M-Net**: Physics-guided deep learning architecture incorporating BRDF mechanisms.
- Comparative benchmarking between data-driven baselines and physics-guided networks.

---

## 4. Author & Contact

- **Author**: Yongyuan Gao
- **Role**: Ph.D. Candidate & Remote Sensing Researcher
- **Institution**: China Agricultural University (CAU)
- **Email**: gaoyy@cau.edu.cn

*This portfolio illustrates my end-to-end capability from physical modeling and data engineering to cloud computing and deep learning.*
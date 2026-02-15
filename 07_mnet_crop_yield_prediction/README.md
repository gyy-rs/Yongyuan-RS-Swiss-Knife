# M-Net: Physics-Guided Deep Learning for Crop Yield Prediction

## 1. Scientific Abstract

This module develops and evaluates **M-Net**, a physics-guided deep learning architecture for county-level crop yield prediction that explicitly incorporates bidirectional reflectance distribution function (BRDF) mechanisms. The scientific objective is to improve **generalisation across satellite view geometries and agro-climatic conditions** by embedding physical knowledge about angular effects into the learning system. M-Net is contrasted against a purely data-driven baseline model (Attention-GRU) to demonstrate the added value of **Knowledge-Guided Machine Learning (KGML)** for large-scale yield forecasting.

*Associated manuscript*: *JRS_Yield_R1_20260112_Clean.pdf*.

---

## 2. Technical Implementation

- **BRDF-aware feature engineering**
  - `brdf_preprocessing_cache_directional_features.py` pre-computes directional vegetation indices and SIF-derived metrics using BRDF-informed geometries (e.g. nadir, hotspot, fixed SZA).
  - Caches multi-angle features into disk-based or in-memory tables to avoid repeated heavy computations.
  - Supports parallel processing over large spatial–temporal domains.

- **M-Net architecture (physics-guided)**
  - Implemented in `tmodel.py` using PyTorch.
  - Ingests multi-temporal, multi-angle feature tensors.
  - Combines a data-driven branch (e.g. temporal encoders) with a branch informed by BRDF-derived directional features.
  - Uses appropriate regularisation, learning-rate scheduling, and GPU/CPU parallelism to stabilise training.

- **Baseline model: Attention-GRU**
  - Implemented in `train_model_comparison_baseline_multiview_atten_gru.py` as an attention-based GRU architecture.
  - Consumes similar multi-temporal inputs but **does not** include explicit BRDF physics.
  - Serves as a strong data-driven reference to quantify the benefits of physics guidance.

- **Experiment management and multiprocessing**
  - `tmodel.py` organises experiments across model families (LSTM/GRU/CNN/RF) and multi-angle configurations.
  - Uses multiprocessing with shared memory (Linux fork semantics) to efficiently sweep over categories and angle combinations.
  - Writes trial-level performance metrics (R²) and errors directly to CSV for post-hoc analysis.

---

## 3. Model Comparison: M-Net vs. Attention-GRU Baseline

- **Attention-GRU (Baseline)**
  - Strengths:
    - Flexible data-driven temporal modelling.
    - Can capture complex, non-linear relationships in the input features.
  - Limitations:
    - Treats angular variability implicitly, which can lead to overfitting to specific sensor geometries.
    - Generalisation across years, regions, or alternative view-angle configurations is limited.

- **M-Net (Physics-Guided)**
  - Strengths:
    - Explicitly integrates BRDF-based directional features, allowing the network to “know” how reflectance and SIF respond to changing geometry.
    - Reduces spurious correlations tied to particular sun–sensor angles, improving robustness across observational conditions.
    - Better suited for extrapolating to unseen combinations of year, sensor configuration, or management regime.
  - Use-cases:
    - Cross-year generalisation in county-level yield prediction.
    - Scenario experiments where sensor viewing geometry differs from the training configuration.

The experimental scripts in this directory are organised to allow side-by-side comparison of these two paradigms, with shared data preprocessing and evaluation protocols.

---

## 4. Key Files Description

- `brdf_preprocessing_cache_directional_features.py`  
  BRDF-informed preprocessing pipeline:
  - Loads BRDF and SIF-related inputs.
  - Computes directional vegetation indices (e.g. NIRv, NDVI, EVI2) at multiple viewing geometries.
  - Caches the resulting features into compressed files for efficient reuse during model training.

- `tmodel.py`  
  Core experimentation and training driver:
  - Defines model lists (LSTM, GRU, CNN, RF) and global hyperparameters.
  - Contains training routines for deep learning and Random Forest baselines.
  - Manages multiprocessing, GPU/CPU worker allocation, and CSV logging of results.

- `train_baseline_multiview_comparison_mnet.py`  
  Training script for the M-Net physics-guided model:
  - Constructs dual-stream input tensors for angular and scalar features.
  - Trains M-Net, tracks validation performance, and stores predictions at the best epoch.
  - Saves outputs as `.pkl` files for subsequent visualisation and analysis.

- `train_model_comparison_baseline_multiview_atten_gru.py`  
  Training script for the Attention-GRU baseline:
  - Implements a GRU-based sequence model with temporal attention.
  - Uses the same experimental setup (target categories, dates, repeats) to ensure fair comparison with M-Net.
  - Produces benchmark results used to quantify the benefit of physics-guided design.

- `Paper - Manuscript - JRS_Yield_R1_20260112_Clean.pdf`  
  Manuscript describing the scientific context, experimental design, and key results of the M-Net study.

---

## 5. Usage / Example

### 5.1. Precompute directional features

```bash
cd 07_mnet_crop_yield_prediction
python brdf_preprocessing_cache_directional_features.py
```

This step generates cached multi-angle features required by both M-Net and the baseline models.

### 5.2. Train and evaluate M-Net (physics-guided)

```bash
python train_baseline_multiview_comparison_mnet.py
```

The script:
- Loads preprocessed directional features.
- Trains the M-Net architecture.
- Saves performance metrics and prediction `.pkl` files for visualisation.

### 5.3. Train and evaluate the Attention-GRU baseline

```bash
python train_model_comparison_baseline_multiview_atten_gru.py
```

The script:
- Uses the same target categories and time windows as M-Net.
- Trains the Atten-GRU model and logs benchmark performance.

The two training scripts together reproduce the **M-Net vs. Baseline** comparison highlighted in the associated JRS manuscript.

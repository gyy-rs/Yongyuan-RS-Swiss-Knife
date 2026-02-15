# GPU-Accelerated BRDF Physical Modeling for Solar-Induced Fluorescence Normalization

## 1. Scientific Abstract

This module implements physically based Bidirectional Reflectance Distribution Function (BRDF) kernels on GPUs to normalize multi-angle satellite observations of solar-induced chlorophyll fluorescence (SIF). The primary scientific objective is to reduce view- and illumination-geometry artefacts in coarse-resolution TROPOMI SIF products, thereby enabling earlier and more robust detection of ecosystem stress signals such as the 2019 North China Plain drought event. By combining the Ross–Thick volumetric kernel and the Li–Sparse geometric kernel with high-throughput GPU computation, this module provides a scalable radiative transfer correction framework suitable for continental- to global-scale drought monitoring studies.

*Associated study*: *Normalized solar-induced fluorescence responds earlier than vegetation indices to the 2019 North China Plain drought*.

---

## 2. Technical Implementation

- **Physics-based BRDF kernels**
  - Implements the Ross–Thick (volumetric scattering) kernel.
  - Implements the Li–Transit / Li–Sparse (geometric–optic) kernel.
  - Uses sensor-sun geometry (solar zenith, viewing zenith, relative azimuth) expressed in radians.

- **GPU acceleration with CuPy**
  - Uses CuPy as a drop-in replacement for NumPy to execute all kernel computations on NVIDIA GPUs.
  - Vectorizes operations over full satellite swaths, avoiding Python loops.
  - Carefully clips trigonometric arguments to valid domains (e.g. for arccos) to ensure numerical stability.

- **Vectorized BRDF evaluation**
  - Accepts angle and kernel coefficient inputs as pandas Series or NumPy arrays.
  - Transfers input arrays from CPU to GPU, performs all BRDF calculations in GPU memory, then returns the result as a NumPy array.
  - Designed to be called from higher-level data-processing pipelines (e.g. within large-scale SIF normalization workflows).

- **Robustness and diagnostics**
  - Prints Python, CuPy, and environment information at import time for reproducibility.
  - Wraps the GPU computation in a try–except block and reports any GPU-related errors.

---

## 3. Key Files Description

- `brdf_gpu_acceleration.py`  
  Core implementation of GPU-accelerated BRDF kernels and their vectorized application to satellite data.
  - `Ross_thick(sZenith, vZenith, rAzimuth)`  
    Computes the Ross–Thick volumetric kernel using CuPy, given solar zenith, view zenith, and relative azimuth angles (in radians).
  - `Li_Transit(sZenith, vZenith, rAzimuth)`  
    Computes the Li–Transit geometric kernel using a canopy-structure parameterization and azimuth folding.
  - `BRDF_degree_vectorized(i, v, r, iso, vol, geo)`  
    High-level driver that converts angles in degrees to radians, evaluates kernels on the GPU, combines isotropic/volumetric/geometric coefficients, and returns BRDF-corrected reflectance factors.

---

## 4. Usage / Example

Below is a minimal example showing how to call the GPU-accelerated BRDF functions on a batch of observations. In practice, these vectors would typically come from a pre-processed satellite swath (e.g. TROPOMI or MODIS) stored in a pandas DataFrame.

```python
import pandas as pd
from brdf_gpu_acceleration import BRDF_degree_vectorized

df = pd.DataFrame({
    "sza_deg": [30.0, 35.0, 40.0],
    "vza_deg": [10.0, 15.0, 20.0],
    "raa_deg": [150.0, 160.0, 170.0],
    "iso":     [0.05, 0.05, 0.05],
    "vol":     [0.10, 0.10, 0.10],
    "geo":     [0.02, 0.02, 0.02],
})

R_brdf = BRDF_degree_vectorized(
    i=df["sza_deg"],
    v=df["vza_deg"],
    r=df["raa_deg"],
    iso=df["iso"],
    vol=df["vol"],
    geo=df["geo"],
)

df["R_brdf"] = R_brdf
print(df)
```

*Note*: A working NVIDIA GPU with a compatible CUDA toolkit and CuPy installed is required.

# UAV Hyperspectral SIF Retrieval with Resonon Pika-L

## 1. Scientific Abstract

This module processes hyperspectral imagery acquired by a **Resonon Pika-L** line-scanning camera mounted on an unmanned aerial vehicle (UAV) to retrieve canopy solar-induced chlorophyll fluorescence (SIF). The scientific objective is to characterise the **decoupling between SIF and gross primary production (GPP)** in irrigated and stressed cotton fields, under varying water and nitrogen treatments. By applying physically motivated Fraunhofer Line Depth (FLD/3FLD) retrievals to radiometrically calibrated hyperspectral cubes, the workflow provides plot-scale SIF maps that bridge the observational gap between leaf/chamber measurements and satellite-based SIF products.

*Associated manuscript*: *Cotton SIF Decoupling Manuscript*.

---

## 2. Technical Implementation

- **Sensor configuration and data ingestion**
  - Assumes input data from a Resonon Pika-L camera covering approximately 400–1,000 nm.
  - Ingests radiance datacubes together with wavelength, dark current, and white reference information.
  - Handles flight-line mosaics or strip-wise acquisitions depending on the experimental setup.

- **Radiometric and geometric preprocessing**
  - Applies dark-current correction and normalisation with respect to calibrated reference panels.
  - Converts raw digital numbers to at-sensor radiance or reflectance units.
  - Optionally co-registers hyperspectral data with ground control points or external orthomosaics.

- **SIF retrieval (FLD / 3FLD family)**
  - Identifies Fraunhofer or atmospheric absorption lines in the spectral domain (e.g. O₂-A band).
  - Implements FLD / 3FLD-style algorithms to separate emitted fluorescence from reflected radiance.
  - Produces SIF estimates at the canopy level for each pixel or aggregated plot.

- **Plot-level aggregation and analysis**
  - Aggregates pixel-level SIF within predefined plot polygons or experimental blocks.
  - Exports SIF statistics suitable for linking with flux-tower GPP, gas-exchange measurements, or crop growth models.

---

## 3. Key Files Description

- `pika_l_sif.py`  
  Main processing script for UAV hyperspectral SIF retrieval:
  - Reads raw or pre-processed Pika-L hyperspectral datacubes and associated metadata.
  - Performs radiometric calibration using dark frames and reference panels.
  - Applies FLD/3FLD SIF retrieval at selected spectral lines.
  - Outputs SIF maps and optional plot-level summary tables for further analysis.

---

## 4. Usage / Example

Below is an indicative usage pattern; exact arguments may differ depending on how the script is parameterised internally.

```bash
cd 04_retrieve_sif_from_hsi_uav_data
python pika_l_sif.py \
  --input_cube   /path/to/pikaL_cube.bil \
  --wavelengths  /path/to/wavelengths.txt \
  --panel_region /path/to/panel_shapefile.shp \
  --plots        /path/to/cotton_plots.shp \
  --output_dir   /path/to/output_sif_maps
```

Typical outputs include:
- Georeferenced SIF maps (e.g. GeoTIFF format).
- Plot-level CSV files summarising SIF statistics by treatment or management regime.

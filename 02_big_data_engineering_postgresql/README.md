# Large-Scale TROPOMI SIF and MODIS Data Engineering with PostgreSQL

## 1. Scientific Abstract

This module provides the data-engineering backbone for large-scale analysis of solar-induced chlorophyll fluorescence (SIF) and optical remote sensing products. It was designed to manage and preprocess on the order of tens of terabytes (~32 TB) of TROPOMI SIF and MODIS surface reflectance data for downstream scientific applications, including BRDF normalization, directional SIF derivation, and crop yield or drought monitoring studies. The pipeline automates the acquisition, integrity checking, and database-ready structuring of satellite products, and integrates them into a PostgreSQL/PostGIS environment with spatial–temporal organization suitable for efficient querying and aggregation.

---

## 2. Technical Implementation

- **Automated TROPOMI SIF metadata harvesting**
  - Uses the official TROPOMI SIF API to incrementally crawl product metadata.
  - Stores per-product JSON descriptors locally for reproducible and offline processing.
  - Handles network errors and API instability with robust retry logic and offset management.

- **Batch downloading and integrity verification**
  - Downloads full-resolution TROPOMI L2B SIF products based on filtered JSON metadata.
  - Uses requests and ThreadPoolExecutor for multithreaded downloading.
  - Implements resume capability via HTTP Range requests for partially downloaded files.
  - Performs multi-algorithm checksum verification (MD5, SHA1, SHA256) and handles provider-specific checksum formats.

- **MODIS surface reflectance ingestion**
  - Downloads MODIS MCD43C1 (BRDF/albedo) products using NASA Earthdata credentials.
  - Leverages .netrc-based authentication and application-approval checks.
  - Streams downloads with per-file progress bars and an overall progress monitor.

- **Metadata extraction and PostgreSQL bulk loading**
  - Parses NetCDF metadata into a compact CSV summary (filename, table_name, file_length, time_baseline).
  - Prints PostgreSQL COPY commands for efficient bulk ingestion into a TROPOMI_SIF_META table.
  - Provides a template for trajectory tables linked to metadata.

- **On-the-fly directional SIF and vegetation index generation**
  - A dedicated SQL script (generate_directional_sif_by_postgresql.sql) computes multi-angle NIRv, NDVI, EVI2, and directional SIF yield metrics directly in PostgreSQL.
  - Uses spatial intersections between TROPOMI trajectories and ROI polygons to export CSV-level data products.

---

## 3. Key Files Description

- batch_download_esa_tropomi_sif_l2b/01_query_tropomi_sif_update  
  Incremental crawler for TROPOMI L2B SIF metadata. Queries the API, saves JSON descriptors, and manages offsets robustly.

- batch_download_esa_tropomi_sif_l2b/02_batch_download_esa_tropomi_sif_l2b.py  
  High-throughput batch downloader that:
  - Filters metadata by acquisition date.
  - Downloads products in parallel.
  - Supports resume and checksum-based validation.

- batch_download_esa_tropomi_sif_l2b/03_create_postgresql_table.py  
  Extracts key NetCDF metadata, writes TROPOMI_SIF_META.csv, and prints a COPY command to ingest it into PostgreSQL.

- batch_download_modis.py  
  MODIS MCD43C1 downloader relying on Earthdata .netrc authentication, with multi-threaded downloads and progress bars.

- generate_directional_sif_by_postgresql.sql  
  PostgreSQL script that intersects TROPOMI trajectories with ROIs and computes directional SIF, NIRv, NDVI, and EVI2 at multiple viewing geometries, exporting CSV products.

---

## 4. Usage / Example

### 4.1. Query and store TROPOMI metadata

```bash
cd 02_big_data_engineering_postgresql/batch_download_esa_tropomi_sif_l2b
python 01_query_tropomi_sif_update
```

### 4.2. Batch download TROPOMI SIF products

```bash
python 02_batch_download_esa_tropomi_sif_l2b.py
```

### 4.3. Generate metadata CSV and ingest into PostgreSQL

```bash
python 03_create_postgresql_table.py
```

Then, in psql:

```sql
COPY TROPOMI_SIF_META (filename, table_name, file_length, time_baseline)
FROM '/nas/@Data.TROPOMI_SIF/@ESA_TROPOMI_SIF_L2B/META/TROPOMI_SIF_META.csv'
DELIMITER ',' CSV HEADER;
```

### 4.4. Download MODIS MCD43C1

```bash
cd 02_big_data_engineering_postgresql
python batch_download_modis.py
```

### 4.5. Generate directional SIF products

In psql:

```sql
\i 02_big_data_engineering_postgresql/generate_directional_sif_by_postgresql.sql
```

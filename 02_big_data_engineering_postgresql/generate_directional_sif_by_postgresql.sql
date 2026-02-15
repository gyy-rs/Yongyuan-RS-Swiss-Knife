DO $$
DECLARE
    r RECORD;
    csv_file text;
BEGIN
    FOR r IN
        SELECT tablename
        FROM pg_tables
        WHERE schemaname = 'public'
          AND tablename LIKE 'sif_tropomi_%'
        ORDER BY tablename
    LOOP
        csv_file := '/pg_disk/@open_data/@Paper3.s1.data_SIF/raw_exports/' || r.tablename || '.csv';

        RAISE NOTICE 'Processing table: %, exporting to: %', r.tablename, csv_file;

        -- Create a temporary table containing rows intersecting the ROI
        EXECUTE format('
            CREATE TEMP TABLE temp_intersect AS
            SELECT 
                roi.name,
                tropomi.*
            FROM public.temp_US_yield_roi AS roi
            JOIN public.%I AS tropomi
                ON ST_Intersects(roi.geom, tropomi.geom)
        ', r.tablename);

        -- Export to CSV, including newly derived directional SIF and vegetation indices
        EXECUTE format('
            COPY (
                SELECT
                    name,
                    site_id_smvi,
                    dt,
                    lat,
                    lon,
                    sif735,
                    sif743,
                    sifc735,
                    sifc743,
                    sife735,
                    sife743,
                    mrad735,
                    mrad743,
                    raa,
                    sza,
                    vza,
                    cld_fr,
                    lcmask,
                    delta_time,
                    iso_r,
                    iso_n,
                    vol_r,
                    vol_n,
                    geo_r,
                    geo_n,
										par,
                    -- Additional derived columns
                    (mcd43c1_band_nir_nadir_sza_sif_vza_0_raa_sif - mcd43c1_band_red_nadir_sza_sif_vza_0_raa_sif) * 
                    mcd43c1_band_nir_nadir_sza_sif_vza_0_raa_sif / 
                    NULLIF((mcd43c1_band_nir_nadir_sza_sif_vza_0_raa_sif + mcd43c1_band_red_nadir_sza_sif_vza_0_raa_sif), 0) AS nirv,
                    
                    2.5 * (mcd43c1_band_nir_nadir_sza_sif_vza_0_raa_sif - mcd43c1_band_red_nadir_sza_sif_vza_0_raa_sif) / 
                    NULLIF((mcd43c1_band_nir_nadir_sza_sif_vza_0_raa_sif + 2.4 * mcd43c1_band_red_nadir_sza_sif_vza_0_raa_sif + 1), 0) AS evi2,
                    
                    (mcd43c1_band_nir_nadir_sza_sif_vza_0_raa_sif - mcd43c1_band_red_nadir_sza_sif_vza_0_raa_sif) / 
                    NULLIF((mcd43c1_band_nir_nadir_sza_sif_vza_0_raa_sif + mcd43c1_band_red_nadir_sza_sif_vza_0_raa_sif), 0) AS ndvi,
                    
                    (mcd43c1_band_nir_sifangel_sza_sif_vza_sif_raa_sif - mcd43c1_band_red_sifangel_sza_sif_vza_sif_raa_sif) * 
                    mcd43c1_band_nir_sifangel_sza_sif_vza_sif_raa_sif / 
                    NULLIF((mcd43c1_band_nir_sifangel_sza_sif_vza_sif_raa_sif + mcd43c1_band_red_sifangel_sza_sif_vza_sif_raa_sif), 0) AS nirv_sifangel,
                    
                    (mcd43c1_band_nir_nadir_sza_sif_vza_0_raa_sif - mcd43c1_band_red_nadir_sza_sif_vza_0_raa_sif) * 
                    mcd43c1_band_nir_nadir_sza_sif_vza_0_raa_sif / 
                    NULLIF((mcd43c1_band_nir_nadir_sza_sif_vza_0_raa_sif + mcd43c1_band_red_nadir_sza_sif_vza_0_raa_sif), 0) AS nirv_nadir,
                    
                    (mcd43c1_band_nir_hotspot_sza_sif_vza_0_raa_0 - mcd43c1_band_red_hotspot_sza_sif_vza_0_raa_0) * 
                    mcd43c1_band_nir_hotspot_sza_sif_vza_0_raa_0 / 
                    NULLIF((mcd43c1_band_nir_hotspot_sza_sif_vza_0_raa_0 + mcd43c1_band_red_hotspot_sza_sif_vza_0_raa_0), 0) AS nirv_hotspot,
                    
                    (mcd43c1_band_nir_fix10sza_sza_10_vza_0_raa_sif - mcd43c1_band_red_fix10sza_sza_10_vza_0_raa_sif) * 
                    mcd43c1_band_nir_fix10sza_sza_10_vza_0_raa_sif / 
                    NULLIF((mcd43c1_band_nir_fix10sza_sza_10_vza_0_raa_sif + mcd43c1_band_red_fix10sza_sza_10_vza_0_raa_sif), 0) AS nirv_fix10sza,
                    
                    (mcd43c1_band_nir_fix30sza_sza_30_vza_0_raa_sif - mcd43c1_band_red_fix30sza_sza_30_vza_0_raa_sif) * 
                    mcd43c1_band_nir_fix30sza_sza_30_vza_0_raa_sif / 
                    NULLIF((mcd43c1_band_nir_fix30sza_sza_30_vza_0_raa_sif + mcd43c1_band_red_fix30sza_sza_30_vza_0_raa_sif), 0) AS nirv_fix30sza,
                    
                    (mcd43c1_band_nir_fix45sza_sza_45_vza_0_raa_sif - mcd43c1_band_red_fix45sza_sza_45_vza_0_raa_sif) * 
                    mcd43c1_band_nir_fix45sza_sza_45_vza_0_raa_sif / 
                    NULLIF((mcd43c1_band_nir_fix45sza_sza_45_vza_0_raa_sif + mcd43c1_band_red_fix45sza_sza_45_vza_0_raa_sif), 0) AS nirv_fix45sza,
                    
                    (mcd43c1_band_nir_fix60sza_sza_60_vza_0_raa_sif - mcd43c1_band_red_fix60sza_sza_60_vza_0_raa_sif) * 
                    mcd43c1_band_nir_fix60sza_sza_60_vza_0_raa_sif / 
                    NULLIF((mcd43c1_band_nir_fix60sza_sza_60_vza_0_raa_sif + mcd43c1_band_red_fix60sza_sza_60_vza_0_raa_sif), 0) AS nirv_fix60sza,
                    
                    -- Ratio between SIF and NIRv
                    sif743 / NULLIF(((mcd43c1_band_nir_sifangel_sza_sif_vza_sif_raa_sif - mcd43c1_band_red_sifangel_sza_sif_vza_sif_raa_sif) * 
                    mcd43c1_band_nir_sifangel_sza_sif_vza_sif_raa_sif / 
                    NULLIF((mcd43c1_band_nir_sifangel_sza_sif_vza_sif_raa_sif + mcd43c1_band_red_sifangel_sza_sif_vza_sif_raa_sif), 0) * par), 0) AS phi_f_743,
                    
                    -- Compute multi-angle SIF based on nirv_x / nirv_sifangel ratios
                    sif743 * (
                        (mcd43c1_band_nir_nadir_sza_sif_vza_0_raa_sif - mcd43c1_band_red_nadir_sza_sif_vza_0_raa_sif) * 
                        mcd43c1_band_nir_nadir_sza_sif_vza_0_raa_sif / 
                        NULLIF((mcd43c1_band_nir_nadir_sza_sif_vza_0_raa_sif + mcd43c1_band_red_nadir_sza_sif_vza_0_raa_sif), 0)
                    ) / NULLIF((
                        (mcd43c1_band_nir_sifangel_sza_sif_vza_sif_raa_sif - mcd43c1_band_red_sifangel_sza_sif_vza_sif_raa_sif) * 
                        mcd43c1_band_nir_sifangel_sza_sif_vza_sif_raa_sif / 
                        NULLIF((mcd43c1_band_nir_sifangel_sza_sif_vza_sif_raa_sif + mcd43c1_band_red_sifangel_sza_sif_vza_sif_raa_sif), 0)
                    ), 0) AS sif743_nadir,
                    
                    sif743 * (
                        (mcd43c1_band_nir_hotspot_sza_sif_vza_0_raa_0 - mcd43c1_band_red_hotspot_sza_sif_vza_0_raa_0) * 
                        mcd43c1_band_nir_hotspot_sza_sif_vza_0_raa_0 / 
                        NULLIF((mcd43c1_band_nir_hotspot_sza_sif_vza_0_raa_0 + mcd43c1_band_red_hotspot_sza_sif_vza_0_raa_0), 0)
                    ) / NULLIF((
                        (mcd43c1_band_nir_sifangel_sza_sif_vza_sif_raa_sif - mcd43c1_band_red_sifangel_sza_sif_vza_sif_raa_sif) * 
                        mcd43c1_band_nir_sifangel_sza_sif_vza_sif_raa_sif / 
                        NULLIF((mcd43c1_band_nir_sifangel_sza_sif_vza_sif_raa_sif + mcd43c1_band_red_sifangel_sza_sif_vza_sif_raa_sif), 0)
                    ), 0) AS sif743_hotspot,
                    
                    sif743 * (
                        (mcd43c1_band_nir_fix10sza_sza_10_vza_0_raa_sif - mcd43c1_band_red_fix10sza_sza_10_vza_0_raa_sif) * 
                        mcd43c1_band_nir_fix10sza_sza_10_vza_0_raa_sif / 
                        NULLIF((mcd43c1_band_nir_fix10sza_sza_10_vza_0_raa_sif + mcd43c1_band_red_fix10sza_sza_10_vza_0_raa_sif), 0)
                    ) / NULLIF((
                        (mcd43c1_band_nir_sifangel_sza_sif_vza_sif_raa_sif - mcd43c1_band_red_sifangel_sza_sif_vza_sif_raa_sif) * 
                        mcd43c1_band_nir_sifangel_sza_sif_vza_sif_raa_sif / 
                        NULLIF((mcd43c1_band_nir_sifangel_sza_sif_vza_sif_raa_sif + mcd43c1_band_red_sifangel_sza_sif_vza_sif_raa_sif), 0)
                    ), 0) AS sif743_fix10sza,
                    
                    sif743 * (
                        (mcd43c1_band_nir_fix30sza_sza_30_vza_0_raa_sif - mcd43c1_band_red_fix30sza_sza_30_vza_0_raa_sif) * 
                        mcd43c1_band_nir_fix30sza_sza_30_vza_0_raa_sif / 
                        NULLIF((mcd43c1_band_nir_fix30sza_sza_30_vza_0_raa_sif + mcd43c1_band_red_fix30sza_sza_30_vza_0_raa_sif), 0)
                    ) / NULLIF((
                        (mcd43c1_band_nir_sifangel_sza_sif_vza_sif_raa_sif - mcd43c1_band_red_sifangel_sza_sif_vza_sif_raa_sif) * 
                        mcd43c1_band_nir_sifangel_sza_sif_vza_sif_raa_sif / 
                        NULLIF((mcd43c1_band_nir_sifangel_sza_sif_vza_sif_raa_sif + mcd43c1_band_red_sifangel_sza_sif_vza_sif_raa_sif), 0)
                    ), 0) AS sif743_fix30sza,
                    
                    sif743 * (
                        (mcd43c1_band_nir_fix45sza_sza_45_vza_0_raa_sif - mcd43c1_band_red_fix45sza_sza_45_vza_0_raa_sif) * 
                        mcd43c1_band_nir_fix45sza_sza_45_vza_0_raa_sif / 
                        NULLIF((mcd43c1_band_nir_fix45sza_sza_45_vza_0_raa_sif + mcd43c1_band_red_fix45sza_sza_45_vza_0_raa_sif), 0)
                    ) / NULLIF((
                        (mcd43c1_band_nir_sifangel_sza_sif_vza_sif_raa_sif - mcd43c1_band_red_sifangel_sza_sif_vza_sif_raa_sif) * 
                        mcd43c1_band_nir_sifangel_sza_sif_vza_sif_raa_sif / 
                        NULLIF((mcd43c1_band_nir_sifangel_sza_sif_vza_sif_raa_sif + mcd43c1_band_red_sifangel_sza_sif_vza_sif_raa_sif), 0)
                    ), 0) AS sif743_fix45sza,
                    
                    sif743 * (
                        (mcd43c1_band_nir_fix60sza_sza_60_vza_0_raa_sif - mcd43c1_band_red_fix60sza_sza_60_vza_0_raa_sif) * 
                        mcd43c1_band_nir_fix60sza_sza_60_vza_0_raa_sif / 
                        NULLIF((mcd43c1_band_nir_fix60sza_sza_60_vza_0_raa_sif + mcd43c1_band_red_fix60sza_sza_60_vza_0_raa_sif), 0)
                    ) / NULLIF((
                        (mcd43c1_band_nir_sifangel_sza_sif_vza_sif_raa_sif - mcd43c1_band_red_sifangel_sza_sif_vza_sif_raa_sif) * 
                        mcd43c1_band_nir_sifangel_sza_sif_vza_sif_raa_sif / 
                        NULLIF((mcd43c1_band_nir_sifangel_sza_sif_vza_sif_raa_sif + mcd43c1_band_red_sifangel_sza_sif_vza_sif_raa_sif), 0)
                    ), 0) AS sif743_fix60sza,
                    
                    -- Compute multi-angle observations of EVI2
                    2.5 * (mcd43c1_band_nir_sifangel_sza_sif_vza_sif_raa_sif - mcd43c1_band_red_sifangel_sza_sif_vza_sif_raa_sif) / 
                    NULLIF((mcd43c1_band_nir_sifangel_sza_sif_vza_sif_raa_sif + 2.4 * mcd43c1_band_red_sifangel_sza_sif_vza_sif_raa_sif + 1), 0) AS evi2_sifangel,
                    
                    2.5 * (mcd43c1_band_nir_nadir_sza_sif_vza_0_raa_sif - mcd43c1_band_red_nadir_sza_sif_vza_0_raa_sif) / 
                    NULLIF((mcd43c1_band_nir_nadir_sza_sif_vza_0_raa_sif + 2.4 * mcd43c1_band_red_nadir_sza_sif_vza_0_raa_sif + 1), 0) AS evi2_nadir,
                    
                    2.5 * (mcd43c1_band_nir_hotspot_sza_sif_vza_0_raa_0 - mcd43c1_band_red_hotspot_sza_sif_vza_0_raa_0) / 
                    NULLIF((mcd43c1_band_nir_hotspot_sza_sif_vza_0_raa_0 + 2.4 * mcd43c1_band_red_hotspot_sza_sif_vza_0_raa_0 + 1), 0) AS evi2_hotspot,
                    
                    2.5 * (mcd43c1_band_nir_fix10sza_sza_10_vza_0_raa_sif - mcd43c1_band_red_fix10sza_sza_10_vza_0_raa_sif) / 
                    NULLIF((mcd43c1_band_nir_fix10sza_sza_10_vza_0_raa_sif + 2.4 * mcd43c1_band_red_fix10sza_sza_10_vza_0_raa_sif + 1), 0) AS evi2_fix10sza,
                    
                    2.5 * (mcd43c1_band_nir_fix30sza_sza_30_vza_0_raa_sif - mcd43c1_band_red_fix30sza_sza_30_vza_0_raa_sif) / 
                    NULLIF((mcd43c1_band_nir_fix30sza_sza_30_vza_0_raa_sif + 2.4 * mcd43c1_band_red_fix30sza_sza_30_vza_0_raa_sif + 1), 0) AS evi2_fix30sza,
                    
                    2.5 * (mcd43c1_band_nir_fix45sza_sza_45_vza_0_raa_sif - mcd43c1_band_red_fix45sza_sza_45_vza_0_raa_sif) / 
                    NULLIF((mcd43c1_band_nir_fix45sza_sza_45_vza_0_raa_sif + 2.4 * mcd43c1_band_red_fix45sza_sza_45_vza_0_raa_sif + 1), 0) AS evi2_fix45sza,
                    
                    2.5 * (mcd43c1_band_nir_fix60sza_sza_60_vza_0_raa_sif - mcd43c1_band_red_fix60sza_sza_60_vza_0_raa_sif) / 
                    NULLIF((mcd43c1_band_nir_fix60sza_sza_60_vza_0_raa_sif + 2.4 * mcd43c1_band_red_fix60sza_sza_60_vza_0_raa_sif + 1), 0) AS evi2_fix60sza,
                    
                    -- Compute multi-angle observations of NDVI
                    (mcd43c1_band_nir_sifangel_sza_sif_vza_sif_raa_sif - mcd43c1_band_red_sifangel_sza_sif_vza_sif_raa_sif) / 
                    NULLIF((mcd43c1_band_nir_sifangel_sza_sif_vza_sif_raa_sif + mcd43c1_band_red_sifangel_sza_sif_vza_sif_raa_sif), 0) AS ndvi_sifangel,
                    
                    (mcd43c1_band_nir_nadir_sza_sif_vza_0_raa_sif - mcd43c1_band_red_nadir_sza_sif_vza_0_raa_sif) / 
                    NULLIF((mcd43c1_band_nir_nadir_sza_sif_vza_0_raa_sif + mcd43c1_band_red_nadir_sza_sif_vza_0_raa_sif), 0) AS ndvi_nadir,
                    
                    (mcd43c1_band_nir_hotspot_sza_sif_vza_0_raa_0 - mcd43c1_band_red_hotspot_sza_sif_vza_0_raa_0) / 
                    NULLIF((mcd43c1_band_nir_hotspot_sza_sif_vza_0_raa_0 + mcd43c1_band_red_hotspot_sza_sif_vza_0_raa_0), 0) AS ndvi_hotspot,
                    
                    (mcd43c1_band_nir_fix10sza_sza_10_vza_0_raa_sif - mcd43c1_band_red_fix10sza_sza_10_vza_0_raa_sif) / 
                    NULLIF((mcd43c1_band_nir_fix10sza_sza_10_vza_0_raa_sif + mcd43c1_band_red_fix10sza_sza_10_vza_0_raa_sif), 0) AS ndvi_fix10sza,
                    
                    (mcd43c1_band_nir_fix30sza_sza_30_vza_0_raa_sif - mcd43c1_band_red_fix30sza_sza_30_vza_0_raa_sif) / 
                    NULLIF((mcd43c1_band_nir_fix30sza_sza_30_vza_0_raa_sif + mcd43c1_band_red_fix30sza_sza_30_vza_0_raa_sif), 0) AS ndvi_fix30sza,
                    
                    (mcd43c1_band_nir_fix45sza_sza_45_vza_0_raa_sif - mcd43c1_band_red_fix45sza_sza_45_vza_0_raa_sif) / 
                    NULLIF((mcd43c1_band_nir_fix45sza_sza_45_vza_0_raa_sif + mcd43c1_band_red_fix45sza_sza_45_vza_0_raa_sif), 0) AS ndvi_fix45sza,
                    
                    (mcd43c1_band_nir_fix60sza_sza_60_vza_0_raa_sif - mcd43c1_band_red_fix60sza_sza_60_vza_0_raa_sif) / 
                    NULLIF((mcd43c1_band_nir_fix60sza_sza_60_vza_0_raa_sif + mcd43c1_band_red_fix60sza_sza_60_vza_0_raa_sif), 0) AS ndvi_fix60sza,
                    
											-- Multi-angle SIF yield calculations
											sif743 / NULLIF(par, 0) AS sif_yield_743_sifangel,

											(sif743 * (
													(mcd43c1_band_nir_nadir_sza_sif_vza_0_raa_sif - mcd43c1_band_red_nadir_sza_sif_vza_0_raa_sif) * 
													mcd43c1_band_nir_nadir_sza_sif_vza_0_raa_sif / 
													NULLIF((mcd43c1_band_nir_nadir_sza_sif_vza_0_raa_sif + mcd43c1_band_red_nadir_sza_sif_vza_0_raa_sif), 0)
											) / NULLIF((
													(mcd43c1_band_nir_sifangel_sza_sif_vza_sif_raa_sif - mcd43c1_band_red_sifangel_sza_sif_vza_sif_raa_sif) * 
													mcd43c1_band_nir_sifangel_sza_sif_vza_sif_raa_sif / 
													NULLIF((mcd43c1_band_nir_sifangel_sza_sif_vza_sif_raa_sif + mcd43c1_band_red_sifangel_sza_sif_vza_sif_raa_sif), 0)
											), 0)) / NULLIF(par, 0) AS sif_yield_743_nadir,

											(sif743 * (
													(mcd43c1_band_nir_hotspot_sza_sif_vza_0_raa_0 - mcd43c1_band_red_hotspot_sza_sif_vza_0_raa_0) * 
													mcd43c1_band_nir_hotspot_sza_sif_vza_0_raa_0 / 
													NULLIF((mcd43c1_band_nir_hotspot_sza_sif_vza_0_raa_0 + mcd43c1_band_red_hotspot_sza_sif_vza_0_raa_0), 0)
											) / NULLIF((
													(mcd43c1_band_nir_sifangel_sza_sif_vza_sif_raa_sif - mcd43c1_band_red_sifangel_sza_sif_vza_sif_raa_sif) * 
													mcd43c1_band_nir_sifangel_sza_sif_vza_sif_raa_sif / 
													NULLIF((mcd43c1_band_nir_sifangel_sza_sif_vza_sif_raa_sif + mcd43c1_band_red_sifangel_sza_sif_vza_sif_raa_sif), 0)
											), 0)) / NULLIF(par, 0) AS sif_yield_743_hotspot,

											(sif743 * (
													(mcd43c1_band_nir_fix10sza_sza_10_vza_0_raa_sif - mcd43c1_band_red_fix10sza_sza_10_vza_0_raa_sif) * 
													mcd43c1_band_nir_fix10sza_sza_10_vza_0_raa_sif / 
													NULLIF((mcd43c1_band_nir_fix10sza_sza_10_vza_0_raa_sif + mcd43c1_band_red_fix10sza_sza_10_vza_0_raa_sif), 0)
											) / NULLIF((
													(mcd43c1_band_nir_sifangel_sza_sif_vza_sif_raa_sif - mcd43c1_band_red_sifangel_sza_sif_vza_sif_raa_sif) * 
													mcd43c1_band_nir_sifangel_sza_sif_vza_sif_raa_sif / 
													NULLIF((mcd43c1_band_nir_sifangel_sza_sif_vza_sif_raa_sif + mcd43c1_band_red_sifangel_sza_sif_vza_sif_raa_sif), 0)
											), 0)) / NULLIF(par, 0) AS sif_yield_743_fix10sza,

											(sif743 * (
													(mcd43c1_band_nir_fix30sza_sza_30_vza_0_raa_sif - mcd43c1_band_red_fix30sza_sza_30_vza_0_raa_sif) * 
													mcd43c1_band_nir_fix30sza_sza_30_vza_0_raa_sif / 
													NULLIF((mcd43c1_band_nir_fix30sza_sza_30_vza_0_raa_sif + mcd43c1_band_red_fix30sza_sza_30_vza_0_raa_sif), 0)
											) / NULLIF((
													(mcd43c1_band_nir_sifangel_sza_sif_vza_sif_raa_sif - mcd43c1_band_red_sifangel_sza_sif_vza_sif_raa_sif) * 
													mcd43c1_band_nir_sifangel_sza_sif_vza_sif_raa_sif / 
													NULLIF((mcd43c1_band_nir_sifangel_sza_sif_vza_sif_raa_sif + mcd43c1_band_red_sifangel_sza_sif_vza_sif_raa_sif), 0)
											), 0)) / NULLIF(par, 0) AS sif_yield_743_fix30sza,

											(sif743 * (
													(mcd43c1_band_nir_fix45sza_sza_45_vza_0_raa_sif - mcd43c1_band_red_fix45sza_sza_45_vza_0_raa_sif) * 
													mcd43c1_band_nir_fix45sza_sza_45_vza_0_raa_sif / 
													NULLIF((mcd43c1_band_nir_fix45sza_sza_45_vza_0_raa_sif + mcd43c1_band_red_fix45sza_sza_45_vza_0_raa_sif), 0)
											) / NULLIF((
													(mcd43c1_band_nir_sifangel_sza_sif_vza_sif_raa_sif - mcd43c1_band_red_sifangel_sza_sif_vza_sif_raa_sif) * 
													mcd43c1_band_nir_sifangel_sza_sif_vza_sif_raa_sif / 
													NULLIF((mcd43c1_band_nir_sifangel_sza_sif_vza_sif_raa_sif + mcd43c1_band_red_sifangel_sza_sif_vza_sif_raa_sif), 0)
											), 0)) / NULLIF(par, 0) AS sif_yield_743_fix45sza,

											(sif743 * (
													(mcd43c1_band_nir_fix60sza_sza_60_vza_0_raa_sif - mcd43c1_band_red_fix60sza_sza_60_vza_0_raa_sif) * 
													mcd43c1_band_nir_fix60sza_sza_60_vza_0_raa_sif / 
													NULLIF((mcd43c1_band_nir_fix60sza_sza_60_vza_0_raa_sif + mcd43c1_band_red_fix60sza_sza_60_vza_0_raa_sif), 0)
											) / NULLIF((
													(mcd43c1_band_nir_sifangel_sza_sif_vza_sif_raa_sif - mcd43c1_band_red_sifangel_sza_sif_vza_sif_raa_sif) * 
													mcd43c1_band_nir_sifangel_sza_sif_vza_sif_raa_sif / 
													NULLIF((mcd43c1_band_nir_sifangel_sza_sif_vza_sif_raa_sif + mcd43c1_band_red_sifangel_sza_sif_vza_sif_raa_sif), 0)
											), 0)) / NULLIF(par, 0) AS sif_yield_743_fix60sza
																					
                FROM temp_intersect
            )
            TO %L
            WITH CSV HEADER
        ', csv_file);
        
        -- Drop the temporary intersection table
        DROP TABLE IF EXISTS temp_intersect;
    END LOOP;
END
$$;

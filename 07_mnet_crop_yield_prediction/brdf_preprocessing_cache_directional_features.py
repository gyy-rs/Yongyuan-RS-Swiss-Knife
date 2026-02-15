import glob
import pandas as pd
import numpy as np
import os
import sys
import warnings
import concurrent.futures
from tqdm import tqdm  # Recommended: install tqdm (pip install tqdm) to display a progress bar

# Ignore pandas fragmentation warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

# Add custom utility library path
sys.path.append('/pg_disk/@open_data/9-ESA_TROPOMI_SIF/py/utils')
try:
    from Make_mat_vectorized_C import  BRDF_degree_vectorized
except ImportError:
    print("Error: failed to import BRDF_degree_vectorized. Please verify the module path.")
    sys.exit(1)

# ================= Configuration =================
output_folder = '/pg_disk/tmp/raw_exports'
save_folder = '/pg_disk/tmp/raw_exports_S2'
file_pattern = f"{output_folder}/*.pkl"
work_dir = '/pg_disk/@open_data/9-ESA_TROPOMI_SIF/'

# Number of parallel workers (e.g. total CPU cores minus 2 to keep the system responsive)
# If memory is constrained, reduce this value manually, for example MAX_WORKERS = 4
MAX_WORKERS = 6

# Compression settings (options: 'gzip', 'zip', 'bz2', 'zstd', None)
# 'gzip' offers good compatibility and a balance between speed and compression ratio
COMPRESSION_METHOD = 'gzip'

# SZA: 0–60 degrees, 5° step, plus 57.5°
sza_list = list(range(0, 65, 5))
if 57.5 not in sza_list:
    sza_list.append(57.5)

# VZA: 0–60 degrees, 5° step, plus 57.5°
vza_list = list(range(0, 65, 5))
if 57.5 not in vza_list:
    vza_list.append(57.5)


# ================= Performance helper class =================
class FastArrayWrapper:
    """
    Wrap a NumPy array to provide a .values attribute and mimic a pandas
    Series, avoiding the overhead of Series objects when calling
    BRDF_degree_vectorized.
    """

    def __init__(self, data):
        self.values = data


# ================= Single-file processing function =================
def process_single_pkl(pkl_file):
    """
    Processing logic for a single file; this function is called in parallel.
    """
    base_name = os.path.basename(pkl_file)
    output_filename = os.path.join(save_folder, f"{base_name}")

    try:
        # 1. Read data
        df = pd.read_pickle(pkl_file)
        n_rows = len(df)

        # === Empty file check ===
        if n_rows == 0:
            return f"[SKIP] {base_name}: file is empty"

        # 2. Data preprocessing: convert to NumPy arrays (float64)
        def get_np(col_name):
            if col_name in df.columns:
                return df[col_name].to_numpy().astype(np.float64)
            else:
                return np.zeros(n_rows, dtype=np.float64)

        raa_np = get_np('raa')
        iso_r_np = get_np('iso_r')
        vol_r_np = get_np('vol_r')
        geo_r_np = get_np('geo_r')
        iso_n_np = get_np('iso_n')
        vol_n_np = get_np('vol_n')
        geo_n_np = get_np('geo_n')
        sif743_np = get_np('sif743')

        # Pre-extract auxiliary columns
        has_yield_info = 'sif_yield_743_sifangel' in df.columns
        if has_yield_info:
            yield_orig_np = get_np('sif_yield_743_sifangel')
            # Compute PAR while avoiding division by zero
            par_np = np.divide(sif743_np, yield_orig_np, out=np.zeros_like(sif743_np), where=yield_orig_np != 0)
        else:
            par_np = np.zeros(n_rows, dtype=np.float64)

        has_nirv_info = 'nirv_sifangel' in df.columns
        if has_nirv_info:
            nirv_orig_np = get_np('nirv_sifangel')

        # Wrap static parameters
        w_raa = FastArrayWrapper(raa_np)
        w_iso_r = FastArrayWrapper(iso_r_np)
        w_vol_r = FastArrayWrapper(vol_r_np)
        w_geo_r = FastArrayWrapper(geo_r_np)
        w_iso_n = FastArrayWrapper(iso_n_np)
        w_vol_n = FastArrayWrapper(vol_n_np)
        w_geo_n = FastArrayWrapper(geo_n_np)

        # ==========================================
        # Core optimization: cache intermediate results in a dictionary
        # ==========================================
        results_dict = {}

        # ----------------------------------------
        # Part A: computations at observed geometry
        # ----------------------------------------
        suffix_raw = "raw_obs"
        w_sza_raw = FastArrayWrapper(get_np('sza'))
        w_vza_raw = FastArrayWrapper(get_np('vza'))

        # Evaluate BRDF for the red band
        brdf_red = BRDF_degree_vectorized(w_sza_raw, w_vza_raw, w_raa, w_iso_r, w_vol_r, w_geo_r, band='Red')

        # Evaluate BRDF for the NIR band
        brdf_nir = BRDF_degree_vectorized(w_sza_raw, w_vza_raw, w_raa, w_iso_n, w_vol_n, w_geo_n, band='NIR')

        results_dict[f'mcd43c1_band_red_{suffix_raw}'] = brdf_red
        results_dict[f'mcd43c1_band_nir_{suffix_raw}'] = brdf_nir

        sum_bands = brdf_nir + brdf_red
        diff_bands = brdf_nir - brdf_red

        # NDVI computation
        results_dict[f'ndvi_{suffix_raw}'] = np.divide(diff_bands, sum_bands, out=np.full(n_rows, np.nan),
                                                       where=sum_bands != 0)

        # EVI2 computation
        evi_denom = brdf_nir + 2.4 * brdf_red + 1
        results_dict[f'evi2_{suffix_raw}'] = np.divide(2.5 * diff_bands, evi_denom, out=np.full(n_rows, np.nan),
                                                       where=evi_denom != 0)

        # NIRv computation
        nirv_curr = np.divide(diff_bands, sum_bands, out=np.full(n_rows, np.nan), where=sum_bands != 0) * brdf_nir
        results_dict[f'nirv_{suffix_raw}'] = nirv_curr

        # SIF and yield computation
        if has_nirv_info:
            ratio = np.divide(nirv_curr, nirv_orig_np, out=np.zeros_like(nirv_curr), where=nirv_orig_np != 0)
            sif_corr = sif743_np * ratio
            results_dict[f'sif743_{suffix_raw}'] = sif_corr

            if has_yield_info:
                results_dict[f'sif_yield_743_{suffix_raw}'] = np.divide(sif_corr, par_np, out=np.full(n_rows, np.nan),
                                                                        where=par_np != 0)

        # ----------------------------------------
        # Part B: multi-angle simulations
        # ----------------------------------------
        # Create a float64 all-ones array for broadcasting
        ones_arr = np.ones(n_rows, dtype=np.float64)

        for sza_val in sza_list:
            sza_arr = ones_arr * sza_val
            w_sza = FastArrayWrapper(sza_arr)

            for vza_val in vza_list:
                vza_arr = ones_arr * vza_val
                w_vza = FastArrayWrapper(vza_arr)

                suffix = f"fix_sza_{sza_val}_vza_{vza_val}"

                # 1. Compute BRDF (explicitly specify band to match hotspot configuration)
                brdf_red = BRDF_degree_vectorized(w_sza, w_vza, w_raa, w_iso_r, w_vol_r, w_geo_r, band='Red')
                brdf_nir = BRDF_degree_vectorized(w_sza, w_vza, w_raa, w_iso_n, w_vol_n, w_geo_n, band='NIR')

                results_dict[f'mcd43c1_band_red_{suffix}'] = brdf_red
                results_dict[f'mcd43c1_band_nir_{suffix}'] = brdf_nir

                # 2. Compute vegetation indices
                sum_bands = brdf_nir + brdf_red
                diff_bands = brdf_nir - brdf_red

                # NDVI
                results_dict[f'ndvi_{suffix}'] = np.divide(diff_bands, sum_bands, out=np.full(n_rows, np.nan),
                                                           where=sum_bands != 0)

                # EVI2
                evi_denom = brdf_nir + 2.4 * brdf_red + 1
                results_dict[f'evi2_{suffix}'] = np.divide(2.5 * diff_bands, evi_denom, out=np.full(n_rows, np.nan),
                                                           where=evi_denom != 0)

                # NIRv
                nirv_curr = np.divide(diff_bands, sum_bands, out=np.full(n_rows, np.nan),
                                      where=sum_bands != 0) * brdf_nir
                results_dict[f'nirv_{suffix}'] = nirv_curr

                # 3. SIF and yield
                if has_nirv_info:
                    ratio = np.divide(nirv_curr, nirv_orig_np, out=np.zeros_like(nirv_curr), where=nirv_orig_np != 0)
                    sif_corr = sif743_np * ratio
                    results_dict[f'sif743_{suffix}'] = sif_corr

                    if has_yield_info:
                        results_dict[f'sif_yield_743_{suffix}'] = np.divide(sif_corr, par_np,
                                                                            out=np.full(n_rows, np.nan),
                                                                            where=par_np != 0)

        # ----------------------------------------
        # Part C: merge with original dataframe and write compressed output
        # ----------------------------------------
        if has_yield_info:
            results_dict['par_calculated'] = par_np

        df_new_cols = pd.DataFrame(results_dict, index=df.index)
        df_final = pd.concat([df, df_new_cols], axis=1)

        # Save as compressed pickle
        df_final.to_pickle(output_filename, compression=COMPRESSION_METHOD)

        return None  # Success

    except Exception as e:
        import traceback
        traceback_str = traceback.format_exc()
        return f"[ERROR] {base_name}: {str(e)}"


# ================= Main entry point =================
if __name__ == '__main__':
    # Change working directory
    os.chdir(work_dir)

    # Ensure that the output directory exists
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Collect all matching input files
    pkl_files = glob.glob(file_pattern)

    if not pkl_files:
        print("No .pkl files were found.")
    else:
        print(f"Found {len(pkl_files)} .pkl files.")
        print(f"Parallel configuration: using {MAX_WORKERS} workers")
        print(f"Compression method: {COMPRESSION_METHOD}")
        print("Starting parallel processing...")

        # Use ProcessPoolExecutor for parallel computation
        with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Submit all tasks and track progress with tqdm
            results = list(tqdm(executor.map(process_single_pkl, pkl_files), total=len(pkl_files), unit="file"))

        # Print a summary of errors and skipped files (if any)
        print("\nProcessing summary:")
        error_count = 0
        skip_count = 0
        for res in results:
            if res:  # Non-None indicates an error or a skipped file
                if "[ERROR]" in res:
                    print(res)
                    error_count += 1
                elif "[SKIP]" in res:
                    skip_count += 1

        print("-" * 30)
        print(f"Successful: {len(pkl_files) - error_count - skip_count}")
        print(f"Skipped: {skip_count}")
        print(f"Failed: {error_count}")
        print("All tasks completed.")

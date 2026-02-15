import sys
import os
import csv
import json
import glob
import time
import random
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from datetime import datetime
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# ================= 1. Paths and environment configuration =================
# Add the project-specific script directory to the Python path
train_script_dir = "/pg_disk/@open_data/@Paper3th.JRS_US_yield_R1/@2.new_deep_learning/experiment_10"
if train_script_dir not in sys.path:
    sys.path.append(train_script_dir)

# Import custom data loader and M-Net model
# Ensure that DataLoader.py and model.py are accessible in the configured paths
from DataLoader import RemoteSensingDataLoader
from model import MNet_DualStream

# ================= 2. Global configuration =================
TRAIN_CSV_PATH = '/home/gyy/data_SIF/train_all_n.csv'
TEST_CSV_PATH = '/home/gyy/data_SIF/test_all_n.csv'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Output directory configuration
CUMULATIVE_CSV_PATH = 'results_plotting_metrics.csv'  # stores R² and RMSE metrics
PRED_SAVE_DIR = 'predictions_pkl_for_plot'  # stores .pkl files for visualization
ATTN_SAVE_DIR = 'attention_weights_json'  # stores attention weights

# Ensure that output directories exist
os.makedirs(PRED_SAVE_DIR, exist_ok=True)
os.makedirs(ATTN_SAVE_DIR, exist_ok=True)

# Define the three target categories
TARGET_CATEGORIES = [
    'STRUCT_MODIS',
    'PURE_SIF',
    'PURE_SIF_VIS'
]

# Define key phenological dates between 07-01 and 08-29
# If only the final date is needed, keep ['08-29'] only
TARGET_DATES = [
    '07-01', '07-15', '07-30', '08-14', '08-29'
]

# Golden angles automatically extracted from the data (populated at runtime)
GOLDEN_ANGLES = []

# Feature-category lookup tables
SCALAR_FEATURES_LUT = [
    'EVI', 'Lai', 'NDVI', 'NIRv', 'cld_fr', 'lcmask', 'par',
    'phi_f_743', 'precipitation', 'precipitation sum',
    'temperature_max', 'temperature_mean', 'temperature_min'
]
ANGULAR_FEATURES_LUT = ['evi2', 'ndvi', 'nirv', 'sif743', 'sif_yield_743']
SCALAR_SET = set(SCALAR_FEATURES_LUT)
ANGULAR_SET = set(ANGULAR_FEATURES_LUT)


# ================= 3. Data processing utilities =================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def get_feature_type(feature_name):
    core_name = feature_name
    if "_fix_sza" in core_name: core_name = core_name.split("_fix_sza")[0]
    if "_raw_obs" in core_name: core_name = core_name.split("_raw_obs")[0]
    if core_name in SCALAR_SET:
        return 'SCALAR', core_name
    elif core_name in ANGULAR_SET:
        return 'ANGULAR', core_name
    else:
        return 'UNKNOWN', core_name


def clean_feature_name(name):
    if "_fix_sza" in name: return name.split("_fix_sza")[0]
    if "_raw_obs" in name: return name.split("_raw_obs")[0]
    return name


def check_and_print_feature_status(loader, category_enum):
    sample_feats = loader.get_feature_names(category_enum, 0, 0)
    found_angulars = set()
    for feat in sample_feats:
        ftype, core_name = get_feature_type(feat)
        if ftype == 'ANGULAR': found_angulars.add(core_name)
    return list(found_angulars)


def get_target_angles():
    sza_list = list(range(0, 65, 5))
    if 57.5 not in sza_list: sza_list.append(57.5)
    sza_list.sort()
    vza_list = list(range(0, 65, 5))
    if 57.5 not in vza_list: vza_list.append(57.5)
    vza_list.sort()
    return sza_list, vza_list


def auto_extract_golden_angles(loader, category_enum):
    """Automatically extract golden viewing angles with the highest yield correlation."""
    angular_feats = check_and_print_feature_status(loader, category_enum)
    if not angular_feats:
        return []

    warnings.filterwarnings("ignore")
    meta_cols = ['YEAR', 'COUNTY', 'INDEX', 'YIELD']
    time_cols = sorted([c for c in loader.df.columns if c not in meta_cols and isinstance(c, str)])
    sza_list, vza_list = get_target_angles()
    angle_stats = []

    # Perform a lightweight sampling analysis without progress bars
    # to avoid interleaved outputs in multiprocessing
    for sza in sza_list:
        for vza in vza_list:
            raw_feats = loader.get_feature_names(category_enum, sza, vza)
            target_feats = [f for f in raw_feats if get_feature_type(f)[0] == 'ANGULAR']
            if not target_feats: continue

            subset = loader.df[loader.df['INDEX'].isin(target_feats)].copy()
            if subset.empty: continue

            vals = np.nan_to_num(subset[time_cols].values)
            mean_vals = np.mean(vals, axis=1)
            yields = subset['YIELD'].values

            if np.std(mean_vals) > 0 and np.std(yields) > 0:
                corr, _ = pearsonr(mean_vals, yields)
                angle_stats.append({'SZA': sza, 'VZA': vza, 'AbsCorr': abs(corr)})

    if not angle_stats: return []
    df_stats = pd.DataFrame(angle_stats).sort_values('AbsCorr', ascending=False)
    top_angles = df_stats.head(10)[['SZA', 'VZA']].values.tolist()
    return [(int(s) if s.is_integer() else s, int(v) if v.is_integer() else v) for s, v in top_angles]


def get_dual_input_tensors(loader, category_enum, golden_angles, start_date=None, end_date=None):
    """Construct dual-stream input tensors for angular and scalar features."""
    base_feats = loader.get_feature_names(category_enum, 0, 0)
    meta_cols = ['YEAR', 'COUNTY', 'INDEX', 'YIELD', 'SZA', 'VZA']
    all_cols = loader.df.columns.tolist()
    all_time_cols = sorted([c for c in all_cols if c not in meta_cols and isinstance(c, str)])

    valid_time_cols = all_time_cols
    if start_date: valid_time_cols = [c for c in valid_time_cols if c >= start_date]
    if end_date: valid_time_cols = [c for c in valid_time_cols if c <= end_date]

    if not valid_time_cols: return None, None, None, [], []

    subset_0 = loader.df[loader.df['INDEX'].isin(base_feats)].copy()
    unique_samples = subset_0[['YEAR', 'COUNTY', 'YIELD']].drop_duplicates().sort_values(['YEAR', 'COUNTY'])

    feature_names_raw = sorted(subset_0['INDEX'].unique())
    all_clean_names = sorted(list(set([clean_feature_name(f) for f in feature_names_raw])))

    ang_feats_list = [f for f in all_clean_names if get_feature_type(f)[0] == 'ANGULAR']
    sca_feats_list = [f for f in all_clean_names if get_feature_type(f)[0] == 'SCALAR']

    N, T, A = len(unique_samples), len(valid_time_cols), len(golden_angles)
    F_ang, F_sca = len(ang_feats_list), len(sca_feats_list)

    X_ang = np.zeros((N, T, A, F_ang), dtype=np.float32)
    X_sca = np.zeros((N, T, F_sca), dtype=np.float32)
    y_tensor = unique_samples['YIELD'].values.astype(np.float32)

    ang_map = {name: i for i, name in enumerate(ang_feats_list)}
    sca_map = {name: i for i, name in enumerate(sca_feats_list)}

    # Populate the angular feature tensor
    if F_ang > 0:
        for a_idx, (sza, vza) in enumerate(golden_angles):
            raw_feats_at_angle = loader.get_feature_names(category_enum, sza, vza)
            clean_to_raw_map = {clean_feature_name(r): r for r in raw_feats_at_angle}
            for feat_name in ang_feats_list:
                target = clean_to_raw_map.get(feat_name)
                if target:
                    sub_f = loader.df[loader.df['INDEX'] == target]
                    if not sub_f.empty:
                        merged = pd.merge(unique_samples[['YEAR', 'COUNTY']], sub_f, on=['YEAR', 'COUNTY'], how='left')
                        X_ang[:, :, a_idx, ang_map[feat_name]] = merged[valid_time_cols].fillna(0).values

    # Populate the scalar feature tensor
    if F_sca > 0:
        for feat_name in sca_feats_list:
            sub_f = loader.df[loader.df['INDEX'] == feat_name]
            if not sub_f.empty:
                merged = pd.merge(unique_samples[['YEAR', 'COUNTY']], sub_f, on=['YEAR', 'COUNTY'], how='left')
                X_sca[:, :, sca_map[feat_name]] = merged[valid_time_cols].fillna(0).values

    return X_ang, X_sca, y_tensor, sca_feats_list, ang_feats_list


# ================= 4. Core training loop (modified version) =================
def train_and_evaluate_mnet_loop(category_enum, category_name, trail_id, golden_angles, start_date=None, end_date=None):
    """
    Modified training loop that records predictions at the epoch
    with the best validation R².
    """
    loader_train = RemoteSensingDataLoader(TRAIN_CSV_PATH)
    loader_test = RemoteSensingDataLoader(TEST_CSV_PATH)

    # Retrieve tensors for training and testing
    Xa_tr, Xs_tr, y_tr, _, _ = get_dual_input_tensors(loader_train, category_enum, golden_angles, start_date, end_date)
    Xa_te, Xs_te, y_te, _, _ = get_dual_input_tensors(loader_test, category_enum, golden_angles, start_date, end_date)

    if y_tr is None: return None

    N, T, A, Fa = Xa_tr.shape
    _, _, Fs = Xs_tr.shape

    # Standardization
    scaler_a, scaler_s, y_scaler = StandardScaler(), StandardScaler(), StandardScaler()

    if Fa > 0:
        Xa_tr = scaler_a.fit_transform(Xa_tr.reshape(-1, Fa)).reshape(N, T, A, Fa)
        Xa_te = scaler_a.transform(Xa_te.reshape(-1, Fa)).reshape(Xa_te.shape[0], T, A, Fa)

    if Fs > 0:
        Xs_tr = scaler_s.fit_transform(Xs_tr.reshape(-1, Fs)).reshape(N, T, Fs)
        Xs_te = scaler_s.transform(Xs_te.reshape(-1, Fs)).reshape(Xs_te.shape[0], T, Fs)

    y_tr_scaled = y_scaler.fit_transform(y_tr.reshape(-1, 1)).flatten()
    y_te_scaled = y_scaler.transform(y_te.reshape(-1, 1)).flatten()

    # DataLoader instances for training and evaluation
    train_dl = DataLoader(
        TensorDataset(torch.tensor(Xa_tr).float(), torch.tensor(Xs_tr).float(), torch.tensor(y_tr_scaled).float()),
        batch_size=32, shuffle=True, drop_last=True)
    test_dl = DataLoader(
        TensorDataset(torch.tensor(Xa_te).float(), torch.tensor(Xs_te).float(), torch.tensor(y_te_scaled).float()),
        batch_size=64, shuffle=False)

    # Model initialization
    model = MNet_DualStream(ang_input_dim=Fa, sca_input_dim=Fs, time_steps=T, num_angles=A, hidden_dim=48,
                            dropout=0.35).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-3)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=15)

    best_r2 = -999.0
    best_rmse = 999.0
    best_attn_weights = None

    # Store predictions and targets at the best-performing epoch
    best_preds_final = []
    best_trues_final = []

    for epoch in range(120):  # slightly fewer epochs to reduce runtime
        model.train()
        for bx_a, bx_s, by in train_dl:
            bx_a, bx_s, by = bx_a.to(DEVICE), bx_s.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            out = model(bx_a, bx_s)
            pred = out[0].squeeze() if isinstance(out, tuple) else out.squeeze()
            loss = criterion(pred, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        preds, trues, attn_list = [], [], []
        with torch.no_grad():
            for bx_a, bx_s, by in test_dl:
                bx_a, bx_s = bx_a.to(DEVICE), bx_s.to(DEVICE)
                out = model(bx_a, bx_s)
                if isinstance(out, tuple):
                    p, attn = out[0].squeeze(), out[1]
                    if attn is not None: attn_list.append(attn.cpu().numpy())
                else:
                    p = out.squeeze()

                # Invert the normalization for evaluation
                preds.extend(y_scaler.inverse_transform(p.reshape(-1, 1).cpu().numpy()).flatten())
                trues.extend(y_scaler.inverse_transform(by.reshape(-1, 1).cpu().numpy()).flatten())

        epoch_r2 = r2_score(trues, preds)
        scheduler.step(epoch_r2)

        if epoch_r2 > best_r2:
            best_r2 = epoch_r2
            best_rmse = np.sqrt(mean_squared_error(trues, preds))
            # Capture predictions at the epoch with the best R²
            best_preds_final = preds
            best_trues_final = trues

            if attn_list:
                all_attn = np.concatenate(attn_list, axis=0)
                best_attn_weights = np.mean(all_attn, axis=0)

    return {
        "R2": best_r2,
        "rmse": best_rmse,
        "attention_weights": best_attn_weights,
        "y_true": best_trues_final,  # true yields
        "y_pred": best_preds_final   # predicted yields
    }


def save_predictions_pkl(y_true, y_pred, category, date, trial):
    """
    Save predictions as a .pkl file compatible with downstream plotting scripts.
    Example for reading: df = pd.read_pickle(filepath)
    """
    # Construct filename: M-Net_{Category}_{Date}_Trial{Trial}.pkl

    # Replace special characters to avoid path issues
    safe_cat = category.replace(" ", "_").replace("/", "_")
    fname = f"M-Net_{safe_cat}_{date}_Trial{trial}.pkl"
    save_path = os.path.join(PRED_SAVE_DIR, fname)

    # Ensure that arrays are one-dimensional
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
    df.to_pickle(save_path)
    return fname


# ================= 5. Multiprocessing worker =================
def run_category_task(cat_name, file_lock):
    print(f"\n[Worker {os.getpid()}] Processing: {cat_name}")

    # Initialize loader and extract golden angles
    loader = RemoteSensingDataLoader(TRAIN_CSV_PATH)
    if not hasattr(loader.CATEGORY, cat_name):
        print(f"  [Error] Category {cat_name} not found.")
        return

    cat_enum = getattr(loader.CATEGORY, cat_name)
    # Each worker extracts golden angles independently (or they could be pre-computed and passed in)
    local_golden_angles = auto_extract_golden_angles(loader, cat_enum)

    # Iterate over target dates
    for date in TARGET_DATES:
        # Loop over five independent trials per configuration
        for trial in range(1, 6):
            set_seed(42 + trial)  # ensure reproducibility

            try:
                res = train_and_evaluate_mnet_loop(
                    cat_enum, cat_name, trial,
                    local_golden_angles,
                    start_date='07-01', end_date=date
                )
            except Exception as e:
                print(f"  [Error] {cat_name} @ {date} Trial {trial}: {e}")
                continue

            if res:
                # 1. Save CSV metrics
                with file_lock:
                    with open(CUMULATIVE_CSV_PATH, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(["M-Net", cat_name, date, trial, f"{res['R2']:.4f}", f"{res['rmse']:.4f}"])

                # 2. Save prediction .pkl files for visualization
                if res['y_true'] and res['y_pred']:
                    pkl_name = save_predictions_pkl(res['y_true'], res['y_pred'], cat_name, date, trial)
                    # print(f"    -> Saved PKL: {pkl_name}")

                # 3. Save attention weights
                if res['attention_weights'] is not None:
                    attn_path = os.path.join(ATTN_SAVE_DIR, f"{cat_name}_T{trial}_{date}_attn.json")
                    w = res['attention_weights']
                    w_list = w.tolist() if isinstance(w, np.ndarray) else w
                    with open(attn_path, 'w') as jf:
                        json.dump({'temporal': w_list, 'date': date}, jf)

                print(f"  > {cat_name} | {date} | T{trial} | R2: {res['R2']:.4f} | RMSE: {res['rmse']:.4f}")

    print(f"[Worker {os.getpid()}] Finished {cat_name}")


# ================= 6. Main entry point =================
if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    # Initialize CSV header if the metrics file does not yet exist
    if not os.path.exists(CUMULATIVE_CSV_PATH):
        with open(CUMULATIVE_CSV_PATH, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(["Model", "Category_ID", "Date", "Trial_ID", "R2", "RMSE"])

    print(f"{'#' * 60}")
    print("Starting M-Net experiment to generate prediction files for visualization")
    print(f"   Target categories: {TARGET_CATEGORIES}")
    print(f"   Target dates: {TARGET_DATES}")
    print(f"   Prediction outputs: {PRED_SAVE_DIR}")
    print(f"{'#' * 60}")

    manager = mp.Manager()
    lock = manager.Lock()

    # Create task list
    tasks = [(cat, lock) for cat in TARGET_CATEGORIES]

    # Launch a process pool (up to one process per task, capped at 4)
    with mp.Pool(processes=min(len(tasks), 4)) as pool:
        pool.starmap(run_category_task, tasks)

    print("\nAll experiments have completed. You can now run plotting scripts to load the generated .pkl files.")

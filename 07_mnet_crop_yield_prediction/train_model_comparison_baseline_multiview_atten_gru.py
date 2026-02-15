import sys
import os
import csv
import json
import random
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# ================= 1. Paths and environment configuration =================
train_script_dir = "/pg_disk/@open_data/@Paper3th.JRS_US_yield_R1/@2.new_deep_learning/experiment_10"
if train_script_dir not in sys.path:
    sys.path.append(train_script_dir)

from DataLoader import RemoteSensingDataLoader

# ================= 2. Global configuration =================
TRAIN_CSV_PATH = '/home/gyy/data_SIF/train_all_n.csv'
TEST_CSV_PATH = '/home/gyy/data_SIF/test_all_n.csv'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Output directory configuration (file names distinguish this ablation experiment)
CUMULATIVE_CSV_PATH = 'results_ablation_gru_attn_final.csv'
PRED_SAVE_DIR = 'predictions_ablation_gru_attn_final'
ATTN_SAVE_DIR = 'attention_weights_gru_attn_final'

os.makedirs(PRED_SAVE_DIR, exist_ok=True)
os.makedirs(ATTN_SAVE_DIR, exist_ok=True)

TARGET_CATEGORIES = ['STRUCT_MODIS', 'PURE_SIF', 'PURE_SIF_VIS']

# Modification: evaluate only the final date instead of running all intermediate dates
TARGET_DATES = ['08-29']

SCALAR_FEATURES_LUT = [
    'EVI', 'Lai', 'NDVI', 'NIRv', 'cld_fr', 'lcmask', 'par',
    'phi_f_743', 'precipitation', 'precipitation sum',
    'temperature_max', 'temperature_mean', 'temperature_min'
]
ANGULAR_FEATURES_LUT = ['evi2', 'ndvi', 'nirv', 'sif743', 'sif_yield_743']
SCALAR_SET = set(SCALAR_FEATURES_LUT)
ANGULAR_SET = set(ANGULAR_FEATURES_LUT)


# ================= 3. Model definition: GRU + temporal attention =================
class TemporalAttention(nn.Module):
    """Temporal attention mechanism over the time dimension."""

    def __init__(self, hidden_dim):
        super(TemporalAttention, self).__init__()
        self.attn_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        # x: (Batch, Time, Hidden)
        scores = self.attn_fc(x)  # (B, T, 1)
        weights = F.softmax(scores, dim=1)  # (B, T, 1)
        context = torch.sum(x * weights, dim=1)  # (B, Hidden) weighted sum over time
        return context, weights.squeeze(-1)


class GRU_Attention_Model(nn.Module):
    def __init__(self, input_dim, time_steps, hidden_dim=48, dropout=0.35):
        super(GRU_Attention_Model, self).__init__()

        # 1. Feature projection (hyperparameters aligned with the M-Net baseline)
        self.feat_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 2. Bi-GRU (hyperparameters aligned with the M-Net baseline)
        self.bi_gru = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim // 2,
                             num_layers=1, batch_first=True, bidirectional=True)
        self.norm = nn.LayerNorm(hidden_dim)

        # 3. Temporal attention layer (replacing mean-pooling)
        self.temporal_attn = TemporalAttention(hidden_dim)

        # 4. Regression head (aligned with the M-Net baseline configuration)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = self.feat_projection(x)
        gru_out, _ = self.bi_gru(x)
        x = self.norm(x + gru_out)
        x_pool, attn_weights = self.temporal_attn(x)
        out = self.regressor(x_pool)
        return out, attn_weights


# ================= 4. Data processing utilities =================
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


def auto_extract_golden_angles(loader, category_enum):
    angular_feats = check_and_print_feature_status(loader, category_enum)
    if not angular_feats: return []

    warnings.filterwarnings("ignore")
    meta_cols = ['YEAR', 'COUNTY', 'INDEX', 'YIELD']
    time_cols = sorted([c for c in loader.df.columns if c not in meta_cols and isinstance(c, str)])

    sza_list = list(range(0, 65, 10))
    vza_list = list(range(0, 65, 10))
    angle_stats = []

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

            if np.std(mean_vals) > 0:
                corr, _ = pearsonr(mean_vals, yields)
                angle_stats.append({'SZA': sza, 'VZA': vza, 'AbsCorr': abs(corr)})

    if not angle_stats: return []
    df_stats = pd.DataFrame(angle_stats).sort_values('AbsCorr', ascending=False)
    top_angles = df_stats.head(5)[['SZA', 'VZA']].values.tolist()
    return [(int(s), int(v)) for s, v in top_angles]


def get_combined_input_tensors(loader, category_enum, golden_angles, start_date=None, end_date=None):
    base_feats = loader.get_feature_names(category_enum, 0, 0)
    meta_cols = ['YEAR', 'COUNTY', 'INDEX', 'YIELD', 'SZA', 'VZA']
    all_time_cols = sorted([c for c in loader.df.columns if c not in meta_cols and isinstance(c, str)])

    valid_time_cols = all_time_cols
    if start_date: valid_time_cols = [c for c in valid_time_cols if c >= start_date]
    if end_date: valid_time_cols = [c for c in valid_time_cols if c <= end_date]

    if not valid_time_cols: return None, None

    subset_0 = loader.df[loader.df['INDEX'].isin(base_feats)].copy()
    unique_samples = subset_0[['YEAR', 'COUNTY', 'YIELD']].drop_duplicates().sort_values(['YEAR', 'COUNTY'])

    feature_names_raw = sorted(subset_0['INDEX'].unique())
    all_clean_names = sorted(list(set([clean_feature_name(f) for f in feature_names_raw])))

    ang_feats_list = [f for f in all_clean_names if get_feature_type(f)[0] == 'ANGULAR']
    sca_feats_list = [f for f in all_clean_names if get_feature_type(f)[0] == 'SCALAR']

    N, T = len(unique_samples), len(valid_time_cols)
    A = len(golden_angles)
    F_ang = len(ang_feats_list)
    F_sca = len(sca_feats_list)

    total_dim = (A * F_ang) + F_sca
    X_combined = np.zeros((N, T, total_dim), dtype=np.float32)
    y_tensor = unique_samples['YIELD'].values.astype(np.float32)

    current_dim_idx = 0

    if F_ang > 0:
        for a_idx, (sza, vza) in enumerate(golden_angles):
            raw_feats_at_angle = loader.get_feature_names(category_enum, sza, vza)
            clean_to_raw_map = {clean_feature_name(r): r for r in raw_feats_at_angle}

            for feat_name in ang_feats_list:
                target = clean_to_raw_map.get(feat_name)
                if target:
                    sub_f = loader.df[loader.df['INDEX'] == target]
                    merged = pd.merge(unique_samples[['YEAR', 'COUNTY']], sub_f, on=['YEAR', 'COUNTY'], how='left')
                    X_combined[:, :, current_dim_idx] = merged[valid_time_cols].fillna(0).values
                current_dim_idx += 1

    if F_sca > 0:
        for feat_name in sca_feats_list:
            sub_f = loader.df[loader.df['INDEX'] == feat_name]
            merged = pd.merge(unique_samples[['YEAR', 'COUNTY']], sub_f, on=['YEAR', 'COUNTY'], how='left')
            X_combined[:, :, current_dim_idx] = merged[valid_time_cols].fillna(0).values
            current_dim_idx += 1

    return X_combined, y_tensor


# ================= 5. Training loop =================
def train_gru_attn(category_enum, category_name, trail_id, golden_angles, start_date=None, end_date=None):
    loader_train = RemoteSensingDataLoader(TRAIN_CSV_PATH)
    loader_test = RemoteSensingDataLoader(TEST_CSV_PATH)

    X_tr, y_tr = get_combined_input_tensors(loader_train, category_enum, golden_angles, start_date, end_date)
    X_te, y_te = get_combined_input_tensors(loader_test, category_enum, golden_angles, start_date, end_date)

    if X_tr is None: return None

    N, T, F_in = X_tr.shape

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    X_tr = scaler_x.fit_transform(X_tr.reshape(-1, F_in)).reshape(N, T, F_in)
    X_te = scaler_x.transform(X_te.reshape(-1, F_in)).reshape(X_te.shape[0], T, F_in)

    y_tr_scaled = scaler_y.fit_transform(y_tr.reshape(-1, 1)).flatten()
    y_te_scaled = scaler_y.transform(y_te.reshape(-1, 1)).flatten()

    train_dl = DataLoader(
        TensorDataset(torch.tensor(X_tr).float(), torch.tensor(y_tr_scaled).float()),
        batch_size=32, shuffle=True, drop_last=True
    )
    test_dl = DataLoader(
        TensorDataset(torch.tensor(X_te).float(), torch.tensor(y_te_scaled).float()),
        batch_size=64, shuffle=False
    )

    model = GRU_Attention_Model(input_dim=F_in, time_steps=T, hidden_dim=48, dropout=0.35).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-3)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=15)

    best_r2 = -999.0
    best_rmse = 999.0
    best_preds, best_trues = [], []
    best_attn_weights = None

    for epoch in range(120):
        model.train()
        for bx, by in train_dl:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            out, _ = model(bx)
            loss = criterion(out.squeeze(), by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        preds, trues, attn_list = [], [], []
        with torch.no_grad():
            for bx, by in test_dl:
                bx = bx.to(DEVICE)
                out, attn = model(bx)

                preds.extend(scaler_y.inverse_transform(out.cpu().numpy()).flatten())
                trues.extend(scaler_y.inverse_transform(by.reshape(-1, 1).cpu().numpy()).flatten())
                attn_list.append(attn.cpu().numpy())

        epoch_r2 = r2_score(trues, preds)
        scheduler.step(epoch_r2)

        if epoch_r2 > best_r2:
            best_r2 = epoch_r2
            best_rmse = np.sqrt(mean_squared_error(trues, preds))
            best_preds = preds
            best_trues = trues
            if attn_list:
                all_attn = np.concatenate(attn_list, axis=0)
                best_attn_weights = np.mean(all_attn, axis=0)

    return {
        "R2": best_r2,
        "rmse": best_rmse,
        "y_true": best_trues,
        "y_pred": best_preds,
        "attention_weights": best_attn_weights
    }


# ================= 6. Multiprocessing worker =================
def run_task(cat_name, file_lock):
    print(f"\n[Worker {os.getpid()}] Processing: {cat_name}")
    loader = RemoteSensingDataLoader(TRAIN_CSV_PATH)
    if not hasattr(loader.CATEGORY, cat_name): return

    cat_enum = getattr(loader.CATEGORY, cat_name)
    local_angles = auto_extract_golden_angles(loader, cat_enum)

    # The loop runs only for the final date '08-29' in this ablation setting
    for date in TARGET_DATES:
        for trial in range(1, 6):
            set_seed(42 + trial)
            try:
                res = train_gru_attn(cat_enum, cat_name, trial, local_angles, '07-01', date)
            except Exception as e:
                print(f"Error {cat_name} {date}: {e}")
                continue

            if res:
                with file_lock:
                    with open(CUMULATIVE_CSV_PATH, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(["GRU-Attn", cat_name, date, trial, f"{res['R2']:.4f}", f"{res['rmse']:.4f}"])

                safe_cat = cat_name.replace(" ", "_")
                fname = f"GRU-Attn_{safe_cat}_{date}_Trial{trial}.pkl"
                pd.DataFrame({'y_true': res['y_true'], 'y_pred': res['y_pred']}).to_pickle(
                    os.path.join(PRED_SAVE_DIR, fname)
                )

                if res['attention_weights'] is not None:
                    with open(os.path.join(ATTN_SAVE_DIR, f"{fname}.json"), 'w') as jf:
                        json.dump(res['attention_weights'].tolist(), jf)

                print(f"  > GRU-Attn | {cat_name} | {date} | T{trial} | R2: {res['R2']:.4f}")


if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True)
    except:
        pass

    if not os.path.exists(CUMULATIVE_CSV_PATH):
        with open(CUMULATIVE_CSV_PATH, 'w', newline='') as f:
            csv.writer(f).writerow(["Model", "Category_ID", "Date", "Trial_ID", "R2", "RMSE"])

    manager = mp.Manager()
    lock = manager.Lock()
    tasks = [(cat, lock) for cat in TARGET_CATEGORIES]

    print("Starting GRU+Attention ablation study (final date only)...")
    print(f"   Target date(s): {TARGET_DATES}")

    with mp.Pool(processes=min(len(tasks), 4)) as pool:
        pool.starmap(run_task, tasks)
    print("Done.")

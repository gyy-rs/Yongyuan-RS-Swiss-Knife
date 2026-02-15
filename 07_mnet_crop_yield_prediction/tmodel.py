import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader as TorchDataLoader
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import os
import csv
import sys
import multiprocessing as mp
from tqdm import tqdm
import time
import threading

# --- Import custom data loader and model definitions ---
from DataLoader import RemoteSensingDataLoader
import tmodel as models

# --- Configuration ---
TRAIN_CSV_PATH = '/home/gyy/data_SIF/train_all_n.csv'
TEST_CSV_PATH = '/home/gyy/data_SIF/test_all_n.csv'
RESULT_CSV_PATH = r'/home/gyy/data_SIF/experiment_results-PURE_v2.csv'

# Use GPU for deep learning workloads; each worker manages its own device context
DEVICE = 'cuda'  # 'cuda'

# Experimental settings
TRIALS = 5  # number of independent runs per configuration
BATCH_SIZE = 32     # batch size; smaller values can be beneficial for small samples
LR = 0.0005         # reduced initial learning rate, used together with a scheduler
EPOCHS = 150        # number of training epochs (can be increased when using a scheduler)

# Hardware configuration
# Random Forest tasks use all available CPU cores (e.g. EPYC)
CPU_WORKERS = 64
# Deep learning tasks limit GPU concurrency (e.g. 8–12 workers on RTX 5090 to avoid memory thrashing)
GPU_WORKERS = 8

# MODEL_LIST = ['LSTM', 'GRU', 'CNN', 'RF']

MODEL_LIST = ['LSTM', 'GRU', 'CNN', 'RF']
# --- Global variables (shared via Linux fork-based copy-on-write semantics) ---
WORKER_ID = 0
GLOBAL_TRAIN_DF = None
GLOBAL_TEST_DF = None
GLOBAL_META_COLS = ['YEAR', 'COUNTY', 'INDEX', 'YIELD']


def process_dataframe_to_tensor_from_memory(df, feature_names):
    """
    Revised version: directly process pre-filtered DataFrames without additional filtering.
    """
    if df.empty:
        return None, None

    # Dynamically identify temporal columns
    time_cols = [c for c in df.columns if c not in GLOBAL_META_COLS]

    yield_df = df[['YEAR', 'COUNTY', 'YIELD']].drop_duplicates().sort_values(['YEAR', 'COUNTY'])
    y_target = yield_df['YIELD'].values
    target_index = pd.MultiIndex.from_arrays([yield_df['YEAR'], yield_df['COUNTY']])

    n_samples = len(yield_df)
    n_time = len(time_cols)
    n_feats = len(feature_names)

    X_tensor = np.zeros((n_samples, n_time, n_feats))

    # Use vectorized operations or lightweight loops to fill the tensor
    for f_idx, feat in enumerate(feature_names):
        try:
            # The dataframe has already been filtered by isin(feature_names)
            # but we still select the specific row index to preserve ordering
            subset = df[df['INDEX'] == feat].set_index(['YEAR', 'COUNTY'])
            subset = subset.reindex(target_index)
            ts_data = subset[time_cols]

            # Interpolate and repair missing values along the temporal axis
            ts_data = ts_data.interpolate(axis=1, limit_direction='both')
            ts_data = ts_data.ffill(axis=1).bfill(axis=1)
            ts_data = ts_data.fillna(0)

            X_tensor[:, :, f_idx] = ts_data.values
        except Exception:
            pass  # Ignore rare missing entries for this feature

    if np.isnan(X_tensor).any():
        return None, None

    return X_tensor, y_target


def train_and_eval_dl(X_train, y_train, X_test, y_test, model_name):
    """Deep learning training routine with learning rate scheduling."""
    torch.set_num_threads(1)

    N, T, F = X_train.shape
    scaler = StandardScaler()

    # Flatten & Scale
    X_train_flat = X_train.reshape(-1, F)
    X_test_flat = X_test.reshape(-1, F)

    with np.errstate(divide='ignore', invalid='ignore'):
        X_train_scaled = scaler.fit_transform(X_train_flat).reshape(N, T, F)
        X_test_scaled = scaler.transform(X_test_flat).reshape(X_test.shape[0], T, F)

    if np.isnan(X_train_scaled).any(): X_train_scaled = np.nan_to_num(X_train_scaled)
    if np.isnan(X_test_scaled).any(): X_test_scaled = np.nan_to_num(X_test_scaled)

    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32).to(DEVICE)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(DEVICE)
    X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32).to(DEVICE)

    # Shuffling the training set is critical for stable optimization
    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = TorchDataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    if model_name == 'LSTM':
        model = models.LSTMModel(input_dim=F).to(DEVICE)
    elif model_name == 'GRU':
        model = models.GRUModel(input_dim=F).to(DEVICE)
    elif model_name == 'CNN':
        model = models.CNN1DModel(input_dim=F, seq_len=T).to(DEVICE)

    criterion = nn.MSELoss()
    # Use weight decay to regularize the model and mitigate overfitting
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

    # --- Learning rate scheduler (CosineAnnealingLR) ---
    # Gradually decays the learning rate over epochs to improve convergence stability
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    model.train()
    for _ in range(EPOCHS):
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

        # Update the learning rate according to the scheduler
        scheduler.step()

    model.eval()
    with torch.no_grad():
        preds = model(X_test_t).cpu().numpy().flatten()

    return r2_score(y_test, preds)

def train_and_eval_rf(X_train, y_train, X_test, y_test):
    """Random Forest training."""
    N, T, F = X_train.shape
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, F)).reshape(N, T, F)
    X_test_scaled = scaler.transform(X_test.reshape(-1, F)).reshape(X_test.shape[0], T, F)

    if np.isnan(X_train_scaled).any(): X_train_scaled = np.nan_to_num(X_train_scaled)
    if np.isnan(X_test_scaled).any(): X_test_scaled = np.nan_to_num(X_test_scaled)

    # Change: remove the 'rf' string argument and explicitly specify n_estimators.
    # The previous implementation passed 'rf' where n_estimators was expected, causing an error.
    # models.SKLearnWrapper is assumed to wrap RandomForestRegressor or a similar estimator.
    model = models.SKLearnWrapper(n_estimators=100, n_jobs=1)
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    return r2_score(y_test, preds)

# --- Worker initialization ---
def worker_init(id_queue):
    global WORKER_ID
    # Retrieve a worker identifier from the queue
    if not id_queue.empty():
        WORKER_ID = id_queue.get()

    # Suppress stdout in workers to avoid interfering with the progress bar
    sys.stdout = open(os.devnull, 'w')


# --- Core worker function ---
def worker_task(args):
    # Note: this worker receives a precomputed indices_list instead of creating its own loader
    model_name, category, sza, vza, indices_list, lock, progress_queue = args

    try:
        # --- 1. In-memory filtering (no additional disk I/O) ---
        # Use the global dataframes shared via Linux fork semantics.
        # Filter rows so that only INDEX values in indices_list are retained.
        # copy() is used to avoid chained-assignment warnings and to isolate worker memory.
        df_train_sub = GLOBAL_TRAIN_DF[GLOBAL_TRAIN_DF['INDEX'].isin(indices_list)].copy()
        df_test_sub = GLOBAL_TEST_DF[GLOBAL_TEST_DF['INDEX'].isin(indices_list)].copy()

        if df_train_sub.empty or df_test_sub.empty:
            # Even if no data are available, still emit progress updates to keep counts consistent
            for _ in range(TRIALS): progress_queue.put(1)
            return None

        # --- 2. Convert to tensors ---
        X_train, y_train = process_dataframe_to_tensor_from_memory(df_train_sub, indices_list)
        X_test, y_test = process_dataframe_to_tensor_from_memory(df_test_sub, indices_list)

        if X_train is None:
            for _ in range(TRIALS): progress_queue.put(1)
            return None

        results_to_write = []

        # --- 3. Loop over repeated trials ---
        for trial in range(TRIALS):
            try:
                r2 = -999.0
                if model_name == 'RF':
                    r2 = train_and_eval_rf(X_train, y_train, X_test, y_test)
                else:
                    r2 = train_and_eval_dl(X_train, y_train, X_test, y_test, model_name)

                # Write each trial result to disk immediately instead of buffering in memory
                with lock:
                    with open(RESULT_CSV_PATH, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([model_name, category, sza, vza, trial, r2, None])

                # Send a progress signal for this trial
                progress_queue.put(1)

            except Exception as e:
                # Immediately log exceptions for failed trials
                with lock:
                    with open(RESULT_CSV_PATH, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([model_name, category, sza, vza, trial, None, str(e)])
                progress_queue.put(1)

        return "OK"

    except Exception as e:
        # In case of a fatal error, emit the remaining progress signals
        for _ in range(TRIALS): progress_queue.put(1)
        return "ERR"


def listener_thread(q, total_trials):
    """
    Dedicated progress-monitoring thread running in the main process.
    """
    pbar = tqdm(total=total_trials, desc="Total Progress (Trials)", unit="trial")
    for _ in range(total_trials):
        q.get()  # Block until a progress signal is received
        pbar.update(1)
    pbar.close()


def main():
    global GLOBAL_TRAIN_DF, GLOBAL_TEST_DF

    # --- 1. Pre-load data into memory in the main process ---
    print(f"Loading Data into Memory from {TRAIN_CSV_PATH} ...")
    start_t = time.time()

    # Instantiate a temporary loader instance to reuse its caching logic
    temp_loader_train = RemoteSensingDataLoader(TRAIN_CSV_PATH)
    GLOBAL_TRAIN_DF = temp_loader_train.df

    temp_loader_test = RemoteSensingDataLoader(TEST_CSV_PATH)
    GLOBAL_TEST_DF = temp_loader_test.df

    print(f"Data loaded in {time.time() - start_t:.1f}s. Train shape: {GLOBAL_TRAIN_DF.shape}")

    # --- 2. Prepare task list and precompute feature indices ---
    if not os.path.exists(RESULT_CSV_PATH):
        with open(RESULT_CSV_PATH, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Model', 'Category', 'SZA', 'VZA', 'Trial', 'R2', 'Error'])

    # Retrieve tasks that have already been completed
    existing_tasks = set()
    if os.path.exists(RESULT_CSV_PATH):
        try:
            df_exist = pd.read_csv(RESULT_CSV_PATH)
            if not df_exist.empty:
                counts = df_exist.groupby(['Model', 'Category', 'SZA', 'VZA']).size()
                finished = counts[counts >= TRIALS].index.tolist()
                for item in finished: existing_tasks.add(item)
        except:
            pass

    # Restrict experiments to the three specified CATEGORY types
    categories = [
        temp_loader_train.CATEGORY.STRUCT_MODIS,
        temp_loader_train.CATEGORY.PURE_SIF,
        temp_loader_train.CATEGORY.PURE_SIF_VIS
    ]

    angle_combinations = temp_loader_train.get_all_angle_combinations()

    manager = mp.Manager()
    lock = manager.Lock()
    progress_queue = manager.Queue()  # queue used for progress monitoring

    # Partition tasks into GPU-bound and CPU-bound groups
    gpu_tasks = []
    cpu_tasks = []

    print("Pre-calculating indices for tasks...")

    for model_name in MODEL_LIST:
        for category in categories:
            for sza, vza in angle_combinations:
                if (model_name, category, sza, vza) in existing_tasks: continue

                # Precompute indices here so that workers do not need to instantiate loaders
                indices = temp_loader_train.get_indices(category, sza, vza)

                task_args = (model_name, category, sza, vza, indices, lock, progress_queue)

                if model_name == 'RF':
                    cpu_tasks.append(task_args)
                else:
                    gpu_tasks.append(task_args)

    # Release temporary loader instances while keeping the global dataframes
    del temp_loader_train
    del temp_loader_test

    total_tasks = len(gpu_tasks) + len(cpu_tasks)
    total_trials = total_tasks * TRIALS

    print(f"\nTask Summary:")
    print(f"  GPU Tasks (DL): {len(gpu_tasks)}")
    print(f"  CPU Tasks (RF): {len(cpu_tasks)}")
    print(f"  Total Trials  : {total_trials}")
    print("=" * 40)
    time.sleep(2)

    # --- 3. Start the progress-monitoring thread ---
    # This thread runs until all expected progress signals have been received.
    watcher = threading.Thread(target=listener_thread, args=(progress_queue, total_trials))
    watcher.start()

    # --- 4. Phase 1: execute GPU tasks with limited concurrency ---
    # Prepare worker identifiers for GPU processes (1–GPU_WORKERS)
    id_queue_gpu = manager.Queue()
    for i in range(1, GPU_WORKERS + 1): id_queue_gpu.put(i)

    if gpu_tasks:
        print(f"\n[Phase 1] Running GPU tasks with {GPU_WORKERS} workers...")
        with mp.Pool(processes=GPU_WORKERS, initializer=worker_init, initargs=(id_queue_gpu,)) as pool:
            # Use map with an external watcher; no per-process progress bar is required here
            pool.map(worker_task, gpu_tasks)

    # --- 5. Phase 2: execute CPU tasks with maximum concurrency ---
    # Prepare worker identifiers for CPU processes (1–CPU_WORKERS)
    id_queue_cpu = manager.Queue()
    for i in range(1, CPU_WORKERS + 1): id_queue_cpu.put(i)

    if cpu_tasks:
        print(f"\n[Phase 2] Running CPU tasks with {CPU_WORKERS} workers...")
        with mp.Pool(processes=CPU_WORKERS, initializer=worker_init, initargs=(id_queue_cpu,)) as pool:
            pool.map(worker_task, cpu_tasks)

    print("\nWaiting for progress bar to finish...")
    watcher.join()
    print("All Done.")

if __name__ == "__main__":
    # Configure multiprocessing start method (WSL typically supports 'fork')
    # If 'fork' fails, consider switching to 'spawn', noting that shared memory will not be available.
    mp.set_start_method('fork', force=True)
    main()

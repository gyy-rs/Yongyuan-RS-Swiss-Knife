import os
import json
import requests
import hashlib
import concurrent.futures
from datetime import datetime
from tqdm import tqdm

# --- Configuration ---

# 1. Directory where metadata JSON files are stored
JSON_DIR = "/nas/@Data.TROPOMI.SIF/@ESA_TROPOMI_SIF_L2B/json"

# 2. Directory where downloaded data files will be saved (created automatically if missing)
DOWNLOAD_DIR = "/nas/@Data.TROPOMI.SIF/@ESA_TROPOMI_SIF_L2B/daily_collection"

# 3. Start date for filtering (only download data on or after this date)
START_DATE = datetime(2024, 8, 1)

# 4. Number of concurrent download threads
MAX_WORKERS = 8

# --- Optional proxy configuration ---
proxies = {
    'http': 'http://127.0.0.1:7897',
    'https': 'http://127.0.0.1:7897',
}
# --- End of proxy configuration ---

# --- Core functions ---

def get_files_to_download(json_dir, start_date):
    """
    Scan the JSON directory, parse each file to extract the acquisition date,
    and build a list of metadata records to download.
    """
    tasks = []
    print(f"Scanning metadata directory: {json_dir}")
    print(f"Filtering JSON contents with date >= {start_date.strftime('%Y-%m-%d')} ...")

    if not os.path.exists(json_dir):
        print(f"ERROR: Metadata directory does not exist: {json_dir}")
        return []

    for filename in os.listdir(json_dir):
        if not filename.endswith(".json"):
            continue

        json_path = os.path.join(json_dir, filename)
        try:
            with open(json_path, 'r') as f:
                meta = json.load(f)
                
                start_datetime_str = meta.get('properties', {}).get('start_datetime')
                if not start_datetime_str:
                    continue
                
                date_part = start_datetime_str.split('T')[0]
                file_date = datetime.strptime(date_part, '%Y-%m-%d')

                if file_date >= start_date:
                    asset = meta.get('assets', {}).get('product', {})
                    if asset and asset.get('href') and asset.get('file:local_path'):
                        tasks.append({
                            'url': asset.get('href'),
                            'local_path': asset.get('file:local_path'),
                            'expected_size': asset.get('file:size'),
                            'expected_checksum': asset.get('file:checksum'),
                            'json_name': filename
                        })
        except Exception as e:
            print(f"WARNING: Failed to process file '{filename}', skipping. Error: {e}")

    print(f"Scan finished. Found {len(tasks)} candidate files to download.")
    return tasks

def calculate_hash(filepath, algorithm, block_size=8192):
    """Generic hash calculation helper."""
    hasher = hashlib.new(algorithm)
    try:
        with open(filepath, "rb") as f:
            while True:
                data = f.read(block_size)
                if not data:
                    break
                hasher.update(data)
    except IOError:
        return None
    return hasher.hexdigest()

def verify_checksums(filepath, expected_checksum):
    """
    Compute multiple hash values and compare them against the expected checksum.
    For this specific data source, MD5 is compared using an "endswith" rule.
    """
    if not expected_checksum:
        return False, {"error": "Expected checksum is empty."}

    # Core logic:
    # 1. Prioritize MD5 because its pattern in the checksum string is known.
    actual_md5 = calculate_hash(filepath, 'md5')
    # 2. Check whether the expected checksum string ends with the computed MD5
    if actual_md5 and expected_checksum.endswith(actual_md5):
        return True, None

    # If MD5 does not match, fall back to other standard hashes
    algorithms_to_check = ['sha1', 'sha256']
    calculated_hashes = {'md5': actual_md5}

    for alg in algorithms_to_check:
        actual_hash = calculate_hash(filepath, alg)
        calculated_hashes[alg] = actual_hash
        if actual_hash == expected_checksum:
            return True, None

    # If none of the algorithms match, return all calculated hashes for debugging
    return False, calculated_hashes

def download_file(task):
    """
    Core worker function for downloading a single file, with resume support
    and multi-hash checksum verification.
    """
    url = task['url']
    local_filename = task['local_path']
    dest_path = os.path.join(DOWNLOAD_DIR, local_filename)
    expected_size = task['expected_size']
    expected_checksum = task['expected_checksum']

    if not all([url, local_filename, expected_size, expected_checksum]):
        return f"SKIP: Incomplete task information for {task.get('json_name')}"

    # Core logic: handle existing files in a clear and robust way
    headers = {}
    current_size = 0
    
    if os.path.exists(dest_path):
        current_size = os.path.getsize(dest_path)

        # 1. Compare file size
        if current_size == expected_size:
            # 2. If sizes match, verify checksum
            is_valid, _ = verify_checksums(dest_path, expected_checksum)
            if is_valid:
                return f"SKIP: {local_filename} already exists and passed checksum verification."
            else:
                # If checksum fails, remove and re-download
                print(f"WARNING: Checksum mismatch for '{local_filename}', re-downloading.")
                os.remove(dest_path)
                current_size = 0
        
        elif current_size < expected_size:
            # Local file is smaller than expected; prepare for resume
            print(f"File '{local_filename}' is incomplete. Resuming from byte {current_size}.")
            headers['Range'] = f'bytes={current_size}-'
        
        else: # current_size > expected_size
            # Local file is larger than expected; remove and re-download
            print(f"WARNING: File '{local_filename}' size ({current_size}) exceeds expected ({expected_size}), re-downloading.")
            os.remove(dest_path)
            current_size = 0

    try:
        # Execute the download (fresh or resumed)
        response = requests.get(url, headers=headers, stream=True, allow_redirects=True, timeout=60, proxies=proxies)
        response.raise_for_status()

        total_size = expected_size
        mode = 'ab' if current_size > 0 else 'wb'

        with open(dest_path, mode) as f, tqdm(
            desc=local_filename,
            initial=current_size,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
            leave=False
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        # Final size and checksum validation after download
        if pbar.n != total_size:
            return f"FAILED: Size mismatch after download: {local_filename}"
        
        is_valid, calculated_hashes = verify_checksums(dest_path, expected_checksum)
        if not is_valid:
            error_msg = f"FAILED: All checksum verifications failed for {local_filename}\n"
            error_msg += f"  - Expected: {expected_checksum}\n"
            for alg, hash_val in calculated_hashes.items():
                error_msg += f"  - Computed {alg.upper()}: {hash_val}\n"
            return error_msg

        return f"SUCCESS: Downloaded {local_filename}"

    except requests.exceptions.RequestException as e:
        return f"FAILED (network error): {local_filename}, {e}"
    except Exception as e:
        return f"FAILED (unexpected error): {local_filename}, {e}"

# --- Main entry point ---

if __name__ == "__main__":
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    download_tasks = get_files_to_download(JSON_DIR, START_DATE)

    if not download_tasks:
        print("No files need to be downloaded.")
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_task = {executor.submit(download_file, task): task for task in download_tasks}
            
            results_iterator = tqdm(concurrent.futures.as_completed(future_to_task), total=len(download_tasks), desc="Overall progress")

            for future in results_iterator:
                task_info = future_to_task[future]
                try:
                    result = future.result()
                    results_iterator.write(result)
                except Exception as exc:
                    results_iterator.write(f"FAILED (exception): Task '{task_info['local_path']}' raised: {exc}")

        print("\nAll download tasks have finished.")

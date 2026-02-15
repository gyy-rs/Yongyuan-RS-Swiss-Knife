#!/usr/bin/env python3
import os
import requests
import concurrent.futures
from tqdm import tqdm
from urllib.parse import urlparse
import logging

# --- Configuration ---

# 1. The full path to the file containing the list of download URLs.
URL_LIST_FILE = "/pg_disk/@open_data/9-ESA_TROPOMI_SIF/py/@MODIS_download_and_processing/MODIS_file_list.list"

# 2. The directory where the downloaded files will be saved.
DOWNLOAD_DIR = "/nas/@Data.MODIS/MCD43C1"

# 3. The number of concurrent download threads.
MAX_WORKERS = 8

# 4. The hostname for Earthdata Login authentication.
EARTHDATA_HOST = "urs.earthdata.nasa.gov"

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def check_prerequisites():
    """Checks if the required .netrc file exists."""
    netrc_path = os.path.expanduser("~/.netrc")
    if not os.path.exists(netrc_path):
        logging.error(f"Authentication file not found at '{netrc_path}'.")
        logging.error("Please create it with the following content:")
        logging.error(f"machine {EARTHDATA_HOST} login YOUR_USERNAME password YOUR_PASSWORD")
        logging.error("Then, set its permissions with: chmod 600 ~/.netrc")
        return False
    return True

def check_app_approval(session, test_url):
    """
    Makes a preliminary request to verify credentials and application approval.
    """
    try:
        logging.info("Verifying application approval with Earthdata Login (using .netrc)...")
        # A HEAD request is efficient as it doesn't download the file body
        response = session.head(test_url, allow_redirects=True, timeout=30)
        response.raise_for_status()
        logging.info("Successfully authenticated and application is approved. ✅")
        return True
    except requests.exceptions.RequestException as e:
        logging.error("\n" + "="*60)
        logging.error("ERROR: Could not verify Earthdata Login credentials or app approval.")
        logging.error(f"DETAILS: {e}")
        logging.error("\nPlease ensure:")
        logging.error("  1. Your ~/.netrc file contains the correct username and password.")
        logging.error(f"  2. You have authorized the 'NASA GESDISC DATA ARCHIVE' application in your Earthdata profile:")
        logging.error(f"     https://{EARTHDATA_HOST}/profile")
        logging.error("="*60 + "\n")
        return False

def download_worker(url, session, dest_folder):
    """
    The core function executed by each thread to download a single file.
    """
    try:
        filename = os.path.basename(urlparse(url).path)
        dest_path = os.path.join(dest_folder, filename)

        if os.path.exists(dest_path):
            return f"SKIPPED: {filename} already exists."

        response = session.get(url, stream=True, allow_redirects=True, timeout=120)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        with open(dest_path, 'wb') as f, tqdm(
            desc=filename, total=total_size, unit='B', unit_scale=True, unit_divisor=1024, leave=False
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        return f"SUCCESS: Downloaded {filename}"
    except requests.exceptions.HTTPError as e:
        return f"FAILED (HTTP Error: {e.response.status_code}): {os.path.basename(url)}"
    except Exception as e:
        return f"FAILED (Error: {e}): {os.path.basename(url)}"

def main():
    """Main execution function."""
    if not check_prerequisites():
        return

    try:
        if not os.path.exists(URL_LIST_FILE):
            logging.error(f"URL list file not found at '{URL_LIST_FILE}'")
            return
        
        with open(URL_LIST_FILE, 'r') as f:
            urls_to_download = [line.strip() for line in f if line.strip()]
        
        if not urls_to_download:
            logging.warning("URL list is empty. Nothing to download.")
            return

        # Create a session that will automatically use the ~/.netrc file
        session = requests.Session()
        
        if not check_app_approval(session, urls_to_download[0]):
            return

        os.makedirs(DOWNLOAD_DIR, exist_ok=True)
        logging.info(f"Files will be downloaded to: {DOWNLOAD_DIR}")
        logging.info(f"Found {len(urls_to_download)} URLs to process with up to {MAX_WORKERS} threads.\n")

        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_url = {executor.submit(download_worker, url, session, DOWNLOAD_DIR): url for url in urls_to_download}
            
            # Use a dedicated tqdm instance for overall progress
            results_pbar = tqdm(concurrent.futures.as_completed(future_to_url), total=len(urls_to_download), desc="Overall Progress")

            for future in results_pbar:
                try:
                    result = future.result()
                    # Use tqdm.write to print messages without breaking the progress bars
                    results_pbar.write(result)
                except Exception as exc:
                    url = future_to_url[future]
                    filename = os.path.basename(urlparse(url).path)
                    results_pbar.write(f"FAILED (Exception): {filename} generated an error: {exc}")

    finally:
        logging.info("\nDownload process finished.")

if __name__ == "__main__":
    main()

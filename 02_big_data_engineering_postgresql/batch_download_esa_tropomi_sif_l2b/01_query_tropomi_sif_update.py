import requests
import os
import time
import json


# Base URL of the TROPOMI SIF API
BASE_URL = "https://data-portal.s5p-pal.com/api/s5p-l2/collections/L2B_SIF___/items"

# Target directory where JSON metadata files will be saved
SAVE_DIR = "/nas/@Data.TROPOMI.SIF/@ESA_TROPOMI_SIF_L2B/json"

# Number of items to request per API call
LIMIT = 10

# Starting offset
# Note: the script automatically increases this offset by LIMIT in each loop
START_OFFSET = 2200

# --- Script ---

def fetch_and_save_sif_metadata():
    """
    Repeatedly query the TROPOMI SIF API, parse the results, and
    download detailed JSON metadata for each product.
    """
    print("Starting TROPOMI SIF metadata crawler...")
    print(f"Metadata will be saved to: {SAVE_DIR}")

    # 1. Ensure the target directory exists
    try:
        os.makedirs(SAVE_DIR, exist_ok=True)
        print(f"Directory '{SAVE_DIR}' is ready.")
    except OSError as e:
        print(f"ERROR: Failed to create directory '{SAVE_DIR}': {e}")
        return  # Abort if the directory cannot be created

    offset = START_OFFSET
    session = requests.Session()  # Use a session to improve efficiency
    total_new_files = 0

    while True:
        # Build the current request URL
        request_url = f"{BASE_URL}?limit={LIMIT}&offset={offset}"
        print(f"\nRequesting: {request_url}")

        try:
            # Send GET request
            response = session.get(request_url, timeout=30)
            response.raise_for_status()  # Raise for HTTP errors (e.g. 404, 500)
            data = response.json()

        except requests.exceptions.RequestException as e:
            print(f"ERROR: Request failed: {e}")
            print("Retrying in 10 seconds...")
            time.sleep(10)
            continue  # Retry the same offset

        except json.JSONDecodeError:
            print("ERROR: Failed to parse JSON response.")
            print("The API may be unstable; skipping the current offset.")
            offset += LIMIT
            continue

        # Check whether the 'features' list is empty
        features = data.get("features", [])
        if not features:
            print("API returned an empty 'features' list. Task is complete.")
            break

        print(f"Successfully retrieved {len(features)} records. Processing...")

        # 2. Iterate over each record
        for feature in features:
            try:
                # 3. Extract the 'href' field from the second element in 'links'
                metadata_url = feature['links'][1]['href']
                
                # Extract the filename from the URL
                filename = os.path.basename(metadata_url)
                file_path = os.path.join(SAVE_DIR, filename)

                # Skip if the file already exists
                if os.path.exists(file_path):
                    continue

                # Download detailed metadata JSON
                print(f"  Downloading: {filename}")
                metadata_response = session.get(metadata_url, timeout=30)
                metadata_response.raise_for_status()
                
                # 4. Save to local JSON file
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata_response.json(), f, indent=4)
                
                total_new_files += 1
                # time.sleep(0.1)  # Optional: short delay to avoid overloading the API

            except (IndexError, KeyError) as e:
                print(f"  WARNING: Failed to parse 'feature'; missing 'links' or unexpected structure: {e}")
                continue
            except requests.exceptions.RequestException as e:
                print(f"  ERROR: Failed to download metadata {metadata_url}: {e}")
                continue

        # Update offset for the next loop
        offset += LIMIT

    print("\n--- Task summary ---")
    print("Crawler finished.")
    print(f"Downloaded {total_new_files} new metadata files in this run.")


if __name__ == "__main__":
    fetch_and_save_sif_metadata()

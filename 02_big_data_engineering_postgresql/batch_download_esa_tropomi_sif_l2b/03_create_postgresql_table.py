import os
import pandas as pd
from glob import glob
import netCDF4 as nc
from tqdm import tqdm

# Set the directory containing NetCDF files and the output CSV path
data_directory = r"/nas/@Data.TROPOMI.SIF/@ESA_TROPOMI_SIF_L2B/daily_collection"
output_csv_path = r"/nas/@Data.TROPOMI.SIF/@ESA_TROPOMI_SIF_L2B/META/TROPOMI_SIF_META.csv"

# Initialize a list to collect metadata records
data = []

# Iterate over all matching NetCDF files
for file_path in tqdm(glob(os.path.join(data_directory, "*.nc"))):
    local_name = os.path.basename(file_path)

    # Open the NetCDF file
    dataset = nc.Dataset(file_path)

    # Access the 'delta_time' variable and its units
    delta_time_var = dataset['PRODUCT/delta_time']
    time_units = delta_time_var.units
    base_time_str = time_units.split('since ')[-1].strip()
    base_time = pd.to_datetime(base_time_str)

    # Generate table_name by removing the trailing component and the "T" character
    table_name = "_".join(local_name.split("_")[:-1]).replace("T", "")

    # File size (bytes) on disk
    file_length = os.path.getsize(file_path)

    # Original filename
    filename = local_name

    # Append a record to the metadata list
    data.append({
        'filename': filename,
        'table_name': "esa_tropomi_sif_traj_"+table_name,
        'file_length': len(dataset['PRODUCT/delta_time']),
        'time_baseline': base_time_str
    })

    # Close the dataset to free resources
    dataset.close()

# Convert the list of records into a DataFrame
df = pd.DataFrame(data)

# Save metadata as CSV
df.to_csv(output_csv_path, index=False)

# Generate a PostgreSQL COPY command for bulk loading
table_name = "TROPOMI_SIF_META"
copy_command = f"COPY {table_name} (filename, table_name, file_length, time_baseline) FROM '{output_csv_path}' DELIMITER ',' CSV HEADER;"

print("CSV file created at:", output_csv_path)
print("COPY command:")
print(copy_command)

import pandas as pd

# Load the generated CSV file
csv_path = output_csv_path
df = pd.read_csv(csv_path)

# Identify rows that share the same file_length (potential duplicates)
duplicate_file_length = df[df.duplicated(['file_length'], keep=False)]

# Print potential duplicates for inspection
print(duplicate_file_length)

"""Local preprocessing script - 5% stratified sampling per attack class.

Reads CSVs from datasets folder, samples 5% per Label group, writes to
datasets/threshold_<THRESHOLD>/reduced_dataset.csv

IMPORTANT: 
- To change the input CSV location, modify LOCAL_DATASETS_DIR below.
- To change the output threshold, modify THRESHOLD below.
- A folder named "datasets" will be created at repo root level.
- The script generates threshold-specific reduced datasets inside datasets/.
"""

import glob
import os
import pandas as pd

# --- CONFIGURATION ---
# Set these paths to your local environment:
#   The script walks up from this file's location to find the repo root,
#   then looks for datasets/ under it.
#   Or set LOCAL_DATASETS_DIR and THRESHOLD explicitly below.
# 
# IMPORTANT: 
# - The script creates datasets/ folder at repo root if missing.
# - It generates threshold-specific reduced datasets: datasets/threshold_<THRESHOLD>/reduced_dataset.csv
# - To change the input CSV location, modify LOCAL_DATASETS_DIR below.
# - To change the threshold value, modify THRESHOLD below (must match COMMON/config.py).
# 
# Option 1: Auto-detect from repo root (default walks up 2 levels from this file)
_current_dir = os.path.dirname(os.path.abspath(__file__))
for _ in range(5):
    if os.path.isdir(os.path.join(_current_dir, 'COMMON')):
        _project_root = _current_dir
        break
    parent = os.path.dirname(_current_dir)
    if parent == _current_dir:
        break
    _current_dir = parent

LOCAL_DATASETS_DIR = os.path.join(_project_root, "datasets") if '_project_root' in dir() else "datasets"
THRESHOLD = "0.10"  # Must match COMMON/config.py THRESHOLD

# Ensure datasets directory exists
os.makedirs(LOCAL_DATASETS_DIR, exist_ok=True)

# Create threshold-specific folder
THRESHOLD_FOLDER = os.path.join(LOCAL_DATASETS_DIR, f"threshold_{THRESHOLD}")
os.makedirs(THRESHOLD_FOLDER, exist_ok=True)

LOCAL_OUTPUT_FILE = os.path.join(THRESHOLD_FOLDER, "reduced_dataset.csv")

# Option 2: Override with explicit paths (uncomment and modify):
# LOCAL_DATASETS_DIR = r"C:\path\to\your\CSVs\..\datasets"
# THRESHOLD = "0.10"
# --------------------------------

print(f"Project root: {_project_root}")
print(f"Dataset dir:  {LOCAL_DATASETS_DIR}")

print(f"Output file:  {LOCAL_OUTPUT_FILE}")

# Get all CSV files from the dataset directory
csv_pattern = os.path.join(LOCAL_DATASETS_DIR, "*.csv")
files = sorted(glob.glob(csv_pattern))

if not files:
    print("\nNo CSV files found at: " + csv_pattern)
    print("   Place your MERGED_CSV *.csv files in the DATASETS folder, or set LOCAL_DATASETS_DIR explicitly.")
    raise SystemExit(1)

print(f"Total files found: {len(files)}")

first_file = True

for file in files:
    print(f"\nProcessing: {os.path.basename(file)}")
    # Read one CSV
    df = pd.read_csv(file)

    # Take 5% sample from every attack class (stratified sampling)
    sampled_df = (
        df.groupby("Label", group_keys=False)
          .apply(lambda x: x.sample(frac=0.05, random_state=42))
          .reset_index(drop=True)
    )

    print(f"  Original shape: {df.shape}")
    print(f"  Sampled shape:  {sampled_df.shape}")

    # Write first file with header
    if first_file:
        sampled_df.to_csv(LOCAL_OUTPUT_FILE, index=False)
        first_file = False
    else:
        # Append remaining files without header
        sampled_df.to_csv(
            LOCAL_OUTPUT_FILE,
            mode='a',
            header=False,
            index=False
        )

    # Free RAM
    del df

print("\nDone! Sampled dataset written to: " + LOCAL_OUTPUT_FILE)
print(f"   Rows: {sum(1 for _ in open(LOCAL_OUTPUT_FILE)) - 1}  (excluding header)")
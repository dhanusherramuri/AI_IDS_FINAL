"""Local preprocessing script - 5% stratified sampling per attack class.

Reads CSVs from DATASETS folder, samples 5% per Label group, writes to
dataset/<user-specified-filename>.csv

IMPORTANT: 
- To change the input CSV location, modify LOCAL_DATASETS_DIR below.
- To change the output filename, modify LOCAL_OUTPUT_FILE below.
- Ensure a folder named "dataset" exists in the same folder as the parent folder
  (i.e., <project_root>/dataset/), or the script will create it automatically.
"""

import glob
import os
import pandas as pd

# --- CONFIGURATION ---
# Set these paths to your local environment:
#   The script walks up from this file's location to find the repo root,
#   then looks for datasets/ under it.
#   Or set LOCAL_DATASETS_DIR and LOCAL_OUTPUT_FILE explicitly below.
# 
# IMPORTANT: 
# - A folder named "dataset" must exist in the project root (parent folder),
#   e.g., <project_root>/dataset/. The script will create it if missing.
# - To change the input CSV location, modify LOCAL_DATASETS_DIR below.
# - To change the output filename (without .csv extension), modify 
#   LOCAL_OUTPUT_FILE stem below.
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

LOCAL_DATASETS_DIR = os.path.join(_project_root, "dataset") if '_project_root' in dir() else "dataset"
LOCAL_OUTPUT_FILE_STEM = "MERGED_CSV_Sampled_5Percent"
LOCAL_OUTPUT_FILE = os.path.join(LOCAL_DATASETS_DIR, LOCAL_OUTPUT_FILE_STEM + ".csv")

# Option 2: Override with explicit paths (uncomment and modify):
# LOCAL_DATASETS_DIR = r"C:\path\to\your\CSVs\..\dataset"
# LOCAL_OUTPUT_FILE_STEM = r"your_output_filename"
# LOCAL_OUTPUT_FILE = r"C:\path\to\output\your_output_filename.csv"
# --------------------------------

print(f"Project root: {_project_root}")
print(f"Dataset dir:  {LOCAL_DATASETS_DIR}")

# Ensure output directory exists
os.makedirs(LOCAL_DATASETS_DIR, exist_ok=True)

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
import os
import sys
import pandas as pd

# Add project root to path so COMMON module can be found
_current_dir = os.path.dirname(os.path.abspath(__file__))
for _ in range(5):
    if os.path.isdir(os.path.join(_current_dir, 'COMMON')):
        sys.path.insert(0, _current_dir)
        break
    parent = os.path.dirname(_current_dir)
    if parent == _current_dir:
        break
    _current_dir = parent

from COMMON.config import PROJECT_ROOT

df = pd.read_csv(os.path.join(PROJECT_ROOT, "DATASETS", "MERGED_CSV_Sampled_5Percent.csv"))

print(df.shape)

print(df["Label"].value_counts())

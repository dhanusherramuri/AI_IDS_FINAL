# AGENTS.md - AI_IDS_MP

## Project Structure
- `COMMON/` - shared utilities: config, dataset loading, metrics, model I/O, preprocessing
- `DL/` - Deep Learning: `train_cnn.py`, `train_ann.py`, `train_cnn_multiclass.py`
- `ML/` - Machine Learning: `train_random_forest.py`, `train_rf_multiclass.py`, `train_decision_tree.py`, `train_dt_multiclass.py`, `train_xgboost_multiclass.py`
- `scripts/` - top-level training scripts
- `AI_IDS.ipynb` - Colab notebook (update paths for local runs)
- `preprocess_local.py` - standalone local preprocessing script

## Virtual Environment
```bash
python -m venv .venv
\.venv\Scripts\activate   # Windows
source .venv/bin/activate   # Linux/macOS
```

## Install Packages
```bash
pip install -r requirements.txt  # if present
# Key: tensorflow, xgboost, scikit-learn, pandas, numpy, joblib, matplotlib, imblearn
```

## Running Training Scripts
All scripts require `AI_IDS_PROJECT_ROOT` pointing to repo root. Set the env var to your path:

```bash
AI_IDS_PROJECT_ROOT=/your/path python scripts/train_pcc_cnn.py       # PCC-CNN (TF Keras, binary by default)
AI_IDS_PROJECT_ROOT=/your/path python ML/scripts/train_random_forest.py  # Random Forest (binary)
AI_IDS_PROJECT_ROOT=/your/path python ML/scripts/train_xgboost_multiclass.py  # XGBoost (multiclass)
AI_IDS_PROJECT_ROOT=/your/path python DL/scripts/train_cnn.py  # CNN model
```

**Important**: First run sets `sys.path` to find `COMMON` and `utils` modules. Do NOT move scripts between directories.

## Key Conventions
- **10-fold cross-validation** is the standard (`N_SPLITS = 10`)
- **Binary vs multiclass** is controlled by `CLASSIFICATION` in `COMMON/config.py` ("binary" or "multiclass")
  - For binary: set `CLASSIFICATION = "binary"` and `THRESHOLD = "0.10"` (or `"0.9"` for stricter)
  - For multiclass: set `CLASSIFICATION = "multiclass"` (e.g., XGBoost uses this default)
- **Threshold** controls binary decision boundary (`THRESHOLD = "0.10"` or `"0.9"`)
- Models saved as `.keras` (TF), `.joblib` (sklearn), `.json` (XGBoost)
- Results saved as CSVs and PNG plots under `RESULTS/` per model type

## Path Configuration
- `COMMON/config.py:11` sets `PROJECT_ROOT` from `AI_IDS_PROJECT_ROOT` environment variable
- Default walks up 2 levels from `COMMON/` to reach repo root
- Set `AI_IDS_PROJECT_ROOT=/your/path python script.py` to override
- **Data under `PROJECT_ROOT/datasets/`** - place merged CSVs here for preprocessing

## Data Preprocessing
Two workflows available — Colab (notebook) or local script.

### Local (preprocess_local.py)
Run from repo root:
```bash
python preprocess_local.py
```
- Auto-detects repo root by walking up looking for `COMMON/`
- Reads all `*.csv` files from `datasets/` folder
- Takes **5% stratified sample** per `Label` group (`random_state=42`)
- Writes single combined `MERGED_CSV_Sampled_5Percent.csv`
- **Output saved to `datasets/` folder** - specify desired filename at top of script (see `LOCAL_OUTPUT_FILE_STEM`)

### Colab (AI_IDS.ipynb)
Update `dataset_path` and `output_file` variables in first cell to local paths.

## Important Quirks (Fixed)
- **Set AI_IDS_PROJECT_ROOT env var** to use custom project root (default walks up 2 levels from `COMMON/`)
- **XGBoost uses SMOTE** for class imbalance (sampling strategy defined per-fold based on training fold class counts)
- **Feature variance thresholding** (`VAR_THRESHOLD = 0.001`) applied before XGBoost training
- **TensorFlow seed** set globally; DL scripts set `tf.random.set_seed(SEED)` at start
- **`utils.plots` import**: 4 ML scripts (`train_random_forest.py`, `train_rf_multiclass.py`, `train_decision_tree.py`, `train_dt_multiclass.py`) include `utils` path-walking so `from utils.plots import ...` resolves correctly without manual `sys.path` edits
- `preprocess_local.py` auto-detects repo root; if CSVs not found in `datasets/`, set `LOCAL_DATASETS_DIR` explicitly

## Using Plots Files (`utils.plots`)
The `utils.plots` module provides helper functions to save visualization outputs. These are used by the ML training scripts to generate:

- **Confusion matrices** (`save_confusion_matrix`)
- **ROC curves** (`save_roc_curve`)  
- **Fold accuracy plots** (`save_fold_accuracy`)
- **Metrics bar charts** (`save_metrics_bar`)
- **Feature importance plots** (`save_feature_importance`)

### What to change if adding plots to a new script:
1. Ensure `utils` directory exists at the repo root (parent of `ML/` and `DL/`)
2. Add path-walking code (same pattern as other ML scripts) to find `utils`:
   ```python
   _current_dir = os.path.dirname(os.path.abspath(__file__))
   for _ in range(5):
       if os.path.isdir(os.path.join(_current_dir, 'utils')):
           sys.path.insert(0, _current_dir)
           break
       parent = os.path.dirname(_current_dir)
       if parent == _current_dir:
           break
       _current_dir = parent
   ```
3. Import the desired plot functions:
   ```python
   from utils.plots import save_confusion_matrix, save_roc_curve, ...
   ```
4. Call the functions with appropriate arguments (results dir, fold number, data)

### Plot output locations:
- PNG plots are saved to `<PROJECT_ROOT>/<MODEL_TYPE>/RESULTS/<model_type>/threshold_<THRESHOLD>/`
- CSV data accompaniment saved alongside plots

## What to Avoid
- Do NOT remove or rename the `utils` or `COMMON` directory from expected relative locations
- Do NOT run training scripts without `AI_IDS_PROJECT_ROOT` set (or ensure you're in the repo root)
- Do NOT modify `COMMON/config.py:_PROJECT_ROOT_LEVELS` unless you understand the directory structure (currently 2 = repo root)
- Do NOT run `preprocess_local.py` from a directory other than repo root or subdirectory — it walks up to find `COMMON/`
- Do NOT change `CLASSIFICATION` without also updating `THRESHOLD` appropriate for binary vs multiclass workflow
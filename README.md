# AI_IDS_MP - AI-Based Intrusion Detection System

## Overview
This repository contains Machine Learning and Deep Learning models for network intrusion detection. Supports both binary and multiclass classification scenarios.

## Project Structure
- `COMMON/` - shared utilities: config, dataset loading, metrics, model I/O, preprocessing
- `DL/` - Deep Learning training scripts (CNN, ANN)
- `ML/` - Machine Learning training scripts (Random Forest, XGBoost, Decision Tree)
- `scripts/` - top-level training scripts
- `AI_IDS.ipynb` - Colab-based notebook
- `preprocess_local.py` - standalone local preprocessing script

## Dataset
Place your merged CSV files in the `datasets/` folder (within the parent folder, i.e., `<project_root>/datasets/`).

The preprocessing script performs 5% stratified sampling per attack class and writes output to:
`<project_root>/dataset/MERGED_CSV_Sampled_5Percent.csv`

**To customize:** Edit `LOCAL_DATASETS_DIR` and `LOCAL_OUTPUT_FILE_STEM` at the top of `preprocess_local.py`.

## Virtual Environment
```bash
python -m venv .venv
\.venv\Scripts\activate   # Windows
source .venv/bin/activate   # Linux/macOS
```

## Installation
```bash
pip install -r requirements.txt
```
Key packages: tensorflow, xgboost, scikit-learn, pandas, numpy, joblib, matplotlib, imblearn

## Running Training Scripts
All scripts require `AI_IDS_PROJECT_ROOT` pointing to repo root:
```bash
AI_IDS_PROJECT_ROOT=/your/path python scripts/train_pcc_cnn.py
AI_IDS_PROJECT_ROOT=/your/path python ML/scripts/train_random_forest.py
AI_IDS_PROJECT_ROOT=/your/path python ML/scripts/train_xgboost_multiclass.py
AI_IDS_PROJECT_ROOT=/your/path python DL/scripts/train_cnn.py
```

**Important:** First run sets `sys.path` to find `COMMON` and `utils` modules. Do NOT move scripts between directories.

## Key Conventions
- **10-fold cross-validation** standard (`N_SPLITS = 10`)
- **Binary vs multiclass** controlled by `CLASSIFICATION` in `COMMON/config.py` ("binary" or "multiclass")
- **Threshold** controls binary decision boundary (`THRESHOLD = "0.10"` or `"0.9"`)
- Models saved as `.keras` (TF), `.joblib` (sklearn), `.json` (XGBoost)
- Results saved as CSVs and PNG plots under `RESULTS/` per model type

## Important Quirks
- Set `AI_IDS_PROJECT_ROOT` env var for custom project root
- XGBoost uses SMOTE for class imbalance
- Feature variance thresholding (`VAR_THRESHOLD = 0.001`) applied before XGBoost training
- 4 ML scripts include `utils` path-walking so `from utils.plots import ...` resolves correctly

## Plots
The `utils.plots` module provides helpers for confusion matrices, ROC curves, fold accuracy, metrics bars, and feature importance plots.

See `AGENTS.md` for detailed guidance on adding plots to new scripts.

## Output Directories
- Models: `<PROJECT_ROOT>/<MODEL_TYPE>/MODELS/<model_type>/threshold_<THRESHOLD>/`
- Results: `<PROJECT_ROOT>/<MODEL_TYPE>/RESULTS/<model_type>/threshold_<THRESHOLD>/`

## License
(Add your license here)
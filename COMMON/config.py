"""
Project Configuration

Controls training parameters, file paths, and classification mode.
Edit values here or override via AI_IDS_PROJECT_ROOT environment variable.
"""

import os

# ==========================================================
# RANDOM SEED
# ==========================================================

SEED = 42  # Global random seed for reproducibility; set in all training scripts

# ==========================================================
# TRAINING PARAMETERS
# ==========================================================

N_SPLITS = 10  # 10-fold cross-validation standard

BATCH_SIZE = 256  # Default batch size for CNN training

EPOCHS = 50  # Default epochs for model training

LEARNING_RATE = 0.001  # Default learning rate for optimizer

CLASSIFICATION = "binary"  # Set to "multiclass" for multi-class classification

# ==========================================================
# FEATURE SELECTION
# ==========================================================

THRESHOLD = "0.10"  # Binary decision threshold; use "0.9" for stricter filtering

# ==========================================================
# PROJECT ROOT
# ==========================================================

import os

_PROJECT_ROOT_LEVELS = 2  # Walk up 2 levels from COMMON/ to reach repo root (AI_IDS_MP/)
_PROJECT_ROOT_BASE = os.path.abspath(__file__)
for _ in range(_PROJECT_ROOT_LEVELS):
    _PROJECT_ROOT_BASE = os.path.dirname(_PROJECT_ROOT_BASE)
PROJECT_ROOT = os.environ.get(
    "AI_IDS_PROJECT_ROOT",
    _PROJECT_ROOT_BASE
)

# ==========================================================
# LINUX PATH COMPATIBILITY
# ==========================================================

# If running on Linux and PROJECT_ROOT doesn't contain 'AI_IDS',
# try to detect based on common Linux directory patterns
if os.name == "posix" and "AI_IDS" not in PROJECT_ROOT:
    # Check if /data/AI_IDS exists (common Linux setup)
    if os.path.isdir("/data/AI_IDS"):
        PROJECT_ROOT = "/data/AI_IDS"
    # Check if /data exists
    elif os.path.isdir("/data"):
        PROJECT_ROOT = "/data"
    # Check if running from /home with project in /home
    elif PROJECT_ROOT.startswith("/home") and os.path.isdir(
        os.path.join(PROJECT_ROOT, "AI_IDS")
    ):
        # Keep original, it already resolved correctly
        pass

# ==========================================================
# COMMON DATA
# ==========================================================

DATASET_ROOT = os.path.join(PROJECT_ROOT, "DATASETS")  # Base path for input CSV files

PCC_RESULT_ROOT = os.path.join(PROJECT_ROOT, "PCC_RESULTS")  # Base path for results/ models

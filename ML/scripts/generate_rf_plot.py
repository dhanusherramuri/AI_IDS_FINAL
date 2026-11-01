import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

from sklearn.metrics import (
    roc_curve,
    auc,
    ConfusionMatrixDisplay
)

from COMMON.config import PROJECT_ROOT

ML_PROJECT_ROOT = os.path.join(PROJECT_ROOT, "ML")

RESULT_PATH = os.path.join(
    ML_PROJECT_ROOT,
    "RESULTS",
    "RandomForest_MULTICLASS",
    "threshold_0.90"
)

REPORT_PATH = os.path.join(
    RESULT_PATH,
    "REPORT"
)

os.makedirs(REPORT_PATH, exist_ok=True)

# ==========================================================
# LOAD RESULTS
# ==========================================================

results = pd.read_csv(
    os.path.join(
        RESULT_PATH,
        "10Fold_Results.csv"
    )
)

# ==========================================================
# FOLD ACCURACY
# ==========================================================

plt.figure(figsize=(8,5))

plt.plot(
    results["Fold"].to_numpy(),
    results["Accuracy"].to_numpy(),
    marker="o",
    linewidth=2
)

plt.title("Fold Accuracy")

plt.xlabel("Fold")

plt.ylabel("Accuracy")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        REPORT_PATH,
        "Fold_Accuracy.png"
    ),
    dpi=300
)

plt.close()

# ==========================================================
# METRIC COMPARISON
# ==========================================================

metrics = [
    "Accuracy",
    "Balanced Accuracy",
    "Precision",
    "Recall",
    "F1 Score",
    "Macro F1",
    "AUC",
    "Specificity"
]

means = results[metrics].mean()

plt.figure(figsize=(10,5))

means.plot(kind="bar")

plt.ylabel("Score")

plt.ylim(0,1.05)

plt.title("Average Metric Comparison")

plt.tight_layout()

plt.savefig(
    os.path.join(
        REPORT_PATH,
        "Metric_Comparison.png"
    ),
    dpi=300
)

plt.close()

# ==========================================================
# METRIC BOXPLOT
# ==========================================================

plt.figure(figsize=(10,5))

results[metrics].boxplot()

plt.xticks(rotation=25)

plt.ylabel("Score")

plt.title("Metric Distribution")

plt.tight_layout()

plt.savefig(
    os.path.join(
        REPORT_PATH,
        "Metric_Boxplot.png"
    ),
    dpi=300
)

plt.close()


# ==========================================================
# AVERAGE CONFUSION MATRIX
# ==========================================================

cm_files = sorted(
    glob.glob(
        os.path.join(
            RESULT_PATH,
            "confusion_matrix_fold_*.csv"
        )
    )
)

first_cm = pd.read_csv(cm_files[0], index_col=0)

num_classes = first_cm.shape[0]

cm_sum = np.zeros((num_classes, num_classes))

for file in cm_files:

    cm = pd.read_csv(
        file,
        index_col=0
    ).to_numpy(dtype=float)

    cm_sum += cm

cm_avg = cm_sum / len(cm_files)

labels = first_cm.index.tolist()

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm_avg,
    display_labels=labels
)

fig, ax = plt.subplots(figsize=(10,10))

disp.plot(
    ax=ax,
    cmap="Blues",
    values_format=".1f",
    xticks_rotation=90,
    colorbar=False
)

plt.title("Average Confusion Matrix")

plt.tight_layout()

plt.savefig(
    os.path.join(
        REPORT_PATH,
        "Average_Confusion_Matrix.png"
    ),
    dpi=300
)

plt.close()

print("="*60)
print("Random Forest Multiclass plots generated successfully.")
print("="*60)

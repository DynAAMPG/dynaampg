import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *


plt.rcParams.update({
    "text.usetex": True,
    "font.family": 'Times New Roman'
})

font_size = 40

# File names for the confusion matrices
csv_files = [os.path.join(SAVED_EVALS_DIR, 'OOD-ISCX-VPN.csv'), os.path.join(SAVED_EVALS_DIR, 'OOD-VNAT.csv'), os.path.join(SAVED_EVALS_DIR, 'OOD-ISCX-Tor.csv')]
dataset_names = ['ISCX-VPN', 'VNAT', 'ISCX-Tor']

# Plot three confusion matrices in a single row
fig, axes = plt.subplots(1, 3, figsize=(30, 10))

for index, csv_file in enumerate(csv_files):
    ax = axes[index]

    df = pd.read_csv(csv_file)
    cm_percentage = df.values.astype(float)
    class_labels = df.columns.tolist()

    im = ax.imshow(cm_percentage, interpolation='nearest', cmap='Blues')
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=font_size)

    # Add class labels
    tick_marks = np.arange(len(class_labels))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_labels, ha="right", fontsize=font_size + 2)
    ax.set_yticklabels(class_labels, fontsize=font_size)

    # Add text annotations
    thresh = cm_percentage.max() / 2.
    for i, j in np.ndindex(cm_percentage.shape):
        ax.text(j, i, f"{cm_percentage[i, j]:.4f}%",
                ha="center", va="center",
                color="white" if cm_percentage[i, j] > thresh else "black",
                fontsize=font_size + 4)

    ax.set_xlabel('Predicted Class', fontsize=font_size)
    ax.set_ylabel('True Class', fontsize=font_size)

    # Extract just the dataset name from the file path for the title
    dataset_name = os.path.basename(csv_file)[:-4]  # Remove .csv extension
    ax.set_title(f'Confusion Matrix for {dataset_name}', fontsize=font_size)

    # Draw outline on four sides of the whole plot
    ax.spines['top'].set_linewidth(1)
    ax.spines['right'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)

plt.tight_layout()
plt.savefig('visualization/fig_ood_confusion_matrices.pdf')
plt.show()

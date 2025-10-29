#!/usr/bin/env python3
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    csv_file = "confusion_matrix.csv"
    output_image = "confusion_matrix.png" 
    dpi = 300  # fixed DPI as before

    # === LOAD DATA ===
    df = pd.read_csv(csv_file, index_col=0)

    # === NORMALIZE TO PERCENTAGES ===
    df_perc = df.div(df.sum(axis=1), axis=0) * 100

    # === PLOT ===
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(df_perc.values, interpolation='nearest', aspect='auto')

    # === COLORBAR ===
    fig.colorbar(im, ax=ax, label='Percentage')

    # === TICKS AND LABELS ===
    classes = df_perc.index.tolist()
    ax.set_xticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=90)
    ax.set_yticks(np.arange(len(classes)))
    ax.set_yticklabels(classes)

    # === ANNOTATE CELLS ===
    for i in range(df_perc.shape[0]):
        for j in range(df_perc.shape[1]):
            ax.text(j, i, f"{df_perc.iat[i, j]:.2f}", 
                    ha='center', va='center', fontsize=6)

    # === LABELS AND TITLE ===
    ax.set_xlabel('Predicted Class')
    ax.set_ylabel('True Class')
    ax.set_title('Normalized Confusion Matrix (%)')

    plt.tight_layout()
    plt.savefig(output_image, dpi=dpi)
    plt.show()

    print(f"✅ Saved confusion matrix heatmap to {output_image} at {dpi} DPI")

if __name__ == "__main__":
    main()

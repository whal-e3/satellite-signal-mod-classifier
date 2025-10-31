#!/usr/bin/env python3
import sys
import re
import pandas as pd
import numpy as np

def main():
    input_file = "result_raw.txt"

    # 1) Parse (GT, Pred) pairs
    entries = []
    current_gt = None
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if "->" not in line:
                current_gt = line.upper()
            else:
                pred = line.split("->")[1].strip().split()[0].upper()
                entries.append((current_gt, pred))

    # 2) Build confusion matrix
    classes = sorted({gt for gt, _ in entries} | {pred for _, pred in entries})
    matrix = pd.DataFrame(0, index=classes, columns=classes)
    for gt, pred in entries:
        matrix.loc[gt, pred] += 1

    # 3) Compute per-class TP, FP, FN, TN and metrics
    N = matrix.values.sum()
    metrics = []
    for cls in classes:
        TP = matrix.loc[cls, cls]
        FP = matrix[cls].sum() - TP
        FN = matrix.loc[cls].sum() - TP
        TN = N - TP - FP - FN
        prec = TP / (TP + FP) if (TP + FP) else 0
        rec  = TP / (TP + FN) if (TP + FN) else 0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
        metrics.append((cls, TP, FP, FN, TN, prec*100, rec*100, f1*100))

    metrics_df = pd.DataFrame(
        metrics,
        columns=["Class","TP","FP","FN","TN","Precision(%)","Recall(%)","F1(%)"]
    ).set_index("Class")

    # 4) Compute overall metrics
    accuracy = np.trace(matrix.values) / N * 100
    macro_prec = metrics_df["Precision(%)"].mean()
    macro_rec  = metrics_df["Recall(%)"].mean()
    macro_f1   = metrics_df["F1(%)"].mean()
    # Cohen's Kappa
    p0 = accuracy/100
    row_sums = matrix.sum(axis=1)
    col_sums = matrix.sum(axis=0)
    pe = sum(row_sums[c]*col_sums[c] for c in classes) / (N**2)
    kappa = (p0 - pe) / (1 - pe) if (1 - pe) else 0

    # 5) Print results
    print("\n=== Confusion Matrix ===")
    print(matrix.to_string())
    print("\n=== Per-Class Metrics (in %) ===")
    print(metrics_df.to_string(float_format="{:.2f}".format))
    print("\n=== Summary ===")
    print(f"Total samples (N): {N}")
    print(f"Macro Precision:    {macro_prec:.2f}%")
    print(f"Macro Recall:       {macro_rec:.2f}%")
    print(f"Macro F1-score:     {macro_f1:.2f}%")
    print(f"Overall Accuracy (OA): {accuracy:.2f}%")
    print(f"Cohen’s Kappa:      {kappa:.4f}\n")

if __name__ == "__main__":
    main()



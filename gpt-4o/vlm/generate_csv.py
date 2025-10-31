#!/usr/bin/env python3
import re
import csv
import argparse
from collections import defaultdict

def main():
    modulations = [
        "16APSK","32APSK","8PSK","BFSK","BPSK",
        "CSS","GFSK","GMSK","NBFM","OOK","QPSK","WBFM"
    ]

    input_file = "result_raw.txt"

    # === Parse raw text into counts ===
    counts = defaultdict(lambda: defaultdict(int))
    current_gt = None
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # If line does not contain '->', it's a ground truth header
            if "->" not in line:
                current_gt = line.upper()
                continue
            # Otherwise parse prediction
            # Format: "<snr>db -> <PRED>"
            parts = line.split("->")
            pred = parts[1].strip().split()[0].upper()
            if current_gt:
                counts[current_gt][pred] += 1

    # === Write CSV ===
    output_csv = "confusion_matrix.csv"
    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        # Header row
        writer.writerow([""] + modulations)
        # Each GT row
        for gt in modulations:
            row = [counts[gt].get(col, 0) for col in modulations]
            writer.writerow([gt] + row)

    print(f"✅ Confusion matrix CSV saved to '{output_csv}'")

if __name__ == "__main__":
    main()
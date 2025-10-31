#!/usr/bin/env python3
import re
import argparse
from collections import defaultdict

def main():
    input_file = "result.txt"
    output_file = "result_raw.txt"
    results = defaultdict(list)

    # 정규표현식 패턴: [번호] 📦 modulation/SNRdb → 🧠 Predicted: prediction
    pattern = re.compile(
        r"\[\d+\] 📦 (\w+)/(-?\d+)db → 🧠 Predicted: (\w+)",
        re.IGNORECASE
    )

    # === File Reading & Parsing ===
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                mod, snr, pred = match.groups()
                results[mod.upper()].append((int(snr), pred.upper()))

    # === Write to Output File ===
    with open(output_file, "w", encoding="utf-8") as out:
        for mod in sorted(results.keys()):
            out.write(f"{mod}\n")
            for snr, pred in sorted(results[mod], key=lambda x: x[0], reverse=True):
                out.write(f"{snr}db -> {pred}\n")
            out.write("\n")

    print(f"✅ Parsed output saved to '{output_file}'")

if __name__ == "__main__":
    from collections import defaultdict
    main()


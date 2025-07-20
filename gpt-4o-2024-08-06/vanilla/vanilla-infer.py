#!/usr/bin/env python3

import os
from pathlib import Path
from openai import OpenAI
from PIL import Image
import base64
import io

# === SETUP ===
client = OpenAI(api_key="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")
ROOT_DIR = Path("/mnt/d/mlwhale-share/rf-sat-hack/signal_dataset/MAIN/custom/image_dataset/image_clean_signal_training0704")

# Prompt
text_prompt = (
    "Given four images showing a radio signal's characteristics "
    "(constellation diagram, time-domain graph, frequency-domain graph, waterfall diagram), "
    "identify the modulation type from: WBFM, NBFM, BPSK, QPSK, 8PSK, 16APSK, 32APSK, "
    "GMSK, CW, CSS, BFSK, GFSK. Output the modulation type only."
)

def encode_image(image_path):
    with Image.open(image_path) as img:
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/png;base64,{encoded}",
            "detail": "high"
        }
    }

def get_all_image_sets(root):
    """
    Return list of (4 image paths) grouped by modulation/SNR.
    """
    image_sets = []
    for modulation_dir in root.iterdir():
        if not modulation_dir.is_dir():
            continue
        for snr_dir in modulation_dir.iterdir():
            if not snr_dir.is_dir():
                continue
            image_paths = sorted(snr_dir.glob("*.png"))
            if len(image_paths) == 4:
                image_sets.append(image_paths)
            else:
                print(f"Skipping {snr_dir}: expected 4 images, found {len(image_paths)}")
    return image_sets

def run_inference(image_sets, output_path="result_raw.txt"):
    with open(output_path, "w", encoding="utf-8") as f_out:
        for image_paths in image_sets:
            try:
                modulation = image_paths[0].parents[1].name.upper()
                snr = image_paths[0].parents[0].name.upper()
            except IndexError:
                modulation = "UNKNOWN"
                snr = "UNKNOWN"

            images = [encode_image(p) for p in image_paths]

            response = client.chat.completions.create(
                model="gpt-4o-2024-08-06",
                messages=[
                    {"role": "system", "content": "You are an expert in radio signal analysis."},
                    {"role": "user", "content": [{"type": "text", "text": text_prompt}] + images}
                ],
                max_tokens=16,
                temperature=0.0
            )

            prediction = response.choices[0].message.content.strip()

            # Output format compatible with confusion matrix script
            print(f"{modulation}/{snr} -> {prediction}")
            f_out.write(f"{modulation}/{snr}\n-> {prediction}\n")

if __name__ == "__main__":
    sets = get_all_image_sets(ROOT_DIR)
    print(f"Found {len(sets)} image sets.")
    run_inference(sets, output_path="result_raw.txt")
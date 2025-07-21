#!/usr/bin/env python3
import os
import json
import time
from pathlib import Path
from tqdm import tqdm
import openai

# === CONFIG ===
API_KEY = "your-api-key-here"  # Replace with your OpenAI API key
BASE_DIR = Path("/your/full/path/to/image_clean_signal_training0704")
OUTPUT_JSONL = "vision_finetune_dataset.jsonl"
ALLOWED_EXTS = {".png", ".jpg", ".jpeg", ".webp"}

openai.api_key = API_KEY

def upload_image(image_path):
    with open(image_path, "rb") as f:
        response = openai.files.create(file=f, purpose="fine-tune")
        return response.id

def build_jsonl_entry(image_file_ids, modulation_label):
    return {
        "messages": [
            {"role": "system", "content": "You are a modulation classification assistant."},
            {"role": "user", "content": "What is the modulation type of this signal?"},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"openai://{fid}"}}
                for fid in image_file_ids
            ]},
            {"role": "assistant", "content": modulation_label}
        ]
    }

def main():
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    examples = []
    total_uploaded = 0

    for mod_dir in sorted(BASE_DIR.iterdir()):
        if not mod_dir.is_dir():
            continue
        modulation = mod_dir.name

        for snr_dir in sorted(mod_dir.iterdir()):
            if not snr_dir.is_dir():
                continue

            image_paths = sorted([
                p for p in snr_dir.glob("*")
                if p.suffix.lower() in ALLOWED_EXTS
            ])

            if len(image_paths) != 4:
                print(f"Skipping {snr_dir} – expected 4 images, found {len(image_paths)}")
                continue

            try:
                file_ids = []
                for path in image_paths:
                    file_id = upload_image(path)
                    file_ids.append(file_id)
                    total_uploaded += 1
                    time.sleep(1)  # Avoid rate limits

                entry = build_jsonl_entry(file_ids, modulation)
                examples.append(entry)

            except Exception as e:
                print(f"[!] Failed in {snr_dir}: {e}")

    # Save to JSONL
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example) + "\n")

    print(f"\n✅ Created {len(examples)} examples, uploaded {total_uploaded} images.")
    print(f"💾 JSONL saved to: {OUTPUT_JSONL}")

if __name__ == "__main__":
    main()
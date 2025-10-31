#!/usr/bin/env python3
import time
import os
os.environ["HF_HOME"] = "/data/buckelwal8979/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/data/buckelwal8979/hf_cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/data/buckelwal8979/hf_cache"
os.environ["TRITON_CACHE_DIR"] = "/data/buckelwal8979/hf_cache/triton"
import tarfile
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import LlavaProcessor, LlavaForConditionalGeneration
import re

# === CONFIG ===
TAR_PATH = "/data/datasets/tarfiles/image_clean_signal_inference0707.tar"
EXTRACT_DIR = "/local_datasets/image_clean_signal_inference0707"


def extract_dataset(tar_path: str, extract_dir: str):
    dest = Path(extract_dir).parent
    if Path(extract_dir).exists():
        print(f"Dataset already extracted at {extract_dir}")
        return
    print(f"Extracting {tar_path} to {dest}...")
    with tarfile.open(tar_path) as tar:
        tar.extractall(path=dest)
    print("Extraction complete.")

def merge_four_images_to_grid(img_dir: Path, mod: str, snr: str) -> Image.Image:
    snr_num = re.sub(r"[^0-9\-]", "", snr)   # '10dB' → '10',  '-5db' → '-5'
    filenames = {
        "constellation": f"{mod}_{snr_num}_constellation.png",
        "waterfall": f"{mod}_{snr_num}_waterfall.png",
        "freq": f"{mod}_{snr_num}_freq.png",
        "time": f"{mod}_{snr_num}_time.png",
    }

    TARGET_SIZE = 168       # 168 * 2 = 336
    
    try:
        img_constellation = Image.open(img_dir / filenames["constellation"]).convert("RGB").resize((TARGET_SIZE, TARGET_SIZE))
        img_waterfall     = Image.open(img_dir / filenames["waterfall"]).convert("RGB").resize((TARGET_SIZE, TARGET_SIZE))
        img_freq          = Image.open(img_dir / filenames["freq"]).convert("RGB").resize((TARGET_SIZE, TARGET_SIZE))
        img_time          = Image.open(img_dir / filenames["time"]).convert("RGB").resize((TARGET_SIZE, TARGET_SIZE))
    except FileNotFoundError as e:
        raise RuntimeError(f"Missing image file in {img_dir}: {e}")

    # 최종 병합 이미지 크기를 2*TARGET_SIZE 로 설정
    merged_img = Image.new("RGB", (TARGET_SIZE * 2, TARGET_SIZE * 2))
    merged_img.paste(img_waterfall,     (0, 0))
    merged_img.paste(img_constellation, (TARGET_SIZE, 0))
    merged_img.paste(img_time,          (0, TARGET_SIZE))
    merged_img.paste(img_freq,          (TARGET_SIZE, TARGET_SIZE))
    return merged_img


def main():
    # 1. Extract dataset
    extract_dataset(TAR_PATH, EXTRACT_DIR)

    # 2. Load processor & model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    processor = LlavaProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
    model = LlavaForConditionalGeneration.from_pretrained(
        "llava-hf/llava-1.5-7b-hf", torch_dtype=torch.float16
    ).to(device).eval()

    # Build Prompt
    IMAGE_TOKEN = processor.tokenizer.image_token  # usually "<image>"
    text_prompt = (
    f"USER: The following image contains four plots describing a radio signal's characteristics, arranged in a 2x2 grid:\n"
    f"- Top-left: Waterfall diagram\n"
    f"- Top-right: Constellation diagram\n"
    f"- Bottom-left: Time-domain graph\n"
    f"- Bottom-right: Frequency-domain graph\n\n"
    f"Based on these plots, identify the modulation type. Choose from the following list: "
    f"WBFM, NBFM, BPSK, QPSK, 8PSK, 16APSK, 32APSK, GMSK, CW, CSS, BFSK, GFSK.\n"
    f"Answer with exactly one word.\n{IMAGE_TOKEN}\nASSISTANT:"
    )
    # text_prompt = (
    #     f"USER: Given four images showing a radio signal's characteristics "
    #     f"(constellation diagram, time‑domain graph, frequency‑domain graph, waterfall diagram), "
    #     f"identify the modulation type from: WBFM, NBFM, BPSK, QPSK, 8PSK, 16APSK, 32APSK, "
    #     f"GMSK, CW, CSS, BFSK, GFSK. "
    #     f"Answer with exactly one word.\n{IMAGE_TOKEN}\nASSISTANT:"
    # )

    # 3. Iterate through each modulation/SNR folder and run inference
    base_dir = Path(EXTRACT_DIR)
    sample_count = 0

    for mod_dir in sorted(base_dir.iterdir()):
        if not mod_dir.is_dir():
            continue
        mod = mod_dir.name
        for snr_dir in sorted(mod_dir.iterdir()):
            if not snr_dir.is_dir():
                continue
            snr = snr_dir.name
            img_paths = sorted(snr_dir.glob("*.png"))
            if len(img_paths) != 4:
                print(f"Skipping {mod}/{snr}: found {len(img_paths)} images (expected 4)")
                continue

            try:
                merged_image = merge_four_images_to_grid(snr_dir, mod, snr)
            except RuntimeError as e:
                print(f"Skipping {mod}/{snr}: {e}")
                continue

            inputs = processor(images=merged_image, text=text_prompt, return_tensors="pt").to(device)

            # Generate answer
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=8,       # few tokens are enough for one word
                    do_sample=False,
                    pad_token_id=processor.tokenizer.pad_token_id,
                    eos_token_id=processor.tokenizer.eos_token_id,
                )
            # keep only the newly generated tokens
            prompt_len = inputs["input_ids"].shape[-1]
            answer_ids    = output_ids[0][prompt_len:]
            answer = processor.decode(answer_ids, skip_special_tokens=True).strip().upper()

            sample_count += 1
            print(f"[{sample_count:04d}] 📦 {mod}/{snr} → 🧠 Answer: {answer}")

if __name__ == "__main__":
    main()
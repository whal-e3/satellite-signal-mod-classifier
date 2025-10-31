#!/usr/bin/env python3
import os
os.environ["HF_HOME"] = "/data/buckelwal8979/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/data/buckelwal8979/hf_cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/data/buckelwal8979/hf_cache"
os.environ["TRITON_CACHE_DIR"] = "/data/buckelwal8979/hf_cache/triton"

import tarfile
from pathlib import Path
from PIL import Image
import torch
from transformers import LlavaProcessor, LlavaForConditionalGeneration
from peft import PeftModel

# === CONFIG ===
TAR_PATH = "/data/datasets/tarfiles/image_clean_signal_training0718.tar"
EXTRACT_DIR = "/local_datasets/image_clean_signal_training0718"
LORA_ADAPTER_PATH = "lora_train_clean0704_allsnr_ep10_2e-4/checkpoint-9840"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_TOKEN = "<image>"
PROMPT = (
    f"USER: The following image contains four plots describing a radio signal's characteristics, arranged in a 2x2 grid:\n"
    f"- Top-left: Constellation diagram\n"
    f"- Top-right: Waterfall diagram\n"
    f"- Bottom-left: Time-domain graph\n"
    f"- Bottom-right: Frequency-domain graph\n\n"
    f"Based on these plots, identify the modulation type. Choose from the following list: "
    f"WBFM, NBFM, BPSK, QPSK, 8PSK, 16APSK, 32APSK, GMSK, CW, CSS, BFSK, GFSK.\n"
    f"Answer with exactly one word.\n{IMAGE_TOKEN}\nASSISTANT:"
)

# === 이미지 병합 ===
def merge_images(image_paths):
    TARGET_SIZE = 168
    FINAL_SIZE = TARGET_SIZE * 2
    merged = Image.new("RGB", (FINAL_SIZE, FINAL_SIZE))
    positions = [(0, 0), (TARGET_SIZE, 0), (0, TARGET_SIZE), (TARGET_SIZE, TARGET_SIZE)]
    for pos, path in zip(positions, image_paths):
        img = Image.open(path).convert("RGB").resize((TARGET_SIZE, TARGET_SIZE))
        merged.paste(img, pos)
    return merged

# === untar ===
def extract_dataset():
    parent = os.path.dirname(EXTRACT_DIR)
    if not os.path.isdir(EXTRACT_DIR):
        os.makedirs(parent, exist_ok=True)
        with tarfile.open(TAR_PATH, "r") as tar:
            tar.extractall(path=parent)
        print(f"✅ Extracted archive into {parent}")
    else:
        print(f"📂 Dataset already extracted at {EXTRACT_DIR}")
    return EXTRACT_DIR

# === Inference ===
@torch.no_grad()
def infer_on_folder(data_root):
    print("🚀 Loading base model and LoRA adapter...")
    base_model = LlavaForConditionalGeneration.from_pretrained(
        "llava-hf/llava-1.5-7b-hf", torch_dtype=torch.float16
    ).to(DEVICE)
    model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH).to(DEVICE)
    model.eval()

    processor = LlavaProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

    count = 0
    for mod in sorted(os.listdir(data_root)):
        mod_dir = os.path.join(data_root, mod)
        if not os.path.isdir(mod_dir):
            continue

        for snr in sorted(os.listdir(mod_dir)):
            snr_dir = os.path.join(mod_dir, snr)
            if not os.path.isdir(snr_dir):
                continue

            image_paths = sorted([
                os.path.join(snr_dir, f)
                for f in os.listdir(snr_dir)
                if f.endswith(".png")
            ])
            if len(image_paths) != 4:
                print(f"⚠️ Skipping {snr_dir}: only {len(image_paths)} images found")
                continue

            try:
                merged_img = merge_images(image_paths)
            except Exception as e:
                print(f"❌ Error merging images in {snr_dir}: {e}")
                continue

            inputs = processor(
                text=PROMPT,
                images=merged_img,
                return_tensors="pt"
            ).to(DEVICE, torch.float16)

            output = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False
            )
            decoded = processor.batch_decode(output, skip_special_tokens=True)[0]
            prediction = decoded.strip().split("ASSISTANT:")[-1].strip().split()[0]

            count += 1
            print(f"[{count:04d}] 📦 {mod}/{snr} → 🧠 Predicted: {prediction}")

    print(f"\n🎯 Completed inference on {count} samples.")
    print(f"* LoRA module used: {LORA_ADAPTER_PATH}")

if __name__ == "__main__":
    data_root = extract_dataset()
    infer_on_folder(data_root)
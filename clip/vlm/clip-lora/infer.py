#!/usr/bin/env python3
import os
os.environ["HF_HOME"] = "/data/buckelwal8979/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/data/buckelwal8979/hf_cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/data/buckelwal8979/hf_cache"
os.environ["TRITON_CACHE_DIR"] = "/data/buckelwal8979/hf_cache/triton"

import torch
from peft import PeftModel
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import tarfile

# === CONFIG ===
TAR_PATH = "/data/datasets/tarfiles/image_clean_signal_inference0707.tar"
EXTRACT_DIR = "/local_datasets/image_clean_signal_inference0707"
# 아래 변수만 변경하면 됨!
ADAPTER_PATH = "clip_lora_adapter200_3e4_r16"
# modulations = [
#     "WBFM", "NBFM", "BPSK", "QPSK", "8PSK", "16APSK", "32APSK",
#     "GMSK", "CW", "CSS", "BFSK", "GFSK"
# ]
modulations = sorted(os.listdir(EXTRACT_DIR))
prompt_prefix = "This radio signal, illustrated by a constellation diagram, a time-domain plot, a frequency-domain graph, and a waterfall diagram, uses"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === STEP 1: Extract Dataset ===
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

# === STEP 2: Load model and run inference ===
def infer_on_folder(data_root):
    print("🚀 Loading base model and LoRA adapter...")
    base_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    # LoRA 모듈 + CLIP 원본
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH).to(device)
    model.eval()

    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

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
                if f.lower().endswith(".png")
            ])
            if len(image_paths) != 4:
                print(f"⚠️ Skipping {snr_dir}: found {len(image_paths)} images")
                continue

            try:
                images = [Image.open(p).convert("RGB") for p in image_paths]
            except Exception as e:
                print(f"❌ Error loading images in {snr_dir}: {e}")
                continue

            text_inputs = [f"{prompt_prefix} {label}" for label in modulations]
            inputs = processor(
                text=text_inputs,
                images=images,
                return_tensors="pt",
                padding=True
            ).to(device)

            with torch.no_grad():
                outputs = model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)
                avg_probs = probs.mean(dim=0)
                top_idx = avg_probs.argmax().item()
                top_pred = modulations[top_idx]

            count += 1
            print(f"[{count:04d}] 📦 {mod}/{snr} → 🧠 Predicted: {top_pred}")
    print(f"\n🎯 Completed inference on {count} samples.")
    print(f"* LoRA module: {ADAPTER_PATH}")

# === MAIN ===
if __name__ == "__main__":
    data_root = extract_dataset()
    infer_on_folder(data_root)

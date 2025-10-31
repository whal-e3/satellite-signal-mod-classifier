#!/usr/bin/env python3
import os
os.environ["HF_HOME"]               = "/data/buckelwal8979/hf_cache"
os.environ["TRANSFORMERS_CACHE"]     = "/data/buckelwal8979/hf_cache"
os.environ["HUGGINGFACE_HUB_CACHE"]  = "/data/buckelwal8979/hf_cache"
os.environ["TRITON_CACHE_DIR"]       = "/data/buckelwal8979/hf_cache/triton"
import tarfile
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch

# === CONFIG ===
TAR_PATH = "/data/datasets/tarfiles/image_clean_signal_inference0707.tar"
EXTRACT_DIR = "/local_datasets/image_clean_signal_inference0707"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === MODULATION CLASSES ===
# modulations = [
#     "WBFM", "NBFM", "BPSK", "QPSK", "8PSK", "16APSK", "32APSK",
#     "GMSK", "CW", "CSS", "BFSK", "GFSK"
# ]
modulations = sorted(os.listdir(EXTRACT_DIR))

# [Vanilla prefix]
# prompt_prefix = "The signal modulation shown in these images is"
prompt_prefix = "This radio signal, illustrated by a constellation diagram, a time-domain plot, a frequency-domain graph, and a waterfall diagram, uses"


# === LOAD CLIP MODEL ===
print("🚀 Loading CLIP model...")
model = CLIPModel.from_pretrained("clip_best_ep100").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# === STEP 1: Untar Dataset ===
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

# === STEP 2: Inference ===
def run_inference(data_root):
    sample_count = 0
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
                logits_per_image = outputs.logits_per_image  # (4, 12)
                probs = logits_per_image.softmax(dim=1)      # (4, 12)
                avg_probs = probs.mean(dim=0)
                top_idx = avg_probs.argmax().item()
                top_pred = modulations[top_idx]

            sample_count += 1
            print(f"[{sample_count:04d}] 📦 {mod}/{snr} → 🧠 Predicted: {top_pred}")
    print(f"\n🎉 Completed inference on {sample_count} samples.")

# === MAIN ===
if __name__ == "__main__":
    data_root = extract_dataset()
    run_inference(data_root)
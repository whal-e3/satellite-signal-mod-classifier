#!/usr/bin/env python3
import time
import os
os.environ["HF_HOME"] = "/data/buckelwal8979/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/data/buckelwal8979/hf_cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/data/buckelwal8979/hf_cache"
os.environ["TRITON_CACHE_DIR"] = "/data/buckelwal8979/hf_cache/triton"
import tarfile
import random
from pathlib import Path
from PIL import Image

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model

# === CONFIG ===
TAR_PATH = "/data/datasets/tarfiles/image_clean_signal_training0704.tar"
EXTRACT_DIR = "/local_datasets/image_clean_signal_training0704"
# modulations = [
#     "WBFM", "NBFM", "BPSK", "QPSK", "8PSK", "16APSK", "32APSK",
#     "GMSK", "CW", "CSS", "BFSK", "GFSK"
# ]
modulations = sorted(os.listdir(EXTRACT_DIR))
prompt_prefix = "This radio signal, illustrated by a constellation diagram, a time-domain plot, a frequency-domain graph, and a waterfall diagram, uses"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 8
# 1차 변수로 사용
epochs = 100
# 2차 변수로 사용
learning_rate = 1e-4

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

# === STEP 2: Dataset Class ===
class ModulationDataset(Dataset):
    def __init__(self, root):
        self.samples = []
        for label, mod in enumerate(sorted(os.listdir(root))):
            mod_dir = Path(root) / mod
            if not mod_dir.is_dir():
                continue
            for snr in sorted(os.listdir(mod_dir)):
                snr_dir = mod_dir / snr
                if not snr_dir.is_dir():
                    continue
                imgs = sorted([snr_dir / f for f in os.listdir(snr_dir) if f.endswith(".png")])
                if len(imgs) == 4:
                    self.samples.append((imgs, label))
        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        paths, label = self.samples[idx]
        images = [Image.open(p).convert("RGB") for p in paths]
        text = [f"{prompt_prefix} {m}" for m in modulations]
        return images, text, label

def collate_fn(batch, processor):
    all_images, all_texts, labels = [], [], []
    for images, texts, label in batch:
        all_images.extend(images)
        all_texts.extend(texts)
        labels.append(label)
    proc = processor(images=all_images, text=all_texts, return_tensors="pt", padding=True)
    B = len(batch)
    proc["input_ids"] = proc["input_ids"].view(B, 12, -1)
    proc["attention_mask"] = proc["attention_mask"].view(B, 12, -1)
    proc["pixel_values"] = proc["pixel_values"].view(B, 4, 3, 224, 224)
    proc["labels"] = torch.tensor(labels)
    return proc

# === STEP 3: Training ===
def train():
    start_time = time.time()
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # LoRA Hyperparameter setting
    lora_cfg = LoraConfig(
        # 3차 변수로 사용
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none"
    )
    model = get_peft_model(model, lora_cfg).to(device)

    dataset = ModulationDataset(EXTRACT_DIR)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            collate_fn=lambda b: collate_fn(b, processor))

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(dataloader) * epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(total_steps * 0.05), total_steps)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for step, batch in enumerate(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            B = batch["pixel_values"].size(0)
            imgs = batch["pixel_values"].view(B * 4, 3, 224, 224)
            txt_ids = batch["input_ids"].view(B * 12, -1)
            txt_mask = batch["attention_mask"].view(B * 12, -1)

            outputs = model(pixel_values=imgs, input_ids=txt_ids, attention_mask=txt_mask)

            # logits = outputs.logits_per_image.view(B, 4, 12).softmax(dim=-1).mean(dim=1)
            logits_all = outputs.logits_per_image  # shape: (B×4, B×12)
            logits_all = logits_all.view(B, 4, B, 12)  # reshape to group per sample
            # 배치 안에서 자기 자신의 텍스트만 비교 (i번째 이미지 vs i번째 텍스트)
            logits_per_sample = logits_all[torch.arange(B), :, torch.arange(B)]  # shape: (B, 4, 12)
            # 소프트맥스 후 평균
            probs = logits_per_sample.softmax(dim=-1).mean(dim=1)  # shape: (B, 12)
            loss = criterion(probs, batch["labels"])
            # loss = criterion(logits, batch["labels"])

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            if step % 50 == 0:
                print(f"Epoch {epoch+1}, Step {step}, Loss: {loss.item():.4f}")
        print(f"✅ Epoch {epoch+1} finished. Avg loss: {total_loss / len(dataloader):.4f}")

    # 세팅 변경시 아래 파일 이름 변경
    model.save_pretrained("clip_lora_adapter-all_ep100_r16_new")
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"⏱️ Training took {elapsed:.2f} seconds (≈ {elapsed/60:.2f} minutes)")
    print("🎉 Training complete. Adapter saved to 'clip_lora_adapter-all_ep100_r16_new'")

# === MAIN ===
if __name__ == "__main__":
    data_root = extract_dataset()
    train()
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
from torch.optim import AdamW

# === CONFIG ===
TAR_PATH = "/data/datasets/tarfiles/image_clean_signal_training0704.tar"
EXTRACT_DIR = "/local_datasets/image_clean_signal_training0704"
modulations = sorted(os.listdir(EXTRACT_DIR))
prompt_prefix = (
    "This radio signal, illustrated by a constellation diagram, a time-domain plot, "
    "a frequency-domain graph, and a waterfall diagram, uses"
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 8
epochs = 30
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
        for label, mod in enumerate(modulations):
            mod_dir = Path(root) / mod
            if not mod_dir.is_dir():
                continue
            for snr in sorted(os.listdir(mod_dir)):
                try:
                    snr_value = int(snr.lower().replace("db", ""))
                except ValueError:
                    continue
                # SNR training 제한: 10 dB 이상
                if snr_value < 10:
                    continue
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
        texts = [f"{prompt_prefix} {m}" for m in modulations]
        return images, texts, label

# === collate_fn ===
def collate_fn(batch, processor):
    all_images, all_texts, labels = [], [], []
    for images, texts, label in batch:
        all_images.extend(images)
        all_texts.extend(texts)
        labels.append(label)
    proc = processor(images=all_images, text=all_texts,
                     return_tensors="pt", padding=True)
    B = len(batch)
    proc["input_ids"] = proc["input_ids"].view(B, len(modulations), -1)
    proc["attention_mask"] = proc["attention_mask"].view(B, len(modulations), -1)
    proc["pixel_values"] = proc["pixel_values"].view(B, 4, 3, 224, 224)
    proc["labels"] = torch.tensor(labels)
    return proc

# === STEP 3: Training with Frozen Backbone ===
def train():
    start_time = time.time()
    # Load CLIP
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Freeze backbone: vision & text encoders
    for param in model.vision_model.parameters():
        param.requires_grad = False
    for param in model.text_model.parameters():
        param.requires_grad = False

    # Prepare data
    dataset = ModulationDataset(EXTRACT_DIR)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            collate_fn=lambda b: collate_fn(b, processor),
                            num_workers=4, pin_memory=True)

    # Optimizer & Scheduler: only train unfrozen params
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = AdamW(trainable_params, lr=learning_rate)
    total_steps = len(dataloader) * epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.05),
        num_training_steps=total_steps
    )
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for step, batch in enumerate(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            B = batch["pixel_values"].size(0)
            imgs = batch["pixel_values"].view(B * 4, 3, 224, 224)
            txt_ids = batch["input_ids"].view(B * len(modulations), -1)
            txt_mask = batch["attention_mask"].view(B * len(modulations), -1)

            outputs = model(
                pixel_values=imgs,
                input_ids=txt_ids,
                attention_mask=txt_mask
            )

            logits_all = outputs.logits_per_image
            logits_all = logits_all.view(B, 4, B, len(modulations))
            logits_per_sample = logits_all[torch.arange(B), :, torch.arange(B)]
            probs = logits_per_sample.softmax(dim=-1).mean(dim=1)
            loss = criterion(probs, batch["labels"])

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            if step % 50 == 0:
                print(f"Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}")
        avg_loss = total_loss / len(dataloader)
        print(f"✅ Epoch {epoch} finished. Avg loss: {avg_loss:.4f}")

    # Save model (backbone frozen)
    save_dir = "clip_frozen_backbone"
    model.save_pretrained(save_dir)
    elapsed = time.time() - start_time
    print(f"⏱️ Training completed in {elapsed:.2f}s ({elapsed/60:.2f}min)")
    print(f"🎉 Model saved to '{save_dir}'")

# === MAIN ===
if __name__ == "__main__":
    extract_dataset()
    train()

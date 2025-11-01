#!/usr/bin/env python3
import os
os.environ["HF_HOME"] = "/data/buckelwal8979/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/data/buckelwal8979/hf_cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/data/buckelwal8979/hf_cache"
os.environ["TRITON_CACHE_DIR"] = "/data/buckelwal8979/hf_cache/triton"

import time
import tarfile
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import torch
from transformers import (
    LlavaProcessor,
    LlavaForConditionalGeneration,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    default_data_collator,
)
from peft import get_peft_model, LoraConfig, TaskType

# === 전역 설정 ===
TAR_DIR = Path("/data/datasets/tarfiles")
EXTRACT_ROOT = Path("/local_datasets")
LORA_MODULE = "./llava-lora_module-ground_train-allsnr-ep5"
TRAIN_TAR_NAME = "ai_amc_ground-train.tar"

IMAGE_TOKEN = "<image>"
text_prompt = (
    f"USER: The following image contains four plots describing a radio signal's characteristics, arranged in a 2x2 grid:\n"
    f"- Top-left: Constellation diagram\n"
    f"- Top-right: Waterfall diagram\n"
    f"- Bottom-left: Time-domain graph\n"
    f"- Bottom-right: Frequency-domain graph\n\n"
    f"Based on these plots, identify the modulation type. Choose from the following list: "
    f"WBFM, NBFM, BPSK, QPSK, 8PSK, 16APSK, 32APSK, GMSK, OOK, CSS, BFSK, GFSK.\n"
    f"Answer with exactly one word.\n{IMAGE_TOKEN}\nASSISTANT:"
)

def merge_images(paths):
    TARGET_SIZE = 168
    FINAL_SIZE = TARGET_SIZE * 2
    merged_image = Image.new("RGB", (FINAL_SIZE, FINAL_SIZE))
    positions = [(0, 0), (TARGET_SIZE, 0), (0, TARGET_SIZE), (TARGET_SIZE, TARGET_SIZE)]
    for pos, img_path in zip(positions, paths):
        img = Image.open(img_path).convert("RGB").resize((TARGET_SIZE, TARGET_SIZE))
        merged_image.paste(img, pos)
    return merged_image

def create_preprocessed_dataset(root_dir, processor):
    samples_to_process = []
    print("Scanning dataset files...")
    for mod_path in tqdm(list(root_dir.rglob("*"))):
        if mod_path.is_dir():
            image_paths = list(mod_path.glob("*.png"))
            if len(image_paths) == 4:
                label = mod_path.parts[-2]
                path_dict = {}
                for path in image_paths:
                    if path.name.endswith("_constellation.png"): path_dict["constellation"] = path
                    elif path.name.endswith("_waterfall.png"): path_dict["waterfall"] = path
                    elif path.name.endswith("_time.png"): path_dict["time"] = path
                    elif path.name.endswith("_freq.png"): path_dict["freq"] = path
                if len(path_dict) == 4:
                    ordered_paths = [
                        path_dict["constellation"], path_dict["waterfall"],
                        path_dict["time"], path_dict["freq"]
                    ]
                    samples_to_process.append((ordered_paths, label))

    print(f"Found {len(samples_to_process)} samples. Pre-processing all data into tensors...")
    processed_dataset = []
    prompt_inputs = processor(text=text_prompt, return_tensors="pt")
    prompt_length = prompt_inputs.input_ids.shape[1]

    for paths, label in tqdm(samples_to_process):
        image = merge_images(paths)
        full_text = f"{text_prompt} {label}"
        inputs = processor(text=full_text, images=image, return_tensors="pt")
        labels = inputs.input_ids.clone()
        labels[:, :prompt_length] = -100
        inputs['labels'] = labels
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        processed_dataset.append(inputs)

    print("Dataset pre-processing complete.")
    return processed_dataset

# === Epoch Loss Callback ===
class PrintEpochLoss(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        # average loss reported in state.log_history for the last step of the epoch
        logs = state.log_history[-1] if state.log_history else {}
        loss = logs.get("loss")
        if loss is not None:
            print(f"✅ Epoch {int(state.epoch)} finished. Avg loss: {loss:.4f}")


def main():
    # === 학습 데이터 분리 및 압축 해제 ===
    train_extract_path = EXTRACT_ROOT / Path(TRAIN_TAR_NAME).stem
    train_tar_path = TAR_DIR / TRAIN_TAR_NAME
    if not train_extract_path.exists() and train_tar_path.exists():
        print(f"Extracting training data: {train_tar_path.name}...")
        with tarfile.open(train_tar_path) as tar:
            tar.extractall(train_extract_path, filter='data')

    # === 모델 및 프로세서 로딩 ===
    print("Loading model and processor...")
    model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
    processor = LlavaProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

    # === LoRA 설정 ===
    print("Setting up LoRA configuration...")
    peft_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # === 데이터셋 생성 ===
    print("Creating training dataset...")
    train_dataset = create_preprocessed_dataset(train_extract_path, processor)

    # === Trainer 설정 ===
    print("Setting up Trainer...")
    training_args = TrainingArguments(
        output_dir=LORA_MODULE,
        per_device_train_batch_size=1,
        num_train_epochs=5,
        learning_rate=1e-4,
        weight_decay=0.01,
        fp16=True,
        label_names=["labels"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=default_data_collator,
        callbacks=[PrintEpochLoss],
    )

    # === 학습 시작 ===
    print("Starting training...")
    start_time = time.time()
    train_result = trainer.train()
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"⏱️ Training took {elapsed:.2f} seconds (≈ {elapsed/60:.2f} minutes)")
    print(f"Final training loss: {train_result.training_loss}")
    print(f"🎉 Training complete. Adapter saved to {LORA_MODULE}")

if __name__ == "__main__":
    main()
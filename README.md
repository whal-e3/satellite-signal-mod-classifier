# Automatic Modulation Classification on Satellite Signal using VLM

This repo is sharing tools and results of using 3 different VLM models as a Satellite Signal AMC (Automatic Modulation Classification) system. 
The repo is also for "AI를 활용한 취약점 발굴 시스템 공모전".

## 팀원
---

황선혁 (경희대학교 융합보안대학원 석사과정)

박철준 (경희대학교 컴퓨터공학과 조교수)

## Descriptions
---

CLIP, LLaVA, GPT-4o were used for the project. The repo provides scripts and "how to" for fine-tuning and inferencing each models. It also provides the result that's done on the models.

### 0. Dataset

12 Modulations is picked based on the Satellite usage.

- WBFM, NBFM
- BPSK, QPSK, 8PSK, 16APSK, 32APSK
- GMSK
- CW (OOK)
- CSS
- BFSK
- GFSK

Each modulation signal has 21 different SNR values from -20db to 20db (2db gap).
4 different Signal Representation images are used for each SNR value.

- Constellation Diagram
- Frequency Domain Graph
- Time Domain Graph
- Waterfall Diagram

### 1. VLM

- CLIP (clip-vit-base-patch32)
- LLaVA (llava-1.5-7b-hf)
- GPT-4o (gpt-4o-2024-08-06)


## Repo Directories
---

- dataset_sample: Sample image dataset of signal representations.

- clip
    - vlm: Scripts for training/inferencing VLM models.
    - results_sample: Sample inference results of vanilla/trained VLM models.
- llava
    - vlm: Scripts for training/inferencing VLM models.
    - results_sample: Sample inference results of vanilla/trained VLM models.
- GPT-4o
    - vlm: Scripts for training/inferencing VLM models.
    - results_sample: Sample inference results of vanilla/trained VLM models.
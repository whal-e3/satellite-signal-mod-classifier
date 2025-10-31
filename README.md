# Satellite AMC (Automatic Modulation Classifier) Identifier 

The repo contains attempts on creating VLM model that does Satellite AMC.
The attempts includes both vanilla use and fine-tuning on various VLM AI models.
The repo also includes the performance results of the models.

## Modulation List
The modulation schemes that are used are like below.

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

- dataset_sample: Sample image dataset of signal representations.

- vlm-finetune: Scripts for running/training VLM models.
    - CLIP
    - LLAVA
    - GPT-4o

- results_sample: Sample inference results of vanilla/trained VLM models.
    - CLIP
    - LLAVA
    - GPT-4o
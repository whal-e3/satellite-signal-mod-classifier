# Satellite AMC (Automatic Modulation Classifier) using VLM models

The repo contains attempts on creating VLM model that does Satellite AMC.
The attempts includes both vanilla use and fine-tuning on various VLM AI models.
The repo also includes the performance results of the models.

## Modulation List
The modulation schemes that are used are like below.

- WBFM, NBFM, BPSK, QPSK, 8PSK, 16APSK, 32APSK, GMSK, CW, CSS, BFSK, GFSK

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

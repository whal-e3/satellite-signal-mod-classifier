# CLIP (clip-vit-base-patch32)

In order to train/run the model, you'll need a GPU that has the capacity.

(I used RTX A5000 for the CLIP model.) 

## vlm-run

training/inference scripts.

- clip-vanilla
- clip-lora
- clip-backbone-freeze
- best-result-settings

* infer.py: inference script. Make sure to change the name of LoRA Adapter or Pretrained Model in to your Adapter/Model.
* train-xx.py: fine-tuning script. xx is the lowest SNR that's used for the fine-tuning (e.g. train-5db.py -> 5db ~ 20db is used.). Change the name of the Adapter/Model to any name that suits you.

## result

Output from the inferences

- vanilla
- lora-fine-tuning
- full-fine-tuning
- best

Use scripts from "confusion_matrix-generator to create confusion_matrix.png from raw result.txt.

## confusion_matrix-generator

With the result you got from inferencing, run the scripts in this order to get the confusion matrix.

Before running the scripts, check if the file names matches your file name.

1. reformat_raw_result.py
2. generate_csv.py
3. generate_confusionMatrix.py

If you also want to check out the evaluation matrics (accuracy, precision, recall, F1) run "generate_metrics.py".
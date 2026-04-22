# CSE4078 Spring 2026 Term Project

This repository contains the codebase for evaluating and fine-tuning small open-source LLMs in the 0.3B to 4B parameter range for the Turkish language, applied to the `Renicames/turkish-law-chatbot` dataset.

## Project Structure

```
.
├── src/
│   ├── data_prep.py   # Dataset loading and prompt formatting
│   ├── evaluate.py    # Inference script for baseline and finetuned comparison
│   ├── train.py       # Fine-tuning script using QLoRA and SFTTrainer
├── tests/             # Pytest directory for basic structure testing
├── requirements.txt   # Dependencies
└── README.md
```

## Setup Instructions

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. (Optional) Run tests to ensure basic logic is intact.
   ```bash
   pytest tests/
   ```

## Workflow & Usage

### 1. Baseline Evaluation
Prior to training, select two small open-source LLMs (0.3B to 4B params) and evaluate them on our common test set. Note that `--load_in_4bit` can be used if running on limited VRAM.

```bash
python -m src.evaluate --model_name "YOUR_SELECTED_MODEL_ID_1"
python -m src.evaluate --model_name "YOUR_SELECTED_MODEL_ID_2"
```

### 2. Fine-Tuning
After selecting the best performing model from the baseline evaluation, train it using Supervised Fine-Tuning (SFTTrainer) + PEFT/QLoRA framework over the `train` dataset constraint split only. *The test set is completely unseen.*

```bash
python -m src.train \
    --model_name "YOUR_SELECTED_MODEL_ID" \
    --output_dir "models/fine_tuned" \
    --epochs 3 \
    --batch_size 4
```

### 3. Post-Training Evaluation
Test the final adapted model over the exact same test dataset.

```bash
python -m src.evaluate \
    --model_name "YOUR_SELECTED_MODEL_ID" \
    --adapter_path "models/fine_tuned/final_model" \
    --sample_size 1500
```

## Note on Resources
4-bit quantization reduces memory drastically, pushing 4B parameters well under strict GPU memory limits. The dataset split rule logic (`train` strictly mapping to LoRA fit algorithm evaluation subset isolation) executes in `src/train.py`.

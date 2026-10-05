# Dialogue Text Summarization Model

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97_Hugging_Face-Transformers-orange)

## Overview

This repository contains a sequence-to-sequence pipeline for generating concise summaries of multi-turn dialogue. The project fine-tunes a BART model from Hugging Face Transformers on the Pulse Quest dialogue-summary dataset.

## Training and optimization

The training workflow uses the `Seq2SeqTrainer` API with:

- Mixed-precision FP16 training to reduce GPU memory usage and training time.
- Gradient accumulation, weight decay, and warm-up steps for stable optimization.
- A reproducible train-validation split using seed `42`.
- ROUGE-based evaluation during training.

## Advanced inference

The inference pipeline is designed to produce readable summaries while limiting repetition:

- **6-beam search** selects a high-probability summary from multiple candidate sequences.
- **3-gram repetition blocking** reduces repeated phrases.
- Configured maximum and minimum summary lengths help keep outputs concise.

## Performance

The best recorded validation result is shown below. Exact results can vary with the GPU, CUDA version, and library versions used during training.

| Metric | Score |
| :--- | :--- |
| **ROUGE-L (Validation)** | **56.43** |

## Repository structure

```text
├── pulse-quest-dataset/
│   ├── train.csv
│   ├── train_2.csv
│   ├── train_3.csv
│   └── test_2.csv
├── pq_submission.ipynb   # Final training, evaluation, and inference workflow
├── certificate.pdf       # Pulse Quest achievement certificate
└── README.md             # Project documentation
```

## How to run

1. Use Python 3.8 or later with a CUDA-enabled GPU recommended.
2. Install the notebook dependencies in your environment:

   ```bash
   pip install transformers[torch] datasets evaluate rouge_score nltk numpy pandas tqdm
   ```

3. Open [`pq_submission.ipynb`](./pq_submission.ipynb) in Jupyter or VS Code and run the cells in order.

The notebook was prepared for Kaggle and searches for the dataset under `/kaggle/input`. To run it locally, update the `get_path` helper to point to `pulse-quest-dataset`, or upload that folder as a Kaggle dataset.

## Dataset format

The training CSV files contain `dialogue` and `summary` columns. The test CSV contains a `dialogue` column. The notebook removes incomplete rows, removes duplicate dialogues, tokenizes the text, and evaluates generated summaries with ROUGE.

The base model is `linydub/bart-large-samsum`. Model weights and evaluation resources are downloaded when the notebook is executed.

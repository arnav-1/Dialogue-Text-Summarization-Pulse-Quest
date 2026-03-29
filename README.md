# 💬 Dialogue Text Summarization Model

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97_Hugging_Face-Transformers-orange)

## 📌 Overview
This repository contains a sequence-to-sequence machine learning pipeline designed to generate highly accurate summaries of complex, multi-turn conversational dialogue. The project leverages the **Hugging Face Transformers** library to fine-tune a **BART** (Bidirectional and Auto-Regressive Transformers) model, adapting it specifically for the nuances of spoken-word transcripts.

## ⚙️ Training & Optimization Architecture
The training pipeline was engineered using the `Seq2SeqTrainer` API, focusing heavily on computational efficiency and model generalization:
* **Mixed-Precision Training (FP16):** Implemented to significantly accelerate training times and reduce GPU memory footprint without sacrificing gradient accuracy.
* **Regularization:** Utilized label smoothing and weight decay to heavily penalize overconfidence and prevent the model from overfitting to the training dialogue dataset.

## 🧠 Advanced Inference Pipeline
Generative models frequently suffer from "looping" or repeating phrases. To ensure high-quality, human-readable summaries, the inference generation step utilizes a custom pipeline:
* **5-Beam Search:** Explores multiple sequence probabilities simultaneously to calculate and select the most logically coherent final summary.
* **Algorithmic Post-Processing:** Engineered custom n-gram penalty algorithms to actively detect and eliminate repetitive word sequences during generation.

## 📊 Performance Metrics
The model was evaluated using standard automated summarization metrics, achieving highly competitive results on the validation set.

| Metric | Score |
| :--- | :--- |
| **ROUGE-L (Validation)** | **56.43** |

## 📂 Repository Structure
```text
├── Best_Model.ipynb           # Finalized training, optimization, and inference pipeline
├── Second_Best_Model.ipynb    # Alternative model architecture and hyperparameter experimentation
└── README.md                  # Project documentation
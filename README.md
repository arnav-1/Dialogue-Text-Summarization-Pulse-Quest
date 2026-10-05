# Dialogue Text Summarization

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-FFD21E?logo=huggingface&logoColor=111111)](https://huggingface.co/docs/transformers)

Fine-tuning experiments for generating concise summaries from multi-turn dialogue. The notebooks use a BART sequence-to-sequence model from Hugging Face and the `Seq2SeqTrainer` API.

## Results

The best recorded validation result was **ROUGE-L: 56.43**. This value comes from the original experiment and is not automatically reproduced on every machine; training requires a compatible GPU and can take substantial time.

The training setup includes:

- BART tokenization with truncated dialogue and summary sequences.
- A reproducible train/validation split using seed `42`.
- Mixed-precision training when supported by the available GPU.
- Beam-search generation and ROUGE evaluation.

## Repository contents

| Path | Description |
| --- | --- |
| `Best_Model.ipynb` | Final training, evaluation, and inference workflow. |
| `Second_Best_Model.ipynb` | Alternative experiment with early stopping and a larger validation split. |
| `train.csv`, `train_2.csv`, `train_3.csv` | Training dialogue-summary data. |
| `test_2.csv` | Test dialogues used by the final workflow. |
| `requirements.txt` | Python dependencies for running the notebooks. |
| `Arnav Jaiswal Certificate - Pulse Quest 1st Place Final.pdf` | Project achievement certificate. |

## Quick start

1. Create and activate a Python 3.9+ environment.
2. Install the dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

3. Open `Best_Model.ipynb` in Jupyter or VS Code and run the cells in order.

`Second_Best_Model.ipynb` expects the CSV files in the notebook working directory. `Best_Model.ipynb` was prepared for Kaggle and searches under `/kaggle/input`; when running locally, either upload the data to a matching Kaggle dataset or update its `get_path` helper to point to the repository directory.

## Data format

Training files must contain:

- `dialogue`: the source conversation.
- `summary`: the reference summary.

The test file must contain a `dialogue` column. The notebooks remove incomplete rows and deduplicate training dialogues before tokenization.

## Reproducibility notes

- The base model is `linydub/bart-large-samsum`.
- Results depend on the Transformers, PyTorch, CUDA, and GPU versions.
- The notebooks download model weights and evaluation resources at runtime.
- Generated checkpoints are written to the notebook runtime directory and are intentionally excluded from version control.
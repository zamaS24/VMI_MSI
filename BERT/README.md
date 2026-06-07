# CamemBERT Text Classification Pipeline

This directory contains a complete CamemBERT pipeline for binary text classification:

- `femme -> 0`
- `homme -> 1`

The data is assumed to be preprocessed before it reaches this project. The scripts read the existing `train`, `val`, and `test` folders, split long texts into 512-token CamemBERT chunks, fine-tune `camembert-base`, evaluate the resulting classifier, and generate LIME/SHAP explanations.

## Architecture

```text
BERT/
├── config.py                  # Defaults, paths, labels, dataclass configs
├── data_loader.py             # Dataset loading and 512-token chunking
├── model.py                   # CamemBERT model/tokenizer helpers
├── train.py                   # Fine-tuning pipeline
├── evaluate.py                # Test-set evaluation
├── explain_lime.py            # LIME explanations
├── explain_shap.py            # SHAP explanations
├── experiment.py              # Plugin-style explainability experiments
├── utils.py                   # Metrics, plots, JSON, reproducibility
├── requirements.txt
├── README.md
├── outputs/
│   ├── checkpoints/
│   ├── models/
│   └── logs/
├── artifacts/
└── vis/
```

## Installation

Create an environment with Python 3.10 or newer, then install dependencies:

```bash
pip install -r BERT/requirements.txt
```

CamemBERT is downloaded from HuggingFace the first time it is used. GPU is used automatically when CUDA is available.

## Dataset Format

Default dataset path:

```text
data/datasetSujet3/content/dataset
├── train/
├── val/
└── test/
```

Each split may contain nested `.txt` files. Labels are inferred from the existing project filename convention:

- fourth parenthesized field equal to `1` means `homme`
- fourth parenthesized field equal to `2` means `femme`

As a fallback, labels can also be inferred from parent folders named `homme` or `femme`.

## Chunking

Texts are tokenized with the CamemBERT tokenizer. Long documents are split into non-overlapping chunks whose final encoded length is 512 tokens including special tokens. Each chunk inherits the original document label.

Training metrics are computed over chunks. Evaluation and explainability use document-level predictions by averaging probabilities over all chunks from the same document.

## Training

```bash
python BERT/train.py --data_dir data/datasetSujet3/content/dataset
```

Useful options:

```bash
python BERT/train.py \
  --batch_size 4 \
  --eval_batch_size 8 \
  --epochs 5 \
  --learning_rate 2e-5 \
  --patience 2
```

Training includes:

- fixed random seeds
- AdamW
- linear warmup/decay scheduler
- gradient clipping
- early stopping on validation loss
- per-epoch checkpoints
- best model selection

Outputs:

```text
BERT/artifacts/metrics.json
BERT/outputs/logs/history.csv
BERT/outputs/models/best_model/
BERT/outputs/checkpoints/epoch_XX/
```

## Evaluation

```bash
python BERT/evaluate.py --data_dir data/datasetSujet3/content/dataset
```

Computes:

- accuracy
- precision
- recall
- F1-score
- confusion matrix
- ROC-AUC

Outputs:

```text
BERT/artifacts/test_predictions.csv
BERT/artifacts/metrics.json
BERT/vis/confusion_matrix.png
BERT/vis/roc_curve.png
```

## LIME Explanations

```bash
python BERT/explain_lime.py --data_dir data/datasetSujet3/content/dataset --n_examples 50
```

Outputs:

```text
BERT/artifacts/lime_results.csv
BERT/vis/lime_local_explanation.html
BERT/vis/lime_top_femme_terms.png
BERT/vis/lime_top_homme_terms.png
```

LIME calls the same document-level prediction function used in evaluation, including 512-token chunking and probability averaging.

## SHAP Explanations

```bash
python BERT/explain_shap.py --data_dir data/datasetSujet3/content/dataset --n_examples 20
```

Outputs:

```text
BERT/artifacts/shap_global.csv
BERT/artifacts/shap_local.csv
BERT/vis/shap_summary.png
BERT/vis/shap_local_explanation.png
```

SHAP uses a HuggingFace-compatible tokenizer masker and the same chunk-aware predictor used by the rest of the pipeline.

## Experiments

The experiment runner compares model variants and explainability methods using a plugin registry.

Raw pretrained CamemBERT with LIME:

```bash
python BERT/experiment.py --model pretrained --method lime
```

Fine-tuned CamemBERT with SHAP:

```bash
python BERT/experiment.py --model finetuned --method shap
```

Experiment rows are appended to:

```text
BERT/artifacts/experiment_results.csv
```

Columns include:

- model type
- explainability method
- predicted label
- confidence
- important tokens
- explanation scores

Future methods such as Integrated Gradients, LRP, attention rollout, or Captum methods can be added by implementing a new `ExplanationPlugin` and registering it in `PLUGIN_REGISTRY`.

## Notes

The explainability outputs describe what the trained classifier learned from this dataset. They should not be interpreted as universal rules about male or female writing.

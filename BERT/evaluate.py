"""Evaluate a trained CamemBERT classifier on the test split."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix

from config import ID2LABEL, PathConfig
from data_loader import documents_to_frame, load_split
from model import load_model_and_tokenizer, predict_proba_for_texts
from utils import compute_metrics, load_json, plot_confusion_matrix, plot_roc_curve, save_json


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    paths = PathConfig()
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned CamemBERT classifier.")
    parser.add_argument("--data_dir", "--data-dir", type=Path, default=paths.data_dir)
    parser.add_argument("--model_dir", "--model-dir", type=Path, default=paths.best_model_dir)
    parser.add_argument("--artifact_dir", "--artifact-dir", type=Path, default=paths.artifact_dir)
    parser.add_argument("--vis_dir", "--vis-dir", type=Path, default=paths.vis_dir)
    parser.add_argument("--max_length", "--max-length", type=int, default=512)
    parser.add_argument("--batch_size", "--batch-size", type=int, default=8)
    return parser.parse_args()


def build_predictions_frame(
    documents: list[Any],
    probabilities: np.ndarray,
) -> pd.DataFrame:
    """Create a document-level predictions DataFrame."""
    frame = documents_to_frame(documents)
    predictions = probabilities.argmax(axis=1)
    confidences = probabilities.max(axis=1)
    frame["predicted_label_id"] = predictions
    frame["predicted_label"] = [ID2LABEL[int(idx)] for idx in predictions]
    frame["confidence"] = confidences
    frame["prob_femme"] = probabilities[:, 0]
    frame["prob_homme"] = probabilities[:, 1]
    frame["correct"] = frame["label_id"] == frame["predicted_label_id"]
    return frame[
        [
            "path",
            "label",
            "label_id",
            "predicted_label",
            "predicted_label_id",
            "confidence",
            "prob_femme",
            "prob_homme",
            "correct",
        ]
    ]


def merge_metrics(metrics_path: Path, test_metrics: dict[str, Any]) -> None:
    """Merge test metrics into the training metrics file when it exists."""
    if metrics_path.exists():
        metrics = load_json(metrics_path)
    else:
        metrics = {}
    metrics["test"] = test_metrics
    save_json(metrics, metrics_path)


def main() -> None:
    """Run test-set evaluation."""
    args = parse_args()
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    args.vis_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model_and_tokenizer(args.model_dir, device)

    test_docs = load_split(args.data_dir, "test")
    texts = [doc.text for doc in test_docs]
    y_true = np.asarray([doc.label_id for doc in test_docs], dtype=np.int64)

    probabilities = predict_proba_for_texts(
        model,
        tokenizer,
        texts,
        device,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )
    y_pred = probabilities.argmax(axis=1)
    y_score = probabilities[:, 1]

    predictions = build_predictions_frame(test_docs, probabilities)
    predictions_path = args.artifact_dir / "test_predictions.csv"
    predictions.to_csv(predictions_path, index=False, encoding="utf-8")

    target_names = [ID2LABEL[0], ID2LABEL[1]]
    test_metrics = compute_metrics(y_true, y_pred, y_score=y_score, target_names=target_names)
    merge_metrics(args.artifact_dir / "metrics.json", test_metrics)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plot_confusion_matrix(cm, target_names, args.vis_dir / "confusion_matrix.png")
    if len(set(y_true.tolist())) > 1:
        plot_roc_curve(y_true, y_score, args.vis_dir / "roc_curve.png")

    print(f"Saved predictions: {predictions_path}")
    print(f"Saved metrics: {args.artifact_dir / 'metrics.json'}")
    print(f"Saved confusion matrix: {args.vis_dir / 'confusion_matrix.png'}")
    print(f"Saved ROC curve: {args.vis_dir / 'roc_curve.png'}")


if __name__ == "__main__":
    main()

"""Experiment runner for comparing explainability methods and model variants."""

from __future__ import annotations

import argparse
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch

from config import ID2LABEL, ModelConfig, PathConfig
from data_loader import load_split
from model import create_model, create_tokenizer, load_model_and_tokenizer, predict_proba_for_texts


@dataclass
class ExplanationResult:
    """Standard experiment output for one explained document."""

    file_path: str
    true_label: str
    predicted_label: str
    confidence: float
    important_tokens: list[str]
    explanation_scores: list[float]


class ExplanationPlugin(ABC):
    """Base class for explainability plugins."""

    name: str

    @abstractmethod
    def explain(
        self,
        texts: list[str],
        documents: list[object],
        probabilities: np.ndarray,
        predictor: Callable[[list[str]], np.ndarray],
        tokenizer: object,
        n_terms: int,
    ) -> list[ExplanationResult]:
        """Explain a list of texts."""


class LimePlugin(ExplanationPlugin):
    """LIME experiment plugin."""

    name = "lime"

    def explain(
        self,
        texts: list[str],
        documents: list[object],
        probabilities: np.ndarray,
        predictor: Callable[[list[str]], np.ndarray],
        tokenizer: object,
        n_terms: int,
    ) -> list[ExplanationResult]:
        from lime.lime_text import LimeTextExplainer

        explainer = LimeTextExplainer(class_names=[ID2LABEL[0], ID2LABEL[1]])
        results: list[ExplanationResult] = []

        for index, document in enumerate(documents):
            predicted_id = int(probabilities[index].argmax())
            explanation = explainer.explain_instance(
                texts[index],
                predictor,
                labels=[predicted_id],
                num_features=n_terms,
            )
            token_scores = explanation.as_list(label=predicted_id)
            results.append(
                ExplanationResult(
                    file_path=document.path,
                    true_label=document.label,
                    predicted_label=ID2LABEL[predicted_id],
                    confidence=float(probabilities[index, predicted_id]),
                    important_tokens=[token for token, _ in token_scores],
                    explanation_scores=[float(score) for _, score in token_scores],
                )
            )

        return results


class ShapPlugin(ExplanationPlugin):
    """SHAP experiment plugin."""

    name = "shap"

    def explain(
        self,
        texts: list[str],
        documents: list[object],
        probabilities: np.ndarray,
        predictor: Callable[[list[str]], np.ndarray],
        tokenizer: object,
        n_terms: int,
    ) -> list[ExplanationResult]:
        import shap

        masker = shap.maskers.Text(tokenizer)
        explainer = shap.Explainer(predictor, masker, output_names=[ID2LABEL[0], ID2LABEL[1]])
        shap_values = explainer(texts)
        results: list[ExplanationResult] = []

        for index, document in enumerate(documents):
            predicted_id = int(probabilities[index].argmax())
            values = shap_values[index].values
            tokens = shap_values[index].data

            if values.ndim == 2:
                scores = values[:, predicted_id]
            else:
                scores = values

            token_scores = [
                (str(token).replace("\n", " ").strip(), float(score))
                for token, score in zip(tokens, scores)
                if str(token).strip()
            ]
            token_scores.sort(key=lambda item: abs(item[1]), reverse=True)
            token_scores = token_scores[:n_terms]

            results.append(
                ExplanationResult(
                    file_path=document.path,
                    true_label=document.label,
                    predicted_label=ID2LABEL[predicted_id],
                    confidence=float(probabilities[index, predicted_id]),
                    important_tokens=[token for token, _ in token_scores],
                    explanation_scores=[float(score) for _, score in token_scores],
                )
            )

        return results


PLUGIN_REGISTRY: dict[str, type[ExplanationPlugin]] = {
    LimePlugin.name: LimePlugin,
    ShapPlugin.name: ShapPlugin,
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    paths = PathConfig()
    model_defaults = ModelConfig()
    parser = argparse.ArgumentParser(description="Run BERT explainability experiments.")
    parser.add_argument("--model", choices=("pretrained", "finetuned"), required=True)
    parser.add_argument("--method", choices=tuple(PLUGIN_REGISTRY.keys()), required=True)
    parser.add_argument("--data_dir", "--data-dir", type=Path, default=paths.data_dir)
    parser.add_argument("--model_name", "--model-name", default=model_defaults.model_name)
    parser.add_argument("--finetuned_model_dir", "--finetuned-model-dir", type=Path, default=paths.best_model_dir)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--n_examples", "--n-examples", type=int, default=10)
    parser.add_argument("--n_terms", "--n-terms", type=int, default=20)
    parser.add_argument("--max_length", "--max-length", type=int, default=model_defaults.max_length)
    parser.add_argument("--batch_size", "--batch-size", type=int, default=8)
    parser.add_argument("--output_path", "--output-path", type=Path, default=paths.artifact_dir / "experiment_results.csv")
    return parser.parse_args()


def load_experiment_model(args: argparse.Namespace, device: torch.device) -> tuple[torch.nn.Module, object]:
    """Load either raw pretrained or fine-tuned CamemBERT."""
    if args.model == "finetuned":
        return load_model_and_tokenizer(args.finetuned_model_dir, device)

    tokenizer = create_tokenizer(args.model_name)
    model = create_model(args.model_name).to(device)
    model.eval()
    return model, tokenizer


def make_predictor(
    model: torch.nn.Module,
    tokenizer: object,
    device: torch.device,
    max_length: int,
    batch_size: int,
) -> Callable[[list[str]], np.ndarray]:
    """Create a chunk-aware document probability predictor."""

    def predict(texts: list[str]) -> np.ndarray:
        return predict_proba_for_texts(
            model,
            tokenizer,
            list(texts),
            device,
            max_length=max_length,
            batch_size=batch_size,
        )

    return predict


def results_to_frame(
    model_type: str,
    method: str,
    results: list[ExplanationResult],
) -> pd.DataFrame:
    """Convert experiment results to the required CSV schema."""
    rows = []
    for result in results:
        rows.append(
            {
                "model_type": model_type,
                "explainability_method": method,
                "file_path": result.file_path,
                "true_label": result.true_label,
                "predicted_label": result.predicted_label,
                "confidence": result.confidence,
                "important_tokens": json.dumps(result.important_tokens, ensure_ascii=False),
                "explanation_scores": json.dumps(result.explanation_scores, ensure_ascii=False),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    """Run one experiment and append its rows to experiment_results.csv."""
    args = parse_args()
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_experiment_model(args, device)
    documents = load_split(args.data_dir, args.split)[: args.n_examples]
    texts = [document.text for document in documents]
    predictor = make_predictor(model, tokenizer, device, args.max_length, args.batch_size)
    probabilities = predictor(texts)

    plugin = PLUGIN_REGISTRY[args.method]()
    results = plugin.explain(texts, documents, probabilities, predictor, tokenizer, args.n_terms)
    frame = results_to_frame(args.model, args.method, results)

    if args.output_path.exists():
        previous = pd.read_csv(args.output_path)
        frame = pd.concat([previous, frame], ignore_index=True)

    frame.to_csv(args.output_path, index=False, encoding="utf-8")
    print(f"Saved experiment results: {args.output_path}")


if __name__ == "__main__":
    main()

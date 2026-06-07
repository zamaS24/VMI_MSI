"""LIME explanations for the CamemBERT classifier."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from lime.lime_text import LimeTextExplainer

from config import ID2LABEL, ModelConfig, PathConfig, TrainingConfig
from data_loader import describe_chunk_selection, load_split
from model import load_model_and_tokenizer, predict_proba_for_texts


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    paths = PathConfig()
    model_defaults = ModelConfig()
    train_defaults = TrainingConfig()
    parser = argparse.ArgumentParser(description="Generate LIME explanations for CamemBERT.")
    parser.add_argument("--data_dir", "--data-dir", type=Path, default=paths.data_dir)
    parser.add_argument("--model_dir", "--model-dir", type=Path, default=paths.best_model_dir)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--n_examples", "--n-examples", type=int, default=50)
    parser.add_argument("--n_terms", "--n-terms", type=int, default=20)
    parser.add_argument("--max_length", "--max-length", type=int, default=model_defaults.max_length)
    parser.add_argument("--num_chunks", "--num-chunks", type=int, default=model_defaults.num_chunks)
    parser.add_argument("--seed", type=int, default=train_defaults.seed)
    parser.add_argument("--batch_size", "--batch-size", type=int, default=8)
    parser.add_argument("--artifact_dir", "--artifact-dir", type=Path, default=paths.artifact_dir)
    parser.add_argument("--vis_dir", "--vis-dir", type=Path, default=paths.vis_dir)
    return parser.parse_args()


def make_predictor(
    model: torch.nn.Module,
    tokenizer: object,
    device: torch.device,
    max_length: int,
    num_chunks: int | None,
    seed: int,
    batch_size: int,
) -> Callable[[list[str]], np.ndarray]:
    """Create a probability function compatible with LIME."""

    def predict(texts: list[str]) -> np.ndarray:
        return predict_proba_for_texts(
            model,
            tokenizer,
            list(texts),
            device,
            max_length=max_length,
            num_chunks=num_chunks,
            seed=seed,
            batch_size=batch_size,
        )

    return predict


def save_top_terms_plot(rows: pd.DataFrame, class_label: str, output_path: Path, n_terms: int) -> None:
    """Save a horizontal bar chart of aggregated positive LIME terms."""
    subset = rows[(rows["explained_class"] == class_label) & (rows["score"] > 0)]
    grouped = (
        subset.groupby("token", as_index=False)["score"]
        .sum()
        .sort_values("score", ascending=False)
        .head(n_terms)
    )

    plt.figure(figsize=(10, 6))
    if grouped.empty:
        plt.title(f"No positive LIME terms for {class_label}")
        plt.axis("off")
    else:
        plt.barh(grouped["token"].iloc[::-1], grouped["score"].iloc[::-1])
        plt.xlabel("Aggregated LIME score")
        plt.title(f"Top LIME terms supporting {class_label}")
        plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def main() -> None:
    """Generate local and aggregate LIME explanations."""
    args = parse_args()
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    args.vis_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model_and_tokenizer(args.model_dir, device)
    print(describe_chunk_selection(args.num_chunks))
    documents = load_split(args.data_dir, args.split)[: args.n_examples]
    predictor = make_predictor(
        model,
        tokenizer,
        device,
        args.max_length,
        args.num_chunks,
        args.seed,
        args.batch_size,
    )
    explainer = LimeTextExplainer(class_names=[ID2LABEL[0], ID2LABEL[1]])

    rows: list[dict[str, object]] = []
    class_counts: dict[str, int] = defaultdict(int)

    for doc_index, document in enumerate(documents):
        probabilities = predictor([document.text])[0]
        predicted_id = int(probabilities.argmax())
        predicted_label = ID2LABEL[predicted_id]
        class_counts[predicted_label] += 1

        explanation = explainer.explain_instance(
            document.text,
            predictor,
            labels=[predicted_id],
            num_features=args.n_terms,
        )

        if doc_index == 0:
            explanation.save_to_file(str(args.vis_dir / "lime_local_explanation.html"))

        for rank, (token, score) in enumerate(explanation.as_list(label=predicted_id), start=1):
            rows.append(
                {
                    "file_path": document.path,
                    "true_label": document.label,
                    "predicted_label": predicted_label,
                    "confidence": float(probabilities[predicted_id]),
                    "explained_class": predicted_label,
                    "rank": rank,
                    "token": token,
                    "score": float(score),
                }
            )

    results = pd.DataFrame(
        rows,
        columns=[
            "file_path",
            "true_label",
            "predicted_label",
            "confidence",
            "explained_class",
            "rank",
            "token",
            "score",
        ],
    )
    results_path = args.artifact_dir / "lime_results.csv"
    results.to_csv(results_path, index=False, encoding="utf-8")

    save_top_terms_plot(results, "femme", args.vis_dir / "lime_top_femme_terms.png", args.n_terms)
    save_top_terms_plot(results, "homme", args.vis_dir / "lime_top_homme_terms.png", args.n_terms)

    print(f"Saved LIME rows: {results_path}")
    print(f"Saved local explanation: {args.vis_dir / 'lime_local_explanation.html'}")
    print(f"Saved femme terms: {args.vis_dir / 'lime_top_femme_terms.png'}")
    print(f"Saved homme terms: {args.vis_dir / 'lime_top_homme_terms.png'}")


if __name__ == "__main__":
    main()

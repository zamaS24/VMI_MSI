"""SHAP explanations for the CamemBERT classifier."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch

from config import ID2LABEL, ModelConfig, PathConfig, TrainingConfig
from data_loader import describe_chunk_selection, format_chunk_sampling_stats, load_split
from model import load_model_and_tokenizer, predict_proba_for_texts


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    paths = PathConfig()
    model_defaults = ModelConfig()
    train_defaults = TrainingConfig()
    parser = argparse.ArgumentParser(description="Generate SHAP explanations for CamemBERT.")
    parser.add_argument("--data_dir", "--data-dir", type=Path, default=paths.data_dir)
    parser.add_argument("--model_dir", "--model-dir", type=Path, default=paths.best_model_dir)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--n_examples", "--n-examples", type=int, default=20)
    parser.add_argument("--n_terms", "--n-terms", type=int, default=20)
    parser.add_argument("--max_length", "--max-length", type=int, default=model_defaults.max_length)
    parser.add_argument("--num_chunks_homme", "--num-chunks-homme", type=int, default=model_defaults.num_chunks_homme)
    parser.add_argument("--num_chunks_femme", "--num-chunks-femme", type=int, default=model_defaults.num_chunks_femme)
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
    num_chunks_homme: int | None,
    num_chunks_femme: int | None,
    seed: int,
    batch_size: int,
    label: str | None = None,
    sample_key: str | None = None,
) -> Callable[[list[str]], np.ndarray]:
    """Create a probability function compatible with SHAP."""

    def predict(texts: list[str]) -> np.ndarray:
        labels = [label] * len(texts) if label is not None else None
        sample_keys = [sample_key] * len(texts) if sample_key is not None else None
        return predict_proba_for_texts(
            model,
            tokenizer,
            list(texts),
            device,
            max_length=max_length,
            labels=labels,
            sample_keys=sample_keys,
            num_chunks_homme=num_chunks_homme,
            num_chunks_femme=num_chunks_femme,
            seed=seed,
            batch_size=batch_size,
        )

    return predict


def clean_token(token: object) -> str:
    """Convert SHAP token objects to compact strings."""
    return str(token).replace("\n", " ").strip()


def shap_rows(
    shap_values: object,
    documents: list[object],
    probabilities: np.ndarray,
    n_terms: int,
) -> pd.DataFrame:
    """Convert SHAP explanations into a long DataFrame."""
    rows: list[dict[str, object]] = []

    for doc_index, document in enumerate(documents):
        predicted_id = int(probabilities[doc_index].argmax())
        predicted_label = ID2LABEL[predicted_id]
        values = shap_values[doc_index].values
        tokens = shap_values[doc_index].data

        if values.ndim == 2:
            class_scores = values[:, predicted_id]
        else:
            class_scores = values

        token_scores = [
            (clean_token(token), float(score))
            for token, score in zip(tokens, class_scores)
            if clean_token(token)
        ]
        token_scores.sort(key=lambda item: abs(item[1]), reverse=True)

        for rank, (token, score) in enumerate(token_scores[:n_terms], start=1):
            rows.append(
                {
                    "file_path": document.path,
                    "true_label": document.label,
                    "predicted_label": predicted_label,
                    "confidence": float(probabilities[doc_index, predicted_id]),
                    "explained_class": predicted_label,
                    "rank": rank,
                    "token": token,
                    "score": score,
                    "absolute_score": abs(score),
                }
            )

    return pd.DataFrame(
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
            "absolute_score",
        ],
    )


def aggregate_global(local_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate local SHAP scores by class and token."""
    if local_df.empty:
        return pd.DataFrame(
            columns=["explained_class", "token", "aggregate_score", "mean_abs_score", "count"]
        )

    grouped = (
        local_df.groupby(["explained_class", "token"], as_index=False)
        .agg(
            aggregate_score=("score", "sum"),
            mean_abs_score=("absolute_score", "mean"),
            count=("score", "size"),
        )
        .sort_values("mean_abs_score", ascending=False)
    )
    return grouped


def save_local_plot(local_df: pd.DataFrame, output_path: Path, n_terms: int) -> None:
    """Save a bar plot for the first local SHAP explanation."""
    plt.figure(figsize=(10, 6))
    if local_df.empty:
        plt.title("No SHAP local explanation available")
        plt.axis("off")
    else:
        first_path = local_df.iloc[0]["file_path"]
        subset = local_df[local_df["file_path"] == first_path].head(n_terms).iloc[::-1]
        colors = ["#4C78A8" if value >= 0 else "#E45756" for value in subset["score"]]
        plt.barh(subset["token"], subset["score"], color=colors)
        plt.xlabel("SHAP score")
        plt.title("Local SHAP explanation")
        plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_summary_plot(global_df: pd.DataFrame, output_path: Path, n_terms: int) -> None:
    """Save a global SHAP summary bar chart."""
    plt.figure(figsize=(10, 6))
    if global_df.empty:
        plt.title("No SHAP global explanation available")
        plt.axis("off")
    else:
        subset = global_df.sort_values("mean_abs_score", ascending=False).head(n_terms).iloc[::-1]
        labels = subset["explained_class"] + ": " + subset["token"]
        plt.barh(labels, subset["mean_abs_score"])
        plt.xlabel("Mean absolute SHAP score")
        plt.title("Global SHAP summary")
        plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def main() -> None:
    """Generate SHAP local and global explanations."""
    args = parse_args()
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    args.vis_dir.mkdir(parents=True, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tokenizer = load_model_and_tokenizer(args.model_dir, device)
    print(describe_chunk_selection(args.num_chunks_homme, args.num_chunks_femme))
    documents = load_split(args.data_dir, args.split)[: args.n_examples]
    texts = [document.text for document in documents]
    labels = [document.label for document in documents]
    sample_keys = [document.path for document in documents]
    probabilities, chunk_stats = predict_proba_for_texts(
        model,
        tokenizer,
        texts,
        device,
        max_length=args.max_length,
        labels=labels,
        sample_keys=sample_keys,
        num_chunks_homme=args.num_chunks_homme,
        num_chunks_femme=args.num_chunks_femme,
        seed=args.seed,
        batch_size=args.batch_size,
        return_chunk_stats=True,
    )
    print(format_chunk_sampling_stats(args.split.capitalize(), chunk_stats))

    masker = shap.maskers.Text(tokenizer)
    local_frames: list[pd.DataFrame] = []
    for index, document in enumerate(documents):
        predictor = make_predictor(
            model,
            tokenizer,
            device,
            args.max_length,
            args.num_chunks_homme,
            args.num_chunks_femme,
            args.seed,
            args.batch_size,
            label=document.label,
            sample_key=document.path,
        )
        explainer = shap.Explainer(predictor, masker, output_names=[ID2LABEL[0], ID2LABEL[1]])
        values = explainer([document.text])
        local_frames.append(
            shap_rows(values, [document], probabilities[index:index + 1], args.n_terms)
        )

    local_df = pd.concat(local_frames, ignore_index=True) if local_frames else shap_rows([], [], probabilities, args.n_terms)
    global_df = aggregate_global(local_df)

    local_path = args.artifact_dir / "shap_local.csv"
    global_path = args.artifact_dir / "shap_global.csv"
    local_df.to_csv(local_path, index=False, encoding="utf-8")
    global_df.to_csv(global_path, index=False, encoding="utf-8")

    save_summary_plot(global_df, args.vis_dir / "shap_summary.png", args.n_terms)
    save_local_plot(local_df, args.vis_dir / "shap_local_explanation.png", args.n_terms)

    print(f"Saved SHAP local rows: {local_path}")
    print(f"Saved SHAP global rows: {global_path}")
    print(f"Saved SHAP summary: {args.vis_dir / 'shap_summary.png'}")
    print(f"Saved SHAP local plot: {args.vis_dir / 'shap_local_explanation.png'}")


if __name__ == "__main__":
    main()

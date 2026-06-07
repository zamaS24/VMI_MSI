"""CamemBERT model helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoTokenizer, CamembertForSequenceClassification

from config import ID2LABEL, LABEL2ID
from data_loader import (
    ChunkSamplingStats,
    empty_chunk_sampling_stats,
    encode_text_chunks,
    select_chunks,
)


def create_tokenizer(model_name_or_path: str | Path) -> Any:
    """Load a CamemBERT tokenizer."""
    return AutoTokenizer.from_pretrained(str(model_name_or_path), use_fast=True)


def create_model(model_name_or_path: str | Path = "camembert-base") -> CamembertForSequenceClassification:
    """Load CamemBERT for binary sequence classification."""
    return CamembertForSequenceClassification.from_pretrained(
        str(model_name_or_path),
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )


def save_model(model: CamembertForSequenceClassification, tokenizer: Any, output_dir: Path | str) -> None:
    """Save a model and tokenizer in HuggingFace format."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)


def load_model_and_tokenizer(model_dir: Path | str, device: torch.device) -> tuple[CamembertForSequenceClassification, Any]:
    """Load a saved model/tokenizer pair and move the model to device."""
    tokenizer = create_tokenizer(model_dir)
    model = create_model(model_dir)
    model.to(device)
    model.eval()
    return model, tokenizer


def batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    """Move model tensor inputs to a device."""
    return {
        "input_ids": batch["input_ids"].to(device),
        "attention_mask": batch["attention_mask"].to(device),
        "labels": batch["labels"].to(device),
    }


def predict_proba_from_arrays(
    model: CamembertForSequenceClassification,
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
    device: torch.device,
    batch_size: int = 8,
) -> np.ndarray:
    """Predict probabilities for already encoded chunks."""
    probabilities: list[np.ndarray] = []
    model.eval()

    with torch.no_grad():
        for start in range(0, len(input_ids), batch_size):
            batch_input_ids = torch.tensor(input_ids[start:start + batch_size], dtype=torch.long).to(device)
            batch_attention = torch.tensor(attention_mask[start:start + batch_size], dtype=torch.long).to(device)
            logits = model(input_ids=batch_input_ids, attention_mask=batch_attention).logits
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())

    return np.vstack(probabilities)


def predict_proba_for_texts(
    model: CamembertForSequenceClassification,
    tokenizer: Any,
    texts: list[str],
    device: torch.device,
    max_length: int = 512,
    labels: list[str] | None = None,
    sample_keys: list[str] | None = None,
    num_chunks_homme: int | None = None,
    num_chunks_femme: int | None = None,
    seed: int = 42,
    batch_size: int = 8,
    return_chunk_stats: bool = False,
) -> np.ndarray | tuple[np.ndarray, ChunkSamplingStats]:
    """Predict document-level probabilities by averaging chunk probabilities."""
    if labels is not None and len(labels) != len(texts):
        raise ValueError("labels must have the same length as texts")
    if sample_keys is not None and len(sample_keys) != len(texts):
        raise ValueError("sample_keys must have the same length as texts")

    document_probs: list[np.ndarray] = []
    chunk_stats = empty_chunk_sampling_stats() if return_chunk_stats else None

    for index, text in enumerate(texts):
        label = labels[index] if labels is not None else None
        sample_key = sample_keys[index] if sample_keys is not None else text
        raw_chunks = encode_text_chunks(text, tokenizer, max_length=max_length)
        chunks = select_chunks(
            raw_chunks,
            label=label,
            num_chunks_homme=num_chunks_homme,
            num_chunks_femme=num_chunks_femme,
            seed=seed,
            sample_key=sample_key,
        )
        if chunk_stats is not None and label is not None:
            chunk_stats.documents[label] = chunk_stats.documents.get(label, 0) + 1
            chunk_stats.chunks_before_sampling[label] = (
                chunk_stats.chunks_before_sampling.get(label, 0) + len(raw_chunks)
            )
            chunk_stats.chunks_after_sampling[label] = (
                chunk_stats.chunks_after_sampling.get(label, 0) + len(chunks)
            )
        if not chunks:
            raise ValueError("A document produced no chunks.")
        input_ids = np.asarray([chunk["input_ids"] for chunk in chunks], dtype=np.int64)
        attention_mask = np.asarray([chunk["attention_mask"] for chunk in chunks], dtype=np.int64)
        chunk_probs = predict_proba_from_arrays(
            model,
            input_ids,
            attention_mask,
            device,
            batch_size=batch_size,
        )
        document_probs.append(chunk_probs.mean(axis=0))

    probabilities = np.vstack(document_probs)
    if return_chunk_stats:
        return probabilities, chunk_stats
    return probabilities

"""Dataset loading and CamemBERT chunking utilities."""

from __future__ import annotations

import hashlib
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from config import LABEL2ID


@dataclass(frozen=True)
class DocumentRecord:
    """One original text document before tokenizer chunking."""

    path: str
    label: str
    label_id: int
    text: str


@dataclass(frozen=True)
class ChunkRecord:
    """One tokenizer chunk inherited from an original document."""

    input_ids: list[int]
    attention_mask: list[int]
    label_id: int
    label: str
    path: str
    document_index: int
    chunk_index: int


@dataclass(frozen=True)
class ChunkSamplingStats:
    """Dataset-level counts before and after label-aware chunk sampling."""

    documents: dict[str, int]
    chunks_before_sampling: dict[str, int]
    chunks_after_sampling: dict[str, int]


def extract_label_from_name(filename: str) -> str | None:
    """Extract labels using the existing dataset filename convention."""
    matches = re.findall(r"\(([^)]*)\)", filename)
    if len(matches) >= 4:
        if matches[3] == "1":
            return "homme"
        if matches[3] == "2":
            return "femme"
    return None


def infer_label_from_path(path: Path) -> str | None:
    """Infer a label from the filename, then from parent folders as fallback."""
    label = extract_label_from_name(path.name)
    if label is not None:
        return label

    lowered_parts = [part.lower() for part in path.parts]
    if "femme" in lowered_parts:
        return "femme"
    if "homme" in lowered_parts:
        return "homme"
    return None


def read_text(path: Path | str) -> str:
    """Read preprocessed text using the same UTF-8 tolerant mode as TF-IDF."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def load_split(data_dir: Path | str, split_name: str) -> list[DocumentRecord]:
    """Load one split as document records."""
    split_dir = Path(data_dir) / split_name
    records: list[DocumentRecord] = []

    for root, _, files in os.walk(split_dir):
        for filename in files:
            if not filename.endswith(".txt"):
                continue
            path = Path(root) / filename
            label = infer_label_from_path(path)
            if label not in LABEL2ID:
                continue
            records.append(
                DocumentRecord(
                    path=str(path),
                    label=label,
                    label_id=LABEL2ID[label],
                    text=read_text(path),
                )
            )

    return records


def load_splits(data_dir: Path | str) -> tuple[list[DocumentRecord], list[DocumentRecord], list[DocumentRecord]]:
    """Load train, validation, and test splits."""
    return (
        load_split(data_dir, "train"),
        load_split(data_dir, "val"),
        load_split(data_dir, "test"),
    )


def documents_to_frame(records: Iterable[DocumentRecord]) -> pd.DataFrame:
    """Convert document records to a DataFrame for reports and exports."""
    return pd.DataFrame([record.__dict__ for record in records])


def chunk_token_ids(input_ids: list[int], chunk_size: int) -> list[list[int]]:
    """Split token ids into non-overlapping chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not input_ids:
        return [[]]
    return [input_ids[start:start + chunk_size] for start in range(0, len(input_ids), chunk_size)]


def _sample_seed(seed: int, sample_key: str | None) -> int:
    """Derive a stable per-document sampling seed."""
    if sample_key is None:
        return seed
    digest = hashlib.blake2b(str(sample_key).encode("utf-8"), digest_size=8).digest()
    return seed + int.from_bytes(digest, byteorder="big", signed=False)


def _validate_chunk_limit(name: str, value: int | None) -> None:
    if value is not None and value < 0:
        raise ValueError(f"{name} must be None or a non-negative integer")


def _chunk_limit_for_label(
    label: str | None,
    num_chunks_homme: int | None,
    num_chunks_femme: int | None,
) -> int | None:
    if label == "homme":
        return num_chunks_homme
    if label == "femme":
        return num_chunks_femme
    return None


def select_chunks(
    chunks: list[dict[str, list[int]]],
    label: str | None = None,
    num_chunks_homme: int | None = None,
    num_chunks_femme: int | None = None,
    seed: int = 42,
    sample_key: str | None = None,
) -> list[dict[str, list[int]]]:
    """Skip the first chunk and optionally sample chunks using the document label."""
    _validate_chunk_limit("num_chunks_homme", num_chunks_homme)
    _validate_chunk_limit("num_chunks_femme", num_chunks_femme)
    if not chunks:
        return []

    remaining_chunks = chunks[1:]
    if not remaining_chunks:
        return chunks[:]

    chunk_limit = _chunk_limit_for_label(label, num_chunks_homme, num_chunks_femme)
    if chunk_limit is None:
        return remaining_chunks
    if len(remaining_chunks) <= chunk_limit:
        return remaining_chunks

    rng = random.Random(_sample_seed(seed, sample_key))
    selected_indices = sorted(rng.sample(range(len(remaining_chunks)), chunk_limit))
    return [remaining_chunks[index] for index in selected_indices]


def _format_chunk_limit(value: int | None) -> str:
    if value is None:
        return "all remaining chunks/document"
    return f"{value} chunks/document"


def describe_chunk_selection(
    num_chunks_homme: int | None,
    num_chunks_femme: int | None,
) -> str:
    """Return a human-readable description of the active chunk selection policy."""
    return "\n".join(
        [
            "Chunk sampling configuration:",
            f"  homme -> {_format_chunk_limit(num_chunks_homme)}",
            f"  femme -> {_format_chunk_limit(num_chunks_femme)}",
            "  first chunk skipped: True",
            "  single-chunk documents kept: True",
        ]
    )


def encode_text_chunks(
    text: str,
    tokenizer: Any,
    max_length: int = 512,
) -> list[dict[str, list[int]]]:
    """Tokenize a document into raw fixed-length CamemBERT-ready chunks."""
    start_token_id = tokenizer.cls_token_id
    if start_token_id is None:
        start_token_id = tokenizer.bos_token_id

    end_token_id = tokenizer.sep_token_id
    if end_token_id is None:
        end_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id

    if start_token_id is None or end_token_id is None or pad_token_id is None:
        raise ValueError("Tokenizer must expose start, end, and pad token ids.")

    payload_size = max_length - 2
    token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]

    chunks: list[dict[str, list[int]]] = []
    for chunk_ids in chunk_token_ids(token_ids, payload_size):
        input_ids = [start_token_id] + chunk_ids[:payload_size] + [end_token_id]
        attention_mask = [1] * len(input_ids)

        padding_length = max_length - len(input_ids)
        input_ids += [pad_token_id] * padding_length
        attention_mask += [0] * padding_length

        chunks.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
        )

    return chunks


def chunk_text(
    text: str,
    tokenizer: Any,
    max_length: int = 512,
    label: str | None = None,
    num_chunks_homme: int | None = None,
    num_chunks_femme: int | None = None,
    seed: int = 42,
    sample_key: str | None = None,
) -> list[dict[str, list[int]]]:
    """Tokenize a document and apply the shared label-aware chunk selection policy."""
    chunks = encode_text_chunks(text, tokenizer, max_length=max_length)
    return select_chunks(
        chunks,
        label=label,
        num_chunks_homme=num_chunks_homme,
        num_chunks_femme=num_chunks_femme,
        seed=seed,
        sample_key=sample_key,
    )


def empty_chunk_sampling_stats() -> ChunkSamplingStats:
    """Create an empty stats object with the expected class keys."""
    labels = ("homme", "femme")
    return ChunkSamplingStats(
        documents={label: 0 for label in labels},
        chunks_before_sampling={label: 0 for label in labels},
        chunks_after_sampling={label: 0 for label in labels},
    )


def format_chunk_sampling_stats(split_name: str, stats: ChunkSamplingStats) -> str:
    """Format dataset-level chunk sampling statistics for logs."""
    lines = [f"{split_name} split:"]
    for label in ("homme", "femme"):
        lines.append(f"  {label} documents: {stats.documents.get(label, 0)}")
    lines.append("")
    for label in ("homme", "femme"):
        lines.append(
            f"  {label} chunks before sampling: "
            f"{stats.chunks_before_sampling.get(label, 0)}"
        )
        lines.append(
            f"  {label} chunks after sampling: "
            f"{stats.chunks_after_sampling.get(label, 0)}"
        )
    return "\n".join(lines)


def build_chunk_records_with_stats(
    documents: list[DocumentRecord],
    tokenizer: Any,
    max_length: int = 512,
    num_chunks_homme: int | None = None,
    num_chunks_femme: int | None = None,
    seed: int = 42,
) -> tuple[list[ChunkRecord], ChunkSamplingStats]:
    """Create chunk records and collect before/after sampling statistics."""
    chunks: list[ChunkRecord] = []
    stats = empty_chunk_sampling_stats()

    for document_index, document in enumerate(documents):
        raw_chunks = encode_text_chunks(document.text, tokenizer, max_length=max_length)
        encoded_chunks = select_chunks(
            raw_chunks,
            label=document.label,
            num_chunks_homme=num_chunks_homme,
            num_chunks_femme=num_chunks_femme,
            seed=seed,
            sample_key=document.path,
        )
        stats.documents[document.label] = stats.documents.get(document.label, 0) + 1
        stats.chunks_before_sampling[document.label] = (
            stats.chunks_before_sampling.get(document.label, 0) + len(raw_chunks)
        )
        stats.chunks_after_sampling[document.label] = (
            stats.chunks_after_sampling.get(document.label, 0) + len(encoded_chunks)
        )

        for chunk_index, encoded in enumerate(encoded_chunks):
            chunks.append(
                ChunkRecord(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                    label_id=document.label_id,
                    label=document.label,
                    path=document.path,
                    document_index=document_index,
                    chunk_index=chunk_index,
                )
            )

    return chunks, stats


def build_chunk_records(
    documents: list[DocumentRecord],
    tokenizer: Any,
    max_length: int = 512,
    num_chunks_homme: int | None = None,
    num_chunks_femme: int | None = None,
    seed: int = 42,
) -> list[ChunkRecord]:
    """Create chunk records from original documents."""
    chunks, _ = build_chunk_records_with_stats(
        documents,
        tokenizer,
        max_length=max_length,
        num_chunks_homme=num_chunks_homme,
        num_chunks_femme=num_chunks_femme,
        seed=seed,
    )
    return chunks


class CamembertChunkDataset(Dataset):
    """PyTorch dataset of fixed-length CamemBERT chunks."""

    def __init__(self, chunks: list[ChunkRecord], chunk_sampling_stats: ChunkSamplingStats | None = None) -> None:
        self.chunks = chunks
        self.chunk_sampling_stats = chunk_sampling_stats

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, index: int) -> dict[str, Any]:
        chunk = self.chunks[index]
        return {
            "input_ids": torch.tensor(chunk.input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(chunk.attention_mask, dtype=torch.long),
            "labels": torch.tensor(chunk.label_id, dtype=torch.long),
            "path": chunk.path,
            "label": chunk.label,
            "document_index": chunk.document_index,
            "chunk_index": chunk.chunk_index,
        }


def create_dataloader(
    documents: list[DocumentRecord],
    tokenizer: Any,
    batch_size: int,
    max_length: int = 512,
    num_chunks_homme: int | None = None,
    num_chunks_femme: int | None = None,
    seed: int = 42,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    """Create a DataLoader over tokenizer chunks."""
    chunks, stats = build_chunk_records_with_stats(
        documents,
        tokenizer,
        max_length=max_length,
        num_chunks_homme=num_chunks_homme,
        num_chunks_femme=num_chunks_femme,
        seed=seed,
    )
    dataset = CamembertChunkDataset(chunks, chunk_sampling_stats=stats)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

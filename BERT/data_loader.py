"""Dataset loading and CamemBERT chunking utilities."""

from __future__ import annotations

import os
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


def chunk_text(text: str, tokenizer: Any, max_length: int = 512) -> list[dict[str, list[int]]]:
    """Tokenize a document into 512-token CamemBERT-ready chunks."""
    special_tokens = tokenizer.num_special_tokens_to_add(pair=False)
    payload_size = max_length - special_tokens
    token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer must define a pad_token_id for fixed-length batching")

    chunks: list[dict[str, list[int]]] = []
    for chunk_ids in chunk_token_ids(token_ids, payload_size):
        input_ids = tokenizer.build_inputs_with_special_tokens(chunk_ids)
        input_ids = input_ids[:max_length]
        attention_mask = [1] * len(input_ids)
        padding_length = max_length - len(input_ids)
        if padding_length > 0:
            input_ids = input_ids + [pad_token_id] * padding_length
            attention_mask = attention_mask + [0] * padding_length

        chunks.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
        )

    return chunks


def build_chunk_records(
    documents: list[DocumentRecord],
    tokenizer: Any,
    max_length: int = 512,
) -> list[ChunkRecord]:
    """Create chunk records from original documents."""
    chunks: list[ChunkRecord] = []
    for document_index, document in enumerate(documents):
        encoded_chunks = chunk_text(document.text, tokenizer, max_length=max_length)
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
    return chunks


class CamembertChunkDataset(Dataset):
    """PyTorch dataset of fixed-length CamemBERT chunks."""

    def __init__(self, chunks: list[ChunkRecord]) -> None:
        self.chunks = chunks

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
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    """Create a DataLoader over tokenizer chunks."""
    chunks = build_chunk_records(documents, tokenizer, max_length=max_length)
    dataset = CamembertChunkDataset(chunks)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

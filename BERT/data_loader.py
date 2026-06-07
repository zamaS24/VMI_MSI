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


def select_chunks(
    chunks: list[dict[str, list[int]]],
    num_chunks: int | None = None,
    seed: int = 42,
    sample_key: str | None = None,
) -> list[dict[str, list[int]]]:
    """Skip the first chunk and optionally sample a fixed number of remaining chunks."""
    remaining_chunks = chunks[1:]
    if num_chunks is None:
        return remaining_chunks
    if num_chunks < 0:
        raise ValueError("num_chunks must be None or a non-negative integer")
    if len(remaining_chunks) <= num_chunks:
        return remaining_chunks

    rng = random.Random(_sample_seed(seed, sample_key))
    selected_indices = sorted(rng.sample(range(len(remaining_chunks)), num_chunks))
    return [remaining_chunks[index] for index in selected_indices]


def describe_chunk_selection(num_chunks: int | None) -> str:
    """Return a human-readable description of the active chunk selection policy."""
    if num_chunks is None:
        return "Skipping first chunk. Using all remaining chunks."
    return f"Skipping first chunk. Randomly sampling {num_chunks} chunks per document."


def chunk_text(
    text: str,
    tokenizer: Any,
    max_length: int = 512,
    num_chunks: int | None = None,
    seed: int = 42,
    sample_key: str | None = None,
) -> list[dict[str, list[int]]]:
    """Tokenize a document into 512-token CamemBERT-ready chunks."""
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

    return select_chunks(chunks, num_chunks=num_chunks, seed=seed, sample_key=sample_key)


def build_chunk_records(
    documents: list[DocumentRecord],
    tokenizer: Any,
    max_length: int = 512,
    num_chunks: int | None = None,
    seed: int = 42,
) -> list[ChunkRecord]:
    """Create chunk records from original documents."""
    chunks: list[ChunkRecord] = []
    for document_index, document in enumerate(documents):
        encoded_chunks = chunk_text(
            document.text,
            tokenizer,
            max_length=max_length,
            num_chunks=num_chunks,
            seed=seed,
            sample_key=document.path,
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
    num_chunks: int | None = None,
    seed: int = 42,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    """Create a DataLoader over tokenizer chunks."""
    chunks = build_chunk_records(
        documents,
        tokenizer,
        max_length=max_length,
        num_chunks=num_chunks,
        seed=seed,
    )
    dataset = CamembertChunkDataset(chunks)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

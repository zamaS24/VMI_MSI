"""Configuration defaults for the CamemBERT classification pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
BERT_ROOT = Path(__file__).resolve().parent

LABEL2ID = {"femme": 0, "homme": 1}
ID2LABEL = {0: "femme", 1: "homme"}


@dataclass
class PathConfig:
    """Filesystem locations used by the BERT pipeline."""

    data_dir: Path = PROJECT_ROOT / "data" / "datasetSujet3" / "content" / "dataset"
    output_dir: Path = BERT_ROOT / "outputs"
    checkpoint_dir: Path = BERT_ROOT / "outputs" / "checkpoints"
    model_dir: Path = BERT_ROOT / "outputs" / "models"
    log_dir: Path = BERT_ROOT / "outputs" / "logs"
    artifact_dir: Path = BERT_ROOT / "artifacts"
    vis_dir: Path = BERT_ROOT / "vis"

    @property
    def best_model_dir(self) -> Path:
        """Directory where the best fine-tuned model is stored."""
        return self.model_dir / "best_model"

    @property
    def metrics_path(self) -> Path:
        """Default metrics JSON path."""
        return self.artifact_dir / "metrics.json"

    @property
    def history_path(self) -> Path:
        """Default training history CSV path."""
        return self.log_dir / "history.csv"

    @property
    def predictions_path(self) -> Path:
        """Default test prediction CSV path."""
        return self.artifact_dir / "test_predictions.csv"


@dataclass
class ModelConfig:
    """Model and tokenizer parameters."""

    model_name: str = "camembert-base"
    num_labels: int = 2
    max_length: int = 512


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    seed: int = 42
    batch_size: int = 4
    eval_batch_size: int = 8
    epochs: int = 5
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    patience: int = 2
    num_workers: int = 0
    use_amp: bool = True


@dataclass
class ExplainabilityConfig:
    """Shared explainability defaults."""

    n_examples: int = 50
    n_terms: int = 20
    batch_size: int = 8


def dataclass_to_dict(value: Any) -> dict[str, Any]:
    """Convert nested dataclasses and paths to JSON-friendly values."""
    raw = asdict(value)

    def convert(item: Any) -> Any:
        if isinstance(item, Path):
            return str(item)
        if isinstance(item, dict):
            return {key: convert(val) for key, val in item.items()}
        if isinstance(item, list):
            return [convert(val) for val in item]
        return item

    return convert(raw)

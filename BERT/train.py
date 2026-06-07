"""Fine-tune CamemBERT for binary text classification."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.optim import AdamW
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

from config import ID2LABEL, ModelConfig, PathConfig, TrainingConfig, dataclass_to_dict
from data_loader import create_dataloader, describe_chunk_selection, format_chunk_sampling_stats, load_splits
from model import batch_to_device, create_model, create_tokenizer, save_model
from utils import compute_metrics, ensure_dir, save_json, seed_everything


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    path_defaults = PathConfig()
    model_defaults = ModelConfig()
    train_defaults = TrainingConfig()

    parser = argparse.ArgumentParser(description="Fine-tune CamemBERT on the dataset splits.")
    parser.add_argument("--data_dir", "--data-dir", type=Path, default=path_defaults.data_dir)
    parser.add_argument("--model_name", "--model-name", default=model_defaults.model_name)
    parser.add_argument("--max_length", "--max-length", type=int, default=model_defaults.max_length)
    parser.add_argument("--num_chunks_homme", "--num-chunks-homme", type=int, default=model_defaults.num_chunks_homme)
    parser.add_argument("--num_chunks_femme", "--num-chunks-femme", type=int, default=model_defaults.num_chunks_femme)
    parser.add_argument("--batch_size", "--batch-size", type=int, default=train_defaults.batch_size)
    parser.add_argument("--eval_batch_size", "--eval-batch-size", type=int, default=train_defaults.eval_batch_size)
    parser.add_argument("--epochs", type=int, default=train_defaults.epochs)
    parser.add_argument("--learning_rate", "--learning-rate", type=float, default=train_defaults.learning_rate)
    parser.add_argument("--weight_decay", "--weight-decay", type=float, default=train_defaults.weight_decay)
    parser.add_argument("--warmup_ratio", "--warmup-ratio", type=float, default=train_defaults.warmup_ratio)
    parser.add_argument("--max_grad_norm", "--max-grad-norm", type=float, default=train_defaults.max_grad_norm)
    parser.add_argument("--patience", type=int, default=train_defaults.patience)
    parser.add_argument("--num_workers", "--num-workers", type=int, default=train_defaults.num_workers)
    parser.add_argument("--seed", type=int, default=train_defaults.seed)
    parser.add_argument("--no_amp", "--no-amp", action="store_true", help="Disable CUDA mixed precision.")
    parser.add_argument("--checkpoint_dir", "--checkpoint-dir", type=Path, default=path_defaults.checkpoint_dir)
    parser.add_argument("--best_model_dir", "--best-model-dir", type=Path, default=path_defaults.best_model_dir)
    parser.add_argument("--log_dir", "--log-dir", type=Path, default=path_defaults.log_dir)
    parser.add_argument("--artifact_dir", "--artifact-dir", type=Path, default=path_defaults.artifact_dir)
    return parser.parse_args()


def prepare_directories(args: argparse.Namespace) -> None:
    """Create output directories used by training."""
    ensure_dir(args.checkpoint_dir)
    ensure_dir(args.best_model_dir)
    ensure_dir(args.log_dir)
    ensure_dir(args.artifact_dir)


def run_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    max_grad_norm: float = 1.0,
    use_amp: bool = True,
) -> dict[str, Any]:
    """Run one training or evaluation epoch over chunk batches."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total_examples = 0
    all_labels: list[int] = []
    all_predictions: list[int] = []
    target_names = [ID2LABEL[0], ID2LABEL[1]]

    progress = tqdm(dataloader, leave=False, desc="train" if is_train else "eval")
    for batch in progress:
        model_inputs = batch_to_device(batch, device)
        labels = model_inputs["labels"]

        with torch.set_grad_enabled(is_train):
            with torch.cuda.amp.autocast(enabled=use_amp and device.type == "cuda"):
                outputs = model(**model_inputs)
                loss = outputs.loss

            if is_train:
                optimizer.zero_grad()
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
                if scheduler is not None:
                    scheduler.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size
        predictions = outputs.logits.argmax(dim=1)
        all_labels.extend(labels.detach().cpu().tolist())
        all_predictions.extend(predictions.detach().cpu().tolist())
        progress.set_postfix(loss=f"{loss.item():.4f}")

    metrics = compute_metrics(all_labels, all_predictions, target_names=target_names)
    metrics["loss"] = total_loss / max(total_examples, 1)
    return metrics


def save_history(history: list[dict[str, Any]], history_path: Path) -> None:
    """Save training history as CSV."""
    history_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(history_path, index=False)


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    """Run CamemBERT fine-tuning from a prepared argument namespace."""
    prepare_directories(args)
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = create_tokenizer(args.model_name)
    model = create_model(args.model_name).to(device)
    print(describe_chunk_selection(args.num_chunks_homme, args.num_chunks_femme))

    train_docs, val_docs, _ = load_splits(args.data_dir)
    train_loader = create_dataloader(
        train_docs,
        tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        num_chunks_homme=args.num_chunks_homme,
        num_chunks_femme=args.num_chunks_femme,
        seed=args.seed,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = create_dataloader(
        val_docs,
        tokenizer,
        batch_size=args.eval_batch_size,
        max_length=args.max_length,
        num_chunks_homme=args.num_chunks_homme,
        num_chunks_femme=args.num_chunks_femme,
        seed=args.seed,
        shuffle=False,
        num_workers=args.num_workers,
    )
    print(format_chunk_sampling_stats("Training", train_loader.dataset.chunk_sampling_stats))
    print(format_chunk_sampling_stats("Validation", val_loader.dataset.chunk_sampling_stats))

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = max(len(train_loader) * args.epochs, 1)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=not args.no_amp and device.type == "cuda")

    history: list[dict[str, Any]] = []
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            max_grad_norm=args.max_grad_norm,
            use_amp=not args.no_amp,
        )
        val_metrics = run_epoch(model, val_loader, device, use_amp=False)

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_precision": train_metrics["precision"],
            "train_recall": train_metrics["recall"],
            "train_f1": train_metrics["f1"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
        }
        history.append(row)
        print(
            "train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            "val_acc={val_accuracy:.4f} val_f1={val_f1:.4f}".format(**row)
        )

        checkpoint_path = args.checkpoint_dir / f"epoch_{epoch:02d}"
        save_model(model, tokenizer, checkpoint_path)

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            epochs_without_improvement = 0
            save_model(model, tokenizer, args.best_model_dir)
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
            break

    history_path = args.log_dir / "history.csv"
    save_history(history, history_path)

    metrics = {
        "model": dataclass_to_dict(
            ModelConfig(
                model_name=args.model_name,
                max_length=args.max_length,
                num_chunks_homme=args.num_chunks_homme,
                num_chunks_femme=args.num_chunks_femme,
            )
        ),
        "training": {
            "seed": args.seed,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "epochs_requested": args.epochs,
            "epochs_completed": len(history),
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "max_grad_norm": args.max_grad_norm,
            "patience": args.patience,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
        },
        "labels": {str(key): value for key, value in ID2LABEL.items()},
        "history": history,
        "paths": {
            "best_model_dir": str(args.best_model_dir),
            "history_path": str(history_path),
        },
    }
    save_json(metrics, args.artifact_dir / "metrics.json")

    print(f"Saved best model: {args.best_model_dir}")
    print(f"Saved history: {history_path}")
    print(f"Saved metrics: {args.artifact_dir / 'metrics.json'}")
    return metrics


def main() -> None:
    """Run CamemBERT fine-tuning from the command line."""
    args = parse_args()
    run_training(args)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from .data import FaceFolderDataset
from .model import TinyFaceMatchTrainingModel, count_parameters


def pick_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def make_loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader, FaceFolderDataset]:
    train_base = FaceFolderDataset(
        root=args.data_dir,
        image_size=112,
        augment=args.augment,
        repeat=args.train_repeat,
    )
    val_base = FaceFolderDataset(
        root=args.data_dir,
        image_size=112,
        augment=False,
        repeat=1,
    )
    sample_count = len(train_base.samples)
    val_size = max(1, int(sample_count * args.val_fraction))
    train_size = sample_count - val_size
    if train_size <= 0:
        raise ValueError("dataset is too small for the requested validation fraction")

    generator = torch.Generator().manual_seed(args.seed)
    permutation = torch.randperm(sample_count, generator=generator).tolist()
    val_indices = permutation[:val_size]
    train_indices = permutation[val_size:]
    repeated_train_indices = [
        index + (repeat_index * sample_count)
        for repeat_index in range(args.train_repeat)
        for index in train_indices
    ]
    train_dataset = Subset(train_base, repeated_train_indices)
    val_dataset = Subset(val_base, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.device == "cuda",
        drop_last=train_size >= args.batch_size,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device == "cuda",
    )
    return train_loader, val_loader, train_base


def train_one_epoch(
    model: TinyFaceMatchTrainingModel,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    use_amp: bool,
    grad_clip: float,
    epoch: int,
    use_arcface_margin: bool,
) -> tuple[float, float, float]:
    model.train()
    total_loss = 0.0
    top1_correct = 0
    top5_correct = 0
    total = 0
    progress = tqdm(loader, desc=f"epoch {epoch}", leave=False)
    for images, labels in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            embeddings, logits = model(images, labels if use_arcface_margin else None)
            loss = criterion(logits, labels)
            metric_logits = model.classifier(embeddings, None)

        scaler.scale(loss).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        batch_size = labels.numel()
        total_loss += float(loss.item()) * batch_size
        top1, top5 = topk_correct(metric_logits.detach(), labels, topk=(1, 5))
        top1_correct += top1
        top5_correct += top5
        total += batch_size
        progress.set_postfix(loss=f"{total_loss / max(total, 1):.4f}")

    return (
        total_loss / max(total, 1),
        top1_correct / max(total, 1),
        top5_correct / max(total, 1),
    )


@torch.no_grad()
def evaluate(
    model: TinyFaceMatchTrainingModel,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_arcface_margin: bool,
) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    top1_correct = 0
    top5_correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        embeddings, logits = model(images, labels if use_arcface_margin else None)
        loss = criterion(logits, labels)
        metric_logits = model.classifier(embeddings, None)
        batch_size = labels.numel()
        total_loss += float(loss.item()) * batch_size
        top1, top5 = topk_correct(metric_logits, labels, topk=(1, 5))
        top1_correct += top1
        top5_correct += top5
        total += batch_size
    return (
        total_loss / max(total, 1),
        top1_correct / max(total, 1),
        top5_correct / max(total, 1),
    )


def topk_correct(logits: torch.Tensor, labels: torch.Tensor, topk: tuple[int, ...]) -> list[int]:
    max_k = min(max(topk), logits.shape[1])
    predictions = logits.topk(max_k, dim=1).indices.t()
    matches = predictions.eq(labels.view(1, -1))
    return [int(matches[: min(k, max_k)].reshape(-1).float().sum().item()) for k in topk]


def save_checkpoint(
    path: Path,
    model: TinyFaceMatchTrainingModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    dataset: FaceFolderDataset,
    args: argparse.Namespace,
    epoch: int,
    best_val_acc: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "best_val_acc": best_val_acc,
        "config": {
            "architecture": "TinyFaceMatch128",
            "embedding_dim": 128,
            "input_size": 112,
            "width_mult": args.width_mult,
            "scale": args.scale,
            "margin": args.margin,
            "margin_warmup_epochs": args.margin_warmup_epochs,
        },
        "class_to_index": dataset.class_to_index,
        "backbone_state_dict": model.backbone.state_dict(),
        "classifier_state_dict": model.classifier.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }
    torch.save(checkpoint, path)
    path.with_suffix(".labels.json").write_text(
        json.dumps(dataset.class_to_index, indent=2),
        encoding="utf-8",
    )


def write_metrics_header(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "epoch",
                "train_loss",
                "train_top1",
                "train_top5",
                "val_loss",
                "val_top1",
                "val_top5",
                "lr",
                "arcface_margin",
            ]
        )


def load_checkpoint_for_training(
    path: Path,
    model: TinyFaceMatchTrainingModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    reset_optimizer: bool,
) -> tuple[int, float]:
    checkpoint = torch.load(path, map_location="cpu")
    model.backbone.load_state_dict(checkpoint["backbone_state_dict"])
    model.classifier.load_state_dict(checkpoint["classifier_state_dict"])

    if not reset_optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if (
        not reset_optimizer
        and "scheduler_state_dict" in checkpoint
        and checkpoint["scheduler_state_dict"] is not None
    ):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    start_epoch = int(checkpoint.get("epoch", 0)) + 1
    best_val_acc = float(checkpoint.get("best_val_acc", -1.0))
    return start_epoch, best_val_acc


def append_metrics(
    path: Path,
    epoch: int,
    train_loss: float,
    train_top1: float,
    train_top5: float,
    val_loss: float,
    val_top1: float,
    val_top5: float,
    lr: float,
    use_arcface_margin: bool,
) -> None:
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                epoch,
                f"{train_loss:.6f}",
                f"{train_top1:.6f}",
                f"{train_top5:.6f}",
                f"{val_loss:.6f}",
                f"{val_top1:.6f}",
                f"{val_top5:.6f}",
                f"{lr:.8f}",
                int(use_arcface_margin),
            ]
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a 128-D TinyFaceMatch model.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/tinyfacematch128"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--width-mult", type=float, default=0.5)
    parser.add_argument("--scale", type=float, default=30.0)
    parser.add_argument("--margin", type=float, default=0.35)
    parser.add_argument("--margin-warmup-epochs", type=int, default=5)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--train-repeat", type=int, default=1)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--reset-optimizer",
        action="store_true",
        help="Load model weights from --resume but start a fresh optimizer/scheduler.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = pick_device(args.device)
    args.device = device.type
    torch.manual_seed(args.seed)

    train_loader, val_loader, dataset = make_loaders(args)
    model = TinyFaceMatchTrainingModel(
        num_classes=len(dataset.classes),
        width_mult=args.width_mult,
        scale=args.scale,
        margin=args.margin,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs, 1),
        eta_min=args.lr * 0.01,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    print(f"device: {device}")
    print(f"identities: {len(dataset.classes)}")
    print(f"source images: {len(dataset.samples)}")
    print(f"train batches: {len(train_loader)}")
    print(f"val batches: {len(val_loader)}")
    print(f"width_mult: {args.width_mult}")
    print(f"embedding dim: 128")
    print(f"backbone channels: {model.backbone.channels}")
    print(f"trainable params: {count_parameters(model):,}")

    metrics_path = args.output_dir / "metrics.csv"
    start_epoch = 1
    best_val_acc = -1.0
    if args.resume is not None:
        start_epoch, best_val_acc = load_checkpoint_for_training(
            args.resume,
            model,
            optimizer,
            scheduler,
            reset_optimizer=args.reset_optimizer,
        )
        print(f"resumed checkpoint: {args.resume}")
        print(f"checkpoint epoch: {start_epoch - 1}")
        print(f"reset optimizer: {args.reset_optimizer}")

    if not metrics_path.exists() or args.resume is None:
        write_metrics_header(metrics_path)

    end_epoch = start_epoch + args.epochs - 1
    for epoch in range(start_epoch, end_epoch + 1):
        resumed_epochs_done = epoch - start_epoch
        use_arcface_margin = resumed_epochs_done >= args.margin_warmup_epochs
        train_loss, train_top1, train_top5 = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            use_amp=args.amp and device.type == "cuda",
            grad_clip=args.grad_clip,
            epoch=epoch,
            use_arcface_margin=use_arcface_margin,
        )
        val_loss, val_top1, val_top5 = evaluate(
            model,
            val_loader,
            criterion,
            device,
            use_arcface_margin=use_arcface_margin,
        )
        lr = float(optimizer.param_groups[0]["lr"])
        scheduler.step()
        append_metrics(
            metrics_path,
            epoch,
            train_loss,
            train_top1,
            train_top5,
            val_loss,
            val_top1,
            val_top5,
            lr,
            use_arcface_margin,
        )

        print(
            f"epoch {epoch:03d} "
            f"train_loss={train_loss:.4f} train_top1={train_top1:.4f} "
            f"train_top5={train_top5:.4f} val_loss={val_loss:.4f} "
            f"val_top1={val_top1:.4f} val_top5={val_top5:.4f} "
            f"margin={int(use_arcface_margin)} lr={lr:.6f}"
        )

        save_checkpoint(
            args.output_dir / "last.pt",
            model,
            optimizer,
            scheduler,
            dataset,
            args,
            epoch,
            best_val_acc,
        )
        if val_top1 > best_val_acc:
            best_val_acc = val_top1
            save_checkpoint(
                args.output_dir / "best.pt",
                model,
                optimizer,
                scheduler,
                dataset,
                args,
                epoch,
                best_val_acc,
            )
            print(f"saved best checkpoint: {args.output_dir / 'best.pt'}")


if __name__ == "__main__":
    main()

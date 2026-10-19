from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from tqdm import tqdm

from .data import IMAGE_EXTENSIONS, PROJECT_MEAN, PROJECT_SCALE
from .model import TinyFaceMatch128


def load_image(path: Path, image_size: int = 112) -> Tensor:
    image = Image.open(path).convert("RGB")
    image = image.resize((image_size, image_size), resample=Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32)
    array = (array - PROJECT_MEAN) / PROJECT_SCALE
    chw = np.transpose(array, (2, 0, 1))
    return torch.from_numpy(np.ascontiguousarray(chw))


def collect_images_by_class(root: Path) -> dict[str, list[Path]]:
    images_by_class: dict[str, list[Path]] = {}
    for class_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        images = [
            path
            for path in sorted(class_dir.rglob("*"))
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if images:
            images_by_class[class_dir.name] = images
    if not images_by_class:
        raise ValueError(f"no images found in identity folders under {root}")
    return images_by_class


def make_pairs(
    images_by_class: dict[str, list[Path]],
    same_pairs: int,
    diff_pairs: int,
    seed: int,
) -> tuple[list[tuple[Path, Path]], np.ndarray]:
    rng = random.Random(seed)
    classes_with_two = [name for name, paths in images_by_class.items() if len(paths) >= 2]
    all_classes = list(images_by_class)
    if len(classes_with_two) < 1:
        raise ValueError("need at least one identity with two or more images for same pairs")
    if len(all_classes) < 2:
        raise ValueError("need at least two identities for different pairs")

    pairs: list[tuple[Path, Path]] = []
    labels: list[int] = []

    for _ in range(same_pairs):
        class_name = rng.choice(classes_with_two)
        left, right = rng.sample(images_by_class[class_name], 2)
        pairs.append((left, right))
        labels.append(1)

    for _ in range(diff_pairs):
        class_a, class_b = rng.sample(all_classes, 2)
        left = rng.choice(images_by_class[class_a])
        right = rng.choice(images_by_class[class_b])
        pairs.append((left, right))
        labels.append(0)

    order = list(range(len(pairs)))
    rng.shuffle(order)
    return [pairs[index] for index in order], np.asarray([labels[index] for index in order])


@torch.no_grad()
def embed_paths(
    model: TinyFaceMatch128,
    paths: list[Path],
    batch_size: int,
    device: torch.device,
) -> dict[Path, np.ndarray]:
    embeddings: dict[Path, np.ndarray] = {}
    model.eval()
    for start in tqdm(range(0, len(paths), batch_size), desc="embedding images", leave=False):
        batch_paths = paths[start : start + batch_size]
        images = torch.stack([load_image(path) for path in batch_paths]).to(device)
        batch_embeddings = model(images).cpu().numpy().astype(np.float32)
        for path, embedding in zip(batch_paths, batch_embeddings, strict=True):
            embeddings[path] = embedding
    return embeddings


def compute_scores(
    pairs: list[tuple[Path, Path]],
    embeddings: dict[Path, np.ndarray],
) -> np.ndarray:
    scores = []
    for left, right in pairs:
        score = float(np.dot(embeddings[left], embeddings[right]))
        scores.append(score)
    return np.asarray(scores, dtype=np.float32)


def evaluate_thresholds(scores: np.ndarray, labels: np.ndarray, steps: int) -> dict[str, float]:
    thresholds = np.linspace(-1.0, 1.0, steps, dtype=np.float32)
    positives = labels == 1
    negatives = labels == 0
    positive_count = max(1, int(positives.sum()))
    negative_count = max(1, int(negatives.sum()))

    best = {
        "threshold": 0.0,
        "accuracy": -1.0,
        "tpr": 0.0,
        "fpr": 0.0,
        "precision": 0.0,
        "eer_threshold": 0.0,
        "eer": 1.0,
    }
    best_eer_gap = float("inf")

    for threshold in thresholds:
        predictions = scores >= threshold
        tp = int(np.logical_and(predictions, positives).sum())
        tn = int(np.logical_and(~predictions, negatives).sum())
        fp = int(np.logical_and(predictions, negatives).sum())
        fn = int(np.logical_and(~predictions, positives).sum())

        accuracy = (tp + tn) / len(labels)
        tpr = tp / positive_count
        fpr = fp / negative_count
        fnr = fn / positive_count
        precision = tp / max(1, tp + fp)

        if accuracy > best["accuracy"]:
            best.update(
                {
                    "threshold": float(threshold),
                    "accuracy": float(accuracy),
                    "tpr": float(tpr),
                    "fpr": float(fpr),
                    "precision": float(precision),
                }
            )

        eer_gap = abs(fpr - fnr)
        if eer_gap < best_eer_gap:
            best_eer_gap = eer_gap
            eer = (fpr + fnr) / 2.0
            best["eer_threshold"] = float(threshold)
            best["eer"] = float(eer)

    return best


def score_summary(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    same_scores = scores[labels == 1]
    diff_scores = scores[labels == 0]
    return {
        "same_mean": float(same_scores.mean()),
        "same_std": float(same_scores.std()),
        "diff_mean": float(diff_scores.mean()),
        "diff_std": float(diff_scores.std()),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate TinyFaceMatch128 pair verification.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--same-pairs", type=int, default=5000)
    parser.add_argument("--diff-pairs", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--threshold-steps", type=int, default=2001)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-json", type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    width_mult = float(checkpoint.get("config", {}).get("width_mult", 0.5))

    model = TinyFaceMatch128(width_mult=width_mult)
    model.load_state_dict(checkpoint["backbone_state_dict"])
    model.to(device)

    images_by_class = collect_images_by_class(args.data_dir)
    pairs, labels = make_pairs(
        images_by_class,
        same_pairs=args.same_pairs,
        diff_pairs=args.diff_pairs,
        seed=args.seed,
    )

    unique_paths = sorted({path for pair in pairs for path in pair})
    embeddings = embed_paths(model, unique_paths, args.batch_size, device)
    scores = compute_scores(pairs, embeddings)
    metrics = evaluate_thresholds(scores, labels, args.threshold_steps)
    metrics.update(score_summary(scores, labels))
    metrics.update(
        {
            "checkpoint": str(args.checkpoint),
            "data_dir": str(args.data_dir),
            "width_mult": width_mult,
            "pairs": int(len(labels)),
            "same_pairs": int((labels == 1).sum()),
            "diff_pairs": int((labels == 0).sum()),
        }
    )

    print(f"checkpoint: {args.checkpoint}")
    print(f"data: {args.data_dir}")
    print(f"pairs: {metrics['pairs']} same={metrics['same_pairs']} diff={metrics['diff_pairs']}")
    print(f"same cosine: mean={metrics['same_mean']:.4f} std={metrics['same_std']:.4f}")
    print(f"diff cosine: mean={metrics['diff_mean']:.4f} std={metrics['diff_std']:.4f}")
    print(
        "best threshold: "
        f"{metrics['threshold']:.4f} "
        f"acc={metrics['accuracy']:.4f} "
        f"tpr={metrics['tpr']:.4f} "
        f"fpr={metrics['fpr']:.4f} "
        f"precision={metrics['precision']:.4f}"
    )
    print(f"eer approx: {metrics['eer']:.4f} at threshold={metrics['eer_threshold']:.4f}")

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"wrote: {args.output_json}")


if __name__ == "__main__":
    main()

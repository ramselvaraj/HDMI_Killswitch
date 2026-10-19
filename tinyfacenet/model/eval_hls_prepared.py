from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from tqdm import tqdm

from .eval_pairs import (
    collect_images_by_class,
    compute_scores,
    evaluate_thresholds,
    load_image,
    make_pairs,
    score_summary,
)
from .model import TinyFaceMatch128


def fake_quant(x: Tensor, frac_bits: int, dtype: torch.dtype = torch.int16) -> Tensor:
    if dtype == torch.int16:
        qmin, qmax = -32768, 32767
    elif dtype == torch.int32:
        qmin, qmax = -2147483648, 2147483647
    else:
        raise ValueError(f"unsupported dtype: {dtype}")
    scale = float(1 << frac_bits)
    q = torch.clamp(torch.round(x * scale), qmin, qmax)
    return q / scale


class PreparedTinyFaceMatchReference(nn.Module):
    def __init__(self, prepared_dir: Path) -> None:
        super().__init__()
        self.prepared_dir = prepared_dir
        self.manifest = json.loads((prepared_dir / "manifest.json").read_text(encoding="utf-8"))
        self.arrays = np.load(prepared_dir / "weights_quantized.npz")
        q = self.manifest["quantization"]
        self.weight_frac = int(q["weight_frac_bits"])
        self.bias_frac = int(q["bias_frac_bits"])
        self.prelu_frac = int(q["prelu_frac_bits"])
        self.act_frac = int(q["activation_frac_bits"])
        self.out_frac = int(q["output_frac_bits"])
        self.layer_map = {layer["name"]: layer for layer in self.manifest["layers"]}

    def deq(self, key: str, frac_bits: int) -> Tensor:
        array = self.arrays[key].astype(np.float32) / float(1 << frac_bits)
        return torch.from_numpy(array)

    def conv(self, x: Tensor, name: str) -> Tensor:
        layer = self.layer_map[name]
        weight = self.deq(layer["weight_key"], self.weight_frac).to(x.device)
        bias = self.deq(layer["bias_key"], self.bias_frac).to(x.device)
        x = F.conv2d(
            x,
            weight,
            bias,
            stride=tuple(layer["stride"]),
            padding=tuple(layer["padding"]),
            groups=int(layer["groups"]),
        )
        if layer["prelu_key"] is not None:
            alpha = self.deq(layer["prelu_key"], self.prelu_frac).to(x.device).view(1, -1, 1, 1)
            x = torch.where(x >= 0, x, x * alpha)
        return fake_quant(x, self.act_frac)

    def linear(self, x: Tensor, name: str) -> Tensor:
        layer = self.layer_map[name]
        weight = self.deq(layer["weight_key"], self.weight_frac).to(x.device)
        bias = self.deq(layer["bias_key"], self.bias_frac).to(x.device)
        x = F.linear(x, weight, bias)
        return fake_quant(x, self.act_frac)

    def bottleneck(self, x: Tensor, index: int, residual: bool) -> Tensor:
        identity = x
        x = self.conv(x, f"b{index}_expand")
        x = self.conv(x, f"b{index}_dw")
        x = self.conv(x, f"b{index}_project")
        if residual:
            x = fake_quant(x + identity, self.act_frac)
        return x

    def forward(self, x: Tensor) -> Tensor:
        x = fake_quant(x, self.act_frac)
        x = self.conv(x, "conv0")
        x = self.conv(x, "dwsep0_dw")
        x = self.conv(x, "dwsep0_pw")

        x = self.bottleneck(x, 0, residual=False)
        x = self.bottleneck(x, 1, residual=True)
        x = self.bottleneck(x, 2, residual=True)
        x = self.bottleneck(x, 3, residual=False)
        x = self.bottleneck(x, 4, residual=True)
        x = self.bottleneck(x, 5, residual=True)
        x = self.bottleneck(x, 6, residual=False)
        x = self.bottleneck(x, 7, residual=True)

        x = self.conv(x, "conv_final")
        x = self.conv(x, "global_dw7")
        x = torch.flatten(x, 1)
        x = self.linear(x, "dense")
        x = F.normalize(x, p=2, dim=1)
        return fake_quant(x, self.out_frac)


@torch.no_grad()
def embed_paths(
    model: nn.Module,
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


@torch.no_grad()
def compare_to_pytorch(
    prepared_model: nn.Module,
    checkpoint: Path,
    paths: list[Path],
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    checkpoint_obj = torch.load(checkpoint, map_location="cpu")
    width_mult = float(checkpoint_obj.get("config", {}).get("width_mult", 0.5))
    pytorch_model = TinyFaceMatch128(width_mult=width_mult)
    pytorch_model.load_state_dict(checkpoint_obj["backbone_state_dict"])
    pytorch_model.to(device).eval()
    prepared_model.to(device).eval()

    max_abs = 0.0
    mean_abs_sum = 0.0
    cosine_sum = 0.0
    count = 0

    for start in tqdm(range(0, len(paths), batch_size), desc="comparing embeddings", leave=False):
        batch_paths = paths[start : start + batch_size]
        images = torch.stack([load_image(path) for path in batch_paths]).to(device)
        fp = pytorch_model(images)
        quant = prepared_model(images)
        diff = torch.abs(fp - quant)
        max_abs = max(max_abs, float(diff.max().item()))
        mean_abs_sum += float(diff.mean().item()) * len(batch_paths)
        cosine_sum += float(F.cosine_similarity(fp, quant, dim=1).sum().item())
        count += len(batch_paths)

    return {
        "embedding_max_abs_error": max_abs,
        "embedding_mean_abs_error": mean_abs_sum / max(1, count),
        "embedding_mean_cosine_to_pytorch": cosine_sum / max(1, count),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate prepared HLS TinyFaceMatch weights.")
    parser.add_argument("--prepared-dir", type=Path, required=True)
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

    images_by_class = collect_images_by_class(args.data_dir)
    pairs, labels = make_pairs(
        images_by_class,
        same_pairs=args.same_pairs,
        diff_pairs=args.diff_pairs,
        seed=args.seed,
    )
    unique_paths = sorted({path for pair in pairs for path in pair})

    model = PreparedTinyFaceMatchReference(args.prepared_dir).to(device)
    embeddings = embed_paths(model, unique_paths, args.batch_size, device)
    scores = compute_scores(pairs, embeddings)
    metrics = evaluate_thresholds(scores, labels, args.threshold_steps)
    metrics.update(score_summary(scores, labels))
    metrics.update(
        compare_to_pytorch(
            prepared_model=model,
            checkpoint=args.checkpoint,
            paths=unique_paths,
            batch_size=args.batch_size,
            device=device,
        )
    )
    metrics.update(
        {
            "prepared_dir": str(args.prepared_dir),
            "checkpoint": str(args.checkpoint),
            "data_dir": str(args.data_dir),
            "pairs": int(len(labels)),
            "same_pairs": int((labels == 1).sum()),
            "diff_pairs": int((labels == 0).sum()),
        }
    )

    print(f"prepared_dir: {args.prepared_dir}")
    print(f"checkpoint: {args.checkpoint}")
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
    print(
        "prepared vs pytorch: "
        f"mean_abs={metrics['embedding_mean_abs_error']:.6f} "
        f"max_abs={metrics['embedding_max_abs_error']:.6f} "
        f"mean_cos={metrics['embedding_mean_cosine_to_pytorch']:.6f}"
    )

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"wrote: {args.output_json}")


if __name__ == "__main__":
    main()

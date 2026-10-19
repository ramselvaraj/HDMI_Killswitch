from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .model import TinyFaceMatch128


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export TinyFaceMatch128 backbone to ONNX.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("tinyfacematch128.onnx"))
    parser.add_argument("--opset", type=int, default=13)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = checkpoint.get("config", {})
    model = TinyFaceMatch128(width_mult=float(config.get("width_mult", 0.5)))
    model.load_state_dict(checkpoint["backbone_state_dict"])
    model.eval()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 3, 112, 112)
    torch.onnx.export(
        model,
        dummy,
        args.output,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["embedding"],
        dynamic_axes={"input": {0: "batch"}, "embedding": {0: "batch"}},
    )
    print(f"exported: {args.output}")
    print("input: N x 3 x 112 x 112")
    print("output: N x 128")


if __name__ == "__main__":
    main()

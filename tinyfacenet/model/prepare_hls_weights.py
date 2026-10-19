from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .model import TinyFaceMatch128


@dataclass
class LayerSpec:
    name: str
    op: str
    weight_key: str
    bias_key: str
    prelu_key: str | None
    weight_shape: list[int]
    bias_shape: list[int]
    stride: list[int] | None = None
    padding: list[int] | None = None
    groups: int | None = None
    offset_bytes: int = 0
    weight_bytes: int = 0
    bias_bytes: int = 0
    prelu_bytes: int = 0


def quantize_signed(array: np.ndarray, frac_bits: int, dtype: np.dtype) -> np.ndarray:
    info = np.iinfo(dtype)
    scale = float(1 << frac_bits)
    quantized = np.round(array * scale)
    quantized = np.clip(quantized, info.min, info.max)
    return quantized.astype(dtype)


def fold_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> tuple[np.ndarray, np.ndarray]:
    weight = conv.weight.detach().cpu()
    if conv.bias is None:
        bias = torch.zeros(conv.out_channels)
    else:
        bias = conv.bias.detach().cpu()

    gamma = bn.weight.detach().cpu()
    beta = bn.bias.detach().cpu()
    mean = bn.running_mean.detach().cpu()
    var = bn.running_var.detach().cpu()
    scale = gamma / torch.sqrt(var + bn.eps)

    folded_weight = weight * scale.reshape(-1, 1, 1, 1)
    folded_bias = beta + (bias - mean) * scale
    return folded_weight.numpy().astype(np.float32), folded_bias.numpy().astype(np.float32)


def fold_linear_bn(linear: nn.Linear, bn: nn.BatchNorm1d) -> tuple[np.ndarray, np.ndarray]:
    weight = linear.weight.detach().cpu()
    if linear.bias is None:
        bias = torch.zeros(linear.out_features)
    else:
        bias = linear.bias.detach().cpu()

    gamma = bn.weight.detach().cpu()
    beta = bn.bias.detach().cpu()
    mean = bn.running_mean.detach().cpu()
    var = bn.running_var.detach().cpu()
    scale = gamma / torch.sqrt(var + bn.eps)

    folded_weight = weight * scale.reshape(-1, 1)
    folded_bias = beta + (bias - mean) * scale
    return folded_weight.numpy().astype(np.float32), folded_bias.numpy().astype(np.float32)


def conv_layer_from_seq(name: str, seq: nn.Sequential) -> tuple[LayerSpec, dict[str, np.ndarray]]:
    conv = seq[0]
    bn = seq[1]
    prelu = seq[2] if len(seq) > 2 and isinstance(seq[2], nn.PReLU) else None
    if not isinstance(conv, nn.Conv2d) or not isinstance(bn, nn.BatchNorm2d):
        raise TypeError(f"{name} is not Conv2d+BatchNorm2d")

    weight, bias = fold_conv_bn(conv, bn)
    arrays: dict[str, np.ndarray] = {
        f"{name}_weight": weight,
        f"{name}_bias": bias,
    }
    prelu_key = None
    if prelu is not None:
        prelu_key = f"{name}_prelu"
        arrays[prelu_key] = prelu.weight.detach().cpu().numpy().astype(np.float32)

    spec = LayerSpec(
        name=name,
        op="conv2d",
        weight_key=f"{name}_weight",
        bias_key=f"{name}_bias",
        prelu_key=prelu_key,
        weight_shape=list(weight.shape),
        bias_shape=list(bias.shape),
        stride=list(conv.stride),
        padding=list(conv.padding),
        groups=conv.groups,
    )
    return spec, arrays


def collect_folded_layers(model: TinyFaceMatch128) -> tuple[list[LayerSpec], dict[str, np.ndarray]]:
    layers: list[LayerSpec] = []
    arrays: dict[str, np.ndarray] = {}

    def add_conv(name: str, seq: nn.Sequential) -> None:
        spec, layer_arrays = conv_layer_from_seq(name, seq)
        layers.append(spec)
        arrays.update(layer_arrays)

    add_conv("conv0", model.features[0])
    add_conv("dwsep0_dw", model.features[1].depthwise)
    add_conv("dwsep0_pw", model.features[1].pointwise)

    for block_index, feature_index in enumerate(range(2, 10)):
        block = model.features[feature_index]
        add_conv(f"b{block_index}_expand", block.layers[0])
        add_conv(f"b{block_index}_dw", block.layers[1])
        add_conv(f"b{block_index}_project", block.layers[2])

    add_conv("conv_final", model.features[10])
    add_conv("global_dw7", model.global_depthwise.layers[:2])

    dense_weight, dense_bias = fold_linear_bn(model.embedding, model.embedding_bn)
    arrays["dense_weight"] = dense_weight
    arrays["dense_bias"] = dense_bias
    layers.append(
        LayerSpec(
            name="dense",
            op="linear",
            weight_key="dense_weight",
            bias_key="dense_bias",
            prelu_key=None,
            weight_shape=list(dense_weight.shape),
            bias_shape=list(dense_bias.shape),
        )
    )

    return layers, arrays


def write_binary(
    path: Path,
    layers: list[LayerSpec],
    quantized_arrays: dict[str, np.ndarray],
) -> None:
    offset = 0
    with path.open("wb") as handle:
        for layer in layers:
            layer.offset_bytes = offset

            weight = np.ascontiguousarray(quantized_arrays[layer.weight_key])
            weight_bytes = weight.tobytes(order="C")
            handle.write(weight_bytes)
            layer.weight_bytes = len(weight_bytes)
            offset += len(weight_bytes)

            bias = np.ascontiguousarray(quantized_arrays[layer.bias_key])
            bias_bytes = bias.tobytes(order="C")
            handle.write(bias_bytes)
            layer.bias_bytes = len(bias_bytes)
            offset += len(bias_bytes)

            if layer.prelu_key is not None:
                prelu = np.ascontiguousarray(quantized_arrays[layer.prelu_key])
                prelu_bytes = prelu.tobytes(order="C")
                handle.write(prelu_bytes)
                layer.prelu_bytes = len(prelu_bytes)
                offset += len(prelu_bytes)


def write_offsets_header(path: Path, manifest: dict) -> None:
    guard = "TINYFACEMATCH_HLS_WEIGHTS_HPP"
    lines = [
        f"#ifndef {guard}",
        f"#define {guard}",
        "",
        "#include <cstdint>",
        "",
        f"static constexpr int TFM_WEIGHT_FRAC_BITS = {manifest['quantization']['weight_frac_bits']};",
        f"static constexpr int TFM_BIAS_FRAC_BITS = {manifest['quantization']['bias_frac_bits']};",
        f"static constexpr int TFM_PRELU_FRAC_BITS = {manifest['quantization']['prelu_frac_bits']};",
        f"static constexpr int TFM_ACT_FRAC_BITS = {manifest['quantization']['activation_frac_bits']};",
        f"static constexpr int TFM_OUT_FRAC_BITS = {manifest['quantization']['output_frac_bits']};",
        f"static constexpr int TFM_WEIGHTS_TOTAL_BYTES = {manifest['total_weight_bytes']};",
        "",
    ]
    for layer in manifest["layers"]:
        prefix = layer["name"].upper()
        lines.extend(
            [
                f"static constexpr int TFM_{prefix}_OFFSET_BYTES = {layer['offset_bytes']};",
                f"static constexpr int TFM_{prefix}_WEIGHT_BYTES = {layer['weight_bytes']};",
                f"static constexpr int TFM_{prefix}_BIAS_BYTES = {layer['bias_bytes']};",
                f"static constexpr int TFM_{prefix}_PRELU_BYTES = {layer['prelu_bytes']};",
                "",
            ]
        )
    lines.append(f"#endif // {guard}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fold and quantize TinyFaceMatch128 weights for HLS.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--weight-frac-bits", type=int, default=12)
    parser.add_argument("--bias-frac-bits", type=int, default=20)
    parser.add_argument("--prelu-frac-bits", type=int, default=12)
    parser.add_argument("--activation-frac-bits", type=int, default=10)
    parser.add_argument("--output-frac-bits", type=int, default=14)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = checkpoint.get("config", {})
    width_mult = float(config.get("width_mult", 0.5))

    model = TinyFaceMatch128(width_mult=width_mult)
    model.load_state_dict(checkpoint["backbone_state_dict"])
    model.eval()

    layers, folded_arrays = collect_folded_layers(model)

    quantized_arrays: dict[str, np.ndarray] = {}
    debug_arrays: dict[str, np.ndarray] = {}
    for key, array in folded_arrays.items():
        debug_arrays[key] = array.astype(np.float32)
        if key.endswith("_bias"):
            quantized_arrays[key] = quantize_signed(array, args.bias_frac_bits, np.int32)
        elif key.endswith("_prelu"):
            quantized_arrays[key] = quantize_signed(array, args.prelu_frac_bits, np.int16)
        else:
            quantized_arrays[key] = quantize_signed(array, args.weight_frac_bits, np.int16)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(args.output_dir / "weights_quantized.npz", **quantized_arrays)
    np.savez(args.output_dir / "weights_folded_fp32.npz", **debug_arrays)

    weights_bin = args.output_dir / "weights.bin"
    write_binary(weights_bin, layers, quantized_arrays)

    manifest = {
        "architecture": "TinyFaceMatch128",
        "checkpoint": str(args.checkpoint),
        "width_mult": width_mult,
        "input": {
            "height": 112,
            "width": 112,
            "channels": 3,
            "layout": "HWC_RGB_uint8_for_HLS_input",
            "normalization": "(pixel - 127.5) / 128.0",
        },
        "output": {
            "embedding_dim": 128,
            "description": "L2-normalized embedding quantized with output_frac_bits",
        },
        "quantization": {
            "weight_dtype": "int16",
            "weight_frac_bits": args.weight_frac_bits,
            "bias_dtype": "int32",
            "bias_frac_bits": args.bias_frac_bits,
            "prelu_dtype": "int16",
            "prelu_frac_bits": args.prelu_frac_bits,
            "activation_dtype": "ap_fixed<16,6>",
            "activation_frac_bits": args.activation_frac_bits,
            "output_dtype": "int16",
            "output_frac_bits": args.output_frac_bits,
        },
        "layers": [asdict(layer) for layer in layers],
        "total_weight_bytes": weights_bin.stat().st_size,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_offsets_header(args.output_dir / "weights_offsets.hpp", manifest)

    print(f"checkpoint: {args.checkpoint}")
    print(f"output_dir: {args.output_dir}")
    print(f"layers: {len(layers)}")
    print(f"weights.bin bytes: {weights_bin.stat().st_size}")
    print(f"wrote: {args.output_dir / 'manifest.json'}")
    print(f"wrote: {args.output_dir / 'weights_offsets.hpp'}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tinyfacematch_pytorch.eval_hls_prepared import PreparedTinyFaceMatchReference
from tinyfacematch_pytorch.eval_pairs import collect_images_by_class, load_image
from tinyfacematch_pytorch.data import PROJECT_MEAN, PROJECT_SCALE


def image_tensor_to_uint8_hwc(tensor: torch.Tensor) -> np.ndarray:
    chw = tensor.numpy()
    hwc = np.transpose(chw, (1, 2, 0))
    image = np.round(hwc * PROJECT_SCALE + PROJECT_MEAN)
    return np.clip(image, 0, 255).astype(np.uint8)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export HLS C-sim test vectors.")
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=1)
    return parser


@torch.no_grad()
def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    images_by_class = collect_images_by_class(args.data_dir)
    selected_paths = []
    for class_name in sorted(images_by_class):
        selected_paths.extend(images_by_class[class_name])
        if len(selected_paths) >= args.count:
            break

    model = PreparedTinyFaceMatchReference(args.prepared_dir).eval()
    metadata = []
    for index, path in enumerate(selected_paths[: args.count]):
        tensor = load_image(path)
        image = image_tensor_to_uint8_hwc(tensor)
        embedding = model(tensor.unsqueeze(0)).squeeze(0).numpy()
        expected = np.round(embedding * (1 << 14))
        expected = np.clip(expected, -32768, 32767).astype(np.int16)

        input_path = args.output_dir / f"input_{index}.bin"
        expected_path = args.output_dir / f"expected_{index}.bin"
        image.tofile(input_path)
        expected.tofile(expected_path)
        metadata.append(
            {
                "index": index,
                "source": str(path),
                "input": str(input_path),
                "expected": str(expected_path),
            }
        )

    (args.output_dir / "manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"wrote {len(metadata)} test vector(s) to {args.output_dir}")


if __name__ == "__main__":
    main()

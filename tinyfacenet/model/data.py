from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import Tensor
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".ppm"}
PROJECT_MEAN = 127.5
PROJECT_SCALE = 128.0


@dataclass(frozen=True)
class Sample:
    path: Path
    label: int


def find_image_classes(root: Path) -> tuple[list[str], dict[str, int]]:
    classes = sorted(path.name for path in root.iterdir() if path.is_dir())
    if not classes:
        raise ValueError(f"no identity folders found in {root}")
    return classes, {class_name: index for index, class_name in enumerate(classes)}


def collect_samples(root: Path, class_to_index: dict[str, int]) -> list[Sample]:
    samples: list[Sample] = []
    for class_name, label in class_to_index.items():
        class_dir = root / class_name
        for path in sorted(class_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append(Sample(path=path, label=label))
    if not samples:
        raise ValueError(f"no images found in {root}")
    return samples


class FaceFolderDataset(Dataset[tuple[Tensor, Tensor]]):
    """ImageFolder-style face dataset.

    Expected layout:

        data_root/
          identity_000/
            img1.jpg
            img2.jpg
          identity_001/
            img1.jpg

    Images are converted to RGB, resized to 112x112, normalized with
    (pixel - 127.5) / 128.0, and returned as CHW float32 tensors.
    """

    def __init__(
        self,
        root: str | Path,
        image_size: int = 112,
        augment: bool = False,
        repeat: int = 1,
    ) -> None:
        self.root = Path(root)
        self.image_size = image_size
        self.augment = augment
        self.repeat = max(1, int(repeat))
        self.classes, self.class_to_index = find_image_classes(self.root)
        self.samples = collect_samples(self.root, self.class_to_index)

    def __len__(self) -> int:
        return len(self.samples) * self.repeat

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        sample = self.samples[index % len(self.samples)]
        image = Image.open(sample.path).convert("RGB")
        image = self._transform(image)
        array = np.asarray(image, dtype=np.float32)
        array = (array - PROJECT_MEAN) / PROJECT_SCALE
        chw = np.transpose(array, (2, 0, 1))
        tensor = torch.from_numpy(np.ascontiguousarray(chw))
        label = torch.tensor(sample.label, dtype=torch.long)
        return tensor, label

    def _transform(self, image: Image.Image) -> Image.Image:
        if self.augment:
            if random.random() < 0.5:
                image = ImageOps.mirror(image)
            scale = random.uniform(0.92, 1.0)
            width, height = image.size
            crop_w = max(1, int(width * scale))
            crop_h = max(1, int(height * scale))
            left = random.randint(0, max(0, width - crop_w))
            top = random.randint(0, max(0, height - crop_h))
            image = image.crop((left, top, left + crop_w, top + crop_h))

        resample = Image.Resampling.BILINEAR
        return image.resize((self.image_size, self.image_size), resample=resample)

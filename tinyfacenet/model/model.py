from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class ConvBnPReLU(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
        groups: int = 1,
    ) -> None:
        if padding is None:
            padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),
        )


class ConvBn(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int | None = None,
        groups: int = 1,
    ) -> None:
        if padding is None:
            padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.depthwise = ConvBnPReLU(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            groups=in_channels,
        )
        self.pointwise = ConvBnPReLU(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.pointwise(self.depthwise(x))


class ResidualBottleneckBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expansion: int = 2,
    ) -> None:
        super().__init__()
        hidden_channels = in_channels * expansion
        self.use_residual = stride == 1 and in_channels == out_channels
        self.layers = nn.Sequential(
            ConvBnPReLU(
                in_channels,
                hidden_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            ConvBnPReLU(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                stride=stride,
                groups=hidden_channels,
            ),
            ConvBn(
                hidden_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self.layers(x)
        if self.use_residual:
            out = out + x
        return out


class GlobalDepthwiseConv(nn.Module):
    def __init__(self, channels: int, spatial_size: int = 7) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=spatial_size,
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.Flatten(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class TinyFaceMatch128(nn.Module):
    """TinyFaceMatch-style embedding backbone with a fixed 128-D output."""

    embedding_dim = 128

    def __init__(self, width_mult: float = 0.5) -> None:
        super().__init__()
        c32 = self._make_divisible(32 * width_mult)
        c64 = self._make_divisible(64 * width_mult)
        c128 = self._make_divisible(128 * width_mult)
        c256 = self._make_divisible(256 * width_mult)
        self.channels = {
            "c32": c32,
            "c64": c64,
            "c128": c128,
            "c256": c256,
        }

        self.features = nn.Sequential(
            ConvBnPReLU(3, c32, kernel_size=3, stride=2),
            DepthwiseSeparableBlock(c32, c64, stride=1),
            ResidualBottleneckBlock(c64, c64, stride=2, expansion=2),
            ResidualBottleneckBlock(c64, c64, stride=1, expansion=2),
            ResidualBottleneckBlock(c64, c64, stride=1, expansion=2),
            ResidualBottleneckBlock(c64, c128, stride=2, expansion=2),
            ResidualBottleneckBlock(c128, c128, stride=1, expansion=2),
            ResidualBottleneckBlock(c128, c128, stride=1, expansion=2),
            ResidualBottleneckBlock(c128, c128, stride=2, expansion=2),
            ResidualBottleneckBlock(c128, c128, stride=1, expansion=2),
            ConvBnPReLU(c128, c256, kernel_size=1, stride=1, padding=0),
        )
        self.global_depthwise = GlobalDepthwiseConv(c256, spatial_size=7)
        self.embedding = nn.Linear(c256, self.embedding_dim, bias=False)
        self.embedding_bn = nn.BatchNorm1d(self.embedding_dim)
        self.apply(self._init_weights)

    @staticmethod
    def _make_divisible(value: float, divisor: int = 8) -> int:
        return max(divisor, int(value + divisor / 2) // divisor * divisor)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(module, nn.BatchNorm2d | nn.BatchNorm1d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.global_depthwise(x)
        x = self.embedding(x)
        x = self.embedding_bn(x)
        return F.normalize(x, p=2, dim=1)


class ArcFaceClassifier(nn.Module):
    """Training-only ArcFace head. Do not export this for FPGA inference."""

    def __init__(
        self,
        num_classes: int,
        embedding_dim: int = 128,
        scale: float = 64.0,
        margin: float = 0.5,
        easy_margin: bool = False,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.scale = scale
        self.margin = margin
        self.easy_margin = easy_margin
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings: Tensor, labels: Tensor | None = None) -> Tensor:
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        if labels is None:
            return cosine * self.scale

        sine = torch.sqrt(torch.clamp(1.0 - cosine.pow(2), min=0.0, max=1.0))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)
        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return logits * self.scale


class TinyFaceMatchTrainingModel(nn.Module):
    """Backbone plus ArcFace head for training."""

    def __init__(
        self,
        num_classes: int,
        width_mult: float = 0.5,
        scale: float = 64.0,
        margin: float = 0.5,
    ) -> None:
        super().__init__()
        self.backbone = TinyFaceMatch128(width_mult=width_mult)
        self.classifier = ArcFaceClassifier(
            num_classes=num_classes,
            embedding_dim=TinyFaceMatch128.embedding_dim,
            scale=scale,
            margin=margin,
        )

    def forward(self, x: Tensor, labels: Tensor | None = None) -> tuple[Tensor, Tensor]:
        embeddings = self.backbone(x)
        logits = self.classifier(embeddings, labels)
        return embeddings, logits


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())

"""Part 2: visual representation learning on Genesis/LeRobot observations.

This file intentionally keeps the phase-2 path narrow:

- load the Genesis Franka pick-and-place LeRobot dataset from phase 1;
- use small collate functions to expose image, class, and xy labels;
- train either a small 4-layer CNN or ResNet-18 for color classification and
  object xy regression;
- expose ``encoder.encode(image) -> (pooled_feature, patch_tokens)`` for the
  behavior-cloning, Transformer, and DiT chapters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from env import COLOR_NAMES
from utils import (
    VisionAugmentation,
    _batch_to_device,
    extract_features,
    make_robustness_vision_collates,
    make_vision_collate,
    pca_2d,
    tsne_2d,
)


@dataclass
class VisionMetrics:
    loss: float
    class_loss: float
    xy_loss: float
    accuracy: float
    xy_error: float


class SmallVisionEncoder(nn.Module):
    """Readable 4-layer CNN with pooled and patch-token outputs."""

    def __init__(self, num_classes: int = len(COLOR_NAMES), feature_dim: int = 64) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.conv = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 48, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, feature_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.class_head = nn.Linear(feature_dim, num_classes)
        self.xy_head = nn.Linear(feature_dim, 2)

    def encode(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feature_map = self.conv(image)
        pooled = self.pool(feature_map).flatten(1)
        patch_tokens = feature_map.flatten(2).transpose(1, 2)
        return pooled, patch_tokens

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        pooled, patch_tokens = self.encode(image)
        return {
            "class_logits": self.class_head(pooled),
            "xy": self.xy_head(pooled).sigmoid(),
            "pooled": pooled,
            "patch_tokens": patch_tokens,
            "patches": patch_tokens,
        }


class ResNet18VisionEncoder(nn.Module):
    """ResNet-18 with classification/regression heads and patch tokens."""

    def __init__(
        self,
        num_classes: int = len(COLOR_NAMES),
        feature_dim: int = 512,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        resnet = resnet18(weights=weights)
        self.pretrained = pretrained
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None])
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layers = nn.Sequential(resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)
        self.project = nn.Identity() if feature_dim == 512 else nn.Conv2d(512, feature_dim, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.class_head = nn.Linear(feature_dim, num_classes)
        self.xy_head = nn.Linear(feature_dim, 2)

    def encode(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.pretrained:
            image = (image - self.mean) / self.std
        feature_map = self.project(self.layers(self.stem(image)))
        pooled = self.pool(feature_map).flatten(1)
        patch_tokens = feature_map.flatten(2).transpose(1, 2)
        return pooled, patch_tokens

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        pooled, patch_tokens = self.encode(image)
        return {
            "class_logits": self.class_head(pooled),
            "xy": self.xy_head(pooled).sigmoid(),
            "pooled": pooled,
            "patch_tokens": patch_tokens,
            "patches": patch_tokens,
        }


def build_vision_encoder(
    name: str = "small",
    num_classes: int = len(COLOR_NAMES),
    feature_dim: int = 64,
    pretrained: bool = True,
) -> nn.Module:
    if name == "small":
        return SmallVisionEncoder(num_classes=num_classes, feature_dim=feature_dim)
    if name == "resnet18":
        return ResNet18VisionEncoder(num_classes=num_classes, feature_dim=feature_dim, pretrained=pretrained)
    raise ValueError(f"unknown encoder: {name}")


# Training and evaluation


def compute_vision_losses(
    outputs: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
    xy_weight: float = 10.0,
) -> dict[str, torch.Tensor]:
    class_loss = nn.functional.cross_entropy(outputs["class_logits"], batch["label"].long())
    xy_loss = nn.functional.mse_loss(outputs["xy"], batch["xy"].float())
    return {"total": class_loss + xy_weight * xy_loss, "class": class_loss, "xy": xy_loss}


def vision_loss(
    outputs: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
    xy_weight: float = 10.0,
) -> torch.Tensor:
    return compute_vision_losses(outputs, batch, xy_weight)["total"]


def _empty_metrics() -> VisionMetrics:
    return VisionMetrics(loss=0.0, class_loss=0.0, xy_loss=0.0, accuracy=0.0, xy_error=0.0)


def train_vision_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: str | torch.device = "cuda",
    xy_weight: float = 10.0,
) -> VisionMetrics:
    model.to(device)
    model.train()
    totals = {"loss": 0.0, "class_loss": 0.0, "xy_loss": 0.0, "correct": 0.0, "xy_error": 0.0, "count": 0}
    for raw_batch in loader:
        batch = _batch_to_device(raw_batch, device)
        optimizer.zero_grad()
        outputs = model(batch["image"])
        losses = compute_vision_losses(outputs, batch, xy_weight)
        losses["total"].backward()
        optimizer.step()
        _accumulate_metrics(totals, outputs, batch, losses)
    return _metrics_from_totals(totals)


@torch.no_grad()
def evaluate_vision(
    model: nn.Module,
    loader,
    device: str | torch.device = "cuda",
    xy_weight: float = 10.0,
) -> VisionMetrics:
    model.to(device)
    model.eval()
    totals = {"loss": 0.0, "class_loss": 0.0, "xy_loss": 0.0, "correct": 0.0, "xy_error": 0.0, "count": 0}
    for raw_batch in loader:
        batch = _batch_to_device(raw_batch, device)
        outputs = model(batch["image"])
        losses = compute_vision_losses(outputs, batch, xy_weight)
        _accumulate_metrics(totals, outputs, batch, losses)
    return _metrics_from_totals(totals)


def _accumulate_metrics(
    totals: dict[str, float],
    outputs: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
    losses: Mapping[str, torch.Tensor],
) -> None:
    n = int(batch["label"].shape[0])
    totals["loss"] += float(losses["total"].item()) * n
    totals["class_loss"] += float(losses["class"].item()) * n
    totals["xy_loss"] += float(losses["xy"].item()) * n
    totals["correct"] += float((outputs["class_logits"].argmax(dim=-1) == batch["label"]).sum().item())
    totals["xy_error"] += float((outputs["xy"] - batch["xy"]).norm(dim=-1).sum().item())
    totals["count"] += n


def _metrics_from_totals(totals: dict[str, float]) -> VisionMetrics:
    count = max(int(totals["count"]), 1)
    if totals["count"] == 0:
        return _empty_metrics()
    return VisionMetrics(
        loss=totals["loss"] / count,
        class_loss=totals["class_loss"] / count,
        xy_loss=totals["xy_loss"] / count,
        accuracy=totals["correct"] / count,
        xy_error=totals["xy_error"] / count,
    )


@torch.no_grad()
def evaluate_robustness(
    model: nn.Module,
    loaders: Mapping[str, object],
    device: str | torch.device = "cuda",
    xy_weight: float = 10.0,
) -> dict[str, VisionMetrics]:
    return {
        name: evaluate_vision(model, loader, device=device, xy_weight=xy_weight)
        for name, loader in loaders.items()
    }

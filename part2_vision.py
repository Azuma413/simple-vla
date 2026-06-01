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
from functools import partial
from typing import Callable, Mapping

import numpy as np
import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms import ColorJitter
from torchvision.transforms import functional as TF

from env import (
    COLOR_NAMES,
    COLORS,
    IMAGE_KEY,
    OBJECT_XYZ_KEY,
    TinyPickPlaceConfig,
)


SampleTransform = Callable[[dict[str, torch.Tensor | str]], dict[str, torch.Tensor | str]]


@dataclass
class VisionMetrics:
    loss: float
    class_loss: float
    xy_loss: float
    accuracy: float
    xy_error: float


def _as_chw_float(image: torch.Tensor | np.ndarray) -> torch.Tensor:
    if not isinstance(image, torch.Tensor):
        image = np.asarray(image)
    image = torch.as_tensor(image)
    if image.ndim == 4:
        image = image[0]
    if image.shape[-1] == 3:
        image = image.permute(2, 0, 1)
    image = image.float()
    if image.max() > 2.0:
        image = image / 255.0
    return image.clamp(0.0, 1.0)


def workspace_xy_to_unit(xyz: torch.Tensor, config: TinyPickPlaceConfig | None = None) -> torch.Tensor:
    cfg = config or TinyPickPlaceConfig()
    xyz = torch.as_tensor(xyz, dtype=torch.float32).flatten()
    return torch.tensor(
        [
            (xyz[0] - cfg.x_range[0]) / (cfg.x_range[1] - cfg.x_range[0]),
            (xyz[1] - cfg.y_range[0]) / (cfg.y_range[1] - cfg.y_range[0]),
        ],
        dtype=torch.float32,
    ).clamp(0.0, 1.0)


def _label_from_item(item: Mapping[str, object], label_key: str = "label") -> torch.Tensor:
    return torch.as_tensor(item[label_key], dtype=torch.long).flatten()[0]


def _xy_from_item(
    item: Mapping[str, object],
    xy_key: str = OBJECT_XYZ_KEY,
    config: TinyPickPlaceConfig | None = None,
) -> torch.Tensor:
    xy_or_xyz = torch.as_tensor(item[xy_key], dtype=torch.float32).flatten()
    if xy_or_xyz.numel() == 2:
        return xy_or_xyz.clamp(0.0, 1.0)
    return workspace_xy_to_unit(xy_or_xyz, config)


def vision_sample_from_lerobot_row(
    row: Mapping[str, object],
    image_key: str = IMAGE_KEY,
    label_key: str = "label",
    xy_key: str = OBJECT_XYZ_KEY,
    config: TinyPickPlaceConfig | None = None,
    condition: str = "lerobot",
) -> dict[str, torch.Tensor | str]:
    """Convert one standard LeRobot row into the minimal phase-2 fields."""

    return {
        "image": _as_chw_float(row[image_key]),
        "label": _label_from_item(row, label_key),
        "xy": _xy_from_item(row, xy_key, config or TinyPickPlaceConfig()),
        "condition": condition,
    }


def object_color_mask(image: torch.Tensor, label: int | torch.Tensor, tolerance: float = 0.32) -> torch.Tensor:
    label = int(torch.as_tensor(label).flatten()[0].item())
    color = COLORS[label].to(image.device, image.dtype)[:, None, None]
    return (image - color).pow(2).sum(dim=0).sqrt() < tolerance


def replace_background(image: torch.Tensor, label: int | torch.Tensor, background: torch.Tensor) -> torch.Tensor:
    mask = object_color_mask(image, label)[None]
    return torch.where(mask, image, background.to(image.device, image.dtype).expand_as(image))


def _apply_visual_condition(
    image: torch.Tensor,
    label: torch.Tensor,
    lighting_range: tuple[float, float] = (1.0, 1.0),
    background_color: tuple[float, float, float] | None = None,
    background_noise: float = 0.0,
    image_noise: float = 0.0,
) -> torch.Tensor:
    lo, hi = lighting_range
    lighting = torch.empty(()).uniform_(lo, hi).item()
    image = (image * lighting).clamp(0.0, 1.0)
    if background_color is not None:
        background = torch.tensor(background_color, dtype=image.dtype)[:, None, None]
        image = replace_background(image, label, background)
    if background_noise > 0:
        noise = torch.rand_like(image) * background_noise
        mask = object_color_mask(image, label)[None]
        image = torch.where(mask, image, (image + noise).clamp(0.0, 1.0))
    if image_noise > 0:
        image = image + torch.randn_like(image) * image_noise
    return image.clamp(0.0, 1.0)


def collate_vision_batch(
    rows: list[Mapping[str, object]],
    image_key: str = IMAGE_KEY,
    label_key: str = "label",
    xy_key: str = OBJECT_XYZ_KEY,
    config: TinyPickPlaceConfig | None = None,
    sample_transform: SampleTransform | None = None,
    condition: str = "lerobot",
    lighting_range: tuple[float, float] = (1.0, 1.0),
    background_color: tuple[float, float, float] | None = None,
    background_noise: float = 0.0,
    image_noise: float = 0.0,
) -> dict[str, torch.Tensor | list[str]]:
    """Collate standard LeRobot rows into image, label, xy, and condition."""

    samples = []
    for row in rows:
        sample = vision_sample_from_lerobot_row(row, image_key, label_key, xy_key, config, condition)
        sample["image"] = _apply_visual_condition(
            torch.as_tensor(sample["image"], dtype=torch.float32),
            torch.as_tensor(sample["label"]),
            lighting_range=lighting_range,
            background_color=background_color,
            background_noise=background_noise,
            image_noise=image_noise,
        )
        if sample_transform is not None:
            sample = sample_transform(sample)
        samples.append(sample)
    return {
        "image": torch.stack([torch.as_tensor(sample["image"], dtype=torch.float32) for sample in samples]),
        "label": torch.stack([torch.as_tensor(sample["label"], dtype=torch.long) for sample in samples]),
        "xy": torch.stack([torch.as_tensor(sample["xy"], dtype=torch.float32) for sample in samples]),
        "condition": [str(sample["condition"]) for sample in samples],
    }


def make_vision_collate(**kwargs) -> Callable[[list[Mapping[str, object]]], dict[str, torch.Tensor | list[str]]]:
    return partial(collate_vision_batch, **kwargs)


def make_robustness_vision_collates() -> dict[str, Callable[[list[Mapping[str, object]]], dict[str, torch.Tensor | list[str]]]]:
    return {
        "train_distribution": make_vision_collate(condition="train"),
        "lighting": make_vision_collate(condition="lighting", lighting_range=(0.45, 1.55)),
        "background": make_vision_collate(condition="background", background_color=(0.05, 0.35, 0.32)),
        "texture_noise": make_vision_collate(condition="texture_noise", background_noise=0.45),
    }


class VisionAugmentation:
    """Color jitter, random crop, and background replacement for phase 2."""

    def __init__(
        self,
        image_size: int = 96,
        color_jitter: bool = True,
        random_crop: bool = True,
        background_replacement: bool = True,
        p_background: float = 0.5,
    ) -> None:
        self.image_size = image_size
        self.random_crop = random_crop
        self.background_replacement = background_replacement
        self.p_background = p_background
        self.jitter = ColorJitter(brightness=0.25, contrast=0.25, saturation=0.20, hue=0.03) if color_jitter else None

    def __call__(self, sample: dict[str, torch.Tensor | str]) -> dict[str, torch.Tensor | str]:
        image = torch.as_tensor(sample["image"], dtype=torch.float32)
        if self.background_replacement and torch.rand(()) < self.p_background:
            bg = torch.rand(3, 1, 1, dtype=image.dtype) * 0.75
            image = replace_background(image, torch.as_tensor(sample["label"]), bg)
        if self.random_crop:
            crop = int(self.image_size * float(torch.empty(()).uniform_(0.85, 1.0)))
            top = int(torch.randint(0, self.image_size - crop + 1, ()).item())
            left = int(torch.randint(0, self.image_size - crop + 1, ()).item())
            image = TF.resized_crop(
                image,
                top,
                left,
                crop,
                crop,
                [self.image_size, self.image_size],
            )
        if self.jitter is not None:
            image = self.jitter(image)
        sample = dict(sample)
        sample["image"] = image.clamp(0.0, 1.0)
        return sample


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
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layers = nn.Sequential(resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)
        self.project = nn.Identity() if feature_dim == 512 else nn.Conv2d(512, feature_dim, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.class_head = nn.Linear(feature_dim, num_classes)
        self.xy_head = nn.Linear(feature_dim, 2)

    def encode(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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


def _batch_to_device(batch: Mapping[str, torch.Tensor | str], device: str | torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}


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


@torch.no_grad()
def extract_features(
    model: nn.Module,
    loader,
    device: str | torch.device = "cuda",
    feature: str = "pooled",
    max_batches: int | None = None,
) -> dict[str, torch.Tensor | list[str]]:
    model.to(device)
    model.eval()
    features: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    xy: list[torch.Tensor] = []
    conditions: list[str] = []
    for batch_index, raw_batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = _batch_to_device(raw_batch, device)
        pooled, patch_tokens = model.encode(batch["image"])
        if feature == "pooled":
            chosen = pooled
        elif feature == "patch_mean":
            chosen = patch_tokens.mean(dim=1)
        else:
            raise ValueError(f"unknown feature type: {feature}")
        features.append(chosen.cpu())
        labels.append(batch["label"].cpu())
        xy.append(batch["xy"].cpu())
        if "condition" in raw_batch:
            condition = raw_batch["condition"]
            conditions.extend(list(condition) if isinstance(condition, (list, tuple)) else [str(condition)])
    return {
        "feature": torch.cat(features),
        "label": torch.cat(labels),
        "xy": torch.cat(xy),
        "condition": conditions,
    }


def pca_2d(features: torch.Tensor) -> torch.Tensor:
    features = features.float().cpu()
    centered = features - features.mean(dim=0, keepdim=True)
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    return centered @ vh[:2].T


def tsne_2d(features: torch.Tensor, perplexity: float = 30.0, seed: int = 0) -> torch.Tensor:
    from sklearn.manifold import TSNE

    embedding = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    ).fit_transform(features.float().cpu().numpy())
    return torch.as_tensor(embedding, dtype=torch.float32)

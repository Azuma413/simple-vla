"""Part 1: a tiny vision encoder for colored-object observations.

The dataset is synthetic on purpose: it runs without Genesis and gives us
class labels, image coordinates, pooled features, and patch tokens for later
chapters.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import Dataset


COLORS = torch.tensor(
    [
        [0.90, 0.10, 0.10],
        [0.10, 0.55, 0.95],
        [0.15, 0.75, 0.20],
    ],
    dtype=torch.float32,
)


@dataclass
class VisionBatch:
    image: torch.Tensor
    label: torch.Tensor
    xy: torch.Tensor


class ColoredSquareDataset(Dataset):
    """Images with one colored square and exact class/position labels."""

    def __init__(
        self,
        n: int = 512,
        image_size: int = 32,
        square_size: int = 6,
        num_classes: int = 3,
        lighting: float = 1.0,
        background: float = 0.05,
        noise: float = 0.01,
        seed: int = 0,
    ) -> None:
        generator = torch.Generator().manual_seed(seed)
        images = torch.full((n, 3, image_size, image_size), background)
        labels = torch.randint(num_classes, (n,), generator=generator)
        xy = torch.rand(n, 2, generator=generator)

        max_start = image_size - square_size - 1
        starts = (xy * max_start).long()
        for i, (row, col) in enumerate(starts):
            color = COLORS[labels[i]] * lighting
            images[i, :, row : row + square_size, col : col + square_size] = color[:, None, None]

        if noise > 0:
            images = images + noise * torch.randn(images.shape, generator=generator)

        self.images = images.clamp(0.0, 1.0)
        self.labels = labels
        self.xy = xy

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"image": self.images[index], "label": self.labels[index], "xy": self.xy[index]}


class SmallVisionEncoder(nn.Module):
    """A small CNN with both pooled and patch-token outputs."""

    def __init__(self, num_classes: int = 3, feature_dim: int = 64) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(48, feature_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.class_head = nn.Linear(feature_dim, num_classes)
        self.xy_head = nn.Linear(feature_dim, 2)

    def encode(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feature_map = self.conv(image)
        pooled = self.pool(feature_map).flatten(1)
        patches = feature_map.flatten(2).transpose(1, 2)
        return pooled, patches

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        pooled, patches = self.encode(image)
        return {
            "class_logits": self.class_head(pooled),
            "xy": self.xy_head(pooled).sigmoid(),
            "pooled": pooled,
            "patches": patches,
        }


def vision_loss(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> torch.Tensor:
    class_loss = nn.functional.cross_entropy(outputs["class_logits"], batch["label"])
    xy_loss = nn.functional.mse_loss(outputs["xy"], batch["xy"])
    return class_loss + xy_loss


def train_vision_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: str | torch.device = "cpu",
) -> float:
    model.train()
    total = 0.0
    count = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        loss = vision_loss(model(batch["image"]), batch)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(batch["label"])
        count += len(batch["label"])
    return total / max(count, 1)


@torch.no_grad()
def evaluate_vision(model: nn.Module, loader, device: str | torch.device = "cpu") -> dict[str, float]:
    model.eval()
    correct = 0
    count = 0
    xy_error = 0.0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(batch["image"])
        correct += (outputs["class_logits"].argmax(dim=-1) == batch["label"]).sum().item()
        xy_error += (outputs["xy"] - batch["xy"]).abs().sum(dim=-1).sum().item()
        count += len(batch["label"])
    return {"accuracy": correct / max(count, 1), "mean_xy_l1": xy_error / max(count, 1)}

"""Part 3: one-step behavior cloning baselines."""

from __future__ import annotations

import torch
from torch import nn

from part1_vision import SmallVisionEncoder


class StateMLPPolicy(nn.Module):
    """Privileged low-dimensional state -> one action."""

    def __init__(self, state_dim: int = 4, action_dim: int = 2, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class ImageBCPolicy(nn.Module):
    """Image -> CNN pooled feature -> one action."""

    def __init__(self, action_dim: int = 2, feature_dim: int = 64) -> None:
        super().__init__()
        self.encoder = SmallVisionEncoder(feature_dim=feature_dim)
        self.head = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        pooled, _ = self.encoder.encode(image)
        return self.head(pooled)


def train_bc_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    input_key: str,
    target_key: str = "action",
    device: str | torch.device = "cpu",
) -> float:
    model.train()
    total = 0.0
    count = 0
    for batch in loader:
        x = batch[input_key].to(device)
        y = batch[target_key].to(device)
        optimizer.zero_grad()
        loss = nn.functional.mse_loss(model(x), y)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(y)
        count += len(y)
    return total / max(count, 1)


@torch.no_grad()
def evaluate_bc_mse(
    model: nn.Module,
    loader,
    input_key: str,
    target_key: str = "action",
    device: str | torch.device = "cpu",
) -> float:
    model.eval()
    total = 0.0
    count = 0
    for batch in loader:
        x = batch[input_key].to(device)
        y = batch[target_key].to(device)
        loss = nn.functional.mse_loss(model(x), y, reduction="sum")
        total += loss.item()
        count += y.numel()
    return total / max(count, 1)

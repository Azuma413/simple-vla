"""Part 3: one-step behavior cloning baselines."""

from __future__ import annotations

import torch
from torch import nn

from env import FRANKA_QPOS_ACTION_DIM, GENESIS_STATE_DIM
from part2_vision import SmallVisionEncoder
from part1_simulator import rollout_policy


class StateMLPPolicy(nn.Module):
    """Privileged low-dimensional state -> one action."""

    def __init__(self, state_dim: int = GENESIS_STATE_DIM, action_dim: int = FRANKA_QPOS_ACTION_DIM, hidden_dim: int = 64) -> None:
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

    def __init__(self, action_dim: int = FRANKA_QPOS_ACTION_DIM, feature_dim: int = 64) -> None:
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


@torch.no_grad()
def evaluate_bc_rollout(
    model: nn.Module,
    dataset,
    input_key: str,
    n_episodes: int | None = 64,
    device: str | torch.device = "cpu",
    env=None,
    initial_states=None,
) -> dict[str, object]:
    """Evaluate a one-step BC policy by closed-loop task success."""

    model.eval()

    def policy(obs: dict[str, torch.Tensor | str]) -> torch.Tensor:
        x = obs[input_key]
        assert isinstance(x, torch.Tensor)
        action = model(x[None].to(device))
        return action[0].cpu()

    rollout_env = env or dataset
    if initial_states is None and env is not None and hasattr(dataset, "episode_initial_state"):
        total = getattr(getattr(dataset, "config", None), "n_episodes", n_episodes or 0)
        initial_states = [dataset.episode_initial_state(i) for i in range(min(n_episodes or total, total))]
    return rollout_policy(policy, rollout_env, initial_states=initial_states, n_episodes=n_episodes)

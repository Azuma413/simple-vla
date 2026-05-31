"""Part 4: single-stream Transformer policy with condition + action tokens."""

from __future__ import annotations

import torch
from torch import nn

from env import FRANKA_QPOS_ACTION_DIM
from part2_vision import SmallVisionEncoder
from part1_simulator import TinyPickPlaceDataset, rollout_policy


def make_block_attention_mask(num_condition: int, chunk_size: int, device=None) -> torch.Tensor:
    """Mask invalid attention pairs for [condition tokens, action tokens].

    True means "masked". Condition tokens cannot read action tokens. Action
    tokens can read condition tokens and past/current action tokens.
    """

    total = num_condition + chunk_size
    mask = torch.zeros(total, total, dtype=torch.bool, device=device)
    mask[:num_condition, num_condition:] = True
    for i in range(chunk_size):
        row = num_condition + i
        future_actions = slice(row + 1, total)
        mask[row, future_actions] = True
    return mask


class TransformerBlock(nn.Module):
    def __init__(self, dim: int = 64, heads: int = 4, mlp_ratio: int = 4) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )
        self.last_attention: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.norm1(x)
        attended, weights = self.attn(
            h,
            h,
            h,
            attn_mask=attn_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        self.last_attention = weights.detach()
        x = x + attended
        return x + self.mlp(self.norm2(x))


class ChunkTransformerPolicy(nn.Module):
    """CNN patch tokens + learned action queries -> action chunk."""

    def __init__(
        self,
        action_dim: int = FRANKA_QPOS_ACTION_DIM,
        chunk_size: int = 4,
        dim: int = 64,
        depth: int = 2,
        heads: int = 4,
    ) -> None:
        super().__init__()
        self.chunk_size = chunk_size
        self.encoder = SmallVisionEncoder(feature_dim=dim)
        self.action_tokens = nn.Parameter(torch.randn(chunk_size, dim) * 0.02)
        self.blocks = nn.ModuleList([TransformerBlock(dim, heads) for _ in range(depth)])
        self.head = nn.Linear(dim, action_dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        _, condition = self.encoder.encode(image)
        batch_size, num_condition, _ = condition.shape
        actions = self.action_tokens[None].expand(batch_size, -1, -1)
        tokens = torch.cat([condition, actions], dim=1)
        mask = make_block_attention_mask(num_condition, self.chunk_size, image.device)
        for block in self.blocks:
            tokens = block(tokens, mask)
        action_tokens = tokens[:, num_condition:]
        return self.head(action_tokens)


def train_chunk_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: str | torch.device = "cpu",
) -> float:
    model.train()
    total = 0.0
    count = 0
    for batch in loader:
        image = batch["image"].to(device)
        target = batch["action_chunk"].to(device)
        optimizer.zero_grad()
        loss = nn.functional.mse_loss(model(image), target)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(target)
        count += len(target)
    return total / max(count, 1)


@torch.no_grad()
def evaluate_chunk_mse(model: nn.Module, loader, device: str | torch.device = "cpu") -> float:
    model.eval()
    total = 0.0
    count = 0
    for batch in loader:
        target = batch["action_chunk"].to(device)
        loss = nn.functional.mse_loss(model(batch["image"].to(device)), target, reduction="sum")
        total += loss.item()
        count += target.numel()
    return total / max(count, 1)


@torch.no_grad()
def evaluate_chunk_rollout(
    model: nn.Module,
    dataset: TinyPickPlaceDataset,
    n_episodes: int | None = 64,
    device: str | torch.device = "cpu",
    execute_chunk: bool = False,
    env=None,
    initial_states=None,
) -> dict[str, object]:
    """Evaluate the Transformer chunk policy in the closed-loop task.

    By default only the first action of the predicted chunk is executed before
    replanning. Set ``execute_chunk=True`` to expose chunk-boundary jitter.
    """

    model.eval()
    pending: list[torch.Tensor] = []

    def policy(obs: dict[str, torch.Tensor | str]) -> torch.Tensor:
        nonlocal pending
        frame_index = obs.get("frame_index")
        if isinstance(frame_index, torch.Tensor) and int(frame_index) == 0:
            pending = []
        if execute_chunk and pending:
            return pending.pop(0)
        image = obs["image"]
        assert isinstance(image, torch.Tensor)
        chunk = model(image[None].to(device))[0].cpu()
        pending = [a for a in chunk[1:]] if execute_chunk else []
        return chunk[0]

    rollout_env = env or dataset
    if initial_states is None and env is not None and hasattr(dataset, "episode_initial_state"):
        total = getattr(getattr(dataset, "config", None), "n_episodes", n_episodes or 0)
        initial_states = [dataset.episode_initial_state(i) for i in range(min(n_episodes or total, total))]
    return rollout_policy(policy, rollout_env, initial_states=initial_states, n_episodes=n_episodes)


@torch.no_grad()
def export_attention_map(model: ChunkTransformerPolicy, image: torch.Tensor, action_token: int = 0) -> torch.Tensor:
    """Return action-token attention over CNN patch tokens from the last block."""

    model.eval()
    _ = model(image[None] if image.ndim == 3 else image)
    weights = model.blocks[-1].last_attention
    if weights is None:
        raise RuntimeError("No attention weights recorded. Run a forward pass first.")
    # [batch, heads, query, key] -> average heads for one action query.
    _, condition = model.encoder.encode(image[None] if image.ndim == 3 else image)
    query = condition.shape[1] + action_token
    return weights[0, :, query, : condition.shape[1]].mean(dim=0)


def chunk_boundary_jitter(action_chunks: torch.Tensor) -> dict[str, float]:
    """Measure speed and boundary discontinuity for predicted chunks."""

    if action_chunks.ndim == 2:
        action_chunks = action_chunks[None]
    step_speed = action_chunks.diff(dim=1).norm(dim=-1)
    boundary_jump = action_chunks[:, 1:] - action_chunks[:, :-1]
    return {
        "mean_step_speed": float(step_speed.mean().item()) if step_speed.numel() else 0.0,
        "mean_boundary_jitter": float(boundary_jump.norm(dim=-1).mean().item()) if boundary_jump.numel() else 0.0,
    }


def plot_expert_vs_rollout(expert_xy: torch.Tensor, rollout_xy: torch.Tensor):
    """Create a matplotlib trajectory plot for expert/rollout drift."""

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(expert_xy[:, 0], expert_xy[:, 1], label="expert", linewidth=2)
    ax.plot(rollout_xy[:, 0], rollout_xy[:, 1], label="rollout", linewidth=2)
    ax.scatter(expert_xy[0, 0], expert_xy[0, 1], marker="o", color="black", s=30)
    ax.scatter(expert_xy[-1, 0], expert_xy[-1, 1], marker="x", color="black", s=40)
    ax.set_aspect("equal", adjustable="box")
    ax.legend()
    return fig

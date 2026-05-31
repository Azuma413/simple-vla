"""Part 5: Flow Matching DiT action expert."""

from __future__ import annotations

import math

import torch
from torch import nn

from env import FRANKA_QPOS_ACTION_DIM
from part2_vision import SmallVisionEncoder
from part1_simulator import TinyPickPlaceDataset, rollout_policy
from part4_transformer import make_block_attention_mask


class AdaLNBlock(nn.Module):
    """Transformer block with timestep-conditioned LayerNorm modulation."""

    def __init__(self, dim: int = 64, heads: int = 4, mlp_ratio: int = 4) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 4))

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        shift1, scale1, shift2, scale2 = self.modulation(t_emb).chunk(4, dim=-1)
        h = self.norm1(x) * (1 + scale1[:, None]) + shift1[:, None]
        x = x + self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)[0]
        h = self.norm2(x) * (1 + scale2[:, None]) + shift2[:, None]
        return x + self.mlp(h)


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freq = torch.exp(-torch.arange(half, device=t.device) * math.log(10000.0) / half)
    args = t[:, None] * freq[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = nn.functional.pad(emb, (0, 1))
    return emb


class FlowMatchingDiT(nn.Module):
    """Predict velocity for noisy action chunks conditioned on image patches."""

    def __init__(
        self,
        action_dim: int = FRANKA_QPOS_ACTION_DIM,
        chunk_size: int = 4,
        dim: int = 64,
        depth: int = 2,
        heads: int = 4,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.dim = dim
        self.encoder = SmallVisionEncoder(feature_dim=dim)
        self.action_in = nn.Linear(action_dim, dim)
        self.time_mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList([AdaLNBlock(dim, heads) for _ in range(depth)])
        self.head = nn.Linear(dim, action_dim)

    def forward(self, image: torch.Tensor, noisy_action: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        _, condition = self.encoder.encode(image)
        action_tokens = self.action_in(noisy_action)
        tokens = torch.cat([condition, action_tokens], dim=1)
        t_emb = self.time_mlp(timestep_embedding(t, self.dim))
        mask = make_block_attention_mask(condition.shape[1], self.chunk_size, image.device)
        for block in self.blocks:
            tokens = block(tokens, t_emb, mask)
        return self.head(tokens[:, condition.shape[1] :])

    @torch.no_grad()
    def sample(self, image: torch.Tensor, steps: int = 8) -> torch.Tensor:
        x = torch.randn(image.shape[0], self.chunk_size, self.action_dim, device=image.device)
        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((image.shape[0],), i / steps, device=image.device)
            x = x + self(image, x, t) * dt
        return x


def flow_matching_loss(model: FlowMatchingDiT, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    clean = batch["action_chunk"]
    noise = torch.randn_like(clean)
    t = torch.rand(clean.shape[0], device=clean.device)
    noisy = (1 - t[:, None, None]) * noise + t[:, None, None] * clean
    target_velocity = clean - noise
    pred_velocity = model(batch["image"], noisy, t)
    return nn.functional.mse_loss(pred_velocity, target_velocity)


def train_flow_epoch(
    model: FlowMatchingDiT,
    loader,
    optimizer: torch.optim.Optimizer,
    device: str | torch.device = "cpu",
) -> float:
    model.train()
    total = 0.0
    count = 0
    for batch in loader:
        tensor_batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        optimizer.zero_grad()
        loss = flow_matching_loss(model, tensor_batch)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(tensor_batch["image"])
        count += len(tensor_batch["image"])
    return total / max(count, 1)


@torch.no_grad()
def evaluate_flow_rollout(
    model: FlowMatchingDiT,
    dataset: TinyPickPlaceDataset,
    n_episodes: int | None = 64,
    device: str | torch.device = "cpu",
    sample_steps: int = 8,
    execute_chunk: bool = False,
    env=None,
    initial_states=None,
) -> dict[str, object]:
    """Evaluate the Flow Matching DiT by sampling actions and rolling out."""

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
        chunk = model.sample(image[None].to(device), steps=sample_steps)[0].cpu()
        pending = [a for a in chunk[1:]] if execute_chunk else []
        return chunk[0]

    rollout_env = env or dataset
    if initial_states is None and env is not None and hasattr(dataset, "episode_initial_state"):
        total = getattr(getattr(dataset, "config", None), "n_episodes", n_episodes or 0)
        initial_states = [dataset.episode_initial_state(i) for i in range(min(n_episodes or total, total))]
    return rollout_policy(policy, rollout_env, initial_states=initial_states, n_episodes=n_episodes)


@torch.no_grad()
def sample_multiple_trajectories(
    model: FlowMatchingDiT,
    image: torch.Tensor,
    n_samples: int = 8,
    steps: int = 8,
) -> torch.Tensor:
    """Sample multiple action chunks for the same observation."""

    model.eval()
    image_batch = image[None].expand(n_samples, -1, -1, -1) if image.ndim == 3 else image.expand(n_samples, -1, -1, -1)
    return model.sample(image_batch.to(next(model.parameters()).device), steps=steps).cpu()


def plot_flow_samples(action_chunks: torch.Tensor):
    """Plot sampled Flow chunks in the first two action dimensions."""

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4, 4))
    chunks = action_chunks.detach().cpu()
    for chunk in chunks:
        if chunk.shape[-1] < 2:
            ax.plot(chunk[:, 0])
        else:
            ax.plot(chunk[:, 0], chunk[:, 1], alpha=0.75)
    ax.set_title("Flow samples")
    ax.set_aspect("equal", adjustable="box") if chunks.shape[-1] >= 2 else None
    return fig


def compare_flow_vs_mse_samples(mse_chunk: torch.Tensor, flow_chunks: torch.Tensor) -> dict[str, float]:
    """Numerical summary for the multimodality demo."""

    center = flow_chunks.mean(dim=0)
    return {
        "mse_to_flow_mean": float((mse_chunk - center).norm(dim=-1).mean().item()),
        "flow_sample_diversity": float(flow_chunks.std(dim=0).norm(dim=-1).mean().item()),
    }

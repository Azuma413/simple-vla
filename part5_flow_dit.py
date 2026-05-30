"""Part 5: Flow Matching DiT action expert."""

from __future__ import annotations

import math

import torch
from torch import nn

from part1_vision import SmallVisionEncoder
from part4_transformer_dit import make_block_attention_mask


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
        action_dim: int = 2,
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

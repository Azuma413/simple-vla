"""Part 4: single-stream Transformer policy with condition + action tokens."""

from __future__ import annotations

import torch
from torch import nn

from part1_vision import SmallVisionEncoder


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
        action_dim: int = 2,
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

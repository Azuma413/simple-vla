"""Part 6: frozen VLM-style prefix tokens connected to the same DiT body.

For a real VLM, replace FrozenTinyVLM with a wrapper that returns hidden states
from Qwen. The connector and DiT-side interface stay the same.
"""

from __future__ import annotations

import torch
from torch import nn

from part4_transformer_dit import TransformerBlock, make_block_attention_mask


class FrozenTinyVLM(nn.Module):
    """A small frozen stand-in for image+language hidden states."""

    def __init__(self, hidden_dim: int = 96, vocab: tuple[str, ...] = ("red", "blue", "green")) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab = vocab
        self.color_embed = nn.Embedding(len(vocab), hidden_dim)
        self.image_proj = nn.Conv2d(3, hidden_dim, kernel_size=4, stride=4)
        for parameter in self.parameters():
            parameter.requires_grad = False

    def _instruction_ids(self, instructions: list[str], device: torch.device) -> torch.Tensor:
        ids = []
        for text in instructions:
            ids.append(next((i for i, word in enumerate(self.vocab) if word in text), 0))
        return torch.tensor(ids, device=device)

    def forward(self, image: torch.Tensor, instructions: list[str]) -> torch.Tensor:
        patch_tokens = self.image_proj(image).flatten(2).transpose(1, 2)
        text_token = self.color_embed(self._instruction_ids(instructions, image.device))[:, None]
        return torch.cat([text_token, patch_tokens], dim=1)


class VLAConnectorPolicy(nn.Module):
    """Frozen VLM hidden states -> linear connector -> same single-stream DiT."""

    def __init__(
        self,
        vlm_hidden_dim: int = 96,
        dim: int = 64,
        action_dim: int = 2,
        chunk_size: int = 4,
        depth: int = 2,
        heads: int = 4,
    ) -> None:
        super().__init__()
        self.vlm = FrozenTinyVLM(hidden_dim=vlm_hidden_dim)
        self.connector = nn.Linear(vlm_hidden_dim, dim)
        self.chunk_size = chunk_size
        self.action_tokens = nn.Parameter(torch.randn(chunk_size, dim) * 0.02)
        self.blocks = nn.ModuleList([TransformerBlock(dim, heads) for _ in range(depth)])
        self.head = nn.Linear(dim, action_dim)

    def forward(self, image: torch.Tensor, instructions: list[str]) -> torch.Tensor:
        condition = self.connector(self.vlm(image, instructions))
        actions = self.action_tokens[None].expand(image.shape[0], -1, -1)
        tokens = torch.cat([condition, actions], dim=1)
        mask = make_block_attention_mask(condition.shape[1], self.chunk_size, image.device)
        for block in self.blocks:
            tokens = block(tokens, mask)
        return self.head(tokens[:, condition.shape[1] :])


def train_vla_epoch(
    model: VLAConnectorPolicy,
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
        instructions = batch["instruction"]
        optimizer.zero_grad()
        loss = nn.functional.mse_loss(model(image, instructions), target)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(target)
        count += len(target)
    return total / max(count, 1)

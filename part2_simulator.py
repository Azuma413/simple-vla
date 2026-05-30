"""Part 2: a tiny pick-and-place style expert dataset.

This file is not a physics simulator. It is a deterministic stand-in that
keeps the data contract stable while Genesis integration is added later.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from part1_vision import COLORS


@dataclass
class TinyPickPlaceConfig:
    n_episodes: int = 256
    horizon: int = 16
    action_dim: int = 2
    image_size: int = 32
    square_size: int = 6
    seed: int = 0


def expert_action_chunk(xy: torch.Tensor, target: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """Return a smooth chunk of delta-position actions from object to target."""

    delta = target - xy
    weights = torch.linspace(1.0, 0.25, chunk_size, device=xy.device)
    actions = delta[:, None, :] * weights[None, :, None] / chunk_size
    return actions


class TinyPickPlaceDataset(Dataset):
    """Synthetic robot dataset with image, state, language, and action chunks."""

    def __init__(self, config: TinyPickPlaceConfig | None = None, chunk_size: int = 4) -> None:
        self.config = config or TinyPickPlaceConfig()
        self.chunk_size = chunk_size
        generator = torch.Generator().manual_seed(self.config.seed)

        n = self.config.n_episodes
        self.label = torch.randint(3, (n,), generator=generator)
        self.xy = torch.rand(n, 2, generator=generator) * 0.8 + 0.1
        self.target = torch.rand(n, 2, generator=generator) * 0.8 + 0.1
        self.state = torch.cat([self.xy, self.target], dim=-1)
        self.action_chunk = expert_action_chunk(self.xy, self.target, chunk_size)
        self.image = self._render_images(generator)
        names = ["red", "blue", "green"]
        self.instruction = [f"pick the {names[int(i)]} cube" for i in self.label]

    def _render_images(self, generator: torch.Generator) -> torch.Tensor:
        size = self.config.image_size
        square = self.config.square_size
        image = torch.full((len(self.label), 3, size, size), 0.05)
        start = (self.xy * (size - square - 1)).long()
        for i, (row, col) in enumerate(start):
            image[i, :, row : row + square, col : col + square] = COLORS[self.label[i]][:, None, None]
        image = image + 0.01 * torch.randn(image.shape, generator=generator)
        return image.clamp(0.0, 1.0)

    def __len__(self) -> int:
        return len(self.label)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        return {
            "image": self.image[index],
            "state": self.state[index],
            "label": self.label[index],
            "xy": self.xy[index],
            "target": self.target[index],
            "action": self.action_chunk[index, 0],
            "action_chunk": self.action_chunk[index],
            "instruction": self.instruction[index],
        }


def collate_tiny_pick_place(batch: list[dict[str, torch.Tensor | str]]) -> dict[str, torch.Tensor | list[str]]:
    keys = batch[0].keys()
    output: dict[str, torch.Tensor | list[str]] = {}
    for key in keys:
        values = [item[key] for item in batch]
        if isinstance(values[0], torch.Tensor):
            output[key] = torch.stack(values)  # type: ignore[arg-type]
        else:
            output[key] = values  # type: ignore[assignment]
    return output

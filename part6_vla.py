"""Part 6: connect real Qwen VLM hidden states to the same DiT body."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from PIL import Image
from torch import nn

from env import FRANKA_QPOS_ACTION_DIM
from part4_transformer import TransformerBlock, make_block_attention_mask
from utils import rollout_policy


@dataclass
class QwenVLMConfig:
    model_id: str = "Qwen/Qwen3.5-0.8B"
    hidden_layer: int = -1
    max_condition_tokens: int = 128
    torch_dtype: str = "auto"
    device_map: str | None = "auto"


def _tensor_image_to_pil(image: torch.Tensor) -> Image.Image:
    array = (image.detach().cpu().permute(1, 2, 0).clamp(0.0, 1.0).numpy() * 255).astype("uint8")
    return Image.fromarray(array)


class QwenVLMBackbone(nn.Module):
    """Frozen Qwen image+text model that returns hidden-state tokens."""

    def __init__(self, config: QwenVLMConfig | None = None) -> None:
        super().__init__()
        self.config = config or QwenVLMConfig()
        try:
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:
            raise ImportError(
                "part6 uses a real VLM. Install dependencies with `uv sync` so "
                "`transformers`, `accelerate`, and `pillow` are available."
            ) from exc

        self.processor = AutoProcessor.from_pretrained(self.config.model_id, trust_remote_code=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.config.model_id,
            torch_dtype=self.config.torch_dtype,
            device_map=self.config.device_map,
            trust_remote_code=True,
        )
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.hidden_dim = int(getattr(self.model.config, "hidden_size", getattr(self.model.config, "text_config", self.model.config).hidden_size))

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _build_messages(self, images: torch.Tensor, instructions: list[str]) -> list[dict[str, Any]]:
        pil_images = [_tensor_image_to_pil(image) for image in images]
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": instruction},
                ],
            }
            for pil_image, instruction in zip(pil_images, instructions, strict=True)
        ]

    def _encode_inputs(self, images: torch.Tensor, instructions: list[str]) -> dict[str, torch.Tensor]:
        messages = self._build_messages(images, instructions)
        try:
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=False,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
        except Exception:
            pil_images = [_tensor_image_to_pil(image) for image in images]
            texts = [f"<image>\n{instruction}" for instruction in instructions]
            inputs = self.processor(text=texts, images=pil_images, return_tensors="pt", padding=True)
        return {key: value.to(self.device) for key, value in inputs.items() if isinstance(value, torch.Tensor)}

    @torch.no_grad()
    def forward(self, image: torch.Tensor, instructions: list[str]) -> torch.Tensor:
        inputs = self._encode_inputs(image, instructions)
        outputs = self.model(**inputs, output_hidden_states=True, use_cache=False, return_dict=True)
        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise RuntimeError("Qwen did not return hidden states. Check the installed transformers version.")
        tokens = hidden_states[self.config.hidden_layer]
        return tokens[:, : self.config.max_condition_tokens].detach()


class VLAConnectorPolicy(nn.Module):
    """Qwen hidden states -> linear connector -> same single-stream DiT."""

    def __init__(
        self,
        vlm: QwenVLMBackbone | None = None,
        qwen_config: QwenVLMConfig | None = None,
        dim: int = 64,
        action_dim: int = FRANKA_QPOS_ACTION_DIM,
        chunk_size: int = 4,
        depth: int = 2,
        heads: int = 4,
    ) -> None:
        super().__init__()
        self.vlm = vlm or QwenVLMBackbone(qwen_config)
        self.connector = nn.Linear(self.vlm.hidden_dim, dim)
        self.chunk_size = chunk_size
        self.action_tokens = nn.Parameter(torch.randn(chunk_size, dim) * 0.02)
        self.blocks = nn.ModuleList([TransformerBlock(dim, heads) for _ in range(depth)])
        self.head = nn.Linear(dim, action_dim)

    def forward(self, image: torch.Tensor, instructions: list[str]) -> torch.Tensor:
        condition = self.connector(self.vlm(image, instructions).to(self.connector.weight.device))
        actions = self.action_tokens[None].expand(image.shape[0], -1, -1)
        tokens = torch.cat([condition, actions], dim=1)
        mask = make_block_attention_mask(condition.shape[1], self.chunk_size, tokens.device)
        for block in self.blocks:
            tokens = block(tokens, mask.to(tokens.device))
        return self.head(tokens[:, condition.shape[1] :])


def freeze_vla_for_connector_training(model: VLAConnectorPolicy) -> None:
    """Freeze VLM and DiT body so only the linear connector trains."""

    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.connector.parameters():
        parameter.requires_grad = True
    model.vlm.eval()
    model.blocks.eval()
    model.head.eval()
    model.action_tokens.requires_grad = False


def connector_optimizer(model: VLAConnectorPolicy, lr: float = 1e-3) -> torch.optim.Optimizer:
    freeze_vla_for_connector_training(model)
    return torch.optim.AdamW(model.connector.parameters(), lr=lr)


def train_vla_connector_epoch(
    model: VLAConnectorPolicy,
    loader,
    optimizer: torch.optim.Optimizer,
    device: str | torch.device = "cpu",
) -> float:
    freeze_vla_for_connector_training(model)
    model.train()
    model.vlm.eval()
    model.blocks.eval()
    model.head.eval()
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


def train_vla_full_epoch(
    model: VLAConnectorPolicy,
    loader,
    optimizer: torch.optim.Optimizer,
    device: str | torch.device = "cpu",
) -> float:
    """Optional ablation that updates connector plus DiT body, never the VLM."""

    for parameter in model.vlm.parameters():
        parameter.requires_grad = False
    for name, parameter in model.named_parameters():
        if not name.startswith("vlm."):
            parameter.requires_grad = True
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


train_vla_epoch = train_vla_connector_epoch


@torch.no_grad()
def evaluate_vla_rollout(
    model: VLAConnectorPolicy,
    dataset,
    n_episodes: int | None = 64,
    device: str | torch.device = "cpu",
    env=None,
    initial_states=None,
) -> dict[str, object]:
    """Evaluate the Qwen-conditioned policy with image+instruction observations."""

    model.eval()

    def policy(obs: dict[str, torch.Tensor | str]) -> torch.Tensor:
        image = obs["image"]
        instruction = obs["instruction"]
        assert isinstance(image, torch.Tensor)
        assert isinstance(instruction, str)
        chunk = model(image[None].to(device), [instruction])[0].cpu()
        return chunk[0]

    rollout_env = env or dataset
    if initial_states is None and env is not None and hasattr(dataset, "episode_initial_state"):
        total = getattr(getattr(dataset, "config", None), "n_episodes", n_episodes or 0)
        initial_states = [dataset.episode_initial_state(i) for i in range(min(n_episodes or total, total))]
    return rollout_policy(policy, rollout_env, initial_states=initial_states, n_episodes=n_episodes)


@torch.no_grad()
def qwen_forward_smoke(
    model_id: str = "Qwen/Qwen3.5-0.8B",
    image_size: int = 64,
    device_map: str | None = "auto",
) -> dict[str, object]:
    """Load Qwen, run one image+instruction forward, and report hidden metadata."""

    config = QwenVLMConfig(model_id=model_id, max_condition_tokens=16, device_map=device_map)
    vlm = QwenVLMBackbone(config)
    image = torch.zeros(1, 3, image_size, image_size)
    image[:, 0] = 0.9
    hidden = vlm(image, ["pick the red cube"])
    return {
        "shape": tuple(hidden.shape),
        "dtype": str(hidden.dtype),
        "device": str(hidden.device),
        "hidden_dim": vlm.hidden_dim,
    }

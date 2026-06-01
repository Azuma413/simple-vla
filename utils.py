"""Small shared helpers used across the SimpleVLA phases."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Callable, Mapping, Protocol, Sequence

import numpy as np
import torch
from torchvision.transforms import ColorJitter
from torchvision.transforms import functional as TF

from env import COLORS, FRANKA_QPOS_ACTION_DIM, FRONT_IMAGE_KEY, OBJECT_XYZ_KEY, TinyPickPlaceConfig


SampleTransform = Callable[[dict[str, torch.Tensor | str]], dict[str, torch.Tensor | str]]


@dataclass
class LearningMetrics:
    loss: float
    mse: float | None = None
    accuracy: float | None = None


@dataclass
class RolloutMetrics:
    success_rate: float
    final_distance: float
    expert_drift: float


@dataclass
class GenerativeMetrics:
    sample_diversity: float
    trajectory_smoothness: float


def _as_chw_float(image: torch.Tensor | np.ndarray) -> torch.Tensor:
    if not isinstance(image, torch.Tensor):
        image = np.asarray(image)
    image = torch.as_tensor(image)
    if image.ndim == 4:
        image = image[0]
    if image.shape[-1] == 3:
        image = image.permute(2, 0, 1)
    image = image.float()
    if image.max() > 2.0:
        image = image / 255.0
    return image.clamp(0.0, 1.0)


def workspace_xy_to_unit(xyz: torch.Tensor, config: TinyPickPlaceConfig | None = None) -> torch.Tensor:
    cfg = config or TinyPickPlaceConfig()
    xyz = torch.as_tensor(xyz, dtype=torch.float32).flatten()
    return torch.tensor(
        [
            (xyz[0] - cfg.x_range[0]) / (cfg.x_range[1] - cfg.x_range[0]),
            (xyz[1] - cfg.y_range[0]) / (cfg.y_range[1] - cfg.y_range[0]),
        ],
        dtype=torch.float32,
    ).clamp(0.0, 1.0)


def _label_from_item(item: Mapping[str, object], label_key: str = "label") -> torch.Tensor:
    return torch.as_tensor(item[label_key], dtype=torch.long).flatten()[0]


def _xy_from_item(
    item: Mapping[str, object],
    xy_key: str = OBJECT_XYZ_KEY,
    config: TinyPickPlaceConfig | None = None,
) -> torch.Tensor:
    xy_or_xyz = torch.as_tensor(item[xy_key], dtype=torch.float32).flatten()
    if xy_or_xyz.numel() == 2:
        return xy_or_xyz.clamp(0.0, 1.0)
    return workspace_xy_to_unit(xy_or_xyz, config)


def vision_sample_from_lerobot_row(
    row: Mapping[str, object],
    image_key: str = FRONT_IMAGE_KEY,
    label_key: str = "label",
    xy_key: str = OBJECT_XYZ_KEY,
    config: TinyPickPlaceConfig | None = None,
    condition: str = "lerobot",
) -> dict[str, torch.Tensor | str]:
    return {
        "image": _as_chw_float(row[image_key]),
        "label": _label_from_item(row, label_key),
        "xy": _xy_from_item(row, xy_key, config or TinyPickPlaceConfig()),
        "condition": condition,
    }


def object_color_mask(image: torch.Tensor, label: int | torch.Tensor, tolerance: float = 0.32) -> torch.Tensor:
    label = int(torch.as_tensor(label).flatten()[0].item())
    color = COLORS[label].to(image.device, image.dtype)[:, None, None]
    return (image - color).pow(2).sum(dim=0).sqrt() < tolerance


def replace_background(image: torch.Tensor, label: int | torch.Tensor, background: torch.Tensor) -> torch.Tensor:
    mask = object_color_mask(image, label)[None]
    return torch.where(mask, image, background.to(image.device, image.dtype).expand_as(image))


def _apply_visual_condition(
    image: torch.Tensor,
    label: torch.Tensor,
    lighting_range: tuple[float, float] = (1.0, 1.0),
    background_color: tuple[float, float, float] | None = None,
    background_noise: float = 0.0,
    image_noise: float = 0.0,
) -> torch.Tensor:
    object_mask = object_color_mask(image, label)[None]
    lo, hi = lighting_range
    lighting = torch.empty(()).uniform_(lo, hi).item()
    image = (image * lighting).clamp(0.0, 1.0)
    if background_color is not None:
        background = torch.tensor(background_color, dtype=image.dtype)[:, None, None]
        image = torch.where(object_mask, image, background.to(image.device).expand_as(image))
    if background_noise > 0:
        noise = torch.rand_like(image) * background_noise
        image = torch.where(object_mask, image, (image + noise).clamp(0.0, 1.0))
    if image_noise > 0:
        image = image + torch.randn_like(image) * image_noise
    return image.clamp(0.0, 1.0)


def collate_vision_batch(
    rows: list[Mapping[str, object]],
    image_key: str = FRONT_IMAGE_KEY,
    label_key: str = "label",
    xy_key: str = OBJECT_XYZ_KEY,
    config: TinyPickPlaceConfig | None = None,
    sample_transform: SampleTransform | None = None,
    condition: str = "lerobot",
    lighting_range: tuple[float, float] = (1.0, 1.0),
    background_color: tuple[float, float, float] | None = None,
    background_noise: float = 0.0,
    image_noise: float = 0.0,
) -> dict[str, torch.Tensor | list[str]]:
    samples = []
    for row in rows:
        sample = vision_sample_from_lerobot_row(row, image_key, label_key, xy_key, config, condition)
        sample["image"] = _apply_visual_condition(
            torch.as_tensor(sample["image"], dtype=torch.float32),
            torch.as_tensor(sample["label"]),
            lighting_range=lighting_range,
            background_color=background_color,
            background_noise=background_noise,
            image_noise=image_noise,
        )
        if sample_transform is not None:
            sample = sample_transform(sample)
        samples.append(sample)
    return {
        "image": torch.stack([torch.as_tensor(sample["image"], dtype=torch.float32) for sample in samples]),
        "label": torch.stack([torch.as_tensor(sample["label"], dtype=torch.long) for sample in samples]),
        "xy": torch.stack([torch.as_tensor(sample["xy"], dtype=torch.float32) for sample in samples]),
        "condition": [str(sample["condition"]) for sample in samples],
    }


def make_vision_collate(**kwargs) -> Callable[[list[Mapping[str, object]]], dict[str, torch.Tensor | list[str]]]:
    return partial(collate_vision_batch, **kwargs)


def make_robustness_vision_collates() -> dict[str, Callable[[list[Mapping[str, object]]], dict[str, torch.Tensor | list[str]]]]:
    return {
        "train_distribution": make_vision_collate(condition="train"),
        "lighting": make_vision_collate(condition="lighting", lighting_range=(0.45, 1.55)),
        "background": make_vision_collate(condition="background", background_color=(0.05, 0.35, 0.32)),
        "texture_noise": make_vision_collate(condition="texture_noise", background_noise=0.45),
    }


class VisionAugmentation:
    def __init__(
        self,
        image_size: int = 96,
        color_jitter: bool = True,
        random_crop: bool = True,
        background_replacement: bool = True,
        p_background: float = 0.5,
    ) -> None:
        self.image_size = image_size
        self.random_crop = random_crop
        self.background_replacement = background_replacement
        self.p_background = p_background
        self.jitter = ColorJitter(brightness=0.25, contrast=0.25, saturation=0.20, hue=0.03) if color_jitter else None

    def __call__(self, sample: dict[str, torch.Tensor | str]) -> dict[str, torch.Tensor | str]:
        image = torch.as_tensor(sample["image"], dtype=torch.float32)
        if self.background_replacement and torch.rand(()) < self.p_background:
            bg = torch.rand(3, 1, 1, dtype=image.dtype) * 0.75
            image = replace_background(image, torch.as_tensor(sample["label"]), bg)
        if self.random_crop:
            height, width = image.shape[-2:]
            crop = int(min(height, width) * float(torch.empty(()).uniform_(0.85, 1.0)))
            top = int(torch.randint(0, height - crop + 1, ()).item())
            left = int(torch.randint(0, width - crop + 1, ()).item())
            image = TF.resized_crop(image, top, left, crop, crop, [self.image_size, self.image_size])
        if self.jitter is not None:
            image = self.jitter(image)
        sample = dict(sample)
        sample["image"] = image.clamp(0.0, 1.0)
        return sample


def _batch_to_device(batch: Mapping[str, torch.Tensor | str], device: str | torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}


@torch.no_grad()
def extract_features(
    model: torch.nn.Module,
    loader,
    device: str | torch.device = "cuda",
    feature: str = "pooled",
    max_batches: int | None = None,
) -> dict[str, torch.Tensor | list[str]]:
    model.to(device)
    model.eval()
    features: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    xy: list[torch.Tensor] = []
    conditions: list[str] = []
    for batch_index, raw_batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = _batch_to_device(raw_batch, device)
        pooled, patch_tokens = model.encode(batch["image"])
        if feature == "pooled":
            chosen = pooled
        elif feature == "patch_mean":
            chosen = patch_tokens.mean(dim=1)
        else:
            raise ValueError(f"unknown feature type: {feature}")
        features.append(chosen.cpu())
        labels.append(batch["label"].cpu())
        xy.append(batch["xy"].cpu())
        if "condition" in raw_batch:
            condition = raw_batch["condition"]
            conditions.extend(list(condition) if isinstance(condition, (list, tuple)) else [str(condition)])
    return {
        "feature": torch.cat(features),
        "label": torch.cat(labels),
        "xy": torch.cat(xy),
        "condition": conditions,
    }


def pca_2d(features: torch.Tensor) -> torch.Tensor:
    features = features.float().cpu()
    centered = features - features.mean(dim=0, keepdim=True)
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    return centered @ vh[:2].T


def tsne_2d(features: torch.Tensor, perplexity: float = 30.0, seed: int = 0) -> torch.Tensor:
    from sklearn.manifold import TSNE

    embedding = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    ).fit_transform(features.float().cpu().numpy())
    return torch.as_tensor(embedding, dtype=torch.float32)


def mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.mean((pred.detach() - target.detach()) ** 2).item())


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return float((logits.argmax(dim=-1) == labels).float().mean().item())


def sample_diversity(samples: torch.Tensor) -> float:
    if samples.shape[0] <= 1:
        return 0.0
    return float(samples.detach().std(dim=0).norm(dim=-1).mean().item())


def trajectory_smoothness(action_chunk: torch.Tensor) -> float:
    if action_chunk.shape[-2] <= 1:
        return 0.0
    diff = action_chunk.detach().diff(dim=-2)
    return float((diff**2).sum(dim=-1).mean().item())


def expert_drift(rollout: torch.Tensor, expert: torch.Tensor) -> float:
    common = min(len(rollout), len(expert))
    if common == 0:
        return float("nan")
    return float((rollout[:common] - expert[:common]).norm(dim=-1).mean().item())


PolicyFn = Callable[[dict[str, torch.Tensor | str]], torch.Tensor]


class RolloutEnv(Protocol):
    config: TinyPickPlaceConfig

    def reset_from_episode(self, episode: Mapping[str, torch.Tensor | str | int | float]) -> dict[str, torch.Tensor | str]:
        ...

    def step(self, action: torch.Tensor) -> tuple[dict[str, torch.Tensor | str], float, bool, dict[str, float]]:
        ...


@torch.no_grad()
def rollout_policy(
    policy: PolicyFn,
    env: RolloutEnv,
    initial_states: Sequence[Mapping[str, torch.Tensor | str | int | float]],
    n_episodes: int | None = None,
    expert_trajectories: Sequence[torch.Tensor] | None = None,
) -> dict[str, object]:
    n = min(n_episodes or len(initial_states), len(initial_states))
    successes: list[float] = []
    final_distances: list[float] = []
    drifts: list[float] = []
    trajectories: list[torch.Tensor] = []

    for episode in range(n):
        obs = env.reset_from_episode(initial_states[episode])
        trajectory = [_position_from_obs(obs)]
        done = False
        info = {"success": 0.0, "distance": float("nan")}
        for _ in range(env.config.horizon):
            action = policy(obs).detach().cpu().flatten()
            obs, _, done, info = env.step(action)
            trajectory.append(_position_from_obs(obs))
            if done:
                break
        rollout_traj = torch.stack(trajectory)
        trajectories.append(rollout_traj)
        successes.append(float(info["success"]))
        final_distances.append(float(info["distance"]))
        if expert_trajectories is not None:
            drifts.append(expert_drift(rollout_traj, expert_trajectories[episode]))

    return {
        "success_rate": float(torch.tensor(successes).float().mean().item()) if successes else 0.0,
        "final_distance": float(torch.tensor(final_distances).float().mean().item()) if final_distances else float("nan"),
        "mean_final_distance": float(torch.tensor(final_distances).float().mean().item()) if final_distances else float("nan"),
        "expert_drift": float(torch.tensor(drifts).float().mean().item()) if drifts else float("nan"),
        "mean_expert_drift": float(torch.tensor(drifts).float().mean().item()) if drifts else float("nan"),
        "trajectories": trajectories,
    }


def _position_from_obs(obs: Mapping[str, torch.Tensor | str]) -> torch.Tensor:
    if OBJECT_XYZ_KEY in obs:
        return torch.as_tensor(obs[OBJECT_XYZ_KEY], dtype=torch.float32).flatten()
    state = torch.as_tensor(obs["state"], dtype=torch.float32).flatten()
    return state[FRANKA_QPOS_ACTION_DIM : FRANKA_QPOS_ACTION_DIM + 3]

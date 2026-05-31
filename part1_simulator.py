"""Part 1: Genesis Franka pick-and-place data collection.

This module is the phase-1 implementation described in PLAN.md:

- build a headless Genesis scene with a Franka arm, colored cubes, a table,
  a target marker, and an RGB camera;
- generate an IK expert trajectory through the fixed stages
  pre-grasp -> grasp -> lift -> move-to-place -> place -> retreat;
- save RGB, robot qpos, object/target poses, gripper width, action qpos
  targets, and task text as a LeRobot dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Protocol, Sequence

import numpy as np
import torch

from env import (
    ACTION_KEY as _ACTION_KEY,
    COLOR_NAMES as _COLOR_NAMES,
    FRANKA_QPOS_ACTION_DIM as _FRANKA_QPOS_ACTION_DIM,
    GRIPPER_WIDTH_KEY as _GRIPPER_WIDTH_KEY,
    IMAGE_KEY as _IMAGE_KEY,
    OBJECT_POSE_KEY as _OBJECT_POSE_KEY,
    OBJECT_XYZ_KEY as _OBJECT_XYZ_KEY,
    ROBOT_QPOS_KEY as _ROBOT_QPOS_KEY,
    STATE_KEY as _STATE_KEY,
    TARGET_POSE_KEY as _TARGET_POSE_KEY,
    TARGET_XYZ_KEY as _TARGET_XYZ_KEY,
    TASK_KEY as _TASK_KEY,
    GenesisFrankaPickPlaceEnv as _GenesisFrankaPickPlaceEnv,
    TinyPickPlaceConfig as _TinyPickPlaceConfig,
)


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


def _chw_float_to_hwc_uint8(image: torch.Tensor) -> np.ndarray:
    return (image.detach().cpu().permute(1, 2, 0).clamp(0.0, 1.0).numpy() * 255).astype(np.uint8)


def lerobot_features(
    image_size: int,
    state_dim: int,
    action_dim: int,
    image_key: str = _IMAGE_KEY,
) -> dict[str, dict[str, object]]:
    return {
        image_key: {"dtype": "video", "shape": (image_size, image_size, 3), "names": ["height", "width", "channels"]},
        _ROBOT_QPOS_KEY: {"dtype": "float32", "shape": (action_dim,), "names": [f"q_{i}" for i in range(action_dim)]},
        _STATE_KEY: {"dtype": "float32", "shape": (state_dim,), "names": [f"state_{i}" for i in range(state_dim)]},
        _ACTION_KEY: {"dtype": "float32", "shape": (action_dim,), "names": [f"target_q_{i}" for i in range(action_dim)]},
        _OBJECT_POSE_KEY: {"dtype": "float32", "shape": (7,), "names": ["x", "y", "z", "qw", "qx", "qy", "qz"]},
        _TARGET_POSE_KEY: {"dtype": "float32", "shape": (7,), "names": ["x", "y", "z", "qw", "qx", "qy", "qz"]},
        _OBJECT_XYZ_KEY: {"dtype": "float32", "shape": (3,), "names": ["object_x", "object_y", "object_z"]},
        _TARGET_XYZ_KEY: {"dtype": "float32", "shape": (3,), "names": ["target_x", "target_y", "target_z"]},
        _GRIPPER_WIDTH_KEY: {"dtype": "float32", "shape": (1,), "names": ["width"]},
        "label": {"dtype": "int64", "shape": (1,), "names": ["label"]},
    }


PolicyFn = Callable[[dict[str, torch.Tensor | str]], torch.Tensor]


class RolloutEnv(Protocol):
    config: _TinyPickPlaceConfig

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
    if _OBJECT_XYZ_KEY in obs:
        return torch.as_tensor(obs[_OBJECT_XYZ_KEY], dtype=torch.float32).flatten()
    state = torch.as_tensor(obs["state"], dtype=torch.float32).flatten()
    return state[_FRANKA_QPOS_ACTION_DIM : _FRANKA_QPOS_ACTION_DIM + 3]


def random_episode_specs(config: _TinyPickPlaceConfig, n_episodes: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(config.seed)
    starts = torch.rand(n_episodes, 2, generator=generator) * 0.8 + 0.1
    targets = torch.rand(n_episodes, 2, generator=generator) * 0.8 + 0.1
    labels = torch.randint(len(_COLOR_NAMES), (n_episodes,), generator=generator)
    return starts, targets, labels


def frame_for_lerobot(obs: Mapping[str, torch.Tensor | str], action: torch.Tensor) -> dict[str, object]:
    return {
        _IMAGE_KEY: _chw_float_to_hwc_uint8(torch.as_tensor(obs[_IMAGE_KEY])),
        _STATE_KEY: torch.as_tensor(obs[_STATE_KEY], dtype=torch.float32).detach().cpu().numpy(),
        _ROBOT_QPOS_KEY: torch.as_tensor(obs[_ROBOT_QPOS_KEY], dtype=torch.float32).detach().cpu().numpy(),
        _ACTION_KEY: action.detach().cpu().float().numpy(),
        _OBJECT_POSE_KEY: torch.as_tensor(obs[_OBJECT_POSE_KEY], dtype=torch.float32).detach().cpu().numpy(),
        _TARGET_POSE_KEY: torch.as_tensor(obs[_TARGET_POSE_KEY], dtype=torch.float32).detach().cpu().numpy(),
        _OBJECT_XYZ_KEY: torch.as_tensor(obs[_OBJECT_XYZ_KEY], dtype=torch.float32).detach().cpu().numpy(),
        _TARGET_XYZ_KEY: torch.as_tensor(obs[_TARGET_XYZ_KEY], dtype=torch.float32).detach().cpu().numpy(),
        _GRIPPER_WIDTH_KEY: torch.as_tensor(obs[_GRIPPER_WIDTH_KEY], dtype=torch.float32).detach().cpu().numpy(),
        "label": torch.as_tensor(obs["label"], dtype=torch.long).detach().cpu().numpy(),
        _TASK_KEY: str(obs[_TASK_KEY]),
    }


def collect_expert_episode(
    env: _GenesisFrankaPickPlaceEnv,
    object_xy: torch.Tensor,
    target_xy: torch.Tensor,
    label: torch.Tensor | int,
    steps_per_segment: int = 12,
) -> tuple[list[dict[str, object]], torch.Tensor]:
    obs = env.reset(object_xy, target_xy, label)
    q_traj = env.expert_qpos_trajectory(steps_per_segment=steps_per_segment)
    rows: list[dict[str, object]] = []
    for qpos in q_traj:
        rows.append(frame_for_lerobot(obs, qpos))
        obs = env.step_qpos(qpos)
    return rows, q_traj


def collect_genesis_franka_lerobot_dataset(
    root: str | Path,
    repo_id: str = "local/simple-vla-genesis-franka-pick-place",
    config: _TinyPickPlaceConfig | None = None,
    n_episodes: int | None = None,
    steps_per_segment: int = 12,
    image_size: int | None = None,
    show_viewer: bool = False,
    backend: str = "gpu",
    vcodec: str = "h264",
):
    """Collect IK expert demonstrations and write a LeRobot dataset."""

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    cfg = config or _TinyPickPlaceConfig()
    n = n_episodes or cfg.n_episodes
    cfg.horizon = steps_per_segment * 6
    cfg.image_size = image_size or cfg.image_size
    env = _GenesisFrankaPickPlaceEnv(cfg, image_size=cfg.image_size, show_viewer=show_viewer, backend=backend)

    starts, targets, labels = random_episode_specs(cfg, n)
    sample_obs = env.reset(starts[0], targets[0], labels[0])
    state_dim = int(torch.as_tensor(sample_obs[_STATE_KEY]).numel())
    action_dim = int(torch.as_tensor(sample_obs[_ROBOT_QPOS_KEY]).numel())
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=cfg.fps,
        root=root,
        robot_type="genesis_franka",
        use_videos=True,
        features=lerobot_features(cfg.image_size, state_dim, action_dim),
        vcodec=vcodec,
    )

    for episode in range(n):
        rows, _ = collect_expert_episode(env, starts[episode], targets[episode], labels[episode], steps_per_segment)
        for row in rows:
            dataset.add_frame(row)
        dataset.save_episode()

    dataset.finalize()
    return dataset


def genesis_integration_notes() -> str:
    return (
        "Phase 1 path: env.GenesisFrankaPickPlaceEnv -> collect_expert_episode -> "
        "collect_genesis_franka_lerobot_dataset. Load data with LeRobotDataset."
    )


def main() -> None:
    collect_genesis_franka_lerobot_dataset(
        root="datasets/simple-vla-genesis-franka-pick-place",
        n_episodes=2,
        steps_per_segment=8,
        image_size=96,
    )


if __name__ == "__main__":
    main()

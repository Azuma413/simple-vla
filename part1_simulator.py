"""Part 1: Genesis Franka pick-and-place data collection.

This module is the phase-1 implementation described in PLAN.md:

- build a headless Genesis scene with a Franka arm, colored cubes, a table,
  a target marker, a front RGB camera, and a wrist RGB camera;
- generate IK expert actions through the fixed stages
  pre-grasp -> grasp -> lift -> move-to-place -> place -> retreat;
- save RGB, robot qpos, object/target poses, gripper width, action qpos
  targets, and task text as a LeRobot dataset.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import shutil
from typing import Mapping

import numpy as np
import torch

from env import (
    ACTION_KEY as _ACTION_KEY,
    COLOR_NAMES as _COLOR_NAMES,
    FRONT_IMAGE_KEY as _FRONT_IMAGE_KEY,
    GRIPPER_WIDTH_KEY as _GRIPPER_WIDTH_KEY,
    IMAGE_KEYS as _IMAGE_KEYS,
    OBJECT_POSE_KEY as _OBJECT_POSE_KEY,
    OBJECT_XYZ_KEY as _OBJECT_XYZ_KEY,
    ROBOT_QPOS_KEY as _ROBOT_QPOS_KEY,
    STATE_KEY as _STATE_KEY,
    TARGET_POSE_KEY as _TARGET_POSE_KEY,
    TARGET_XYZ_KEY as _TARGET_XYZ_KEY,
    TASK_KEY as _TASK_KEY,
    WRIST_IMAGE_KEY as _WRIST_IMAGE_KEY,
    GenesisFrankaPickPlaceEnv as _GenesisFrankaPickPlaceEnv,
    TinyPickPlaceConfig as _TinyPickPlaceConfig,
)

DEFAULT_STAGE_STEPS = {
    "pre-grasp": 40,
    "grasp": 25,
    "lift": 45,
    "move-to-place": 55,
    "place": 25,
    "retreat": 30,
}

VISUAL_DATASET_VARIANTS = {
    "train": {},
    "lighting_dim": {"ambient_light": (0.03, 0.03, 0.03), "directional_light_intensity": 2.0},
    "lighting_bright": {"ambient_light": (0.22, 0.22, 0.22), "directional_light_intensity": 8.0},
    "texture": {"table_texture_path": "checker"},
}


def _chw_float_to_hwc_uint8(image: torch.Tensor) -> np.ndarray:
    return (image.detach().cpu().permute(1, 2, 0).clamp(0.0, 1.0).numpy() * 255).astype(np.uint8)


def lerobot_features(
    image_size: int,
    state_dim: int,
    action_dim: int,
    image_keys: tuple[str, ...] = _IMAGE_KEYS,
) -> dict[str, dict[str, object]]:
    image_features = {
        key: {"dtype": "video", "shape": (image_size, image_size, 3), "names": ["height", "width", "channels"]}
        for key in image_keys
    }
    return {
        **image_features,
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


def random_episode_specs(config: _TinyPickPlaceConfig, n_episodes: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(config.seed)
    starts = torch.rand(n_episodes, 2, generator=generator) * 0.8 + 0.1
    targets = torch.rand(n_episodes, 2, generator=generator) * 0.8 + 0.1
    labels = torch.randint(len(_COLOR_NAMES), (n_episodes,), generator=generator)
    return starts, targets, labels


def frame_for_lerobot(obs: Mapping[str, torch.Tensor | str], action: torch.Tensor) -> dict[str, object]:
    return {
        _FRONT_IMAGE_KEY: _chw_float_to_hwc_uint8(torch.as_tensor(obs[_FRONT_IMAGE_KEY])),
        _WRIST_IMAGE_KEY: _chw_float_to_hwc_uint8(torch.as_tensor(obs[_WRIST_IMAGE_KEY])),
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
    stage_steps: Mapping[str, int] | None = None,
) -> tuple[list[dict[str, object]], torch.Tensor]:
    obs = env.reset(object_xy, target_xy, label)
    actions: list[torch.Tensor] = []
    rows: list[dict[str, object]] = []
    for stage, steps in (stage_steps or DEFAULT_STAGE_STEPS).items():
        for _ in range(steps):
            action = env.expert_stage_action(stage)
            rows.append(frame_for_lerobot(obs, action))
            actions.append(action)
            obs = env.step_qpos(action)
    return rows, torch.stack(actions)


def collect_genesis_franka_lerobot_dataset(
    root: str | Path,
    repo_id: str = "local/simple-vla-genesis-franka-pick-place",
    config: _TinyPickPlaceConfig | None = None,
    n_episodes: int | None = None,
    stage_steps: Mapping[str, int] | None = None,
    image_size: int | None = None,
    show_viewer: bool = False,
    backend: str = "gpu",
    vcodec: str = "h264",
    overwrite: bool = False,
):
    """Collect IK expert demonstrations and write a LeRobot dataset."""

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = Path(root)
    if overwrite and root.exists():
        shutil.rmtree(root)

    stage_steps = dict(stage_steps or DEFAULT_STAGE_STEPS)
    base_cfg = config or _TinyPickPlaceConfig()
    cfg = replace(base_cfg, horizon=sum(stage_steps.values()), image_size=image_size or base_cfg.image_size)
    n = n_episodes or cfg.n_episodes
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
        rows, _ = collect_expert_episode(env, starts[episode], targets[episode], labels[episode], stage_steps)
        for row in rows:
            dataset.add_frame(row)
        dataset.save_episode()

    dataset.finalize()
    return dataset


def collect_phase1_dataset_suite(
    root: str | Path = "datasets/simple-vla-genesis-franka-pick-place",
    repo_id_prefix: str = "local/simple-vla-genesis-franka-pick-place",
    config: _TinyPickPlaceConfig | None = None,
    n_episodes: int | None = None,
    stage_steps: Mapping[str, int] | None = None,
    image_size: int = 225,
    variants: Mapping[str, Mapping[str, float]] | None = None,
    show_viewer: bool = False,
    backend: str = "gpu",
    overwrite: bool = True,
) -> dict[str, object]:
    """Generate train and robustness datasets used by later phases."""

    root = Path(root)
    base_cfg = config or _TinyPickPlaceConfig()
    datasets = {}
    for name, visual_kwargs in (variants or VISUAL_DATASET_VARIANTS).items():
        variant_cfg = replace(base_cfg, image_size=image_size, **visual_kwargs)
        datasets[name] = collect_genesis_franka_lerobot_dataset(
            root=root / name,
            repo_id=f"{repo_id_prefix}-{name}",
            config=variant_cfg,
            n_episodes=n_episodes or variant_cfg.n_episodes,
            stage_steps=stage_steps,
            image_size=image_size,
            show_viewer=show_viewer,
            backend=backend,
            overwrite=overwrite,
        )
    return datasets


def genesis_integration_notes() -> str:
    return (
        "Phase 1 path: env.GenesisFrankaPickPlaceEnv -> collect_expert_episode -> "
        "collect_genesis_franka_lerobot_dataset. Load data with LeRobotDataset."
    )


def main() -> None:
    collect_phase1_dataset_suite(
        root="datasets/simple-vla-genesis-franka-pick-place",
        n_episodes=2,
        image_size=225,
    )


if __name__ == "__main__":
    main()

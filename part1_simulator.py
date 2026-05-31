"""Part 1: Genesis Franka pick-and-place data collection.

This module is the phase-1 implementation described in PLAN.md:

- build a headless Genesis scene with a Franka arm, colored cubes, a table,
  a target marker, and an RGB camera;
- generate an IK expert trajectory through the fixed stages
  pre-grasp -> grasp -> lift -> move-to-place -> place -> retreat;
- save RGB, robot qpos, object/target poses, gripper width, action qpos
  targets, and task text as a LeRobot dataset;
- provide a small stable Dataset/DataLoader adapter used by later chapters.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Protocol, Sequence

import imageio.v3 as iio
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from env import (
    ACTION_CHUNK_KEY as _ACTION_CHUNK_KEY,
    ACTION_KEY as _ACTION_KEY,
    COLOR_NAMES as _COLOR_NAMES,
    FRANKA_QPOS_ACTION_DIM as _FRANKA_QPOS_ACTION_DIM,
    GENESIS_STATE_DIM as _GENESIS_STATE_DIM,
    GRIPPER_WIDTH_KEY as _GRIPPER_WIDTH_KEY,
    IMAGE_KEY as _IMAGE_KEY,
    INSTRUCTION_KEY as _INSTRUCTION_KEY,
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


def _as_tensor(value, dtype: torch.dtype | None = None) -> torch.Tensor:
    tensor = torch.as_tensor(value)
    return tensor.to(dtype=dtype) if dtype is not None else tensor


def _image_to_chw_float(image) -> torch.Tensor:
    if hasattr(image, "__array__") and not isinstance(image, torch.Tensor):
        image = np.asarray(image)
    tensor = torch.as_tensor(image)
    if tensor.ndim == 3 and tensor.shape[-1] in (1, 3, 4):
        tensor = tensor[..., :3].permute(2, 0, 1)
    tensor = tensor.float()
    if tensor.max() > 1.0:
        tensor = tensor / 255.0
    return tensor.clamp(0.0, 1.0)


def _chw_float_to_hwc_uint8(image: torch.Tensor) -> np.ndarray:
    return (image.detach().cpu().permute(1, 2, 0).clamp(0.0, 1.0).numpy() * 255).astype(np.uint8)


def _task_from_row(row: Mapping) -> str:
    return str(row.get(_TASK_KEY, row.get(_INSTRUCTION_KEY, "pick the cube")))


def _canonical_sample(row: Mapping, chunk: torch.Tensor | None = None) -> dict[str, torch.Tensor | str]:
    image = _image_to_chw_float(row[_IMAGE_KEY if _IMAGE_KEY in row else "image"])
    state = _as_tensor(row[_STATE_KEY if _STATE_KEY in row else "state"], torch.float32).flatten()
    action = _as_tensor(row[_ACTION_KEY], torch.float32).flatten()
    task = _task_from_row(row)
    out: dict[str, torch.Tensor | str] = {
        _IMAGE_KEY: image,
        _STATE_KEY: state,
        _ACTION_KEY: action,
        _ACTION_CHUNK_KEY: chunk if chunk is not None else action[None],
        _TASK_KEY: task,
        _INSTRUCTION_KEY: task,
        "image": image,
        "state": state,
        "episode_index": _as_tensor(row.get("episode_index", 0), torch.long),
        "frame_index": _as_tensor(row.get("frame_index", 0), torch.long),
    }
    for key in (
        _ROBOT_QPOS_KEY,
        _OBJECT_POSE_KEY,
        _TARGET_POSE_KEY,
        _OBJECT_XYZ_KEY,
        _TARGET_XYZ_KEY,
        _GRIPPER_WIDTH_KEY,
        "timestamp",
        "next.done",
        "label",
    ):
        if key in row:
            dtype = torch.float32
            if key in ("next.done", "label"):
                dtype = None
            out[key] = _as_tensor(row[key], dtype).flatten()
    return out


class TinyPickPlaceDataset(Dataset):
    """In-memory frames using the same keys as the collected LeRobot data."""

    def __init__(
        self,
        frames: Sequence[Mapping],
        chunk_size: int = 4,
        initial_states: Sequence[Mapping] | None = None,
    ) -> None:
        self.frames = list(frames)
        self.chunk_size = chunk_size
        self._episode_to_indices = _build_episode_index(self.frames)
        self._initial_states = list(initial_states) if initial_states is not None else None
        sample = self[0]
        self.action_dim = int(torch.as_tensor(sample[_ACTION_KEY]).numel())
        self.state_dim = int(torch.as_tensor(sample["state"]).numel())
        self.config = _TinyPickPlaceConfig(
            n_episodes=len(self._episode_to_indices),
            horizon=max(len(v) for v in self._episode_to_indices.values()),
            action_dim=self.action_dim,
            state_dim=self.state_dim,
        )

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        row = self.frames[index]
        episode = int(torch.as_tensor(row.get("episode_index", 0)).item())
        indices = self._episode_to_indices[episode]
        local = indices.index(index)
        chunk = []
        for offset in range(self.chunk_size):
            chunk_index = indices[min(local + offset, len(indices) - 1)]
            chunk.append(_as_tensor(self.frames[chunk_index][_ACTION_KEY], torch.float32).flatten())
        return _canonical_sample(row, torch.stack(chunk))

    def episode_initial_state(self, episode: int) -> dict[str, torch.Tensor | int]:
        if self._initial_states is not None:
            return {k: _as_tensor(v).clone() if not isinstance(v, str) else v for k, v in self._initial_states[episode].items()}
        first = self[self._episode_to_indices[episode][0]]
        return _initial_state_from_sample(first)


class LeRobotPickPlaceDataset(Dataset):
    """LeRobot adapter with stable keys for parts 2-6."""

    def __init__(
        self,
        root: str | Path,
        repo_id: str | None = None,
        chunk_size: int = 4,
        image_key: str = _IMAGE_KEY,
        video_backend: str = "pyav",
    ) -> None:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self.root = Path(root)
        self.repo_id = repo_id or _infer_lerobot_repo_id(self.root)
        self.chunk_size = chunk_size
        self.image_key = image_key
        self.dataset = LeRobotDataset(self.repo_id, root=self.root, video_backend=video_backend)
        self._episode_to_indices = _build_episode_index(self.dataset)
        sample = self[0]
        self.action_dim = int(torch.as_tensor(sample[_ACTION_KEY]).numel())
        self.state_dim = int(torch.as_tensor(sample["state"]).numel())
        self.config = _TinyPickPlaceConfig(
            n_episodes=len(self._episode_to_indices),
            horizon=max(len(v) for v in self._episode_to_indices.values()),
            action_dim=self.action_dim,
            state_dim=self.state_dim,
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        row = dict(self.dataset[index])
        if self.image_key != _IMAGE_KEY:
            row[_IMAGE_KEY] = row[self.image_key]
        episode = int(torch.as_tensor(row.get("episode_index", 0)).item())
        indices = self._episode_to_indices[episode]
        local = indices.index(index)
        chunk = []
        for offset in range(self.chunk_size):
            chunk_index = indices[min(local + offset, len(indices) - 1)]
            chunk.append(_as_tensor(_lerobot_feature_at(self.dataset, chunk_index, _ACTION_KEY), torch.float32).flatten())
        return _canonical_sample(row, torch.stack(chunk))

    def episode_initial_state(self, episode: int) -> dict[str, torch.Tensor | int]:
        first = self[self._episode_to_indices[episode][0]]
        return _initial_state_from_sample(first)


def _infer_lerobot_repo_id(root: Path) -> str:
    info_path = root / "meta" / "info.json"
    if info_path.exists():
        info = json.loads(info_path.read_text())
        value = info.get("repo_id")
        if isinstance(value, str):
            return value
    return "local/simple-vla-genesis-franka-pick-place"


def _build_episode_index(dataset) -> dict[int, list[int]]:
    meta = getattr(dataset, "meta", None)
    episode_rows = getattr(meta, "episodes", None)
    if episode_rows is not None:
        episodes: dict[int, list[int]] = {}
        for row in episode_rows:
            if "dataset_from_index" in row and "dataset_to_index" in row:
                episode = int(row["episode_index"])
                start = int(row["dataset_from_index"])
                end = int(row["dataset_to_index"])
                episodes[episode] = list(range(start, end))
        if episodes:
            return episodes

    episodes: dict[int, list[int]] = {}
    for idx in range(len(dataset)):
        row = dataset[idx]
        episode = int(torch.as_tensor(row.get("episode_index", 0)).item())
        episodes.setdefault(episode, []).append(idx)
    return episodes


def _lerobot_feature_at(dataset, index: int, key: str):
    reader = dataset._ensure_reader()
    if reader.hf_dataset is None:
        reader.load_and_activate()
    return reader.hf_dataset[index][key]


def _initial_state_from_sample(sample: Mapping[str, torch.Tensor | str]) -> dict[str, torch.Tensor | int]:
    if _OBJECT_XYZ_KEY in sample and _TARGET_XYZ_KEY in sample:
        label_value = sample.get("label", torch.tensor([0]))
        return {
            _OBJECT_XYZ_KEY: torch.as_tensor(sample[_OBJECT_XYZ_KEY], dtype=torch.float32).flatten(),
            _TARGET_XYZ_KEY: torch.as_tensor(sample[_TARGET_XYZ_KEY], dtype=torch.float32).flatten(),
            "label": int(torch.as_tensor(label_value).flatten()[0].item()),
        }
    state = torch.as_tensor(sample["state"], dtype=torch.float32).flatten()
    return {
        _OBJECT_XYZ_KEY: state[_FRANKA_QPOS_ACTION_DIM : _FRANKA_QPOS_ACTION_DIM + 3],
        _TARGET_XYZ_KEY: state[_FRANKA_QPOS_ACTION_DIM + 3 : _FRANKA_QPOS_ACTION_DIM + 6],
        "label": 0,
    }


def pick_place_collate(batch: list[dict[str, torch.Tensor | str]]) -> dict[str, torch.Tensor | list[str]]:
    output: dict[str, torch.Tensor | list[str]] = {}
    for key in batch[0]:
        values = [item[key] for item in batch]
        if isinstance(values[0], torch.Tensor):
            output[key] = torch.stack(values)  # type: ignore[arg-type]
        else:
            output[key] = values  # type: ignore[assignment]
    return output


collate_tiny_pick_place = pick_place_collate


def make_pick_place_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=pick_place_collate)


def lerobot_features(
    image_size: int,
    state_dim: int,
    action_dim: int,
    image_key: str = _IMAGE_KEY,
    image_dtype: str = "video",
) -> dict[str, dict[str, object]]:
    return {
        image_key: {"dtype": image_dtype, "shape": (image_size, image_size, 3), "names": ["height", "width", "channels"]},
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


def save_rollout_video(frames: Sequence[np.ndarray | torch.Tensor], path: str | Path, fps: int = 10) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    images = []
    for frame in frames:
        tensor = _image_to_chw_float(frame)
        images.append(_chw_float_to_hwc_uint8(tensor))
    iio.imwrite(path, images, fps=fps)
    return path


def collect_expert_episode(
    env: _GenesisFrankaPickPlaceEnv,
    object_xy: torch.Tensor,
    target_xy: torch.Tensor,
    label: torch.Tensor | int,
    steps_per_segment: int = 12,
) -> tuple[list[dict[str, object]], list[np.ndarray], torch.Tensor]:
    obs = env.reset(object_xy, target_xy, label)
    q_traj = env.expert_qpos_trajectory(steps_per_segment=steps_per_segment)
    rows: list[dict[str, object]] = []
    frames: list[np.ndarray] = []
    for qpos in q_traj:
        rows.append(frame_for_lerobot(obs, qpos))
        frames.append(_chw_float_to_hwc_uint8(torch.as_tensor(obs[_IMAGE_KEY])))
        obs = env.step_qpos(qpos)
    return rows, frames, q_traj


def collect_genesis_franka_lerobot_dataset(
    root: str | Path,
    repo_id: str = "local/simple-vla-genesis-franka-pick-place",
    config: _TinyPickPlaceConfig | None = None,
    n_episodes: int | None = None,
    steps_per_segment: int = 12,
    image_size: int | None = None,
    show_viewer: bool = False,
    backend: str = "gpu",
    video_dir: str | Path | None = None,
    use_videos: bool = True,
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
        use_videos=use_videos,
        features=lerobot_features(cfg.image_size, state_dim, action_dim, image_dtype="video" if use_videos else "image"),
        vcodec=vcodec,
    )

    for episode in range(n):
        rows, frames, _ = collect_expert_episode(env, starts[episode], targets[episode], labels[episode], steps_per_segment)
        for row in rows:
            dataset.add_frame(row)
        dataset.save_episode()
        if video_dir is not None:
            save_rollout_video(frames, Path(video_dir) / f"episode_{episode:03d}.mp4", fps=cfg.fps)

    dataset.finalize()
    return dataset


def genesis_integration_notes() -> str:
    return (
        "Phase 1 path: env.GenesisFrankaPickPlaceEnv -> collect_expert_episode -> "
        "collect_genesis_franka_lerobot_dataset. Load data with "
        "LeRobotPickPlaceDataset and make_pick_place_dataloader."
    )


def main() -> None:
    collect_genesis_franka_lerobot_dataset(
        root="datasets/simple-vla-genesis-franka-pick-place",
        n_episodes=2,
        steps_per_segment=8,
        image_size=96,
        video_dir="rollouts/part1",
    )


if __name__ == "__main__":
    main()

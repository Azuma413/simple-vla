"""Part 1 / Phase 0 common infrastructure for SimpleVLA.

The project standard path is Genesis + Franka + LeRobot Dataset.  This module
therefore defines the shared sample schema, LeRobot dataset adapter, rollout
metrics, and a thin Genesis environment/collector entry point used by later
chapters.  It intentionally does not provide toy, mock, or CPU-only fallback
tasks.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Protocol, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


IMAGE_KEY = "observation.images.camera"
STATE_KEY = "observation.state"
ACTION_KEY = "action"
ACTION_CHUNK_KEY = "action_chunk"
INSTRUCTION_KEY = "instruction"
TASK_KEY = "task"

FRANKA_QPOS_ACTION_DIM = 9
GENESIS_STATE_DIM = FRANKA_QPOS_ACTION_DIM + 3 + 3

COLOR_NAMES = ("red", "blue", "green")
COLORS = torch.tensor(
    [
        [0.90, 0.10, 0.10],
        [0.10, 0.55, 0.95],
        [0.15, 0.75, 0.20],
    ],
    dtype=torch.float32,
)


@dataclass
class TinyPickPlaceConfig:
    """Small shared config object used by notebooks and rollout helpers.

    The name is kept for notebook compatibility, but the standard dataset path
    is LeRobot data collected from Genesis, not an in-code toy generator.
    """

    n_episodes: int = 1
    horizon: int = 64
    image_size: int = 96
    dt: float = 0.1
    action_dim: int = FRANKA_QPOS_ACTION_DIM
    state_dim: int = GENESIS_STATE_DIM
    success_threshold: float = 0.05
    seed: int = 0


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
    """Mean per-step standard deviation for [samples, chunk, action_dim]."""

    if samples.shape[0] <= 1:
        return 0.0
    return float(samples.detach().std(dim=0).norm(dim=-1).mean().item())


def trajectory_smoothness(action_chunk: torch.Tensor) -> float:
    """Mean squared finite difference of a trajectory or action chunk."""

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
    return str(row.get(TASK_KEY, row.get(INSTRUCTION_KEY, "pick the cube")))


def _canonical_sample(row: Mapping, chunk: torch.Tensor | None = None) -> dict[str, torch.Tensor | str]:
    image = _image_to_chw_float(row[IMAGE_KEY if IMAGE_KEY in row else "image"])
    state = _as_tensor(row[STATE_KEY if STATE_KEY in row else "state"], torch.float32).flatten()
    action = _as_tensor(row[ACTION_KEY], torch.float32).flatten()
    task = _task_from_row(row)
    out: dict[str, torch.Tensor | str] = {
        IMAGE_KEY: image,
        STATE_KEY: state,
        ACTION_KEY: action,
        ACTION_CHUNK_KEY: chunk if chunk is not None else action[None],
        TASK_KEY: task,
        INSTRUCTION_KEY: task,
        "image": image,
        "state": state,
        "episode_index": _as_tensor(row.get("episode_index", 0), torch.long),
        "frame_index": _as_tensor(row.get("frame_index", 0), torch.long),
    }
    for key in ("timestamp", "next.done", "object_xyz", "target_xyz", "label"):
        if key in row:
            dtype = torch.float32 if key in ("timestamp", "object_xyz", "target_xyz") else None
            out[key] = _as_tensor(row[key], dtype).flatten()
    return out


class TinyPickPlaceDataset(Dataset):
    """In-memory pick-and-place frames using the same schema as LeRobot data.

    This class is useful for small subsets loaded from notebooks or tests.  It
    does not synthesize demonstrations; every frame must be provided by a real
    data source and contain image, state, action, and episode/frame metadata.
    """

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
        self.action_dim = int(torch.as_tensor(sample[ACTION_KEY]).numel())
        self.state_dim = int(torch.as_tensor(sample["state"]).numel())
        self.config = TinyPickPlaceConfig(
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
            chunk.append(_as_tensor(self.frames[chunk_index][ACTION_KEY], torch.float32).flatten())
        return _canonical_sample(row, torch.stack(chunk))

    def episode_initial_state(self, episode: int) -> dict[str, torch.Tensor | int]:
        if self._initial_states is not None:
            return {k: _as_tensor(v).clone() if not isinstance(v, str) else v for k, v in self._initial_states[episode].items()}
        first = self[self._episode_to_indices[episode][0]]
        return _initial_state_from_sample(first)


class LeRobotPickPlaceDataset(Dataset):
    """LeRobot adapter with stable keys for parts 3-6.

    Each item contains both LeRobot-compatible names and short aliases:
    ``observation.images.camera``/``image``, ``observation.state``/``state``,
    ``action``, ``action_chunk``, ``task``, and ``instruction``.
    """

    def __init__(
        self,
        root: str | Path,
        repo_id: str | None = None,
        chunk_size: int = 4,
        image_key: str = IMAGE_KEY,
    ) -> None:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self.root = Path(root)
        self.repo_id = repo_id or _infer_lerobot_repo_id(self.root)
        self.chunk_size = chunk_size
        self.image_key = image_key
        self.dataset = LeRobotDataset(self.repo_id, root=self.root)
        self._episode_to_indices = _build_episode_index(self.dataset)
        sample = self[0]
        self.action_dim = int(torch.as_tensor(sample[ACTION_KEY]).numel())
        self.state_dim = int(torch.as_tensor(sample["state"]).numel())
        self.config = TinyPickPlaceConfig(
            n_episodes=len(self._episode_to_indices),
            horizon=max(len(v) for v in self._episode_to_indices.values()),
            action_dim=self.action_dim,
            state_dim=self.state_dim,
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        row = dict(self.dataset[index])
        if self.image_key != IMAGE_KEY:
            row[IMAGE_KEY] = row[self.image_key]
        episode = int(torch.as_tensor(row.get("episode_index", 0)).item())
        indices = self._episode_to_indices[episode]
        local = indices.index(index)
        chunk = []
        for offset in range(self.chunk_size):
            chunk_index = indices[min(local + offset, len(indices) - 1)]
            chunk.append(_as_tensor(self.dataset[chunk_index][ACTION_KEY], torch.float32).flatten())
        return _canonical_sample(row, torch.stack(chunk))

    def episode_initial_state(self, episode: int) -> dict[str, torch.Tensor | int]:
        first = self[self._episode_to_indices[episode][0]]
        return _initial_state_from_sample(first)


def _infer_lerobot_repo_id(root: Path) -> str:
    info_path = root / "meta" / "info.json"
    if info_path.exists():
        info = json.loads(info_path.read_text())
        for key in ("repo_id", "codebase_version"):
            value = info.get(key)
            if isinstance(value, str) and "/" in value:
                return value
    return "local/simple-vla-genesis-franka-pick-place"


def _build_episode_index(dataset) -> dict[int, list[int]]:
    episodes: dict[int, list[int]] = {}
    for idx in range(len(dataset)):
        row = dataset[idx]
        episode = int(torch.as_tensor(row.get("episode_index", 0)).item())
        episodes.setdefault(episode, []).append(idx)
    return episodes


def _initial_state_from_sample(sample: Mapping[str, torch.Tensor | str]) -> dict[str, torch.Tensor | int]:
    if "object_xyz" in sample and "target_xyz" in sample:
        label_value = sample.get("label", torch.tensor([0]))
        return {
            "object_xyz": torch.as_tensor(sample["object_xyz"], dtype=torch.float32).flatten(),
            "target_xyz": torch.as_tensor(sample["target_xyz"], dtype=torch.float32).flatten(),
            "label": int(torch.as_tensor(label_value).flatten()[0].item()),
        }
    state = torch.as_tensor(sample["state"], dtype=torch.float32).flatten()
    if state.numel() >= 15:
        return {"object_xyz": state[-6:-3], "target_xyz": state[-3:], "label": 0}
    raise KeyError("Dataset frames need object_xyz/target_xyz or a Genesis state suffix.")


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
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=pick_place_collate,
    )


def lerobot_features(
    image_size: int,
    state_dim: int,
    action_dim: int,
    image_key: str = IMAGE_KEY,
) -> dict[str, dict[str, object]]:
    return {
        image_key: {
            "dtype": "image",
            "shape": (image_size, image_size, 3),
            "names": ["height", "width", "channels"],
        },
        STATE_KEY: {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": [f"state_{i}" for i in range(state_dim)],
        },
        ACTION_KEY: {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": [f"action_{i}" for i in range(action_dim)],
        },
        "object_xyz": {
            "dtype": "float32",
            "shape": (3,),
            "names": ["object_x", "object_y", "object_z"],
        },
        "target_xyz": {
            "dtype": "float32",
            "shape": (3,),
            "names": ["target_x", "target_y", "target_z"],
        },
        "label": {
            "dtype": "int64",
            "shape": (1,),
            "names": ["label"],
        },
    }


PolicyFn = Callable[[dict[str, torch.Tensor | str]], torch.Tensor]


class RolloutEnv(Protocol):
    config: TinyPickPlaceConfig

    def reset_from_episode(self, episode: Mapping[str, torch.Tensor | str | int | float]) -> dict[str, torch.Tensor | str]:
        ...

    def step(self, action: torch.Tensor) -> tuple[dict[str, torch.Tensor | str], float, bool, dict[str, float]]:
        ...


def _initial_states_from_dataset(dataset, n_episodes: int | None = None) -> list[dict[str, torch.Tensor | int]]:
    total = getattr(getattr(dataset, "config", None), "n_episodes", None) or len(getattr(dataset, "_episode_to_indices"))
    n = min(n_episodes or total, total)
    return [dataset.episode_initial_state(i) for i in range(n)]


@torch.no_grad()
def rollout_policy(
    policy: PolicyFn,
    env: RolloutEnv,
    initial_states: Sequence[Mapping[str, torch.Tensor | str | int | float]] | None = None,
    n_episodes: int | None = None,
    expert_trajectories: Sequence[torch.Tensor] | None = None,
) -> dict[str, object]:
    """Roll out a policy in a real environment exposing reset/step.

    Returns the project-wide rollout metrics: success rate, final distance, and
    expert drift.  Passing a Dataset as ``env`` is intentionally unsupported;
    create ``GenesisFrankaPickPlaceEnv`` and pass dataset initial states.
    """

    if not hasattr(env, "reset_from_episode") or not hasattr(env, "step"):
        raise TypeError("rollout_policy requires a real env with reset_from_episode() and step().")
    if initial_states is None:
        if hasattr(env, "episode_initial_state"):
            initial_states = _initial_states_from_dataset(env, n_episodes)
        else:
            raise ValueError("Pass initial_states from LeRobotPickPlaceDataset.episode_initial_state().")

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
        max_steps = int(getattr(getattr(env, "config", None), "horizon", 0))
        steps = 0
        while not done and (max_steps <= 0 or steps < max_steps):
            action = policy(obs).detach().cpu()
            if action.ndim == 2:
                action = action[0]
            if action.ndim == 3:
                action = action[0, 0]
            obs, _, done, info = env.step(action)
            trajectory.append(_position_from_obs(obs))
            steps += 1

        rollout_traj = torch.stack(trajectory)
        trajectories.append(rollout_traj)
        successes.append(float(info.get("success", 0.0)))
        final_distances.append(float(info.get("distance", float("nan"))))
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
    if "object_xyz" in obs:
        return torch.as_tensor(obs["object_xyz"], dtype=torch.float32).flatten()
    if "xy" in obs:
        xy = torch.as_tensor(obs["xy"], dtype=torch.float32).flatten()
        return torch.tensor([xy[0], xy[1], 0.0], dtype=torch.float32)
    state = torch.as_tensor(obs["state"], dtype=torch.float32).flatten()
    return state[-6:-3]


class GenesisFrankaPickPlaceEnv:
    """Thin Genesis Franka pick-and-place environment for collection/eval."""

    def __init__(
        self,
        config: TinyPickPlaceConfig | None = None,
        image_size: int | None = None,
        backend: str = "gpu",
    ) -> None:
        self.config = config or TinyPickPlaceConfig()
        self.image_size = image_size or self.config.image_size
        self.backend = backend
        self._built = False
        self.scene = None
        self.franka = None
        self.cube = None
        self.cubes = []
        self.goal_marker = None
        self.camera = None
        self.ee_link = None
        self.object_xyz = torch.zeros(3)
        self.target_xyz = torch.zeros(3)
        self.label = torch.tensor(0)
        self.t = 0
        self._last_qpos: torch.Tensor | None = None

    def build(self) -> None:
        import genesis as gs

        if not getattr(gs, "_initialized", False):
            backend = gs.gpu if self.backend == "gpu" else gs.cpu
            gs.init(backend=backend, logging_level="ERROR")

        self.scene = gs.Scene(show_viewer=False, renderer=gs.renderers.Rasterizer())
        self.scene.add_entity(gs.morphs.Plane())
        self.franka = self.scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
        self.cubes = []
        for idx, color in enumerate([(0.9, 0.1, 0.1, 1.0), (0.1, 0.45, 0.95, 1.0), (0.1, 0.75, 0.2, 1.0)]):
            self.cubes.append(
                self.scene.add_entity(
                    gs.morphs.Box(pos=(0.45, 0.0, -0.2 - 0.1 * idx), size=(0.04, 0.04, 0.04)),
                    surface=gs.surfaces.Default(color=color),
                    name=f"{COLOR_NAMES[idx]}_cube",
                )
            )
        self.goal_marker = self.scene.add_entity(
            gs.morphs.Box(pos=(0.55, 0.15, 0.004), size=(0.07, 0.07, 0.008), collision=False),
            surface=gs.surfaces.Default(color=(1.0, 1.0, 0.1, 0.55)),
            name="goal_marker",
        )
        self.camera = self.scene.add_camera(
            res=(self.image_size, self.image_size),
            pos=(0.55, -0.85, 0.65),
            lookat=(0.50, 0.0, 0.03),
            fov=48,
        )
        self.scene.build()
        self.ee_link = self.franka.get_link("hand")
        self._built = True

    def reset(
        self,
        object_xy: torch.Tensor,
        target_xy: torch.Tensor,
        label: torch.Tensor | int,
    ) -> dict[str, torch.Tensor | str]:
        if not self._built:
            self.build()
        assert self.scene is not None and self.goal_marker is not None and self.cubes
        self.t = 0
        self.label = torch.as_tensor(label, dtype=torch.long)
        self.cube = self.cubes[int(self.label)]
        self.object_xyz = torch.tensor([0.35 + float(object_xy[0]) * 0.35, -0.22 + float(object_xy[1]) * 0.44, 0.025])
        self.target_xyz = torch.tensor([0.35 + float(target_xy[0]) * 0.35, -0.22 + float(target_xy[1]) * 0.44, 0.025])
        for idx, cube in enumerate(self.cubes):
            if idx == int(self.label):
                cube.set_pos(self.object_xyz, zero_velocity=True)
            else:
                cube.set_pos((0.15 + 0.05 * idx, 0.35, -0.2), zero_velocity=True)
        self.goal_marker.set_pos((float(self.target_xyz[0]), float(self.target_xyz[1]), 0.004), zero_velocity=True)
        self.scene.step()
        self._last_qpos = torch.as_tensor(self.franka.get_qpos(), dtype=torch.float32).flatten()
        return self.observation()

    def reset_from_episode(self, episode: Mapping[str, torch.Tensor | str | int | float]) -> dict[str, torch.Tensor | str]:
        label = int(torch.as_tensor(episode.get("label", 0)).flatten()[0].item())
        if "object_xy" in episode and "target_xy" in episode:
            return self.reset(
                torch.as_tensor(episode["object_xy"], dtype=torch.float32),
                torch.as_tensor(episode["target_xy"], dtype=torch.float32),
                label,
            )
        object_xyz = torch.as_tensor(episode["object_xyz"], dtype=torch.float32)
        target_xyz = torch.as_tensor(episode["target_xyz"], dtype=torch.float32)
        object_xy = torch.stack([(object_xyz[0] - 0.35) / 0.35, (object_xyz[1] + 0.22) / 0.44]).clamp(0.0, 1.0)
        target_xy = torch.stack([(target_xyz[0] - 0.35) / 0.35, (target_xyz[1] + 0.22) / 0.44]).clamp(0.0, 1.0)
        return self.reset(object_xy, target_xy, label)

    def observation(self) -> dict[str, torch.Tensor | str]:
        task = f"pick the {COLOR_NAMES[int(self.label)]} cube"
        image = self.render()
        state = self.lowdim_state()
        return {
            IMAGE_KEY: image,
            STATE_KEY: state,
            "image": image,
            "state": state,
            "object_xyz": self.object_xyz.clone(),
            "target_xyz": self.target_xyz.clone(),
            "label": self.label.clone(),
            "frame_index": torch.tensor(self.t, dtype=torch.long),
            TASK_KEY: task,
            INSTRUCTION_KEY: task,
        }

    def lowdim_state(self) -> torch.Tensor:
        assert self.franka is not None
        qpos = torch.as_tensor(self.franka.get_qpos(), dtype=torch.float32).flatten()
        return torch.cat([qpos, self.object_xyz, self.target_xyz])

    def render(self) -> torch.Tensor:
        assert self.camera is not None
        rgb = self.camera.render(rgb=True, depth=False, segmentation=False)[0]
        return torch.as_tensor(rgb.copy(), dtype=torch.float32).permute(2, 0, 1) / 255.0

    def ik_qpos_for(self, xyz: torch.Tensor, init_qpos: torch.Tensor | None = None) -> torch.Tensor:
        assert self.franka is not None and self.ee_link is not None
        qpos = self.franka.inverse_kinematics(
            link=self.ee_link,
            pos=xyz.detach().cpu().numpy(),
            quat=np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
            init_qpos=None if init_qpos is None else init_qpos.detach().cpu().numpy(),
            rot_mask=[False, False, True],
            max_solver_iters=40,
        )
        return torch.as_tensor(qpos, dtype=torch.float32).flatten()

    def expert_qpos_trajectory(self, steps_per_segment: int = 8) -> torch.Tensor:
        assert self.franka is not None
        current = torch.as_tensor(self.franka.get_qpos(), dtype=torch.float32).flatten()
        above_pick = self.object_xyz + torch.tensor([0.0, 0.0, 0.18])
        grasp = self.object_xyz + torch.tensor([0.0, 0.0, 0.075])
        above_place = self.target_xyz + torch.tensor([0.0, 0.0, 0.18])
        place = self.target_xyz + torch.tensor([0.0, 0.0, 0.075])
        waypoints = [above_pick, grasp, above_pick, above_place, place, above_place]
        q_waypoints = []
        init = current
        for xyz in waypoints:
            q = self.ik_qpos_for(xyz, init_qpos=init)
            q_waypoints.append(q)
            init = q
        trajectory = []
        prev = current
        for q in q_waypoints:
            alpha = torch.linspace(0.0, 1.0, steps_per_segment + 1)[1:, None]
            trajectory.append(prev[None] * (1.0 - alpha) + q[None] * alpha)
            prev = q
        return torch.cat(trajectory, dim=0)

    def success(self) -> bool:
        assert self.cube is not None
        cube_pos = torch.as_tensor(self.cube.get_pos(), dtype=torch.float32).flatten()
        xy_close = torch.dist(cube_pos[:2], self.target_xyz[:2]).item() <= self.config.success_threshold
        low_enough = float(cube_pos[2]) <= 0.055
        return bool(xy_close and low_enough)

    def step(self, action: torch.Tensor) -> tuple[dict[str, torch.Tensor | str], float, bool, dict[str, float]]:
        obs = self.step_qpos(action)
        assert self.cube is not None
        cube_pos = torch.as_tensor(self.cube.get_pos(), dtype=torch.float32).flatten()
        distance = torch.dist(cube_pos[:2], self.target_xyz[:2]).item()
        success = self.success()
        done = success or self.t >= self.config.horizon
        reward = 1.0 if success else -distance
        return obs, reward, done, {"success": float(success), "distance": distance}

    def step_qpos(self, qpos: torch.Tensor) -> dict[str, torch.Tensor | str]:
        assert self.scene is not None and self.franka is not None and self.cube is not None
        qpos = qpos.detach().float().flatten()
        if self._last_qpos is not None and qpos.numel() != self._last_qpos.numel():
            raise ValueError(f"Franka qpos action_dim must be {self._last_qpos.numel()}, got {qpos.numel()}")
        self.franka.control_dofs_position(qpos.cpu().numpy())
        self.scene.step()
        self.object_xyz = torch.as_tensor(self.cube.get_pos(), dtype=torch.float32).flatten()
        self._last_qpos = qpos.clone()
        self.t += 1
        return self.observation()


def collect_genesis_franka_lerobot_dataset(
    root: str | Path,
    repo_id: str = "local/simple-vla-genesis-franka-pick-place",
    config: TinyPickPlaceConfig | None = None,
    n_episodes: int = 8,
    steps_per_segment: int = 4,
    backend: str = "gpu",
    image_size: int = 96,
):
    """Collect IK demonstrations in Genesis and save them as LeRobot data."""

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    cfg = config or TinyPickPlaceConfig(n_episodes=n_episodes, horizon=steps_per_segment * 6, image_size=image_size)
    env = GenesisFrankaPickPlaceEnv(cfg, image_size=image_size, backend=backend)
    generator = torch.Generator().manual_seed(cfg.seed)
    starts = torch.rand(n_episodes, 2, generator=generator) * 0.8 + 0.1
    targets = torch.rand(n_episodes, 2, generator=generator) * 0.8 + 0.1
    labels = torch.randint(len(COLOR_NAMES), (n_episodes,), generator=generator)

    sample_obs = env.reset(starts[0], targets[0], labels[0])
    state = torch.as_tensor(sample_obs["state"], dtype=torch.float32)
    action_dim = int(env.franka.n_qs)  # type: ignore[union-attr]
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=root,
        fps=int(round(1.0 / cfg.dt)),
        features=lerobot_features(image_size=image_size, state_dim=state.numel(), action_dim=action_dim),
        robot_type="genesis_franka",
        use_videos=False,
    )

    for episode in range(n_episodes):
        obs = env.reset(starts[episode], targets[episode], labels[episode])
        q_traj = env.expert_qpos_trajectory(steps_per_segment=steps_per_segment)
        for qpos in q_traj:
            image = torch.as_tensor(obs["image"], dtype=torch.float32)
            state = torch.as_tensor(obs["state"], dtype=torch.float32)
            dataset.add_frame(
                {
                    IMAGE_KEY: _chw_float_to_hwc_uint8(image),
                    STATE_KEY: state.numpy().astype(np.float32),
                    ACTION_KEY: qpos.numpy().astype(np.float32),
                    "object_xyz": torch.as_tensor(obs["object_xyz"]).numpy().astype(np.float32),
                    "target_xyz": torch.as_tensor(obs["target_xyz"]).numpy().astype(np.float32),
                    "label": np.array([int(labels[episode])], dtype=np.int64),
                    TASK_KEY: obs[TASK_KEY],
                }
            )
            obs = env.step_qpos(qpos)
        dataset.save_episode()

    dataset.finalize()
    return dataset


def genesis_integration_notes() -> str:
    return (
        "Use collect_genesis_franka_lerobot_dataset(...) to create LeRobot data, "
        "LeRobotPickPlaceDataset(...) to build DataLoaders, and rollout_policy("
        "policy, GenesisFrankaPickPlaceEnv(...), initial_states) for evaluation. "
        "The shared keys are image/observation.images.camera, state/"
        "observation.state, action, action_chunk, task, and instruction."
    )

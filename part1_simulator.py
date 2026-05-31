"""Part 2: tiny pick-and-place task, LeRobot-style records, and rollouts.

This module still runs without Genesis. The important change from the first
mock is that the data is episode/frame based and has a simulator-like rollout
API, so later chapters can be evaluated by success rate instead of only MSE.

The Genesis integration point is intentionally narrow: environments expose
``reset_from_episode``, ``render`` and ``step``. The standard robot action is a
Franka qpos target; the 2D delta action is kept as a toy fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

import numpy as np
import torch
from torch.utils.data import Dataset

from part2_vision import COLORS

TOY_DELTA_ACTION_DIM = 2
FRANKA_QPOS_ACTION_DIM = 9
GENESIS_STATE_DIM = FRANKA_QPOS_ACTION_DIM + 3 + 3


@dataclass
class TinyPickPlaceConfig:
    n_episodes: int = 256
    horizon: int = 16
    action_dim: int = TOY_DELTA_ACTION_DIM
    image_size: int = 32
    square_size: int = 6
    dt: float = 0.1
    max_action: float = 0.08
    success_threshold: float = 0.045
    seed: int = 0


@dataclass
class TinyPickPlaceState:
    object_xy: torch.Tensor
    target_xy: torch.Tensor
    label: torch.Tensor
    t: int = 0

    @property
    def lowdim(self) -> torch.Tensor:
        return torch.cat([self.object_xy, self.target_xy], dim=-1)


def expert_delta_action(
    xy: torch.Tensor,
    target: torch.Tensor,
    max_action: float = 0.08,
    gain: float = 0.75,
) -> torch.Tensor:
    """Proportional expert in normalized table coordinates."""

    delta = (target - xy) * gain
    norm = delta.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    scale = torch.clamp(max_action / norm, max=1.0)
    return delta * scale


def expert_action_chunk(
    xy: torch.Tensor,
    target: torch.Tensor,
    chunk_size: int,
    max_action: float = 0.08,
) -> torch.Tensor:
    """Roll out the scripted expert and return the next action chunk."""

    actions = []
    current = xy
    for _ in range(chunk_size):
        action = expert_delta_action(current, target, max_action=max_action)
        actions.append(action)
        current = (current + action).clamp(0.0, 1.0)
    return torch.stack(actions, dim=1)


class TinyPickPlaceWorld:
    """A deterministic 2D task with the same contract as a robot env.

    State is normalized table coordinates. The action is a delta end-effector
    command; in this tiny world the grasped object follows that delta. This is
    not physics, but it gives the curriculum real rollouts, success rates and
    compounding-error behavior before Genesis is connected.
    """

    color_names = ("red", "blue", "green")

    def __init__(self, config: TinyPickPlaceConfig | None = None) -> None:
        self.config = config or TinyPickPlaceConfig()
        self.state: TinyPickPlaceState | None = None

    def reset(
        self,
        object_xy: torch.Tensor,
        target_xy: torch.Tensor,
        label: torch.Tensor | int,
    ) -> dict[str, torch.Tensor | str]:
        label_tensor = torch.as_tensor(label, dtype=torch.long)
        self.state = TinyPickPlaceState(
            object_xy=object_xy.detach().float().clone(),
            target_xy=target_xy.detach().float().clone(),
            label=label_tensor,
            t=0,
        )
        return self.observation()

    def reset_from_episode(self, episode: dict[str, torch.Tensor | str | int | float]) -> dict[str, torch.Tensor | str]:
        object_xy = torch.as_tensor(episode["object_xy"], dtype=torch.float32)
        target_xy = torch.as_tensor(episode["target_xy"], dtype=torch.float32)
        label = int(torch.as_tensor(episode.get("label", 0)).item())
        return self.reset(object_xy, target_xy, label)

    def observation(self) -> dict[str, torch.Tensor | str]:
        assert self.state is not None
        instruction = f"pick the {self.color_names[int(self.state.label)]} cube"
        return {
            "image": self.render(),
            "state": self.state.lowdim.clone(),
            "xy": self.state.object_xy.clone(),
            "target": self.state.target_xy.clone(),
            "label": self.state.label.clone(),
            "frame_index": torch.tensor(self.state.t, dtype=torch.long),
            "instruction": instruction,
        }

    def render(self) -> torch.Tensor:
        assert self.state is not None
        return render_square_image(
            self.state.object_xy[None],
            self.state.label[None],
            image_size=self.config.image_size,
            square_size=self.config.square_size,
        )[0]

    def step(self, action: torch.Tensor) -> tuple[dict[str, torch.Tensor | str], float, bool, dict[str, float]]:
        assert self.state is not None
        action = action.detach().float().flatten()[:TOY_DELTA_ACTION_DIM].clamp(-self.config.max_action, self.config.max_action)
        self.state.object_xy = (self.state.object_xy + action).clamp(0.0, 1.0)
        self.state.t += 1
        distance = torch.dist(self.state.object_xy, self.state.target_xy).item()
        success = distance <= self.config.success_threshold
        done = success or self.state.t >= self.config.horizon
        reward = 1.0 if success else -distance
        return self.observation(), reward, done, {"success": float(success), "distance": distance}


class GenesisFrankaPickPlaceEnv:
    """Genesis Franka scene for the curriculum's real simulator path.

    The task contract mirrors ``TinyPickPlaceWorld`` but uses Genesis entities,
    a rendered camera, and Franka qpos target actions. By default the cube is
    left to contact/gripper physics. ``scripted_grasp=True`` is an explicit
    deterministic fallback for lectures where physics grasping is not the focus.
    """

    def __init__(
        self,
        config: TinyPickPlaceConfig | None = None,
        image_size: int | None = None,
        backend: str = "cpu",
        scripted_grasp: bool = False,
    ) -> None:
        self.config = config or TinyPickPlaceConfig()
        self.image_size = image_size or self.config.image_size
        self.backend = backend
        self.scripted_grasp = scripted_grasp
        self._built = False
        self._gs = None
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
        self._grasped = False
        self._last_qpos: torch.Tensor | None = None

    def build(self) -> None:
        import genesis as gs

        if not getattr(gs, "_initialized", False):
            backend = gs.gpu if self.backend == "gpu" else gs.cpu
            gs.init(backend=backend, logging_level="ERROR")

        self._gs = gs
        self.scene = gs.Scene(show_viewer=False, renderer=gs.renderers.Rasterizer())
        self.scene.add_entity(gs.morphs.Plane())
        self.franka = self.scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
        self.cubes = []
        for idx, color in enumerate([(0.9, 0.1, 0.1, 1.0), (0.1, 0.45, 0.95, 1.0), (0.1, 0.75, 0.2, 1.0)]):
            self.cubes.append(
                self.scene.add_entity(
                    gs.morphs.Box(pos=(0.45, 0.0, -0.2 - 0.1 * idx), size=(0.04, 0.04, 0.04)),
                    surface=gs.surfaces.Default(color=color),
                    name=f"{TinyPickPlaceWorld.color_names[idx]}_cube",
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
        self._grasped = False
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

    def reset_from_episode(self, episode: dict[str, torch.Tensor | str | int | float]) -> dict[str, torch.Tensor | str]:
        """Reset from an episode descriptor produced by datasets/adapters.

        Expected keys are either normalized ``object_xy``/``target_xy`` or world
        ``object_xyz``/``target_xyz``. This lets the same rollout function run
        on tiny and Genesis environments without hard-coding the dataset type.
        """

        label = episode.get("label", 0)
        if "object_xy" in episode and "target_xy" in episode:
            object_xy = torch.as_tensor(episode["object_xy"], dtype=torch.float32)
            target_xy = torch.as_tensor(episode["target_xy"], dtype=torch.float32)
            return self.reset(object_xy, target_xy, int(torch.as_tensor(label).item()))
        if "object_xyz" in episode and "target_xyz" in episode:
            object_xyz = torch.as_tensor(episode["object_xyz"], dtype=torch.float32)
            target_xyz = torch.as_tensor(episode["target_xyz"], dtype=torch.float32)
            object_xy = torch.stack([(object_xyz[0] - 0.35) / 0.35, (object_xyz[1] + 0.22) / 0.44]).clamp(0.0, 1.0)
            target_xy = torch.stack([(target_xyz[0] - 0.35) / 0.35, (target_xyz[1] + 0.22) / 0.44]).clamp(0.0, 1.0)
            return self.reset(object_xy, target_xy, int(torch.as_tensor(label).item()))
        raise KeyError("episode must contain object_xy/target_xy or object_xyz/target_xyz")

    def observation(self) -> dict[str, torch.Tensor | str]:
        instruction = f"pick the {TinyPickPlaceWorld.color_names[int(self.label)]} cube"
        return {
            "image": self.render(),
            "state": self.lowdim_state(),
            "xy": self.object_xyz[:2].clone(),
            "target": self.target_xyz[:2].clone(),
            "object_xyz": self.object_xyz.clone(),
            "target_xyz": self.target_xyz.clone(),
            "label": self.label.clone(),
            "frame_index": torch.tensor(self.t, dtype=torch.long),
            "instruction": instruction,
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
        traj = []
        prev = current
        for q in q_waypoints:
            alpha = torch.linspace(0.0, 1.0, steps_per_segment + 1)[1:, None]
            traj.append(prev[None] * (1.0 - alpha) + q[None] * alpha)
            prev = q
        return torch.cat(traj, dim=0)

    def _ee_pos(self) -> torch.Tensor:
        assert self.franka is not None and self.ee_link is not None
        return torch.as_tensor(self.franka.get_links_pos([self.ee_link.idx_local]), dtype=torch.float32).flatten()

    def _update_grasp_state_from_contact(self, qpos: torch.Tensor) -> None:
        """Approximate gripper/contact state using EE distance and finger target.

        Genesis contact APIs vary across versions, so the standard path leaves
        the cube to physics and only records whether the command is a plausible
        closed-gripper contact. ``scripted_grasp=True`` is the explicit teaching
        fallback that attaches the cube for deterministic demos.
        """

        ee_to_cube = torch.dist(self._ee_pos(), self.object_xyz).item()
        gripper_closed = qpos.numel() < 2 or float(qpos[-2:].mean()) < 0.025
        if gripper_closed and ee_to_cube < 0.07:
            self._grasped = True
        if not gripper_closed:
            self._grasped = False

    def success(self) -> bool:
        assert self.cube is not None
        cube_pos = torch.as_tensor(self.cube.get_pos(), dtype=torch.float32).flatten()
        xy_close = torch.dist(cube_pos[:2], self.target_xyz[:2]).item() <= self.config.success_threshold
        low_enough = float(cube_pos[2]) <= 0.055
        return bool(xy_close and low_enough)

    def step(self, action: torch.Tensor) -> tuple[dict[str, torch.Tensor | str], float, bool, dict[str, float]]:
        """Step Genesis with a Franka qpos target action."""

        obs = self.step_qpos(action)
        cube_pos = torch.as_tensor(self.cube.get_pos(), dtype=torch.float32).flatten()  # type: ignore[union-attr]
        distance = torch.dist(cube_pos[:2], self.target_xyz[:2]).item()
        success = self.success()
        done = success or self.t >= self.config.horizon
        reward = 1.0 if success else -distance
        return obs, reward, done, {"success": float(success), "distance": distance}

    def step_qpos(self, qpos: torch.Tensor, grasped: bool | None = None) -> dict[str, torch.Tensor | str]:
        assert self.scene is not None and self.franka is not None and self.cube is not None and self.ee_link is not None
        qpos = qpos.detach().float().flatten()
        if self._last_qpos is not None and qpos.numel() != self._last_qpos.numel():
            raise ValueError(f"Genesis qpos action_dim must be {self._last_qpos.numel()}, got {qpos.numel()}")
        self.franka.control_dofs_position(qpos.cpu().numpy())
        self.scene.step()
        self.object_xyz = torch.as_tensor(self.cube.get_pos(), dtype=torch.float32).flatten()
        if grasped is not None:
            self._grasped = bool(grasped)
        else:
            self._update_grasp_state_from_contact(qpos)
        if self.scripted_grasp and self._grasped:
            ee_pos = torch.as_tensor(self.franka.get_links_pos([self.ee_link.idx_local]), dtype=torch.float32).flatten()
            self.object_xyz = torch.tensor([ee_pos[0], ee_pos[1], max(float(ee_pos[2]) - 0.05, 0.025)])
            self.cube.set_pos(self.object_xyz, zero_velocity=True)
            self.scene.step()
        else:
            self.object_xyz = torch.as_tensor(self.cube.get_pos(), dtype=torch.float32).flatten()
        self._last_qpos = qpos.clone()
        self.t += 1
        return self.observation()


def render_square_image(
    xy: torch.Tensor,
    label: torch.Tensor,
    image_size: int = 32,
    square_size: int = 6,
    noise: float = 0.01,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    image = torch.full((len(label), 3, image_size, image_size), 0.05, device=xy.device)
    start = (xy * (image_size - square_size - 1)).long().clamp(0, image_size - square_size)
    colors = COLORS.to(xy.device)
    for i, (row, col) in enumerate(start):
        image[i, :, row : row + square_size, col : col + square_size] = colors[label[i]][:, None, None]
    if noise > 0:
        kwargs = {"generator": generator} if generator is not None and xy.device.type == "cpu" else {}
        image = image + noise * torch.randn(image.shape, device=xy.device, **kwargs)
    return image.clamp(0.0, 1.0)


def generate_expert_episodes(config: TinyPickPlaceConfig, chunk_size: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(config.seed)
    labels = torch.randint(3, (config.n_episodes,), generator=generator)
    starts = torch.rand(config.n_episodes, 2, generator=generator) * 0.8 + 0.1
    targets = torch.rand(config.n_episodes, 2, generator=generator) * 0.8 + 0.1

    xy = torch.zeros(config.n_episodes, config.horizon, 2)
    actions = torch.zeros(config.n_episodes, config.horizon, config.action_dim)
    distance = torch.zeros(config.n_episodes, config.horizon)
    xy[:, 0] = starts
    for t in range(config.horizon):
        current = xy[:, t]
        action = expert_delta_action(current, targets, max_action=config.max_action)
        actions[:, t, :TOY_DELTA_ACTION_DIM] = action
        distance[:, t] = (targets - current).norm(dim=-1)
        if t + 1 < config.horizon:
            xy[:, t + 1] = (current + action).clamp(0.0, 1.0)

    action_chunks = torch.zeros(config.n_episodes, config.horizon, chunk_size, config.action_dim)
    for t in range(config.horizon):
        current = xy[:, t]
        action_chunks[:, t, :, :TOY_DELTA_ACTION_DIM] = expert_action_chunk(current, targets, chunk_size, config.max_action)

    return {
        "label": labels,
        "start_xy": starts,
        "target": targets,
        "xy": xy,
        "action": actions,
        "action_chunk": action_chunks,
        "distance": distance,
    }


class TinyPickPlaceDataset(Dataset):
    """Frame dataset with both old keys and LeRobot-style observation keys."""

    def __init__(
        self,
        config: TinyPickPlaceConfig | None = None,
        chunk_size: int = 4,
        frames_per_episode: int | None = None,
    ) -> None:
        self.config = config or TinyPickPlaceConfig()
        self.chunk_size = chunk_size
        self.episodes = generate_expert_episodes(self.config, chunk_size)
        self.frames_per_episode = frames_per_episode or self.config.horizon
        self.indices = [
            (episode, frame)
            for episode in range(self.config.n_episodes)
            for frame in range(min(self.frames_per_episode, self.config.horizon))
        ]
        self.instruction = [
            f"pick the {TinyPickPlaceWorld.color_names[int(i)]} cube" for i in self.episodes["label"]
        ]
        episode_index = torch.tensor([episode for episode, _ in self.indices], dtype=torch.long)
        frame_index = torch.tensor([frame for _, frame in self.indices], dtype=torch.long)
        self.label = self.episodes["label"][episode_index]
        self.xy = self.episodes["xy"][episode_index, frame_index]
        self.target = self.episodes["target"][episode_index]
        self.state = torch.cat([self.xy, self.target], dim=-1)
        self.action = self.episodes["action"][episode_index, frame_index]
        self.action_chunk = self.episodes["action_chunk"][episode_index, frame_index]
        self.image = render_square_image(
            self.xy,
            self.label,
            image_size=self.config.image_size,
            square_size=self.config.square_size,
            noise=0.0,
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        episode, frame = self.indices[index]
        label = self.episodes["label"][episode]
        xy = self.episodes["xy"][episode, frame]
        target = self.episodes["target"][episode]
        state = torch.cat([xy, target], dim=-1)
        image = render_square_image(
            xy[None],
            label[None],
            image_size=self.config.image_size,
            square_size=self.config.square_size,
            noise=0.01,
        )[0]
        action = self.episodes["action"][episode, frame]
        action_chunk = self.episodes["action_chunk"][episode, frame]
        done = frame == self.config.horizon - 1
        timestamp = torch.tensor(frame * self.config.dt, dtype=torch.float32)

        return {
            # LeRobot-style names.
            "observation.images.camera": image,
            "observation.state": state,
            "action": action,
            "episode_index": torch.tensor(episode, dtype=torch.long),
            "frame_index": torch.tensor(frame, dtype=torch.long),
            "timestamp": timestamp,
            "next.done": torch.tensor(done),
            "task": self.instruction[episode],
            # Backward-compatible aliases used by the rest of the notebook.
            "image": image,
            "state": state,
            "label": label,
            "xy": xy,
            "target": target,
            "object_xyz": torch.tensor([xy[0], xy[1], 0.025], dtype=torch.float32),
            "target_xyz": torch.tensor([target[0], target[1], 0.025], dtype=torch.float32),
            "action_chunk": action_chunk,
            "instruction": self.instruction[episode],
        }

    def episode_initial_state(self, episode: int) -> dict[str, torch.Tensor]:
        start_xy = self.episodes["start_xy"][episode]
        target = self.episodes["target"][episode]
        label = self.episodes["label"][episode]
        return {"object_xy": start_xy, "target_xy": target, "label": label}


def collate_tiny_pick_place(batch: list[dict[str, torch.Tensor | str]]) -> dict[str, torch.Tensor | list[str]]:
    output: dict[str, torch.Tensor | list[str]] = {}
    for key in batch[0].keys():
        values = [item[key] for item in batch]
        if isinstance(values[0], torch.Tensor):
            output[key] = torch.stack(values)  # type: ignore[arg-type]
        else:
            output[key] = values  # type: ignore[assignment]
    return output


class LeRobotPickPlaceDataset(Dataset):
    """Adapter that makes a real LeRobotDataset look like the chapter datasets.

    Output keys are the curriculum standard:
    ``image``, ``state``, ``action``, ``action_chunk``, ``task`` and
    ``instruction``. The adapter does not assume tiny 2D actions; chunks are
    built directly from the stored action vectors, so Franka qpos targets flow
    into parts 3-6 unchanged.
    """

    def __init__(
        self,
        root: str | Path,
        repo_id: str | None = None,
        chunk_size: int = 4,
        image_key: str = "observation.images.camera",
    ) -> None:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self.root = Path(root)
        self.chunk_size = chunk_size
        self.image_key = image_key
        self.dataset = LeRobotDataset(repo_id=repo_id, root=self.root) if repo_id else LeRobotDataset(root=self.root)
        self._episode_to_indices = self._build_episode_index()
        sample = self[0]
        self.action_dim = int(sample["action"].numel())  # type: ignore[union-attr]
        self.state_dim = int(sample["state"].numel())  # type: ignore[union-attr]
        self.config = TinyPickPlaceConfig(
            n_episodes=len(self._episode_to_indices),
            horizon=max(len(v) for v in self._episode_to_indices.values()),
            action_dim=self.action_dim,
        )

    def _build_episode_index(self) -> dict[int, list[int]]:
        episodes: dict[int, list[int]] = {}
        for idx in range(len(self.dataset)):
            row = self.dataset[idx]
            episode = int(torch.as_tensor(row.get("episode_index", 0)).item())
            episodes.setdefault(episode, []).append(idx)
        return episodes

    @staticmethod
    def _image_to_chw_float(image) -> torch.Tensor:
        tensor = torch.as_tensor(image)
        if tensor.ndim == 3 and tensor.shape[-1] == 3:
            tensor = tensor.permute(2, 0, 1)
        tensor = tensor.float()
        if tensor.max() > 1.0:
            tensor = tensor / 255.0
        return tensor.clamp(0.0, 1.0)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        row = self.dataset[index]
        image = self._image_to_chw_float(row[self.image_key])
        state = torch.as_tensor(row["observation.state"], dtype=torch.float32).flatten()
        action = torch.as_tensor(row["action"], dtype=torch.float32).flatten()
        episode = int(torch.as_tensor(row.get("episode_index", 0)).item())
        indices = self._episode_to_indices[episode]
        local = indices.index(index)
        chunk = []
        for offset in range(self.chunk_size):
            chunk_index = indices[min(local + offset, len(indices) - 1)]
            chunk.append(torch.as_tensor(self.dataset[chunk_index]["action"], dtype=torch.float32).flatten())
        task = str(row.get("task", "pick the cube"))
        out: dict[str, torch.Tensor | str] = {
            "observation.images.camera": image,
            "observation.state": state,
            "image": image,
            "state": state,
            "action": action,
            "action_chunk": torch.stack(chunk),
            "episode_index": torch.tensor(episode, dtype=torch.long),
            "frame_index": torch.as_tensor(row.get("frame_index", local), dtype=torch.long),
            "task": task,
            "instruction": task,
        }
        for optional in ("object_xyz", "target_xyz", "label"):
            if optional in row:
                out[optional] = torch.as_tensor(row[optional]).flatten()
        return out

    def episode_initial_state(self, episode: int) -> dict[str, torch.Tensor | str | int | float]:
        first = self[self._episode_to_indices[episode][0]]
        if "object_xyz" in first and "target_xyz" in first:
            return {
                "object_xyz": first["object_xyz"],  # type: ignore[dict-item]
                "target_xyz": first["target_xyz"],  # type: ignore[dict-item]
                "label": int(torch.as_tensor(first.get("label", torch.tensor([0]))).flatten()[0].item()),
            }
        state = torch.as_tensor(first["state"], dtype=torch.float32)
        if state.numel() >= 15:
            return {"object_xyz": state[-6:-3], "target_xyz": state[-3:], "label": 0}
        raise KeyError("LeRobot dataset needs object_xyz/target_xyz fields or a Genesis state suffix.")


def lerobot_features(
    image_size: int,
    state_dim: int,
    action_dim: int,
    image_key: str = "observation.images.camera",
) -> dict[str, dict[str, object]]:
    return {
        image_key: {
            "dtype": "image",
            "shape": (image_size, image_size, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": [f"state_{i}" for i in range(state_dim)],
        },
        "action": {
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


def _chw_float_to_hwc_uint8(image: torch.Tensor) -> np.ndarray:
    return (image.detach().cpu().permute(1, 2, 0).clamp(0.0, 1.0).numpy() * 255).astype(np.uint8)


def save_lerobot_dataset(
    dataset: TinyPickPlaceDataset,
    root: str | Path,
    repo_id: str = "local/simple-vla-tiny-pick-place",
    image_key: str = "observation.images.camera",
    use_videos: bool = False,
):
    """Write the dataset with the real LeRobotDataset writer."""

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = Path(root)
    features = lerobot_features(
        image_size=dataset.config.image_size,
        state_dim=dataset.state.shape[-1],
        action_dim=dataset.config.action_dim,
        image_key=image_key,
    )
    lr_dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=root,
        fps=int(round(1.0 / dataset.config.dt)),
        features=features,
        robot_type="tiny_pick_place",
        use_videos=use_videos,
    )
    frames_per_episode = min(dataset.frames_per_episode, dataset.config.horizon)
    for episode in range(dataset.config.n_episodes):
        for frame in range(frames_per_episode):
            row = dataset[episode * frames_per_episode + frame]
            image = row[image_key]
            state = row["observation.state"]
            action = row["action"]
            assert isinstance(image, torch.Tensor)
            assert isinstance(state, torch.Tensor)
            assert isinstance(action, torch.Tensor)
            lr_dataset.add_frame(
                {
                    image_key: _chw_float_to_hwc_uint8(image),
                    "observation.state": state.numpy().astype(np.float32),
                    "action": action.numpy().astype(np.float32),
                    "object_xyz": np.array([row["xy"][0], row["xy"][1], 0.025], dtype=np.float32),
                    "target_xyz": np.array([row["target"][0], row["target"][1], 0.025], dtype=np.float32),
                    "label": np.array([int(row["label"])], dtype=np.int64),
                    "task": row["task"],
                }
            )
        lr_dataset.save_episode()
    lr_dataset.finalize()
    return lr_dataset


def save_lerobot_style_dataset(dataset: TinyPickPlaceDataset, root: str | Path) -> None:
    """Backward-compatible alias for notebooks; now writes real LeRobot data."""

    save_lerobot_dataset(dataset, root)


PolicyFn = Callable[[dict[str, torch.Tensor | str]], torch.Tensor]


class RolloutEnv(Protocol):
    config: TinyPickPlaceConfig

    def reset_from_episode(self, episode: dict[str, torch.Tensor | str | int | float]) -> dict[str, torch.Tensor | str]:
        ...

    def step(self, action: torch.Tensor) -> tuple[dict[str, torch.Tensor | str], float, bool, dict[str, float]]:
        ...


def _initial_states_from_dataset(dataset, n_episodes: int | None = None) -> list[dict[str, torch.Tensor | str | int | float]]:
    if hasattr(dataset, "episode_initial_state"):
        total = getattr(getattr(dataset, "config", None), "n_episodes", None) or getattr(dataset, "num_episodes", 0)
        n = min(n_episodes or total, total)
        return [dataset.episode_initial_state(i) for i in range(n)]
    if hasattr(dataset, "episode_initial_states"):
        states = list(dataset.episode_initial_states())
        return states[: n_episodes or len(states)]
    raise TypeError("Pass initial_states explicitly or use a dataset with episode_initial_state(s).")


@torch.no_grad()
def rollout_policy(
    policy: PolicyFn,
    env: RolloutEnv | TinyPickPlaceDataset,
    initial_states: list[dict[str, torch.Tensor | str | int | float]] | None = None,
    n_episodes: int | None = None,
    config: TinyPickPlaceConfig | None = None,
    ) -> dict[str, object]:
    """Roll out a policy in any env exposing reset_from_episode/step.

    Backward compatibility: passing ``TinyPickPlaceDataset`` as the second
    argument still evaluates in ``TinyPickPlaceWorld``. The preferred standard
    path is ``rollout_policy(policy, env, initial_states)`` so Genesis success
    rate is measured in Genesis, not in the 2D toy world.
    """

    dataset = env if isinstance(env, TinyPickPlaceDataset) else None
    if dataset is not None:
        rollout_config = config or dataset.config
        world: RolloutEnv = TinyPickPlaceWorld(rollout_config)  # type: ignore[assignment]
        initial_states = initial_states or _initial_states_from_dataset(dataset, n_episodes)
    else:
        world = env  # type: ignore[assignment]
        initial_states = initial_states or []
    n = min(n_episodes or len(initial_states), len(initial_states))
    successes = []
    final_distances = []
    drift_from_expert = []
    trajectories = []

    for episode in range(n):
        init = initial_states[episode]
        obs = world.reset_from_episode(init)
        start_xy = obs.get("xy", obs.get("object_xyz"))
        traj = [start_xy.clone()] if isinstance(start_xy, torch.Tensor) else []
        done = False
        info = {"distance": float("nan"), "success": 0.0}
        while not done:
            action = policy(obs).detach().cpu()
            if action.ndim == 2:
                action = action[0]
            if action.ndim == 3:
                action = action[0, 0]
            obs, _, done, info = world.step(action)
            pos = obs.get("xy", obs.get("object_xyz"))
            if isinstance(pos, torch.Tensor):
                traj.append(pos.clone())

        trajectory = torch.stack(traj) if traj else torch.empty(0)
        if dataset is not None and trajectory.numel() > 0:
            expert_xy = dataset.episodes["xy"][episode, : len(trajectory)]
            common = min(len(trajectory), len(expert_xy))
            drift_from_expert.append((trajectory[:common] - expert_xy[:common]).norm(dim=-1).mean())
        successes.append(info["success"])
        final_distances.append(info["distance"])
        trajectories.append(trajectory)

    drift = float(torch.stack(drift_from_expert).mean().item()) if drift_from_expert else float("nan")
    return {
        "success_rate": float(torch.tensor(successes).float().mean().item()),
        "mean_final_distance": float(torch.tensor(final_distances).float().mean().item()),
        "mean_expert_drift": drift,
        "trajectories": trajectories,
    }


def scripted_expert_policy(obs: dict[str, torch.Tensor | str], max_action: float = 0.08) -> torch.Tensor:
    xy = obs["xy"]
    target = obs["target"]
    assert isinstance(xy, torch.Tensor)
    assert isinstance(target, torch.Tensor)
    return expert_delta_action(xy, target, max_action=max_action)


def collect_genesis_franka_lerobot_dataset(
    root: str | Path,
    repo_id: str = "local/simple-vla-genesis-franka-pick-place",
    config: TinyPickPlaceConfig | None = None,
    n_episodes: int = 8,
    steps_per_segment: int = 4,
    backend: str = "cpu",
    image_size: int = 96,
    scripted_grasp: bool = False,
):
    """Collect Franka IK demonstrations in Genesis and save real LeRobot data."""

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    cfg = config or TinyPickPlaceConfig(n_episodes=n_episodes, horizon=steps_per_segment * 6, image_size=image_size)
    env = GenesisFrankaPickPlaceEnv(cfg, image_size=image_size, backend=backend, scripted_grasp=scripted_grasp)
    generator = torch.Generator().manual_seed(cfg.seed)
    starts = torch.rand(n_episodes, 2, generator=generator) * 0.8 + 0.1
    targets = torch.rand(n_episodes, 2, generator=generator) * 0.8 + 0.1
    labels = torch.randint(3, (n_episodes,), generator=generator)

    sample_obs = env.reset(starts[0], targets[0], labels[0])
    state = sample_obs["state"]
    assert isinstance(state, torch.Tensor)
    action_dim = int(env.franka.n_qs)  # type: ignore[union-attr]
    features = lerobot_features(image_size=image_size, state_dim=state.numel(), action_dim=action_dim)
    lr_dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=root,
        fps=int(round(1.0 / cfg.dt)),
        features=features,
        robot_type="genesis_franka",
        use_videos=False,
    )

    for episode in range(n_episodes):
        obs = env.reset(starts[episode], targets[episode], labels[episode])
        q_traj = env.expert_qpos_trajectory(steps_per_segment=steps_per_segment)
        grasp_start = steps_per_segment
        release_start = steps_per_segment * 4
        for step, qpos in enumerate(q_traj):
            grasped = grasp_start <= step < release_start
            image = obs["image"]
            state = obs["state"]
            assert isinstance(image, torch.Tensor)
            assert isinstance(state, torch.Tensor)
            lr_dataset.add_frame(
                {
                    "observation.images.camera": _chw_float_to_hwc_uint8(image),
                    "observation.state": state.numpy().astype(np.float32),
                    "action": qpos.numpy().astype(np.float32),
                    "object_xyz": obs["object_xyz"].numpy().astype(np.float32),  # type: ignore[union-attr]
                    "target_xyz": obs["target_xyz"].numpy().astype(np.float32),  # type: ignore[union-attr]
                    "label": np.array([int(labels[episode])], dtype=np.int64),
                    "task": obs["instruction"],
                }
            )
            obs = env.step_qpos(qpos, grasped=grasped if scripted_grasp else None)
        lr_dataset.save_episode()

    lr_dataset.finalize()
    return lr_dataset


def genesis_integration_notes() -> str:
    """Return the concrete replacement points for the next implementation step."""

    return (
        "Use GenesisFrankaPickPlaceEnv plus collect_genesis_franka_lerobot_dataset: "
        "the collector resets cube and goal pose, renders observation.images.camera, "
        "solves Franka IK waypoints, linearly interpolates qpos, and writes episodes "
        "with the real LeRobotDataset writer. Closed-loop evaluation should call "
        "rollout_policy(policy, genesis_env, initial_states). Set scripted_grasp=True "
        "only for the deterministic teaching fallback; the standard path leaves "
        "grasping to gripper/contact physics and evaluates cube pose via success()."
    )

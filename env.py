"""Genesis Franka pick-and-place environment for SimpleVLA."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch


IMAGE_KEY = "observation.images.camera"
STATE_KEY = "observation.state"
ROBOT_QPOS_KEY = "observation.robot_qpos"
ACTION_KEY = "action"
ACTION_CHUNK_KEY = "action_chunk"
OBJECT_POSE_KEY = "object_pose"
TARGET_POSE_KEY = "target_pose"
OBJECT_XYZ_KEY = "object_xyz"
TARGET_XYZ_KEY = "target_xyz"
GRIPPER_WIDTH_KEY = "gripper_width"
TASK_KEY = "task"
INSTRUCTION_KEY = "instruction"

COLOR_NAMES = ("red", "blue", "green")
COLORS = torch.tensor(
    [
        [0.90, 0.10, 0.10],
        [0.10, 0.55, 0.95],
        [0.15, 0.75, 0.20],
    ],
    dtype=torch.float32,
)

FRANKA_QPOS_ACTION_DIM = 9
GENESIS_STATE_DIM = FRANKA_QPOS_ACTION_DIM + 3 + 3 + 1


@dataclass
class TinyPickPlaceConfig:
    """Shared configuration for phase 1 and the later notebooks."""

    n_episodes: int = 8
    horizon: int = 72
    image_size: int = 96
    fps: int = 10
    dt: float = 0.1
    seed: int = 0
    action_dim: int = FRANKA_QPOS_ACTION_DIM
    state_dim: int = GENESIS_STATE_DIM
    success_threshold: float = 0.05
    table_size: tuple[float, float, float] = (0.80, 0.60, 0.05)
    cube_size: float = 0.04
    x_range: tuple[float, float] = (0.34, 0.64)
    y_range: tuple[float, float] = (-0.22, 0.22)
    gripper_open: float = 0.04
    gripper_closed: float = 0.0
    home_qpos: tuple[float, ...] = (0.0, -0.4, 0.0, -2.2, 0.0, 2.0, 0.8, 0.04, 0.04)


@dataclass
class ExpertWaypoint:
    stage: str
    xyz: torch.Tensor
    finger_qpos: float


class GenesisFrankaPickPlaceEnv:
    """Headless Genesis + Franka + colored cube pick-and-place environment."""

    def __init__(
        self,
        config: TinyPickPlaceConfig | None = None,
        image_size: int | None = None,
        show_viewer: bool = False,
        backend: str = "gpu",
    ) -> None:
        self.config = config or TinyPickPlaceConfig()
        self.image_size = image_size or self.config.image_size
        self.show_viewer = show_viewer
        self.backend = backend
        self._built = False
        self.scene = None
        self.franka = None
        self.table = None
        self.cubes = []
        self.cube = None
        self.goal_marker = None
        self.camera = None
        self.ee_link = None
        self.motors_dof = np.arange(7)
        self.fingers_dof = np.arange(7, 9)
        self.object_xyz = torch.zeros(3)
        self.target_xyz = torch.zeros(3)
        self.label = torch.tensor(0)
        self.t = 0

    def build(self) -> None:
        import genesis as gs

        if not getattr(gs, "_initialized", False):
            gs.init(backend=gs.gpu if self.backend == "gpu" else gs.cpu)

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=0.01),
            rigid_options=gs.options.RigidOptions(box_box_detection=True),
            show_viewer=self.show_viewer,
            renderer=gs.renderers.Rasterizer(),
        )
        self.scene.add_entity(gs.morphs.Plane())
        self.table = self.scene.add_entity(
            gs.morphs.Box(pos=(0.50, 0.0, -self.config.table_size[2] / 2), size=self.config.table_size, fixed=True),
            surface=gs.surfaces.Default(color=(0.72, 0.72, 0.68, 1.0)),
        )
        self.franka = self.scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
        self.cubes = [
            self.scene.add_entity(
                gs.morphs.Box(
                    pos=(0.20 + 0.06 * i, 0.34, -0.20),
                    size=(self.config.cube_size, self.config.cube_size, self.config.cube_size),
                ),
                material=gs.materials.Rigid(rho=50, friction=1.5, coup_friction=1.0, coup_softness=0.001),
                surface=gs.surfaces.Default(color=(*COLORS[i].tolist(), 1.0)),
                name=f"{COLOR_NAMES[i]}_cube",
            )
            for i in range(len(COLOR_NAMES))
        ]
        self.goal_marker = self.scene.add_entity(
            gs.morphs.Box(pos=(0.55, 0.12, 0.004), size=(0.07, 0.07, 0.008), collision=False, fixed=True),
            surface=gs.surfaces.Default(color=(1.0, 0.90, 0.10, 0.55)),
            name="target_marker",
        )
        self.camera = self.scene.add_camera(
            res=(self.image_size, self.image_size),
            pos=(0.62, -0.82, 0.64),
            lookat=(0.50, 0.00, 0.04),
            fov=48,
        )
        self.scene.build()
        self.ee_link = self.franka.get_link("hand")
        self._built = True

    def reset(self, object_xy: torch.Tensor, target_xy: torch.Tensor, label: torch.Tensor | int) -> dict[str, torch.Tensor | str]:
        if not self._built:
            self.build()
        assert self.scene is not None and self.franka is not None and self.goal_marker is not None
        self.t = 0
        self.label = torch.as_tensor(label, dtype=torch.long).flatten()[0]
        self.cube = self.cubes[int(self.label)]
        self.object_xyz = self._workspace_xyz(object_xy)
        self.target_xyz = self._workspace_xyz(target_xy)

        qpos = torch.tensor(self.config.home_qpos, dtype=torch.float32)
        self.franka.set_dofs_kp(np.array([3000, 2500, 2000, 2000, 1500, 1500, 1500, 100, 100]))
        self.franka.set_dofs_kv(np.array([600, 600, 500, 500, 400, 400, 400, 20, 20]))
        self.franka.set_dofs_force_range(
            np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
            np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]),
        )
        self.franka.set_qpos(qpos.detach().cpu().numpy(), zero_velocity=True)
        self.franka.control_dofs_position(qpos[:7].detach().cpu().numpy(), self.motors_dof)
        self.franka.control_dofs_position(qpos[7:].detach().cpu().numpy(), self.fingers_dof)
        for idx, cube in enumerate(self.cubes):
            if idx == int(self.label):
                cube.set_pos(self.object_xyz.detach().cpu().numpy(), zero_velocity=True)
            else:
                cube.set_pos((0.20 + 0.06 * idx, 0.34, -0.20), zero_velocity=True)
        self.goal_marker.set_pos((float(self.target_xyz[0]), float(self.target_xyz[1]), 0.004), zero_velocity=True)
        self.scene.step()
        return self.observation()

    def reset_random(self, generator: torch.Generator | None = None) -> dict[str, torch.Tensor | str]:
        object_xy = torch.rand(2, generator=generator) * 0.8 + 0.1
        target_xy = torch.rand(2, generator=generator) * 0.8 + 0.1
        label = torch.randint(len(COLOR_NAMES), (1,), generator=generator)[0]
        return self.reset(object_xy, target_xy, label)

    def reset_from_episode(self, episode: Mapping[str, torch.Tensor | str | int | float]) -> dict[str, torch.Tensor | str]:
        label = int(torch.as_tensor(episode.get("label", 0)).flatten()[0].item())
        object_xyz = torch.as_tensor(episode[OBJECT_XYZ_KEY], dtype=torch.float32).flatten()
        target_xyz = torch.as_tensor(episode[TARGET_XYZ_KEY], dtype=torch.float32).flatten()
        return self.reset(self._normalize_xy(object_xyz), self._normalize_xy(target_xyz), label)

    def _workspace_xyz(self, xy01: torch.Tensor) -> torch.Tensor:
        xy01 = torch.as_tensor(xy01, dtype=torch.float32).flatten().clamp(0.0, 1.0)
        x = self.config.x_range[0] + xy01[0] * (self.config.x_range[1] - self.config.x_range[0])
        y = self.config.y_range[0] + xy01[1] * (self.config.y_range[1] - self.config.y_range[0])
        z = self.config.cube_size / 2 + 0.002
        return torch.tensor([x, y, z], dtype=torch.float32)

    def _normalize_xy(self, xyz: torch.Tensor) -> torch.Tensor:
        return torch.tensor(
            [
                (xyz[0] - self.config.x_range[0]) / (self.config.x_range[1] - self.config.x_range[0]),
                (xyz[1] - self.config.y_range[0]) / (self.config.y_range[1] - self.config.y_range[0]),
            ],
            dtype=torch.float32,
        ).clamp(0.0, 1.0)

    def observation(self) -> dict[str, torch.Tensor | str]:
        assert self.franka is not None and self.cube is not None
        qpos = torch.as_tensor(self.franka.get_dofs_position(), dtype=torch.float32).flatten()
        object_pose = self.object_pose()
        target_pose = torch.cat([self.target_xyz, torch.tensor([1.0, 0.0, 0.0, 0.0])])
        gripper_width = qpos[-2:].sum()[None]
        state = torch.cat([qpos, self.object_xyz, self.target_xyz, gripper_width])
        task = f"pick the {COLOR_NAMES[int(self.label)]} cube and place it on the target"
        image = self.render()
        return {
            IMAGE_KEY: image,
            STATE_KEY: state,
            ROBOT_QPOS_KEY: qpos,
            OBJECT_POSE_KEY: object_pose,
            TARGET_POSE_KEY: target_pose,
            OBJECT_XYZ_KEY: self.object_xyz.clone(),
            TARGET_XYZ_KEY: self.target_xyz.clone(),
            GRIPPER_WIDTH_KEY: gripper_width,
            "label": self.label.clone()[None],
            "image": image,
            "state": state,
            "frame_index": torch.tensor(self.t, dtype=torch.long),
            TASK_KEY: task,
            INSTRUCTION_KEY: task,
        }

    def object_pose(self) -> torch.Tensor:
        assert self.cube is not None
        self.object_xyz = torch.as_tensor(self.cube.get_pos(), dtype=torch.float32).flatten()
        quat = torch.as_tensor(self.cube.get_quat(), dtype=torch.float32).flatten()
        return torch.cat([self.object_xyz, quat])

    def render(self) -> torch.Tensor:
        assert self.camera is not None
        rgb = self.camera.render(rgb=True, depth=False, segmentation=False)[0]
        return torch.as_tensor(rgb.copy(), dtype=torch.float32).permute(2, 0, 1) / 255.0

    def ik_qpos_for(self, xyz: torch.Tensor, finger_qpos: float, init_qpos: torch.Tensor | None = None) -> torch.Tensor:
        assert self.franka is not None and self.ee_link is not None
        qpos = self.franka.inverse_kinematics(
            link=self.ee_link,
            pos=xyz.detach().cpu().numpy(),
            quat=np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
            init_qpos=None if init_qpos is None else init_qpos.detach().cpu().numpy(),
            rot_mask=[False, False, True],
            max_solver_iters=40,
        )
        qpos = torch.as_tensor(qpos, dtype=torch.float32).flatten()
        qpos[-2:] = float(finger_qpos)
        return qpos

    def expert_waypoints(self) -> list[ExpertWaypoint]:
        open_q = self.config.gripper_open
        closed_q = self.config.gripper_closed
        return [
            ExpertWaypoint("pre-grasp", self.object_xyz + torch.tensor([0.0, 0.0, 0.18]), open_q),
            ExpertWaypoint("grasp", self.object_xyz + torch.tensor([0.0, 0.0, 0.065]), closed_q),
            ExpertWaypoint("lift", self.object_xyz + torch.tensor([0.0, 0.0, 0.20]), closed_q),
            ExpertWaypoint("move-to-place", self.target_xyz + torch.tensor([0.0, 0.0, 0.20]), closed_q),
            ExpertWaypoint("place", self.target_xyz + torch.tensor([0.0, 0.0, 0.065]), open_q),
            ExpertWaypoint("retreat", self.target_xyz + torch.tensor([0.0, 0.0, 0.22]), open_q),
        ]

    def expert_qpos_trajectory(self, steps_per_segment: int = 12) -> torch.Tensor:
        assert self.franka is not None
        current = torch.as_tensor(self.franka.get_dofs_position(), dtype=torch.float32).flatten()
        q_waypoints = []
        init = current
        for waypoint in self.expert_waypoints():
            q = self.ik_qpos_for(waypoint.xyz, waypoint.finger_qpos, init_qpos=init)
            q_waypoints.append(q)
            init = q

        trajectory = []
        prev = current
        for target in q_waypoints:
            alpha = torch.linspace(0.0, 1.0, steps_per_segment + 1)[1:, None]
            trajectory.append(prev[None] * (1.0 - alpha) + target[None] * alpha)
            prev = target
        return torch.cat(trajectory, dim=0)

    def step(self, action: torch.Tensor) -> tuple[dict[str, torch.Tensor | str], float, bool, dict[str, float]]:
        obs = self.step_qpos(action)
        distance = torch.dist(torch.as_tensor(obs[OBJECT_XYZ_KEY])[:2], self.target_xyz[:2]).item()
        success = self.success()
        done = success or self.t >= self.config.horizon
        return obs, 1.0 if success else -distance, done, {"success": float(success), "distance": distance}

    def step_qpos(self, qpos: torch.Tensor) -> dict[str, torch.Tensor | str]:
        assert self.scene is not None and self.franka is not None
        qpos = qpos.detach().float().flatten()
        self.franka.control_dofs_position(qpos[:7].detach().cpu().numpy(), self.motors_dof)
        self.franka.control_dofs_position(qpos[7:].detach().cpu().numpy(), self.fingers_dof)
        self.scene.step()
        self.t += 1
        return self.observation()

    def success(self) -> bool:
        assert self.cube is not None
        cube_pos = torch.as_tensor(self.cube.get_pos(), dtype=torch.float32).flatten()
        xy_close = torch.dist(cube_pos[:2], self.target_xyz[:2]).item() <= self.config.success_threshold
        released = torch.as_tensor(self.franka.get_dofs_position(), dtype=torch.float32).flatten()[-2:].sum().item() > 0.05
        return bool(xy_close and released)

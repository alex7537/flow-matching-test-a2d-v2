from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _recovery_joint_targets(
    start: np.ndarray,
    target: np.ndarray,
    *,
    open_steps: int,
    retreat_steps: int,
    settle_steps: int,
) -> np.ndarray:
    start = np.asarray(start, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    if start.shape != (13,) or target.shape != (13,):
        raise ValueError("recovery start and target must have shape (13,)")
    if not np.isfinite(start).all() or not np.isfinite(target).all():
        raise ValueError("recovery start and target must be finite")
    if open_steps < 1 or retreat_steps < 1 or settle_steps < 0:
        raise ValueError("recovery open/retreat steps must be positive and settle non-negative")

    hand_open = start.copy()
    hand_open[7:] = target[7:]
    commands = [
        start + (float(index) / float(open_steps)) * (hand_open - start)
        for index in range(1, open_steps + 1)
    ]
    commands.extend(
        hand_open + (float(index) / float(retreat_steps)) * (target - hand_open)
        for index in range(1, retreat_steps + 1)
    )
    commands.extend(target.copy() for _ in range(settle_steps))
    return np.asarray(commands, dtype=np.float32)


class GraspEnv:
    """Thin Isaac Sim adapter; construct only after SimulationApp exists."""

    def __init__(self, *, physics_dt: float, render_dt: float, cfg: dict[str, Any]) -> None:
        sim = cfg.get("sim", {})
        required = [
            "scene_usd",
            "robot_prim_path",
            "object_prim_path",
            "eef_prim_path",
            "camera_prim_paths",
            "articulation_joint_order",
            "contact_sensor_prim_paths",
        ]
        missing = [key for key in required if not sim.get(key)]
        if missing:
            raise ValueError(f"eval grid sim section is missing configured values: {missing}")
        if any(not value for value in sim["camera_prim_paths"].values()):
            raise ValueError("every sim.camera_prim_paths value must be configured")
        if len(sim["articulation_joint_order"]) != 13:
            raise ValueError("sim.articulation_joint_order must contain 13 joint names")
        if len(sim["contact_sensor_prim_paths"]) < 2:
            raise ValueError("at least two configured contact sensors are required for close success")
        scene_usd = Path(str(sim["scene_usd"])).expanduser()
        if not scene_usd.is_file():
            raise FileNotFoundError(f"scene USD does not exist: {scene_usd}")

        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation, SingleXFormPrim
        from isaacsim.core.utils.stage import add_reference_to_stage
        from isaacsim.core.utils.types import ArticulationAction
        from isaacsim.sensors.camera import Camera

        self._ArticulationAction = ArticulationAction
        self.world = World(physics_dt=physics_dt, rendering_dt=render_dt, stage_units_in_meters=1.0)
        add_reference_to_stage(str(scene_usd), "/World/RolloutScene")
        self.robot = SingleArticulation(prim_path=str(sim["robot_prim_path"]), name="rollout_robot")
        self.object = SingleXFormPrim(str(sim["object_prim_path"]), name="rollout_object")
        self.eef = SingleXFormPrim(str(sim["eef_prim_path"]), name="rollout_eef")
        self.world.scene.add(self.robot)
        self.world.scene.add(self.object)
        self.world.scene.add(self.eef)

        camera_cfg = {camera["name"]: camera for camera in cfg["obs"]["cameras"]}
        self.cameras = {}
        for name, prim_path in sim["camera_prim_paths"].items():
            if name not in camera_cfg:
                continue
            width, height = camera_cfg[name]["resolution"]
            camera = Camera(prim_path=str(prim_path), name=f"rollout_{name}", resolution=(width, height))
            self.world.scene.add(camera)
            self.cameras[name] = camera
        if set(self.cameras) != set(camera_cfg):
            missing_cameras = sorted(set(camera_cfg).difference(self.cameras))
            raise ValueError(f"sim camera_prim_paths is missing bundle cameras: {missing_cameras}")

        self.world.reset()
        self.joint_order = list(cfg["joint_order"])
        articulation_joint_order = list(sim["articulation_joint_order"])
        dof_names = list(self.robot.dof_names)
        missing_joints = [name for name in articulation_joint_order if name not in dof_names]
        if missing_joints:
            raise ValueError(f"articulation is missing bundle joints: {missing_joints}")
        self.joint_indices = np.asarray([dof_names.index(name) for name in articulation_joint_order], dtype=np.int64)
        self.physics_substeps = round(render_dt / physics_dt)
        if self.physics_substeps <= 0 or not np.isclose(
            self.physics_substeps * physics_dt, render_dt, rtol=0.0, atol=1e-9
        ):
            raise ValueError("render_dt must be an integer multiple of physics_dt")
        self.contact_sensors = self._build_contact_sensors(sim["contact_sensor_prim_paths"])
        self.initial_joint_positions = np.asarray(sim.get("initial_joint_positions", [0.0] * 13), dtype=np.float32)
        if self.initial_joint_positions.shape != (13,) or not np.isfinite(
            self.initial_joint_positions
        ).all():
            raise ValueError("sim.initial_joint_positions must contain 13 finite values")
        self.current_initial_joint_positions = self.initial_joint_positions.copy()
        self.frames: list[np.ndarray] = []

    def _build_contact_sensors(self, prim_paths: list[str]) -> list[Any]:
        if not prim_paths:
            return []
        from isaacsim.sensors.physics import ContactSensor

        sensors = []
        for index, prim_path in enumerate(prim_paths):
            sensor = ContactSensor(prim_path=str(prim_path), name=f"grasp_contact_{index}")
            self.world.scene.add(sensor)
            sensors.append(sensor)
        return sensors

    def reset(self, trial: dict[str, Any]) -> dict[str, Any]:
        self.world.reset()
        position = np.asarray(trial["object_position"], dtype=np.float32)
        orientation = np.asarray(trial.get("object_orientation_wxyz", [1.0, 0.0, 0.0, 0.0]), dtype=np.float32)
        self.object.set_world_pose(position=position, orientation=orientation)
        initial = np.asarray(trial.get("initial_joint_positions", self.initial_joint_positions), dtype=np.float32)
        if initial.shape != (13,) or not np.isfinite(initial).all():
            raise ValueError(
                f"trial {trial.get('id')} initial_joint_positions must contain 13 finite values"
            )
        self.current_initial_joint_positions = initial.copy()
        all_positions = self.robot.get_joint_positions().copy()
        all_positions[self.joint_indices] = initial
        self.robot.set_joint_positions(all_positions)
        self.robot.set_joint_velocities(np.zeros_like(all_positions))
        self.frames = []
        self.world.step(render=True)
        return self.observe()

    def _rgba(self, camera: Any) -> np.ndarray:
        rgba = np.asarray(camera.get_rgba())
        if rgba.dtype != np.uint8:
            rgba = np.clip(rgba * (255.0 if rgba.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
        return rgba[..., :3]

    def observe(self) -> dict[str, Any]:
        images = {name: self._rgba(camera) for name, camera in self.cameras.items()}
        if images:
            self.frames.append(next(iter(images.values())).copy())
        joint_positions = np.asarray(self.robot.get_joint_positions(), dtype=np.float32)
        return {"images": images, "proprio": joint_positions[self.joint_indices]}

    def step(self, action: np.ndarray) -> dict[str, Any]:
        target = np.asarray(action, dtype=np.float32)
        if target.shape != (13,):
            raise ValueError(f"action must have shape (13,), got {target.shape}")
        command = self._ArticulationAction(joint_positions=target, joint_indices=self.joint_indices)
        self.robot.apply_action(command)
        for substep in range(self.physics_substeps):
            self.world.step(render=substep == self.physics_substeps - 1)
        return self.observe()

    def recover(
        self,
        *,
        open_steps: int,
        retreat_steps: int,
        settle_steps: int,
        joint_positions: tuple[float, ...] | None = None,
    ) -> dict[str, Any]:
        """Open the hand, retreat to pregrasp, then settle without resetting the object."""
        target = np.asarray(
            self.current_initial_joint_positions if joint_positions is None else joint_positions,
            dtype=np.float32,
        )
        if target.shape != (13,) or not np.isfinite(target).all():
            raise ValueError("recovery joint positions must contain 13 finite values")
        all_positions = np.asarray(self.robot.get_joint_positions(), dtype=np.float32)
        start = all_positions[self.joint_indices].copy()
        commands = _recovery_joint_targets(
            start,
            target,
            open_steps=open_steps,
            retreat_steps=retreat_steps,
            settle_steps=settle_steps,
        )
        obs: dict[str, Any] | None = None
        for command in commands:
            obs = self.step(command)
        assert obs is not None
        return obs

    def state(self) -> dict[str, Any]:
        object_position, _ = self.object.get_world_pose()
        eef_position, _ = self.eef.get_world_pose()
        contacts = 0
        for sensor in self.contact_sensors:
            frame = sensor.get_current_frame()
            force = np.asarray(frame.get("force", 0.0), dtype=np.float32)
            contacts += int(float(np.linalg.norm(force)) > 0.0)
        return {
            "object_position": np.asarray(object_position),
            "eef_position": np.asarray(eef_position),
            "contact_count": contacts,
        }

    def save_video(self, path: str | Path, fps: int = 30) -> None:
        if not self.frames:
            return
        import imageio.v2 as imageio

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(path, self.frames, fps=fps, macro_block_size=1)

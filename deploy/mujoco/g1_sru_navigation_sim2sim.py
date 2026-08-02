"""Closed-loop MuJoCo sim2sim for the SRU navigator and parkour controller."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import deque
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "deploy"))

from instinct_nav_onboard.config import OnboardConfig  # noqa: E402
from instinct_nav_onboard.onnx_high_level_runtime import OnnxHighLevelRuntime  # noqa: E402
from instinct_nav_onboard.packets import SensorPacket  # noqa: E402

from g1_parkour_sim2sim import (  # noqa: E402
    JOINTS,
    URDF,
    body_obs,
    build_model,
    ids,
    load_params,
    terrain_config_from_args,
)

OFFICIAL_PARKOUR = ROOT / "data&model/checkpoints/parkour_onboard_preview_stair"
HIGH_LEVEL = ROOT / "artifacts/high_level/sru_scene_token"


def _yaw_from_matrix(rotation: np.ndarray) -> float:
    return math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))


def _goal_in_body(goal_w: np.ndarray, base_pos_w: np.ndarray, rotation_wb: np.ndarray) -> np.ndarray:
    delta_b = rotation_wb.T @ (goal_w - base_pos_w)
    distance = float(np.linalg.norm(delta_b))
    if distance < 1.0e-6:
        return np.zeros(4, dtype=np.float32)
    return np.asarray(
        [delta_b[0] / distance, delta_b[1] / distance, delta_b[2] / distance, math.log1p(distance)],
        dtype=np.float32,
    )


class RayDepthCamera:
    """Small pinhole depth camera matching the shared-lowcam training geometry."""

    def __init__(self, model: mujoco.MjModel, torso_body_id: int, cfg: OnboardConfig):
        self.model = model
        self.torso_body_id = torso_body_id
        self.cfg = cfg.camera
        self.geom_group = np.zeros(6, dtype=np.uint8)
        self.geom_group[0] = 1

        width, height = self.cfg.width, self.cfg.height
        tan_h = math.tan(math.radians(self.cfg.horizontal_fov_deg) * 0.5)
        tan_v = math.tan(math.radians(self.cfg.vertical_fov_deg) * 0.5)
        cols = ((np.arange(width, dtype=np.float64) + 0.5) / width * 2.0 - 1.0) * tan_h
        rows = ((np.arange(height, dtype=np.float64) + 0.5) / height * 2.0 - 1.0) * tan_v
        row_grid, col_grid = np.meshgrid(rows, cols, indexing="ij")
        rays = np.stack(
            [np.ones_like(row_grid), -col_grid, -row_grid],
            axis=-1,
        )
        rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
        self.local_rays = rays.reshape(-1, 3)
        self.forward_components = self.local_rays[:, 0].copy()
        self.offset = np.asarray(self.cfg.offset_pos_xyz, dtype=np.float64)
        self.camera_rotation = self._quat_to_matrix(np.asarray(self.cfg.offset_quat_wxyz, dtype=np.float64))

    @staticmethod
    def _quat_to_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
        matrix = np.empty(9, dtype=np.float64)
        mujoco.mju_quat2Mat(matrix, quat_wxyz)
        return matrix.reshape(3, 3)

    def capture(self, data: mujoco.MjData) -> np.ndarray:
        torso_rotation = data.xmat[self.torso_body_id].reshape(3, 3)
        origin = data.xpos[self.torso_body_id] + torso_rotation @ self.offset
        rotation = torso_rotation @ self.camera_rotation
        world_rays = self.local_rays @ rotation.T
        depth = np.full(len(world_rays), 8.0, dtype=np.float32)
        geom_id = np.asarray([-1], dtype=np.int32)
        for index, direction in enumerate(world_rays):
            distance = mujoco.mj_ray(
                self.model,
                data,
                origin,
                direction,
                self.geom_group,
                1,
                self.torso_body_id,
                geom_id,
            )
            if distance >= 0.0:
                depth[index] = float(distance * self.forward_components[index])
        return depth.reshape(self.cfg.height, self.cfg.width)


def _append_connector(scene, geom_type, start, end, width, rgba):
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        geom_type,
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        np.eye(3, dtype=np.float64).reshape(-1),
        np.asarray(rgba, dtype=np.float32),
    )
    mujoco.mjv_connector(
        geom,
        geom_type,
        width,
        np.asarray(start, dtype=np.float64),
        np.asarray(end, dtype=np.float64),
    )
    scene.ngeom += 1


def _append_sphere(scene, position, radius, rgba):
    if scene.ngeom >= scene.maxgeom:
        return
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom],
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.asarray([radius, radius, radius], dtype=np.float64),
        np.asarray(position, dtype=np.float64),
        np.eye(3, dtype=np.float64).reshape(-1),
        np.asarray(rgba, dtype=np.float32),
    )
    scene.ngeom += 1


def _update_debug_visuals(viewer, data, camera, torso_id, goal_w, frustum_length):
    torso_rotation = data.xmat[torso_id].reshape(3, 3)
    torso_position = data.xpos[torso_id]
    camera_origin = torso_position + torso_rotation @ camera.offset
    camera_rotation = torso_rotation @ camera.camera_rotation
    corner_indices = (
        0,
        camera.cfg.width - 1,
        camera.cfg.height * camera.cfg.width - 1,
        (camera.cfg.height - 1) * camera.cfg.width,
    )
    corner_rays = camera.local_rays[list(corner_indices)] @ camera_rotation.T
    corners = camera_origin[None, :] + float(frustum_length) * corner_rays

    with viewer.lock():
        scene = viewer.user_scn
        scene.ngeom = 0

        frustum_color = (1.0, 0.78, 0.05, 0.9)
        for corner in corners:
            _append_connector(
                scene, mujoco.mjtGeom.mjGEOM_LINE, camera_origin, corner, 2.0, frustum_color
            )
        for index in range(4):
            _append_connector(
                scene,
                mujoco.mjtGeom.mjGEOM_LINE,
                corners[index],
                corners[(index + 1) % 4],
                2.0,
                frustum_color,
            )

        frame_origin = torso_position + torso_rotation @ np.asarray([0.0, 0.0, 0.55])
        axis_colors = (
            (0.95, 0.10, 0.10, 1.0),
            (0.10, 0.85, 0.15, 1.0),
            (0.10, 0.35, 1.0, 1.0),
        )
        for axis, color in enumerate(axis_colors):
            endpoint = frame_origin + 0.30 * torso_rotation[:, axis]
            _append_connector(
                scene, mujoco.mjtGeom.mjGEOM_ARROW, frame_origin, endpoint, 0.015, color
            )

        goal_position = np.asarray(goal_w, dtype=np.float64).copy()
        goal_position[2] = max(goal_position[2], 0.10)
        _append_sphere(scene, goal_position, 0.10, (0.1, 1.0, 0.2, 0.9))
        _append_connector(
            scene,
            mujoco.mjtGeom.mjGEOM_LINE,
            (goal_position[0], goal_position[1], 0.0),
            (goal_position[0], goal_position[1], goal_position[2] + 0.50),
            4.0,
            (0.1, 1.0, 0.2, 0.8),
        )
def _prepare_raycast_groups(model: mujoco.MjModel) -> None:
    """Keep robot geometry out of camera rays while preserving contact behavior."""
    model.geom_group[:] = 1
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if name == "floor" or name.startswith("terrain_"):
            model.geom_group[geom_id] = 0


def _low_level_depth_history(frames: deque[np.ndarray], cfg: OnboardConfig) -> np.ndarray:
    indices = np.linspace(-43, -1, cfg.camera.low_level_history).astype(int)
    selected = np.stack([frames[index] for index in indices], axis=0)
    top, bottom, left, right = cfg.camera.low_level_crop_region
    height, width = selected.shape[-2:]
    selected = selected[:, top : height - bottom, left : width - right]
    depth_min, depth_max = cfg.camera.low_level_depth_range_m
    return np.clip((selected - depth_min) / (depth_max - depth_min), 0.0, 1.0)[None].astype(np.float32)


def run(args: argparse.Namespace) -> dict:
    cfg = OnboardConfig(
        command_min=tuple(-abs(v) for v in args.high_command_limit),
        command_max=tuple(abs(v) for v in args.high_command_limit),
    )
    params = load_params(args.low_policy_dir)
    model = build_model(args.urdf, args.dt, terrain_config_from_args(args))
    _prepare_raycast_groups(model)
    data = mujoco.MjData(model)
    qpos_ids, dof_ids = ids(model)
    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    model.dof_armature[dof_ids] = params["armature"]

    low_actor = ort.InferenceSession(
        str(args.low_policy_dir / "exported/actor.onnx"), providers=["CPUExecutionProvider"]
    )
    low_encoder = ort.InferenceSession(
        str(args.low_policy_dir / "exported/0-depth_encoder.onnx"), providers=["CPUExecutionProvider"]
    )
    high_runtime = OnnxHighLevelRuntime(
        cfg,
        policy_onnx_path=args.high_policy,
        depth_encoder_onnx_path=args.high_depth_encoder,
        providers=["CPUExecutionProvider"],
        manifest_path=args.manifest,
    )

    data.qpos[:7] = [0.0, 0.0, args.height, 1.0, 0.0, 0.0, 0.0]
    data.qpos[qpos_ids] = params["default"]
    mujoco.mj_forward(model, data)

    camera = RayDepthCamera(model, torso_id, cfg)
    depth = camera.capture(data)
    depth_frames: deque[np.ndarray] = deque([depth.copy() for _ in range(60)], maxlen=60)
    last_low_action = np.zeros(29, dtype=np.float32)
    low_terms = {
        "ang": np.zeros(3, dtype=np.float32),
        "grav": np.asarray([0.0, 0.0, -1.0], dtype=np.float32),
        "cmd": np.zeros(3, dtype=np.float32),
        "q": np.zeros(29, dtype=np.float32),
        "dq": np.zeros(29, dtype=np.float32),
        "act": last_low_action,
    }
    low_history = {
        key: deque([value.copy() for _ in range(8)], maxlen=8) for key, value in low_terms.items()
    }
    target_joint_pos = params["default"].copy()
    velocity_command = np.zeros(3, dtype=np.float32)
    goal_w = np.asarray(args.goal, dtype=np.float64)
    next_camera = next_low = next_high = 0.0
    fell = reached = False
    command_records: list[np.ndarray] = []
    path_records: list[np.ndarray] = []

    viewer = None
    if args.viewer:
        from mujoco import viewer as mj_viewer

        viewer = mj_viewer.launch_passive(model, data)

    while data.time < args.duration:
        if data.time + 1.0e-9 >= next_camera:
            depth = camera.capture(data)
            depth_frames.append(depth.copy())
            next_camera += 1.0 / args.camera_hz

        if data.time + 1.0e-9 >= next_high:
            velocity = np.zeros(6, dtype=np.float64)
            mujoco.mj_objectVelocity(
                model, data, mujoco.mjtObj.mjOBJ_BODY, torso_id, velocity, 1
            )
            torso_rotation = data.xmat[torso_id].reshape(3, 3)
            goal_b = _goal_in_body(goal_w, data.qpos[:3], torso_rotation)
            packet = SensorPacket(
                timestamp_s=float(data.time),
                base_lin_vel_b=velocity[3:].astype(np.float32),
                base_ang_vel_b=velocity[:3].astype(np.float32),
                projected_gravity_b=(
                    torso_rotation.T @ np.asarray([0.0, 0.0, -1.0])
                ).astype(np.float32),
                joint_pos=data.qpos[qpos_ids].astype(np.float32),
                joint_vel=data.qvel[dof_ids].astype(np.float32),
                depth_m=depth.astype(np.float32),
                goal_b=goal_b,
                reset=next_high == 0.0,
                base_pos_w=data.qpos[:3].astype(np.float32),
                base_yaw_w=_yaw_from_matrix(torso_rotation),
            )
            velocity_command = high_runtime.step(packet).as_array()
            command_records.append(velocity_command.copy())
            next_high += 1.0 / cfg.high_level_hz

        if data.time + 1.0e-9 >= next_low:
            angular_velocity, gravity = body_obs(model, data, torso_id)
            low_terms = {
                "ang": angular_velocity * 0.25,
                "grav": gravity,
                "cmd": velocity_command,
                "q": data.qpos[qpos_ids].astype(np.float32) - params["default"],
                "dq": data.qvel[dof_ids].astype(np.float32) * 0.05,
                "act": last_low_action,
            }
            for key, value in low_terms.items():
                low_history[key].append(value.copy())
            low_depth = _low_level_depth_history(depth_frames, cfg)
            latent = low_encoder.run(
                None, {low_encoder.get_inputs()[0].name: low_depth}
            )[0].reshape(-1)
            low_obs = np.concatenate(
                [np.stack(low_history[key]).reshape(-1) for key in low_history] + [latent]
            ).reshape(1, -1).astype(np.float32)
            last_low_action = low_actor.run(
                None, {low_actor.get_inputs()[0].name: low_obs}
            )[0].reshape(-1).astype(np.float32)
            target_joint_pos = params["default"] + params["scale"] * last_low_action
            next_low += 1.0 / cfg.low_level_hz

        torque = np.clip(
            params["kp"] * (target_joint_pos - data.qpos[qpos_ids])
            - params["kd"] * data.qvel[dof_ids],
            -params["effort"],
            params["effort"],
        )
        data.qfrc_applied[dof_ids] = torque
        mujoco.mj_step(model, data)
        path_records.append(data.qpos[:3].copy())

        torso_rotation = data.xmat[torso_id].reshape(3, 3)
        gravity_b = torso_rotation.T @ np.asarray([0.0, 0.0, -1.0])
        tilt_deg = math.degrees(math.acos(float(np.clip(-gravity_b[2], -1.0, 1.0))))
        distance_xy = float(np.linalg.norm(goal_w[:2] - data.qpos[:2]))
        if distance_xy <= args.goal_threshold:
            reached = True
            break
        if data.qpos[2] < args.fall_height or tilt_deg > args.fall_tilt:
            fell = True
            break

        if viewer is not None:
            if not args.no_debug_visuals:
                _update_debug_visuals(
                    viewer, data, camera, torso_id, goal_w, args.camera_frustum_length
                )
            viewer.sync()
            time.sleep(args.dt)

    if viewer is not None:
        viewer.close()

    commands = np.stack(command_records) if command_records else np.zeros((0, 3))
    path = np.stack(path_records) if path_records else np.zeros((0, 3))
    result = {
        "duration_s": float(data.time),
        "terrain": args.terrain,
        "goal_w": goal_w.tolist(),
        "reached": reached,
        "fell": fell,
        "final_position": data.qpos[:3].tolist(),
        "final_distance_xy": float(np.linalg.norm(goal_w[:2] - data.qpos[:2])),
        "command_samples": int(len(commands)),
        "command_mean": commands.mean(axis=0).tolist() if len(commands) else [0.0] * 3,
        "command_abs_max": np.abs(commands).max(axis=0).tolist() if len(commands) else [0.0] * 3,
        "path_xy_length": float(np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1).sum())
        if len(path) > 1
        else 0.0,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2) + "\n")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--low_policy_dir", type=Path, default=OFFICIAL_PARKOUR)
    parser.add_argument("--high_policy", type=Path, default=HIGH_LEVEL / "sru_high_level_scene_token.onnx")
    parser.add_argument(
        "--high_depth_encoder",
        type=Path,
        default=HIGH_LEVEL / "sru_high_level_depth_encoder.onnx",
    )
    parser.add_argument("--manifest", type=Path, default=HIGH_LEVEL / "deploy_manifest.json")
    parser.add_argument("--urdf", type=Path, default=URDF)
    parser.add_argument("--goal", type=float, nargs=3, default=(2.0, 0.0, 0.0))
    parser.add_argument("--goal_threshold", type=float, default=0.5)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--camera_hz", type=float, default=60.0)
    parser.add_argument("--height", type=float, default=0.9)
    parser.add_argument("--high_command_limit", type=float, nargs=3, default=(0.3, 0.2, 0.4))
    parser.add_argument("--fall_height", type=float, default=0.45)
    parser.add_argument("--fall_tilt", type=float, default=60.0)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--no_debug_visuals", action="store_true")
    parser.add_argument("--camera_frustum_length", type=float, default=3.0)
    parser.add_argument("--terrain", choices=("flat", "stairs", "ramp"), default="flat")
    parser.add_argument("--terrain_start_x", type=float, default=1.5)
    parser.add_argument("--terrain_width", type=float, default=2.0)
    parser.add_argument("--stair_height", type=float, default=0.12)
    parser.add_argument("--stair_depth", type=float, default=0.50)
    parser.add_argument("--stair_steps", type=int, default=4)
    parser.add_argument("--ramp_angle_deg", type=float, default=12.0)
    parser.add_argument("--ramp_length", type=float, default=3.0)
    parser.add_argument(
        "--json_out",
        type=Path,
        default=ROOT / "artifacts/mujoco/sru_navigation.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2))

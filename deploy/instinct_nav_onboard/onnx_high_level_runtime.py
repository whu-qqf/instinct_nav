from __future__ import annotations

import hashlib
import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort

from .config import OnboardConfig
from .packets import SensorPacket, VelocityCommand


@dataclass(frozen=True)
class OnnxPolicyContract:
    obs_dim: int
    num_actions: int
    rnn_num_layers: int
    rnn_hidden_size: int
    scene_token_num_tokens: int | None = None
    scene_token_dim: int | None = None
    scene_token_features_total: int | None = None
    encoded_depth_dim: int | None = None
    proprio_dim: int | None = None

@dataclass(frozen=True)
class OnnxDebugDumpConfig:
    dump_dir: Path
    every_steps: int = 1
    max_steps: int = 1000
    save_arrays: bool = True


class OnnxHighLevelRuntime:
    """Torch-free high-level SRU runtime using ONNXRuntime."""

    def __init__(
        self,
        cfg: OnboardConfig,
        policy_onnx_path: str | Path,
        depth_encoder_onnx_path: str | Path,
        command_sink=None,
        providers: list[str] | None = None,
        debug_dump_config: OnnxDebugDumpConfig | None = None,
        manifest_path: str | Path | None = None,
    ):
        self.cfg = cfg
        self.command_sink = command_sink
        self.providers = providers or ort.get_available_providers()
        self.policy_onnx_path = Path(policy_onnx_path).expanduser()
        self.depth_encoder_onnx_path = Path(depth_encoder_onnx_path).expanduser()
        self.policy_session = ort.InferenceSession(str(self.policy_onnx_path), providers=self.providers)
        self.depth_session = ort.InferenceSession(str(self.depth_encoder_onnx_path), providers=self.providers)
        self.contract = self._load_and_validate_contract(manifest_path)
        self.h = np.zeros((self.contract.rnn_num_layers, 1, self.contract.rnn_hidden_size), dtype=np.float32)
        self.c = np.zeros((self.contract.rnn_num_layers, 1, self.contract.rnn_hidden_size), dtype=np.float32)
        self.last_action = np.zeros(3, dtype=np.float32)
        self.visibility = np.zeros((1, 1, 64, 64), dtype=np.float32)
        self.last_base_pos_w = np.zeros(3, dtype=np.float32)
        self.last_base_yaw_w = 0.0
        self.visibility_pose_valid = False
        self.debug_cfg = debug_dump_config
        self.step_count = 0
        self.saved_count = 0
        if self.debug_cfg is not None:
            self.debug_cfg.dump_dir.mkdir(parents=True, exist_ok=True)
            self.index_path = self.debug_cfg.dump_dir / "index.jsonl"

    def reset(self) -> None:
        self.h.fill(0.0)
        self.c.fill(0.0)
        self.last_action.fill(0.0)
        self.visibility.fill(0.0)
        self.last_base_pos_w.fill(0.0)
        self.last_base_yaw_w = 0.0
        self.visibility_pose_valid = False

    def step(self, packet: SensorPacket) -> VelocityCommand:
        if packet.reset:
            self.reset()
        depth = np.asarray(packet.depth_m, dtype=np.float32).reshape(1, 1, 36, 64)
        encoded_depth, filtered_depth = self.depth_session.run(None, {"depth_m": depth})
        scene_tokens = self.build_scene_tokens(
            depth.reshape(36, 64), base_pos_w=packet.base_pos_w, base_yaw_w=packet.base_yaw_w
        )
        obs = self.build_actor_obs(packet, encoded_depth.reshape(1, -1), scene_tokens.reshape(1, -1))
        actions, self.h, self.c = self.policy_session.run(None, {"obs": obs, "h_in": self.h, "c_in": self.c})
        action = np.asarray(actions, dtype=np.float32).reshape(-1)[:3]
        self.last_action = action.copy()
        command = self._action_to_command(action, packet.timestamp_s)
        self._maybe_dump(packet, encoded_depth, filtered_depth, scene_tokens, obs, action, command)
        return command


    def _load_and_validate_contract(self, manifest_path: str | Path | None) -> OnnxPolicyContract:
        manifest = _read_manifest(manifest_path) if manifest_path is not None else _read_manifest(self.policy_onnx_path.with_name("deploy_manifest.json"))
        inferred = self._infer_contract_from_onnx()
        if manifest is None:
            contract = inferred
        else:
            expected_name = manifest.get("policy_onnx")
            if expected_name is not None and Path(expected_name).name != self.policy_onnx_path.name:
                raise ValueError(f"policy ONNX filename mismatch: manifest={expected_name}, runtime={self.policy_onnx_path.name}")
            expected_sha256 = manifest.get("policy_onnx_sha256")
            if expected_sha256 is not None:
                actual_sha256 = _sha256(self.policy_onnx_path)
                if actual_sha256 != expected_sha256:
                    raise ValueError(
                        f"policy ONNX sha256 mismatch: manifest={expected_sha256}, runtime={actual_sha256}"
                    )
            contract = OnnxPolicyContract(
                obs_dim=int(manifest["obs_dim"]),
                num_actions=int(manifest["num_actions"]),
                rnn_num_layers=int(manifest["rnn_num_layers"]),
                rnn_hidden_size=int(manifest["rnn_hidden_size"]),
                scene_token_num_tokens=int(manifest["scene_token_num_tokens"]) if manifest.get("scene_token_num_tokens") is not None else None,
                scene_token_dim=int(manifest["scene_token_dim"]) if manifest.get("scene_token_dim") is not None else None,
                scene_token_features_total=int(manifest["scene_token_features_total"]) if manifest.get("scene_token_features_total") is not None else None,
                encoded_depth_dim=int(manifest["encoded_depth_dim"]) if manifest.get("encoded_depth_dim") is not None else None,
                proprio_dim=int(manifest["proprio_dim"]) if manifest.get("proprio_dim") is not None else None,
            )
            mismatches = []
            for field in ("obs_dim", "num_actions", "rnn_num_layers", "rnn_hidden_size"):
                if getattr(contract, field) != getattr(inferred, field):
                    mismatches.append(f"{field}: manifest={getattr(contract, field)} onnx={getattr(inferred, field)}")
            expected_metadata = {
                "scene_token_num_tokens": self.cfg.layout.scene_token_num_tokens,
                "scene_token_dim": self.cfg.layout.scene_token_feature_dim,
                "scene_token_features_total": self.cfg.layout.scene_token_dim,
                "encoded_depth_dim": self.cfg.layout.encoded_depth_dim,
                "proprio_dim": self.cfg.layout.proprio_dim,
            }
            for field, expected in expected_metadata.items():
                actual = getattr(contract, field)
                if actual is not None and int(actual) != int(expected):
                    mismatches.append(f"{field}: manifest={actual} runtime={expected}")
            if mismatches:
                raise ValueError("ONNX contract mismatch: " + "; ".join(mismatches))
        if contract.obs_dim != self.cfg.layout.actor_obs_dim:
            raise ValueError(f"runtime obs layout mismatch: contract={contract.obs_dim}, cfg={self.cfg.layout.actor_obs_dim}")
        if contract.num_actions != 3:
            raise ValueError(f"expected 3 high-level actions, got {contract.num_actions}")
        return contract

    def _infer_contract_from_onnx(self) -> OnnxPolicyContract:
        inputs = {item.name: item.shape for item in self.policy_session.get_inputs()}
        outputs = {item.name: item.shape for item in self.policy_session.get_outputs()}
        obs_shape = inputs.get("obs")
        h_shape = inputs.get("h_in")
        actions_shape = outputs.get("actions")
        if obs_shape is None or h_shape is None or actions_shape is None:
            raise ValueError("policy ONNX must expose inputs obs/h_in and output actions")
        obs_dim = _static_dim(obs_shape, 1, "obs_dim")
        rnn_num_layers = _static_dim(h_shape, 0, "rnn_num_layers")
        rnn_hidden_size = _static_dim(h_shape, 2, "rnn_hidden_size")
        num_actions = _static_dim(actions_shape, 1, "num_actions")
        return OnnxPolicyContract(
            obs_dim=obs_dim,
            num_actions=num_actions,
            rnn_num_layers=rnn_num_layers,
            rnn_hidden_size=rnn_hidden_size,
            scene_token_num_tokens=self.cfg.layout.scene_token_num_tokens,
            scene_token_dim=self.cfg.layout.scene_token_feature_dim,
            scene_token_features_total=self.cfg.layout.scene_token_dim,
            encoded_depth_dim=self.cfg.layout.encoded_depth_dim,
            proprio_dim=self.cfg.layout.proprio_dim,
        )

    def build_actor_obs(self, packet: SensorPacket, encoded_depth: np.ndarray, scene_tokens: np.ndarray) -> np.ndarray:
        proprio = np.concatenate(
            [
                np.asarray(packet.base_lin_vel_b, dtype=np.float32).reshape(3),
                np.asarray(packet.base_ang_vel_b, dtype=np.float32).reshape(3),
                np.asarray(packet.projected_gravity_b, dtype=np.float32).reshape(3),
                self.last_action.reshape(3),
                np.asarray(packet.goal_b, dtype=np.float32).reshape(4),
            ],
            axis=0,
        ).reshape(1, -1)
        obs = np.concatenate([proprio, encoded_depth.reshape(1, -1), scene_tokens.reshape(1, -1)], axis=1).astype(np.float32)
        if obs.shape[1] != self.contract.obs_dim:
            raise ValueError(f"actor obs dim mismatch: got {obs.shape[1]}, expected {self.contract.obs_dim}")
        return obs

    def build_scene_tokens(
        self,
        depth_m: np.ndarray,
        *,
        base_pos_w: np.ndarray | None = None,
        base_yaw_w: float | None = None,
    ) -> np.ndarray:
        token_grid = 8
        grid_size = 64
        size_m = 6.4
        half_size = size_m / 2.0
        depth_min_m = 0.5
        depth_max_m = 8.0
        depth = np.asarray(depth_m, dtype=np.float32).reshape(36, 64)
        valid = np.isfinite(depth) & (depth > depth_min_m) & (depth < depth_max_m)
        if base_pos_w is not None and base_yaw_w is not None:
            self._warp_visibility_to_current_pose(
                np.asarray(base_pos_w, dtype=np.float32).reshape(3),
                float(base_yaw_w),
                size_m=size_m,
                grid_size=grid_size,
            )
        current_visibility = self._depth_to_current_visibility(depth, valid, size_m=size_m, grid_size=grid_size)
        if self.cfg.visibility_decay < 1.0:
            self.visibility *= self.cfg.visibility_decay
            self.visibility += self.cfg.visibility_score_alpha_up * current_visibility * (1.0 - self.visibility)
            np.clip(self.visibility, 0.0, 1.0, out=self.visibility)
        else:
            self.visibility = np.maximum(self.visibility, current_visibility)

        vis = self.visibility.reshape(1, 1, 8, 8, 8, 8).mean(axis=(3, 5)).reshape(1, 64)
        rows, cols = np.meshgrid(np.arange(token_grid, dtype=np.float32), np.arange(token_grid, dtype=np.float32), indexing="ij")
        patch_m = size_m / float(token_grid)
        x_center_m = half_size - (rows + 0.5) * patch_m
        y_center_m = half_size - (cols + 0.5) * patch_m
        x_center = (x_center_m / half_size).reshape(1, -1)
        y_center = (y_center_m / half_size).reshape(1, -1)

        depth_norm = np.where(valid, np.clip((depth - depth_min_m) / max(depth_max_m - depth_min_m, 1.0e-6), 0.0, 1.0), 1.0).astype(np.float32)
        depth_mean = _adaptive_avg_pool2d(depth_norm, token_grid, token_grid).reshape(1, -1)
        grad_x = np.pad(np.abs(depth_norm[:, 1:] - depth_norm[:, :-1]), ((0, 0), (0, 1)))
        grad_y = np.pad(np.abs(depth_norm[1:, :] - depth_norm[:-1, :]), ((0, 1), (0, 0)))
        depth_edge = np.clip(_adaptive_avg_pool2d(grad_x + grad_y, token_grid, token_grid), 0.0, 1.0).reshape(1, -1)

        frontier_source = self.visibility[0, 0] if self.cfg.visibility_decay < 1.0 else (self.visibility[0, 0] > 0.5).astype(np.float32)
        frontier_x = np.pad(np.abs(frontier_source[:, 1:] - frontier_source[:, :-1]), ((0, 0), (0, 1)))
        frontier_y = np.pad(np.abs(frontier_source[1:, :] - frontier_source[:-1, :]), ((0, 1), (0, 0)))
        frontier_grid = np.clip(frontier_x + frontier_y, 0.0, 1.0).reshape(8, 8, 8, 8).mean(axis=(1, 3))
        frontier = frontier_grid.reshape(1, -1)

        tokens = np.stack([x_center, y_center, vis, depth_mean, depth_edge, frontier], axis=-1)
        return tokens.reshape(1, -1).astype(np.float32)

    def _warp_visibility_to_current_pose(
        self,
        base_pos_w: np.ndarray,
        base_yaw_w: float,
        *,
        size_m: float,
        grid_size: int,
    ) -> None:
        if not np.isfinite(base_pos_w).all() or not np.isfinite(base_yaw_w):
            return
        if not self.visibility_pose_valid:
            self.last_base_pos_w = base_pos_w.copy()
            self.last_base_yaw_w = float(base_yaw_w)
            self.visibility_pose_valid = True
            return

        half_size = size_m * 0.5
        cell_size = size_m / float(grid_size)
        rows, cols = np.meshgrid(
            np.arange(grid_size, dtype=np.float32),
            np.arange(grid_size, dtype=np.float32),
            indexing="ij",
        )
        x_cur = half_size - (rows + 0.5) * cell_size
        y_cur = half_size - (cols + 0.5) * cell_size
        cos_cur = np.cos(base_yaw_w)
        sin_cur = np.sin(base_yaw_w)
        world_x = base_pos_w[0] + cos_cur * x_cur - sin_cur * y_cur
        world_y = base_pos_w[1] + sin_cur * x_cur + cos_cur * y_cur

        rel_x = world_x - self.last_base_pos_w[0]
        rel_y = world_y - self.last_base_pos_w[1]
        cos_prev = np.cos(self.last_base_yaw_w)
        sin_prev = np.sin(self.last_base_yaw_w)
        x_prev = cos_prev * rel_x + sin_prev * rel_y
        y_prev = -sin_prev * rel_x + cos_prev * rel_y
        row_prev = (half_size - x_prev) / cell_size - 0.5
        col_prev = (half_size - y_prev) / cell_size - 0.5
        warped = _bilinear_sample_grid(self.visibility[0, 0], row_prev, col_prev)
        if self.cfg.visibility_decay < 1.0:
            self.visibility[0, 0] = warped
        else:
            self.visibility[0, 0] = (warped > 0.5).astype(np.float32)
        self.last_base_pos_w = base_pos_w.copy()
        self.last_base_yaw_w = float(base_yaw_w)

    def _depth_to_current_visibility(self, depth: np.ndarray, valid: np.ndarray, size_m: float, grid_size: int) -> np.ndarray:
        height, width = depth.shape
        pixel_stride = 2
        sampled_rows = np.arange(0, height, pixel_stride, dtype=np.float32)
        sampled_cols = np.arange(0, width, pixel_stride, dtype=np.float32)
        row_grid, col_grid = np.meshgrid(sampled_rows, sampled_cols, indexing="ij")
        depth = depth[::pixel_stride, ::pixel_stride]
        valid = valid[::pixel_stride, ::pixel_stride]
        h_fov = np.deg2rad(self.cfg.camera.horizontal_fov_deg)
        v_fov = np.deg2rad(self.cfg.camera.vertical_fov_deg)
        fx = (width * 0.5) / np.tan(h_fov * 0.5)
        fy = (height * 0.5) / np.tan(v_fov * 0.5)
        cx = (width - 1) * 0.5
        cy = (height - 1) * 0.5
        points_camera = np.stack(
            (
                depth,
                -((col_grid - cx) / fx) * depth,
                -((row_grid - cy) / fy) * depth,
            ),
            axis=-1,
        )
        camera_rotation_b = _quat_wxyz_to_matrix(
            np.asarray(self.cfg.camera.offset_quat_wxyz, dtype=np.float32)
        )
        points_body = points_camera @ camera_rotation_b.T
        points_body += np.asarray(self.cfg.camera.offset_pos_xyz, dtype=np.float32).reshape(1, 1, 3)
        x_forward = points_body[..., 0]
        y_left = points_body[..., 1]
        z_up = points_body[..., 2]
        valid_points = valid & np.isfinite(x_forward) & np.isfinite(y_left) & np.isfinite(z_up)
        half_size = size_m / 2.0
        cell_size = size_m / float(grid_size)
        row = np.floor((half_size - x_forward) / cell_size).astype(np.int64)
        col = np.floor((half_size - y_left) / cell_size).astype(np.int64)
        in_grid = valid_points & (row >= 0) & (row < grid_size) & (col >= 0) & (col < grid_size)
        visibility = np.zeros((1, 1, grid_size, grid_size), dtype=np.float32)
        visibility[0, 0, row[in_grid], col[in_grid]] = 1.0
        return visibility

    def _action_to_command(self, action: np.ndarray, timestamp_s: float) -> VelocityCommand:
        mins = np.asarray(self.cfg.command_min, dtype=np.float32)
        maxs = np.asarray(self.cfg.command_max, dtype=np.float32)
        # Training uses a Gaussian policy and applies tanh in the navigation
        # action term before writing the low-level velocity command. Keep the
        # raw action in last_action, but mirror that command transform here.
        command = np.clip(np.tanh(action.astype(np.float32)), mins, maxs)
        return VelocityCommand(float(command[0]), float(command[1]), float(command[2]), timestamp_s=timestamp_s)

    def _maybe_dump(self, packet, encoded_depth, filtered_depth, scene_tokens, actor_obs, raw_action, command) -> None:
        if self.debug_cfg is None:
            return
        self.step_count += 1
        if self.debug_cfg.every_steps <= 0 or self.step_count % self.debug_cfg.every_steps != 0:
            return
        if self.saved_count >= self.debug_cfg.max_steps:
            return
        self.saved_count += 1
        tag = f"step_{self.step_count:08d}"
        tanh_action = np.tanh(np.asarray(raw_action, dtype=np.float32).reshape(-1))
        command_array = command.as_array().astype(np.float32)
        command_clip_mask = np.abs(tanh_action - command_array) > 1.0e-6
        policy_saturation_mask = np.abs(tanh_action) >= 0.95
        record = {
            "step": self.step_count,
            "saved_index": self.saved_count,
            "timestamp_s": float(packet.timestamp_s),
            "reset": bool(packet.reset),
            "file": f"{tag}.npz" if self.debug_cfg.save_arrays else None,
            "goal_b": np.asarray(packet.goal_b, dtype=np.float32).reshape(-1).astype(float).tolist(),
            "raw_action": np.asarray(raw_action, dtype=np.float32).reshape(-1).astype(float).tolist(),
            "tanh_action": tanh_action.astype(float).tolist(),
            "command": command_array.astype(float).tolist(),
            "policy_saturation_mask": policy_saturation_mask.tolist(),
            "command_clip_mask": command_clip_mask.tolist(),
            "depth_stats": _stats(packet.depth_m),
            "encoded_depth_stats": _stats(encoded_depth),
            "scene_token_stats": _stats(scene_tokens),
            "actor_obs_stats": _stats(actor_obs),
        }
        if self.debug_cfg.save_arrays:
            np.savez_compressed(
                self.debug_cfg.dump_dir / f"{tag}.npz",
                timestamp_s=np.asarray(packet.timestamp_s, dtype=np.float64),
                base_pos_w=np.asarray(packet.base_pos_w, dtype=np.float32),
                base_yaw_w=np.asarray(packet.base_yaw_w, dtype=np.float32),
                base_lin_vel_b=np.asarray(packet.base_lin_vel_b, dtype=np.float32),
                base_ang_vel_b=np.asarray(packet.base_ang_vel_b, dtype=np.float32),
                projected_gravity_b=np.asarray(packet.projected_gravity_b, dtype=np.float32),
                joint_pos=np.asarray(packet.joint_pos, dtype=np.float32),
                joint_vel=np.asarray(packet.joint_vel, dtype=np.float32),
                depth_m=np.asarray(packet.depth_m, dtype=np.float32),
                filtered_depth=np.asarray(filtered_depth, dtype=np.float32),
                goal_b=np.asarray(packet.goal_b, dtype=np.float32),
                encoded_depth=np.asarray(encoded_depth, dtype=np.float32).reshape(-1),
                scene_tokens=np.asarray(scene_tokens, dtype=np.float32).reshape(-1),
                actor_obs=np.asarray(actor_obs, dtype=np.float32).reshape(-1),
                raw_action=np.asarray(raw_action, dtype=np.float32),
                tanh_action=tanh_action,
                command=command_array,
                policy_saturation_mask=policy_saturation_mask,
                command_clip_mask=command_clip_mask,
            )
        with self.index_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _adaptive_avg_pool2d(image: np.ndarray, output_height: int, output_width: int) -> np.ndarray:
    """Match torch adaptive_avg_pool2d for one 2-D image."""
    image = np.asarray(image, dtype=np.float32)
    input_height, input_width = image.shape
    pooled = np.empty((output_height, output_width), dtype=np.float32)
    for out_row in range(output_height):
        row_start = (out_row * input_height) // output_height
        row_end = ((out_row + 1) * input_height + output_height - 1) // output_height
        for out_col in range(output_width):
            col_start = (out_col * input_width) // output_width
            col_end = ((out_col + 1) * input_width + output_width - 1) // output_width
            pooled[out_row, out_col] = image[row_start:row_end, col_start:col_end].mean()
    return pooled


def _bilinear_sample_grid(image: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Sample a 2-D map like grid_sample with zero padding and align_corners."""
    height, width = image.shape
    row0 = np.floor(rows).astype(np.int64)
    col0 = np.floor(cols).astype(np.int64)
    row1 = row0 + 1
    col1 = col0 + 1
    row_weight = rows - row0
    col_weight = cols - col0

    def gather(sample_rows: np.ndarray, sample_cols: np.ndarray) -> np.ndarray:
        valid = (sample_rows >= 0) & (sample_rows < height) & (sample_cols >= 0) & (sample_cols < width)
        values = np.zeros(rows.shape, dtype=np.float32)
        values[valid] = image[sample_rows[valid], sample_cols[valid]]
        return values

    top_left = gather(row0, col0)
    top_right = gather(row0, col1)
    bottom_left = gather(row1, col0)
    bottom_right = gather(row1, col1)
    return (
        top_left * (1.0 - row_weight) * (1.0 - col_weight)
        + top_right * (1.0 - row_weight) * col_weight
        + bottom_left * row_weight * (1.0 - col_weight)
        + bottom_right * row_weight * col_weight
    ).astype(np.float32)


def _quat_wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion, dtype=np.float64).reshape(4)
    norm = np.linalg.norm((w, x, y, z))
    if not np.isfinite(norm) or norm < 1.0e-8:
        raise ValueError("camera offset quaternion is invalid")
    w, x, y, z = (w / norm, x / norm, y / norm, z / norm)
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_manifest(path: str | Path | None) -> dict | None:
    if path is None:
        return None
    manifest_path = Path(path).expanduser()
    if not manifest_path.exists():
        return None
    with manifest_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _static_dim(shape: list, index: int, name: str) -> int:
    try:
        value = shape[index]
    except IndexError as exc:
        raise ValueError(f"missing ONNX dimension {name} at index {index}: shape={shape}") from exc
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"ONNX dimension {name} must be static positive int, got {value!r} from shape={shape}")
    return int(value)


def _stats(value: np.ndarray) -> dict[str, float | int]:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    finite = np.isfinite(arr)
    if not finite.any():
        return {"numel": int(arr.size), "finite_ratio": 0.0, "min": float("nan"), "max": float("nan"), "mean": float("nan"), "std": float("nan")}
    finite_arr = arr[finite]
    return {
        "numel": int(arr.size),
        "finite_ratio": float(finite.mean()),
        "min": float(finite_arr.min()),
        "max": float(finite_arr.max()),
        "mean": float(finite_arr.mean()),
        "std": float(finite_arr.std()),
    }

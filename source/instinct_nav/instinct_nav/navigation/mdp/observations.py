# Copyright (c) 2022-2025, Fan Yang and Per Frivik, ETH Zurich.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Observation functions for navigation tasks.

These functions can be passed to :class:`isaaclab.managers.ObservationTermCfg`
to specify observations for the policy.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional, cast

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

import instinct_nav.navigation.mdp as mdp

from .depth_utils.camera_config import CameraConfig, DEFAULT_CAMERA_CONFIG, get_camera_config
from .depth_utils.depth_noise_encoder import DepthNoiseEncoder
from .heightscan_utils.heightscan_encoder import HeightScanFeatEncoder

if TYPE_CHECKING:
    from instinct_nav.navigation.mdp import PerceptiveNavigationSE2Action
    from instinct_nav.navigation.mdp.navigation.goal_commands import RobotNavigationGoalCommand

# Visualization state
_DEPTH_VIZ_WINDOWS: dict[str, dict] = {}

# Global depth noise generator - will be initialized with camera config
DEPTH_NOISE_GENERATOR: Optional[DepthNoiseEncoder] = None
JIT_DEPTH_NOISE_GENERATOR: Optional[torch.jit.ScriptModule] = None

# Global height scan encoder - initialized once on first use
HEIGHTSCAN_FEAT_ENCODER: Optional[HeightScanFeatEncoder] = None
JIT_HEIGHTSCAN_FEAT_ENCODER: Optional[torch.jit.ScriptModule] = None

# Global flags/params for depth encoding
use_jit = True
min_depth = 0.0
max_depth = 0.0


def _quat_wxyz_to_yaw(quat: torch.Tensor) -> torch.Tensor:
    """Return yaw from quaternions in wxyz convention."""
    w, x, y, z = quat.unbind(dim=-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _quat_wxyz_to_matrix(quat: torch.Tensor) -> torch.Tensor:
    """Convert quaternions in wxyz convention to rotation matrices."""
    quat = F.normalize(quat, dim=-1)
    w, x, y, z = quat.unbind(dim=-1)
    two = 2.0
    matrix = torch.empty((*quat.shape[:-1], 3, 3), dtype=quat.dtype, device=quat.device)
    matrix[..., 0, 0] = 1.0 - two * (y * y + z * z)
    matrix[..., 0, 1] = two * (x * y - z * w)
    matrix[..., 0, 2] = two * (x * z + y * w)
    matrix[..., 1, 0] = two * (x * y + z * w)
    matrix[..., 1, 1] = 1.0 - two * (x * x + z * z)
    matrix[..., 1, 2] = two * (y * z - x * w)
    matrix[..., 2, 0] = two * (x * z - y * w)
    matrix[..., 2, 1] = two * (y * z + x * w)
    matrix[..., 2, 2] = 1.0 - two * (x * x + y * y)
    return matrix


class LocalVisibilityMapManager:
    """Maintain a robot-centric visibility map from noisy, delayed depth samples.

    The map stores only whether nearby terrain has been observed. It intentionally
    does not encode height, danger, terrain-generator information, or confidence.
    Previous map state is warped into the current robot-yaw frame and re-binarized
    so memory remains spatially meaningful as the robot moves.
    """

    def __init__(
        self,
        num_envs: int,
        device: str | torch.device,
        size_m: float = 6.4,
        grid_size: int = 64,
        depth_min_m: float = 0.5,
        depth_max_m: float = 8.0,
        pixel_stride: int = 2,
        decay: float = 1.0,
    ):
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.size_m = max(float(size_m), 0.5)
        self.grid_size = max(int(grid_size), 8)
        self.depth_min_m = max(float(depth_min_m), 0.0)
        self.depth_max_m = max(float(depth_max_m), self.depth_min_m + 1.0e-3)
        self.pixel_stride = max(int(pixel_stride), 1)
        self.decay = 1.0
        self.cell_size = self.size_m / float(self.grid_size)
        self.half_size = self.size_m / 2.0

        self.map = torch.zeros((self.num_envs, 1, self.grid_size, self.grid_size), device=self.device)
        self.last_root_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
        self.last_root_yaw = torch.zeros((self.num_envs,), device=self.device)
        self.pose_valid = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)

        rows = torch.arange(self.grid_size, device=self.device, dtype=torch.float32)
        cols = torch.arange(self.grid_size, device=self.device, dtype=torch.float32)
        row_grid, col_grid = torch.meshgrid(rows, cols, indexing="ij")
        self.cell_x_local = self.half_size - (row_grid + 0.5) * self.cell_size
        self.cell_y_local = self.half_size - (col_grid + 0.5) * self.cell_size
        self._pixel_cache: dict[tuple[int, int, int], tuple[torch.Tensor, torch.Tensor]] = {}

    def reset(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            self.map.zero_()
            self.pose_valid.zero_()
            return
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        self.map[env_ids] = 0.0
        self.pose_valid[env_ids] = False

    def _pixel_grid(self, height: int, width: int) -> tuple[torch.Tensor, torch.Tensor]:
        key = (int(height), int(width), self.pixel_stride)
        if key not in self._pixel_cache:
            rows = torch.arange(0, height, self.pixel_stride, device=self.device, dtype=torch.float32)
            cols = torch.arange(0, width, self.pixel_stride, device=self.device, dtype=torch.float32)
            row_grid, col_grid = torch.meshgrid(rows, cols, indexing="ij")
            self._pixel_cache[key] = (row_grid.reshape(1, -1), col_grid.reshape(1, -1))
        return self._pixel_cache[key]

    def _warp_memory_to_current_frame(self, root_pos_w: torch.Tensor, root_yaw: torch.Tensor):
        if not torch.any(self.pose_valid):
            return

        x_cur = self.cell_x_local.unsqueeze(0)
        y_cur = self.cell_y_local.unsqueeze(0)
        cos_cur = torch.cos(root_yaw).view(-1, 1, 1)
        sin_cur = torch.sin(root_yaw).view(-1, 1, 1)
        world_x = root_pos_w[:, 0].view(-1, 1, 1) + cos_cur * x_cur - sin_cur * y_cur
        world_y = root_pos_w[:, 1].view(-1, 1, 1) + sin_cur * x_cur + cos_cur * y_cur

        prev_pos = torch.where(self.pose_valid[:, None], self.last_root_pos_w, root_pos_w)
        prev_yaw = torch.where(self.pose_valid, self.last_root_yaw, root_yaw)
        rel_x = world_x - prev_pos[:, 0].view(-1, 1, 1)
        rel_y = world_y - prev_pos[:, 1].view(-1, 1, 1)
        cos_prev = torch.cos(prev_yaw).view(-1, 1, 1)
        sin_prev = torch.sin(prev_yaw).view(-1, 1, 1)
        x_prev = cos_prev * rel_x + sin_prev * rel_y
        y_prev = -sin_prev * rel_x + cos_prev * rel_y

        row_prev = (self.half_size - x_prev) / self.cell_size - 0.5
        col_prev = (self.half_size - y_prev) / self.cell_size - 0.5
        norm_x = 2.0 * col_prev / max(self.grid_size - 1, 1) - 1.0
        norm_y = 2.0 * row_prev / max(self.grid_size - 1, 1) - 1.0
        sample_grid = torch.stack((norm_x, norm_y), dim=-1)
        warped = F.grid_sample(
            self.map, sample_grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )
        self.map = torch.where(self.pose_valid.view(-1, 1, 1, 1), (warped > 0.5).to(warped.dtype), torch.zeros_like(warped))

    def update(
        self,
        depth: torch.Tensor,
        camera_pos_w: torch.Tensor,
        camera_quat_w: torch.Tensor,
        intrinsics: torch.Tensor,
        root_pos_w: torch.Tensor,
        root_quat_w: torch.Tensor,
    ) -> torch.Tensor:
        depth = depth.to(self.device)
        if depth.dim() == 4:
            depth = depth[:, 0]
        num_envs, height, width = depth.shape
        if num_envs != self.num_envs:
            raise ValueError(f"Expected {self.num_envs} envs, got {num_envs}")

        camera_pos_w = camera_pos_w.to(self.device)
        camera_quat_w = camera_quat_w.to(self.device)
        intrinsics = intrinsics.to(self.device)
        root_pos_w = root_pos_w.to(self.device)
        root_quat_w = root_quat_w.to(self.device)
        root_yaw = _quat_wxyz_to_yaw(root_quat_w)

        self._warp_memory_to_current_frame(root_pos_w, root_yaw)

        depth_sub = depth[:, :: self.pixel_stride, :: self.pixel_stride].reshape(num_envs, -1)
        row_grid, col_grid = self._pixel_grid(height, width)
        valid_depth = torch.isfinite(depth_sub) & (depth_sub >= self.depth_min_m) & (depth_sub <= self.depth_max_m)

        fx = intrinsics[:, 0, 0].clamp_min(1.0e-6).unsqueeze(1)
        fy = intrinsics[:, 1, 1].clamp_min(1.0e-6).unsqueeze(1)
        cx = intrinsics[:, 0, 2].unsqueeze(1)
        cy = intrinsics[:, 1, 2].unsqueeze(1)
        x_cam = depth_sub
        y_cam = -((col_grid - cx) / fx) * depth_sub
        z_cam = -((row_grid - cy) / fy) * depth_sub
        points_cam = torch.stack((x_cam, y_cam, z_cam), dim=-1)

        rotation = _quat_wxyz_to_matrix(camera_quat_w)
        points_w = torch.bmm(points_cam, rotation.transpose(1, 2)) + camera_pos_w.unsqueeze(1)
        finite_points = torch.isfinite(points_w).all(dim=-1)

        rel = points_w - root_pos_w.unsqueeze(1)
        cos_yaw = torch.cos(root_yaw).unsqueeze(1)
        sin_yaw = torch.sin(root_yaw).unsqueeze(1)
        x_local = cos_yaw * rel[..., 0] + sin_yaw * rel[..., 1]
        y_local = -sin_yaw * rel[..., 0] + cos_yaw * rel[..., 1]

        row = torch.floor((self.half_size - x_local) / self.cell_size).long()
        col = torch.floor((self.half_size - y_local) / self.cell_size).long()
        valid = (
            valid_depth
            & finite_points
            & (row >= 0)
            & (row < self.grid_size)
            & (col >= 0)
            & (col < self.grid_size)
        )
        linear_idx = (row.clamp(0, self.grid_size - 1) * self.grid_size + col.clamp(0, self.grid_size - 1))
        updates = torch.zeros((self.num_envs, self.grid_size * self.grid_size), device=self.device)
        values = valid.to(dtype=updates.dtype)
        updates.scatter_reduce_(1, linear_idx, values, reduce="amax", include_self=True)
        self.map = torch.maximum(self.map, updates.view_as(self.map))

        self.last_root_pos_w.copy_(root_pos_w)
        self.last_root_yaw.copy_(root_yaw)
        self.pose_valid[:] = True
        self.map = (self.map > 0.5).to(self.map.dtype)
        return self.map.flatten(start_dim=1)


def _get_or_create_visibility_map_manager(
    env: ManagerBasedEnv,
    size_m: float,
    grid_size: int,
    depth_min_m: float,
    depth_max_m: float,
    pixel_stride: int,
    decay: float,
) -> LocalVisibilityMapManager:
    manager = getattr(env, "local_visibility_map_manager", None)
    if manager is None:
        manager = LocalVisibilityMapManager(
            num_envs=env.num_envs,
            device=env.device,
            size_m=size_m,
            grid_size=grid_size,
            depth_min_m=depth_min_m,
            depth_max_m=depth_max_m,
            pixel_stride=pixel_stride,
            decay=decay,
        )
        env.local_visibility_map_manager = manager
    return manager


def _cache_noisy_depth_sample(env, camera_name: str, noisy_depth: torch.Tensor, depth_camera) -> None:
    cache = getattr(env, "_nav_noisy_depth_cache", None)
    if cache is None:
        cache = {}
        env._nav_noisy_depth_cache = cache
    cache[camera_name] = {
        "depth": noisy_depth.detach(),
        "camera_pos_w": depth_camera.data.pos_w.detach(),
        "camera_quat_w": depth_camera.data.quat_w_world.detach(),
        "intrinsics": depth_camera.data.intrinsic_matrices.detach(),
    }


def _get_noisy_depth_sample(env, sensor_cfg) -> dict[str, torch.Tensor]:
    camera_name = sensor_cfg.name if hasattr(sensor_cfg, "name") else "raycast_camera"
    cache = getattr(env, "_nav_noisy_depth_cache", {})
    if camera_name in cache:
        return cache[camera_name]

    _ensure_depth_noise_generator_initialized()
    depth_camera = env.scene.sensors[sensor_cfg.name]
    depth_tensor = depth_camera.data.output["distance_to_image_plane"].view(env.num_envs, -1)
    depth_tensor = torch.nan_to_num(depth_tensor, nan=50.0, posinf=50.0, neginf=0.0)
    height, width = depth_camera.image_shape
    depth_tensor = depth_tensor.view(-1, 1, height, width)

    model = cast(torch.jit.ScriptModule, JIT_DEPTH_NOISE_GENERATOR)
    if use_jit:
        depth_tensor = depth_tensor.clone()
        depth_tensor[depth_tensor > max_depth] = 0.0
        depth_tensor[depth_tensor < min_depth] = 0.0
        noisy_depth_tensor = depth_tensor
    else:
        _, noisy_depth_tensor = model(depth_tensor)
    _cache_noisy_depth_sample(env, camera_name, noisy_depth_tensor, depth_camera)
    return env._nav_noisy_depth_cache[camera_name]


def _update_depth_window(camera_name: str, depth_tensor: torch.Tensor, height: int, width: int, title: str) -> None:
    """Update (or lazily create) a persistent matplotlib window for a camera.

    This enables continuous, non-blocking visualization across steps for one or more cameras.
    """
    try:
        # Convert to numpy and handle different tensor shapes
        if depth_tensor.dim() == 4:  # [B, 1, H, W]
            depth_np = depth_tensor[0, 0].detach().cpu().numpy()
        elif depth_tensor.dim() == 2:  # [B, H*W]
            depth_np = depth_tensor[0].detach().cpu().numpy().reshape(height, width)
        else:
            return

        # Lazily create window if missing
        if camera_name not in _DEPTH_VIZ_WINDOWS:
            plt.ion()
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            im = ax.imshow(depth_np, cmap="plasma", aspect="equal")
            cbar = plt.colorbar(im, ax=ax, label="Depth (meters)")
            ax.set_title(title)
            ax.set_xlabel("Width (pixels)")
            ax.set_ylabel("Height (pixels)")
            _DEPTH_VIZ_WINDOWS[camera_name] = {"fig": fig, "ax": ax, "im": im, "cbar": cbar}
        else:
            handle = _DEPTH_VIZ_WINDOWS[camera_name]
            im = handle["im"]
            ax = handle["ax"]
            ax.set_title(title)
            im.set_data(depth_np)
            # Optional: keep consistent color scaling per window
            im.set_clim(vmin=depth_np.min(), vmax=depth_np.max())

        # Lightweight draw
        _DEPTH_VIZ_WINDOWS[camera_name]["fig"].canvas.draw_idle()
        plt.pause(0.001)
    except Exception as e:
        print(f"Error updating depth window for {camera_name}: {e}")


def _ensure_depth_noise_generator_initialized(
    camera_config: Optional[CameraConfig] = None,
    use_jit_precompiled: bool = True,
    feature_dim: int = 64,
):
    """Ensure the depth noise generator is initialized with the correct configuration.

    This is called automatically by observation functions that need depth encoding.
    If not explicitly initialized via initialize_depth_noise_generator(), it will use defaults.

    Args:
        camera_config: The camera configuration to use. If None, uses DEFAULT_CAMERA_CONFIG.
        use_jit_precompiled: Whether to use JIT compilation for faster inference.
        feature_dim: Feature dimension for the encoder output.
    """
    global DEPTH_NOISE_GENERATOR, JIT_DEPTH_NOISE_GENERATOR, use_jit, min_depth, max_depth

    # Only initialize if not already done
    if DEPTH_NOISE_GENERATOR is not None:
        return

    # Use provided camera config or default
    config = camera_config if camera_config is not None else DEFAULT_CAMERA_CONFIG

    # Extract camera parameters
    min_depth = config.min_depth
    max_depth = config.max_depth
    resolution = config.resolution

    print("=" * 80)
    print("Initializing depth noise generator for navigation observation:")
    print(f"  Resolution: {resolution}")
    print(f"  Depth range: [{min_depth}, {max_depth}]")
    print(f"  Feature dim: {feature_dim}")
    print(f"  Encoder path: {config.depth_encoder_path}")
    print(f"  Use JIT: {use_jit_precompiled}")

    # Initialize encoder with the new simplified API
    # The DepthNoiseEncoder now takes camera_config directly
    DEPTH_NOISE_GENERATOR = DepthNoiseEncoder(
        feature_dim=feature_dim,
        camera_config=config,
    ).to(torch.device("cuda"))
    DEPTH_NOISE_GENERATOR.eval()

    # Create JIT version for inference (optional optimization)
    use_jit = use_jit_precompiled
    if use_jit:
        # Try to find a JIT compiled version
        jit_path = config.depth_encoder_path.replace('.pth', '_jit.pt') if config.depth_encoder_path else None
        if jit_path and os.path.exists(jit_path):
            print(f"  Loading precompiled JIT model from: {jit_path}")
            JIT_DEPTH_NOISE_GENERATOR = torch.jit.load(jit_path, map_location="cuda")
            JIT_DEPTH_NOISE_GENERATOR = torch.jit.optimize_for_inference(JIT_DEPTH_NOISE_GENERATOR)
        else:
            print(f"  JIT compilation requested but no precompiled model found.")
            print(f"  Creating JIT model from encoder...")
            example_input = torch.randn(1, 1, resolution[1], resolution[0]).cuda()  # (B, C, H, W)
            JIT_DEPTH_NOISE_GENERATOR = torch.jit.trace(DEPTH_NOISE_GENERATOR, example_input)
            JIT_DEPTH_NOISE_GENERATOR = torch.jit.optimize_for_inference(JIT_DEPTH_NOISE_GENERATOR)
    else:
        JIT_DEPTH_NOISE_GENERATOR = DEPTH_NOISE_GENERATOR

    print("  Depth noise generator initialized successfully")
    print("=" * 80)


def initialize_depth_noise_generator(
    camera_config: Optional[CameraConfig] = None,
    robot_name: Optional[str] = None,
    use_jit_precompiled: bool = True,
    feature_dim: int = 64,
):
    """Initialize the depth noise generator with specific configuration.

    This function should be called in the environment's __post_init__() to set up
    the depth encoder before any observations are computed.

    Args:
        camera_config: The camera configuration to use. If None and robot_name is provided,
                      uses the config for that robot. If both are None, uses DEFAULT_CAMERA_CONFIG.
        robot_name: Name of the robot (e.g., 'b2w', 'aow_d'). If provided and
                   camera_config is None, automatically loads the appropriate camera config.
        use_jit_precompiled: Whether to use JIT compilation for faster inference. Defaults to True.
        feature_dim: Feature dimension for the encoder output. Defaults to 64.

    Examples:
        # Using robot name (recommended for multi-robot training)
        initialize_depth_noise_generator(robot_name="b2w")

        # Using explicit camera config
        initialize_depth_noise_generator(camera_config=ZEDX_CAMERA_CONFIG)
    """
    # If camera_config not provided, try to get it from robot_name
    if camera_config is None and robot_name is not None:
        camera_config = get_camera_config(robot_name, use_default_fallback=False)

    _ensure_depth_noise_generator_initialized(camera_config, use_jit_precompiled, feature_dim)


# ============================================================================
# Observation Functions
# ============================================================================


def generated_commands_reshaped(
    env: ManagerBasedRLEnv, command_name: str, unsqueeze_pos: int = 1, flatten: bool = False
) -> torch.Tensor:
    """The generated command from command term in the command manager with the given name."""
    if flatten:
        return env.command_manager.get_command(command_name)
    return env.command_manager.get_command(command_name).unsqueeze(unsqueeze_pos)


def base_lin_vel_delayed(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Compute delayed root linear velocity.

    Requires env.delay_manager to exist. Each environment has a random delay
    sampled from [0, max_delay] at episode reset.

    Args:
        env: The environment object (must have delay_manager attribute).
        asset_cfg: The name of the asset.

    Returns:
        The delayed linear velocity in the asset's root frame.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    lin_vel = asset.data.root_lin_vel_b
    return env.delay_manager.compute_delayed_lin_vel(lin_vel)


def base_ang_vel_delayed(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Compute delayed root angular velocity.

    Requires env.delay_manager to exist. Each environment has a random delay
    sampled from [0, max_delay] at episode reset.

    Args:
        env: The environment object (must have delay_manager attribute).
        asset_cfg: The name of the asset.

    Returns:
        The delayed angular velocity in the asset's root frame.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    ang_vel = asset.data.root_ang_vel_b
    return env.delay_manager.compute_delayed_ang_vel(ang_vel)


def projected_gravity_delayed(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Compute delayed projected gravity.

    Requires env.delay_manager to exist. Each environment has a random delay
    sampled from [0, max_delay] at episode reset.

    Args:
        env: The environment object (must have delay_manager attribute).
        asset_cfg: The name of the asset.

    Returns:
        The delayed projected gravity in the asset's root frame.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    projected_gravity = asset.data.projected_gravity_b

    return env.delay_manager.compute_delayed_projected_gravity(projected_gravity)


def height_scan_feat(
    env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner_critic"), offset: float = 0.5
) -> torch.Tensor:
    """Height scan feature from a ray caster sensor, encoded using a pre-trained VAE encoder.

    The height scan is reshaped to a 51x51 grid and encoded using a VAE encoder that
    outputs a 64-channel, 7x7 spatial feature map (64*7*7 = 3136 features).

    Args:
        env: The environment object.
        sensor_cfg: The configuration of the height scanner sensor.
        offset: Offset to subtract from the height values. Defaults to 0.5.

    Returns:
        The encoded height scan features of shape (num_envs, 3136).
    """
    global HEIGHTSCAN_FEAT_ENCODER, JIT_HEIGHTSCAN_FEAT_ENCODER

    # Initialize the encoder on first call
    if HEIGHTSCAN_FEAT_ENCODER is None:
        print("Initializing height scan feature encoder...")
        HEIGHTSCAN_FEAT_ENCODER = HeightScanFeatEncoder(feature_dim=64).to(torch.device("cuda"))
        HEIGHTSCAN_FEAT_ENCODER.eval()
        JIT_HEIGHTSCAN_FEAT_ENCODER = torch.jit.script(HEIGHTSCAN_FEAT_ENCODER)
        JIT_HEIGHTSCAN_FEAT_ENCODER = torch.jit.optimize_for_inference(JIT_HEIGHTSCAN_FEAT_ENCODER)

    # Get height scanner data
    height_scanner = env.scene.sensors[sensor_cfg.name]

    # Compute height scan: sensor_height - hit_z - offset
    scan_data = height_scanner.data.pos_w[:, 2].unsqueeze(1) - height_scanner.data.ray_hits_w[..., 2] - offset

    # Clamp the height scan data to the range [-5, 5]
    scan_data = torch.clamp(scan_data, min=-5.0, max=5.0)

    # Expected grid size for the height scanner (51x51 = 2601 points)
    H = W = 51

    # Reshape the height scan data to a 2D grid
    scan_data = scan_data.view(-1, H, W)

    # Encode using the pre-trained encoder
    with torch.no_grad():
        encoded_scan = JIT_HEIGHTSCAN_FEAT_ENCODER(scan_data)

    # Flatten and return: (batch, 64, 7, 7) -> (batch, 3136)
    return encoded_scan.view(env.num_envs, -1)


def generated_actions(env: ManagerBasedRLEnv, action_name: str) -> torch.Tensor:
    """The generated action from action term in the action manager with the given name.

    Args:
        env: The environment object.
        action_name: The name of the action term.

    Returns:
        The processed actions from the action term.
    """
    return env.action_manager.get_term(action_name).processed_actions


def generated_commands_reshaped_delayed(
    env: ManagerBasedRLEnv,
    command_name: str,
    unsqueeze_pos: int = 1,
    flatten: bool = False,
) -> torch.Tensor:
    """The generated command with delay applied.

    Requires env.delay_manager to exist. Each environment has a random delay
    sampled from [0, max_delay] at episode reset.

    Args:
        env: The environment object (must have delay_manager attribute).
        command_name: The name of the command term.
        unsqueeze_pos: Position to unsqueeze the command.
        flatten: Whether to flatten the command.

    Returns:
        The delayed command.
    """
    if flatten:
        command = env.command_manager.get_command(command_name)
        return env.delay_manager.compute_delayed_target_position(command)

    command = env.command_manager.get_command(command_name).unsqueeze(unsqueeze_pos)
    return env.delay_manager.compute_delayed_target_position(command)


def last_low_level_action(
    env: ManagerBasedEnv, action_term: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """The last low-level action.

    Args:
        env: The environment object.
        action_term: The name of the action term.

    Returns:
        The last low-level action.
    """
    action_term: PerceptiveNavigationSE2Action = env.action_manager._terms[action_term]
    return action_term.low_level_actions[:, asset_cfg.joint_ids]


@torch.inference_mode()
def depth_image_prefect(env, sensor_cfg):
    """Return the perfect (non-noisy) encoded depth image from the camera.

    Args:
        env: The environment object.
        sensor_cfg: The sensor configuration.

    Returns:
        The encoded depth image features.
    """
    # Ensure encoder is initialized
    _ensure_depth_noise_generator_initialized()

    depth_camera = env.scene.sensors[sensor_cfg.name]

    # Get depth image tensor
    depth_tensor = depth_camera.data.output["distance_to_image_plane"].view(env.num_envs, -1)

    # Handle NaN values by replacing them with 50.0
    depth_tensor = torch.nan_to_num(depth_tensor, nan=50.0, posinf=50.0, neginf=0.0)

    # Reshape the tensor to [B, 1, H, W]
    H, W = depth_camera.image_shape
    depth_tensor = depth_tensor.view(-1, 1, H, W)

    assert JIT_DEPTH_NOISE_GENERATOR is not None, (
        "Depth encoder JIT model is not initialized. Call initialize_depth_noise_generator first."
    )
    model = cast(torch.jit.ScriptModule, JIT_DEPTH_NOISE_GENERATOR)
    if use_jit:
        depth_tensor[depth_tensor > max_depth] = 0.0
        encoded_depth_tensor = model(depth_tensor)
    else:
        encoded_depth_tensor, _ = model(depth_tensor)

    return encoded_depth_tensor.view(env.num_envs, -1)


@torch.no_grad()
def depth_image_noisy_delayed(
    env, sensor_cfg, visualize: bool = False
):
    """Return the noisy and delayed encoded depth image from the camera.

    Requires env.delay_manager to exist. Each environment has a random delay
    sampled from [0, max_delay] at episode reset.

    Args:
        env: The environment object (must have delay_manager attribute).
        sensor_cfg: The sensor configuration.
        visualize: Whether to visualize the depth image.

    Returns:
        The delayed encoded depth image features.
    """
    # Ensure encoder is initialized
    _ensure_depth_noise_generator_initialized()

    depth_camera = env.scene.sensors[sensor_cfg.name]

    # Get depth image tensor
    depth_tensor = depth_camera.data.output["distance_to_image_plane"].view(env.num_envs, -1)

    # Handle NaN values by replacing them with 50.0
    depth_tensor = torch.nan_to_num(depth_tensor, nan=50.0, posinf=50.0, neginf=0.0)

    # Reshape the tensor to [B, 1, H, W]
    H, W = depth_camera.image_shape
    depth_tensor = depth_tensor.view(-1, 1, H, W)

    assert JIT_DEPTH_NOISE_GENERATOR is not None, (
        "Depth encoder JIT model is not initialized. Call initialize_depth_noise_generator first."
    )
    model = cast(torch.jit.ScriptModule, JIT_DEPTH_NOISE_GENERATOR)
    if use_jit:
        depth_tensor[depth_tensor > max_depth] = 0.0  # depth larger than depth max is invalid
        depth_tensor[depth_tensor < min_depth] = 0.0  # depth smaller than depth min is invalid
        encoded_depth_tensor = model(depth_tensor)
        noisy_depth_tensor = depth_tensor
    else:
        encoded_depth_tensor, noisy_depth_tensor = model(depth_tensor)

    camera_name = sensor_cfg.name if hasattr(sensor_cfg, "name") else "depth"
    _cache_noisy_depth_sample(env, camera_name, noisy_depth_tensor, depth_camera)

    # Continuous visualization (single-env) with persistent windows
    if visualize and env.num_envs == 1 and not use_jit:
        _update_depth_window(camera_name, noisy_depth_tensor, H, W, title=f"Depth (Noisy Delayed) - {camera_name}")

    encoded_depth_tensor_reshaped = encoded_depth_tensor.view(env.num_envs, -1)

    # Apply delay using env's delay manager
    return env.delay_manager.compute_delayed_depth(encoded_depth_tensor_reshaped, camera_name)


def local_visibility_map(
    env,
    sensor_cfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    size_m: float = 6.4,
    grid_size: int = 64,
    depth_min_m: float = 0.5,
    depth_max_m: float = 8.0,
    pixel_stride: int = 2,
    decay: float = 1.0,
):
    """Return a flattened binary robot-centric visibility map built from noisy, delayed depth.

    The delayed raw sample packs depth, camera pose, and intrinsics together so the
    back-projection uses a temporally consistent camera frame. The map contains no
    privileged terrain height or danger labels.
    """
    sample = _get_noisy_depth_sample(env, sensor_cfg)
    depth = sample["depth"]
    if depth.dim() == 4:
        depth_flat = depth[:, 0].flatten(start_dim=1)
        height, width = depth.shape[-2:]
    else:
        height, width = env.scene.sensors[sensor_cfg.name].image_shape
        depth_flat = depth.view(env.num_envs, -1)

    packed = torch.cat(
        (
            depth_flat,
            sample["camera_pos_w"],
            sample["camera_quat_w"],
            sample["intrinsics"].reshape(env.num_envs, -1),
        ),
        dim=1,
    )
    camera_name = sensor_cfg.name if hasattr(sensor_cfg, "name") else "depth"
    delayed = env.delay_manager.compute_delayed_depth(packed, f"{camera_name}_visibility_sample")

    depth_size = height * width
    delayed_depth = delayed[:, :depth_size].view(env.num_envs, 1, height, width)
    camera_pos_w = delayed[:, depth_size : depth_size + 3]
    camera_quat_w = delayed[:, depth_size + 3 : depth_size + 7]
    intrinsics = delayed[:, depth_size + 7 : depth_size + 16].view(env.num_envs, 3, 3)

    robot = env.scene[asset_cfg.name]
    manager = _get_or_create_visibility_map_manager(
        env,
        size_m=size_m,
        grid_size=grid_size,
        depth_min_m=depth_min_m,
        depth_max_m=depth_max_m,
        pixel_stride=pixel_stride,
        decay=decay,
    )
    return manager.update(
        delayed_depth,
        camera_pos_w,
        camera_quat_w,
        intrinsics,
        robot.data.root_pos_w,
        robot.data.root_quat_w,
    )


def in_goal(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    distance_threshold: float = 0.5,
    goal_cmd_name: str = "robot_goal",
) -> torch.Tensor:
    """Check if the robot is within the goal distance threshold.

    Args:
        env: The learning environment.
        asset_cfg: The name of the robot asset.
        distance_threshold: The distance threshold to the goal.
        goal_cmd_name: The name of the goal command.

    Returns:
        Boolean tensor indicating whether the robot is within the goal.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    goal_cmd_generator: RobotNavigationGoalCommand = env.command_manager._terms[goal_cmd_name]
    distance_goal = torch.norm(asset.data.root_pos_w[:, :2] - goal_cmd_generator.pos_command_w[:, :2], dim=1, p=2, keepdim=True)
    return distance_goal < distance_threshold


def time_normalized(env: ManagerBasedRLEnv, command_name: str = "robot_goal") -> torch.Tensor:
    """Time normalized to the maximum episode length.

    Args:
        env: The learning environment.
        command_name: The name of the goal command.

    Returns:
        The normalized time (current step / max steps).
    """
    T_max = env.max_episode_length
    if hasattr(env, "episode_length_buf"):
        t = env.episode_length_buf.unsqueeze(-1)
    else:
        t = torch.tensor([0.0]).repeat(env.num_envs, 1).to(env.device)
    return t / T_max

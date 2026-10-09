# Copyright (c) 2022-2025, Fan Yang and Per Frivik, ETH Zurich.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Observation functions for navigation tasks.

This module is the observation-side glue between IsaacLab managers, the SRU
high-level navigation policy, and the frozen low-level locomotion policies.
Functions in this file are referenced by ``ObservationTermCfg`` entries in the
environment config. IsaacLab calls them every environment step when it builds an
observation group.

Important conventions used throughout this file:

- ``env.scene[...]`` reads simulation state from IsaacLab assets and sensors.
- Actor observations may be noisy and delayed; critic observations are usually
  cleaner and may include privileged information such as encoded height scans.
- Depth observations are not fed to the SRU as raw pixels. They are encoded by a
  pre-trained depth encoder into ``[64, 5, 8]`` features and then flattened.
- The local visibility map is actor-side explicit memory built from the same
  noisy/delayed depth stream. It is a binary "observed / not observed" BEV map;
  it is not an occupancy map, traversability map, or height map.
- Tensor shapes are documented in comments because network slicing in
  ``ActorCriticSRU`` assumes these exact layouts.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, Literal, Optional, cast

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
from isaaclab.envs.mdp.events import _randomize_prop_by_op
from isaaclab.managers import ManagerTermBase, ManagerTermBaseCfg, SceneEntityCfg

import instinct_nav.navigation.mdp as mdp

from .depth_utils.camera_config import CameraConfig, DEFAULT_CAMERA_CONFIG, get_camera_config
from .depth_utils.depth_noise_encoder import DepthNoiseEncoder
from .heightscan_utils.heightscan_encoder import HeightScanFeatEncoder

if TYPE_CHECKING:
    from instinct_nav.navigation.mdp import PerceptiveNavigationSE2Action
    from instinct_nav.navigation.mdp.navigation.goal_commands import RobotNavigationGoalCommand

# Matplotlib windows used only by optional single-env depth visualization.
# They are kept globally so repeated observation calls update the same window
# instead of opening a new one at every step.
_DEPTH_VIZ_WINDOWS: dict[str, dict] = {}

# Depth encoder singleton.
#
# The encoder is expensive to construct and has fixed weights, so all envs in a
# process share one instance. ``initialize_depth_noise_generator(...)`` should be
# called from the environment config with the robot/camera-specific parameters
# before the first observation call. Observation functions still call
# ``_ensure_depth_noise_generator_initialized`` defensively in case the config did
# not initialize it explicitly.
DEPTH_NOISE_GENERATOR: Optional[DepthNoiseEncoder] = None
JIT_DEPTH_NOISE_GENERATOR: Optional[torch.jit.ScriptModule] = None

# Height-scan encoder singleton used by the critic. It is initialized lazily the
# first time ``height_scan_feat`` is requested.
HEIGHTSCAN_FEAT_ENCODER: Optional[HeightScanFeatEncoder] = None
JIT_HEIGHTSCAN_FEAT_ENCODER: Optional[torch.jit.ScriptModule] = None

# Runtime depth preprocessing flags copied from the selected camera config.
# ``min_depth`` and ``max_depth`` define what the depth encoder considers a valid
# range; invalid values are set to 0.0 before encoding when JIT mode is used.
use_jit = True
min_depth = 0.0
max_depth = 0.0


def _quat_wxyz_to_yaw(quat: torch.Tensor) -> torch.Tensor:
    """Return yaw from quaternions in wxyz convention.

    IsaacLab stores root orientation as ``[w, x, y, z]``. Visibility maps are
    robot-centric but yaw-aligned: we care about the robot's heading in the
    horizontal plane, not roll/pitch. This helper extracts that yaw angle.
    """
    w, x, y, z = quat.unbind(dim=-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _quat_wxyz_to_matrix(quat: torch.Tensor) -> torch.Tensor:
    """Convert quaternions in wxyz convention to rotation matrices.

    The visibility map construction back-projects depth pixels into camera-frame
    3D points and then rotates them into world frame. This helper builds that
    rotation matrix for every environment in the vectorized batch.
    """
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
    """Maintain a robot-centric local visibility map.

    This is the explicit-memory component used by the visibility-map fusion
    ablations. It turns the actor's depth stream into a 2D binary BEV grid:

    - Shape kept internally: ``[num_envs, 1, grid_size, grid_size]``.
    - Shape returned to the actor: ``[num_envs, grid_size * grid_size]``.
    - Default coverage: ``6.4m x 6.4m`` around the robot.
    - Default resolution: ``64 x 64``, so each grid cell is ``0.1m x 0.1m``.

    Semantics are deliberately weak and non-privileged:

    - ``1`` means this local cell has been hit/observed by delayed depth.
    - ``0`` means the cell has not been observed recently/in this episode.
    - It does NOT mean obstacle/free space/traversable/unsafe/height.

    The map is stored in the robot's current yaw-aligned frame. When the robot
    moves, old map values are warped from the previous robot frame into the new
    frame before current depth hits are inserted. This keeps explicit memory
    spatially meaningful without exposing global terrain coordinates to the
    policy.
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
        # Store all map state on the simulation device. Visibility map update is
        # called inside the observation pipeline, so moving tensors to CPU here
        # would severely slow training.
        self.num_envs = int(num_envs)
        self.device = torch.device(device)

        # Clamp config values to avoid degenerate maps. In normal G1 runs these
        # remain: size_m=6.4, grid_size=64, cell_size=0.1m.
        self.size_m = max(float(size_m), 0.5)
        self.grid_size = max(int(grid_size), 8)
        self.depth_min_m = max(float(depth_min_m), 0.0)
        self.depth_max_m = max(float(depth_max_m), self.depth_min_m + 1.0e-3)
        self.pixel_stride = max(int(pixel_stride), 1)

        # decay=1 preserves the legacy binary map; lower values enable confidence.
        self.decay = min(max(float(decay), 0.0), 1.0)
        self.score_mode = self.decay < 1.0
        self.score_alpha_up = 0.15
        # Repeated hits rise toward one while stale cells decay gradually.
        self.cell_size = self.size_m / float(self.grid_size)
        self.half_size = self.size_m / 2.0

        # Binary map memory. ``pose_valid`` marks whether a map has a previous
        # robot pose; without it we cannot warp old memory into the new frame.
        self.map = torch.zeros((self.num_envs, 1, self.grid_size, self.grid_size), device=self.device)
        self.current_dynamic_risk = torch.zeros_like(self.map)
        # Two ego-motion-compensated dynamic occupancy frames. They are kept
        # separate from persistent visibility memory so moving objects never
        # leave permanent "ghost" cells in the static map.
        self.dynamic_history: torch.Tensor | None = None
        self.last_root_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
        self.last_root_yaw = torch.zeros((self.num_envs,), device=self.device)
        self.pose_valid = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)

        # Precompute each BEV cell center in the robot-local frame. The indexing
        # convention is chosen so row 0 is in front of the robot and later rows
        # move backward; columns span left/right in robot yaw frame.
        rows = torch.arange(self.grid_size, device=self.device, dtype=torch.float32)
        cols = torch.arange(self.grid_size, device=self.device, dtype=torch.float32)
        row_grid, col_grid = torch.meshgrid(rows, cols, indexing="ij")
        self.cell_x_local = self.half_size - (row_grid + 0.5) * self.cell_size
        self.cell_y_local = self.half_size - (col_grid + 0.5) * self.cell_size

        # Pixel grids depend only on camera resolution and stride. Cache them so
        # every observation call does not rebuild the same meshgrid.
        self._pixel_cache: dict[tuple[int, int, int], tuple[torch.Tensor, torch.Tensor]] = {}

    def reset(self, env_ids: torch.Tensor | None = None):
        """Clear visibility memory for all envs or a subset of reset envs.

        This must run on episode reset. Otherwise a new episode could inherit a
        map built at another spawn/goal, leaking invalid spatial memory into the
        actor observation.
        """
        if env_ids is None:
            self.map.zero_()
            self.current_dynamic_risk.zero_()
            if self.dynamic_history is not None:
                self.dynamic_history.zero_()
            self.pose_valid.zero_()
            return
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        self.map[env_ids] = 0.0
        self.current_dynamic_risk[env_ids] = 0.0
        if self.dynamic_history is not None:
            self.dynamic_history[env_ids] = 0.0
        self.pose_valid[env_ids] = False

    @property
    def risk_adjusted_map(self) -> torch.Tensor:
        """Current visibility with transient dynamic hazards rendered as black."""
        return self.map * (1.0 - self.current_dynamic_risk)

    def _pixel_grid(self, height: int, width: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return flattened image row/column coordinates for sampled pixels.

        ``pixel_stride`` subsamples the depth image. With a 40x64 depth image and
        stride=2, only 20x32 rays are inserted into the map each policy step.
        That is enough for coarse visibility memory and keeps the observation
        path cheap for thousands of envs.
        """
        key = (int(height), int(width), self.pixel_stride)
        if key not in self._pixel_cache:
            rows = torch.arange(0, height, self.pixel_stride, device=self.device, dtype=torch.float32)
            cols = torch.arange(0, width, self.pixel_stride, device=self.device, dtype=torch.float32)
            row_grid, col_grid = torch.meshgrid(rows, cols, indexing="ij")
            self._pixel_cache[key] = (row_grid.reshape(1, -1), col_grid.reshape(1, -1))
        return self._pixel_cache[key]

    def _warp_memory_to_current_frame(self, root_pos_w: torch.Tensor, root_yaw: torch.Tensor):
        """Warp the previous robot-centric map into the current robot frame.

        The map is stored in local robot coordinates, not world coordinates. If
        the robot moves forward or turns, the same world point should appear at a
        different local cell. Before inserting new depth hits, we therefore sample
        the old map at the previous-frame coordinates corresponding to each
        current-frame cell.

        This is the key step that turns per-frame depth observations into short
        horizon explicit memory.
        """
        if not torch.any(self.pose_valid):
            return

        # Current local BEV cell centers -> world XY. These are the world points
        # represented by each cell if the robot is at its current pose.
        x_cur = self.cell_x_local.unsqueeze(0)
        y_cur = self.cell_y_local.unsqueeze(0)
        cos_cur = torch.cos(root_yaw).view(-1, 1, 1)
        sin_cur = torch.sin(root_yaw).view(-1, 1, 1)
        world_x = root_pos_w[:, 0].view(-1, 1, 1) + cos_cur * x_cur - sin_cur * y_cur
        world_y = root_pos_w[:, 1].view(-1, 1, 1) + sin_cur * x_cur + cos_cur * y_cur

        # World XY -> previous robot-local frame. We ask: where did this current
        # cell's world point live in the old map? That previous coordinate is the
        # sampling location used by grid_sample below.
        prev_pos = torch.where(self.pose_valid[:, None], self.last_root_pos_w, root_pos_w)
        prev_yaw = torch.where(self.pose_valid, self.last_root_yaw, root_yaw)
        rel_x = world_x - prev_pos[:, 0].view(-1, 1, 1)
        rel_y = world_y - prev_pos[:, 1].view(-1, 1, 1)
        cos_prev = torch.cos(prev_yaw).view(-1, 1, 1)
        sin_prev = torch.sin(prev_yaw).view(-1, 1, 1)
        x_prev = cos_prev * rel_x + sin_prev * rel_y
        y_prev = -sin_prev * rel_x + cos_prev * rel_y

        # Previous local meters -> previous image/grid coordinates. grid_sample
        # expects normalized coordinates in [-1, 1], where x indexes columns and
        # y indexes rows.
        row_prev = (self.half_size - x_prev) / self.cell_size - 0.5
        col_prev = (self.half_size - y_prev) / self.cell_size - 0.5
        norm_x = 2.0 * col_prev / max(self.grid_size - 1, 1) - 1.0
        norm_y = 2.0 * row_prev / max(self.grid_size - 1, 1) - 1.0
        sample_grid = torch.stack((norm_x, norm_y), dim=-1)

        # Bilinear sampling can produce fractional values near cell boundaries.
        # The map semantics are binary, so threshold back to {0, 1} after warp.
        # Invalid envs use an all-zero map until their first valid pose arrives.
        warped = F.grid_sample(
            self.map, sample_grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )
        warped_memory = warped if self.score_mode else (warped > 0.5).to(warped.dtype)
        self.map = torch.where(
            self.pose_valid.view(-1, 1, 1, 1),
            warped_memory,
            torch.zeros_like(warped),
        )
        if self.dynamic_history is not None:
            warped_dynamic = F.grid_sample(
                self.dynamic_history, sample_grid, mode="bilinear", padding_mode="zeros", align_corners=True
            )
            self.dynamic_history = torch.where(
                self.pose_valid.view(-1, 1, 1, 1),
                (warped_dynamic > 0.5).to(warped_dynamic.dtype),
                torch.zeros_like(warped_dynamic),
            )

    def update(
        self,
        depth: torch.Tensor,
        camera_pos_w: torch.Tensor,
        camera_quat_w: torch.Tensor,
        intrinsics: torch.Tensor,
        root_pos_w: torch.Tensor,
        root_quat_w: torch.Tensor,
        return_components: bool = False,
        pixel_mask: torch.Tensor | None = None,
        exclude_masked_pixels_from_visibility: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Insert one delayed depth frame into the local visibility map.

        Inputs are batched over environments. The function performs the complete
        geometry pipeline:

        ``depth pixel -> camera-frame 3D -> world-frame 3D -> robot-local BEV cell``

        It returns the flattened actor observation ``[num_envs, grid_size^2]``.
        """
        # Accept either [B, 1, H, W] or [B, H, W]. Internally we use [B, H, W]
        # before flattening sampled pixels.
        depth = depth.to(self.device)
        if depth.dim() == 4:
            depth = depth[:, 0]
        num_envs, height, width = depth.shape
        if num_envs != self.num_envs:
            raise ValueError(f"Expected {self.num_envs} envs, got {num_envs}")
        self.current_dynamic_risk.zero_()

        pixel_mask_sub = None
        if pixel_mask is not None:
            pixel_mask = pixel_mask.to(self.device)
            if pixel_mask.dim() == 4:
                pixel_mask = pixel_mask[:, 0]
            if pixel_mask.shape != depth.shape:
                raise ValueError(
                    f"pixel_mask shape {tuple(pixel_mask.shape)} must match depth shape {tuple(depth.shape)}"
                )
            pixel_mask_sub = pixel_mask[:, :: self.pixel_stride, :: self.pixel_stride].reshape(
                num_envs, -1
            )

        # Keep all tensors on the map device and extract yaw-only robot heading.
        # Roll/pitch are deliberately ignored for BEV memory alignment.
        camera_pos_w = camera_pos_w.to(self.device)
        camera_quat_w = camera_quat_w.to(self.device)
        intrinsics = intrinsics.to(self.device)
        root_pos_w = root_pos_w.to(self.device)
        root_quat_w = root_quat_w.to(self.device)
        root_yaw = _quat_wxyz_to_yaw(root_quat_w)

        # First move previous memory into the current robot frame. Keep this
        # pre-update state for DynamicMapCross so its history K/V does not already
        # contain the current frame used by the query.
        self._warp_memory_to_current_frame(root_pos_w, root_yaw)
        history_before_update = self.map.clone() if return_components else None

        # Subsample the depth image to reduce update cost. The row/col grids are
        # the pixel centers that correspond to each selected depth value.
        depth_sub = depth[:, :: self.pixel_stride, :: self.pixel_stride].reshape(num_envs, -1)
        row_grid, col_grid = self._pixel_grid(height, width)
        valid_depth = torch.isfinite(depth_sub) & (depth_sub >= self.depth_min_m) & (depth_sub <= self.depth_max_m)

        # Back-project pinhole-camera depth to camera-frame 3D points.
        # This raycaster camera uses distance_to_image_plane; in this coordinate
        # convention the forward axis is x_cam. The signs for y/z match the
        # camera frame convention used by IsaacLab raycast camera intrinsics.
        fx = intrinsics[:, 0, 0].clamp_min(1.0e-6).unsqueeze(1)
        fy = intrinsics[:, 1, 1].clamp_min(1.0e-6).unsqueeze(1)
        cx = intrinsics[:, 0, 2].unsqueeze(1)
        cy = intrinsics[:, 1, 2].unsqueeze(1)
        x_cam = depth_sub
        y_cam = -((col_grid - cx) / fx) * depth_sub
        z_cam = -((row_grid - cy) / fy) * depth_sub
        points_cam = torch.stack((x_cam, y_cam, z_cam), dim=-1)

        # Camera frame -> world frame using delayed camera pose. It is important
        # that depth, camera pose, and intrinsics come from the same delayed
        # sample; otherwise moving/turning robots would smear the map.
        rotation = _quat_wxyz_to_matrix(camera_quat_w)
        points_w = torch.bmm(points_cam, rotation.transpose(1, 2)) + camera_pos_w.unsqueeze(1)
        finite_points = torch.isfinite(points_w).all(dim=-1)

        # World frame -> current robot yaw-aligned local frame. x_local points
        # forward, y_local points laterally in the robot frame.
        rel = points_w - root_pos_w.unsqueeze(1)
        cos_yaw = torch.cos(root_yaw).unsqueeze(1)
        sin_yaw = torch.sin(root_yaw).unsqueeze(1)
        x_local = cos_yaw * rel[..., 0] + sin_yaw * rel[..., 1]
        y_local = -sin_yaw * rel[..., 0] + cos_yaw * rel[..., 1]

        # Local meters -> BEV grid row/column. Points outside the square map or
        # outside valid depth range are ignored.
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

        # Scatter all valid depth hits into a flat [B, grid_size^2] update map.
        # scatter_reduce(..., reduce="amax") implements a vectorized OR: if any
        # depth point lands in a cell, that cell becomes visible.
        linear_idx = (row.clamp(0, self.grid_size - 1) * self.grid_size + col.clamp(0, self.grid_size - 1))
        updates = torch.zeros((self.num_envs, self.grid_size * self.grid_size), device=self.device)
        visibility_valid = valid
        if exclude_masked_pixels_from_visibility and pixel_mask_sub is not None:
            visibility_valid = valid & (pixel_mask_sub < 0.5)
        values = visibility_valid.to(dtype=updates.dtype)
        updates.scatter_reduce_(1, linear_idx, values, reduce="amax", include_self=True)

        dynamic_updates = None
        if pixel_mask_sub is not None:
            dynamic_updates = torch.zeros_like(updates)
            dynamic_values = (valid & (pixel_mask_sub >= 0.5)).to(dtype=updates.dtype)
            dynamic_updates.scatter_reduce_(
                1, linear_idx, dynamic_values, reduce="amax", include_self=True
            )

        # Preserve current-frame depth in the same BEV coordinates. Empty cells
        # are masked to zero; callers use current_hits to distinguish them.
        depth_values = ((depth_sub - self.depth_min_m) / (self.depth_max_m - self.depth_min_m)).clamp(0.0, 1.0)
        depth_values = torch.where(visibility_valid, depth_values, torch.ones_like(depth_values))
        current_depth = torch.ones_like(updates)
        current_depth.scatter_reduce_(1, linear_idx, depth_values, reduce="amin", include_self=True)

        current_hits = updates.view_as(self.map)
        current_depth = current_depth.view_as(self.map)
        current_depth = torch.where(current_hits > 0.0, current_depth, torch.zeros_like(current_depth))
        if self.score_mode:
            self.map.mul_(self.decay)
            self.map.add_(self.score_alpha_up * current_hits * (1.0 - self.map))
            self.map.clamp_(0.0, 1.0)
        else:
            self.map = torch.maximum(self.map, current_hits)

        self.last_root_pos_w.copy_(root_pos_w)
        self.last_root_yaw.copy_(root_yaw)
        self.pose_valid[:] = True
        if not self.score_mode:
            self.map = (self.map > 0.5).to(self.map.dtype)
        updated_history = self.map.flatten(start_dim=1)
        if not return_components:
            return updated_history
        assert history_before_update is not None
        components = (
            updated_history,
            history_before_update.flatten(start_dim=1),
            current_hits.flatten(start_dim=1),
            current_depth.flatten(start_dim=1),
        )
        if dynamic_updates is None:
            return components
        return components + (dynamic_updates,)


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


def _get_delayed_visual_packet(env, sensor_cfg) -> dict[str, torch.Tensor]:
    """Return one noisy, encoded, and delayed visual packet per camera step.

    Actor depth, critic depth, Current BEV, and History BEV must observe the same
    camera frame.  The complete packet is delayed by one buffer and cached by the
    environment step, which also prevents repeated observation-group evaluation
    from advancing that buffer more than once.
    """
    camera_name = sensor_cfg.name if hasattr(sensor_cfg, "name") else "raycast_camera"
    step = int(getattr(env, "common_step_counter", -1))
    cache = getattr(env, "_nav_delayed_visual_packet_cache", None)
    if cache is None:
        cache = {}
        env._nav_delayed_visual_packet_cache = cache
    cached = cache.get(camera_name)
    if cached is not None and cached["step"] == step:
        return cached["packet"]

    _ensure_depth_noise_generator_initialized()
    depth_camera = env.scene.sensors[sensor_cfg.name]
    height, width = depth_camera.image_shape
    depth_tensor = depth_camera.data.output["distance_to_image_plane"].view(env.num_envs, -1)
    depth_tensor = torch.nan_to_num(depth_tensor, nan=50.0, posinf=50.0, neginf=0.0)
    depth_tensor = depth_tensor.view(-1, 1, height, width)

    assert JIT_DEPTH_NOISE_GENERATOR is not None
    model = cast(torch.jit.ScriptModule, JIT_DEPTH_NOISE_GENERATOR)
    if use_jit:
        noisy_depth_tensor = depth_tensor.clone()
        noisy_depth_tensor[noisy_depth_tensor > max_depth] = 0.0
        noisy_depth_tensor[noisy_depth_tensor < min_depth] = 0.0
        encoded_depth_tensor = model(noisy_depth_tensor)
    else:
        encoded_depth_tensor, noisy_depth_tensor = model(depth_tensor)

    encoded_flat = encoded_depth_tensor.reshape(env.num_envs, -1)
    noisy_flat = noisy_depth_tensor.reshape(env.num_envs, -1)
    camera_pos_w = depth_camera.data.pos_w
    camera_quat_w = depth_camera.data.quat_w_world
    intrinsics_flat = depth_camera.data.intrinsic_matrices.reshape(env.num_envs, -1)
    packed = torch.cat((encoded_flat, noisy_flat, camera_pos_w, camera_quat_w, intrinsics_flat), dim=1).detach()
    delayed = env.delay_manager.compute_delayed_visual_packet(packed, camera_name)

    encoded_size = encoded_flat.shape[1]
    depth_size = noisy_flat.shape[1]
    offset = 0
    delayed_encoded = delayed[:, offset : offset + encoded_size]
    offset += encoded_size
    delayed_depth = delayed[:, offset : offset + depth_size].view(env.num_envs, 1, height, width)
    offset += depth_size
    delayed_camera_pos = delayed[:, offset : offset + 3]
    offset += 3
    delayed_camera_quat = delayed[:, offset : offset + 4]
    offset += 4
    delayed_intrinsics = delayed[:, offset : offset + 9].view(env.num_envs, 3, 3)

    packet = {
        "encoded_depth": delayed_encoded,
        "depth": delayed_depth,
        "camera_pos_w": delayed_camera_pos,
        "camera_quat_w": delayed_camera_quat,
        "intrinsics": delayed_intrinsics,
    }
    cache[camera_name] = {"step": step, "packet": packet}
    return packet


def _get_noisy_depth_sample(env, sensor_cfg) -> dict[str, torch.Tensor]:
    """Compatibility alias returning the shared noisy, delayed visual packet."""
    return _get_delayed_visual_packet(env, sensor_cfg)


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


class delayed_visualizable_image(ManagerTermBase):
    """Return delayed image-history frames from a noisy camera sensor.

    This is the local instinct_nav copy of the small InstinctLab observation term
    used by the frozen G1 parkour locomotion policy. The low-level checkpoint was
    trained with ``NoisyGroupedRayCasterCamera`` history data, so the observation
    must read from the sensor's history buffer instead of from the high-level SRU
    ``raycast_camera`` path.

    Expected sensor output shape is ``[num_envs, history, H, W, C]``. Current G1
    config uses ``distance_to_image_plane_noised_history`` with one channel, so
    this term squeezes the channel dimension and returns:

    ``[num_envs, num_output_frames, H, W]``.

    With the active low-level config that becomes ``[num_envs, 8, 18, 32]``.
    """

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.sensor_cfg = cfg.params.get("sensor_cfg", SceneEntityCfg("camera"))
        self.data_type = cfg.params["data_type"]
        if "history" not in self.data_type:
            raise ValueError("data_type must contain 'history' so delayed frames can be sampled")

        self.sensor = env.scene.sensors[self.sensor_cfg.name]
        self.delayed_frame_ranges = cfg.params.get("delayed_frame_ranges", (0, 0))
        self.delayed_frame_distribution: Literal["uniform", "log_uniform"] = cfg.params.get(
            "delayed_frame_distribution", "uniform"
        )
        self._num_delayed_frames = torch.zeros(env.num_envs, device=env.device)
        self.history_skip_frames = max(cfg.params.get("history_skip_frames", 1), 1)
        self.num_output_frames = max(cfg.params.get("num_output_frames", 0), 1)

        history_tensor = self.sensor.data.output[self.data_type]
        if len(history_tensor.shape) < 5:
            raise ValueError(
                f"sensor data {self.data_type!r} should have [N, history, H, W, C] shape, "
                f"got {tuple(history_tensor.shape)}"
            )
        self.sensor_history_length = history_tensor.shape[1]

        # frame_offset is ordered old -> new. Because the newest sensor frame is
        # stored at index -1, the first returned output frame uses the largest
        # offset and the last returned output frame is closest to the present.
        self.frame_offset = torch.flip(
            torch.arange(
                0,
                self.num_output_frames * self.history_skip_frames,
                self.history_skip_frames,
                device=env.device,
            ),
            dims=(0,),
        )
        self.check_delay_bounds()

    def check_delay_bounds(self) -> None:
        """Validate that requested delay/history settings fit in sensor history."""
        max_delayed_frames = self.delayed_frame_ranges[1]
        frames_needed_if_no_delay = (self.num_output_frames - 1) * self.history_skip_frames + 1
        total_needed = frames_needed_if_no_delay + max_delayed_frames
        if total_needed > self.sensor_history_length:
            raise ValueError(
                "The delayed frame ranges are too large for the sensor history length. "
                f"max_delay={max_delayed_frames}, frames_without_delay={frames_needed_if_no_delay}, "
                f"total_needed={total_needed}, history_length={self.sensor_history_length}."
            )

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Sample each environment's frame delay at episode reset."""
        if env_ids is None:
            env_ids = slice(None)
        self._num_delayed_frames[env_ids] = _randomize_prop_by_op(
            self._num_delayed_frames[env_ids].unsqueeze(-1),
            self.delayed_frame_ranges,
            None,
            slice(None),
            operation="abs",
            distribution=self.delayed_frame_distribution,
        ).squeeze(-1)

    def __call__(
        self,
        env: ManagerBasedEnv,
        data_type: str,
        sensor_cfg: SceneEntityCfg = SceneEntityCfg("camera"),
        history_skip_frames: int = 0,
        num_output_frames: int = 0,
        delayed_frame_ranges: tuple[int, int] = (0, 0),
        delayed_frame_distribution: Literal["uniform", "log_uniform"] = "uniform",
        debug_vis: bool = False,
        scale_up_vis: int = 5,
    ) -> torch.Tensor:
        """Read delayed frames from the configured sensor history buffer."""
        images = self.sensor.data.output[self.data_type].clone()
        images = images.squeeze(-1)

        frame_indices = (
            self.sensor_history_length - self.frame_offset.unsqueeze(0) - self._num_delayed_frames.unsqueeze(1) - 1
        ).to(torch.long)
        if not (frame_indices >= 0).all():
            raise RuntimeError(f"frame_indices should be non-negative, got {frame_indices}")
        if not (frame_indices < self.sensor_history_length).all():
            raise RuntimeError(
                f"frame_indices should be less than sensor history length {self.sensor_history_length}, "
                f"got {frame_indices}"
            )

        batch_indices = (
            torch.arange(images.shape[0], device=images.device)
            .unsqueeze(1)
            .expand(-1, frame_indices.shape[1])
            .to(torch.long)
        )
        delayed_frames = images[batch_indices, frame_indices]
        if debug_vis and env.num_envs == 1:
            _update_depth_window(
                self.sensor_cfg.name,
                delayed_frames[:, -1:].contiguous(),
                delayed_frames.shape[-2],
                delayed_frames.shape[-1],
                title=f"Delayed Image - {self.sensor_cfg.name}",
            )
        return delayed_frames


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
    """Return critic-only encoded height scan features.

    This is privileged terrain information for the value function. The actor does
    not receive it. The raw ray caster produces a 51x51 grid around the robot;
    the pre-trained heightscan VAE compresses it into ``[64, 7, 7]`` and the
    observation manager flattens it to ``3136`` features.

    Important distinction from visibility_map:

    - height_scan_feat is a height/terrain feature and is critic-only.
    - local_visibility_map is a binary observed/not-observed memory and is actor-side.

    Args:
        env: The environment object.
        sensor_cfg: The configuration of the height scanner sensor.
        offset: Offset subtracted from height values before encoding.

    Returns:
        Encoded height scan features with shape ``[num_envs, 3136]``.
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
    """Return the high-level action after IsaacLab action-term processing.

    In this project the trainable SRU policy does not directly output low-level
    joint targets. It outputs a navigation/action command consumed by an action
    term, and that action term may clamp, scale, or otherwise process the raw
    policy output. This observation exposes the processed action, which is useful
    when the actor/critic need action history or when logging reward terms such as
    action-rate penalties.

    Args:
        env: The environment object.
        action_name: The name of the action term.

    Returns:
        Processed high-level action tensor with shape determined by the selected
        action term.
    """
    return env.action_manager.get_term(action_name).processed_actions


def generated_commands_reshaped_delayed(
    env: ManagerBasedRLEnv,
    command_name: str,
    unsqueeze_pos: int = 1,
    flatten: bool = False,
) -> torch.Tensor:
    """Return a delayed command from the command manager.

    For navigation, the command is usually the target/goal representation used by
    the policy. Applying delay here makes command perception consistent with the
    delayed proprioception and delayed depth observations used by the actor.

    Args:
        env: The environment object (must have delay_manager attribute).
        command_name: The name of the command term.
        unsqueeze_pos: Position to unsqueeze the command when a sequence/channel
            dimension is expected by the observation group.
        flatten: If True, keep the command as ``[num_envs, command_dim]``.

    Returns:
        Delayed command tensor. Shape is either flat ``[num_envs, command_dim]``
        or unsqueezed according to ``unsqueeze_pos``.
    """
    if flatten:
        command = env.command_manager.get_command(command_name)
        return env.delay_manager.compute_delayed_target_position(command)

    command = env.command_manager.get_command(command_name).unsqueeze(unsqueeze_pos)
    return env.delay_manager.compute_delayed_target_position(command)


def last_low_level_action(
    env: ManagerBasedEnv, action_term: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Return the most recent frozen low-level locomotion action.

    The SRU policy is the high-level navigator. The G1 locomotion policy is
    frozen underneath it. This observation exposes the previous low-level joint
    action for the selected joints so the high-level policy/critic can infer how
    the frozen controller is currently behaving.

    Args:
        env: The environment object.
        action_term: The name of the action term.

    Returns:
        Low-level action tensor indexed by ``asset_cfg.joint_ids``.
    """
    action_term: PerceptiveNavigationSE2Action = env.action_manager._terms[action_term]
    return action_term.low_level_actions[:, asset_cfg.joint_ids]


@torch.inference_mode()
def depth_image_prefect(env, sensor_cfg):
    """Return perfect/non-noisy encoded depth features.

    The spelling ``prefect`` is kept for compatibility with existing config
    files. Conceptually this is ``perfect`` depth.

    This path is used where we want camera geometry without actor-side corruption:

    1. Read IsaacLab ray-caster depth ``distance_to_image_plane``.
    2. Replace NaN/Inf values with finite sentinels.
    3. Reshape raw depth from flattened camera output to ``[B, 1, H, W]``.
    4. Apply the frozen depth encoder.
    5. Flatten encoded features before handing them to the observation manager.

    No random noise and no delay are applied here. In contrast,
    ``depth_image_noisy_delayed`` is the actor-facing observation used during
    SRU training.

    Args:
        env: The environment object.
        sensor_cfg: The sensor configuration.

    Returns:
        Flattened encoded depth features with shape ``[num_envs, 64 * 5 * 8]``
        for the current G1 camera/encoder setup.
    """
    # The depth encoder is a singleton. This call is cheap after first use and
    # protects play/eval scripts that may not initialize it in config __post_init__.
    _ensure_depth_noise_generator_initialized()

    depth_camera = env.scene.sensors[sensor_cfg.name]

    # IsaacLab ray-caster output is ``[B, H, W, 1]`` or equivalent contiguous
    # camera layout depending on sensor internals. The encoder expects
    # ``[B, 1, H, W]``, so we first flatten per environment and then reshape.
    depth_tensor = depth_camera.data.output["distance_to_image_plane"].view(env.num_envs, -1)

    # Invalid misses may appear as NaN or Inf. Large positive values are outside
    # the valid depth range and are later zeroed in JIT mode.
    depth_tensor = torch.nan_to_num(depth_tensor, nan=50.0, posinf=50.0, neginf=0.0)

    # Camera ``image_shape`` is ``(H, W)`` in IsaacLab sensor data.
    H, W = depth_camera.image_shape
    depth_tensor = depth_tensor.view(-1, 1, H, W)

    assert JIT_DEPTH_NOISE_GENERATOR is not None, (
        "Depth encoder JIT model is not initialized. Call initialize_depth_noise_generator first."
    )
    model = cast(torch.jit.ScriptModule, JIT_DEPTH_NOISE_GENERATOR)
    if use_jit:
        # The JIT encoder path only returns encoded features, so invalid depth is
        # filtered here before inference.
        depth_tensor[depth_tensor > max_depth] = 0.0
        encoded_depth_tensor = model(depth_tensor)
    else:
        # Non-JIT DepthNoiseEncoder returns ``(encoded, noisy_depth)``. The
        # perfect path ignores the noisy image and only keeps encoded features.
        encoded_depth_tensor, _ = model(depth_tensor)

    return encoded_depth_tensor.view(env.num_envs, -1)


@torch.no_grad()
def depth_image_noisy_delayed(
    env, sensor_cfg, visualize: bool = False
):
    """Return actor-facing noisy and delayed encoded depth features.

    This is the main visual observation consumed by the SRU actor. The intended
    data flow is:

    1. Read raw depth from the ray-caster camera.
    2. Replace invalid values and reshape to ``[B, 1, H, W]``.
    3. Apply depth preprocessing/noise/encoding.
    4. Cache the noisy raw depth sample for ``local_visibility_map``.
    5. Flatten the encoded feature map.
    6. Apply per-env observation delay to the encoded feature vector.

    The depth encoder output is conceptually ``[B, 64, 5, 8]``. The observation
    manager receives the flattened version ``[B, 2560]``. ``ActorCriticSRU`` then
    slices this block according to the policy observation layout.

    Args:
        env: The environment object (must have delay_manager attribute).
        sensor_cfg: The sensor configuration.
        visualize: Whether to visualize the depth image.

    Returns:
        Delayed flattened encoded depth features with shape
        ``[num_envs, 64 * 5 * 8]``.
    """
    packet = _get_delayed_visual_packet(env, sensor_cfg)

    # Continuous visualization (single-env) with persistent windows
    if visualize and env.num_envs == 1 and not use_jit:
        camera_name = sensor_cfg.name if hasattr(sensor_cfg, "name") else "depth"
        height, width = packet["depth"].shape[-2:]
        _update_depth_window(
            camera_name, packet["depth"], height, width, title=f"Depth (Noisy Delayed) - {camera_name}"
        )

    return packet["encoded_depth"]


@torch.no_grad()
def raw_depth_delayed(env, sensor_cfg):
    """Return the raw noisy+delayed depth image, flattened.

    This is the pixel-space depth (not the encoded VAE features) that the
    dynamic-obstacle segmentation head consumes. It comes from the same delayed
    visual packet as the encoded depth, so it is temporally aligned with the
    actor's encoded-depth observation and with the mask label.

    Returns:
        Flattened depth ``[num_envs, H * W]`` where ``(H, W)`` is the camera
        ``image_shape`` (currently 36 x 64 for the shared G1 camera).
    """
    packet = _get_delayed_visual_packet(env, sensor_cfg)
    return packet["depth"].flatten(start_dim=1)


@torch.no_grad()
def dynamic_obstacle_mask_label(
    env,
    sensor_cfg,
    half_extent_m: float | None = None,
    half_height_m: float | None = None,
    depth_tol_m: float = 0.35,
    return_instance_visibility: bool = False,
):
    """Generate a training-only ground-truth dynamic-obstacle segmentation mask.

    Projects each axis-aligned obstacle cuboid (world ``positions_w`` + half
    extents) into the *delayed* camera frame and rasterizes its 2D silhouette,
    marking a pixel only when the obstacle is the nearest surface (a simple
    depth-consistency occlusion test against the delayed raw depth). The output
    matches the raw depth resolution. In simulation it supplies the mask input
    and, for the legacy prediction task, the auxiliary segmentation label.

    Args:
        env: The environment (must have ``_dynamic_obstacle_manager``).
        sensor_cfg: The depth sensor configuration (same one as the depth obs).
        half_extent_m: Optional XY half-width override; defaults to the spawned cuboid size.
        half_height_m: Optional Z half-height override; defaults to the spawned cuboid size.
        depth_tol_m: Tolerance for the depth-consistency occlusion check.

    Returns:
        Flattened binary mask ``[num_envs, H * W]``.
    """
    manager = getattr(env, "_dynamic_obstacle_manager", None)
    packet = _get_delayed_visual_packet(env, sensor_cfg)
    depth = packet["depth"]  # [B, 1, H, W]
    B, _, H, W = depth.shape
    mask = torch.zeros(B, 1, H, W, device=env.device, dtype=torch.float32)

    if manager is None:
        if return_instance_visibility:
            return mask.flatten(start_dim=1), torch.zeros(B, 0, device=env.device, dtype=torch.bool)
        return mask.flatten(start_dim=1)

    positions = getattr(manager, "positions_w", None)  # [B, K, 3]
    if positions is None or positions.shape[1] == 0:
        if return_instance_visibility:
            return mask.flatten(start_dim=1), torch.zeros(B, 0, device=env.device, dtype=torch.bool)
        return mask.flatten(start_dim=1)
    K = positions.shape[1]
    if half_extent_m is None:
        half_extent_m = manager.body_half_extent
    if half_height_m is None:
        half_height_m = manager.obstacle_half_height

    # Align obstacle state with the delayed camera frame: index the manager's
    # per-step state history (positions_w + activation state) by the same visual lag as the
    # delayed depth/camera packet. This is exact across spawn / direction-reversal
    # / stop-and-go transitions (a velocity extrapolation would miss them).
    positions = positions.clone()
    spawned = getattr(manager, "spawned_obstacles", None)
    if spawned is None:
        spawned = getattr(manager, "spawned", torch.zeros(B, dtype=torch.bool, device=env.device))
    spawned = spawned.clone()
    curriculum = getattr(manager, "curriculum_envs", torch.ones(B, dtype=torch.bool, device=env.device)).clone()
    lag = torch.zeros(B, device=env.device, dtype=torch.long)
    delay_manager = getattr(env, "delay_manager", None)
    if delay_manager is not None and getattr(delay_manager, "enabled", False):
        buffers = getattr(delay_manager, "_delay_depth_buffers", {})
        camera_name = sensor_cfg.name if hasattr(sensor_cfg, "name") else None
        state = buffers.get(f"{camera_name}_visual_packet") if camera_name is not None else None
        if state is None and len(buffers) == 1:
            state = next(iter(buffers.values()))
        if state is not None and getattr(state, "time_lags", None) is not None:
            lag = state.time_lags.to(env.device)
    pos_history = getattr(manager, "_positions_history", None)
    spawned_history = getattr(manager, "_spawned_history", None)
    curriculum_history = getattr(manager, "_curriculum_envs_history", None)
    if pos_history and spawned_history and curriculum_history and bool(lag.any().item()):
        # history[-1] is the current step; history[-(d+1)] is `d` steps ago.
        for d in range(1, len(pos_history)):
            mask_d = lag == d
            if bool(mask_d.any().item()):
                positions[mask_d] = pos_history[-(d + 1)][mask_d]
                spawned[mask_d] = spawned_history[-(d + 1)][mask_d]
                curriculum[mask_d] = curriculum_history[-(d + 1)][mask_d]

    # Only spawned obstacles in curriculum envs are active; the rest stay parked
    # off-camera and contribute nothing to the label.
    if spawned.ndim == 1:
        spawned = spawned.unsqueeze(1).expand(B, K)
    active_k = spawned.bool() & curriculum.bool().unsqueeze(1)  # [B, K]

    fx = packet["intrinsics"][:, 0, 0].clamp_min(1.0e-6)
    fy = packet["intrinsics"][:, 1, 1].clamp_min(1.0e-6)
    cx = packet["intrinsics"][:, 0, 2]
    cy = packet["intrinsics"][:, 1, 2]

    # Axis-aligned cuboid corners (identity orientation). The rotation matrix R
    # maps camera->world (see LocalVisibilityMapManager.update), so world->camera
    # is ``(p_w - cam_pos) @ R``.
    corner_signs = torch.tensor(
        [
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [-1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        device=env.device,
        dtype=positions.dtype,
    )  # [8, 3]
    half = torch.tensor([half_extent_m, half_extent_m, half_height_m], device=env.device, dtype=positions.dtype)
    R = _quat_wxyz_to_matrix(packet["camera_quat_w"])  # [B, 3, 3]
    cam_pos = packet["camera_pos_w"]  # [B, 3]

    corners_w = positions.unsqueeze(2) + (corner_signs * half).unsqueeze(0).unsqueeze(0)  # [B, K, 8, 3]
    corners_cam = torch.einsum("bkna,bac->bknc", corners_w - cam_pos.unsqueeze(1).unsqueeze(1), R)  # [B, K, 8, 3]
    xc = corners_cam[..., 0]  # forward depth
    yc = corners_cam[..., 1]
    zc = corners_cam[..., 2]

    col = cx.unsqueeze(1).unsqueeze(1) - fx.unsqueeze(1).unsqueeze(1) * yc / xc.clamp_min(1.0e-3)
    row = cy.unsqueeze(1).unsqueeze(1) - fy.unsqueeze(1).unsqueeze(1) * zc / xc.clamp_min(1.0e-3)

    # Obstacle is valid only if every corner is in front of the camera.
    valid = (xc > 0.05).all(dim=-1) & active_k  # [B, K]
    front_depth = xc.amin(dim=-1)  # [B, K] closest corner depth (front face proxy)

    # Rasterize each obstacle's projected bounding box. Keep the small obstacle loop,
    # but process every environment and pixel in one CUDA broadcast. This avoids the
    # O(K * B) Python loop and thousands of .item() synchronizations at 4096 envs.
    pixel_rows = torch.arange(H, device=env.device).view(1, H, 1)
    pixel_cols = torch.arange(W, device=env.device).view(1, 1, W)
    mask_2d = mask[:, 0]
    instance_visibility = torch.zeros(B, K, device=env.device, dtype=torch.bool)
    for k in range(K):
        raw_r_min = row[:, k].amin(dim=-1)
        raw_r_max = row[:, k].amax(dim=-1)
        raw_c_min = col[:, k].amin(dim=-1)
        raw_c_max = col[:, k].amax(dim=-1)
        intersects_image = (
            (raw_r_max >= 0)
            & (raw_r_min <= H - 1)
            & (raw_c_max >= 0)
            & (raw_c_min <= W - 1)
        )
        r_min = raw_r_min.clamp(0, H - 1).floor().long().view(B, 1, 1)
        r_max = raw_r_max.clamp(0, H - 1).ceil().long().view(B, 1, 1)
        c_min = raw_c_min.clamp(0, W - 1).floor().long().view(B, 1, 1)
        c_max = raw_c_max.clamp(0, W - 1).ceil().long().view(B, 1, 1)
        inside_bbox = (
            (pixel_rows >= r_min)
            & (pixel_rows <= r_max)
            & (pixel_cols >= c_min)
            & (pixel_cols <= c_max)
        )
        depth_consistent = depth[:, 0] >= (front_depth[:, k].view(B, 1, 1) - depth_tol_m)
        obstacle_mask = (
            valid[:, k].view(B, 1, 1)
            & intersects_image.view(B, 1, 1)
            & inside_bbox
            & depth_consistent
        )
        instance_visibility[:, k] = obstacle_mask.flatten(start_dim=1).any(dim=1)
        mask_2d.logical_or_(obstacle_mask)

    flat_mask = mask.flatten(start_dim=1)
    if return_instance_visibility:
        return flat_mask, instance_visibility
    return flat_mask


def _delayed_dynamic_state_for_risk(env, sensor_cfg):
    """Return obstacle position/velocity synchronized with the delayed camera packet."""
    manager = getattr(env, "_dynamic_obstacle_manager", None)
    if manager is None:
        return None
    positions = manager.positions_w.clone()
    spawned = manager.spawned_obstacles.clone()
    curriculum = manager.curriculum_envs.clone()
    velocity = manager.dirs * (
        manager.speeds
        * (~manager.paused).to(manager.speeds.dtype)
        * manager.spawned_obstacles.to(manager.speeds.dtype)
    ).unsqueeze(-1)

    lag = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    delay_manager = getattr(env, "delay_manager", None)
    if delay_manager is not None and getattr(delay_manager, "enabled", False):
        buffers = getattr(delay_manager, "_delay_depth_buffers", {})
        camera_name = getattr(sensor_cfg, "name", None)
        state = buffers.get(f"{camera_name}_visual_packet") if camera_name is not None else None
        if state is None and len(buffers) == 1:
            state = next(iter(buffers.values()))
        if state is not None and getattr(state, "time_lags", None) is not None:
            lag = state.time_lags.to(env.device)

    pos_history = getattr(manager, "_positions_history", None)
    spawned_history = getattr(manager, "_spawned_history", None)
    curriculum_history = getattr(manager, "_curriculum_envs_history", None)
    velocity_history = getattr(manager, "_velocities_history", None)
    if pos_history and spawned_history and curriculum_history and velocity_history:
        history_len = min(
            len(pos_history), len(spawned_history), len(curriculum_history), len(velocity_history)
        )
        for delay in range(1, history_len):
            use_history = lag == delay
            if bool(use_history.any().item()):
                positions[use_history] = pos_history[-(delay + 1)][use_history]
                spawned[use_history] = spawned_history[-(delay + 1)][use_history]
                curriculum[use_history] = curriculum_history[-(delay + 1)][use_history]
                velocity[use_history] = velocity_history[-(delay + 1)][use_history]
    active = spawned.bool() & curriculum.bool().unsqueeze(1)
    return positions, velocity, active


def _render_dynamic_sector_risk(
    positions_w: torch.Tensor,
    velocities_w: torch.Tensor,
    active: torch.Tensor,
    visible: torch.Tensor,
    root_pos_w: torch.Tensor,
    root_quat_w: torch.Tensor,
    *,
    size_m: float,
    grid_size: int,
    body_half_extent_m: float,
    clearance_m: float,
    horizon_s: float,
    sigma_long_m: float,
    sigma_lat_m: float,
    softness_m: float,
    stationary_speed_mps: float,
) -> torch.Tensor:
    """Rasterize current occupancy plus a velocity-aligned soft risk sector."""
    batch, obstacle_count = positions_w.shape[:2]
    device = positions_w.device
    dtype = positions_w.dtype
    yaw = _quat_wxyz_to_yaw(root_quat_w)
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)

    relative = positions_w[..., :2] - root_pos_w[:, None, :2]
    center_x = cos_yaw[:, None] * relative[..., 0] + sin_yaw[:, None] * relative[..., 1]
    center_y = -sin_yaw[:, None] * relative[..., 0] + cos_yaw[:, None] * relative[..., 1]
    velocity_x = cos_yaw[:, None] * velocities_w[..., 0] + sin_yaw[:, None] * velocities_w[..., 1]
    velocity_y = -sin_yaw[:, None] * velocities_w[..., 0] + cos_yaw[:, None] * velocities_w[..., 1]

    cell_size = float(size_m) / int(grid_size)
    half_size = 0.5 * float(size_m)
    axis = half_size - (torch.arange(grid_size, device=device, dtype=dtype) + 0.5) * cell_size
    grid_x = axis.view(1, grid_size, 1)
    grid_y = axis.view(1, 1, grid_size)
    risk = torch.zeros(batch, 1, grid_size, grid_size, device=device, dtype=dtype)
    base_radius = float(body_half_extent_m) + float(clearance_m)
    softness = max(float(softness_m), 1.0e-3)

    for obstacle_index in range(obstacle_count):
        enabled = active[:, obstacle_index] & visible[:, obstacle_index]
        if not bool(enabled.any().item()):
            continue
        dx = grid_x - center_x[:, obstacle_index, None, None]
        dy = grid_y - center_y[:, obstacle_index, None, None]
        speed = torch.sqrt(
            velocity_x[:, obstacle_index].square() + velocity_y[:, obstacle_index].square()
        )
        unit_x = velocity_x[:, obstacle_index] / speed.clamp_min(1.0e-6)
        unit_y = velocity_y[:, obstacle_index] / speed.clamp_min(1.0e-6)
        along = dx * unit_x[:, None, None] + dy * unit_y[:, None, None]
        lateral = torch.abs(-dx * unit_y[:, None, None] + dy * unit_x[:, None, None])
        travel = speed * float(horizon_s) + float(sigma_long_m)
        progress = (along / travel[:, None, None].clamp_min(1.0e-3)).clamp(0.0, 1.0)
        width = base_radius + float(sigma_lat_m) * (1.0 + progress)
        longitudinal_gate = torch.sigmoid((along + base_radius) / softness)
        longitudinal_gate *= torch.sigmoid((travel[:, None, None] + base_radius - along) / softness)
        lateral_gate = torch.sigmoid((width - lateral) / softness)
        sector = longitudinal_gate * lateral_gate
        current_disk = torch.sigmoid(
            (base_radius - torch.sqrt(dx.square() + dy.square())) / softness
        )
        moving = speed >= float(stationary_speed_mps)
        obstacle_risk = torch.where(moving[:, None, None], torch.maximum(sector, current_disk), current_disk)
        obstacle_risk *= enabled[:, None, None].to(dtype)
        risk[:, 0] = torch.maximum(risk[:, 0], obstacle_risk)
    return risk.clamp_(0.0, 1.0)


@torch.no_grad()
def dynamic_obstacle_mask_observation(
    env,
    sensor_cfg,
    output_height: int = 9,
    output_width: int = 16,
    apply_corruption: bool = True,
    frame_dropout_prob: float = 0.08,
    pixel_dropout_prob: float = 0.05,
    false_positive_prob: float = 0.001,
    erode_prob: float = 0.10,
    dilate_prob: float = 0.15,
    max_shift_pixels: int = 1,
    include_masked_depth: bool = False,
    force_zero: bool = False,
    return_instance_visibility: bool = False,
):
    """Return a compact dynamic-obstacle mask for direct policy input.

    Simulation ground truth is projected into the same delayed camera frame as
    the depth observation, then max-pooled to a compact grid. During training,
    simple mask-space corruption approximates deployment-time segmentation
    failures: latency is already provided by the delayed visual packet, while
    this function adds frame drops, holes, false positives, boundary changes,
    and spatial jitter.

    The result is cached per environment step so actor and critic observe the
    same corrupted mask.
    """
    step = int(getattr(env, "common_step_counter", -1))
    camera_name = sensor_cfg.name if hasattr(sensor_cfg, "name") else "camera"
    obstacle_manager = getattr(env, "_dynamic_obstacle_manager", None)
    obstacle_positions = getattr(obstacle_manager, "positions_w", None)
    obstacle_count = int(obstacle_positions.shape[1]) if obstacle_positions is not None else 0
    cache_key = (
        step,
        camera_name,
        obstacle_count,
        int(output_height),
        int(output_width),
        bool(apply_corruption),
        float(frame_dropout_prob),
        float(pixel_dropout_prob),
        float(false_positive_prob),
        float(erode_prob),
        float(dilate_prob),
        int(max_shift_pixels),
        bool(include_masked_depth),
        bool(force_zero),
    )
    cache = getattr(env, "_instinct_nav_direct_mask_cache", None)
    if cache is not None and cache.get("key") == cache_key:
        manager = getattr(env, "_dynamic_obstacle_manager", None)
        if manager is not None:
            manager._last_direct_mask_clean_positive_rate = cache["clean_positive_rate"]
            manager._last_direct_mask_input_positive_rate = cache["input_positive_rate"]
        if return_instance_visibility:
            return cache["mask"], cache["instance_visibility"]
        return cache["mask"]

    packet = _get_delayed_visual_packet(env, sensor_cfg)
    batch_size = packet["depth"].shape[0]
    input_height, input_width = packet["depth"].shape[-2:]
    flat_mask, instance_visibility = dynamic_obstacle_mask_label(
        env, sensor_cfg, return_instance_visibility=True
    )
    mask = flat_mask.reshape(batch_size, 1, input_height, input_width)
    mask = F.adaptive_max_pool2d(mask, (int(output_height), int(output_width)))
    clean_positive_rate = mask.mean().detach()

    if apply_corruption:
        out_h, out_w = mask.shape[-2:]

        if max_shift_pixels > 0:
            shift_y = torch.randint(
                -max_shift_pixels, max_shift_pixels + 1, (batch_size,), device=env.device
            )
            shift_x = torch.randint(
                -max_shift_pixels, max_shift_pixels + 1, (batch_size,), device=env.device
            )
            rows = torch.arange(out_h, device=env.device).view(1, out_h, 1) - shift_y.view(-1, 1, 1)
            cols = torch.arange(out_w, device=env.device).view(1, 1, out_w) - shift_x.view(-1, 1, 1)
            valid = (rows >= 0) & (rows < out_h) & (cols >= 0) & (cols < out_w)
            source = rows.clamp(0, out_h - 1) * out_w + cols.clamp(0, out_w - 1)
            source = source.expand(-1, out_h, out_w).reshape(batch_size, 1, -1)
            mask = mask.flatten(2).gather(2, source).reshape(batch_size, 1, out_h, out_w)
            mask = mask * valid.unsqueeze(1)

        morphology_choice = torch.rand(batch_size, 1, 1, 1, device=env.device)
        eroded = 1.0 - F.max_pool2d(1.0 - mask, kernel_size=3, stride=1, padding=1)
        dilated = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
        mask = torch.where(
            morphology_choice < erode_prob,
            eroded,
            torch.where(morphology_choice < erode_prob + dilate_prob, dilated, mask),
        )

        if pixel_dropout_prob > 0.0:
            keep = torch.rand_like(mask) >= pixel_dropout_prob
            mask = mask * keep
        if false_positive_prob > 0.0:
            false_positive_seed = (torch.rand_like(mask) < false_positive_prob).float()
            false_positive = F.max_pool2d(false_positive_seed, kernel_size=3, stride=1, padding=1)
            mask = torch.maximum(mask, false_positive)
        if frame_dropout_prob > 0.0:
            keep_frame = (
                torch.rand(batch_size, 1, 1, 1, device=env.device) >= frame_dropout_prob
            )
            mask = mask * keep_frame

    result = (mask >= 0.5).float().flatten(start_dim=1)
    input_positive_rate = result.mean().detach()
    if include_masked_depth:
        depth = packet["depth"].reshape(batch_size, 1, input_height, input_width)
        depth = F.adaptive_avg_pool2d(depth, (int(output_height), int(output_width)))
        valid = torch.isfinite(depth) & (depth > 0.0)
        depth = torch.where(valid, depth.clamp(0.0, 8.0) / 8.0, 0.0)
        binary = result.reshape(batch_size, 1, int(output_height), int(output_width))
        result = torch.cat((binary, binary * depth), dim=1).flatten(1)
    if force_zero:
        # Preserve the normal mask path and its RNG consumption so a seeded
        # ablation changes only the tensor presented to the network.
        result = torch.zeros_like(result)
        input_positive_rate = result.new_tensor(0.0)
    manager = getattr(env, "_dynamic_obstacle_manager", None)
    if manager is not None:
        manager._last_direct_mask_clean_positive_rate = clean_positive_rate
        manager._last_direct_mask_input_positive_rate = input_positive_rate
    env._instinct_nav_direct_mask_cache = {
        "key": cache_key,
        "mask": result,
        "clean_positive_rate": clean_positive_rate,
        "input_positive_rate": input_positive_rate,
        "instance_visibility": instance_visibility,
    }
    if return_instance_visibility:
        return result, instance_visibility
    return result


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
    """Return a robot-centric visibility memory built from noisy, delayed depth.

    ``decay=1`` preserves a binary seen/unseen map; ``decay<1`` returns confidence scores.

    The delayed raw sample packs depth, camera pose, and intrinsics together so the
    back-projection uses a temporally consistent camera frame. The map contains no
    privileged terrain height or danger labels.

    Why pack pose/intrinsics together with depth?

    - If only depth were delayed, the map would combine an old image with the
      current camera pose, producing spatially wrong points.
    - Packing ``depth + camera_pos + camera_quat + intrinsics`` into one vector
      and passing it through ``compute_delayed_depth`` makes all fields share the
      same sampled delay for each environment.

    Returned shape is ``[num_envs, grid_size * grid_size]``. With defaults this
    is ``[num_envs, 4096]``.
    """
    # All visual consumers share this exact delayed packet and capture pose.
    sample = _get_delayed_visual_packet(env, sensor_cfg)
    depth = sample["depth"]

    robot = env.scene[asset_cfg.name]
    # The manager owns map memory across steps and handles reset/warp/update.
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
        depth,
        sample["camera_pos_w"],
        sample["camera_quat_w"],
        sample["intrinsics"],
        robot.data.root_pos_w,
        robot.data.root_quat_w,
    )


def scene_tokens_from_visibility_map(
    env,
    sensor_cfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    goal_cmd_name: str = "robot_goal",
    size_m: float = 6.4,
    grid_size: int = 64,
    token_grid_size: int = 8,
    depth_min_m: float = 0.5,
    depth_max_m: float = 8.0,
    pixel_stride: int = 2,
    decay: float = 1.0,
) -> torch.Tensor:
    """Return fixed local 3D scene tokens derived from the visibility map.

    This is the first non-PQ3D token provider. It converts the actor-side binary
    visibility memory into a compact token set so ``ActorCriticSRU`` can learn the
    same token-attention interface that will later consume PQ3D object/frontier
    embeddings.

    Current token layout with default ``token_grid_size=8``:

    - logical token shape: ``[num_envs, 64, 6]``
    - observation-manager shape: ``[num_envs, 384]``
    - per-token features: ``[x_center, y_center, visible_ratio, dist_to_goal, cos_to_goal, sin_to_goal]``

    Coordinates are robot-local and normalized by map size. ``visible_ratio`` is
    the average binary visibility value inside the corresponding BEV patch. The
    tensor is flattened before returning because IsaacLab concatenated observation
    groups require every term to be one-dimensional per environment.
    """
    visibility_flat = local_visibility_map(
        env,
        sensor_cfg=sensor_cfg,
        asset_cfg=asset_cfg,
        size_m=size_m,
        grid_size=grid_size,
        depth_min_m=depth_min_m,
        depth_max_m=depth_max_m,
        pixel_stride=pixel_stride,
        decay=decay,
    )
    visibility = visibility_flat.view(env.num_envs, 1, grid_size, grid_size)
    patch_size = grid_size // token_grid_size
    if patch_size * token_grid_size != grid_size:
        raise ValueError(f"grid_size={grid_size} must be divisible by token_grid_size={token_grid_size}")

    # Average visibility over each BEV patch: [B, 1, 64, 64] -> [B, 8, 8] -> [B, 64].
    visible_ratio = F.avg_pool2d(visibility, kernel_size=patch_size, stride=patch_size).flatten(start_dim=1)

    device = visibility.device
    dtype = visibility.dtype
    rows = torch.arange(token_grid_size, device=device, dtype=dtype)
    cols = torch.arange(token_grid_size, device=device, dtype=dtype)
    row_grid, col_grid = torch.meshgrid(rows, cols, indexing="ij")

    # Token centers in robot-local meters, following LocalVisibilityMapManager's
    # convention: x is forward, y is lateral. Normalize by half map size so values
    # stay roughly in [-1, 1].
    patch_m = size_m / float(token_grid_size)
    half_size = size_m / 2.0
    x_center_m = half_size - (row_grid + 0.5) * patch_m
    y_center_m = half_size - (col_grid + 0.5) * patch_m
    x_center = (x_center_m / half_size).reshape(1, -1).expand(env.num_envs, -1)
    y_center = (y_center_m / half_size).reshape(1, -1).expand(env.num_envs, -1)

    goal = env.command_manager.get_command(goal_cmd_name)
    if goal.shape[-1] < 2:
        goal_xy = torch.zeros((env.num_envs, 2), device=device, dtype=dtype)
    else:
        goal_xy = goal[:, :2].to(device=device, dtype=dtype)
    goal_x = goal_xy[:, 0:1]
    goal_y = goal_xy[:, 1:2]

    dx = goal_x - x_center_m.reshape(1, -1)
    dy = goal_y - y_center_m.reshape(1, -1)
    dist = torch.sqrt(dx * dx + dy * dy + 1.0e-6)
    dist_norm = (dist / max(size_m, 1.0e-6)).clamp(0.0, 4.0)
    cos_to_goal = dx / dist
    sin_to_goal = dy / dist

    tokens = torch.stack(
        (x_center, y_center, visible_ratio, dist_norm, cos_to_goal, sin_to_goal),
        dim=-1,
    )
    return tokens.reshape(env.num_envs, -1)



def scene_tokens_risk_from_visibility_map(
    env,
    sensor_cfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    goal_cmd_name: str = "robot_goal",
    size_m: float = 6.4,
    grid_size: int = 64,
    token_grid_size: int = 8,
    depth_min_m: float = 0.5,
    depth_max_m: float = 8.0,
    pixel_stride: int = 2,
    decay: float = 1.0,
) -> torch.Tensor:
    """Return 9D scene tokens with lightweight terrain-risk cues.

    Layout: [x, y, visible_ratio, dist_to_goal, cos_to_goal, sin_to_goal,
    depth_mean, depth_edge, frontier]. The last three features are actor-safe:
    they come from the same raycast camera and local visibility memory, not from
    the critic-only height scanner.
    """
    cache_key = (
        "scene_tokens_risk_from_visibility_map",
        getattr(sensor_cfg, "name", "raycast_camera"),
        int(getattr(env, "common_step_counter", -1)),
        int(grid_size),
        int(token_grid_size),
        float(size_m),
        float(depth_min_m),
        float(depth_max_m),
        int(pixel_stride),
        float(decay),
    )
    cache = getattr(env, "_instinct_nav_scene_token_cache", None)
    if cache is not None and cache.get("key") == cache_key:
        return cache["tokens"]

    visibility_flat = local_visibility_map(
        env,
        sensor_cfg=sensor_cfg,
        asset_cfg=asset_cfg,
        size_m=size_m,
        grid_size=grid_size,
        depth_min_m=depth_min_m,
        depth_max_m=depth_max_m,
        pixel_stride=pixel_stride,
        decay=decay,
    )
    visibility = visibility_flat.view(env.num_envs, 1, grid_size, grid_size)
    patch_size = grid_size // token_grid_size
    if patch_size * token_grid_size != grid_size:
        raise ValueError(f"grid_size={grid_size} must be divisible by token_grid_size={token_grid_size}")

    visible_ratio = F.avg_pool2d(visibility, kernel_size=patch_size, stride=patch_size).flatten(start_dim=1)

    device = visibility.device
    dtype = visibility.dtype
    rows = torch.arange(token_grid_size, device=device, dtype=dtype)
    cols = torch.arange(token_grid_size, device=device, dtype=dtype)
    row_grid, col_grid = torch.meshgrid(rows, cols, indexing="ij")
    patch_m = size_m / float(token_grid_size)
    half_size = size_m / 2.0
    x_center_m = half_size - (row_grid + 0.5) * patch_m
    y_center_m = half_size - (col_grid + 0.5) * patch_m
    x_center = (x_center_m / half_size).reshape(1, -1).expand(env.num_envs, -1)
    y_center = (y_center_m / half_size).reshape(1, -1).expand(env.num_envs, -1)

    goal = env.command_manager.get_command(goal_cmd_name)
    if goal.shape[-1] < 2:
        goal_xy = torch.zeros((env.num_envs, 2), device=device, dtype=dtype)
    else:
        goal_xy = goal[:, :2].to(device=device, dtype=dtype)
    dx = goal_xy[:, 0:1] - x_center_m.reshape(1, -1)
    dy = goal_xy[:, 1:2] - y_center_m.reshape(1, -1)
    dist = torch.sqrt(dx * dx + dy * dy + 1.0e-6)
    dist_norm = (dist / max(size_m, 1.0e-6)).clamp(0.0, 4.0)
    cos_to_goal = dx / dist
    sin_to_goal = dy / dist

    sample = _get_noisy_depth_sample(env, sensor_cfg)
    depth = sample["depth"]
    if depth.dim() == 4:
        depth_image = depth[:, 0]
    else:
        height, width = env.scene.sensors[sensor_cfg.name].image_shape
        depth_image = depth.view(env.num_envs, height, width)
    valid_depth = torch.isfinite(depth_image) & (depth_image > depth_min_m) & (depth_image < depth_max_m)
    depth_norm = torch.where(
        valid_depth,
        ((depth_image - depth_min_m) / max(depth_max_m - depth_min_m, 1.0e-6)).clamp(0.0, 1.0),
        torch.ones_like(depth_image),
    )
    depth_mean = F.adaptive_avg_pool2d(depth_norm.unsqueeze(1), (token_grid_size, token_grid_size)).flatten(start_dim=1)
    grad_x = F.pad((depth_norm[..., 1:] - depth_norm[..., :-1]).abs(), (0, 1, 0, 0))
    grad_y = F.pad((depth_norm[..., 1:, :] - depth_norm[..., :-1, :]).abs(), (0, 0, 0, 1))
    depth_edge = F.adaptive_avg_pool2d((grad_x + grad_y).unsqueeze(1), (token_grid_size, token_grid_size)).flatten(start_dim=1)
    depth_edge = depth_edge.clamp(0.0, 1.0)

    visibility_binary = (visibility > 0.5).to(dtype=dtype)
    unknown = 1.0 - visibility_binary
    frontier_x = F.pad((unknown[..., 1:] - unknown[..., :-1]).abs(), (0, 1, 0, 0))
    frontier_y = F.pad((unknown[..., 1:, :] - unknown[..., :-1, :]).abs(), (0, 0, 0, 1))
    frontier = F.avg_pool2d((frontier_x + frontier_y).clamp(0.0, 1.0), kernel_size=patch_size, stride=patch_size)
    frontier = frontier.flatten(start_dim=1)

    tokens = torch.stack(
        (x_center, y_center, visible_ratio, dist_norm, cos_to_goal, sin_to_goal, depth_mean, depth_edge, frontier),
        dim=-1,
    ).reshape(env.num_envs, -1)
    env._instinct_nav_scene_token_cache = {"key": cache_key, "tokens": tokens}
    return tokens


def scene_tokens_risk_no_goal_from_visibility_map(
    env,
    sensor_cfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    size_m: float = 6.4,
    grid_size: int = 64,
    token_grid_size: int = 8,
    depth_min_m: float = 0.5,
    depth_max_m: float = 8.0,
    pixel_stride: int = 2,
    decay: float = 1.0,
) -> torch.Tensor:
    """Return 6D scene-risk tokens without goal-relative features.

    Layout: [x, y, visible_ratio, depth_mean, depth_edge, frontier].
    Goal information stays in the normal proprio/command observation and enters
    attention through the recurrent query rather than being duplicated inside
    every scene token.
    """
    cache_key = (
        "scene_tokens_risk_no_goal_from_visibility_map",
        getattr(sensor_cfg, "name", "raycast_camera"),
        int(getattr(env, "common_step_counter", -1)),
        int(grid_size),
        int(token_grid_size),
        float(size_m),
        float(depth_min_m),
        float(depth_max_m),
        int(pixel_stride),
        float(decay),
    )
    cache = getattr(env, "_instinct_nav_scene_token_cache", None)
    if cache is not None and cache.get("key") == cache_key:
        return cache["tokens"]

    visibility_flat = local_visibility_map(
        env,
        sensor_cfg=sensor_cfg,
        asset_cfg=asset_cfg,
        size_m=size_m,
        grid_size=grid_size,
        depth_min_m=depth_min_m,
        depth_max_m=depth_max_m,
        pixel_stride=pixel_stride,
        decay=decay,
    )
    visibility = visibility_flat.view(env.num_envs, 1, grid_size, grid_size)
    patch_size = grid_size // token_grid_size
    if patch_size * token_grid_size != grid_size:
        raise ValueError(f"grid_size={grid_size} must be divisible by token_grid_size={token_grid_size}")

    visible_ratio = F.avg_pool2d(visibility, kernel_size=patch_size, stride=patch_size).flatten(start_dim=1)

    device = visibility.device
    dtype = visibility.dtype
    rows = torch.arange(token_grid_size, device=device, dtype=dtype)
    cols = torch.arange(token_grid_size, device=device, dtype=dtype)
    row_grid, col_grid = torch.meshgrid(rows, cols, indexing="ij")
    patch_m = size_m / float(token_grid_size)
    half_size = size_m / 2.0
    x_center_m = half_size - (row_grid + 0.5) * patch_m
    y_center_m = half_size - (col_grid + 0.5) * patch_m
    x_center = (x_center_m / half_size).reshape(1, -1).expand(env.num_envs, -1)
    y_center = (y_center_m / half_size).reshape(1, -1).expand(env.num_envs, -1)

    sample = _get_noisy_depth_sample(env, sensor_cfg)
    depth = sample["depth"]
    if depth.dim() == 4:
        depth_image = depth[:, 0]
    else:
        height, width = env.scene.sensors[sensor_cfg.name].image_shape
        depth_image = depth.view(env.num_envs, height, width)
    valid_depth = torch.isfinite(depth_image) & (depth_image > depth_min_m) & (depth_image < depth_max_m)
    depth_norm = torch.where(
        valid_depth,
        ((depth_image - depth_min_m) / max(depth_max_m - depth_min_m, 1.0e-6)).clamp(0.0, 1.0),
        torch.ones_like(depth_image),
    )
    depth_mean = F.adaptive_avg_pool2d(depth_norm.unsqueeze(1), (token_grid_size, token_grid_size)).flatten(start_dim=1)
    grad_x = F.pad((depth_norm[..., 1:] - depth_norm[..., :-1]).abs(), (0, 1, 0, 0))
    grad_y = F.pad((depth_norm[..., 1:, :] - depth_norm[..., :-1, :]).abs(), (0, 0, 0, 1))
    depth_edge = F.adaptive_avg_pool2d((grad_x + grad_y).unsqueeze(1), (token_grid_size, token_grid_size)).flatten(start_dim=1)
    depth_edge = depth_edge.clamp(0.0, 1.0)

    frontier_source = visibility if decay < 1.0 else (visibility > 0.5).to(dtype=dtype)
    frontier_x = F.pad(
        (frontier_source[..., 1:] - frontier_source[..., :-1]).abs(), (0, 1, 0, 0)
    )
    frontier_y = F.pad(
        (frontier_source[..., 1:, :] - frontier_source[..., :-1, :]).abs(), (0, 0, 0, 1)
    )
    frontier = F.avg_pool2d((frontier_x + frontier_y).clamp(0.0, 1.0), kernel_size=patch_size, stride=patch_size)
    frontier = frontier.flatten(start_dim=1)

    tokens = torch.stack((x_center, y_center, visible_ratio, depth_mean, depth_edge, frontier), dim=-1)
    tokens = tokens.reshape(env.num_envs, -1)
    env._instinct_nav_scene_token_cache = {"key": cache_key, "tokens": tokens}
    return tokens


def dynamic_map_patches_from_visibility_map(
    env,
    sensor_cfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    size_m: float = 6.4,
    grid_size: int = 64,
    token_grid_size: int = 8,
    depth_min_m: float = 0.5,
    depth_max_m: float = 8.0,
    pixel_stride: int = 2,
    decay: float = 0.995,
    include_dynamic_mask: bool = False,
    mask_frame_dropout_prob: float = 0.08,
    mask_pixel_dropout_prob: float = 0.05,
    mask_false_positive_prob: float = 0.001,
    mask_erode_prob: float = 0.10,
    mask_dilate_prob: float = 0.15,
    mask_max_shift_pixels: int = 3,
    merge_dynamic_as_risk: bool = False,
    dynamic_risk_inflation_cells: int = 0,
    exclude_dynamic_from_visibility_history: bool = False,
    dynamic_risk_mode: str = "occupancy",
    sector_horizon_s: float = 1.2,
    sector_clearance_m: float = 0.30,
    sector_sigma_long_m: float = 0.15,
    sector_sigma_lat_m: float = 0.12,
    sector_softness_m: float = 0.08,
    sector_stationary_speed_mps: float = 0.10,
) -> torch.Tensor:
    """Return spatially aligned history/current BEV patches.

    The base layout per patch is: pre-update history score, current BEV hit
    ratio, current BEV normalized depth, current BEV depth edge, and
    current-minus-history score. With include_dynamic_mask enabled, the default
    behavior appends a sixth current-frame dynamic-occupancy channel. When
    ``merge_dynamic_as_risk`` is enabled instead, dynamic BEV cells are rendered
    as black in the existing five visibility features. The risk overlay is
    current-only and never modifies persistent visibility memory.
    """
    cache_key = (
        "dynamic_map_patches_from_visibility_map",
        getattr(sensor_cfg, "name", "raycast_camera"),
        int(getattr(env, "common_step_counter", -1)),
        int(grid_size),
        int(token_grid_size),
        float(decay),
        bool(include_dynamic_mask),
        float(mask_frame_dropout_prob),
        float(mask_pixel_dropout_prob),
        float(mask_false_positive_prob),
        float(mask_erode_prob),
        float(mask_dilate_prob),
        int(mask_max_shift_pixels),
        bool(merge_dynamic_as_risk),
        int(dynamic_risk_inflation_cells),
        bool(exclude_dynamic_from_visibility_history),
        str(dynamic_risk_mode),
        float(sector_horizon_s),
        float(sector_clearance_m),
        float(sector_sigma_long_m),
        float(sector_sigma_lat_m),
        float(sector_softness_m),
        float(sector_stationary_speed_mps),
    )
    cache = getattr(env, "_instinct_nav_dynamic_map_patch_cache", None)
    if cache is not None and cache.get("key") == cache_key:
        return cache["patches"]

    sample = _get_delayed_visual_packet(env, sensor_cfg)
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
    pixel_mask = None
    instance_visibility = None
    if include_dynamic_mask:
        image_height, image_width = sample["depth"].shape[-2:]
        mask_result = dynamic_obstacle_mask_observation(
            env,
            sensor_cfg,
            output_height=image_height,
            output_width=image_width,
            apply_corruption=True,
            frame_dropout_prob=mask_frame_dropout_prob,
            pixel_dropout_prob=mask_pixel_dropout_prob,
            false_positive_prob=mask_false_positive_prob,
            erode_prob=mask_erode_prob,
            dilate_prob=mask_dilate_prob,
            max_shift_pixels=mask_max_shift_pixels,
            return_instance_visibility=dynamic_risk_mode == "sector",
        )
        if dynamic_risk_mode == "sector":
            pixel_mask, instance_visibility = mask_result
        else:
            pixel_mask = mask_result
        pixel_mask = pixel_mask.reshape(env.num_envs, 1, image_height, image_width)

    components = manager.update(
        sample["depth"],
        sample["camera_pos_w"],
        sample["camera_quat_w"],
        sample["intrinsics"],
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        return_components=True,
        pixel_mask=pixel_mask,
        exclude_masked_pixels_from_visibility=exclude_dynamic_from_visibility_history,
    )
    if include_dynamic_mask:
        _, history_flat, current_hit_flat, current_depth_flat, dynamic_hit_flat = components
    else:
        _, history_flat, current_hit_flat, current_depth_flat = components
        dynamic_hit_flat = None
    history = history_flat.view(env.num_envs, 1, grid_size, grid_size)
    current_hit_bev = current_hit_flat.view(env.num_envs, 1, grid_size, grid_size)
    current_depth_bev = current_depth_flat.view(env.num_envs, 1, grid_size, grid_size)
    dynamic_hit_bev = None
    if dynamic_hit_flat is not None:
        dynamic_hit_bev = dynamic_hit_flat.view(env.num_envs, 1, grid_size, grid_size)
        if dynamic_risk_mode == "sector":
            delayed_state = _delayed_dynamic_state_for_risk(env, sensor_cfg)
            obstacle_manager = getattr(env, "_dynamic_obstacle_manager", None)
            if delayed_state is not None and instance_visibility is not None:
                positions_w, velocities_w, active = delayed_state
                obstacle_count = active.shape[1]
                if instance_visibility.shape[1] < obstacle_count:
                    instance_visibility = F.pad(
                        instance_visibility,
                        (0, obstacle_count - instance_visibility.shape[1]),
                        value=False,
                    )
                elif instance_visibility.shape[1] > obstacle_count:
                    instance_visibility = instance_visibility[:, :obstacle_count]
                # A simulated detector frame drop must suppress its tracks as
                # well; otherwise the sector leaks privileged GT information.
                frame_has_mask = pixel_mask.flatten(start_dim=1).any(dim=1)
                visible = instance_visibility & frame_has_mask.unsqueeze(1)
                body_half_extent = float(getattr(obstacle_manager, "body_half_extent", 0.25))
                sector_risk = _render_dynamic_sector_risk(
                    positions_w,
                    velocities_w,
                    active,
                    visible,
                    robot.data.root_pos_w,
                    robot.data.root_quat_w,
                    size_m=size_m,
                    grid_size=grid_size,
                    body_half_extent_m=body_half_extent,
                    clearance_m=sector_clearance_m,
                    horizon_s=sector_horizon_s,
                    sigma_long_m=sector_sigma_long_m,
                    sigma_lat_m=sector_sigma_lat_m,
                    softness_m=sector_softness_m,
                    stationary_speed_mps=sector_stationary_speed_mps,
                )
                dynamic_hit_bev = torch.maximum(dynamic_hit_bev, sector_risk)
                if obstacle_manager is not None:
                    obstacle_manager._last_dynamic_sector_risk_rate = (
                        (sector_risk >= 0.5).float().mean().detach()
                    )
                    obstacle_manager._last_dynamic_visible_instances = (
                        visible.float().sum(dim=1).mean().detach()
                    )
        elif dynamic_risk_mode != "occupancy":
            raise ValueError(f"Unknown dynamic_risk_mode={dynamic_risk_mode!r}")
        inflation = max(0, int(dynamic_risk_inflation_cells))
        if inflation > 0:
            kernel_size = 2 * inflation + 1
            dynamic_hit_bev = F.max_pool2d(
                dynamic_hit_bev, kernel_size=kernel_size, stride=1, padding=inflation
            )
        if merge_dynamic_as_risk:
            manager.current_dynamic_risk.copy_(dynamic_hit_bev)
            safe = 1.0 - dynamic_hit_bev
            history = history * safe
            current_hit_bev = current_hit_bev * safe
            current_depth_bev = current_depth_bev * safe
    patch_size = grid_size // token_grid_size
    if patch_size * token_grid_size != grid_size:
        raise ValueError(f"grid_size={grid_size} must be divisible by token_grid_size={token_grid_size}")
    history_score = F.avg_pool2d(history, patch_size, patch_size).flatten(start_dim=1)

    current_hit = F.avg_pool2d(current_hit_bev, patch_size, patch_size).flatten(start_dim=1)
    depth_sum = F.avg_pool2d(
        current_depth_bev * current_hit_bev, patch_size, patch_size
    ).flatten(start_dim=1)
    current_depth = torch.where(
        current_hit > 0.0,
        depth_sum / current_hit.clamp_min(1.0e-6),
        torch.ones_like(depth_sum),
    )
    edge_source = torch.where(
        current_hit_bev > 0.0, current_depth_bev, torch.ones_like(current_depth_bev)
    )
    grad_x = F.pad(
        (edge_source[..., 1:] - edge_source[..., :-1]).abs(), (0, 1, 0, 0)
    )
    grad_y = F.pad(
        (edge_source[..., 1:, :] - edge_source[..., :-1, :]).abs(), (0, 0, 0, 1)
    )
    current_edge = F.avg_pool2d(
        (grad_x + grad_y).clamp(0.0, 1.0), patch_size, patch_size
    ).flatten(start_dim=1)
    visibility_delta = (current_hit - history_score).clamp(-1.0, 1.0)
    patch_channels = (
        history_score,
        current_hit,
        current_depth,
        current_edge,
        visibility_delta,
    )
    if dynamic_hit_bev is not None:
        dynamic_presence = F.max_pool2d(
            dynamic_hit_bev, patch_size, patch_size
        ).flatten(start_dim=1)
        obstacle_manager = getattr(env, "_dynamic_obstacle_manager", None)
        if obstacle_manager is not None:
            obstacle_manager._last_dynamic_bev_token_positive_rate = (
                dynamic_presence.mean().detach()
            )
        if not merge_dynamic_as_risk:
            patch_channels = patch_channels + (dynamic_presence,)
    patches = torch.stack(patch_channels, dim=-1).reshape(env.num_envs, -1)
    env._instinct_nav_dynamic_map_patch_cache = {"key": cache_key, "patches": patches}
    return patches


@torch.no_grad()
def learned_dynamic_bev_observation(
    env,
    sensor_cfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    size_m: float = 6.4,
    grid_size: int = 64,
    output_grid_size: int = 32,
    depth_min_m: float = 0.5,
    depth_max_m: float = 8.0,
    pixel_stride: int = 2,
    decay: float = 0.995,
    mask_frame_dropout_prob: float = 0.08,
    mask_pixel_dropout_prob: float = 0.05,
    mask_false_positive_prob: float = 0.0,
    mask_erode_prob: float = 0.10,
    mask_dilate_prob: float = 0.15,
    mask_max_shift_pixels: int = 2,
    dynamic_history_offsets_s: tuple[float, float] | None = None,
    dynamic_dilation_kernel: int = 7,
) -> torch.Tensor:
    """Build a six-channel BEV image for learned spatial query fusion.

    Channels are persistent static visibility, current static depth hits,
    normalized current static depth, current dynamic occupancy, and the two
    preceding dynamic occupancy frames (or the configured time offsets). The dynamic history is warped into the
    current robot frame before it is returned, making temporal differences
    describe obstacle motion rather than ego motion.

    Geometry is accumulated at ``grid_size`` resolution, then downsampled to
    ``output_grid_size`` before PPO storage. Actor and critic calls share the
    same step cache, so temporal history advances exactly once per policy step.
    """
    if grid_size % output_grid_size != 0:
        raise ValueError(
            f"grid_size={grid_size} must be divisible by output_grid_size={output_grid_size}"
        )
    if dynamic_dilation_kernel < 1 or dynamic_dilation_kernel % 2 != 1:
        raise ValueError("dynamic_dilation_kernel must be a positive odd integer")
    history_steps = (1, 2)
    if dynamic_history_offsets_s is not None:
        if len(dynamic_history_offsets_s) != 2 or not 0 < dynamic_history_offsets_s[0] < dynamic_history_offsets_s[1]:
            raise ValueError("dynamic_history_offsets_s must contain two increasing positive times")
        dt = float(env.step_dt)
        history_steps = tuple(max(1, round(offset / dt)) for offset in dynamic_history_offsets_s)
        if history_steps[0] >= history_steps[1]:
            raise ValueError("dynamic history offsets must resolve to distinct policy steps")
    cache_key = (
        "learned_dynamic_bev_observation",
        getattr(sensor_cfg, "name", "camera"),
        int(getattr(env, "common_step_counter", -1)),
        int(grid_size),
        int(output_grid_size),
        float(decay),
        float(mask_frame_dropout_prob),
        float(mask_pixel_dropout_prob),
        float(mask_false_positive_prob),
        float(mask_erode_prob),
        float(mask_dilate_prob),
        int(mask_max_shift_pixels),
        history_steps,
        int(dynamic_dilation_kernel),
    )
    cache = getattr(env, "_instinct_nav_learned_bev_cache", None)
    if cache is not None and cache.get("key") == cache_key:
        return cache["bev"]

    sample = _get_delayed_visual_packet(env, sensor_cfg)
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
    if manager.dynamic_history is None or manager.dynamic_history.shape[1] != history_steps[-1]:
        manager.dynamic_history = torch.zeros(
            (env.num_envs, history_steps[-1], grid_size, grid_size), device=manager.device
        )
    image_height, image_width = sample["depth"].shape[-2:]
    pixel_mask = dynamic_obstacle_mask_observation(
        env,
        sensor_cfg,
        output_height=image_height,
        output_width=image_width,
        apply_corruption=True,
        frame_dropout_prob=mask_frame_dropout_prob,
        pixel_dropout_prob=mask_pixel_dropout_prob,
        false_positive_prob=mask_false_positive_prob,
        erode_prob=mask_erode_prob,
        dilate_prob=mask_dilate_prob,
        max_shift_pixels=mask_max_shift_pixels,
    ).reshape(env.num_envs, 1, image_height, image_width)

    _, _, current_hit_flat, current_depth_flat, dynamic_hit_flat = manager.update(
        sample["depth"],
        sample["camera_pos_w"],
        sample["camera_quat_w"],
        sample["intrinsics"],
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        return_components=True,
        pixel_mask=pixel_mask,
        exclude_masked_pixels_from_visibility=True,
    )
    current_hit_native = current_hit_flat.view(env.num_envs, 1, grid_size, grid_size)
    current_depth = current_depth_flat.view(env.num_envs, 1, grid_size, grid_size)
    current_dynamic = dynamic_hit_flat.view(env.num_envs, 1, grid_size, grid_size)

    # Preserve a small safety margin around a detected body at the native
    # 0.1 m grid. This is detector-space morphology, not a hand-designed future
    # trajectory; motion must be inferred from the temporal channels.
    current_dynamic = F.max_pool2d(
        current_dynamic, kernel_size=dynamic_dilation_kernel,
        stride=1, padding=dynamic_dilation_kernel // 2,
    )
    previous_dynamic = manager.dynamic_history[:, [step - 1 for step in history_steps]].clone()
    manager.dynamic_history[:, 1:].copy_(manager.dynamic_history[:, :-1].clone())
    manager.dynamic_history[:, 0].copy_(current_dynamic[:, 0])
    manager.current_dynamic_risk.copy_(current_dynamic)

    downsample = grid_size // output_grid_size
    static_history = F.avg_pool2d(manager.map, downsample, downsample)
    current_hit = F.avg_pool2d(current_hit_native, downsample, downsample)
    depth_sum = F.avg_pool2d(
        current_depth * (current_hit_native > 0.0), downsample, downsample
    )
    current_depth = torch.where(
        current_hit > 0.0,
        depth_sum / current_hit.clamp_min(1.0e-6),
        torch.zeros_like(depth_sum),
    )
    current_dynamic = F.max_pool2d(current_dynamic, downsample, downsample)
    previous_dynamic = F.max_pool2d(previous_dynamic, downsample, downsample)
    bev = torch.cat(
        (
            static_history,
            current_hit,
            current_depth,
            current_dynamic,
            previous_dynamic[:, 0:1],
            previous_dynamic[:, 1:2],
        ),
        dim=1,
    ).clamp_(0.0, 1.0)

    obstacle_manager = getattr(env, "_dynamic_obstacle_manager", None)
    if obstacle_manager is not None:
        obstacle_manager._last_dynamic_bev_token_positive_rate = current_dynamic.mean().detach()
    bev_flat = bev.flatten(start_dim=1)
    env._instinct_nav_learned_bev_cache = {"key": cache_key, "bev": bev_flat}
    return bev_flat


def bev_odometry_pose(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """Planar odometry for temporal-map alignment only, not navigation features."""
    robot = env.scene[asset_cfg.name]
    return torch.cat((robot.data.root_pos_w[:, :2],
                      _quat_wxyz_to_yaw(robot.data.root_quat_w)[:, None]), 1)


def future_occupancy_teacher_state(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """Training-only actual obstacle state; never append to actor/critic inputs."""
    robot = env.scene[asset_cfg.name]
    pose = torch.cat((robot.data.root_pos_w[:, :2], _quat_wxyz_to_yaw(robot.data.root_quat_w)[:, None]), 1)
    manager = getattr(env, "_dynamic_obstacle_manager", None)
    count = int(env.cfg.num_obstacles)
    slots = torch.zeros(env.num_envs, count, 4, device=env.device)
    if manager is not None:
        for index in range(count):
            slots[:, index, :2] = env.scene[f"obstacle_{index}"].data.root_pos_w[:, :2]
        slots[..., 2] = (manager.spawned_obstacles & manager.curriculum_envs[:, None] & bool(manager.enabled)).float()
        # The BEV observation has already computed visibility from the aligned
        # visual packet; do not run or corrupt the detector a second time.
        cache = getattr(env, "_instinct_nav_direct_mask_cache", None)
        if (cache is not None and cache["key"][0] == int(env.common_step_counter)
                and cache["instance_visibility"].shape == slots[..., 3].shape):
            slots[..., 3] = cache["instance_visibility"].float()
        relative = slots[..., :2] - pose[:, None, :2]
        cos, sin = pose[:, 2].cos()[:, None], pose[:, 2].sin()[:, None]
        local_x = cos * relative[..., 0] + sin * relative[..., 1]
        local_y = -sin * relative[..., 0] + cos * relative[..., 1]
        half_size = float(env.cfg.observations.policy.visibility_map.params["size_m"]) / 2
        slots[..., 3] *= ((local_x.abs() <= half_size + 0.25)
                         & (local_y.abs() <= half_size + 0.25)).float()
    return torch.cat((pose, slots.flatten(1)), 1)


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

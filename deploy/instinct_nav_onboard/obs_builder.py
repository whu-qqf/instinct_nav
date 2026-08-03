from __future__ import annotations

from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

from .config import OnboardConfig
from .packets import SensorPacket


class HighLevelObsBuilder:
    """Build the current SRU actor observation from real robot sensor packets.

    This class is intentionally conservative. It provides the deployment seam for
    real depth encoding and scene-token construction, but it does not pretend the
    sim observation pipeline is solved. The first robot integration should log all
    intermediate tensors and compare them with IsaacLab evaluation samples.
    """

    def __init__(self, cfg: OnboardConfig, device: str = "cpu"):
        self.cfg = cfg
        self.device = torch.device(device)
        self.last_action = torch.zeros(3, device=self.device)
        self.visibility = torch.zeros(1, 1, 64, 64, device=self.device)
        self.depth_history: deque[torch.Tensor] = deque(maxlen=cfg.camera.low_level_history)

    def reset(self) -> None:
        self.last_action.zero_()
        self.visibility.zero_()
        self.depth_history.clear()

    def build_actor_obs(self, packet: SensorPacket, encoded_depth: torch.Tensor, scene_tokens: torch.Tensor) -> torch.Tensor:
        """Return flat actor obs: proprio(16) + encoded_depth(2560) + scene_tokens(384)."""
        if packet.reset:
            self.reset()

        proprio = self._build_proprio(packet)
        encoded_depth = encoded_depth.to(device=self.device, dtype=torch.float32).reshape(1, -1)
        scene_tokens = scene_tokens.to(device=self.device, dtype=torch.float32).reshape(1, -1)
        obs = torch.cat((proprio, encoded_depth, scene_tokens), dim=-1)
        expected = self.cfg.layout.actor_obs_dim
        if obs.shape[-1] != expected:
            raise ValueError(f"actor obs dim mismatch: got {obs.shape[-1]}, expected {expected}")
        return obs

    def _build_proprio(self, packet: SensorPacket) -> torch.Tensor:
        base_lin_vel = torch.as_tensor(packet.base_lin_vel_b, device=self.device, dtype=torch.float32).reshape(1, 3)
        base_ang_vel = torch.as_tensor(packet.base_ang_vel_b, device=self.device, dtype=torch.float32).reshape(1, 3)
        projected_gravity = torch.as_tensor(packet.projected_gravity_b, device=self.device, dtype=torch.float32).reshape(1, 3)
        goal = torch.as_tensor(packet.goal_b, device=self.device, dtype=torch.float32).reshape(1, 4)
        last_action = self.last_action.reshape(1, 3)
        return torch.cat((base_lin_vel, base_ang_vel, projected_gravity, last_action, goal), dim=-1)

    def update_last_action(self, action: torch.Tensor) -> None:
        self.last_action = action.detach().reshape(-1)[:3].to(self.device)

    def make_low_level_depth(self, packet: SensorPacket) -> torch.Tensor:
        """Approximate the low-level camera preprocessing used by InstinctLab.

        Real integration should compare this tensor against instinct_onboard's
        native preprocessing. If instinct_onboard already builds the low-level
        depth history, prefer that implementation and use this only for tests.
        """
        depth = torch.as_tensor(packet.depth_m, device=self.device, dtype=torch.float32).reshape(1, 1, 36, 64)
        y, x, h, w = self.cfg.camera.low_level_crop_region
        cropped = depth[:, :, y : y + h, x : x + w]
        resized = F.interpolate(cropped, size=(18, 32), mode="bilinear", align_corners=False)
        lo, hi = self.cfg.camera.low_level_depth_range_m
        normalized = ((resized - lo) / max(hi - lo, 1.0e-6)).clamp(0.0, 1.0)
        self.depth_history.append(normalized.squeeze(0))
        while len(self.depth_history) < self.cfg.camera.low_level_history:
            self.depth_history.append(normalized.squeeze(0).clone())
        return torch.cat(list(self.depth_history), dim=0).reshape(1, self.cfg.camera.low_level_history, 18, 32)

    def build_scene_tokens_from_depth_visibility(self, depth_m: np.ndarray) -> torch.Tensor:
        """Build 6D scene-risk tokens from the live depth frame and local visibility memory.

        Training uses IsaacLab camera/root poses to back-project delayed depth into
        a yaw-aligned 64x64 BEV memory. On the robot we keep the same token layout
        but only use the current torso-camera depth until odom/root-pose wiring is
        available. This is intentionally a conservative approximation, not a
        claim of perfect sim-to-real parity.
        """
        device = self.device
        token_grid = 8
        grid_size = 64
        size_m = 6.4
        half_size = size_m / 2.0
        depth_min_m = 0.5
        depth_max_m = 8.0

        depth = torch.as_tensor(depth_m, device=device, dtype=torch.float32).reshape(1, 1, *depth_m.shape)
        depth_image = depth[:, 0]
        valid = torch.isfinite(depth_image) & (depth_image > depth_min_m) & (depth_image < depth_max_m)

        current_visibility = self._depth_to_current_visibility(depth_image, valid, size_m=size_m, grid_size=grid_size)
        if self.cfg.visibility_decay < 1.0:
            self.visibility.mul_(self.cfg.visibility_decay)
            self.visibility.add_(
                self.cfg.visibility_score_alpha_up * current_visibility * (1.0 - self.visibility)
            )
            self.visibility.clamp_(0.0, 1.0)
        else:
            self.visibility = torch.maximum(self.visibility, current_visibility)

        patch_size = grid_size // token_grid
        visible_ratio = F.avg_pool2d(self.visibility, kernel_size=patch_size, stride=patch_size).flatten(start_dim=1)

        rows = torch.arange(token_grid, device=device, dtype=torch.float32)
        cols = torch.arange(token_grid, device=device, dtype=torch.float32)
        row_grid, col_grid = torch.meshgrid(rows, cols, indexing="ij")
        patch_m = size_m / float(token_grid)
        x_center_m = half_size - (row_grid + 0.5) * patch_m
        y_center_m = half_size - (col_grid + 0.5) * patch_m
        x_center = (x_center_m / half_size).reshape(1, -1)
        y_center = (y_center_m / half_size).reshape(1, -1)

        depth_norm = torch.where(
            valid,
            ((depth_image - depth_min_m) / max(depth_max_m - depth_min_m, 1.0e-6)).clamp(0.0, 1.0),
            torch.ones_like(depth_image),
        )
        depth_mean = F.adaptive_avg_pool2d(depth_norm.unsqueeze(1), (token_grid, token_grid)).flatten(start_dim=1)
        grad_x = F.pad((depth_norm[..., 1:] - depth_norm[..., :-1]).abs(), (0, 1, 0, 0))
        grad_y = F.pad((depth_norm[..., 1:, :] - depth_norm[..., :-1, :]).abs(), (0, 0, 0, 1))
        depth_edge = F.adaptive_avg_pool2d((grad_x + grad_y).unsqueeze(1), (token_grid, token_grid)).flatten(start_dim=1)
        depth_edge = depth_edge.clamp(0.0, 1.0)

        frontier_source = (
            self.visibility
            if self.cfg.visibility_decay < 1.0
            else (self.visibility > 0.5).to(dtype=torch.float32)
        )
        frontier_x = F.pad(
            (frontier_source[..., 1:] - frontier_source[..., :-1]).abs(), (0, 1, 0, 0)
        )
        frontier_y = F.pad(
            (frontier_source[..., 1:, :] - frontier_source[..., :-1, :]).abs(), (0, 0, 0, 1)
        )
        frontier = F.avg_pool2d((frontier_x + frontier_y).clamp(0.0, 1.0), kernel_size=patch_size, stride=patch_size)
        frontier = frontier.flatten(start_dim=1)

        tokens = torch.stack((x_center, y_center, visible_ratio, depth_mean, depth_edge, frontier), dim=-1)
        return tokens.reshape(1, -1)

    def build_scene_tokens_from_visibility_stub(self, depth_m: np.ndarray) -> torch.Tensor:
        """Backward-compatible alias for dry-run scripts."""
        return self.build_scene_tokens_from_depth_visibility(depth_m)

    def _depth_to_current_visibility(
        self, depth_image: torch.Tensor, valid: torch.Tensor, size_m: float, grid_size: int
    ) -> torch.Tensor:
        """Project current depth pixels into a robot-local 64x64 visibility grid.

        This uses camera FOV from training config and assumes the camera is
        yaw-aligned with the torso. It intentionally avoids fake global pose
        integration; odom-based warping should be added when root pose is
        available on the robot.
        """
        _, height, width = depth_image.shape
        rows = torch.arange(height, device=self.device, dtype=torch.float32)
        cols = torch.arange(width, device=self.device, dtype=torch.float32)
        row_grid, col_grid = torch.meshgrid(rows, cols, indexing="ij")

        h_fov = torch.deg2rad(torch.tensor(self.cfg.camera.horizontal_fov_deg, device=self.device))
        v_fov = torch.deg2rad(torch.tensor(self.cfg.camera.vertical_fov_deg, device=self.device))
        fx = (width * 0.5) / torch.tan(h_fov * 0.5)
        fy = (height * 0.5) / torch.tan(v_fov * 0.5)
        cx = (width - 1) * 0.5
        cy = (height - 1) * 0.5

        z = depth_image[0]
        x_forward = z
        y_left = -((col_grid - cx) / fx) * z
        z_up = -((row_grid - cy) / fy) * z

        # Ignore rays that likely terminate on the robot body or ceiling; the
        # token branch is primarily interested in horizontal local structure.
        valid_points = valid[0] & torch.isfinite(x_forward) & torch.isfinite(y_left) & torch.isfinite(z_up)
        half_size = size_m / 2.0
        cell_size = size_m / float(grid_size)
        row = torch.floor((half_size - x_forward) / cell_size).long()
        col = torch.floor((half_size - y_left) / cell_size).long()
        in_grid = valid_points & (row >= 0) & (row < grid_size) & (col >= 0) & (col < grid_size)

        visibility_flat = torch.zeros(1, grid_size * grid_size, device=self.device)
        linear_idx = (row.clamp(0, grid_size - 1) * grid_size + col.clamp(0, grid_size - 1)).reshape(1, -1)
        values = in_grid.reshape(1, -1).to(dtype=visibility_flat.dtype)
        visibility_flat.scatter_reduce_(1, linear_idx, values, reduce="amax", include_self=True)
        return visibility_flat.view(1, 1, grid_size, grid_size)

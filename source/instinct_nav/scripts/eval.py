#!/usr/bin/env python3
# Copyright (c) 2022-2025, Fan Yang and Per Frivik, ETH Zurich.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Evaluate a trained navigation policy and summarize approximate paper-style metrics.

This script is intended for approximate reproduction of the SRU paper results on the
currently available workspace/runtime. It loads a checkpoint, runs batched episodes,
and reports:

- overall episode success rate
- per-termination counts / rates
- terrain-type grouped success rates when terrain metadata is available

It works with standard RSL-RL checkpoints (``model_*.pt``) and TorchScript deployment
policies. For TorchScript policies, only ``num_envs=1`` is supported.

Examples:
    python scripts/eval.py --task Isaac-Nav-PPO-AoW-D-v0 --checkpoint logs/.../model_6500.pt \
        --num_envs 256 --num_episodes 1000 --headless

    python scripts/eval.py --task Isaac-Nav-PPO-AoW-D-Play-v0 --num_episodes 200 --headless
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import scipy.ndimage as ndi

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Evaluate a trained navigation policy.")
parser.add_argument("--task", type=str, default=None, help="Registered Gym task id.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint (.pt). Defaults to latest.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of parallel environments.")
parser.add_argument("--num_episodes", type=int, default=200, help="Number of finished episodes to aggregate.")
parser.add_argument("--seed", type=int, default=None, help="Seed for evaluation.")
parser.add_argument(
    "--episode_length_s",
    type=float,
    default=None,
    help="Optional evaluation-only episode timeout in seconds. Defaults to the environment config value.",
)
parser.add_argument(
    "--initial_env_cases_only",
    action="store_true",
    help=(
        "Only aggregate the first episode from each environment. Use with num_envs == num_episodes "
        "to compare policies on identical initial navigation cases without reset-order drift."
    ),
)
parser.add_argument(
    "--disable_eval_overrides",
    action="store_true",
    help="Do not disable observation corruption / interval pushes during evaluation.",
)
parser.add_argument(
    "--showcase_scene",
    type=str,
    default="none",
    choices=["none", "realistic", "rivermark", "outdoor_campus", "official_warehouse"],
    help=(
        "Showcase scene preset for GIF evaluation. "
        "\"realistic\" keeps the procedural terrain but improves material/sky lighting; "
        "\"rivermark\" loads the downloaded outdoor Rivermark USD scene and uses fixed point-nav cases."
    ),
)
parser.add_argument(
    "--json_out",
    type=str,
    default=None,
    help="Optional path to write the summary JSON.",
)
parser.add_argument(
    "--debug_dump_dir",
    type=str,
    default=None,
    help="Optional directory for IsaacLab eval observation/action dumps matching deploy debug format.",
)
parser.add_argument(
    "--debug_dump_env_id",
    type=int,
    default=0,
    help="Environment id to dump when --debug_dump_dir is enabled.",
)
parser.add_argument(
    "--debug_dump_every_steps",
    type=int,
    default=1,
    help="Dump every N eval policy steps when --debug_dump_dir is enabled.",
)
parser.add_argument(
    "--debug_dump_max_steps",
    type=int,
    default=1000,
    help="Maximum number of eval policy steps to dump.",
)
parser.add_argument(
    "--debug_dump_stats_only",
    action="store_true",
    default=False,
    help="Only write index.jsonl statistics; skip per-step npz arrays.",
)
parser.add_argument(
    "--print_every",
    type=int,
    default=50,
    help="Print progress every N finished episodes.",
)
parser.add_argument(
    "--max_base_contact_records",
    type=int,
    default=100,
    help="Maximum number of detailed base-contact failure records to store in the JSON summary.",
)
parser.add_argument(
    "--failure_topdown_dir",
    type=str,
    default=None,
    help="Optional directory where per-failure top-down PNG snapshots will be written.",
)
parser.add_argument(
    "--failure_topdown_limit",
    type=int,
    default=0,
    help="Maximum number of failed episodes to render as top-down snapshots. 0 disables rendering.",
)
parser.add_argument(
    "--failure_topdown_crop_size_m",
    type=float,
    default=0.0,
    help="Optional square crop size in meters around the robot for failure top-down renders. <=0 renders the full terrain tile.",
)
parser.add_argument(
    "--failure_topdown_dpi",
    type=int,
    default=150,
    help="DPI used when saving top-down failure renders.",
)
parser.add_argument(
    "--failure_rgb_dir",
    type=str,
    default=None,
    help="Optional directory where real 3D RGB screenshots will be saved for failed episodes.",
)
parser.add_argument(
    "--failure_rgb_limit",
    type=int,
    default=0,
    help="Maximum number of failed episodes to export as real 3D RGB screenshots. 0 disables RGB capture.",
)
parser.add_argument(
    "--failure_rgb_success_limit",
    type=int,
    default=None,
    help="Maximum number of successful episodes to export as RGB GIF/PNG. None falls back to --failure_rgb_limit when successes are included.",
)
parser.add_argument(
    "--failure_rgb_failure_limit",
    type=int,
    default=None,
    help="Maximum number of failed episodes to export as RGB GIF/PNG. None falls back to --failure_rgb_limit.",
)
parser.add_argument(
    "--failure_rgb_success_min_initial_distance",
    type=float,
    default=0.0,
    help="Only export successful RGB GIF/PNG cases whose initial spawn-goal XY distance is at least this value in meters. 0 disables filtering.",
)
parser.add_argument(
    "--failure_rgb_success_min_goal_height_delta",
    type=float,
    default=0.0,
    help=(
        "Only export successful RGB GIF/PNG cases whose sampled goal terrain height is at least this many meters "
        "above the sampled spawn terrain height. 0 disables filtering."
    ),
)
parser.add_argument(
    "--failure_rgb_terrain_name",
    type=str,
    default=None,
    help="Only export RGB GIF/PNG cases from this terrain name, for example 'stairs'. None disables terrain filtering.",
)
parser.add_argument(
    "--failure_rgb_base_contact_only",
    action="store_true",
    help="Only export real 3D RGB captures for base_contact failures.",
)
parser.add_argument(
    "--failure_rgb_include_successes",
    action="store_true",
    default=True,
    help="Export successful episodes too. Enabled by default; filenames use the 'success' label.",
)
parser.add_argument(
    "--failure_rgb_include_observations",
    action="store_true",
    default=True,
    help="When exporting RGB GIFs, compose each RGB frame with depth, visibility, and scene-token observation panels. Enabled by default.",
)
parser.add_argument(
    "--depth_pit_to_obstacle",
    action="store_true",
    help="Zero-shot eval-only preprocessing: detect pit-like depth discontinuities and encode them as closer obstacles.",
)
parser.add_argument(
    "--depth_pit_jump_m",
    type=float,
    default=0.30,
    help="Minimum below-ground height in meters to classify a region as pit-like.",
)
parser.add_argument(
    "--depth_pit_local_drop_m",
    type=float,
    default=0.12,
    help="Minimum nearby local below-ground jump in meters used as weak pit-edge evidence.",
)
parser.add_argument(
    "--depth_pit_obstacle_margin_m",
    type=float,
    default=0.30,
    help="How much closer than the local foreground depth to render detected pits.",
)
parser.add_argument(
    "--depth_pit_obstacle_height_m",
    type=float,
    default=0.60,
    help="Vertical height in meters used to extrude detected pits into above-ground obstacle silhouettes.",
)
parser.add_argument(
    "--depth_pit_dilation_px",
    type=int,
    default=5,
    help="Odd pixel kernel size used to dilate detected pit regions in depth preprocessing.",
)
parser.add_argument(
    "--depth_pit_min_row_frac",
    type=float,
    default=0.25,
    help="Ignore pit detections above this image row fraction to reduce far-field false positives.",
)
parser.add_argument(
    "--depth_pit_max_distance_m",
    type=float,
    default=4.5,
    help="Maximum raw depth in meters eligible for pit-to-obstacle rewriting.",
)
parser.add_argument(
    "--failure_rgb_width",
    type=int,
    default=960,
    help="Width in pixels for RGB failure screenshots.",
)
parser.add_argument(
    "--failure_rgb_height",
    type=int,
    default=540,
    help="Height in pixels for RGB failure screenshots.",
)
parser.add_argument(
    "--failure_rgb_distance_m",
    type=float,
    default=0.0,
    help="Horizontal camera offset in meters for RGB failure screenshots. Defaults to 0 for top-down capture.",
)
parser.add_argument(
    "--failure_rgb_height_m",
    type=float,
    default=10.0,
    help="Camera height above the robot in meters for RGB failure screenshots. Defaults to a top-down view covering roughly 10 m vertically.",
)
parser.add_argument(
    "--failure_rgb_focal_length",
    type=float,
    default=12.0,
    help="Camera focal length for RGB failure screenshots. Lower values give a wider top-down field of view.",
)
parser.add_argument(
    "--failure_rgb_lookahead_m",
    type=float,
    default=0.0,
    help="Optional horizontal look-ahead in meters for RGB failure screenshots. Defaults to 0 for true vertical top-down capture.",
)
parser.add_argument(
    "--failure_rgb_center_mode",
    type=str,
    choices=("root", "collision", "midpoint", "spawn_goal"),
    default="root",
    help="Center mode for RGB failure screenshots: 'root' keeps the robot centered, 'collision' centers the approximated collision point, 'midpoint' uses the midpoint between them, and 'spawn_goal' fixes the view to the navigation case midpoint.",
)
parser.add_argument(
    "--failure_rgb_follow_robot",
    action="store_true",
    default=True,
    help="For RGB GIF history export, recenter the BEV RGB camera on the robot root at every history frame. Enabled by default.",
)
parser.add_argument(
    "--failure_rgb_include_rear_view",
    action="store_true",
    help="When composing observation GIFs, insert an oblique rear camera view immediately after the BEV panel.",
)
parser.add_argument(
    "--failure_rgb_rear_height_m",
    type=float,
    default=5.0,
    help="Height above the robot for the oblique rear camera panel.",
)
parser.add_argument(
    "--failure_rgb_rear_distance_m",
    type=float,
    default=4.0,
    help="Backward offset from the robot for the oblique rear camera panel.",
)
parser.add_argument(
    "--failure_rgb_rear_lateral_m",
    type=float,
    default=1.5,
    help="Right-side lateral offset for the oblique rear camera panel.",
)
parser.add_argument(
    "--failure_rgb_rear_lookahead_m",
    type=float,
    default=1.0,
    help="Target height above the robot root for the rear gravity-aligned look-at camera panel.",
)
parser.add_argument(
    "--failure_rgb_export_mode",
    type=str,
    choices=("png", "gif", "both"),
    default="gif",
    help="Whether to save a single PNG, a multi-frame GIF, or both for each failure RGB capture.",
)
parser.add_argument(
    "--failure_rgb_history_frames",
    type=int,
    default=10,
    help="Number of pre-impact control-step frames to include in each RGB failure GIF. Values <=1 disable GIF history and use only the final pre-impact frame.",
)
parser.add_argument(
    "--failure_rgb_full_episode",
    action="store_true",
    default=True,
    help=(
        "When exporting RGB/observation GIFs, keep enough history to render the full episode "
        "from reset to termination instead of only the last --failure_rgb_history_frames frames. "
        "Enabled by default; use modest num_envs because observation panels are cached for every env."
    ),
)

parser.add_argument(
    "--failure_rgb_min_episode",
    type=int,
    default=0,
    help="Only export failure/success RGB GIFs for episodes whose global episode index is >= this value. 0 disables filtering.",
)

parser.add_argument(
    "--force_dynamic_obstacles",
    action="store_true",
    default=False,
    help="Force-enable scripted dynamic obstacles for evaluation, so moving obstacles are active from the first episode.",
)
parser.add_argument(
    "--failure_rgb_gif_frame_ms",
    type=int,
    default=250,
    help="Frame duration in milliseconds for failure RGB GIF export.",
)
parser.add_argument(
    "--observation_gif_dir",
    type=str,
    default=None,
    help="Optional directory where per-base_contact failure depth/height GIFs will be written.",
)
parser.add_argument(
    "--observation_gif_frames",
    type=int,
    default=10,
    help="Number of pre-impact control-step frames to include in each base_contact depth/height GIF.",
)
parser.add_argument(
    "--observation_gif_frame_ms",
    type=int,
    default=250,
    help="Frame duration in milliseconds for depth/height observation GIF export.",
)
parser.add_argument(
    "--observation_gif_limit",
    type=int,
    default=0,
    help="Maximum number of base_contact failures to export as depth/height GIFs. 0 disables export.",
)
parser.add_argument(
    "--local_terrain_map",
    action="store_true",
    help="Eval-only visualization: maintain a local BEV terrain map from recent depth ray hits and robot pose.",
)
parser.add_argument(
    "--local_terrain_map_size_m",
    type=float,
    default=6.0,
    help="Square local terrain map size in meters.",
)
parser.add_argument(
    "--local_terrain_map_resolution_m",
    type=float,
    default=0.08,
    help="Local terrain map grid resolution in meters.",
)
parser.add_argument(
    "--local_terrain_map_history_frames",
    type=int,
    default=12,
    help="Number of recent depth frames to fuse into the local terrain map.",
)
parser.add_argument(
    "--local_terrain_map_stride",
    type=int,
    default=2,
    help="Pixel stride for adding depth ray hits to the local terrain map.",
)
parser.add_argument(
    "--local_terrain_map_depth_min_m",
    type=float,
    default=0.5,
    help="Minimum depth, in meters, used when projecting depth pixels into the local BEV terrain map.",
)
parser.add_argument(
    "--local_terrain_map_depth_max_m",
    type=float,
    default=8.0,
    help="Maximum depth, in meters, used when projecting depth pixels into the local BEV terrain map.",
)
parser.add_argument(
    "--local_terrain_map_height_range_m",
    type=float,
    default=1.5,
    help="Relative height range around the robot used for local terrain map coloring.",
)
parser.add_argument(
    "--local_terrain_map_gradient_threshold_m",
    type=float,
    default=10000,
    help="Mark BEV grid cells whose neighbor height jump exceeds this threshold in meters.",
)
parser.add_argument(
    "--case_manifest_out",
    type=str,
    default=None,
    help="Optional path to write the initial per-env navigation case manifest used for exact policy comparisons.",
)
parser.add_argument(
    "--case_manifest_in",
    type=str,
    default=None,
    help="Optional path to an initial per-env navigation case manifest to apply before evaluation starts.",
)
parser.add_argument(
    "--compare_checkpoint",
    type=str,
    default=None,
    help="Optional second checkpoint to evaluate in the same Isaac process and terrain scene after the first checkpoint.",
)
parser.add_argument(
    "--compare_failure_rgb_dir",
    type=str,
    default=None,
    help="Output directory for the second checkpoint when --compare_checkpoint is used.",
)
parser.add_argument(
    "--compare_json_out",
    type=str,
    default=None,
    help="JSON summary path for the second checkpoint when --compare_checkpoint is used.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if (
    args_cli.failure_rgb_dir is not None
    or args_cli.compare_failure_rgb_dir is not None
    or args_cli.failure_rgb_limit > 0
    or (args_cli.failure_rgb_success_limit is not None and args_cli.failure_rgb_success_limit > 0)
    or (args_cli.failure_rgb_failure_limit is not None and args_cli.failure_rgb_failure_limit > 0)
):
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from rsl_rl.runners import OnPolicyRunner
try:
    from scripts.eval_depth_viz import rear_gravity_aligned_camera_pose, select_delayed_depth_frames, select_episode_history_for_env
except ModuleNotFoundError:
    from eval_depth_viz import rear_gravity_aligned_camera_pose, select_delayed_depth_frames, select_episode_history_for_env
try:
    from scripts.showcase_scene import apply_showcase_scene_preset
except ModuleNotFoundError:
    from showcase_scene import apply_showcase_scene_preset

import isaaclab_tasks  # noqa: F401
import instinct_nav  # noqa: F401

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors.camera import Camera, CameraCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import SPHERE_MARKER_CFG
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab.utils import math as math_utils
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from instinct_nav.navigation.mdp import observations as nav_observations

_failure_goal_marker: VisualizationMarkers | None = None


def get_failure_goal_marker() -> VisualizationMarkers:
    global _failure_goal_marker
    if _failure_goal_marker is None:
        cfg = SPHERE_MARKER_CFG.copy()
        cfg.prim_path = "/Visuals/Eval/point_navigation_goal"
        cfg.markers["sphere"].radius = 0.55
        cfg.markers["sphere"].visual_material.diffuse_color = (1.0, 0.15, 0.0)
        _failure_goal_marker = VisualizationMarkers(cfg)
    return _failure_goal_marker


def visualize_goal_marker_for_snapshot(
    snapshot: dict[str, torch.Tensor], env_id: int, device: torch.device | str
) -> None:
    goal_positions = snapshot.get("goal_position_world")
    if goal_positions is None:
        return
    goal_position = goal_positions[env_id].detach().clone().to(device=device, dtype=torch.float32)
    goal_position[2] += 0.35
    get_failure_goal_marker().visualize(translations=goal_position.view(1, 3))


class LegacyRslRlVecEnvWrapper(RslRlVecEnvWrapper):
    def _ensure_batched_obs(self, value):
        if torch.is_tensor(value) and value.dim() == 1:
            return value.unsqueeze(0)
        return value

    def _to_legacy_obs(self, obs, extras=None):
        obs_keys = obs.keys() if hasattr(obs, "keys") else ()
        if "policy" in obs_keys:
            policy_obs = self._ensure_batched_obs(obs["policy"])
            critic_obs = obs["critic"] if "critic" in obs_keys else policy_obs
            critic_obs = self._ensure_batched_obs(critic_obs)
            return policy_obs, {"observations": {"critic": critic_obs}}
        observation_extras = (extras or {}).get("observations", {})
        critic_obs = observation_extras.get("critic", obs) if isinstance(observation_extras, dict) else obs
        obs = self._ensure_batched_obs(obs)
        critic_obs = self._ensure_batched_obs(critic_obs)
        return obs, {"observations": {"critic": critic_obs}}

    def reset(self):
        obs, extras = super().reset()
        policy_obs, legacy_extras = self._to_legacy_obs(obs, extras)
        extras.setdefault("observations", {}).update(legacy_extras["observations"])
        return policy_obs, extras

    def get_observations(self):
        result = super().get_observations()
        if isinstance(result, tuple) and len(result) == 2:
            obs, extras = result
        else:
            obs, extras = result, {}
        policy_obs, legacy_extras = self._to_legacy_obs(obs, extras)
        extras.setdefault("observations", {}).update(legacy_extras["observations"])
        return policy_obs, extras

    def step(self, actions):
        obs, rew, dones, extras = super().step(actions)
        policy_obs, legacy_extras = self._to_legacy_obs(obs, extras)
        extras.setdefault("observations", {}).update(legacy_extras["observations"])
        return policy_obs, rew, dones, extras


def _preprocess_depth_pits_to_obstacles(depth_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Detect pit edges from depth and vertical-coordinate gradients, then render vertical columns."""
    if depth_tensor.dim() != 4:
        raise ValueError(f"Expected depth tensor shape [B, 1, H, W], got {tuple(depth_tensor.shape)}")

    min_depth_value = float(nav_observations.min_depth) if nav_observations.min_depth > 0.0 else 0.25
    max_depth_value = float(nav_observations.max_depth) if nav_observations.max_depth > 0.0 else 10.0

    depth = torch.nan_to_num(depth_tensor, nan=50.0, posinf=50.0, neginf=0.0)
    eligible_max_depth = min(max_depth_value, float(args_cli.depth_pit_max_distance_m))
    valid = torch.logical_and(depth >= min_depth_value, depth <= eligible_max_depth)
    batch, _, height, width = depth.shape
    device = depth.device
    dtype = depth.dtype

    # RayCasterCameraCfg is created from a 192x120 ROS camera_info with downsample=3.
    fx = torch.as_tensor(72.7025 / 3.0, device=device, dtype=dtype)
    fy = torch.as_tensor(72.7025 / 3.0, device=device, dtype=dtype)
    cx = torch.as_tensor(94.4457 / 3.0, device=device, dtype=dtype)
    cy = torch.as_tensor(62.5424 / 3.0, device=device, dtype=dtype)
    u = torch.arange(width, device=device, dtype=dtype).view(1, 1, 1, width)
    v = torch.arange(height, device=device, dtype=dtype).view(1, 1, height, 1)
    x_ray = (u - cx) / fx
    y_ray = (v - cy) / fy

    # ROS optical camera frame: X right, Y down, Z forward. A pit edge should be
    # farther in image-y direction and lower in camera vertical coordinate.
    y_coord = y_ray * depth

    min_row = int(max(0.0, min(1.0, float(args_cli.depth_pit_min_row_frac))) * height)
    row_gate = torch.zeros_like(valid)
    if min_row > 0:
        row_gate[..., min_row:, :] = True
    else:
        row_gate[...] = True

    depth_grad_y = depth[..., 1:, :] - depth[..., :-1, :]
    y_coord_grad_y = y_coord[..., 1:, :] - y_coord[..., :-1, :]
    valid_pair = torch.logical_and(valid[..., 1:, :], valid[..., :-1, :])
    valid_pair = torch.logical_and(valid_pair, row_gate[..., 1:, :])
    # pit_edge = torch.logical_and(depth_grad_y >= float(args_cli.depth_pit_local_drop_m), valid_pair)
    pit_edge = torch.logical_and(depth_grad_y <= -0.005, valid_pair)
    # pit_edge = torch.logical_and(pit_edge, y_coord_grad_y >= float(args_cli.depth_pit_jump_m))
    pit_edge = torch.logical_and(pit_edge, y_coord_grad_y <= -0.00)

    pit_mask = torch.zeros_like(valid)
    pit_mask[..., 1:, :] = pit_edge

    pit_mask_np = pit_mask.detach().cpu().numpy()
    filtered_mask_np = np.zeros_like(pit_mask_np, dtype=bool)
    structure = np.ones((3, 3), dtype=np.uint8)
    for batch_index in range(batch):
        labels, num_labels = ndi.label(pit_mask_np[batch_index, 0], structure=structure)
        if num_labels == 0:
            continue
        counts = np.bincount(labels.reshape(-1))
        keep_labels = counts > 5
        keep_labels[0] = False
        filtered_mask_np[batch_index, 0] = keep_labels[labels]
    pit_mask = torch.as_tensor(filtered_mask_np, device=device)

    dilation_kernel = max(1, int(args_cli.depth_pit_dilation_px))
    if dilation_kernel % 2 == 0:
        dilation_kernel += 1
    if dilation_kernel > 1:
        pit_mask = torch.nn.functional.max_pool2d(
            pit_mask.float(),
            kernel_size=dilation_kernel,
            stride=1,
            padding=dilation_kernel // 2,
        ) > 0.0
        pit_mask = torch.logical_and(pit_mask, torch.logical_and(valid, row_gate))

    obstacle_height_m = max(0.0, float(args_cli.depth_pit_obstacle_height_m))
    extrude_rows = torch.ceil(fy * obstacle_height_m / torch.clamp(depth, min=min_depth_value)).to(torch.long)
    extrude_rows = torch.clamp(extrude_rows, min=1, max=height)
    obstacle_mask = torch.zeros_like(pit_mask)
    obstacle_silhouette_depth = torch.full_like(depth, max_depth_value)
    for offset in range(height):
        source_mask = pit_mask[..., offset:, :]
        if source_mask.numel() == 0:
            continue
        source_mask = torch.logical_and(source_mask, extrude_rows[..., offset:, :] >= offset + 1)
        column_depth = torch.clamp(depth[..., offset:, :], min=min_depth_value, max=max_depth_value)
        obstacle_mask[..., : height - offset, :] = torch.logical_or(
            obstacle_mask[..., : height - offset, :],
            source_mask,
        )
        obstacle_silhouette_depth[..., : height - offset, :] = torch.minimum(
            obstacle_silhouette_depth[..., : height - offset, :],
            torch.where(
                source_mask,
                column_depth,
                torch.full_like(column_depth, max_depth_value),
            ),
        )

    processed = torch.where(obstacle_mask, torch.minimum(depth, obstacle_silhouette_depth), depth)
    processed = torch.where(processed > max_depth_value, torch.zeros_like(processed), processed)
    processed = torch.where(processed < min_depth_value, torch.zeros_like(processed), processed)
    return processed, obstacle_mask


@torch.no_grad()
def depth_image_pit_to_obstacle_noisy_delayed(env, sensor_cfg, visualize: bool = False):
    """Depth observation wrapper that performs zero-shot pit-to-obstacle preprocessing before encoding."""
    nav_observations._ensure_depth_noise_generator_initialized()

    depth_camera = env.scene.sensors[sensor_cfg.name]
    depth_tensor = depth_camera.data.output["distance_to_image_plane"].view(env.num_envs, -1)
    depth_tensor = torch.nan_to_num(depth_tensor, nan=50.0, posinf=50.0, neginf=0.0)
    height, width = depth_camera.image_shape
    depth_tensor = depth_tensor.view(-1, 1, height, width)
    processed_depth, _ = _preprocess_depth_pits_to_obstacles(depth_tensor)

    model = nav_observations.JIT_DEPTH_NOISE_GENERATOR
    assert model is not None, "Depth encoder model is not initialized."
    if nav_observations.use_jit:
        encoded_depth_tensor = model(processed_depth)
    else:
        encoded_depth_tensor, _ = model(processed_depth)

    encoded_depth_tensor_reshaped = encoded_depth_tensor.view(env.num_envs, -1)
    camera_name = sensor_cfg.name if hasattr(sensor_cfg, "name") else "depth"
    return env.delay_manager.compute_delayed_depth(encoded_depth_tensor_reshaped, camera_name)


def try_load_torchscript_policy(checkpoint_path: str, device: str | torch.device):
    try:
        policy = torch.jit.load(checkpoint_path, map_location=device)
    except (RuntimeError, ValueError, TypeError, OSError):
        return None

    policy.eval()
    print(f"[INFO] Loaded TorchScript policy from: {checkpoint_path}")
    return policy


def find_latest_checkpoint(log_path: str, checkpoint_pattern: str = "model_.*.pt") -> str:
    if not os.path.exists(log_path):
        raise ValueError(f"Log path does not exist: {log_path}")

    run_dirs = []
    for entry in os.scandir(log_path):
        if entry.is_dir() and re.match(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", entry.name):
            run_dirs.append(entry.name)

    if not run_dirs:
        raise ValueError(f"No run directories found in: {log_path}")

    run_dirs.sort()
    latest_run = run_dirs[-1]
    run_path = os.path.join(log_path, latest_run)

    checkpoint_files = [f for f in os.listdir(run_path) if re.match(checkpoint_pattern, f)]
    if not checkpoint_files:
        raise ValueError(f"No checkpoint files matching '{checkpoint_pattern}' found in: {run_path}")

    checkpoint_files.sort(key=lambda name: f"{name:0>15}")
    return os.path.join(run_path, checkpoint_files[-1])


def load_checkpoint_with_fallback(runner: OnPolicyRunner, checkpoint_path: str):
    print(f"[INFO] Loading checkpoint from: {checkpoint_path}")
    loaded_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if runner.is_mdpo:
        runner.alg.actor_critic_1.load_state_dict(loaded_dict["model_state_dict"], strict=True)
        runner.alg.actor_critic_2.load_state_dict(loaded_dict["model_state_dict"], strict=True)
    else:
        runner.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"], strict=True)

    if runner.empirical_normalization:
        runner.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
        runner.critic_obs_normalizer.load_state_dict(loaded_dict["critic_obs_norm_state_dict"])

    runner.current_learning_iteration = loaded_dict["iter"]
    print(f"[INFO] Loaded checkpoint from iteration {loaded_dict['iter']}")


def load_checkpoint_dict(checkpoint_path: str) -> dict:
    return torch.load(checkpoint_path, map_location="cpu", weights_only=False)


def maybe_apply_legacy_checkpoint_policy_overrides(
    env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg, checkpoint_dict: dict
):
    model_state_dict = checkpoint_dict.get("model_state_dict", {})

    has_actor_height_cnn = any(key.startswith("actor_height_cnn.") for key in model_state_dict)
    if getattr(agent_cfg.policy, "use_actor_height_input", False) and not has_actor_height_cnn:
        print("[INFO] Legacy checkpoint detected; disabling actor-only height branch for compatibility.")
        agent_cfg.policy.use_actor_height_input = False
        agent_cfg.policy.actor_height_input_dims = None
        agent_cfg.policy.actor_height_hidden_dims = None
        if hasattr(env_cfg, "observations") and hasattr(env_cfg.observations, "policy"):
            if hasattr(env_cfg.observations.policy, "height_scan_actor"):
                env_cfg.observations.policy.height_scan_actor = None

    has_visibility_cnn = any(key.startswith("visibility_map_encoder.") for key in model_state_dict)
    if getattr(agent_cfg.policy, "use_actor_visibility_map", False) and not has_visibility_cnn:
        print("[INFO] Legacy checkpoint detected; disabling actor visibility-map branch for compatibility.")
        agent_cfg.policy.use_actor_visibility_map = False
        if hasattr(env_cfg, "observations") and hasattr(env_cfg.observations, "policy"):
            if hasattr(env_cfg.observations.policy, "visibility_map"):
                env_cfg.observations.policy.visibility_map = None


def maybe_apply_eval_overrides(env_cfg: ManagerBasedRLEnvCfg):
    if hasattr(env_cfg, "observations") and hasattr(env_cfg.observations, "policy"):
        if hasattr(env_cfg.observations.policy, "enable_corruption"):
            env_cfg.observations.policy.enable_corruption = False

    if hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "robot_goal"):
        if hasattr(env_cfg.commands.robot_goal, "debug_vis"):
            env_cfg.commands.robot_goal.debug_vis = False

    if hasattr(env_cfg, "events"):
        for event_name in ("push_robot", "base_external_force_torque"):
            if hasattr(env_cfg.events, event_name):
                setattr(env_cfg.events, event_name, None)


def compute_bucket_sizes(num_cols: int, proportions: list[float]) -> list[int]:
    raw = [num_cols * proportion for proportion in proportions]
    base = [math.floor(value) for value in raw]
    remaining = num_cols - sum(base)
    remainders = sorted(
        enumerate([value - math.floor(value) for value in raw]),
        key=lambda item: item[1],
        reverse=True,
    )
    for index, _ in remainders[:remaining]:
        base[index] += 1
    return base


def build_terrain_type_lookup(env) -> dict[int, str]:
    terrain = getattr(env.unwrapped.scene, "terrain", None)
    terrain_cfg = getattr(terrain.cfg, "terrain_generator", None) if terrain is not None else None
    if terrain is None or terrain_cfg is None or not hasattr(terrain, "terrain_types"):
        return {}

    sub_terrains = getattr(terrain_cfg, "sub_terrains", None)
    if not sub_terrains:
        return {}

    terrain_names = list(sub_terrains.keys())
    terrain_proportions = [sub_terrains[name].proportion for name in terrain_names]
    bucket_sizes = compute_bucket_sizes(terrain_cfg.num_cols, terrain_proportions)

    lookup = {}
    current = 0
    for name, bucket_size in zip(terrain_names, bucket_sizes):
        for terrain_type in range(current, current + bucket_size):
            lookup[terrain_type] = name
        current += bucket_size
    return lookup


def terrain_geometry_signature(env) -> dict[str, object]:
    terrain = getattr(env.unwrapped.scene, "terrain", None)
    if terrain is None:
        return {"available": False}

    parts: list[tuple[str, object]] = []
    for name in (
        "_height_field_visual",
        "_height_field_valid_mask",
        "terrain_origins",
        "terrain_levels",
        "terrain_types",
    ):
        value = getattr(terrain, name, None)
        arr = _to_numpy(value)
        if arr is not None:
            parts.append((name, np.ascontiguousarray(arr)))

    digest = hashlib.sha256()
    shapes = {}
    for name, arr in parts:
        digest.update(name.encode("utf-8"))
        digest.update(str(arr.shape).encode("utf-8"))
        digest.update(str(arr.dtype).encode("utf-8"))
        digest.update(arr.tobytes())
        shapes[name] = list(arr.shape)

    return {
        "available": bool(parts),
        "sha256": digest.hexdigest() if parts else None,
        "shapes": shapes,
    }


def summarize_rates(successes: int, total: int) -> float:
    return float(successes) / float(total) if total > 0 else 0.0


def _tensor_to_list(tensor: torch.Tensor) -> list[float]:
    return [float(value) for value in tensor.detach().cpu().tolist()]


def _safe_body_name(body_names: list[str], body_id: int) -> str:
    if 0 <= body_id < len(body_names):
        return body_names[body_id]
    return f"body_{body_id}"


def _to_numpy(data) -> np.ndarray:
    if data is None:
        return None
    if torch.is_tensor(data):
        return data.detach().cpu().numpy()
    return np.asarray(data)


def _quat_wxyz_to_yaw(quat_wxyz: np.ndarray) -> float:
    qw, qx, qy, qz = [float(value) for value in quat_wxyz]
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def _quat_wxyz_to_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = [float(value) for value in quat_wxyz]
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm <= 1.0e-8:
        return np.eye(3, dtype=np.float32)
    qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm
    return np.array(
        [
            [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw), 2.0 * (qx * qz + qy * qw)],
            [2.0 * (qx * qy + qz * qw), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw)],
            [2.0 * (qx * qz - qy * qw), 2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx * qx + qy * qy)],
        ],
        dtype=np.float32,
    )


def create_failure_rgb_camera(env) -> Camera:
    count = int(getattr(create_failure_rgb_camera, "_count", 0))
    create_failure_rgb_camera._count = count + 1
    camera_cfg = CameraCfg(
        prim_path=f"/World/FailureCaptureCamera_{count}",
        update_period=0,
        update_latest_camera_pose=True,
        height=int(args_cli.failure_rgb_height),
        width=int(args_cli.failure_rgb_width),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=float(args_cli.failure_rgb_focal_length),
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 1.0e5),
        ),
    )
    camera = Camera(cfg=camera_cfg)
    if not camera.is_initialized:
        camera._initialize_callback(None)
    for _ in range(2):
        env.unwrapped.sim.render()
    camera.update(env.unwrapped.physics_dt, force_recompute=True)
    return camera


def _save_rgb_tensor_to_png(rgb_tensor: torch.Tensor, output_path: str):
    from PIL import Image

    rgb_np = rgb_tensor.detach().cpu().numpy()
    if rgb_np.shape[-1] == 4:
        rgb_np = rgb_np[..., :3]
    rgb_np = np.asarray(rgb_np, dtype=np.uint8)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    Image.fromarray(rgb_np).save(output_path)


def _rgb_tensor_to_numpy(rgb_tensor: torch.Tensor) -> np.ndarray:
    rgb_np = rgb_tensor.detach().cpu().numpy()
    if rgb_np.shape[-1] == 4:
        rgb_np = rgb_np[..., :3]
    return np.asarray(rgb_np, dtype=np.uint8)


def _save_rgb_frames_to_gif(rgb_frames: list[np.ndarray], output_path: str, frame_duration_ms: int):
    from PIL import Image

    if not rgb_frames:
        return
    pil_frames = [Image.fromarray(frame) for frame in rgb_frames]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=max(1, int(frame_duration_ms)),
        loop=0,
    )


def _capture_observation_delay_snapshot(env) -> dict[str, torch.Tensor] | None:
    delay_manager = getattr(env.unwrapped, "delay_manager", None)
    if delay_manager is None or not getattr(delay_manager, "enabled", False):
        return None

    def _clone_lags(state) -> torch.Tensor | None:
        if state is None or not hasattr(state, "time_lags"):
            return None
        time_lags = getattr(state, "time_lags")
        if time_lags is None:
            return None
        return time_lags.detach().clone()

    snapshot = {
        "lin_vel": _clone_lags(getattr(delay_manager, "_delay_lin_vel", None)),
        "ang_vel": _clone_lags(getattr(delay_manager, "_delay_ang_vel", None)),
        "projected_gravity": _clone_lags(getattr(delay_manager, "_delay_projected_gravity", None)),
        "target_position": _clone_lags(getattr(delay_manager, "_delay_target_position", None)),
        "depth": None,
    }
    depth_buffers = getattr(delay_manager, "_delay_depth_buffers", {})
    if hasattr(depth_buffers, "get"):
        depth_state = depth_buffers.get("raycast_camera")
        if depth_state is None and len(depth_buffers) == 1:
            depth_state = next(iter(depth_buffers.values()))
        snapshot["depth"] = _clone_lags(depth_state)
    return snapshot


def _append_observation_delay_suffix(
    output_stem: str,
    env,
    env_id: int,
    delay_snapshot: dict[str, torch.Tensor] | None = None,
) -> str:
    delay_manager = getattr(env.unwrapped, "delay_manager", None)
    if delay_manager is None or not getattr(delay_manager, "enabled", False):
        return f"{output_stem}__obsdelay_off"

    def _lag_value_from_tensor(time_lags) -> str:
        if time_lags is None:
            return "na"
        if time_lags is None or env_id < 0 or env_id >= int(time_lags.shape[0]):
            return "na"
        return str(int(time_lags[env_id].item()))

    if delay_snapshot is None:
        delay_snapshot = _capture_observation_delay_snapshot(env)
    if delay_snapshot is None:
        return f"{output_stem}__obsdelay_na"

    return (
        f"{output_stem}__obsdelay"
        f"_lv{_lag_value_from_tensor(delay_snapshot.get('lin_vel'))}"
        f"_av{_lag_value_from_tensor(delay_snapshot.get('ang_vel'))}"
        f"_pg{_lag_value_from_tensor(delay_snapshot.get('projected_gravity'))}"
        f"_tp{_lag_value_from_tensor(delay_snapshot.get('target_position'))}"
        f"_d{_lag_value_from_tensor(delay_snapshot.get('depth'))}"
    )


def _delay_snapshot_for_env(delay_snapshot: dict[str, torch.Tensor] | None, env_id: int) -> dict[str, int | None]:
    def _value(name: str) -> int | None:
        if delay_snapshot is None:
            return None
        values = delay_snapshot.get(name)
        if values is None or env_id < 0 or env_id >= int(values.shape[0]):
            return None
        return int(values[env_id].item())

    return {
        "lin_vel": _value("lin_vel"),
        "ang_vel": _value("ang_vel"),
        "projected_gravity": _value("projected_gravity"),
        "target_position": _value("target_position"),
        "depth": _value("depth"),
    }


def _snapshot_vector(snapshot: dict[str, torch.Tensor], key: str, env_id: int) -> list[float] | None:
    value = snapshot.get(key)
    if value is None or env_id < 0 or env_id >= int(value.shape[0]):
        return None
    return [float(x) for x in value[env_id].detach().cpu().tolist()]


def _snapshot_scalar(snapshot: dict[str, torch.Tensor], key: str, env_id: int) -> float | None:
    value = snapshot.get(key)
    if value is None or env_id < 0 or env_id >= int(value.shape[0]):
        return None
    return float(value[env_id].detach().cpu().item())


def _sample_height_field_z_for_positions(
    env,
    snapshot: dict[str, torch.Tensor],
    env_ids: torch.Tensor,
    positions_world: torch.Tensor,
) -> torch.Tensor | None:
    terrain = getattr(env.unwrapped.scene, "terrain", None)
    heights_raw = getattr(terrain, "_height_field_visual", None) if terrain is not None else None
    terrain_origins_raw = getattr(terrain, "terrain_origins", None) if terrain is not None else None
    terrain_levels = snapshot.get("terrain_levels")
    terrain_types = snapshot.get("terrain_types")
    if heights_raw is None or terrain_origins_raw is None or terrain_levels is None or terrain_types is None:
        return None

    terrain_cfg = getattr(getattr(terrain, "cfg", None), "terrain_generator", None)
    if terrain_cfg is None or not hasattr(terrain_cfg, "horizontal_scale") or not hasattr(terrain_cfg, "size"):
        return None

    device = positions_world.device
    heights = heights_raw.to(device=device) if isinstance(heights_raw, torch.Tensor) else torch.as_tensor(heights_raw, device=device)
    terrain_origins = (
        terrain_origins_raw.to(device=device)
        if isinstance(terrain_origins_raw, torch.Tensor)
        else torch.as_tensor(terrain_origins_raw, device=device)
    )
    levels = terrain_levels[env_ids].to(device=device, dtype=torch.long)
    types = terrain_types[env_ids].to(device=device, dtype=torch.long)
    terrain_indices = levels * int(terrain_origins.shape[1]) + types

    horizontal_scale = float(terrain_cfg.horizontal_scale)
    terrain_size = terrain_cfg.size
    mesh_center_x = float(terrain_size[0]) * 0.5
    mesh_center_y = float(terrain_size[1]) * 0.5
    # PositionSampler uses the same one-cell mesh border offset for local xy <-> height-field index conversion.
    border_pixels = 1

    origins = terrain_origins[levels, types]
    local_xy = positions_world[env_ids, :2] - origins[:, :2]
    x_idx = torch.round((local_xy[:, 0] + mesh_center_x) / horizontal_scale - border_pixels).to(dtype=torch.long)
    y_idx = torch.round((local_xy[:, 1] + mesh_center_y) / horizontal_scale - border_pixels).to(dtype=torch.long)
    x_idx = torch.clamp(x_idx, 0, int(heights.shape[1]) - 1)
    y_idx = torch.clamp(y_idx, 0, int(heights.shape[2]) - 1)

    vertical_scale = float(getattr(terrain_cfg, "vertical_scale", 0.005))
    return heights[terrain_indices, x_idx, y_idx].float() * vertical_scale


def _write_case_metadata(output_path: str, metadata: dict[str, object]):
    metadata_path = os.path.splitext(output_path)[0] + ".json"
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)


def _tensor_row_to_list(tensor: torch.Tensor | None, env_id: int) -> list[float] | None:
    if tensor is None or env_id < 0 or env_id >= int(tensor.shape[0]):
        return None
    return [float(x) for x in tensor[env_id].detach().cpu().tolist()]


def _get_navigation_action_term(env):
    terms = getattr(env.unwrapped.action_manager, "_terms", {})
    if hasattr(terms, "get"):
        return terms.get("velocity_command")
    return None


def _get_raycast_camera(env):
    sensors = getattr(env.unwrapped.scene, "sensors", {})
    if hasattr(sensors, "get"):
        return sensors.get("raycast_camera")
    return None


def _capture_initial_case_manifest(env, num_cases: int) -> dict[str, object]:
    num_cases = min(int(num_cases), int(env.unwrapped.num_envs))
    case_ids = list(range(num_cases))
    robot = env.unwrapped.scene["robot"]
    terrain = getattr(env.unwrapped.scene, "terrain", None)
    goal_term = env.unwrapped.command_manager._terms["robot_goal"]
    delay_snapshot = _capture_observation_delay_snapshot(env)
    action_term = _get_navigation_action_term(env)
    camera = _get_raycast_camera(env)

    cases = []
    for env_id in case_ids:
        case = {
            "case_index": env_id + 1,
            "env_id": env_id,
            "terrain_level": int(terrain.terrain_levels[env_id].item()) if terrain is not None and hasattr(terrain, "terrain_levels") else None,
            "terrain_type_index": int(terrain.terrain_types[env_id].item()) if terrain is not None and hasattr(terrain, "terrain_types") else None,
            "spawn_position_world": _tensor_row_to_list(goal_term.spawn_position_world, env_id),
            "goal_position_world": _tensor_row_to_list(goal_term.goal_position_world, env_id),
            "root_pos_world": _tensor_row_to_list(robot.data.root_pos_w, env_id),
            "root_quat_world": _tensor_row_to_list(robot.data.root_quat_w, env_id),
            "root_lin_vel_world": _tensor_row_to_list(robot.data.root_lin_vel_w, env_id),
            "root_ang_vel_world": _tensor_row_to_list(robot.data.root_ang_vel_w, env_id),
            "joint_pos": _tensor_row_to_list(robot.data.joint_pos, env_id),
            "joint_vel": _tensor_row_to_list(robot.data.joint_vel, env_id),
            "observation_delay": _delay_snapshot_for_env(delay_snapshot, env_id),
        }
        if action_term is not None:
            case["policy_scaling"] = _tensor_row_to_list(getattr(action_term, "_policy_scaling", None), env_id)
            case["policy_bias"] = _tensor_row_to_list(getattr(action_term, "_policy_bias", None), env_id)
            case["low_pass_alpha"] = _tensor_row_to_list(
                getattr(action_term, "_per_env_per_dim_low_pass_alpha", None), env_id
            )
        if camera is not None:
            case["camera_offset_pos"] = _tensor_row_to_list(getattr(camera, "_offset_pos", None), env_id)
            case["camera_offset_quat"] = _tensor_row_to_list(getattr(camera, "_offset_quat", None), env_id)
        cases.append(case)

    return {
        "seed": None if args_cli.seed is None else int(args_cli.seed),
        "num_envs": int(env.unwrapped.num_envs),
        "num_cases": num_cases,
        "terrain_geometry": terrain_geometry_signature(env),
        "cases": cases,
    }


def _write_case_manifest(env, path: str, num_cases: int):
    manifest = _capture_initial_case_manifest(env, num_cases)
    path = os.path.abspath(os.path.expanduser(path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"[INFO] Wrote initial case manifest to: {path}")


def _set_delay_lags_from_case(delay_snapshot: dict[str, int | None], env_id_tensor: torch.Tensor, env):
    delay_manager = getattr(env.unwrapped, "delay_manager", None)
    if delay_manager is None or not getattr(delay_manager, "enabled", False):
        return

    def _set_one(state, key: str):
        value = delay_snapshot.get(key)
        if state is None or value is None:
            return
        state.set_lags(torch.tensor([int(value)], device=env.unwrapped.device, dtype=torch.int), env_id_tensor)

    _set_one(getattr(delay_manager, "_delay_lin_vel", None), "lin_vel")
    _set_one(getattr(delay_manager, "_delay_ang_vel", None), "ang_vel")
    _set_one(getattr(delay_manager, "_delay_projected_gravity", None), "projected_gravity")
    _set_one(getattr(delay_manager, "_delay_target_position", None), "target_position")
    depth_buffers = getattr(delay_manager, "_delay_depth_buffers", {})
    if hasattr(depth_buffers, "get"):
        depth_state = depth_buffers.get("raycast_camera")
        if depth_state is None and hasattr(delay_manager, "get_or_create_depth_delay_buffer"):
            depth_state = delay_manager.get_or_create_depth_delay_buffer("raycast_camera")
        if depth_state is None and len(depth_buffers) == 1:
            depth_state = next(iter(depth_buffers.values()))
        _set_one(depth_state, "depth")


def _apply_initial_case_manifest_data(env, manifest: dict[str, object], source: str = "<memory>"):
    cases = manifest.get("cases", [])
    if len(cases) > int(env.unwrapped.num_envs):
        raise ValueError(
            f"Case manifest has {len(cases)} cases but env only has {env.unwrapped.num_envs} envs."
        )

    robot = env.unwrapped.scene["robot"]
    terrain = getattr(env.unwrapped.scene, "terrain", None)
    goal_term = env.unwrapped.command_manager._terms["robot_goal"]
    action_term = _get_navigation_action_term(env)
    camera = _get_raycast_camera(env)
    device = env.unwrapped.device

    for case in cases:
        env_id = int(case["env_id"])
        env_id_tensor = torch.tensor([env_id], device=device, dtype=torch.long)

        spawn = torch.tensor(case["spawn_position_world"], device=device, dtype=torch.float32)
        goal = torch.tensor(case["goal_position_world"], device=device, dtype=torch.float32)
        root_pos = torch.tensor(case["root_pos_world"], device=device, dtype=torch.float32)
        root_quat = torch.tensor(case["root_quat_world"], device=device, dtype=torch.float32)
        root_lin_vel = torch.tensor(case["root_lin_vel_world"], device=device, dtype=torch.float32)
        root_ang_vel = torch.tensor(case["root_ang_vel_world"], device=device, dtype=torch.float32)
        joint_pos = torch.tensor(case["joint_pos"], device=device, dtype=torch.float32).unsqueeze(0)
        joint_vel = torch.tensor(case["joint_vel"], device=device, dtype=torch.float32).unsqueeze(0)

        goal_term.spawn_position_world[env_id] = spawn
        goal_term.goal_position_world[env_id] = goal
        if terrain is not None and hasattr(terrain, "env_origins"):
            terrain.env_origins[env_id] = spawn

        robot.write_root_pose_to_sim(torch.cat((root_pos, root_quat)).unsqueeze(0), env_ids=env_id_tensor)
        robot.write_root_velocity_to_sim(torch.cat((root_lin_vel, root_ang_vel)).unsqueeze(0), env_ids=env_id_tensor)
        robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_id_tensor)
        robot.set_joint_position_target(joint_pos, env_ids=env_id_tensor)
        robot.set_joint_velocity_target(joint_vel, env_ids=env_id_tensor)

        _set_delay_lags_from_case(case.get("observation_delay", {}), env_id_tensor, env)

        if action_term is not None:
            if case.get("policy_scaling") is not None and hasattr(action_term, "_policy_scaling"):
                action_term._policy_scaling[env_id] = torch.tensor(case["policy_scaling"], device=device, dtype=torch.float32)
            if case.get("policy_bias") is not None and hasattr(action_term, "_policy_bias"):
                action_term._policy_bias[env_id] = torch.tensor(case["policy_bias"], device=device, dtype=torch.float32)
            if case.get("low_pass_alpha") is not None and hasattr(action_term, "_per_env_per_dim_low_pass_alpha"):
                action_term._per_env_per_dim_low_pass_alpha[env_id] = torch.tensor(
                    case["low_pass_alpha"], device=device, dtype=torch.float32
                )
            if hasattr(action_term, "reset_low_pass_filter"):
                action_term.reset_low_pass_filter(env_id_tensor)

        if camera is not None:
            if case.get("camera_offset_pos") is not None and hasattr(camera, "_offset_pos"):
                camera._offset_pos[env_id] = torch.tensor(case["camera_offset_pos"], device=device, dtype=torch.float32)
            if case.get("camera_offset_quat") is not None and hasattr(camera, "_offset_quat"):
                camera._offset_quat[env_id] = torch.tensor(case["camera_offset_quat"], device=device, dtype=torch.float32)

    if hasattr(goal_term, "_reset_tracking_state"):
        goal_term._reset_tracking_state(
            torch.tensor([int(case["env_id"]) for case in cases], device=device, dtype=torch.long)
        )
    robot.write_data_to_sim()
    env.unwrapped.sim.forward()
    print(f"[INFO] Applied initial case manifest from: {source}")


def _apply_initial_case_manifest(env, path: str):
    path = os.path.abspath(os.path.expanduser(path))
    with open(path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    _apply_initial_case_manifest_data(env, manifest, source=path)


def _save_rgb_array_to_png(rgb_frame: np.ndarray, output_path: str):
    from PIL import Image

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    Image.fromarray(np.asarray(rgb_frame, dtype=np.uint8)).save(output_path)


def _scalar_frame_to_rgb(frame: np.ndarray, value_min: float, value_max: float, cmap_name: str) -> np.ndarray:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import cm

    frame = np.asarray(frame, dtype=np.float32)
    frame = np.nan_to_num(frame, nan=value_max, posinf=value_max, neginf=value_min)
    scale = max(float(value_max) - float(value_min), 1.0e-6)
    normalized = np.clip((frame - float(value_min)) / scale, 0.0, 1.0)
    rgba = cm.get_cmap(cmap_name)(normalized)
    return np.asarray(rgba[..., :3] * 255.0, dtype=np.uint8)


class DepthPoseLocalTerrainMap:
    """Eval-only local terrain map built from delayed noisy depth and camera pose."""

    def __init__(
        self,
        num_envs: int,
        size_m: float,
        resolution_m: float,
        history_frames: int,
        stride: int,
        depth_min_m: float,
        depth_max_m: float,
        height_range_m: float,
        gradient_threshold_m: float,
    ):
        self.num_envs = int(num_envs)
        self.size_m = max(float(size_m), 0.5)
        self.resolution_m = max(float(resolution_m), 0.01)
        self.grid_size = max(8, int(round(self.size_m / self.resolution_m)))
        self.history_frames = max(1, int(history_frames))
        self.stride = max(1, int(stride))
        self.depth_min_m = max(float(depth_min_m), 0.0)
        self.depth_max_m = max(float(depth_max_m), self.depth_min_m + 1.0e-3)
        self.height_range_m = max(float(height_range_m), 0.1)
        self.gradient_threshold_m = max(float(gradient_threshold_m), 0.0)
        self.image_y_drop_threshold_m = 0.25
        self.image_y_drop_min_neighbor_pairs = 2
        self.image_y_drop_min_component_pixels = 5
        self.bev_gradient_min_points_per_cell = 1
        self.bev_gradient_min_neighbor_cells = 1
        self.bev_gradient_min_component_pixels = 2
        self._history: list[deque[np.ndarray]] = [deque(maxlen=self.history_frames) for _ in range(self.num_envs)]
        self._danger_history: list[deque[np.ndarray]] = [
            deque(maxlen=self.history_frames) for _ in range(self.num_envs)
        ]
        self._sensor_history: list[deque[dict[str, np.ndarray]]] = [
            deque(maxlen=max(1, self.history_frames + 4)) for _ in range(self.num_envs)
        ]
        self._pixel_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
        self._last_gradient_frames: list[np.ndarray | None] = [None] * self.num_envs
        self._last_danger_masks: list[np.ndarray | None] = [None] * self.num_envs
        self._last_depth_frames: list[np.ndarray | None] = [None] * self.num_envs

    def reset(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            for history in self._history:
                history.clear()
            for danger_history in self._danger_history:
                danger_history.clear()
            for sensor_history in self._sensor_history:
                sensor_history.clear()
            self._last_gradient_frames = [None] * self.num_envs
            self._last_danger_masks = [None] * self.num_envs
            self._last_depth_frames = [None] * self.num_envs
            return
        for env_id in env_ids.detach().cpu().tolist():
            if 0 <= int(env_id) < self.num_envs:
                self._history[int(env_id)].clear()
                self._danger_history[int(env_id)].clear()
                self._sensor_history[int(env_id)].clear()
                self._last_gradient_frames[int(env_id)] = None
                self._last_danger_masks[int(env_id)] = None
                self._last_depth_frames[int(env_id)] = None

    def update(self, env):
        sensors = getattr(env.unwrapped.scene, "sensors", None)
        if sensors is None or "raycast_camera" not in sensors:
            return
        depth_camera = sensors["raycast_camera"]
        if "distance_to_image_plane" not in depth_camera.data.output:
            return

        height, width = depth_camera.image_shape
        depth_tensor = depth_camera.data.output["distance_to_image_plane"].view(env.unwrapped.num_envs, 1, height, width)
        depth_tensor = torch.nan_to_num(depth_tensor, nan=50.0, posinf=50.0, neginf=0.0)
        depth_tensor = self._apply_policy_depth_noise(depth_tensor)
        depth_cpu = depth_tensor.detach().cpu().numpy()[:, 0]

        camera_pos_cpu = depth_camera.data.pos_w.detach().cpu().numpy().astype(np.float32, copy=False)
        camera_quat_cpu = depth_camera.data.quat_w_world.detach().cpu().numpy().astype(np.float32, copy=False)
        intrinsics_cpu = depth_camera.data.intrinsic_matrices.detach().cpu().numpy().astype(np.float32, copy=False)
        depth_lags = self._depth_lags(env)

        for env_id in range(min(self.num_envs, depth_cpu.shape[0])):
            self._sensor_history[env_id].append(
                {
                    "depth": depth_cpu[env_id].astype(np.float32, copy=True),
                    "camera_pos_w": camera_pos_cpu[env_id].copy(),
                    "camera_quat_w": camera_quat_cpu[env_id].copy(),
                    "intrinsics": intrinsics_cpu[env_id].copy(),
                }
            )
            lag = 0 if depth_lags is None else int(depth_lags[env_id])
            sensor_sample = self._delayed_sensor_sample(env_id, lag)
            if sensor_sample is None:
                continue
            backprojected = self._backproject_depth_sample(sensor_sample)
            if backprojected is None:
                continue
            points, danger_points, danger_mask = backprojected
            self._last_depth_frames[env_id] = np.asarray(sensor_sample["depth"], dtype=np.float32).copy()
            self._last_danger_masks[env_id] = danger_mask
            if points.size > 0:
                self._history[env_id].append(points)
            if danger_points.size > 0:
                self._danger_history[env_id].append(danger_points)

    def _apply_policy_depth_noise(self, depth_tensor: torch.Tensor) -> torch.Tensor:
        nav_observations._ensure_depth_noise_generator_initialized()
        min_depth_value = float(nav_observations.min_depth) if nav_observations.min_depth > 0.0 else 0.25
        max_depth_value = float(nav_observations.max_depth) if nav_observations.max_depth > 0.0 else 10.0
        model = nav_observations.JIT_DEPTH_NOISE_GENERATOR
        if model is None:
            return depth_tensor
        if nav_observations.use_jit:
            noisy_depth = depth_tensor.clone()
            noisy_depth[noisy_depth > max_depth_value] = 0.0
            noisy_depth[noisy_depth < min_depth_value] = 0.0
            return noisy_depth
        _, noisy_depth = model(depth_tensor)
        return noisy_depth

    def _depth_lags(self, env) -> np.ndarray | None:
        delay_manager = getattr(env.unwrapped, "delay_manager", None)
        if delay_manager is None or not getattr(delay_manager, "enabled", False):
            return None
        depth_buffers = getattr(delay_manager, "_delay_depth_buffers", {})
        depth_state = depth_buffers.get("raycast_camera") if hasattr(depth_buffers, "get") else None
        if depth_state is None and hasattr(delay_manager, "get_or_create_depth_delay_buffer"):
            depth_state = delay_manager.get_or_create_depth_delay_buffer("raycast_camera")
        if depth_state is None or not hasattr(depth_state, "time_lags"):
            return None
        return depth_state.time_lags.detach().cpu().numpy().astype(np.int32, copy=False)

    def _delayed_sensor_sample(self, env_id: int, lag: int) -> dict[str, np.ndarray] | None:
        history = self._sensor_history[env_id]
        if not history:
            return None
        lag = max(0, int(lag))
        if lag >= len(history):
            lag = 0
        return history[-1 - lag]

    def _pixel_grid(self, height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
        key = (int(height), int(width))
        if key not in self._pixel_cache:
            rows = np.arange(height, dtype=np.float32)[:, None]
            cols = np.arange(width, dtype=np.float32)[None, :]
            self._pixel_cache[key] = (rows, cols)
        return self._pixel_cache[key]

    def _backproject_depth_sample(self, sample: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        depth = np.asarray(sample["depth"], dtype=np.float32)
        height, width = depth.shape
        rows, cols = self._pixel_grid(height, width)
        intrinsics = np.asarray(sample["intrinsics"], dtype=np.float32)
        fx = max(float(intrinsics[0, 0]), 1.0e-6)
        fy = max(float(intrinsics[1, 1]), 1.0e-6)
        cx = float(intrinsics[0, 2])
        cy = float(intrinsics[1, 2])

        depth_sub = depth[:: self.stride, :: self.stride]
        rows_sub = rows[:: self.stride, :]
        cols_sub = cols[:, :: self.stride]
        valid = np.isfinite(depth_sub) & (depth_sub >= self.depth_min_m) & (depth_sub <= self.depth_max_m)
        if not np.any(valid):
            return None

        vertical_cam = ((rows - cy) / fy) * depth
        valid_vertical = np.isfinite(vertical_cam) & np.isfinite(depth) & (depth >= 0.25) & (depth <= 10.0)
        upper_is_lower = (
            valid_vertical[:-1, :]
            & valid_vertical[1:, :]
            & ((vertical_cam[1:, :] - vertical_cam[:-1, :]) < -self.image_y_drop_threshold_m)
        )
        if np.any(upper_is_lower):
            neighbor_pairs = ndi.convolve(upper_is_lower.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), mode="constant")
            upper_is_lower &= neighbor_pairs >= int(self.image_y_drop_min_neighbor_pairs)
        if np.any(upper_is_lower):
            labels, num_labels = ndi.label(upper_is_lower, structure=np.ones((3, 3), dtype=np.uint8))
            if num_labels > 0:
                component_sizes = np.bincount(labels.ravel())
                keep_labels = component_sizes >= int(self.image_y_drop_min_component_pixels)
                keep_labels[0] = False
                upper_is_lower = keep_labels[labels]
        danger_mask = np.zeros_like(depth, dtype=bool)
        if np.any(upper_is_lower):
            danger_mask[:-1, :] |= upper_is_lower
            danger_mask[1:, :] |= upper_is_lower
        danger_full = danger_mask & valid_vertical
        danger_full &= vertical_cam >= -0.25

        # Isaac "world" camera convention: +X forward, +Z up. Image right maps to -Y.
        x_cam = depth_sub
        y_cam = -((cols_sub - cx) / fx) * depth_sub
        z_cam = -((rows_sub - cy) / fy) * depth_sub
        x_cam_full = depth
        y_cam_full = -((cols - cx) / fx) * depth
        z_cam_full = -((rows - cy) / fy) * depth
        points_cam = np.stack(
            [
                x_cam[valid],
                y_cam[valid],
                z_cam[valid],
            ],
            axis=1,
        ).astype(np.float32, copy=False)

        rotation = _quat_wxyz_to_matrix(np.asarray(sample["camera_quat_w"], dtype=np.float32))
        camera_pos = np.asarray(sample["camera_pos_w"], dtype=np.float32)
        points_w = points_cam @ rotation.T + camera_pos[None, :]
        finite_mask = np.isfinite(points_w).all(axis=1)
        points_w = points_w[finite_mask].astype(np.float32, copy=False)

        danger_points_w = np.empty((0, 3), dtype=np.float32)
        if np.any(danger_full):
            danger_points_cam = np.stack(
                [
                    x_cam_full[danger_full],
                    y_cam_full[danger_full],
                    z_cam_full[danger_full],
                ],
                axis=1,
            ).astype(np.float32, copy=False)
            danger_points_w = danger_points_cam @ rotation.T + camera_pos[None, :]
            danger_finite = np.isfinite(danger_points_w).all(axis=1)
            danger_points_w = danger_points_w[danger_finite].astype(np.float32, copy=False)
        return points_w, danger_points_w, danger_full.astype(bool, copy=True)

    def danger_mask(self, env_id: int) -> np.ndarray | None:
        if env_id < 0 or env_id >= self.num_envs:
            return None
        return self._last_danger_masks[env_id]

    def depth_frame(self, env_id: int) -> np.ndarray | None:
        if env_id < 0 or env_id >= self.num_envs:
            return None
        return self._last_depth_frames[env_id]

    def render(self, env, env_id: int) -> np.ndarray | None:
        if env_id < 0 or env_id >= self.num_envs or not self._history[env_id]:
            return None

        robot = env.unwrapped.scene["robot"]
        root_pos = robot.data.root_pos_w[env_id].detach().cpu().numpy().astype(np.float32)
        root_quat = robot.data.root_quat_w[env_id].detach().cpu().numpy().astype(np.float32)
        yaw = _quat_wxyz_to_yaw(root_quat)
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)

        points_w = np.concatenate(list(self._history[env_id]), axis=0)
        rel = points_w - root_pos[None, :]
        # World to robot-yaw frame: +x is robot forward, +y is robot left.
        x_local = cos_yaw * rel[:, 0] + sin_yaw * rel[:, 1]
        y_local = -sin_yaw * rel[:, 0] + cos_yaw * rel[:, 1]
        z_rel = rel[:, 2]

        half_size = self.size_m / 2.0
        valid = (
            (x_local >= -half_size)
            & (x_local < half_size)
            & (y_local >= -half_size)
            & (y_local < half_size)
            & np.isfinite(z_rel)
        )
        if not np.any(valid):
            return None

        x_local = x_local[valid]
        y_local = y_local[valid]
        z_rel = z_rel[valid]
        xi = np.floor((x_local + half_size) / self.resolution_m).astype(np.int32)
        yi = np.floor((y_local + half_size) / self.resolution_m).astype(np.int32)
        inside = (xi >= 0) & (xi < self.grid_size) & (yi >= 0) & (yi < self.grid_size)
        if not np.any(inside):
            return None
        xi = xi[inside]
        yi = yi[inside]
        z_rel = z_rel[inside]

        flat_index = yi * self.grid_size + xi
        flat_count = np.bincount(flat_index, minlength=self.grid_size * self.grid_size).astype(np.float32)
        flat_sum = np.bincount(flat_index, weights=z_rel, minlength=self.grid_size * self.grid_size).astype(np.float32)
        observed = flat_count > 0.0
        height_grid = np.full(self.grid_size * self.grid_size, np.nan, dtype=np.float32)
        height_grid[observed] = flat_sum[observed] / flat_count[observed]
        height_grid = height_grid.reshape(self.grid_size, self.grid_size)
        observed_grid = observed.reshape(self.grid_size, self.grid_size)

        rgb = _scalar_frame_to_rgb(
            np.nan_to_num(height_grid, nan=0.0),
            value_min=-self.height_range_m,
            value_max=self.height_range_m,
            cmap_name="coolwarm",
        )
        rgb[~observed_grid] = np.array([20, 20, 24], dtype=np.uint8)

        steep = np.zeros_like(observed_grid, dtype=bool)
        flat_count_grid = flat_count.reshape(self.grid_size, self.grid_size)
        robust_observed_grid = observed_grid & (flat_count_grid >= float(self.bev_gradient_min_points_per_cell))
        valid_x = robust_observed_grid[1:, :] & robust_observed_grid[:-1, :]
        if np.any(valid_x):
            delta_x = np.abs(height_grid[1:, :] - height_grid[:-1, :])
            steep_x = valid_x & (delta_x > self.gradient_threshold_m)
            steep[1:, :] |= steep_x
            steep[:-1, :] |= steep_x
        valid_y = robust_observed_grid[:, 1:] & robust_observed_grid[:, :-1]
        if np.any(valid_y):
            delta_y = np.abs(height_grid[:, 1:] - height_grid[:, :-1])
            steep_y = valid_y & (delta_y > self.gradient_threshold_m)
            steep[:, 1:] |= steep_y
            steep[:, :-1] |= steep_y
        steep &= observed_grid
        if np.any(steep):
            neighbor_cells = ndi.convolve(steep.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), mode="constant")
            steep &= neighbor_cells >= int(self.bev_gradient_min_neighbor_cells)
        if np.any(steep):
            labels, num_labels = ndi.label(steep, structure=np.ones((3, 3), dtype=np.uint8))
            if num_labels > 0:
                component_sizes = np.bincount(labels.ravel())
                keep_labels = component_sizes >= int(self.bev_gradient_min_component_pixels)
                keep_labels[0] = False
                steep = keep_labels[labels]
        steep &= observed_grid
        gradient_rgb = np.zeros_like(rgb)
        gradient_rgb[~observed_grid] = np.array([20, 20, 24], dtype=np.uint8)
        gradient_rgb[observed_grid] = np.array([46, 50, 58], dtype=np.uint8)

        center = self.grid_size // 2
        radius = max(1, int(round(0.18 / self.resolution_m)))
        y0, y1 = max(0, center - radius), min(self.grid_size, center + radius + 1)
        x0, x1 = max(0, center - radius), min(self.grid_size, center + radius + 1)
        rgb[y0:y1, x0:x1] = np.array([255, 40, 60], dtype=np.uint8)
        gradient_rgb[y0:y1, x0:x1] = np.array([255, 40, 60], dtype=np.uint8)

        # Grid is indexed as [local_y, local_x]. Display as image rows=forward/back and cols=left/right.
        # Image up is robot forward (+x), and image right is robot right (-y).
        self._last_gradient_frames[env_id] = np.flip(np.flip(np.transpose(gradient_rgb, (1, 0, 2)), axis=0), axis=1)
        return np.flip(np.flip(np.transpose(rgb, (1, 0, 2)), axis=0), axis=1)

    def gradient_frame(self, env_id: int) -> np.ndarray | None:
        if env_id < 0 or env_id >= self.num_envs:
            return None
        return self._last_gradient_frames[env_id]


def _capture_depth_frame_rgb(env, env_id: int) -> np.ndarray | None:
    sensors = getattr(env.unwrapped.scene, "sensors", None)
    if sensors is None or "raycast_camera" not in sensors:
        return None

    depth_camera = sensors["raycast_camera"]
    if env_id < 0 or env_id >= int(env.unwrapped.num_envs):
        return None

    height, width = depth_camera.image_shape
    depth_tensor = depth_camera.data.output["distance_to_image_plane"][env_id].view(height, width)
    depth_np = depth_tensor.detach().cpu().numpy()

    return _scalar_frame_to_rgb(depth_np, value_min=0.0, value_max=10.0, cmap_name="plasma_r")


def _orient_height_scan_for_display(scan_np: np.ndarray) -> np.ndarray:
    """Orient GridPattern height scans as robot-centric BEV: up=+x forward, right=-y right."""
    return np.flip(np.flip(scan_np.T, axis=0), axis=1)


def _capture_height_frame_rgb(env, env_id: int, offset: float = 0.5) -> np.ndarray | None:
    sensors = getattr(env.unwrapped.scene, "sensors", None)
    if sensors is None or "height_scanner_critic" not in sensors:
        return None

    height_scanner = sensors["height_scanner_critic"]
    if env_id < 0 or env_id >= int(env.unwrapped.num_envs):
        return None

    scan_tensor = height_scanner.data.pos_w[env_id, 2] - height_scanner.data.ray_hits_w[env_id, ..., 2] - offset
    scan_tensor = torch.clamp(scan_tensor, min=-5.0, max=5.0)
    num_points = int(scan_tensor.shape[0])
    grid_size = int(num_points ** 0.5)
    if grid_size * grid_size != num_points:
        return None

    scan_np = scan_tensor.view(grid_size, grid_size).detach().cpu().numpy()
    scan_np = _orient_height_scan_for_display(scan_np)
    return _scalar_frame_to_rgb(scan_np, value_min=-5.0, value_max=5.0, cmap_name="viridis")


def _height_scanner_steep_params(env) -> tuple[float, float]:
    default_threshold = 0.30
    default_hard_change = 0.80
    reward_manager = getattr(env.unwrapped, "reward_manager", None)
    if reward_manager is None:
        return default_threshold, default_hard_change

    term_cfg = None
    find_term_cfg = getattr(reward_manager, "find_term_cfg", None)
    if callable(find_term_cfg):
        try:
            term_cfg = find_term_cfg("steep_height_change")
        except Exception:
            term_cfg = None
    if term_cfg is None:
        term_cfgs = getattr(reward_manager, "_term_cfgs", None)
        if isinstance(term_cfgs, dict):
            term_cfg = term_cfgs.get("steep_height_change")
        elif isinstance(term_cfgs, (list, tuple)):
            term_names = getattr(reward_manager, "_term_names", [])
            if "steep_height_change" in term_names:
                term_cfg = term_cfgs[list(term_names).index("steep_height_change")]

    params = getattr(term_cfg, "params", None) if term_cfg is not None else None
    if not isinstance(params, dict):
        return default_threshold, default_hard_change
    threshold = float(params.get("height_threshold", default_threshold))
    hard_change = float(params.get("hard_height_change", default_hard_change))
    if hard_change <= threshold:
        hard_change = threshold + 1.0e-6
    return threshold, hard_change


def _capture_steep_frame_rgbs(env, height_sensor) -> list[np.ndarray | None]:
    num_envs = int(env.unwrapped.num_envs)
    cached_frames: list[np.ndarray | None] = [None] * num_envs
    if height_sensor is None:
        return cached_frames

    ray_hits_z = height_sensor.data.ray_hits_w[..., 2]
    num_points = int(ray_hits_z.shape[-1])
    grid_size = int(num_points ** 0.5)
    if grid_size * grid_size != num_points:
        return cached_frames

    height_grid = ray_hits_z.view(num_envs, grid_size, grid_size)
    finite_mask = torch.isfinite(height_grid)
    fallback_height = height_sensor.data.pos_w[:, 2].view(num_envs, 1, 1)
    height_grid = torch.where(finite_mask, height_grid, fallback_height)

    height_threshold, hard_height_change = _height_scanner_steep_params(env)
    risk_scale = max(hard_height_change - height_threshold, 1.0e-6)
    diff_x = torch.abs(height_grid[:, 1:, :] - height_grid[:, :-1, :])
    diff_y = torch.abs(height_grid[:, :, 1:] - height_grid[:, :, :-1])
    risk_x = torch.clamp((diff_x - height_threshold) / risk_scale, min=0.0, max=1.0)
    risk_y = torch.clamp((diff_y - height_threshold) / risk_scale, min=0.0, max=1.0)

    cell_risk = torch.zeros_like(height_grid)
    cell_risk[:, 1:, :] = torch.maximum(cell_risk[:, 1:, :], risk_x)
    cell_risk[:, :-1, :] = torch.maximum(cell_risk[:, :-1, :], risk_x)
    cell_risk[:, :, 1:] = torch.maximum(cell_risk[:, :, 1:], risk_y)
    cell_risk[:, :, :-1] = torch.maximum(cell_risk[:, :, :-1], risk_y)
    cell_risk = cell_risk.detach().cpu()

    for env_id in range(min(num_envs, int(cell_risk.shape[0]))):
        risk_np = cell_risk[env_id].numpy()
        risk_np = _orient_height_scan_for_display(risk_np)
        cached_frames[env_id] = _scalar_frame_to_rgb(risk_np, value_min=0.0, value_max=1.0, cmap_name="inferno")
    return cached_frames


def _visibility_map_to_rgb(visibility_map: np.ndarray) -> np.ndarray:
    """Render a binary actor visibility observation as a robot-centric BEV panel."""
    visibility = np.asarray(visibility_map, dtype=np.float32)
    observed = visibility > 0.5
    rgb = np.zeros((*observed.shape, 3), dtype=np.uint8)
    rgb[~observed] = np.array([20, 20, 24], dtype=np.uint8)
    rgb[observed] = np.array([220, 235, 245], dtype=np.uint8)
    center_y = observed.shape[0] // 2
    center_x = observed.shape[1] // 2
    radius = max(1, min(observed.shape) // 32)
    y0, y1 = max(0, center_y - radius), min(observed.shape[0], center_y + radius + 1)
    x0, x1 = max(0, center_x - radius), min(observed.shape[1], center_x + radius + 1)
    rgb[y0:y1, x0:x1] = np.array([255, 40, 60], dtype=np.uint8)
    return rgb


def _visibility_score_map_to_rgb(visibility_map: np.ndarray) -> np.ndarray:
    """Render a continuous visibility-confidence (ScoreVis) map as a BEV heatmap."""
    visibility = np.asarray(visibility_map, dtype=np.float32)
    rgb = _scalar_frame_to_rgb(visibility, value_min=0.0, value_max=1.0, cmap_name="magma")
    center_y = visibility.shape[0] // 2
    center_x = visibility.shape[1] // 2
    radius = max(1, min(visibility.shape) // 32)
    y0, y1 = max(0, center_y - radius), min(visibility.shape[0], center_y + radius + 1)
    x0, x1 = max(0, center_x - radius), min(visibility.shape[1], center_x + radius + 1)
    rgb[y0:y1, x0:x1] = np.array([255, 40, 60], dtype=np.uint8)
    return rgb


def _scene_tokens_to_rgb(scene_tokens: np.ndarray, token_grid_size: int = 8, layout: str = "v1") -> np.ndarray:
    """Render scene-token observations as compact BEV heatmaps.

    ``layout="v1"`` renders the legacy 6D tokens ``[x, y, visible_ratio,
    dist_to_goal, cos_to_goal, sin_to_goal]`` side by side.

    ``layout="no_goal"`` renders the 6D no-goal risk tokens used by the
    v3/ScoreVis tasks: ``[x, y, visibility_score, depth_mean, depth_edge,
    frontier]``, showing the continuous visibility score and normalized depth.
    """
    tokens = np.asarray(scene_tokens, dtype=np.float32)
    if tokens.ndim == 1:
        if tokens.size % 6 != 0:
            raise ValueError(f"Expected flattened 6D scene tokens, got {tokens.size} values")
        tokens = tokens.reshape(-1, 6)
    expected_tokens = token_grid_size * token_grid_size
    if tokens.shape[0] != expected_tokens or tokens.shape[1] < 4:
        raise ValueError(f"Expected [{expected_tokens}, >=4] scene tokens, got {tokens.shape}")

    if layout == "no_goal":
        visibility_score = tokens[:, 2].reshape(token_grid_size, token_grid_size)
        depth_mean = tokens[:, 3].reshape(token_grid_size, token_grid_size)
        visibility_rgb = _scalar_frame_to_rgb(visibility_score, value_min=0.0, value_max=1.0, cmap_name="magma")
        depth_rgb = _scalar_frame_to_rgb(depth_mean, value_min=0.0, value_max=1.0, cmap_name="viridis")
        separator = np.full((token_grid_size, 1, 3), 245, dtype=np.uint8)
        return np.concatenate([visibility_rgb, separator, depth_rgb], axis=1)

    visible_ratio = tokens[:, 2].reshape(token_grid_size, token_grid_size)
    dist_to_goal = tokens[:, 3].reshape(token_grid_size, token_grid_size)
    visible_rgb = _scalar_frame_to_rgb(visible_ratio, value_min=0.0, value_max=1.0, cmap_name="magma")
    dist_rgb = _scalar_frame_to_rgb(dist_to_goal, value_min=0.0, value_max=4.0, cmap_name="viridis_r")
    separator = np.full((token_grid_size, 1, 3), 245, dtype=np.uint8)
    return np.concatenate([visible_rgb, separator, dist_rgb], axis=1)


def _overlay_danger_mask_on_depth_rgb(depth_rgb: np.ndarray, danger_mask: np.ndarray | None) -> np.ndarray:
    if danger_mask is None:
        return depth_rgb
    mask = np.asarray(danger_mask, dtype=bool)
    if mask.shape != depth_rgb.shape[:2] or not np.any(mask):
        return depth_rgb
    mask = ndi.binary_dilation(mask, iterations=1)
    overlaid = depth_rgb.copy()
    danger_color = np.array([0, 255, 255], dtype=np.uint8)
    overlaid[mask] = (0.25 * overlaid[mask].astype(np.float32) + 0.75 * danger_color.astype(np.float32)).astype(np.uint8)
    return overlaid


def _resolve_depth_visualization_sensor_cfg(env) -> SceneEntityCfg | None:
    """Resolve the depth sensor used by the active high-level navigation observation."""
    sensors = getattr(env.unwrapped.scene, "sensors", {})
    policy_group = getattr(getattr(env.unwrapped.cfg, "observations", None), "policy", None)
    depth_term = getattr(policy_group, "depth_image", None) if policy_group is not None else None
    sensor_cfg = None
    if depth_term is not None:
        params = getattr(depth_term, "params", {}) or {}
        sensor_cfg = params.get("sensor_cfg")
        sensor_name = getattr(sensor_cfg, "name", None)
        if sensor_name is not None and hasattr(sensors, "get") and sensors.get(sensor_name) is not None:
            return sensor_cfg

    for sensor_name in ("raycast_camera", "camera"):
        if hasattr(sensors, "get") and sensors.get(sensor_name) is not None:
            return SceneEntityCfg(sensor_name)
    return None


def _capture_observation_frame_cache(
    env, local_terrain_map: DepthPoseLocalTerrainMap | None = None
) -> dict[str, list[np.ndarray | None]]:
    """Cache rendered depth/height observation frames for all envs without forcing sensor updates."""
    num_envs = int(env.unwrapped.num_envs)
    cached_frames: dict[str, list[np.ndarray | None]] = {
        "depth": [None] * num_envs,
        "processed_depth": [None] * num_envs,
        "height": [None] * num_envs,
        "steep": [None] * num_envs,
        "local_map": [None] * num_envs,
        "local_gradient": [None] * num_envs,
        "visibility_obs": [None] * num_envs,
        "scene_tokens": [None] * num_envs,
    }
    sensors = getattr(env.unwrapped.scene, "sensors", {})

    depth_sensor_cfg = _resolve_depth_visualization_sensor_cfg(env)
    depth_sensor_name = getattr(depth_sensor_cfg, "name", None)
    depth_sensor = sensors.get(depth_sensor_name) if depth_sensor_name is not None and hasattr(sensors, "get") else None
    if depth_sensor is not None and "distance_to_image_plane" in depth_sensor.data.output:
        height, width = depth_sensor.image_shape
        depth_tensor_device = depth_sensor.data.output["distance_to_image_plane"].view(num_envs, 1, height, width)
        display_depth_tensor = depth_tensor_device
        try:
            noisy_sample = nav_observations._get_noisy_depth_sample(env.unwrapped, depth_sensor_cfg)
            noisy_depth = noisy_sample["depth"]
            if noisy_depth.dim() in (2, 3):
                noisy_depth = noisy_depth.view(num_envs, 1, height, width)
            elif noisy_depth.dim() != 4:
                raise ValueError(f"unexpected noisy depth shape {tuple(noisy_depth.shape)}")

            noisy_history = getattr(_capture_observation_frame_cache, "_noisy_depth_history", None)
            if noisy_history is None:
                noisy_history = deque(maxlen=max(1, int(args_cli.failure_rgb_history_frames)))
                _capture_observation_frame_cache._noisy_depth_history = noisy_history
            noisy_history.append(noisy_depth.detach().clone())

            depth_lags = None
            delay_manager = getattr(env.unwrapped, "delay_manager", None)
            if delay_manager is not None and getattr(delay_manager, "enabled", False):
                depth_buffers = getattr(delay_manager, "_delay_depth_buffers", {})
                depth_state = depth_buffers.get(depth_sensor_name) if depth_sensor_name is not None and hasattr(depth_buffers, "get") else None
                if depth_state is None and hasattr(depth_buffers, "__len__") and len(depth_buffers) == 1:
                    depth_state = next(iter(depth_buffers.values()))
                if depth_state is not None and hasattr(depth_state, "time_lags"):
                    depth_lags = depth_state.time_lags
            display_depth_tensor = select_delayed_depth_frames(noisy_history, depth_lags)
        except Exception as exc:
            if not getattr(_capture_observation_frame_cache, "_warned_noisy_depth_fallback", False):
                print(f"[WARN] Falling back to raw depth visualization; noisy/delayed depth unavailable: {exc}")
                _capture_observation_frame_cache._warned_noisy_depth_fallback = True
        processed_depth_tensor = None
        if args_cli.depth_pit_to_obstacle:
            processed_depth_tensor, _ = _preprocess_depth_pits_to_obstacles(display_depth_tensor)
            processed_depth_tensor = processed_depth_tensor.detach().cpu()
        depth_tensor = display_depth_tensor.detach().cpu()
        for env_id in range(min(num_envs, int(depth_tensor.shape[0]))):
            depth_np = depth_tensor[env_id].view(height, width).numpy()
            if local_terrain_map is not None:
                delayed_depth_np = local_terrain_map.depth_frame(env_id)
                if delayed_depth_np is not None and delayed_depth_np.shape == depth_np.shape:
                    depth_np = delayed_depth_np
            depth_rgb = _scalar_frame_to_rgb(depth_np, value_min=0.0, value_max=10.0, cmap_name="plasma_r")
            if local_terrain_map is not None:
                depth_rgb = _overlay_danger_mask_on_depth_rgb(depth_rgb, local_terrain_map.danger_mask(env_id))
            cached_frames["depth"][env_id] = depth_rgb
            if processed_depth_tensor is not None:
                processed_depth_np = processed_depth_tensor[env_id].view(height, width).numpy()
                cached_frames["processed_depth"][env_id] = _scalar_frame_to_rgb(
                    processed_depth_np, value_min=0.0, value_max=10.0, cmap_name="plasma_r"
                )

    height_sensor = sensors.get("height_scanner_critic") if hasattr(sensors, "get") else None
    if height_sensor is not None:
        scan_tensor = height_sensor.data.pos_w[:, 2:3] - height_sensor.data.ray_hits_w[..., 2] - 0.5
        scan_tensor = torch.clamp(scan_tensor, min=-5.0, max=5.0).detach().cpu()
        num_points = int(scan_tensor.shape[-1])
        grid_size = int(num_points ** 0.5)
        if grid_size * grid_size == num_points:
            for env_id in range(min(num_envs, int(scan_tensor.shape[0]))):
                scan_np = scan_tensor[env_id].view(grid_size, grid_size).numpy()
                scan_np = _orient_height_scan_for_display(scan_np)
                cached_frames["height"][env_id] = _scalar_frame_to_rgb(
                    scan_np, value_min=-5.0, value_max=5.0, cmap_name="viridis"
                )
        cached_frames["steep"] = _capture_steep_frame_rgbs(env, height_sensor)

    if local_terrain_map is not None:
        for env_id in range(num_envs):
            cached_frames["local_map"][env_id] = local_terrain_map.render(env, env_id)
            cached_frames["local_gradient"][env_id] = local_terrain_map.gradient_frame(env_id)

    if depth_sensor is not None and depth_sensor_cfg is not None:
        # Match the scene-token observation term the environment actually uses so
        # the GIF panels show the same visibility decay / token layout as the
        # policy. ScoreVis tasks set decay=0.995 (continuous confidence).
        policy_obs_cfg = getattr(env.unwrapped.cfg.observations, "policy", None)
        scene_tokens_term = getattr(policy_obs_cfg, "scene_tokens", None)
        token_func_name = getattr(getattr(scene_tokens_term, "func", None), "__name__", "")
        term_params = getattr(scene_tokens_term, "params", None) or {}
        try:
            decay = float(term_params.get("decay", 1.0))
        except (TypeError, ValueError):
            decay = 1.0
        score_mode = decay < 1.0
        use_no_goal_tokens = token_func_name == "scene_tokens_risk_no_goal_from_visibility_map"
        try:
            visibility_obs = nav_observations.local_visibility_map(
                env.unwrapped,
                sensor_cfg=depth_sensor_cfg,
                asset_cfg=SceneEntityCfg("robot"),
                size_m=6.4,
                grid_size=64,
                depth_min_m=0.5,
                depth_max_m=8.0,
                pixel_stride=2,
                decay=decay,
            )
            visibility_np = visibility_obs.detach().cpu().view(num_envs, 64, 64).numpy()
            for env_id in range(num_envs):
                if score_mode:
                    cached_frames["visibility_obs"][env_id] = _visibility_score_map_to_rgb(visibility_np[env_id])
                else:
                    cached_frames["visibility_obs"][env_id] = _visibility_map_to_rgb(visibility_np[env_id])
        except Exception as exc:
            print(f"[WARN] Failed to capture actor visibility-map observation: {exc}")
        try:
            if use_no_goal_tokens:
                scene_tokens = nav_observations.scene_tokens_risk_no_goal_from_visibility_map(
                    env.unwrapped,
                    sensor_cfg=depth_sensor_cfg,
                    asset_cfg=SceneEntityCfg("robot"),
                    size_m=6.4,
                    grid_size=64,
                    token_grid_size=8,
                    depth_min_m=0.5,
                    depth_max_m=8.0,
                    pixel_stride=2,
                    decay=decay,
                )
            else:
                scene_tokens = nav_observations.scene_tokens_from_visibility_map(
                    env.unwrapped,
                    sensor_cfg=depth_sensor_cfg,
                    asset_cfg=SceneEntityCfg("robot"),
                    size_m=6.4,
                    grid_size=64,
                    token_grid_size=8,
                    depth_min_m=0.5,
                    depth_max_m=8.0,
                    pixel_stride=2,
                    decay=decay,
                )
            scene_tokens_np = scene_tokens.detach().cpu().view(num_envs, 64, 6).numpy()
            for env_id in range(num_envs):
                cached_frames["scene_tokens"][env_id] = _scene_tokens_to_rgb(
                    scene_tokens_np[env_id], layout="no_goal" if use_no_goal_tokens else "v1"
                )
        except Exception as exc:
            print(f"[WARN] Failed to capture actor scene-token observation: {exc}")

    return cached_frames


def _force_sensor_update(sensor, dt: float):
    update_fn = getattr(sensor, "update", None)
    if not callable(update_fn):
        return
    try:
        update_fn(dt, force_recompute=True)
    except TypeError:
        update_fn(dt)


def capture_failure_observation_gifs(
    env,
    done_ids: torch.Tensor,
    done_term_values: dict[str, list[float]],
    pre_step_snapshot: dict[str, torch.Tensor] | None,
    pre_step_delay_snapshot: dict[str, torch.Tensor] | None,
    pre_step_snapshot_history: list[dict[str, torch.Tensor]],
    pre_step_observation_history: list[dict[str, list[np.ndarray | None]]],
    render_context: dict[str, object] | None,
    terrain_lookup: dict[int, str],
    base_contact_records: list[dict[str, object]],
    episode_start_index: int,
    output_dir: str | None,
    remaining_budget: int,
) -> tuple[int, list[str]]:
    if (
        output_dir is None
        or remaining_budget <= 0
        or pre_step_snapshot is None
        or done_ids.numel() == 0
        or "base_contact" not in done_term_values
    ):
        return 0, []

    base_contact_mask = torch.tensor(done_term_values["base_contact"], device=done_ids.device, dtype=torch.bool)
    local_indices = torch.nonzero(base_contact_mask, as_tuple=False).squeeze(-1)
    if local_indices.numel() == 0:
        return 0, []
    local_indices = local_indices[:remaining_budget]

    robot = env.unwrapped.scene["robot"]
    sensors = getattr(env.unwrapped.scene, "sensors", {})
    depth_sensor = sensors.get("raycast_camera") if hasattr(sensors, "get") else None
    height_sensor = sensors.get("height_scanner_critic") if hasattr(sensors, "get") else None
    if depth_sensor is None and height_sensor is None:
        return 0, []

    collision_lookup = {int(record["finished_episode_index"]): record for record in base_contact_records}

    history_frames = max(1, int(args_cli.observation_gif_frames))

    saved_paths: list[str] = []
    exported_failures = 0
    for local_index in local_indices.detach().cpu().tolist():
        env_id = int(done_ids[local_index].item())
        env_id_tensor = torch.tensor([env_id], device=env.unwrapped.device, dtype=torch.long)
        selected_snapshots = list(pre_step_snapshot_history[-history_frames:]) if pre_step_snapshot_history else [pre_step_snapshot]
        selected_snapshots, selected_observations = select_episode_history_for_env(
            snapshots=selected_snapshots,
            observations=list(pre_step_observation_history[-len(selected_snapshots):]) if selected_snapshots else [],
            env_id=env_id,
            reference_snapshot=pre_step_snapshot,
        )
        finished_episode_index = _finished_case_index(done_ids, local_index, episode_start_index)

        current_root_pose = torch.cat(
            (robot.data.root_pos_w[env_id_tensor], robot.data.root_quat_w[env_id_tensor]), dim=-1
        ).detach().clone()
        current_root_velocity = torch.cat(
            (robot.data.root_lin_vel_w[env_id_tensor], robot.data.root_ang_vel_w[env_id_tensor]), dim=-1
        ).detach().clone()
        current_joint_pos = robot.data.joint_pos[env_id_tensor].detach().clone()
        current_joint_vel = robot.data.joint_vel[env_id_tensor].detach().clone()

        terrain_type_index = int(pre_step_snapshot["terrain_types"][env_id].item()) if "terrain_types" in pre_step_snapshot else -1
        terrain_level = int(pre_step_snapshot["terrain_levels"][env_id].item()) if "terrain_levels" in pre_step_snapshot else -1
        terrain_name = _terrain_name_from_metadata(terrain_lookup, terrain_type_index)
        output_stem = _build_failure_output_stem(
            output_dir,
            finished_episode_index,
            "base_contact",
            terrain_name,
            terrain_level,
            terrain_type_index,
        )
        collision_record = collision_lookup.get(finished_episode_index)
        collision_pos = None
        collision_body_name = None
        if collision_record is not None:
            collision_body_name = collision_record.get("colliding_body_name")
            if collision_record.get("approx_collision_position_w") is not None:
                collision_pos = np.asarray(collision_record["approx_collision_position_w"], dtype=np.float32)

        goal_pos_w = None
        if "goal_position_world" in pre_step_snapshot:
            goal_pos_w = pre_step_snapshot["goal_position_world"][env_id].detach().cpu().numpy()
        spawn_pos_w = None
        if "spawn_position_world" in pre_step_snapshot:
            spawn_pos_w = pre_step_snapshot["spawn_position_world"][env_id].detach().cpu().numpy()

        root_pos_w = pre_step_snapshot["root_pos_w"][env_id].detach().cpu().numpy()
        root_quat_w = pre_step_snapshot["root_quat_w"][env_id].detach().cpu().numpy()
        root_lin_speed_b = float(torch.norm(pre_step_snapshot["root_lin_vel_b"][env_id], dim=-1).item())
        termination_labels = [name for name, values in done_term_values.items() if bool(values[local_index])]

        topdown_frame = None
        if render_context is not None and terrain_type_index >= 0 and terrain_level >= 0:
            topdown_frame = _render_failure_topdown_rgb(
                render_context=render_context,
                terrain_name=terrain_name,
                terrain_level=terrain_level,
                terrain_type_index=terrain_type_index,
                root_pos_w=root_pos_w,
                root_quat_w=root_quat_w,
                goal_pos_w=goal_pos_w,
                spawn_pos_w=spawn_pos_w,
                termination_labels=termination_labels,
                finished_episode_index=finished_episode_index,
                root_lin_speed_b=root_lin_speed_b,
                collision_pos_w=collision_pos,
                collision_body_name=collision_body_name,
                crop_size_m=args_cli.failure_topdown_crop_size_m,
                dpi=args_cli.failure_topdown_dpi,
            )

        rendered_combined_frames: list[np.ndarray] = []
        for history_index, history_snapshot in enumerate(selected_snapshots):
            hist_root_pose = torch.cat(
                (
                    history_snapshot["root_pos_w"][env_id_tensor],
                    history_snapshot["root_quat_w"][env_id_tensor],
                ),
                dim=-1,
            )
            hist_root_velocity = torch.cat(
                (
                    history_snapshot["root_lin_vel_w"][env_id_tensor],
                    history_snapshot["root_ang_vel_w"][env_id_tensor],
                ),
                dim=-1,
            )
            hist_joint_pos = history_snapshot["joint_pos"][env_id_tensor]
            hist_joint_vel = history_snapshot["joint_vel"][env_id_tensor]

            robot.write_root_pose_to_sim(hist_root_pose, env_ids=env_id_tensor)
            robot.write_root_velocity_to_sim(hist_root_velocity, env_ids=env_id_tensor)
            robot.write_joint_state_to_sim(hist_joint_pos, hist_joint_vel, env_ids=env_id_tensor)
            robot.set_joint_position_target(hist_joint_pos, env_ids=env_id_tensor)
            robot.set_joint_velocity_target(hist_joint_vel, env_ids=env_id_tensor)
            robot.write_data_to_sim()

            env.unwrapped.sim.render()
            env.unwrapped.sim.render()
            depth_frame = None
            if depth_sensor is not None:
                _force_sensor_update(depth_sensor, env.unwrapped.physics_dt)
                depth_frame = _capture_depth_frame_rgb(env, env_id)
            height_frame = None
            if height_sensor is not None:
                _force_sensor_update(height_sensor, env.unwrapped.physics_dt)
                height_frame = _capture_height_frame_rgb(env, env_id)
            steep_frame = None
            steep_frames = _capture_steep_frame_rgbs(env, height_sensor) if height_sensor is not None else []
            if env_id < len(steep_frames):
                steep_frame = steep_frames[env_id]
            local_map_frame = None
            local_gradient_frame = None
            visibility_obs_frame = None
            scene_token_frame = None
            observation_cache = selected_observations[history_index] if history_index < len(selected_observations) else None
            if observation_cache is not None:
                local_map_frames = observation_cache.get("local_map", [])
                local_gradient_frames = observation_cache.get("local_gradient", [])
                visibility_obs_frames = observation_cache.get("visibility_obs", [])
                scene_token_frames = observation_cache.get("scene_tokens", [])
                if env_id < len(local_map_frames):
                    local_map_frame = local_map_frames[env_id]
                if env_id < len(local_gradient_frames):
                    local_gradient_frame = local_gradient_frames[env_id]
                if env_id < len(visibility_obs_frames):
                    visibility_obs_frame = visibility_obs_frames[env_id]
                if env_id < len(scene_token_frames):
                    scene_token_frame = scene_token_frames[env_id]

            if (
                topdown_frame is not None
                or depth_frame is not None
                or height_frame is not None
                or steep_frame is not None
                or local_map_frame is not None
                or local_gradient_frame is not None
                or visibility_obs_frame is not None
                or scene_token_frame is not None
            ):
                rendered_combined_frames.append(
                    _compose_observation_triptych(
                        topdown_frame,
                        None,
                        depth_frame,
                        height_frame,
                        steep_frame,
                        local_map_frame=local_map_frame,
                        local_gradient_frame=local_gradient_frame,
                        visibility_obs_frame=visibility_obs_frame,
                        scene_token_frame=scene_token_frame,
                    )
                )

        if rendered_combined_frames:
            combined_output_path = (
                f"{_append_observation_delay_suffix(output_stem, env, env_id, pre_step_delay_snapshot)}"
                "__topdown_depth_height.gif"
            )
            _save_rgb_frames_to_gif(rendered_combined_frames, combined_output_path, args_cli.observation_gif_frame_ms)
            saved_paths.append(combined_output_path)

        if rendered_combined_frames:
            exported_failures += 1

        robot.write_root_pose_to_sim(current_root_pose, env_ids=env_id_tensor)
        robot.write_root_velocity_to_sim(current_root_velocity, env_ids=env_id_tensor)
        robot.write_joint_state_to_sim(current_joint_pos, current_joint_vel, env_ids=env_id_tensor)
        robot.set_joint_position_target(current_joint_pos, env_ids=env_id_tensor)
        robot.set_joint_velocity_target(current_joint_vel, env_ids=env_id_tensor)
        robot.write_data_to_sim()

    return exported_failures, saved_paths


def _build_failure_output_stem(
    output_dir: str,
    finished_episode_index: int,
    safe_terms: str,
    terrain_name: str,
    terrain_level: int,
    terrain_type_index: int,
) -> str:
    id_prefix = "case" if args_cli.initial_env_cases_only else "ep"
    return os.path.join(
        output_dir,
        f"{id_prefix}_{finished_episode_index:04d}__{safe_terms}__{terrain_name}__level_{terrain_level}__type_{terrain_type_index}",
    )


def _finished_case_index(done_ids: torch.Tensor, local_index: int, episode_start_index: int) -> int:
    if args_cli.initial_env_cases_only:
        return int(done_ids[local_index].item()) + 1
    return int(episode_start_index + local_index + 1)


def _terrain_metadata_for_env_ids(
    pre_step_snapshot: dict[str, torch.Tensor], env_ids: torch.Tensor
) -> tuple[np.ndarray, np.ndarray]:
    count = int(env_ids.numel())
    if "terrain_types" in pre_step_snapshot:
        terrain_type_indices = pre_step_snapshot["terrain_types"][env_ids].detach().cpu().numpy()
    else:
        terrain_type_indices = np.full(count, -1, dtype=np.int64)
    if "terrain_levels" in pre_step_snapshot:
        terrain_levels = pre_step_snapshot["terrain_levels"][env_ids].detach().cpu().numpy()
    else:
        terrain_levels = np.full(count, -1, dtype=np.int64)
    return terrain_type_indices, terrain_levels


def _terrain_name_from_metadata(terrain_lookup: dict[int, str], terrain_type_index: int) -> str:
    if terrain_type_index < 0:
        return "usd_scene"
    return terrain_lookup.get(terrain_type_index, f"terrain_type_{terrain_type_index}")


def _resolved_failure_rgb_success_limit() -> int:
    if args_cli.failure_rgb_success_limit is not None:
        return max(0, int(args_cli.failure_rgb_success_limit))
    if not args_cli.failure_rgb_include_successes:
        return 0
    return max(0, int(args_cli.failure_rgb_limit))


def _resolved_failure_rgb_failure_limit() -> int:
    if args_cli.failure_rgb_failure_limit is not None:
        return max(0, int(args_cli.failure_rgb_failure_limit))
    return max(0, int(args_cli.failure_rgb_limit))


def capture_failure_rgb_screenshots(
    env,
    failure_camera: Camera | None,
    done_ids: torch.Tensor,
    successes: torch.Tensor,
    done_term_values: dict[str, list[float]],
    pre_step_snapshot: dict[str, torch.Tensor] | None,
    pre_step_delay_snapshot: dict[str, torch.Tensor] | None,
    pre_step_snapshot_history: list[dict[str, torch.Tensor]],
    pre_step_observation_history: list[dict[str, list[np.ndarray | None]]],
    terrain_lookup: dict[int, str],
    base_contact_records: list[dict[str, object]],
    episode_start_index: int,
    output_dir: str | None,
    success_remaining_budget: int,
    failure_remaining_budget: int,
) -> tuple[int, int, int, int, list[str]]:
    if (
        failure_camera is None
        or output_dir is None
        or (success_remaining_budget <= 0 and failure_remaining_budget <= 0)
        or pre_step_snapshot is None
        or done_ids.numel() == 0
    ):
        return 0, 0, 0, 0, []

    terrain_filter_mask = None
    requested_terrain_name = args_cli.failure_rgb_terrain_name
    if requested_terrain_name:
        terrain_type_indices_all, _ = _terrain_metadata_for_env_ids(pre_step_snapshot, done_ids)
        terrain_matches = [
            _terrain_name_from_metadata(terrain_lookup, int(terrain_type_index)) == requested_terrain_name
            for terrain_type_index in terrain_type_indices_all
        ]
        terrain_filter_mask = torch.tensor(terrain_matches, device=done_ids.device, dtype=torch.bool)

    skipped_success_distance_count = 0
    skipped_success_height_delta_count = 0
    capture_indices: list[torch.Tensor] = []
    if args_cli.failure_rgb_include_successes and success_remaining_budget > 0:
        success_capture_mask = successes.clone()
        if terrain_filter_mask is not None:
            success_capture_mask = torch.logical_and(success_capture_mask, terrain_filter_mask)
        min_success_distance = float(args_cli.failure_rgb_success_min_initial_distance)
        if min_success_distance > 0.0:
            initial_distance = None
            if "initial_distance_to_goal" in pre_step_snapshot:
                initial_distance = pre_step_snapshot["initial_distance_to_goal"][done_ids]
            elif "spawn_position_world" in pre_step_snapshot and "goal_position_world" in pre_step_snapshot:
                spawn_xy = pre_step_snapshot["spawn_position_world"][done_ids, :2]
                goal_xy = pre_step_snapshot["goal_position_world"][done_ids, :2]
                initial_distance = torch.norm(spawn_xy - goal_xy, dim=1)
            if initial_distance is not None:
                distance_mask = initial_distance >= min_success_distance
                skipped_success_distance_count = int(torch.logical_and(success_capture_mask, torch.logical_not(distance_mask)).sum().item())
                success_capture_mask = torch.logical_and(success_capture_mask, distance_mask)
        min_goal_height_delta = float(args_cli.failure_rgb_success_min_goal_height_delta)
        if min_goal_height_delta > 0.0 and "spawn_position_world" in pre_step_snapshot and "goal_position_world" in pre_step_snapshot:
            spawn_positions = pre_step_snapshot["spawn_position_world"]
            goal_positions = pre_step_snapshot["goal_position_world"]
            spawn_z = _sample_height_field_z_for_positions(env, pre_step_snapshot, done_ids, spawn_positions)
            goal_z = _sample_height_field_z_for_positions(env, pre_step_snapshot, done_ids, goal_positions)
            if spawn_z is None or goal_z is None:
                spawn_z = spawn_positions[done_ids, 2]
                goal_z = goal_positions[done_ids, 2]
            height_delta_mask = (goal_z - spawn_z) >= min_goal_height_delta
            skipped_success_height_delta_count = int(
                torch.logical_and(success_capture_mask, torch.logical_not(height_delta_mask)).sum().item()
            )
            success_capture_mask = torch.logical_and(success_capture_mask, height_delta_mask)
        success_local_indices = torch.nonzero(success_capture_mask, as_tuple=False).squeeze(-1)
        if success_local_indices.numel() > 0:
            capture_indices.append(success_local_indices[:success_remaining_budget])

    failure_capture_mask = torch.logical_not(successes)
    if terrain_filter_mask is not None:
        failure_capture_mask = torch.logical_and(failure_capture_mask, terrain_filter_mask)
    if args_cli.failure_rgb_base_contact_only:
        if "base_contact" not in done_term_values:
            return 0, 0, skipped_success_distance_count, skipped_success_height_delta_count, []
        base_contact_mask = torch.tensor(done_term_values["base_contact"], device=done_ids.device, dtype=torch.bool)
        failure_capture_mask = torch.logical_and(failure_capture_mask, base_contact_mask)
    if failure_remaining_budget > 0:
        failure_local_indices = torch.nonzero(failure_capture_mask, as_tuple=False).squeeze(-1)
        if failure_local_indices.numel() > 0:
            capture_indices.append(failure_local_indices[:failure_remaining_budget])

    if not capture_indices:
        return 0, 0, skipped_success_distance_count, skipped_success_height_delta_count, []
    capture_local_indices = torch.cat(capture_indices).sort().values
    capture_env_ids = done_ids[capture_local_indices]
    collision_lookup = {int(record["finished_episode_index"]): record for record in base_contact_records}
    robot = env.unwrapped.scene["robot"]

    root_pos_w = pre_step_snapshot["root_pos_w"][capture_env_ids]
    root_quat_w = pre_step_snapshot["root_quat_w"][capture_env_ids]
    terrain_type_indices, terrain_levels = _terrain_metadata_for_env_ids(pre_step_snapshot, capture_env_ids)

    saved_paths: list[str] = []
    rendered_success_cases = 0
    rendered_failure_cases = 0
    for batch_index, local_index in enumerate(capture_local_indices.detach().cpu().tolist()):
        finished_episode_index = _finished_case_index(done_ids, local_index, episode_start_index)
        env_id = int(capture_env_ids[batch_index].item())
        env_id_tensor = torch.tensor([env_id], device=env.unwrapped.device, dtype=torch.long)

        current_root_pose = torch.cat(
            (robot.data.root_pos_w[env_id_tensor], robot.data.root_quat_w[env_id_tensor]), dim=-1
        ).detach().clone()
        current_root_velocity = torch.cat(
            (robot.data.root_lin_vel_w[env_id_tensor], robot.data.root_ang_vel_w[env_id_tensor]), dim=-1
        ).detach().clone()
        current_joint_pos = robot.data.joint_pos[env_id_tensor].detach().clone()
        current_joint_vel = robot.data.joint_vel[env_id_tensor].detach().clone()

        root_pos = root_pos_w[batch_index].detach().cpu().numpy()
        collision_record = collision_lookup.get(finished_episode_index)
        collision_pos = None
        if collision_record is not None and collision_record.get("approx_collision_position_w") is not None:
            collision_pos = np.asarray(collision_record["approx_collision_position_w"], dtype=np.float32)

        center_pos = np.array(root_pos, dtype=np.float32)
        if args_cli.failure_rgb_center_mode == "collision" and collision_pos is not None:
            center_pos = collision_pos
        elif args_cli.failure_rgb_center_mode == "midpoint" and collision_pos is not None:
            center_pos = 0.5 * (center_pos + collision_pos)
        elif args_cli.failure_rgb_center_mode == "spawn_goal":
            spawn_pos = pre_step_snapshot.get("spawn_position_world")
            goal_pos = pre_step_snapshot.get("goal_position_world")
            if spawn_pos is not None and goal_pos is not None:
                spawn_np = spawn_pos[env_id].detach().cpu().numpy()
                goal_np = goal_pos[env_id].detach().cpu().numpy()
                center_pos = 0.5 * (np.asarray(spawn_np, dtype=np.float32) + np.asarray(goal_np, dtype=np.float32))

        eye = np.array(center_pos, dtype=np.float32) + np.array([0.0, 0.0, args_cli.failure_rgb_height_m], dtype=np.float32)
        if args_cli.failure_rgb_distance_m != 0.0 or args_cli.failure_rgb_lookahead_m != 0.0:
            root_quat = root_quat_w[batch_index].detach().cpu().numpy()
            yaw = _quat_wxyz_to_yaw(root_quat)
            forward = np.array([math.cos(yaw), math.sin(yaw), 0.0], dtype=np.float32)
            eye[:2] -= args_cli.failure_rgb_distance_m * forward[:2]
            target = np.array(center_pos, dtype=np.float32) + args_cli.failure_rgb_lookahead_m * forward
        else:
            target = np.array(center_pos, dtype=np.float32)
        target[2] = center_pos[2]

        eye_t = torch.tensor(eye[None, :], device=env.unwrapped.device, dtype=torch.float32)
        target_t = torch.tensor(target[None, :], device=env.unwrapped.device, dtype=torch.float32)
        view_rotations = math_utils.create_rotation_matrix_from_view(
            eye_t,
            target_t,
            up_axis="Z",
            device=env.unwrapped.device,
        )
        view_quaternions = math_utils.quat_from_matrix(view_rotations)
        failure_camera.set_world_poses(eye_t, view_quaternions, convention="opengl")

        env.unwrapped.sim.render()
        env.unwrapped.sim.render()
        failure_camera.update(env.unwrapped.physics_dt, force_recompute=True)

        terrain_type_index = int(terrain_type_indices[batch_index])
        terrain_level = int(terrain_levels[batch_index])
        terrain_name = _terrain_name_from_metadata(terrain_lookup, terrain_type_index)
        termination_labels = [name for name, values in done_term_values.items() if bool(values[local_index])]
        if bool(successes[local_index].item()):
            safe_terms = "success"
        else:
            safe_terms = "-".join(termination_labels) if termination_labels else "failure"
        output_stem = _build_failure_output_stem(
            output_dir,
            finished_episode_index,
            safe_terms,
            terrain_name,
            terrain_level,
            terrain_type_index,
        )
        spawn_position_world = _snapshot_vector(pre_step_snapshot, "spawn_position_world", env_id)
        goal_position_world = _snapshot_vector(pre_step_snapshot, "goal_position_world", env_id)
        goal_height_delta_m = None
        spawn_positions_tensor = pre_step_snapshot.get("spawn_position_world")
        goal_positions_tensor = pre_step_snapshot.get("goal_position_world")
        if spawn_positions_tensor is not None and goal_positions_tensor is not None:
            sampled_spawn_z = _sample_height_field_z_for_positions(env, pre_step_snapshot, env_id_tensor, spawn_positions_tensor)
            sampled_goal_z = _sample_height_field_z_for_positions(env, pre_step_snapshot, env_id_tensor, goal_positions_tensor)
            if sampled_spawn_z is not None and sampled_goal_z is not None:
                goal_height_delta_m = float((sampled_goal_z[0] - sampled_spawn_z[0]).detach().cpu().item())
            elif spawn_position_world is not None and goal_position_world is not None:
                goal_height_delta_m = float(goal_position_world[2] - spawn_position_world[2])

        case_metadata = {
            "seed": None if args_cli.seed is None else int(args_cli.seed),
            "case_index": int(finished_episode_index),
            "env_id": int(env_id),
            "outcome": safe_terms,
            "success": bool(successes[local_index].item()),
            "termination_labels": termination_labels,
            "terrain_name": terrain_name,
            "terrain_level": int(terrain_level),
            "terrain_type_index": int(terrain_type_index),
            "observation_delay": _delay_snapshot_for_env(pre_step_delay_snapshot, env_id),
            "spawn_position_world": spawn_position_world,
            "goal_position_world": goal_position_world,
            "initial_distance_to_goal_m": _snapshot_scalar(pre_step_snapshot, "initial_distance_to_goal", env_id),
            "goal_height_delta_m": goal_height_delta_m,
            "final_root_position_world": _snapshot_vector(pre_step_snapshot, "root_pos_w", env_id),
            "final_root_quat_world": _snapshot_vector(pre_step_snapshot, "root_quat_w", env_id),
            "terrain_geometry": terrain_geometry_signature(env),
            "checkpoint": os.path.abspath(os.path.expanduser(args_cli.checkpoint)) if args_cli.checkpoint else None,
        }

        if args_cli.failure_rgb_full_episode:
            selected_snapshots = list(pre_step_snapshot_history) if pre_step_snapshot_history else [pre_step_snapshot]
        else:
            history_frames = max(1, int(args_cli.failure_rgb_history_frames))
            selected_snapshots = list(pre_step_snapshot_history[-history_frames:]) if pre_step_snapshot_history else [pre_step_snapshot]
        observation_candidates = []
        if args_cli.failure_rgb_include_observations:
            observation_candidates = list(pre_step_observation_history[-len(selected_snapshots):]) if selected_snapshots else []
        selected_snapshots, selected_observations = select_episode_history_for_env(
            snapshots=selected_snapshots,
            observations=observation_candidates,
            env_id=env_id,
            reference_snapshot=pre_step_snapshot,
        )

        rendered_rgb_frames: list[np.ndarray] = []
        for history_index, history_snapshot in enumerate(selected_snapshots):
            hist_root_pose = torch.cat(
                (
                    history_snapshot["root_pos_w"][env_id_tensor],
                    history_snapshot["root_quat_w"][env_id_tensor],
                ),
                dim=-1,
            )
            hist_root_velocity = torch.cat(
                (
                    history_snapshot["root_lin_vel_w"][env_id_tensor],
                    history_snapshot["root_ang_vel_w"][env_id_tensor],
                ),
                dim=-1,
            )
            hist_joint_pos = history_snapshot["joint_pos"][env_id_tensor]
            hist_joint_vel = history_snapshot["joint_vel"][env_id_tensor]

            robot.write_root_pose_to_sim(hist_root_pose, env_ids=env_id_tensor)
            robot.write_root_velocity_to_sim(hist_root_velocity, env_ids=env_id_tensor)
            robot.write_joint_state_to_sim(hist_joint_pos, hist_joint_vel, env_ids=env_id_tensor)
            robot.set_joint_position_target(hist_joint_pos, env_ids=env_id_tensor)
            robot.set_joint_velocity_target(hist_joint_vel, env_ids=env_id_tensor)
            robot.write_data_to_sim()
            visualize_goal_marker_for_snapshot(history_snapshot, env_id, env.unwrapped.device)

            if args_cli.failure_rgb_follow_robot:
                hist_center_pos = history_snapshot["root_pos_w"][env_id].detach().cpu().numpy().astype(np.float32)
                hist_eye = hist_center_pos + np.array([0.0, 0.0, args_cli.failure_rgb_height_m], dtype=np.float32)
                if args_cli.failure_rgb_distance_m != 0.0 or args_cli.failure_rgb_lookahead_m != 0.0:
                    hist_root_quat = history_snapshot["root_quat_w"][env_id].detach().cpu().numpy()
                    hist_yaw = _quat_wxyz_to_yaw(hist_root_quat)
                    hist_forward = np.array([math.cos(hist_yaw), math.sin(hist_yaw), 0.0], dtype=np.float32)
                    hist_eye[:2] -= args_cli.failure_rgb_distance_m * hist_forward[:2]
                    hist_target = hist_center_pos + args_cli.failure_rgb_lookahead_m * hist_forward
                else:
                    hist_target = hist_center_pos.copy()
                hist_target[2] = hist_center_pos[2]

                hist_eye_t = torch.tensor(hist_eye[None, :], device=env.unwrapped.device, dtype=torch.float32)
                hist_target_t = torch.tensor(hist_target[None, :], device=env.unwrapped.device, dtype=torch.float32)
                hist_view_rotations = math_utils.create_rotation_matrix_from_view(
                    hist_eye_t,
                    hist_target_t,
                    up_axis="Z",
                    device=env.unwrapped.device,
                )
                hist_view_quaternions = math_utils.quat_from_matrix(hist_view_rotations)
                failure_camera.set_world_poses(hist_eye_t, hist_view_quaternions, convention="opengl")

            env.unwrapped.sim.render()
            env.unwrapped.sim.render()
            failure_camera.update(env.unwrapped.physics_dt, force_recompute=True)
            rgb_frame = _rgb_tensor_to_numpy(failure_camera.data.output["rgb"][0])
            rear_view_frame = None
            observation_cache = selected_observations[history_index] if history_index < len(selected_observations) else None
            if args_cli.failure_rgb_include_rear_view:
                hist_center_pos = history_snapshot["root_pos_w"][env_id].detach().cpu().numpy().astype(np.float32)
                hist_root_quat = history_snapshot["root_quat_w"][env_id].detach().cpu().numpy()
                hist_yaw = _quat_wxyz_to_yaw(hist_root_quat)
                rear_eye, rear_target = rear_gravity_aligned_camera_pose(
                    hist_center_pos,
                    hist_yaw,
                    height_m=args_cli.failure_rgb_rear_height_m,
                    distance_m=args_cli.failure_rgb_rear_distance_m,
                    lateral_m=args_cli.failure_rgb_rear_lateral_m,
                    target_height_m=args_cli.failure_rgb_rear_lookahead_m,
                )
                rear_eye_t = rear_eye.to(device=env.unwrapped.device, dtype=torch.float32).unsqueeze(0)
                rear_target_t = rear_target.to(device=env.unwrapped.device, dtype=torch.float32).unsqueeze(0)
                rear_view_rotations = math_utils.create_rotation_matrix_from_view(
                    rear_eye_t,
                    rear_target_t,
                    up_axis="Z",
                    device=env.unwrapped.device,
                )
                rear_view_quaternions = math_utils.quat_from_matrix(rear_view_rotations)
                failure_camera.set_world_poses(rear_eye_t, rear_view_quaternions, convention="opengl")
                env.unwrapped.sim.render()
                env.unwrapped.sim.render()
                failure_camera.update(env.unwrapped.physics_dt, force_recompute=True)
                rear_view_frame = _rgb_tensor_to_numpy(failure_camera.data.output["rgb"][0])
            if args_cli.failure_rgb_include_observations:
                depth_frame = None
                height_frame = None
                steep_frame = None
                local_map_frame = None
                local_gradient_frame = None
                visibility_obs_frame = None
                scene_token_frame = None
                if observation_cache is not None:
                    depth_frames = observation_cache.get("depth", [])
                    processed_depth_frames = observation_cache.get("processed_depth", [])
                    height_frames = observation_cache.get("height", [])
                    steep_frames = observation_cache.get("steep", [])
                    local_map_frames = observation_cache.get("local_map", [])
                    local_gradient_frames = observation_cache.get("local_gradient", [])
                    visibility_obs_frames = observation_cache.get("visibility_obs", [])
                    scene_token_frames = observation_cache.get("scene_tokens", [])
                    if env_id < len(depth_frames):
                        depth_frame = depth_frames[env_id]
                    processed_depth_frame = None
                    if env_id < len(processed_depth_frames):
                        processed_depth_frame = processed_depth_frames[env_id]
                    if env_id < len(height_frames):
                        height_frame = height_frames[env_id]
                    if env_id < len(steep_frames):
                        steep_frame = steep_frames[env_id]
                    if env_id < len(local_map_frames):
                        local_map_frame = local_map_frames[env_id]
                    if env_id < len(local_gradient_frames):
                        local_gradient_frame = local_gradient_frames[env_id]
                    if env_id < len(visibility_obs_frames):
                        visibility_obs_frame = visibility_obs_frames[env_id]
                    if env_id < len(scene_token_frames):
                        scene_token_frame = scene_token_frames[env_id]
                else:
                    processed_depth_frame = None
                rgb_frame = _compose_observation_triptych(
                    rgb_frame,
                    rear_view_frame,
                    depth_frame,
                    height_frame,
                    steep_frame,
                    processed_depth_frame=processed_depth_frame,
                    local_map_frame=local_map_frame,
                    local_gradient_frame=local_gradient_frame,
                    visibility_obs_frame=visibility_obs_frame,
                    scene_token_frame=scene_token_frame,
                )
            rendered_rgb_frames.append(rgb_frame)

        if args_cli.failure_rgb_export_mode in ("png", "both") and rendered_rgb_frames:
            png_output_path = f"{output_stem}.png"
            _save_rgb_array_to_png(rendered_rgb_frames[-1], png_output_path)
            _write_case_metadata(png_output_path, case_metadata)
            saved_paths.append(png_output_path)

        if args_cli.failure_rgb_export_mode in ("gif", "both") and rendered_rgb_frames:
            gif_output_path = f"{_append_observation_delay_suffix(output_stem, env, env_id, pre_step_delay_snapshot)}.gif"
            _save_rgb_frames_to_gif(rendered_rgb_frames, gif_output_path, args_cli.failure_rgb_gif_frame_ms)
            _write_case_metadata(gif_output_path, case_metadata)
            saved_paths.append(gif_output_path)
        if rendered_rgb_frames:
            if bool(successes[local_index].item()):
                rendered_success_cases += 1
            else:
                rendered_failure_cases += 1

        robot.write_root_pose_to_sim(current_root_pose, env_ids=env_id_tensor)
        robot.write_root_velocity_to_sim(current_root_velocity, env_ids=env_id_tensor)
        robot.write_joint_state_to_sim(current_joint_pos, current_joint_vel, env_ids=env_id_tensor)
        robot.set_joint_position_target(current_joint_pos, env_ids=env_id_tensor)
        robot.set_joint_velocity_target(current_joint_vel, env_ids=env_id_tensor)
        robot.write_data_to_sim()

    return rendered_success_cases, rendered_failure_cases, skipped_success_distance_count, skipped_success_height_delta_count, saved_paths


def _terrain_instance_index(level: int, terrain_type_index: int, render_context: dict[str, object]) -> int:
    if render_context["curriculum"]:
        return level + terrain_type_index * render_context["num_rows"]
    return level * render_context["num_cols"] + terrain_type_index


def resolve_topdown_render_context(env) -> dict[str, object] | None:
    terrain = getattr(env.unwrapped.scene, "terrain", None)
    terrain_cfg = getattr(terrain.cfg, "terrain_generator", None) if terrain is not None else None
    if terrain is None or terrain_cfg is None:
        return None

    heights = _to_numpy(getattr(terrain, "_height_field_visual", None))
    if heights is None:
        return None

    valid_mask = _to_numpy(getattr(terrain, "_height_field_valid_mask", None))
    terrain_origins = _to_numpy(getattr(terrain, "terrain_origins", None))
    if terrain_origins is None:
        return None

    return {
        "heights": heights,
        "valid_mask": valid_mask,
        "terrain_origins": terrain_origins,
        "terrain_size": float(terrain_cfg.size[0]),
        "horizontal_scale": float(terrain_cfg.horizontal_scale),
        "num_rows": int(terrain_cfg.num_rows),
        "num_cols": int(terrain_cfg.num_cols),
        "curriculum": bool(terrain_cfg.curriculum),
    }


def _build_topdown_rgb(tile_heights: np.ndarray, tile_valid_mask: np.ndarray | None) -> np.ndarray:
    tile_heights = tile_heights.astype(np.float32)
    rgb = np.full((*tile_heights.shape, 3), 0.86, dtype=np.float32)

    positive_peak = max(float(tile_heights.max()), 1.0)
    negative_peak = max(float((-tile_heights).max()), 1.0)
    positive = np.clip(tile_heights / positive_peak, 0.0, 1.0)
    negative = np.clip((-tile_heights) / negative_peak, 0.0, 1.0)

    rgb -= positive[..., None] * np.array([0.55, 0.55, 0.50], dtype=np.float32)
    rgb += negative[..., None] * np.array([0.00, 0.06, 0.20], dtype=np.float32)

    if tile_valid_mask is not None:
        invalid_mask = np.logical_not(tile_valid_mask.astype(bool))
        positive_obstacles = np.logical_and(invalid_mask, tile_heights >= 0)
        negative_obstacles = np.logical_and(invalid_mask, tile_heights < 0)
        rgb[positive_obstacles] = np.array([0.12, 0.12, 0.16], dtype=np.float32)
        rgb[negative_obstacles] = np.array([0.06, 0.15, 0.28], dtype=np.float32)

    return np.clip(rgb, 0.0, 1.0)


def _render_failure_topdown_rgb(
    render_context: dict[str, object],
    terrain_name: str,
    terrain_level: int,
    terrain_type_index: int,
    root_pos_w: np.ndarray,
    root_quat_w: np.ndarray,
    goal_pos_w: np.ndarray | None,
    spawn_pos_w: np.ndarray | None,
    termination_labels: list[str],
    finished_episode_index: int,
    root_lin_speed_b: float,
    collision_pos_w: np.ndarray | None = None,
    collision_body_name: str | None = None,
    crop_size_m: float = 0.0,
    dpi: int = 150,
) -> np.ndarray:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    terrain_index = _terrain_instance_index(terrain_level, terrain_type_index, render_context)
    tile_heights = render_context["heights"][terrain_index]
    tile_valid_mask = None
    if render_context["valid_mask"] is not None:
        tile_valid_mask = render_context["valid_mask"][terrain_index]
    terrain_origin = render_context["terrain_origins"][terrain_level, terrain_type_index]
    terrain_half_size = float(render_context["terrain_size"]) / 2.0

    def world_to_local(point_w: np.ndarray | None) -> np.ndarray | None:
        if point_w is None:
            return None
        return np.asarray(point_w[:2], dtype=np.float32) - np.asarray(terrain_origin[:2], dtype=np.float32)

    robot_local = world_to_local(root_pos_w)
    goal_local = world_to_local(goal_pos_w)
    spawn_local = world_to_local(spawn_pos_w)
    collision_local = world_to_local(collision_pos_w)
    yaw = _quat_wxyz_to_yaw(root_quat_w)

    rgb = _build_topdown_rgb(tile_heights, tile_valid_mask)
    fig, ax = plt.subplots(figsize=(6, 6), dpi=dpi)
    ax.imshow(
        rgb,
        origin="lower",
        extent=[-terrain_half_size, terrain_half_size, -terrain_half_size, terrain_half_size],
        interpolation="nearest",
    )

    if spawn_local is not None:
        ax.scatter(spawn_local[0], spawn_local[1], s=35, c="#3498db", marker="s", label="spawn")
    if goal_local is not None:
        ax.scatter(goal_local[0], goal_local[1], s=65, c="#2ecc71", marker="*", label="goal")
    ax.scatter(robot_local[0], robot_local[1], s=45, c="#ff4d4f", marker="o", label="robot")
    ax.arrow(
        robot_local[0],
        robot_local[1],
        1.2 * math.cos(yaw),
        1.2 * math.sin(yaw),
        color="#ff4d4f",
        width=0.05,
        head_width=0.45,
        length_includes_head=True,
    )
    if collision_local is not None:
        collision_label = f"collision ({collision_body_name})" if collision_body_name else "collision"
        ax.scatter(collision_local[0], collision_local[1], s=40, c="#f1c40f", marker="x", label=collision_label)

    if crop_size_m > 0.0:
        half_crop = crop_size_m / 2.0
        ax.set_xlim(robot_local[0] - half_crop, robot_local[0] + half_crop)
        ax.set_ylim(robot_local[1] - half_crop, robot_local[1] + half_crop)
    else:
        ax.set_xlim(-terrain_half_size, terrain_half_size)
        ax.set_ylim(-terrain_half_size, terrain_half_size)

    ax.set_title(
        f"Ep {finished_episode_index} | {terrain_name} | level {terrain_level} | {'/'.join(termination_labels) if termination_labels else 'failure'}\n"
        f"speed_b={root_lin_speed_b:.2f} m/s"
    )
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[..., :3].copy()
    plt.close(fig)
    return image


def _resize_rgb_frame(frame: np.ndarray, target_height: int) -> np.ndarray:
    from PIL import Image

    if frame.shape[0] == target_height:
        return np.asarray(frame, dtype=np.uint8)
    target_width = max(1, int(round(frame.shape[1] * (target_height / float(frame.shape[0])))))
    resized = Image.fromarray(np.asarray(frame, dtype=np.uint8)).resize((target_width, target_height), Image.BILINEAR)
    return np.asarray(resized, dtype=np.uint8)


def _label_rgb_frame(frame: np.ndarray, label: str) -> np.ndarray:
    from PIL import Image, ImageDraw

    image = Image.fromarray(np.asarray(frame, dtype=np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(image)
    pad_x = 6
    pad_y = 4
    text_bbox = draw.textbbox((0, 0), label)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]
    draw.rectangle((0, 0, text_w + 2 * pad_x, text_h + 2 * pad_y), fill=(0, 0, 0))
    draw.text((pad_x, pad_y), label, fill=(255, 255, 255))
    return np.asarray(image, dtype=np.uint8)


def _compose_observation_triptych(
    topdown_frame: np.ndarray,
    rear_view_frame: np.ndarray | None,
    depth_frame: np.ndarray | None,
    height_frame: np.ndarray | None,
    steep_frame: np.ndarray | None = None,
    processed_depth_frame: np.ndarray | None = None,
    local_map_frame: np.ndarray | None = None,
    local_gradient_frame: np.ndarray | None = None,
    visibility_obs_frame: np.ndarray | None = None,
    scene_token_frame: np.ndarray | None = None,
) -> np.ndarray:
    labeled_frames = [
        ("rgb", topdown_frame),
        ("rear", rear_view_frame),
        ("depth", depth_frame),
        ("processed_depth", processed_depth_frame),
        ("height", height_frame),
        ("steep", steep_frame),
        ("local_map", local_map_frame),
        ("local_gradient", local_gradient_frame),
        ("visibility", visibility_obs_frame),
        ("scene_tokens", scene_token_frame),
    ]
    labeled_frames = [(label, frame) for label, frame in labeled_frames if frame is not None]
    if not labeled_frames:
        raise ValueError("At least one frame is required to compose observation visualization.")
    target_height = max(frame.shape[0] for _, frame in labeled_frames)
    panels = [_label_rgb_frame(_resize_rgb_frame(frame, target_height), label) for label, frame in labeled_frames]
    return np.concatenate(panels, axis=1)


def render_failure_topdown_snapshot(
    output_path: str,
    render_context: dict[str, object],
    terrain_name: str,
    terrain_level: int,
    terrain_type_index: int,
    root_pos_w: np.ndarray,
    root_quat_w: np.ndarray,
    goal_pos_w: np.ndarray | None,
    spawn_pos_w: np.ndarray | None,
    termination_labels: list[str],
    finished_episode_index: int,
    root_lin_speed_b: float,
    collision_pos_w: np.ndarray | None = None,
    collision_body_name: str | None = None,
    crop_size_m: float = 0.0,
    dpi: int = 150,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    terrain_index = _terrain_instance_index(terrain_level, terrain_type_index, render_context)
    tile_heights = render_context["heights"][terrain_index]
    tile_valid_mask = None
    if render_context["valid_mask"] is not None:
        tile_valid_mask = render_context["valid_mask"][terrain_index]
    terrain_origin = render_context["terrain_origins"][terrain_level, terrain_type_index]
    terrain_half_size = float(render_context["terrain_size"]) / 2.0

    def world_to_local(point_w: np.ndarray | None) -> np.ndarray | None:
        if point_w is None:
            return None
        return np.asarray(point_w[:2], dtype=np.float32) - np.asarray(terrain_origin[:2], dtype=np.float32)

    robot_local = world_to_local(root_pos_w)
    goal_local = world_to_local(goal_pos_w)
    spawn_local = world_to_local(spawn_pos_w)
    collision_local = world_to_local(collision_pos_w)
    yaw = _quat_wxyz_to_yaw(root_quat_w)

    rgb = _build_topdown_rgb(tile_heights, tile_valid_mask)
    fig, ax = plt.subplots(figsize=(6, 6), dpi=dpi)
    ax.imshow(
        rgb,
        origin="lower",
        extent=[-terrain_half_size, terrain_half_size, -terrain_half_size, terrain_half_size],
        interpolation="nearest",
    )

    if spawn_local is not None:
        ax.scatter(spawn_local[0], spawn_local[1], s=35, c="#3498db", marker="s", label="spawn")
    if goal_local is not None:
        ax.scatter(goal_local[0], goal_local[1], s=65, c="#2ecc71", marker="*", label="goal")
    ax.scatter(robot_local[0], robot_local[1], s=45, c="#ff4d4f", marker="o", label="robot")
    ax.arrow(
        robot_local[0],
        robot_local[1],
        1.2 * math.cos(yaw),
        1.2 * math.sin(yaw),
        color="#ff4d4f",
        width=0.05,
        head_width=0.45,
        length_includes_head=True,
    )
    if collision_local is not None:
        collision_label = f"collision ({collision_body_name})" if collision_body_name else "collision"
        ax.scatter(collision_local[0], collision_local[1], s=65, c="#f39c12", marker="x", label=collision_label)

    if crop_size_m > 0.0:
        half_crop = crop_size_m / 2.0
        ax.set_xlim(robot_local[0] - half_crop, robot_local[0] + half_crop)
        ax.set_ylim(robot_local[1] - half_crop, robot_local[1] + half_crop)
    else:
        ax.set_xlim(-terrain_half_size, terrain_half_size)
        ax.set_ylim(-terrain_half_size, terrain_half_size)

    termination_text = ",".join(termination_labels) if termination_labels else "failure"
    ax.set_title(
        f"ep={finished_episode_index} | {termination_text}\n"
        f"terrain={terrain_name} level={terrain_level} type_idx={terrain_type_index} | speed_b={root_lin_speed_b:.2f}"
    )
    ax.set_xlabel("local x [m]")
    ax.set_ylabel("local y [m]")
    ax.grid(color="white", alpha=0.08, linewidth=0.5)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def render_failure_topdown_images(
    done_ids: torch.Tensor,
    successes: torch.Tensor,
    done_term_values: dict[str, list[float]],
    pre_step_snapshot: dict[str, torch.Tensor] | None,
    render_context: dict[str, object] | None,
    terrain_lookup: dict[int, str],
    base_contact_records: list[dict[str, object]],
    episode_start_index: int,
    output_dir: str,
    crop_size_m: float,
    dpi: int,
    remaining_budget: int,
) -> tuple[int, list[str]]:
    if (
        remaining_budget <= 0
        or pre_step_snapshot is None
        or render_context is None
        or done_ids.numel() == 0
    ):
        return 0, []

    failure_mask = torch.logical_not(successes)
    failure_local_indices = torch.nonzero(failure_mask, as_tuple=False).squeeze(-1).detach().cpu().tolist()
    if not failure_local_indices:
        return 0, []
    failure_local_indices = failure_local_indices[:remaining_budget]

    collision_lookup = {int(record["finished_episode_index"]): record for record in base_contact_records}

    root_pos_w = pre_step_snapshot["root_pos_w"][done_ids].detach().cpu().numpy()
    root_quat_w = pre_step_snapshot["root_quat_w"][done_ids].detach().cpu().numpy()
    root_lin_vel_b = pre_step_snapshot["root_lin_vel_b"][done_ids].detach().cpu().numpy()
    terrain_type_indices, terrain_levels = _terrain_metadata_for_env_ids(pre_step_snapshot, done_ids)
    goal_positions = pre_step_snapshot.get("goal_position_world")
    spawn_positions = pre_step_snapshot.get("spawn_position_world")
    if goal_positions is not None:
        goal_positions = goal_positions[done_ids].detach().cpu().numpy()
    if spawn_positions is not None:
        spawn_positions = spawn_positions[done_ids].detach().cpu().numpy()

    rendered_paths: list[str] = []
    for local_index in failure_local_indices:
        finished_episode_index = _finished_case_index(done_ids, local_index, episode_start_index)
        terrain_type_index = int(terrain_type_indices[local_index])
        terrain_level = int(terrain_levels[local_index])
        terrain_name = _terrain_name_from_metadata(terrain_lookup, terrain_type_index)
        termination_labels = [name for name, values in done_term_values.items() if bool(values[local_index])]
        collision_record = collision_lookup.get(finished_episode_index)
        collision_pos_w = None if collision_record is None else np.asarray(collision_record["approx_collision_position_w"])
        collision_body_name = None if collision_record is None else str(collision_record["colliding_body_name"])
        root_lin_speed_b = float(np.linalg.norm(root_lin_vel_b[local_index]))
        safe_terms = "-".join(termination_labels) if termination_labels else "failure"
        id_prefix = "case" if args_cli.initial_env_cases_only else "ep"
        file_name = (
            f"{id_prefix}_{finished_episode_index:04d}__{safe_terms}__{terrain_name}__level_{terrain_level}"
            f"__type_{terrain_type_index}.png"
        )
        output_path = os.path.join(output_dir, file_name)
        render_failure_topdown_snapshot(
            output_path=output_path,
            render_context=render_context,
            terrain_name=terrain_name,
            terrain_level=terrain_level,
            terrain_type_index=terrain_type_index,
            root_pos_w=root_pos_w[local_index],
            root_quat_w=root_quat_w[local_index],
            goal_pos_w=None if goal_positions is None else goal_positions[local_index],
            spawn_pos_w=None if spawn_positions is None else spawn_positions[local_index],
            termination_labels=termination_labels,
            finished_episode_index=finished_episode_index,
            root_lin_speed_b=root_lin_speed_b,
            collision_pos_w=collision_pos_w,
            collision_body_name=collision_body_name,
            crop_size_m=crop_size_m,
            dpi=dpi,
        )
        rendered_paths.append(output_path)

    return len(rendered_paths), rendered_paths


def resolve_base_contact_context(env):
    termination_manager = env.unwrapped.termination_manager
    if "base_contact" not in termination_manager.active_terms:
        return None

    term_cfg = termination_manager.get_term_cfg("base_contact")
    params = getattr(term_cfg, "params", {}) or {}
    sensor_cfg = params.get("sensor_cfg")
    if sensor_cfg is None:
        return None

    body_ids = getattr(sensor_cfg, "body_ids", None)
    if body_ids is None:
        return None

    sensor = env.unwrapped.scene.sensors[sensor_cfg.name]
    robot = env.unwrapped.scene["robot"]

    resolved_body_ids = [int(body_id) for body_id in body_ids]
    sensor_body_names = list(sensor.body_names)
    robot_body_names = list(robot.data.body_names)
    resolved_body_names = [_safe_body_name(sensor_body_names, body_id) for body_id in resolved_body_ids]
    robot_body_ids = [robot_body_names.index(body_name) for body_name in resolved_body_names]

    return {
        "sensor": sensor,
        "sensor_name": sensor_cfg.name,
        "body_ids": resolved_body_ids,
        "body_names": resolved_body_names,
        "robot_body_ids": robot_body_ids,
        "threshold": float(params.get("threshold", 0.0)),
        "robot": robot,
    }


def capture_base_contact_snapshot(env, base_contact_context: dict | None) -> dict[str, torch.Tensor] | None:
    robot = base_contact_context["robot"] if base_contact_context is not None else env.unwrapped.scene["robot"]
    terrain = getattr(env.unwrapped.scene, "terrain", None)

    snapshot = {
        "root_pos_w": robot.data.root_pos_w.detach().clone(),
        "root_quat_w": robot.data.root_quat_w.detach().clone(),
        "root_lin_vel_w": robot.data.root_lin_vel_w.detach().clone(),
        "root_ang_vel_w": robot.data.root_ang_vel_w.detach().clone(),
        "root_lin_vel_b": robot.data.root_lin_vel_b.detach().clone(),
        "root_ang_vel_b": robot.data.root_ang_vel_b.detach().clone(),
        "joint_pos": robot.data.joint_pos.detach().clone(),
        "joint_vel": robot.data.joint_vel.detach().clone(),
        "body_state_w": robot.data.body_state_w.detach().clone(),
    }

    if terrain is not None and hasattr(terrain, "terrain_types"):
        snapshot["terrain_types"] = terrain.terrain_types.detach().clone()
    if terrain is not None and hasattr(terrain, "terrain_levels"):
        snapshot["terrain_levels"] = terrain.terrain_levels.detach().clone()

    command_terms = getattr(env.unwrapped.command_manager, "_terms", {})
    goal_term = command_terms.get("robot_goal") if isinstance(command_terms, dict) else None
    if goal_term is not None and hasattr(goal_term, "goal_position_world"):
        snapshot["goal_position_world"] = goal_term.goal_position_world.detach().clone()
    if goal_term is not None and hasattr(goal_term, "spawn_position_world"):
        snapshot["spawn_position_world"] = goal_term.spawn_position_world.detach().clone()
    if goal_term is not None and hasattr(goal_term, "initial_distance_to_goal"):
        snapshot["initial_distance_to_goal"] = goal_term.initial_distance_to_goal.detach().clone()

    return snapshot


def collect_base_contact_failure_records(
    env,
    done_ids: torch.Tensor,
    base_contact_mask: torch.Tensor,
    base_contact_context: dict,
    pre_step_snapshot: dict[str, torch.Tensor] | None,
    terrain_lookup: dict[int, str],
    episode_start_index: int,
) -> list[dict[str, object]]:
    if done_ids.numel() == 0 or not torch.any(base_contact_mask):
        return []

    sensor = base_contact_context["sensor"]
    robot = base_contact_context["robot"]
    monitored_body_ids = base_contact_context["body_ids"]
    monitored_robot_body_ids = base_contact_context["robot_body_ids"]
    net_force_history = sensor.data.net_forces_w_history[done_ids][:, :, monitored_body_ids, :]
    force_magnitudes = torch.norm(net_force_history, dim=-1)
    flattened_force_magnitudes = force_magnitudes.view(force_magnitudes.shape[0], -1)
    peak_force_values, peak_force_indices = torch.max(flattened_force_magnitudes, dim=1)

    state_source = pre_step_snapshot if pre_step_snapshot is not None else {}
    root_pos_w = state_source.get("root_pos_w", robot.data.root_pos_w)[done_ids]
    root_quat_w = state_source.get("root_quat_w", robot.data.root_quat_w)[done_ids]
    root_lin_vel_w = state_source.get("root_lin_vel_w", robot.data.root_lin_vel_w)[done_ids]
    root_lin_vel_b = state_source.get("root_lin_vel_b", robot.data.root_lin_vel_b)[done_ids]
    root_ang_vel_b = state_source.get("root_ang_vel_b", robot.data.root_ang_vel_b)[done_ids]
    body_state_w = state_source.get("body_state_w", robot.data.body_state_w)[done_ids]
    terrain = getattr(env.unwrapped.scene, "terrain", None)
    terrain_types_source = state_source.get("terrain_types")
    terrain_levels_source = state_source.get("terrain_levels")
    terrain_types = (
        terrain_types_source[done_ids]
        if terrain_types_source is not None
        else terrain.terrain_types[done_ids]
        if terrain is not None and hasattr(terrain, "terrain_types")
        else None
    )
    terrain_levels = (
        terrain_levels_source[done_ids]
        if terrain_levels_source is not None
        else terrain.terrain_levels[done_ids]
        if terrain is not None and hasattr(terrain, "terrain_levels")
        else None
    )

    records = []
    monitored_body_count = len(monitored_body_ids)
    for local_index in torch.nonzero(base_contact_mask, as_tuple=False).squeeze(-1).detach().cpu().tolist():
        peak_flat_index = int(peak_force_indices[local_index].item())
        body_offset = peak_flat_index % monitored_body_count
        colliding_body_sensor_id = int(monitored_body_ids[body_offset])
        colliding_body_id = int(monitored_robot_body_ids[body_offset])
        colliding_body_name = _safe_body_name(list(robot.data.body_names), colliding_body_id)
        colliding_body_state = body_state_w[local_index, colliding_body_id]

        terrain_type_name = None
        if terrain_types is not None:
            terrain_type_value = int(terrain_types[local_index].item())
            terrain_type_name = terrain_lookup.get(terrain_type_value, f"terrain_type_{terrain_type_value}")

        terrain_level_value = None
        if terrain_levels is not None:
            terrain_level_value = int(terrain_levels[local_index].item())

        root_lin_speed_b = float(torch.norm(root_lin_vel_b[local_index], dim=-1).item())
        root_lin_speed_w = float(torch.norm(root_lin_vel_w[local_index], dim=-1).item())
        root_ang_speed_b = float(torch.norm(root_ang_vel_b[local_index], dim=-1).item())
        colliding_body_lin_speed_w = float(torch.norm(colliding_body_state[7:10], dim=-1).item())
        colliding_body_ang_speed_w = float(torch.norm(colliding_body_state[10:13], dim=-1).item())

        records.append(
            {
                "finished_episode_index": _finished_case_index(done_ids, local_index, episode_start_index),
                "env_id": int(done_ids[local_index].item()),
                "terrain_type": terrain_type_name,
                "terrain_level": terrain_level_value,
                "colliding_body_sensor_id": colliding_body_sensor_id,
                "colliding_body_id": colliding_body_id,
                "colliding_body_name": colliding_body_name,
                "peak_contact_force": float(peak_force_values[local_index].item()),
                "approx_collision_position_w": _tensor_to_list(colliding_body_state[0:3]),
                "approx_collision_position_note": (
                    "Body origin in world frame from the last pre-termination frame; exact contact point is unavailable "
                    "because the current contact sensor configuration does not track contact points."
                ),
                "body_lin_vel_w": _tensor_to_list(colliding_body_state[7:10]),
                "body_lin_speed_w": colliding_body_lin_speed_w,
                "body_ang_vel_w": _tensor_to_list(colliding_body_state[10:13]),
                "body_ang_speed_w": colliding_body_ang_speed_w,
                "root_pos_w": _tensor_to_list(root_pos_w[local_index]),
                "root_quat_w": _tensor_to_list(root_quat_w[local_index]),
                "root_lin_vel_w": _tensor_to_list(root_lin_vel_w[local_index]),
                "root_lin_speed_w": root_lin_speed_w,
                "root_lin_vel_b": _tensor_to_list(root_lin_vel_b[local_index]),
                "root_lin_speed_b": root_lin_speed_b,
                "root_ang_vel_b": _tensor_to_list(root_ang_vel_b[local_index]),
                "root_ang_speed_b": root_ang_speed_b,
            }
        )

    return records



def _np_stats(value: np.ndarray) -> dict[str, float | int]:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    finite = np.isfinite(arr)
    if not finite.any():
        return {
            "numel": int(arr.size),
            "finite_ratio": 0.0,
            "min": float("nan"),
            "max": float("nan"),
            "mean": float("nan"),
            "std": float("nan"),
        }
    finite_arr = arr[finite]
    return {
        "numel": int(arr.size),
        "finite_ratio": float(finite.mean()),
        "min": float(finite_arr.min()),
        "max": float(finite_arr.max()),
        "mean": float(finite_arr.mean()),
        "std": float(finite_arr.std()),
    }


def _tensor_env_np(value: torch.Tensor, env_id: int) -> np.ndarray:
    if value.dim() == 0:
        return value.detach().cpu().numpy()
    env_id = max(0, min(int(env_id), int(value.shape[0]) - 1))
    return value[env_id].detach().cpu().numpy().astype(np.float32, copy=False)


def _capture_eval_depth_m(env, env_id: int) -> np.ndarray:
    sensors = getattr(env.unwrapped.scene, "sensors", None)
    depth_sensor = None
    if sensors is not None and hasattr(sensors, "get"):
        depth_sensor = sensors.get("raycast_camera") or sensors.get("camera")
    if depth_sensor is None or "distance_to_image_plane" not in depth_sensor.data.output:
        return np.zeros((0,), dtype=np.float32)
    height, width = depth_sensor.image_shape
    depth = depth_sensor.data.output["distance_to_image_plane"].view(env.unwrapped.num_envs, height, width)
    depth = torch.nan_to_num(depth, nan=50.0, posinf=50.0, neginf=0.0)
    return _tensor_env_np(depth, env_id).astype(np.float32, copy=True)


def _capture_eval_robot_state(env, env_id: int) -> dict[str, np.ndarray]:
    robot = env.unwrapped.scene["robot"]
    data = robot.data
    zeros3 = np.zeros(3, dtype=np.float32)
    return {
        "base_lin_vel_b": _tensor_env_np(getattr(data, "root_lin_vel_b", torch.zeros(env.unwrapped.num_envs, 3, device=env.unwrapped.device)), env_id).reshape(-1)[:3].astype(np.float32, copy=True),
        "base_ang_vel_b": _tensor_env_np(getattr(data, "root_ang_vel_b", torch.zeros(env.unwrapped.num_envs, 3, device=env.unwrapped.device)), env_id).reshape(-1)[:3].astype(np.float32, copy=True),
        "projected_gravity_b": _tensor_env_np(getattr(data, "projected_gravity_b", torch.zeros(env.unwrapped.num_envs, 3, device=env.unwrapped.device)), env_id).reshape(-1)[:3].astype(np.float32, copy=True),
        "joint_pos": _tensor_env_np(getattr(data, "joint_pos", torch.zeros(env.unwrapped.num_envs, 0, device=env.unwrapped.device)), env_id).astype(np.float32, copy=True),
        "joint_vel": _tensor_env_np(getattr(data, "joint_vel", torch.zeros(env.unwrapped.num_envs, 0, device=env.unwrapped.device)), env_id).astype(np.float32, copy=True),
    }


def _capture_eval_goal_b(env, obs_env: np.ndarray) -> np.ndarray:
    # The current SRU actor layout is proprio(16) = base_lin_vel(3), base_ang_vel(3),
    # projected_gravity(3), last_action(3), goal_b(4). Prefer the exact actor obs
    # slice so this dump matches the network input even if command internals change.
    if obs_env.shape[0] >= 16:
        return obs_env[12:16].astype(np.float32, copy=True)
    return np.zeros(4, dtype=np.float32)


class EvalDebugDumper:
    """Dump IsaacLab eval observations/actions in deploy-compatible format."""

    def __init__(self, dump_dir: str | None, env_id: int, every_steps: int, max_steps: int, save_arrays: bool, label: str):
        self.enabled = dump_dir is not None
        self.env_id = int(env_id)
        self.every_steps = int(every_steps)
        self.max_steps = int(max_steps)
        self.save_arrays = bool(save_arrays)
        self.step_count = 0
        self.saved_count = 0
        self.label = str(label)
        self.dump_dir = None
        self.index_path = None
        if self.enabled:
            base_dir = Path(os.path.abspath(os.path.expanduser(str(dump_dir))))
            self.dump_dir = base_dir if label == "primary" else base_dir / label
            self.dump_dir.mkdir(parents=True, exist_ok=True)
            self.index_path = self.dump_dir / "index.jsonl"
            print(
                f"[INFO] Eval debug dump enabled: dir={self.dump_dir}, env_id={self.env_id}, "
                f"every_steps={self.every_steps}, max_steps={self.max_steps}, save_arrays={self.save_arrays}"
            )

    def maybe_dump(self, env, obs: torch.Tensor, actions: torch.Tensor) -> None:
        if not self.enabled or self.every_steps <= 0:
            return
        self.step_count += 1
        if self.step_count % self.every_steps != 0 or self.saved_count >= self.max_steps:
            return
        env_id = max(0, min(self.env_id, int(obs.shape[0]) - 1))
        self.saved_count += 1
        tag = f"step_{self.step_count:08d}"

        obs_env = _tensor_env_np(obs, env_id).reshape(-1).astype(np.float32, copy=True)
        action_env = _tensor_env_np(actions, env_id).reshape(-1)[:3].astype(np.float32, copy=True)
        encoded_depth = obs_env[16:2576].astype(np.float32, copy=True) if obs_env.shape[0] >= 2576 else np.zeros(0, dtype=np.float32)
        scene_tokens = obs_env[2576:2960].astype(np.float32, copy=True) if obs_env.shape[0] >= 2960 else np.zeros(0, dtype=np.float32)
        depth_m = _capture_eval_depth_m(env, env_id)
        robot_state = _capture_eval_robot_state(env, env_id)
        goal_b = _capture_eval_goal_b(env, obs_env)

        record = {
            "step": self.step_count,
            "saved_index": self.saved_count,
            "env_id": int(env_id),
            "label": self.label,
            "file": f"{tag}.npz" if self.save_arrays else None,
            "base_lin_vel_b": robot_state["base_lin_vel_b"].astype(float).tolist(),
            "base_ang_vel_b": robot_state["base_ang_vel_b"].astype(float).tolist(),
            "projected_gravity_b": robot_state["projected_gravity_b"].astype(float).tolist(),
            "goal_b": goal_b.astype(float).tolist(),
            "raw_action": action_env.astype(float).tolist(),
            "command": action_env.astype(float).tolist(),
            "depth_stats": _np_stats(depth_m),
            "encoded_depth_stats": _np_stats(encoded_depth),
            "scene_token_stats": _np_stats(scene_tokens),
            "actor_obs_stats": _np_stats(obs_env),
        }

        if self.save_arrays:
            np.savez_compressed(
                self.dump_dir / f"{tag}.npz",
                base_lin_vel_b=robot_state["base_lin_vel_b"],
                base_ang_vel_b=robot_state["base_ang_vel_b"],
                projected_gravity_b=robot_state["projected_gravity_b"],
                joint_pos=robot_state["joint_pos"],
                joint_vel=robot_state["joint_vel"],
                depth_m=depth_m,
                goal_b=goal_b,
                encoded_depth=encoded_depth,
                scene_tokens=scene_tokens,
                actor_obs=obs_env,
                raw_action=action_env,
                command=action_env,
            )

        with self.index_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


class EvalActionStats:
    """Accumulate batched policy action statistics for simulation/deployment comparison."""

    def __init__(self):
        self._raw_chunks: list[np.ndarray] = []
        self._goal_chunks: list[np.ndarray] = []

    def update(self, obs: torch.Tensor, actions: torch.Tensor) -> None:
        raw = actions.detach().cpu().numpy().astype(np.float32, copy=False)
        self._raw_chunks.append(raw[:, :3].copy())
        if obs.shape[1] >= 16:
            self._goal_chunks.append(obs[:, 12:16].detach().cpu().numpy().astype(np.float32, copy=True))

    def summarize(self) -> dict[str, object]:
        if not self._raw_chunks:
            return {"samples": 0}
        raw = np.concatenate(self._raw_chunks, axis=0)
        processed = np.tanh(raw)
        names = ("vx", "vy", "wz")

        def summarize_array(values: np.ndarray) -> dict[str, object]:
            return {
                "mean": dict(zip(names, np.mean(values, axis=0).astype(float).tolist())),
                "std": dict(zip(names, np.std(values, axis=0).astype(float).tolist())),
                "min": dict(zip(names, np.min(values, axis=0).astype(float).tolist())),
                "p01": dict(zip(names, np.quantile(values, 0.01, axis=0).astype(float).tolist())),
                "p05": dict(zip(names, np.quantile(values, 0.05, axis=0).astype(float).tolist())),
                "p50": dict(zip(names, np.quantile(values, 0.50, axis=0).astype(float).tolist())),
                "p95": dict(zip(names, np.quantile(values, 0.95, axis=0).astype(float).tolist())),
                "p99": dict(zip(names, np.quantile(values, 0.99, axis=0).astype(float).tolist())),
                "max": dict(zip(names, np.max(values, axis=0).astype(float).tolist())),
                "negative_rate": dict(zip(names, np.mean(values < 0.0, axis=0).astype(float).tolist())),
                "positive_rate": dict(zip(names, np.mean(values > 0.0, axis=0).astype(float).tolist())),
            }

        summary = {
            "samples": int(raw.shape[0]),
            "raw": summarize_array(raw),
            "tanh": summarize_array(processed),
            "tanh_saturation_rate_abs_ge_0_95": dict(
                zip(names, np.mean(np.abs(processed) >= 0.95, axis=0).astype(float).tolist())
            ),
            "tanh_saturation_rate_abs_ge_0_99": dict(
                zip(names, np.mean(np.abs(processed) >= 0.99, axis=0).astype(float).tolist())
            ),
        }
        if self._goal_chunks:
            goal = np.concatenate(self._goal_chunks, axis=0)
            goal_xy = goal[:, :2]
            for action_index, action_name in enumerate(names[:2]):
                for goal_index, goal_name in enumerate(("goal_x", "goal_y")):
                    summary[f"correlation_{action_name}_{goal_name}"] = float(
                        np.corrcoef(processed[:, action_index], goal_xy[:, goal_index])[0, 1]
                    )
            goal_y_sign = np.sign(goal_xy[:, 1])
            action_y_sign = np.sign(processed[:, 1])
            nonzero_goal_y = np.abs(goal_xy[:, 1]) > 1.0e-3
            summary["vy_goal_y_sign_agreement_rate"] = float(
                np.mean(action_y_sign[nonzero_goal_y] == goal_y_sign[nonzero_goal_y])
            )
        return summary

def _load_inference_policy_for_env(env, agent_cfg: RslRlOnPolicyRunnerCfg, checkpoint_path: str):
    checkpoint_path = os.path.abspath(os.path.expanduser(checkpoint_path))
    policy = try_load_torchscript_policy(checkpoint_path, "cpu")
    if policy is not None:
        policy = policy.to(env.unwrapped.device)
        if hasattr(policy, "reset"):
            policy.reset()
        return policy, None, True, checkpoint_path

    checkpoint_dict = load_checkpoint_dict(checkpoint_path)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    if runner.is_mdpo:
        runner.alg.actor_critic_1.load_state_dict(checkpoint_dict["model_state_dict"], strict=True)
        runner.alg.actor_critic_2.load_state_dict(checkpoint_dict["model_state_dict"], strict=True)
    else:
        runner.alg.actor_critic.load_state_dict(checkpoint_dict["model_state_dict"], strict=True)

    if runner.empirical_normalization:
        runner.obs_normalizer.load_state_dict(checkpoint_dict["obs_norm_state_dict"])
        runner.critic_obs_normalizer.load_state_dict(checkpoint_dict["critic_obs_norm_state_dict"])

    runner.current_learning_iteration = checkpoint_dict["iter"]
    print(f"[INFO] Loaded checkpoint from iteration {checkpoint_dict['iter']}: {checkpoint_path}")
    return (
        runner.get_inference_policy(device=env.unwrapped.device),
        runner.get_policy_reset(device=env.unwrapped.device),
        False,
        checkpoint_path,
    )


class _PolicyObservationSliceEnvAdapter:
    """Adapter used to build legacy runners against a prefix of policy observations."""

    def __init__(self, env, policy_obs_dim: int):
        self._env = env
        self._policy_obs_dim = int(policy_obs_dim)
        self.num_obs = self._policy_obs_dim

    def __getattr__(self, name: str):
        return getattr(self._env, name)

    def get_observations(self):
        obs, extras = self._env.get_observations()
        return obs[:, : self._policy_obs_dim], extras


def _checkpoint_has_state_prefix(checkpoint_dict: dict, prefix: str) -> bool:
    return any(key.startswith(prefix) for key in checkpoint_dict.get("model_state_dict", {}))


def _actor_visibility_feature_count(agent_cfg: RslRlOnPolicyRunnerCfg) -> int:
    policy_cfg = agent_cfg.policy
    if not getattr(policy_cfg, "use_actor_visibility_map", False):
        return 0
    dims = getattr(policy_cfg, "visibility_map_input_dims", None)
    if dims is None:
        return 0
    return int(math.prod(dims))


def _load_compare_inference_policy_for_env(env, agent_cfg: RslRlOnPolicyRunnerCfg, checkpoint_path: str):
    checkpoint_path = os.path.abspath(os.path.expanduser(checkpoint_path))
    policy = try_load_torchscript_policy(checkpoint_path, "cpu")
    if policy is not None:
        policy = policy.to(env.unwrapped.device)
        if hasattr(policy, "reset"):
            policy.reset()
        return policy, None, True, checkpoint_path

    checkpoint_dict = load_checkpoint_dict(checkpoint_path)
    compare_agent_cfg = copy.deepcopy(agent_cfg)
    visibility_features = _actor_visibility_feature_count(compare_agent_cfg)
    crop_policy_obs_dim = None

    if visibility_features > 0 and not _checkpoint_has_state_prefix(checkpoint_dict, "visibility_map_encoder."):
        obs, _ = env.get_observations()
        crop_policy_obs_dim = int(obs.shape[1]) - visibility_features
        compare_agent_cfg.policy.use_actor_visibility_map = False
        compare_agent_cfg.policy.use_actor_visibility_query_attention = False
        print(
            "[INFO] Compare checkpoint has no actor visibility-map branch; "
            f"using policy observation prefix dim {crop_policy_obs_dim} inside the same env scene."
        )

    runner_env = env
    if crop_policy_obs_dim is not None:
        runner_env = _PolicyObservationSliceEnvAdapter(env, crop_policy_obs_dim)

    runner = OnPolicyRunner(runner_env, compare_agent_cfg.to_dict(), log_dir=None, device=compare_agent_cfg.device)
    if runner.is_mdpo:
        runner.alg.actor_critic_1.load_state_dict(checkpoint_dict["model_state_dict"], strict=True)
        runner.alg.actor_critic_2.load_state_dict(checkpoint_dict["model_state_dict"], strict=True)
    else:
        runner.alg.actor_critic.load_state_dict(checkpoint_dict["model_state_dict"], strict=True)

    if runner.empirical_normalization:
        runner.obs_normalizer.load_state_dict(checkpoint_dict["obs_norm_state_dict"])
        runner.critic_obs_normalizer.load_state_dict(checkpoint_dict["critic_obs_norm_state_dict"])

    runner.current_learning_iteration = checkpoint_dict["iter"]
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    if crop_policy_obs_dim is not None:
        base_policy = policy

        def policy(obs):
            return base_policy(obs[:, :crop_policy_obs_dim])

    print(f"[INFO] Loaded compare checkpoint from iteration {checkpoint_dict['iter']}: {checkpoint_path}")
    return (
        policy,
        runner.get_policy_reset(device=env.unwrapped.device),
        False,
        checkpoint_path,
    )


def _run_eval_pass(
    *,
    env,
    policy,
    policy_reset,
    using_torchscript_policy: bool,
    checkpoint_path: str,
    failure_rgb_output_dir: str | None,
    json_out: str | None,
    label: str,
) -> dict[str, object]:
    obs, _ = env.get_observations()
    reset_policy_state = True
    debug_dumper = EvalDebugDumper(
        dump_dir=args_cli.debug_dump_dir,
        env_id=args_cli.debug_dump_env_id,
        every_steps=args_cli.debug_dump_every_steps,
        max_steps=args_cli.debug_dump_max_steps,
        save_arrays=not args_cli.debug_dump_stats_only,
        label=label,
    )
    action_stats = EvalActionStats()

    goal_term = env.unwrapped.command_manager._terms["robot_goal"]
    success_tracker = goal_term.success_tracker
    termination_manager = env.unwrapped.termination_manager
    termination_terms = list(termination_manager.active_terms)
    terrain_lookup = build_terrain_type_lookup(env)
    base_contact_context = resolve_base_contact_context(env)
    topdown_render_context = resolve_topdown_render_context(env)

    failure_rgb_camera = None
    failure_rgb_success_limit = _resolved_failure_rgb_success_limit()
    failure_rgb_failure_limit = _resolved_failure_rgb_failure_limit()
    failure_rgb_success_rendered_count = 0
    failure_rgb_failure_rendered_count = 0
    failure_rgb_success_distance_skipped_count = 0
    failure_rgb_success_height_delta_skipped_count = 0
    failure_rgb_rendered_paths: list[str] = []
    if failure_rgb_output_dir is not None and (failure_rgb_success_limit != 0 or failure_rgb_failure_limit != 0):
        failure_rgb_output_dir = os.path.abspath(os.path.expanduser(failure_rgb_output_dir))
        os.makedirs(failure_rgb_output_dir, exist_ok=True)
        failure_rgb_camera = create_failure_rgb_camera(env)

    if args_cli.failure_rgb_full_episode:
        episode_history_frames = int(getattr(env.unwrapped, "max_episode_length", 0) or 0) + 2
        if episode_history_frames <= 2:
            episode_history_frames = max(1, int(round(float(env.unwrapped.cfg.episode_length_s) / float(env.unwrapped.step_dt)))) + 2
    else:
        episode_history_frames = 1
    pre_step_history_frames = max(
        1,
        int(args_cli.failure_rgb_history_frames),
        int(args_cli.observation_gif_frames),
        int(args_cli.local_terrain_map_history_frames) if args_cli.local_terrain_map else 1,
        episode_history_frames,
    )
    pre_step_snapshot_history: deque[dict[str, torch.Tensor]] = deque(maxlen=pre_step_history_frames)
    pre_step_observation_history: deque[dict[str, list[np.ndarray | None]]] = deque(maxlen=pre_step_history_frames)
    episode_ids = torch.zeros(env.unwrapped.num_envs, dtype=torch.long, device=env.unwrapped.device)
    _capture_observation_frame_cache._noisy_depth_history = deque(maxlen=pre_step_history_frames)
    _capture_observation_frame_cache._warned_noisy_depth_fallback = False
    local_terrain_map = None
    if args_cli.local_terrain_map:
        local_terrain_map = DepthPoseLocalTerrainMap(
            num_envs=env.unwrapped.num_envs,
            size_m=args_cli.local_terrain_map_size_m,
            resolution_m=args_cli.local_terrain_map_resolution_m,
            history_frames=args_cli.local_terrain_map_history_frames,
            stride=args_cli.local_terrain_map_stride,
            depth_min_m=args_cli.local_terrain_map_depth_min_m,
            depth_max_m=args_cli.local_terrain_map_depth_max_m,
            height_range_m=args_cli.local_terrain_map_height_range_m,
            gradient_threshold_m=args_cli.local_terrain_map_gradient_threshold_m,
        )

    total_episodes = 0
    total_successes = 0
    termination_counts = {name: 0 for name in termination_terms}
    base_contact_failure_count = 0
    base_contact_failure_records: list[dict[str, object]] = []
    base_contact_body_stats: dict[str, dict[str, float]] = defaultdict(
        lambda: {"count": 0.0, "sum_peak_contact_force": 0.0, "sum_root_lin_speed_b": 0.0, "max_peak_contact_force": 0.0}
    )
    terrain_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"episodes": 0, "successes": 0})
    terrain_level_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"episodes": 0, "successes": 0})
    terrain_failure_stats: dict[str, dict[str, object]] = defaultdict(
        lambda: {
            "episodes": 0,
            "successes": 0,
            "termination_counts": {name: 0 for name in termination_terms},
        }
    )
    terrain_level_failure_stats: dict[str, dict[str, object]] = defaultdict(
        lambda: {
            "episodes": 0,
            "successes": 0,
            "termination_counts": {name: 0 for name in termination_terms},
        }
    )
    initial_case_done_mask = None
    if args_cli.initial_env_cases_only:
        if args_cli.num_episodes > env.unwrapped.num_envs:
            raise ValueError(
                "--initial_env_cases_only requires --num_episodes <= num_envs so every counted case can be an initial env episode, "
                f"got num_episodes={args_cli.num_episodes}, num_envs={env.unwrapped.num_envs}."
            )
        initial_case_done_mask = torch.zeros(env.unwrapped.num_envs, dtype=torch.bool, device=env.unwrapped.device)

    print(
        f"[INFO] Starting evaluation pass '{label}': task={args_cli.task}, num_envs={env.unwrapped.num_envs}, "
        f"target_episodes={args_cli.num_episodes}"
    )

    while simulation_app.is_running() and total_episodes < args_cli.num_episodes:
        pre_step_snapshot = capture_base_contact_snapshot(env, base_contact_context)
        pre_step_delay_snapshot = _capture_observation_delay_snapshot(env)
        if pre_step_snapshot is not None:
            pre_step_snapshot["episode_ids"] = episode_ids.detach().clone()
            pre_step_snapshot_history.append(pre_step_snapshot)
            if local_terrain_map is not None:
                local_terrain_map.update(env)
            if args_cli.failure_rgb_include_observations or local_terrain_map is not None:
                pre_step_observation_history.append(_capture_observation_frame_cache(env, local_terrain_map))
        with torch.no_grad():
            if using_torchscript_policy:
                actions = policy(obs, reset_policy_state)
                reset_policy_state = False
            else:
                actions = policy(obs)
            action_stats.update(obs, actions)
            debug_dumper.maybe_dump(env, obs, actions)
            obs, _, dones, _ = env.step(actions)

        raw_done_ids = torch.nonzero(dones > 0, as_tuple=False).squeeze(-1)
        if raw_done_ids.numel() == 0:
            continue
        done_ids = raw_done_ids
        if initial_case_done_mask is not None:
            first_case_mask = torch.logical_not(initial_case_done_mask[done_ids])
            done_ids = done_ids[first_case_mask]
            if done_ids.numel() == 0:
                episode_ids[raw_done_ids] += 1
                continue

        if using_torchscript_policy:
            reset_policy_state = True
        elif policy_reset is not None:
            with torch.no_grad():
                policy_reset(dones > 0)

        remaining = args_cli.num_episodes - total_episodes
        if done_ids.numel() > remaining:
            done_ids = done_ids[:remaining]
        if initial_case_done_mask is not None:
            initial_case_done_mask[done_ids] = True

        latest_indices = (success_tracker.write_index[done_ids] - 1) % success_tracker.buffer_size
        latest_results = success_tracker.buffer[done_ids, latest_indices]
        successes = latest_results > 0.0
        episode_start_index = total_episodes

        total_episodes += int(done_ids.numel())
        total_successes += int(successes.sum().item())

        done_term_values = {}
        for term_name in termination_terms:
            term_values = termination_manager.get_term(term_name)[done_ids]
            done_term_values[term_name] = term_values.detach().cpu().tolist()
            termination_counts[term_name] += int(term_values.sum().item())

        if base_contact_context is not None and "base_contact" in done_term_values:
            base_contact_mask = termination_manager.get_term("base_contact")[done_ids] > 0
            base_contact_records = collect_base_contact_failure_records(
                env=env,
                done_ids=done_ids,
                base_contact_mask=base_contact_mask,
                base_contact_context=base_contact_context,
                pre_step_snapshot=pre_step_snapshot,
                terrain_lookup=terrain_lookup,
                episode_start_index=episode_start_index,
            )
            base_contact_failure_count += len(base_contact_records)
            for record in base_contact_records:
                body_name = str(record["colliding_body_name"])
                peak_contact_force = float(record["peak_contact_force"])
                root_lin_speed_b = float(record["root_lin_speed_b"])
                body_stats = base_contact_body_stats[body_name]
                body_stats["count"] += 1.0
                body_stats["sum_peak_contact_force"] += peak_contact_force
                body_stats["sum_root_lin_speed_b"] += root_lin_speed_b
                body_stats["max_peak_contact_force"] = max(body_stats["max_peak_contact_force"], peak_contact_force)

            remaining_record_budget = max(0, args_cli.max_base_contact_records - len(base_contact_failure_records))
            if remaining_record_budget > 0:
                base_contact_failure_records.extend(base_contact_records[:remaining_record_budget])
        else:
            base_contact_records = []

        if failure_rgb_output_dir is not None and episode_start_index >= args_cli.failure_rgb_min_episode:
            remaining_success_budget = max(0, failure_rgb_success_limit - failure_rgb_success_rendered_count)
            remaining_failure_budget = max(0, failure_rgb_failure_limit - failure_rgb_failure_rendered_count)
            rendered_success_count, rendered_failure_count, skipped_success_distance_count, skipped_success_height_delta_count, rendered_paths = capture_failure_rgb_screenshots(
                env=env,
                failure_camera=failure_rgb_camera,
                done_ids=done_ids,
                successes=successes,
                done_term_values=done_term_values,
                pre_step_snapshot=pre_step_snapshot,
                pre_step_delay_snapshot=pre_step_delay_snapshot,
                pre_step_snapshot_history=list(pre_step_snapshot_history),
                pre_step_observation_history=list(pre_step_observation_history),
                terrain_lookup=terrain_lookup,
                base_contact_records=base_contact_records,
                episode_start_index=episode_start_index,
                output_dir=failure_rgb_output_dir,
                success_remaining_budget=remaining_success_budget,
                failure_remaining_budget=remaining_failure_budget,
            )
            failure_rgb_success_rendered_count += rendered_success_count
            failure_rgb_failure_rendered_count += rendered_failure_count
            failure_rgb_success_distance_skipped_count += skipped_success_distance_count
            failure_rgb_success_height_delta_skipped_count += skipped_success_height_delta_count
            failure_rgb_rendered_paths.extend(rendered_paths)

        if local_terrain_map is not None:
            local_terrain_map.reset(done_ids)

        terrain = getattr(env.unwrapped.scene, "terrain", None)
        if terrain is not None and hasattr(terrain, "terrain_types"):
            terrain_types = terrain.terrain_types[done_ids].detach().cpu().tolist()
            terrain_levels = terrain.terrain_levels[done_ids].detach().cpu().tolist() if hasattr(terrain, "terrain_levels") else None
            success_list = successes.detach().cpu().tolist()
            for index, terrain_type in enumerate(terrain_types):
                terrain_name = terrain_lookup.get(int(terrain_type), f"terrain_type_{int(terrain_type)}")
                terrain_stats[terrain_name]["episodes"] += 1
                terrain_stats[terrain_name]["successes"] += int(bool(success_list[index]))
                terrain_failure_stats[terrain_name]["episodes"] += 1
                terrain_failure_stats[terrain_name]["successes"] += int(bool(success_list[index]))
                for term_name in termination_terms:
                    terrain_failure_stats[terrain_name]["termination_counts"][term_name] += int(
                        bool(done_term_values[term_name][index])
                    )
                if terrain_levels is not None:
                    level_name = f"level_{int(terrain_levels[index])}"
                    terrain_level_stats[level_name]["episodes"] += 1
                    terrain_level_stats[level_name]["successes"] += int(bool(success_list[index]))
                    terrain_level_failure_stats[level_name]["episodes"] += 1
                    terrain_level_failure_stats[level_name]["successes"] += int(bool(success_list[index]))
                    for term_name in termination_terms:
                        terrain_level_failure_stats[level_name]["termination_counts"][term_name] += int(
                            bool(done_term_values[term_name][index])
                        )

        episode_ids[raw_done_ids] += 1

        if args_cli.print_every > 0 and total_episodes % args_cli.print_every == 0:
            print(
                f"[INFO] Progress {label}: {total_episodes}/{args_cli.num_episodes} episodes, "
                f"success_rate={summarize_rates(total_successes, total_episodes):.4f}"
            )

    summary = {
        "task": args_cli.task,
        "checkpoint": checkpoint_path,
        "debug_dump": {
            "enabled": bool(debug_dumper.enabled),
            "dir": str(debug_dumper.dump_dir) if debug_dumper.dump_dir is not None else None,
            "env_id": int(debug_dumper.env_id),
            "saved_steps": int(debug_dumper.saved_count),
            "stats_only": bool(not debug_dumper.save_arrays),
        },
        "num_envs": int(env.unwrapped.num_envs),
        "episodes": total_episodes,
        "successes": total_successes,
        "success_rate": summarize_rates(total_successes, total_episodes),
        "action_stats": action_stats.summarize(),
        "termination_rates": {
            term_name: {"count": count, "rate": summarize_rates(count, total_episodes)}
            for term_name, count in termination_counts.items()
        },
        "terrain_geometry": terrain_geometry_signature(env),
        "terrain_type_success": {
            name: {
                "episodes": values["episodes"],
                "successes": values["successes"],
                "success_rate": summarize_rates(values["successes"], values["episodes"]),
            }
            for name, values in sorted(terrain_stats.items())
        },
        "terrain_level_success": {
            name: {
                "episodes": values["episodes"],
                "successes": values["successes"],
                "success_rate": summarize_rates(values["successes"], values["episodes"]),
            }
            for name, values in sorted(terrain_level_stats.items())
        },
        "terrain_type_failure_breakdown": {
            name: {
                "episodes": int(values["episodes"]),
                "successes": int(values["successes"]),
                "failures": int(values["episodes"] - values["successes"]),
                "success_rate": summarize_rates(int(values["successes"]), int(values["episodes"])),
                "termination_rates": {
                    term_name: {"count": int(count), "rate": summarize_rates(int(count), int(values["episodes"]))}
                    for term_name, count in values["termination_counts"].items()
                },
            }
            for name, values in sorted(terrain_failure_stats.items())
        },
        "terrain_level_failure_breakdown": {
            name: {
                "episodes": int(values["episodes"]),
                "successes": int(values["successes"]),
                "failures": int(values["episodes"] - values["successes"]),
                "success_rate": summarize_rates(int(values["successes"]), int(values["episodes"])),
                "termination_rates": {
                    term_name: {"count": int(count), "rate": summarize_rates(int(count), int(values["episodes"]))}
                    for term_name, count in values["termination_counts"].items()
                },
            }
            for name, values in sorted(terrain_level_failure_stats.items())
        },
        "base_contact_diagnostics": {
            "enabled": base_contact_context is not None,
            "sensor_name": base_contact_context["sensor_name"] if base_contact_context is not None else None,
            "threshold": base_contact_context["threshold"] if base_contact_context is not None else None,
            "monitored_bodies": base_contact_context["body_names"] if base_contact_context is not None else [],
            "num_failures": int(base_contact_failure_count),
            "recorded_failures": int(len(base_contact_failure_records)),
            "record_limit": int(args_cli.max_base_contact_records),
            "body_breakdown": {
                body_name: {
                    "count": int(values["count"]),
                    "rate_within_base_contact_failures": summarize_rates(int(values["count"]), int(base_contact_failure_count)),
                    "mean_peak_contact_force": float(values["sum_peak_contact_force"] / values["count"]),
                    "max_peak_contact_force": float(values["max_peak_contact_force"]),
                    "mean_root_lin_speed_b": float(values["sum_root_lin_speed_b"] / values["count"]),
                }
                for body_name, values in sorted(
                    base_contact_body_stats.items(), key=lambda item: (-item[1]["count"], item[0])
                )
                if values["count"] > 0
            },
            "failures": base_contact_failure_records,
        },
        "failure_rgb_renders": {
            "enabled": failure_rgb_output_dir is not None,
            "output_dir": failure_rgb_output_dir,
            "render_limit": int(args_cli.failure_rgb_limit),
            "success_render_limit": int(failure_rgb_success_limit),
            "failure_render_limit": int(failure_rgb_failure_limit),
            "success_min_initial_distance_m": float(args_cli.failure_rgb_success_min_initial_distance),
            "success_min_goal_height_delta_m": float(args_cli.failure_rgb_success_min_goal_height_delta),
            "terrain_name_filter": args_cli.failure_rgb_terrain_name,
            "success_distance_skipped_count": int(failure_rgb_success_distance_skipped_count),
            "success_height_delta_skipped_count": int(failure_rgb_success_height_delta_skipped_count),
            "base_contact_only": bool(args_cli.failure_rgb_base_contact_only),
            "include_successes": bool(args_cli.failure_rgb_include_successes),
            "include_observations": bool(args_cli.failure_rgb_include_observations),
            "width": int(args_cli.failure_rgb_width),
            "height": int(args_cli.failure_rgb_height),
            "camera_height_m": float(args_cli.failure_rgb_height_m),
            "focal_length": float(args_cli.failure_rgb_focal_length),
            "center_mode": str(args_cli.failure_rgb_center_mode),
            "follow_robot": bool(args_cli.failure_rgb_follow_robot),
            "export_mode": str(args_cli.failure_rgb_export_mode),
            "history_frames": int(args_cli.failure_rgb_history_frames),
            "full_episode": bool(args_cli.failure_rgb_full_episode),
            "gif_frame_ms": int(args_cli.failure_rgb_gif_frame_ms),
            "include_rear_view": bool(args_cli.failure_rgb_include_rear_view),
            "rear_view_height_m": float(args_cli.failure_rgb_rear_height_m),
            "rear_view_distance_m": float(args_cli.failure_rgb_rear_distance_m),
            "rear_view_lateral_m": float(args_cli.failure_rgb_rear_lateral_m),
            "rear_view_target_height_m": float(args_cli.failure_rgb_rear_lookahead_m),
            "rear_view_orientation": "gravity_aligned_look_at",
            "success_rendered_count": int(failure_rgb_success_rendered_count),
            "failure_rendered_count": int(failure_rgb_failure_rendered_count),
            "rendered_count": int(failure_rgb_success_rendered_count + failure_rgb_failure_rendered_count),
            "sample_paths": failure_rgb_rendered_paths[:10],
        },
        "eval_overrides_enabled": not args_cli.disable_eval_overrides,
    }

    print(f"\n[RESULT] Evaluation summary ({label})")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if json_out:
        output_path = os.path.abspath(os.path.expanduser(json_out))
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as output_file:
            json.dump(summary, output_file, indent=2, sort_keys=True)
            output_file.write("\n")
        print(f"[INFO] Wrote JSON summary to: {output_path}")
    return summary


def main():
    if args_cli.task is None:
        raise ValueError("Argument '--task' is required. Pass a registered Gym environment id.")

    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    agent_cfg: RslRlOnPolicyRunnerCfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.episode_length_s is not None:
        if args_cli.episode_length_s <= 0.0:
            raise ValueError(f"--episode_length_s must be positive, got {args_cli.episode_length_s}.")
        env_cfg.episode_length_s = args_cli.episode_length_s
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed
        agent_cfg.seed = args_cli.seed
    if not args_cli.disable_eval_overrides:
        maybe_apply_eval_overrides(env_cfg)
    project_root = Path(__file__).resolve().parents[3]
    showcase_asset_root = None
    if args_cli.showcase_scene == "rivermark":
        showcase_asset_root = str(project_root / "assets" / "showcase" / "rivermark")
    elif args_cli.showcase_scene == "outdoor_campus":
        showcase_asset_root = str(project_root / "assets" / "showcase" / "outdoor_campus")
    elif args_cli.showcase_scene == "official_warehouse":
        showcase_asset_root = str(project_root / "assets" / "showcase" / "official_warehouse")
    if apply_showcase_scene_preset(
        env_cfg,
        args_cli.showcase_scene,
        sim_utils=sim_utils,
        isaac_nucleus_dir=ISAAC_NUCLEUS_DIR,
        isaaclab_nucleus_dir=ISAACLAB_NUCLEUS_DIR,
        asset_root=showcase_asset_root,
    ):
        print(f"[INFO] Applied showcase scene preset: {args_cli.showcase_scene}")

    if args_cli.checkpoint:
        resume_path = args_cli.checkpoint
    else:
        log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        resume_path = find_latest_checkpoint(log_root_path, checkpoint_pattern="model_.*.pt")

    resume_path = os.path.abspath(os.path.expanduser(resume_path))

    policy = try_load_torchscript_policy(resume_path, "cpu")
    using_torchscript_policy = policy is not None
    checkpoint_dict = None

    if not using_torchscript_policy:
        checkpoint_dict = load_checkpoint_dict(resume_path)
        maybe_apply_legacy_checkpoint_policy_overrides(env_cfg, agent_cfg, checkpoint_dict)

    if args_cli.depth_pit_to_obstacle:
        if not hasattr(env_cfg, "observations") or not hasattr(env_cfg.observations, "policy"):
            raise ValueError("--depth_pit_to_obstacle requires a policy observation group.")
        if not hasattr(env_cfg.observations.policy, "depth_image"):
            raise ValueError("--depth_pit_to_obstacle requires a policy.depth_image observation term.")
        env_cfg.observations.policy.depth_image.func = depth_image_pit_to_obstacle_noisy_delayed
        print(
            "[INFO] Enabled zero-shot depth pit-to-obstacle preprocessing for policy depth observation "
            f"(jump>={args_cli.depth_pit_jump_m}m, margin={args_cli.depth_pit_obstacle_margin_m}m)."
        )

    if using_torchscript_policy:
        if env_cfg.scene.num_envs != 1:
            if args_cli.num_envs is None:
                print("[INFO] TorchScript deployment policy detected; overriding num_envs to 1.")
                env_cfg.scene.num_envs = 1
            else:
                raise ValueError(
                    f"TorchScript deployment checkpoints only support '--num_envs 1', got {env_cfg.scene.num_envs}."
                )

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = LegacyRslRlVecEnvWrapper(env)

    if args_cli.force_dynamic_obstacles:
        from instinct_nav.navigation.mdp.dynamic_obstacles import _get_or_create_dynamic_obstacle_manager

        # Build the manager explicitly from the configured event params instead of
        # relying on the reset chain having already created it. This guarantees the
        # flag is never a silent no-op.
        update_term = getattr(getattr(env.unwrapped.cfg, "events", None), "dynamic_obstacles_update", None)
        params = dict(getattr(update_term, "params", {}) or {})
        obstacle_names = params.pop("obstacle_names", None)
        if obstacle_names:
            manager = _get_or_create_dynamic_obstacle_manager(
                env.unwrapped, obstacle_names=list(obstacle_names), **params
            )
            manager.enable()
            print("[INFO] Dynamic obstacles forced ENABLED for evaluation.")
        else:
            print(
                "[WARN] --force_dynamic_obstacles set but task has no dynamic-obstacle "
                "event term; ignoring."
            )

    policy_reset = None
    if using_torchscript_policy:
        policy = policy.to(env.unwrapped.device)
        if hasattr(policy, "reset"):
            policy.reset()
    else:
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        if checkpoint_dict is None:
            load_checkpoint_with_fallback(runner, resume_path)
        else:
            if runner.is_mdpo:
                runner.alg.actor_critic_1.load_state_dict(checkpoint_dict["model_state_dict"], strict=True)
                runner.alg.actor_critic_2.load_state_dict(checkpoint_dict["model_state_dict"], strict=True)
            else:
                runner.alg.actor_critic.load_state_dict(checkpoint_dict["model_state_dict"], strict=True)

            if runner.empirical_normalization:
                runner.obs_normalizer.load_state_dict(checkpoint_dict["obs_norm_state_dict"])
                runner.critic_obs_normalizer.load_state_dict(checkpoint_dict["critic_obs_norm_state_dict"])

            runner.current_learning_iteration = checkpoint_dict["iter"]
            print(f"[INFO] Loaded checkpoint from iteration {checkpoint_dict['iter']}")
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        policy_reset = runner.get_policy_reset(device=env.unwrapped.device)

    if args_cli.case_manifest_in is not None:
        _apply_initial_case_manifest(env, args_cli.case_manifest_in)
    if args_cli.case_manifest_out is not None:
        _write_case_manifest(env, args_cli.case_manifest_out, args_cli.num_episodes)
    compare_case_manifest = None
    if args_cli.compare_checkpoint is not None:
        compare_case_manifest = _capture_initial_case_manifest(env, args_cli.num_episodes)

    obs, _ = env.get_observations()
    reset_policy_state = True

    goal_term = env.unwrapped.command_manager._terms["robot_goal"]
    success_tracker = goal_term.success_tracker
    termination_manager = env.unwrapped.termination_manager
    termination_terms = list(termination_manager.active_terms)
    terrain_lookup = build_terrain_type_lookup(env)
    base_contact_context = resolve_base_contact_context(env)
    topdown_render_context = resolve_topdown_render_context(env)
    failure_rgb_camera = None
    failure_rgb_output_dir = None
    failure_rgb_success_limit = _resolved_failure_rgb_success_limit()
    failure_rgb_failure_limit = _resolved_failure_rgb_failure_limit()
    failure_rgb_success_rendered_count = 0
    failure_rgb_failure_rendered_count = 0
    failure_rgb_success_distance_skipped_count = 0
    failure_rgb_success_height_delta_skipped_count = 0
    failure_rgb_rendered_paths: list[str] = []
    if args_cli.failure_rgb_dir and (failure_rgb_success_limit != 0 or failure_rgb_failure_limit != 0):
        failure_rgb_output_dir = os.path.abspath(os.path.expanduser(args_cli.failure_rgb_dir))
        os.makedirs(failure_rgb_output_dir, exist_ok=True)
        failure_rgb_camera = create_failure_rgb_camera(env)
    observation_gif_output_dir = None
    observation_gif_rendered_count = 0
    observation_gif_rendered_paths: list[str] = []
    if args_cli.observation_gif_dir and args_cli.observation_gif_limit != 0:
        observation_gif_output_dir = os.path.abspath(os.path.expanduser(args_cli.observation_gif_dir))
        os.makedirs(observation_gif_output_dir, exist_ok=True)
    local_terrain_map = None
    if args_cli.local_terrain_map:
        local_terrain_map = DepthPoseLocalTerrainMap(
            num_envs=env.unwrapped.num_envs,
            size_m=args_cli.local_terrain_map_size_m,
            resolution_m=args_cli.local_terrain_map_resolution_m,
            history_frames=args_cli.local_terrain_map_history_frames,
            stride=args_cli.local_terrain_map_stride,
            depth_min_m=args_cli.local_terrain_map_depth_min_m,
            depth_max_m=args_cli.local_terrain_map_depth_max_m,
            height_range_m=args_cli.local_terrain_map_height_range_m,
            gradient_threshold_m=args_cli.local_terrain_map_gradient_threshold_m,
        )
        print(
            "[INFO] Enabled eval-only local terrain map visualization "
            f"(size={local_terrain_map.size_m:.2f}m, res={local_terrain_map.resolution_m:.3f}m, "
            f"history={local_terrain_map.history_frames}, stride={local_terrain_map.stride}, "
            f"depth_range=[{local_terrain_map.depth_min_m:.2f}, {local_terrain_map.depth_max_m:.2f}]m)."
        )
    pre_step_snapshot = capture_base_contact_snapshot(env, base_contact_context)
    pre_step_delay_snapshot = _capture_observation_delay_snapshot(env)
    if args_cli.failure_rgb_full_episode:
        episode_history_frames = int(getattr(env.unwrapped, "max_episode_length", 0) or 0) + 2
        if episode_history_frames <= 2:
            episode_history_frames = max(1, int(round(float(env.unwrapped.cfg.episode_length_s) / float(env.unwrapped.step_dt)))) + 2
    else:
        episode_history_frames = 1
    pre_step_history_frames = max(
        1,
        int(args_cli.failure_rgb_history_frames),
        int(args_cli.observation_gif_frames),
        int(args_cli.local_terrain_map_history_frames) if args_cli.local_terrain_map else 1,
        episode_history_frames,
    )
    pre_step_snapshot_history: deque[dict[str, torch.Tensor]] = deque(maxlen=pre_step_history_frames)
    pre_step_observation_history: deque[dict[str, list[np.ndarray | None]]] = deque(maxlen=pre_step_history_frames)
    episode_ids = torch.zeros(env.unwrapped.num_envs, dtype=torch.long, device=env.unwrapped.device)
    _capture_observation_frame_cache._noisy_depth_history = deque(maxlen=pre_step_history_frames)
    _capture_observation_frame_cache._warned_noisy_depth_fallback = False
    topdown_rendered_count = 0
    topdown_rendered_paths: list[str] = []
    topdown_output_dir = None
    if args_cli.failure_topdown_dir and args_cli.failure_topdown_limit != 0:
        topdown_output_dir = os.path.abspath(os.path.expanduser(args_cli.failure_topdown_dir))
        os.makedirs(topdown_output_dir, exist_ok=True)

    total_episodes = 0
    total_successes = 0
    termination_counts = {name: 0 for name in termination_terms}
    base_contact_failure_count = 0
    base_contact_failure_records: list[dict[str, object]] = []
    base_contact_body_stats: dict[str, dict[str, float]] = defaultdict(
        lambda: {"count": 0.0, "sum_peak_contact_force": 0.0, "sum_root_lin_speed_b": 0.0, "max_peak_contact_force": 0.0}
    )
    terrain_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"episodes": 0, "successes": 0})
    terrain_level_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"episodes": 0, "successes": 0})
    terrain_failure_stats: dict[str, dict[str, object]] = defaultdict(
        lambda: {
            "episodes": 0,
            "successes": 0,
            "termination_counts": {name: 0 for name in termination_terms},
        }
    )
    terrain_level_failure_stats: dict[str, dict[str, object]] = defaultdict(
        lambda: {
            "episodes": 0,
            "successes": 0,
            "termination_counts": {name: 0 for name in termination_terms},
        }
    )
    initial_case_done_mask = None
    if args_cli.initial_env_cases_only:
        if args_cli.num_episodes > env.unwrapped.num_envs:
            raise ValueError(
                "--initial_env_cases_only requires --num_episodes <= num_envs so every counted case can be an initial env episode, "
                f"got num_episodes={args_cli.num_episodes}, num_envs={env.unwrapped.num_envs}."
            )
        initial_case_done_mask = torch.zeros(env.unwrapped.num_envs, dtype=torch.bool, device=env.unwrapped.device)

    debug_dumper = EvalDebugDumper(
        dump_dir=args_cli.debug_dump_dir,
        env_id=args_cli.debug_dump_env_id,
        every_steps=args_cli.debug_dump_every_steps,
        max_steps=args_cli.debug_dump_max_steps,
        save_arrays=not args_cli.debug_dump_stats_only,
        label="primary",
    )
    action_stats = EvalActionStats()

    print(
        f"[INFO] Starting evaluation: task={args_cli.task}, num_envs={env.unwrapped.num_envs}, "
        f"target_episodes={args_cli.num_episodes}"
    )

    while simulation_app.is_running() and total_episodes < args_cli.num_episodes:
        pre_step_snapshot = capture_base_contact_snapshot(env, base_contact_context)
        pre_step_delay_snapshot = _capture_observation_delay_snapshot(env)
        if pre_step_snapshot is not None:
            pre_step_snapshot["episode_ids"] = episode_ids.detach().clone()
            pre_step_snapshot_history.append(pre_step_snapshot)
            if local_terrain_map is not None:
                local_terrain_map.update(env)
            if args_cli.failure_rgb_include_observations or local_terrain_map is not None:
                pre_step_observation_history.append(_capture_observation_frame_cache(env, local_terrain_map))
        with torch.no_grad():
            if using_torchscript_policy:
                actions = policy(obs, reset_policy_state)
                reset_policy_state = False
            else:
                actions = policy(obs)

            action_stats.update(obs, actions)
            debug_dumper.maybe_dump(env, obs, actions)
            obs, _, dones, _ = env.step(actions)

        raw_done_ids = torch.nonzero(dones > 0, as_tuple=False).squeeze(-1)
        if raw_done_ids.numel() == 0:
            continue
        done_ids = raw_done_ids
        if initial_case_done_mask is not None:
            first_case_mask = torch.logical_not(initial_case_done_mask[done_ids])
            done_ids = done_ids[first_case_mask]
            if done_ids.numel() == 0:
                episode_ids[raw_done_ids] += 1
                continue

        if using_torchscript_policy:
            reset_policy_state = True
        elif policy_reset is not None:
            with torch.no_grad():
                policy_reset(dones > 0)

        remaining = args_cli.num_episodes - total_episodes
        if done_ids.numel() > remaining:
            done_ids = done_ids[:remaining]
        if initial_case_done_mask is not None:
            initial_case_done_mask[done_ids] = True

        latest_indices = (success_tracker.write_index[done_ids] - 1) % success_tracker.buffer_size
        latest_results = success_tracker.buffer[done_ids, latest_indices]
        successes = latest_results > 0.0
        episode_start_index = total_episodes

        total_episodes += int(done_ids.numel())
        total_successes += int(successes.sum().item())

        done_term_values = {}
        for term_name in termination_terms:
            term_values = termination_manager.get_term(term_name)[done_ids]
            done_term_values[term_name] = term_values.detach().cpu().tolist()
            termination_counts[term_name] += int(term_values.sum().item())

        if base_contact_context is not None and "base_contact" in done_term_values:
            base_contact_mask = termination_manager.get_term("base_contact")[done_ids] > 0
            base_contact_records = collect_base_contact_failure_records(
                env=env,
                done_ids=done_ids,
                base_contact_mask=base_contact_mask,
                base_contact_context=base_contact_context,
                pre_step_snapshot=pre_step_snapshot,
                terrain_lookup=terrain_lookup,
                episode_start_index=episode_start_index,
            )
            base_contact_failure_count += len(base_contact_records)
            for record in base_contact_records:
                body_name = str(record["colliding_body_name"])
                peak_contact_force = float(record["peak_contact_force"])
                root_lin_speed_b = float(record["root_lin_speed_b"])
                body_stats = base_contact_body_stats[body_name]
                body_stats["count"] += 1.0
                body_stats["sum_peak_contact_force"] += peak_contact_force
                body_stats["sum_root_lin_speed_b"] += root_lin_speed_b
                body_stats["max_peak_contact_force"] = max(body_stats["max_peak_contact_force"], peak_contact_force)

            remaining_record_budget = max(0, args_cli.max_base_contact_records - len(base_contact_failure_records))
            if remaining_record_budget > 0:
                base_contact_failure_records.extend(base_contact_records[:remaining_record_budget])
        else:
            base_contact_records = []

        if topdown_output_dir is not None:
            remaining_render_budget = max(0, args_cli.failure_topdown_limit - topdown_rendered_count)
            rendered_count, rendered_paths = render_failure_topdown_images(
                done_ids=done_ids,
                successes=successes,
                done_term_values=done_term_values,
                pre_step_snapshot=pre_step_snapshot,
                render_context=topdown_render_context,
                terrain_lookup=terrain_lookup,
                base_contact_records=base_contact_records,
                episode_start_index=episode_start_index,
                output_dir=topdown_output_dir,
                crop_size_m=args_cli.failure_topdown_crop_size_m,
                dpi=args_cli.failure_topdown_dpi,
                remaining_budget=remaining_render_budget,
            )
            topdown_rendered_count += rendered_count
            topdown_rendered_paths.extend(rendered_paths)

        if failure_rgb_output_dir is not None and episode_start_index >= args_cli.failure_rgb_min_episode:
            remaining_success_budget = max(0, failure_rgb_success_limit - failure_rgb_success_rendered_count)
            remaining_failure_budget = max(0, failure_rgb_failure_limit - failure_rgb_failure_rendered_count)
            rendered_success_count, rendered_failure_count, skipped_success_distance_count, skipped_success_height_delta_count, rendered_paths = capture_failure_rgb_screenshots(
                env=env,
                failure_camera=failure_rgb_camera,
                done_ids=done_ids,
                successes=successes,
                done_term_values=done_term_values,
                pre_step_snapshot=pre_step_snapshot,
                pre_step_delay_snapshot=pre_step_delay_snapshot,
                pre_step_snapshot_history=list(pre_step_snapshot_history),
                pre_step_observation_history=list(pre_step_observation_history),
                terrain_lookup=terrain_lookup,
                base_contact_records=base_contact_records,
                episode_start_index=episode_start_index,
                output_dir=failure_rgb_output_dir,
                success_remaining_budget=remaining_success_budget,
                failure_remaining_budget=remaining_failure_budget,
            )
            failure_rgb_success_rendered_count += rendered_success_count
            failure_rgb_failure_rendered_count += rendered_failure_count
            failure_rgb_success_distance_skipped_count += skipped_success_distance_count
            failure_rgb_success_height_delta_skipped_count += skipped_success_height_delta_count
            failure_rgb_rendered_paths.extend(rendered_paths)

        if observation_gif_output_dir is not None:
            remaining_observation_budget = max(0, args_cli.observation_gif_limit - observation_gif_rendered_count)
            rendered_count, rendered_paths = capture_failure_observation_gifs(
                env=env,
                done_ids=done_ids,
                done_term_values=done_term_values,
                pre_step_snapshot=pre_step_snapshot,
                pre_step_delay_snapshot=pre_step_delay_snapshot,
                pre_step_snapshot_history=list(pre_step_snapshot_history),
                pre_step_observation_history=list(pre_step_observation_history),
                render_context=topdown_render_context,
                terrain_lookup=terrain_lookup,
                base_contact_records=base_contact_records,
                episode_start_index=episode_start_index,
                output_dir=observation_gif_output_dir,
                remaining_budget=remaining_observation_budget,
            )
            observation_gif_rendered_count += rendered_count
            observation_gif_rendered_paths.extend(rendered_paths)

        if local_terrain_map is not None:
            local_terrain_map.reset(done_ids)

        terrain = getattr(env.unwrapped.scene, "terrain", None)
        if terrain is not None and hasattr(terrain, "terrain_types"):
            terrain_types = terrain.terrain_types[done_ids].detach().cpu().tolist()
            terrain_levels = terrain.terrain_levels[done_ids].detach().cpu().tolist() if hasattr(terrain, "terrain_levels") else None
            success_list = successes.detach().cpu().tolist()

            for index, terrain_type in enumerate(terrain_types):
                terrain_name = terrain_lookup.get(int(terrain_type), f"terrain_type_{int(terrain_type)}")
                terrain_stats[terrain_name]["episodes"] += 1
                terrain_stats[terrain_name]["successes"] += int(bool(success_list[index]))
                terrain_failure_stats[terrain_name]["episodes"] += 1
                terrain_failure_stats[terrain_name]["successes"] += int(bool(success_list[index]))
                for term_name in termination_terms:
                    terrain_failure_stats[terrain_name]["termination_counts"][term_name] += int(
                        bool(done_term_values[term_name][index])
                    )
                if terrain_levels is not None:
                    level_name = f"level_{int(terrain_levels[index])}"
                    terrain_level_stats[level_name]["episodes"] += 1
                    terrain_level_stats[level_name]["successes"] += int(bool(success_list[index]))
                    terrain_level_failure_stats[level_name]["episodes"] += 1
                    terrain_level_failure_stats[level_name]["successes"] += int(bool(success_list[index]))
                    for term_name in termination_terms:
                        terrain_level_failure_stats[level_name]["termination_counts"][term_name] += int(
                            bool(done_term_values[term_name][index])
                        )

        episode_ids[raw_done_ids] += 1

        if args_cli.print_every > 0 and total_episodes % args_cli.print_every == 0:
            print(
                f"[INFO] Progress: {total_episodes}/{args_cli.num_episodes} episodes, "
                f"success_rate={summarize_rates(total_successes, total_episodes):.4f}"
            )

    summary = {
        "task": args_cli.task,
        "checkpoint": resume_path,
        "debug_dump": {
            "enabled": bool(debug_dumper.enabled),
            "dir": str(debug_dumper.dump_dir) if debug_dumper.dump_dir is not None else None,
            "env_id": int(debug_dumper.env_id),
            "saved_steps": int(debug_dumper.saved_count),
            "stats_only": bool(not debug_dumper.save_arrays),
        },
        "num_envs": int(env.unwrapped.num_envs),
        "episodes": total_episodes,
        "successes": total_successes,
        "success_rate": summarize_rates(total_successes, total_episodes),
        "action_stats": action_stats.summarize(),
        "termination_rates": {
            term_name: {
                "count": count,
                "rate": summarize_rates(count, total_episodes),
            }
            for term_name, count in termination_counts.items()
        },
        "terrain_type_success": {
            name: {
                "episodes": values["episodes"],
                "successes": values["successes"],
                "success_rate": summarize_rates(values["successes"], values["episodes"]),
            }
            for name, values in sorted(terrain_stats.items())
        },
        "terrain_level_success": {
            name: {
                "episodes": values["episodes"],
                "successes": values["successes"],
                "success_rate": summarize_rates(values["successes"], values["episodes"]),
            }
            for name, values in sorted(terrain_level_stats.items())
        },
        "terrain_type_failure_breakdown": {
            name: {
                "episodes": int(values["episodes"]),
                "successes": int(values["successes"]),
                "failures": int(values["episodes"] - values["successes"]),
                "success_rate": summarize_rates(int(values["successes"]), int(values["episodes"])),
                "termination_rates": {
                    term_name: {
                        "count": int(count),
                        "rate": summarize_rates(int(count), int(values["episodes"])),
                    }
                    for term_name, count in values["termination_counts"].items()
                },
            }
            for name, values in sorted(terrain_failure_stats.items())
        },
        "terrain_level_failure_breakdown": {
            name: {
                "episodes": int(values["episodes"]),
                "successes": int(values["successes"]),
                "failures": int(values["episodes"] - values["successes"]),
                "success_rate": summarize_rates(int(values["successes"]), int(values["episodes"])),
                "termination_rates": {
                    term_name: {
                        "count": int(count),
                        "rate": summarize_rates(int(count), int(values["episodes"])),
                    }
                    for term_name, count in values["termination_counts"].items()
                },
            }
            for name, values in sorted(terrain_level_failure_stats.items())
        },
        "base_contact_diagnostics": {
            "enabled": base_contact_context is not None,
            "sensor_name": base_contact_context["sensor_name"] if base_contact_context is not None else None,
            "threshold": base_contact_context["threshold"] if base_contact_context is not None else None,
            "monitored_bodies": base_contact_context["body_names"] if base_contact_context is not None else [],
            "num_failures": int(base_contact_failure_count),
            "recorded_failures": int(len(base_contact_failure_records)),
            "record_limit": int(args_cli.max_base_contact_records),
            "collision_position_note": (
                "Approximate collision positions use the colliding body origin from the last pre-termination frame. "
                "The exact contact point is unavailable unless contact-point tracking is enabled in the contact sensor."
                if base_contact_context is not None
                else None
            ),
            "body_breakdown": {
                body_name: {
                    "count": int(values["count"]),
                    "rate_within_base_contact_failures": summarize_rates(int(values["count"]), int(base_contact_failure_count)),
                    "mean_peak_contact_force": float(values["sum_peak_contact_force"] / values["count"]),
                    "max_peak_contact_force": float(values["max_peak_contact_force"]),
                    "mean_root_lin_speed_b": float(values["sum_root_lin_speed_b"] / values["count"]),
                }
                for body_name, values in sorted(
                    base_contact_body_stats.items(), key=lambda item: (-item[1]["count"], item[0])
                )
                if values["count"] > 0
            },
            "failures": base_contact_failure_records,
        },
        "failure_topdown_renders": {
            "enabled": topdown_output_dir is not None and topdown_render_context is not None,
            "output_dir": topdown_output_dir,
            "render_limit": int(args_cli.failure_topdown_limit),
            "crop_size_m": float(args_cli.failure_topdown_crop_size_m),
            "rendered_count": int(topdown_rendered_count),
            "sample_paths": topdown_rendered_paths[:10],
        },
        "failure_rgb_renders": {
            "enabled": failure_rgb_output_dir is not None,
            "output_dir": failure_rgb_output_dir,
            "render_limit": int(args_cli.failure_rgb_limit),
            "success_render_limit": int(failure_rgb_success_limit),
            "failure_render_limit": int(failure_rgb_failure_limit),
            "success_min_initial_distance_m": float(args_cli.failure_rgb_success_min_initial_distance),
            "success_min_goal_height_delta_m": float(args_cli.failure_rgb_success_min_goal_height_delta),
            "terrain_name_filter": args_cli.failure_rgb_terrain_name,
            "success_distance_skipped_count": int(failure_rgb_success_distance_skipped_count),
            "success_height_delta_skipped_count": int(failure_rgb_success_height_delta_skipped_count),
            "base_contact_only": bool(args_cli.failure_rgb_base_contact_only),
            "include_successes": bool(args_cli.failure_rgb_include_successes),
            "include_observations": bool(args_cli.failure_rgb_include_observations),
            "width": int(args_cli.failure_rgb_width),
            "height": int(args_cli.failure_rgb_height),
            "camera_height_m": float(args_cli.failure_rgb_height_m),
            "focal_length": float(args_cli.failure_rgb_focal_length),
            "center_mode": str(args_cli.failure_rgb_center_mode),
            "follow_robot": bool(args_cli.failure_rgb_follow_robot),
            "export_mode": str(args_cli.failure_rgb_export_mode),
            "history_frames": int(args_cli.failure_rgb_history_frames),
            "full_episode": bool(args_cli.failure_rgb_full_episode),
            "gif_frame_ms": int(args_cli.failure_rgb_gif_frame_ms),
            "include_rear_view": bool(args_cli.failure_rgb_include_rear_view),
            "rear_view_height_m": float(args_cli.failure_rgb_rear_height_m),
            "rear_view_distance_m": float(args_cli.failure_rgb_rear_distance_m),
            "rear_view_lateral_m": float(args_cli.failure_rgb_rear_lateral_m),
            "rear_view_target_height_m": float(args_cli.failure_rgb_rear_lookahead_m),
            "rear_view_orientation": "gravity_aligned_look_at",
            "success_rendered_count": int(failure_rgb_success_rendered_count),
            "failure_rendered_count": int(failure_rgb_failure_rendered_count),
            "rendered_count": int(failure_rgb_success_rendered_count + failure_rgb_failure_rendered_count),
            "sample_paths": failure_rgb_rendered_paths[:10],
        },
        "base_contact_observation_gifs": {
            "enabled": observation_gif_output_dir is not None,
            "output_dir": observation_gif_output_dir,
            "render_limit": int(args_cli.observation_gif_limit),
            "history_frames": int(args_cli.observation_gif_frames),
            "full_episode": bool(args_cli.failure_rgb_full_episode),
            "gif_frame_ms": int(args_cli.observation_gif_frame_ms),
            "content": "combined topdown + depth + height + visibility + scene token",
            "rendered_count": int(observation_gif_rendered_count),
            "sample_paths": observation_gif_rendered_paths[:10],
        },
        "eval_overrides_enabled": not args_cli.disable_eval_overrides,
    }

    print("\n[RESULT] Evaluation summary")
    print(json.dumps(summary, indent=2, sort_keys=True))

    if args_cli.json_out:
        output_path = os.path.abspath(os.path.expanduser(args_cli.json_out))
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as output_file:
            json.dump(summary, output_file, indent=2, sort_keys=True)
            output_file.write("\n")
        print(f"[INFO] Wrote JSON summary to: {output_path}")

    if args_cli.compare_checkpoint is not None:
        if compare_case_manifest is None:
            compare_case_manifest = _capture_initial_case_manifest(env, args_cli.num_episodes)
        print("[INFO] Resetting the same environment scene for second checkpoint comparison.")
        env.reset()
        _apply_initial_case_manifest_data(env, compare_case_manifest, source="in-memory compare manifest")
        compare_policy, compare_policy_reset, compare_using_ts, compare_checkpoint_path = _load_compare_inference_policy_for_env(
            env, agent_cfg, args_cli.compare_checkpoint
        )
        compare_output_dir = args_cli.compare_failure_rgb_dir
        if compare_output_dir is None and args_cli.failure_rgb_dir is not None:
            compare_output_dir = os.path.join(os.path.dirname(args_cli.failure_rgb_dir), "compare_second")
        _run_eval_pass(
            env=env,
            policy=compare_policy,
            policy_reset=compare_policy_reset,
            using_torchscript_policy=compare_using_ts,
            checkpoint_path=compare_checkpoint_path,
            failure_rgb_output_dir=compare_output_dir,
            json_out=args_cli.compare_json_out,
            label="compare",
        )

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

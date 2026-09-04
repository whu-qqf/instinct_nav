# Copyright (c) 2022-2025, Fan Yang and Per Frivik, ETH Zurich.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Reward functions for navigation tasks.

These functions can be passed to :class:`isaaclab.managers.RewardTermCfg`
to specify the reward function and its parameters.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

from instinct_nav.navigation.mdp.navigation.goal_commands import RobotNavigationGoalCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def action_rate_l1(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Penalize the rate of change of the actions using L1 kernel."""
    return torch.sum(torch.abs(env.action_manager.action - env.action_manager.prev_action), dim=1)


def lateral_movement(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward the agent for moving laterally using L1-Kernel.

    Args:
        env: The learning environment.
        asset_cfg: The name of the robot asset.

    Returns:
        Dense reward [0, +1] based on the lateral velocity.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    # compute the reward
    lateral_velocity = asset.data.root_lin_vel_b[:, 1]
    reward = torch.abs(lateral_velocity)
    return reward


def rot_movement(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward the agent for rotating around the z-axis using an L2-Kernel.

    Args:
        env: The learning environment.
        asset_cfg: The name of the robot asset.

    Returns:
        Dense reward [0, +1] based on the rotational velocity.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    rot_vel_norm = torch.norm(asset.data.root_ang_vel_b, dim=1)
    return rot_vel_norm


def reach_goal_xyz(
    env: ManagerBasedRLEnv,
    command_name: str,
    sigmoid: float,
    T_r: float,
    probability: float,
    flat: bool,
    ratio: bool,
) -> torch.Tensor:
    """Reward goal reaching with configurable sigmoid shaping.

    Args:
        env: The learning environment.
        command_name: Name of the goal command.
        sigmoid: Sigmoid parameter for shaping.
        T_r: Time reward scaling factor.
        probability: Probability of random sampling.
        flat: Whether to only consider xy error (ignore z).
        ratio: Whether to scale by travel distance ratio.

    Returns:
        Dense reward based on distance to goal.
    """
    goal_cmd_generator: RobotNavigationGoalCommand = env.command_manager._terms[command_name]

    t = env.episode_length_buf
    T = env.max_episode_length

    if flat:
        xyz_error = torch.norm(goal_cmd_generator._get_unscaled_command()[:, :2], dim=1)
    else:
        xyz_error = torch.norm(goal_cmd_generator._get_unscaled_command(), dim=1)

    reward = 1 / (1 + torch.square(xyz_error / sigmoid)) / T_r

    timeup_mask = t > (T - goal_cmd_generator.required_time_at_goal_in_steps)
    random_mask = torch.rand_like(t.float()) < probability
    timeup_mask = torch.logical_or(timeup_mask, random_mask)

    arrive_mask = goal_cmd_generator.time_at_goal > 0.0
    reward_mask = torch.logical_or(timeup_mask, arrive_mask)

    if ratio:
        # Calculate the travel distance ratio relative to the initial goal distance
        travel_distance = torch.max(
            goal_cmd_generator.distance_traveled, goal_cmd_generator.initial_distance_to_goal
        )
        travel_distance_ratio = goal_cmd_generator.initial_distance_to_goal / (travel_distance + 1e-6)
    else:
        travel_distance_ratio = torch.ones_like(reward)

    reward = reward * reward_mask.float() * travel_distance_ratio

    return reward


def backward_movement_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Small penalty for backward movement as a regularization term.

    Args:
        env: The learning environment.
        asset_cfg: The name of the robot asset.

    Returns:
        Penalty [0, +1] based on backward velocity (to be used with negative weight).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    # compute the penalty
    forward_velocity = asset.data.root_lin_vel_b[:, 0]
    # Only penalize negative forward velocity (backward movement)
    backward_velocity = torch.clamp(-forward_velocity, min=0.0, max=1.0)
    return backward_velocity


def stair_centerline_alignment(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward the robot for approaching/ascending stairs along their center-line.

    Uses privileged geometry baked at terrain generation: the per-cell centerline
    field ``env.scene.terrain._height_field_stair_centerline`` (shape
    ``(num_terrains, W, H)``, values in ``[0, 1]``). The reward is the field value
    at the robot's current cell — high on the lateral mid-line of a staircase,
    decaying to its exposed lateral edges. The field is 0 everywhere off stairs, so
    flat ground and platforms yield exactly 0 (no false-positive shaping). Purely
    positive; use with a positive weight.

    Args:
        env: The learning environment.
        command_name: Name of the navigation goal command (reused for the terrain
            index/coordinate convention).
        asset_cfg: The robot asset.

    Returns:
        Dense reward in ``[0, 1]``.
    """
    terrain = env.scene.terrain
    reward = torch.zeros(env.num_envs, device=env.device)

    # Cache the field on GPU once (raw storage from patches lives on CPU).
    field = getattr(terrain, "_height_field_stair_centerline_gpu", None)
    if field is None:
        raw = getattr(terrain, "_height_field_stair_centerline", None)
        if raw is None:
            return reward
        field = raw.to(env.device)
        terrain._height_field_stair_centerline_gpu = field

    num_terrains, W, H = field.shape

    cmd: RobotNavigationGoalCommand = env.command_manager._terms[command_name]
    cmd._initialize_position_sampling()
    sampler = cmd._position_sampler

    asset: Articulation = env.scene[asset_cfg.name]
    env_ids = torch.arange(env.num_envs, device=env.device)

    # Per-env terrain index into the (num_terrains, W, H) field, and world origin.
    terrain_idx = cmd._get_terrain_indices(env_ids).clamp(0, num_terrains - 1)
    levels = terrain.terrain_levels[env_ids]
    types = terrain.terrain_types[env_ids]
    origin_xy = terrain.terrain_origins[levels, types][:, :2]

    # World -> cell, inverting the PositionSampler convention (goal_commands.py:283-284):
    #   local = (idx + border_pixels) * cell_size - mesh_center
    local_xy = asset.data.root_pos_w[:, :2] - origin_xy
    i = torch.round((local_xy[:, 0] + sampler.mesh_center) / sampler.cell_size - sampler.border_pixels).long()
    j = torch.round((local_xy[:, 1] + sampler.mesh_center) / sampler.cell_size - sampler.border_pixels).long()
    in_bounds = (i >= 0) & (i < W) & (j >= 0) & (j < H)
    i = i.clamp(0, W - 1)
    j = j.clamp(0, H - 1)

    values = field[terrain_idx, i, j]
    return torch.where(in_bounds, values, reward)


def velocity_into_unobserved_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    speed_threshold: float = 0.1,
    max_range_m: float = 1.5,
) -> torch.Tensor:
    """Penalize moving toward low-score ("black") cells of the vis_score field.

    Treats the robot-centric visibility/score map as a scalar field ``s(x, y)``
    (values in ``[0, 1]``, ``1`` == observed) and the robot's *actual* velocity as
    a vector ``v``. The penalty is the blackness integrated along the ray ``v`` —
    a discrete line integral ``(1/L) int_0^L (1 - s(robot + v_hat * t)) dt``, i.e.
    "how much of the path ahead, in the direction of motion, is unobserved". No
    hemisphere/cone restriction: ``v`` may point in any direction, so moving
    backward/sideways into unobserved regions is penalized too, while moving
    forward through open (observed) terrain yields ~0.

    The map lives in the robot yaw frame (local ``+x`` forward, ``+y`` left, robot
    at the center cell), matching ``cell_x_local``/``cell_y_local`` on the manager.

    The map only encodes *observed vs. not-observed* — a low score means the cell
    is occluded, outside the forward FOV, or beyond range (NOT that it is an
    obstacle). Gated by a minimum speed (a stationary robot is never penalized)
    and by the map having been populated. Purely positive; use with a negative
    weight. Returns zeros when the visibility manager is absent (non-ScoreVis
    tasks) so it is safe to leave in the base config at weight 0.

    Args:
        env: The learning environment.
        asset_cfg: The robot asset.
        speed_threshold: Minimum planar speed (m/s) below which no penalty is
            applied.
        max_range_m: Distance (m) along the velocity direction over which the score
            field is integrated (a discrete line integral, one sample per grid cell).

    Returns:
        Dense penalty in ``[0, 1]`` (0 when not moving / no map).
    """
    reward = torch.zeros(env.num_envs, device=env.device)

    mgr = getattr(env, "local_visibility_map_manager", None)
    if mgr is None:
        return reward

    grid = mgr.map  # [N, 1, G, G], 1 == observed
    G = mgr.grid_size
    half = mgr.half_size
    cell = mgr.cell_size

    asset: Articulation = env.scene[asset_cfg.name]
    # The vis map is stored in a YAW-ONLY robot frame, so the velocity must be
    # expressed in that same frame. root_lin_vel_b uses the full body frame
    # (includes pitch/roll), which rotates v_hat off the map axes when the
    # humanoid pitches/rolls while walking. Rotate world velocity by yaw only.
    yaw_q = yaw_quat(asset.data.root_quat_w)
    vel_xy = quat_apply_inverse(yaw_q, asset.data.root_lin_vel_w)[:, :2]
    speed = torch.norm(vel_xy, dim=1)
    v_hat = vel_xy / speed.clamp_min(1e-6).unsqueeze(1)  # [N, 2] yaw frame (forward, left)

    env_ids = torch.arange(env.num_envs, device=env.device)
    # Dense ray-march along the velocity direction: one sample per grid cell over
    # [cell, max_range_m]. This is the discrete line integral of blackness along v.
    # Sample points start at `cell` (0.1 m), NOT the robot center cell, so the
    # robot's own map state does not directly drive the penalty.
    n_steps = max(1, int(round(max_range_m / cell)))
    dists = (torch.arange(1, n_steps + 1, device=env.device, dtype=torch.float32)) * cell  # [n_steps]

    px = v_hat[:, 0:1] * dists.unsqueeze(0)  # [N, n_steps] forward-local offsets
    py = v_hat[:, 1:2] * dists.unsqueeze(0)  # [N, n_steps] left-local offsets
    row = torch.round((half - px) / cell - 0.5).long()
    col = torch.round((half - py) / cell - 0.5).long()

    # Out-of-bounds samples (robot near / leaving the map) are excluded from the
    # average instead of being clamped to the edge, which would double-count edge
    # blackness when the robot drives toward the map border.
    in_bounds = (row >= 0) & (row < G) & (col >= 0) & (col < G)
    row_c = row.clamp(0, G - 1)
    col_c = col.clamp(0, G - 1)
    score = grid[env_ids.unsqueeze(1), 0, row_c, col_c]  # [N, n_steps], 1 == observed
    blackness = 1.0 - score

    denom = in_bounds.float().sum(dim=1).clamp_min(1.0)
    penalty = (blackness * in_bounds.float()).sum(dim=1) / denom

    gate = speed > speed_threshold
    pose_valid = getattr(mgr, "pose_valid", None)
    if pose_valid is not None:
        gate = gate & pose_valid
    return torch.where(gate, penalty, reward)

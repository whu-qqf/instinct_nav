"""Dynamic obstacle curriculum support for navigation tasks.

Stage-3 obstacle-avoidance training adds per-env scripted rigid obstacles that:

- are ray-castable by the shared depth camera (tracked mesh transforms),
- move through path interactions (cross, head-on, follow, merge, stop-and-go),
- reset along the robot-to-goal corridor with clearance constraints,
- drive an obstacle-proximity reward and an obstacle-collision termination.

The low-level parkour policy stays frozen; obstacles only add scene dynamics.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject

from instinct_nav.terrains.terrain_constants import VERTICAL_SCALE

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _build_terrain_lookup(env: "ManagerBasedRLEnv") -> dict:
    """Cache terrain height-field geometry used to place obstacles on the ground."""
    terrain = env.scene.terrain
    heights_raw = getattr(terrain, "_height_field_visual", None)
    if heights_raw is None:
        raise RuntimeError("Dynamic obstacles require a height-field terrain with _height_field_visual.")
    terrain_cfg = terrain.cfg.terrain_generator
    return {
        "heights": heights_raw.to(env.device),
        "num_rows": int(terrain_cfg.num_rows),
        "num_cols": int(terrain_cfg.num_cols),
        "curriculum": bool(terrain_cfg.curriculum),
        "cell": float(terrain_cfg.horizontal_scale),
        "border_pixels": int(float(terrain_cfg.border_width) / float(terrain_cfg.horizontal_scale)) + 1,
        "mesh_center": float(terrain_cfg.size[0]) / 2.0,
    }


def _terrain_indices(env: "ManagerBasedRLEnv", env_ids: torch.Tensor, lookup: dict) -> torch.Tensor:
    """Flat per-tile terrain index, matching RobotNavigationGoalCommand._get_terrain_indices."""
    terrain = env.scene.terrain
    levels = terrain.terrain_levels[env_ids]
    types = terrain.terrain_types[env_ids]
    if lookup["curriculum"]:
        return levels + types * lookup["num_rows"]
    return levels * lookup["num_cols"] + types


def _terrain_z_at(
    env: "ManagerBasedRLEnv", lookup: dict, positions_w: torch.Tensor, env_ids: torch.Tensor
) -> torch.Tensor:
    """Height-field z (meters) under world XY positions for the given env ids.

    ``positions_w``: [N, 3] world positions; ``env_ids``: [N] matching env indices.
    Returns [N] terrain heights.
    """
    terrain = env.scene.terrain
    levels = terrain.terrain_levels[env_ids]
    types = terrain.terrain_types[env_ids]
    origins = terrain.terrain_origins[levels, types][:, :2]  # [N, 2]
    local = positions_w[:, :2] - origins
    i = torch.floor((local[:, 0] + lookup["mesh_center"]) / lookup["cell"] - lookup["border_pixels"]).long()
    j = torch.floor((local[:, 1] + lookup["mesh_center"]) / lookup["cell"] - lookup["border_pixels"]).long()
    heights = lookup["heights"]
    i = i.clamp(0, heights.shape[1] - 1)
    j = j.clamp(0, heights.shape[2] - 1)
    tile = _terrain_indices(env, env_ids, lookup)
    return heights[tile, i, j] * VERTICAL_SCALE


class DynamicObstacleManager:
    """Scripted per-env rigid obstacles with reset and per-step motion."""

    def __init__(
        self,
        env: "ManagerBasedRLEnv",
        obstacle_names: list[str],
        speed_range: tuple[float, float] = (0.3, 0.6),
        pattern: str = "patrol",
        half_extent_m: float = 2.0,
        min_start_dist: float = 1.5,
        max_start_dist: float = 3.2,
        goal_clearance_m: float = 1.0,
        obstacle_half_height: float = 0.4,
    ):
        self.env = env
        self.num_envs = int(env.num_envs)
        self.device = env.device
        self.obstacle_names = list(obstacle_names)
        self.num_obstacles = len(self.obstacle_names)
        self.obstacles: list[RigidObject] = [env.scene[name] for name in self.obstacle_names]
        if self.num_obstacles == 0:
            raise ValueError("DynamicObstacleManager requires at least one obstacle name.")

        self.speed_range = (float(speed_range[0]), float(speed_range[1]))
        self.pattern = pattern
        self.half_extent_m = float(half_extent_m)
        self.min_start_dist = float(min_start_dist)
        self.max_start_dist = float(max_start_dist)
        self.goal_clearance_m = float(goal_clearance_m)
        self.obstacle_half_height = float(obstacle_half_height)
        self.terrain_lookup = _build_terrain_lookup(env)

        # In-run curriculum state: obstacles stay parked until the fixed PPO
        # iteration threshold. The switch is deliberately independent of success.
        self.enabled = False
        self.park_offset_m = 40.0
        self.auto_switch_success = 0.9
        self.max_switch_iteration = 3003
        self.env_steps_per_iteration = 16
        self.switch_check_interval = 100
        self._last_switch_check = -1

        # [num_envs, K, 3] world position; [num_envs, K, 2] direction/center; [num_envs, K] speed/timer
        self.positions_w = torch.zeros(self.num_envs, self.num_obstacles, 3, device=self.device)
        self.dirs = torch.zeros(self.num_envs, self.num_obstacles, 2, device=self.device)
        self.speeds = torch.zeros(self.num_envs, self.num_obstacles, device=self.device)
        self.centers = torch.zeros(self.num_envs, self.num_obstacles, 2, device=self.device)
        self.timers = torch.zeros(self.num_envs, self.num_obstacles, device=self.device)
        self.starts = torch.zeros(self.num_envs, self.num_obstacles, 2, device=self.device)
        self.ends = torch.zeros(self.num_envs, self.num_obstacles, 2, device=self.device)
        self.behaviors = torch.zeros(self.num_envs, self.num_obstacles, device=self.device, dtype=torch.long)
        self.paused = torch.zeros(self.num_envs, self.num_obstacles, device=self.device, dtype=torch.bool)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _sample_trajectories(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample interaction trajectories around the robot-to-goal corridor.

        Behaviors: 0 cross, 1 head-on, 2 same-direction, 3 diagonal merge,
        4 stop-and-go. Rejection sampling enforces robot/goal clearance and
        obstacle separation; unresolved candidates fall back to spaced corridor
        crossings instead of accepting overlaps.
        """
        n = env_ids.numel()
        robot_xy = self.env.scene["robot"].data.root_pos_w[env_ids, :2]
        goal_cmd = getattr(getattr(self.env, "command_manager", None), "_terms", {}).get("robot_goal", None)
        if goal_cmd is not None and hasattr(goal_cmd, "goal_position_world"):
            goal_xy = goal_cmd.goal_position_world[env_ids, :2]
        else:
            goal_xy = robot_xy + torch.tensor([4.0, 0.0], device=self.device)
        corridor = goal_xy - robot_xy
        corridor_len = torch.norm(corridor, dim=-1, keepdim=True).clamp_min(2.0)
        forward = corridor / corridor_len
        lateral = torch.stack([-forward[:, 1], forward[:, 0]], dim=-1)

        starts = torch.zeros(n, self.num_obstacles, 2, device=self.device)
        ends = torch.zeros_like(starts)
        behaviors = torch.multinomial(
            torch.tensor([0.30, 0.25, 0.15, 0.20, 0.10], device=self.device),
            n * self.num_obstacles, replacement=True,
        ).view(n, self.num_obstacles)
        for k in range(self.num_obstacles):
            accepted = torch.zeros(n, device=self.device, dtype=torch.bool)
            for attempt in range(12):
                t = torch.rand(n, 1, device=self.device) * 0.50 + 0.25
                anchor = robot_xy + corridor * t
                side = torch.where(torch.rand(n, 1, device=self.device) < 0.5, -1.0, 1.0)
                span = torch.rand(n, 1, device=self.device) * 1.0 + 1.2
                mode = behaviors[:, k]
                cross_start = anchor + lateral * side * span
                cross_end = anchor - lateral * side * span
                head_start = anchor + forward * (torch.rand(n, 1, device=self.device) * 0.8 + 0.8)
                head_end = robot_xy + forward * 0.8
                same_start = robot_xy + forward * 1.5 + lateral * side * 0.6
                same_end = goal_xy - forward * self.goal_clearance_m + lateral * side * 0.6
                merge_start = anchor + lateral * side * span + forward * 0.8
                merge_end = anchor - lateral * side * 0.3 - forward * 0.8
                candidate_start = torch.where((mode == 1).unsqueeze(-1), head_start, cross_start)
                candidate_end = torch.where((mode == 1).unsqueeze(-1), head_end, cross_end)
                candidate_start = torch.where((mode == 2).unsqueeze(-1), same_start, candidate_start)
                candidate_end = torch.where((mode == 2).unsqueeze(-1), same_end, candidate_end)
                candidate_start = torch.where((mode == 3).unsqueeze(-1), merge_start, candidate_start)
                candidate_end = torch.where((mode == 3).unsqueeze(-1), merge_end, candidate_end)
                clear = (torch.norm(candidate_start - robot_xy, dim=-1) >= self.min_start_dist)
                clear &= (torch.norm(candidate_start - goal_xy, dim=-1) >= self.goal_clearance_m)
                clear &= (torch.norm(candidate_end - goal_xy, dim=-1) >= self.goal_clearance_m)
                if k > 0:
                    clear &= (torch.norm(candidate_start.unsqueeze(1) - starts[:, :k], dim=-1).min(dim=1).values >= 0.9)
                take = clear & ~accepted
                starts[:, k] = torch.where(take.unsqueeze(-1), candidate_start, starts[:, k])
                ends[:, k] = torch.where(take.unsqueeze(-1), candidate_end, ends[:, k])
                accepted |= clear
                if bool(accepted.all().item()):
                    break
            fallback_anchor = robot_xy + forward * (2.0 + 0.6 * k)
            fallback_start = fallback_anchor + lateral * (1.4 + 0.35 * k)
            fallback_end = fallback_anchor - lateral * (1.4 + 0.35 * k)
            starts[:, k] = torch.where(accepted.unsqueeze(-1), starts[:, k], fallback_start)
            ends[:, k] = torch.where(accepted.unsqueeze(-1), ends[:, k], fallback_end)
        return starts, ends, behaviors

    def _set_pose_and_velocity(self, env_ids: torch.Tensor, positions: torch.Tensor):
        """Write root poses (and scripted velocities) to the simulator for env_ids."""
        quat = torch.zeros(*positions.shape[:-1], 4, device=self.device)
        quat[..., 0] = 1.0
        pose = torch.cat([positions, quat], dim=-1)  # [n, K, 7]
        for k, obstacle in enumerate(self.obstacles):
            obstacle.write_root_pose_to_sim(pose[:, k], env_ids=env_ids)

        active_speed = self.speeds[env_ids] * (~self.paused[env_ids]).to(self.speeds.dtype)
        lin_vel = torch.cat([self.dirs[env_ids] * active_speed.unsqueeze(-1), torch.zeros_like(positions[:, :, :1])], dim=-1)
        ang_vel = torch.zeros_like(lin_vel)
        velocity = torch.cat([lin_vel, ang_vel], dim=-1)  # [n, K, 6]
        for k, obstacle in enumerate(self.obstacles):
            obstacle.write_root_velocity_to_sim(velocity[:, k], env_ids=env_ids)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def reset(self, env_ids: Sequence[int] | torch.Tensor | slice | None = None):
        if env_ids is None or isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        else:
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        n = env_ids.numel()

        if not self.enabled:
            # Static phase: park obstacles far from the robot so depth/visibility
            # never sees them and proximity/collision terms stay ~zero.
            center = (
                self.env.scene.env_origins[env_ids][:, :2].unsqueeze(1)
                + torch.tensor([[-self.park_offset_m, 0.0]], device=self.device)
            ).expand(n, self.num_obstacles, 2)
            speeds = torch.zeros(n, self.num_obstacles, device=self.device)
            dirs = torch.zeros(n, self.num_obstacles, 2, device=self.device)
        else:
            starts, ends, behaviors = self._sample_trajectories(env_ids)
            center = (starts + ends) * 0.5
            speeds = (
                torch.rand(n, self.num_obstacles, device=self.device) * (self.speed_range[1] - self.speed_range[0])
                + self.speed_range[0]
            )
            dirs = torch.nn.functional.normalize(ends - starts, dim=-1)
            self.starts[env_ids] = starts
            self.ends[env_ids] = ends
            self.behaviors[env_ids] = behaviors
            self.paused[env_ids] = False

        positions = torch.zeros(n, self.num_obstacles, 3, device=self.device)
        positions[:, :, :2] = self.starts[env_ids] if self.enabled else center
        if self.enabled:
            flat_pos = positions.reshape(-1, 3)
            flat_ids = env_ids.repeat_interleave(self.num_obstacles)
            z = _terrain_z_at(self.env, self.terrain_lookup, flat_pos, flat_ids).reshape(n, self.num_obstacles)
            positions[:, :, 2] = z + self.obstacle_half_height
        else:
            # Parked far outside the terrain tile; gravity is disabled so the
            # obstacles float in place and can never interact with the robot.
            positions[:, :, 2] = 0.5

        self.centers[env_ids] = center
        self.speeds[env_ids] = speeds
        self.dirs[env_ids] = dirs
        self.timers[env_ids] = torch.rand(n, self.num_obstacles, device=self.device) * 3.0 + 1.0
        self.positions_w[env_ids] = positions
        self._set_pose_and_velocity(env_ids, positions)

    def update(self, dt: float):
        """Advance scripted motion for every env and write poses to the simulator."""
        dt = float(dt)
        self._log_metrics()
        if not self.enabled:
            # Static-phase obstacles stay parked until the fixed iteration threshold.
            self._maybe_switch_curriculum()
            return
        # Stop-and-go trajectories pause for 0.5--1.5 s, then resume.
        stop_go = self.behaviors == 4
        self.timers -= dt
        expire = stop_go & (self.timers <= 0.0)
        if bool(expire.any().item()):
            self.paused = torch.where(expire, ~self.paused, self.paused)
            moving_time = torch.rand_like(self.timers) * 2.0 + 1.5
            pause_time = torch.rand_like(self.timers) + 0.5
            self.timers = torch.where(expire, torch.where(self.paused, pause_time, moving_time), self.timers)

        # Integrate motion.
        active_speed = self.speeds * (~self.paused).to(self.speeds.dtype)
        self.positions_w[:, :, :2] += self.dirs * active_speed.unsqueeze(-1) * dt
        target = torch.where(
            (torch.sum(self.dirs * (self.ends - self.starts), dim=-1) >= 0.0).unsqueeze(-1),
            self.ends, self.starts,
        )
        reached = torch.sum((target - self.positions_w[:, :, :2]) * self.dirs, dim=-1) <= 0.0
        self.positions_w[:, :, :2] = torch.where(reached.unsqueeze(-1), target, self.positions_w[:, :, :2])
        self.dirs = torch.where(reached.unsqueeze(-1), -self.dirs, self.dirs)

        # Refresh z so obstacles track terrain height as they move.
        all_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        flat_pos = self.positions_w.reshape(-1, 3)
        flat_ids = all_ids.repeat_interleave(self.num_obstacles)
        z = _terrain_z_at(self.env, self.terrain_lookup, flat_pos, flat_ids).reshape(
            self.num_envs, self.num_obstacles
        )
        self.positions_w[:, :, 2] = z + self.obstacle_half_height

        self._set_pose_and_velocity(all_ids, self.positions_w)

    def enable(self, env_ids: Sequence[int] | torch.Tensor | slice | None = None):
        """Switch the curriculum on: place obstacles into the action ring and start motion."""
        if self.enabled:
            return
        self.enabled = True
        self.reset(env_ids)
        print(f"[DynamicObstacleManager] Dynamic obstacles ENABLED (step {self.env.common_step_counter}).")
        self._log_metrics()

    def _log_metrics(self):
        """Expose dynamic-obstacle state to TensorBoard/WandB via env.extras log."""
        extras = getattr(self.env, "extras", None)
        if extras is None:
            return
        log = extras.setdefault("log", {})
        log["Metrics/dynamic_obstacles/enabled"] = float(self.enabled)
        if self.enabled:
            log["Metrics/dynamic_obstacles/min_dist_mean"] = float(
                self.min_dist_to_robot().mean().item()
            )
        else:
            log["Metrics/dynamic_obstacles/min_dist_mean"] = float(
                self.park_offset_m
            )

    def _mean_success_rate(self) -> float:
        """Windowed mean success rate from the robot-goal command tracker."""
        goal_cmd = getattr(getattr(self.env, "command_manager", None), "_terms", {}).get("robot_goal", None)
        if goal_cmd is None or not hasattr(goal_cmd, "get_success_rate"):
            return 0.0
        rate = goal_cmd.get_success_rate()
        if rate.numel() == 0:
            return 0.0
        return float(rate.mean().item())

    def _maybe_switch_curriculum(self):
        """One-way switch from the static phase to the dynamic phase.

        Triggered only at the configured fixed PPO iteration threshold.
        """
        step = int(self.env.common_step_counter)
        if step < 2:
            return
        steps_per_iter = max(1, int(self.env_steps_per_iteration))
        cap_env_steps = int(self.max_switch_iteration) * steps_per_iter
        if step >= cap_env_steps:
            approx_iteration = step // steps_per_iter
            print(
                f"[DynamicObstacleManager] Switching curriculum at env step {step} "
                f"(~iteration {approx_iteration}) (fixed iteration {self.max_switch_iteration})."
            )
            self.enable()

    def min_dist_to_robot(self) -> torch.Tensor:
        """Minimum XY distance from the robot root to any obstacle: [num_envs]."""
        robot = self.env.scene["robot"]
        root_xy = robot.data.root_pos_w[:, :2].unsqueeze(1)  # [num_envs, 1, 2]
        dist = torch.norm(self.positions_w[:, :, :2] - root_xy, dim=-1)  # [num_envs, K]
        return dist.min(dim=-1).values

    def closing_speed_to_robot(self) -> torch.Tensor:
        """Maximum approach speed of any obstacle toward the robot: [num_envs].

        Uses the scripted obstacle velocity (world XY) and the robot root velocity;
        positive values mean the obstacle is closing in on the robot.
        """
        robot = self.env.scene["robot"]
        root_xy = robot.data.root_pos_w[:, :2].unsqueeze(1)  # [num_envs, 1, 2]
        to_obs = self.positions_w[:, :, :2] - root_xy  # [num_envs, K, 2]
        dist = torch.norm(to_obs, dim=-1).clamp_min(1.0e-3)  # [num_envs, K]
        unit = to_obs / dist.unsqueeze(-1)

        obs_vel = self.dirs * self.speeds.unsqueeze(-1)  # [num_envs, K, 2] world XY
        robot_vel = robot.data.root_lin_vel_w[:, :2].unsqueeze(1)  # [num_envs, 1, 2]
        # closing speed = -d(dist)/dt = dot(robot_vel - obs_vel, unit); positive
        # when the obstacle is approaching (distance shrinking).
        closing = torch.sum((robot_vel - obs_vel) * unit, dim=-1)  # [num_envs, K]
        return closing.clamp_min(0.0).max(dim=-1).values


def _get_or_create_dynamic_obstacle_manager(
    env: "ManagerBasedRLEnv",
    obstacle_names: list[str],
    speed_range: tuple[float, float] = (0.3, 0.6),
    pattern: str = "patrol",
    half_extent_m: float = 2.0,
    min_start_dist: float = 1.5,
    max_start_dist: float = 3.2,
    goal_clearance_m: float = 1.0,
    obstacle_half_height: float = 0.4,
    auto_switch_success: float = 0.9,
    max_switch_iteration: int = 3003,
    env_steps_per_iteration: int = 16,
    switch_check_interval: int = 100,
) -> DynamicObstacleManager:
    manager = getattr(env, "_dynamic_obstacle_manager", None)
    if manager is None:
        manager = DynamicObstacleManager(
            env,
            obstacle_names=list(obstacle_names),
            speed_range=speed_range,
            pattern=pattern,
            half_extent_m=half_extent_m,
            min_start_dist=min_start_dist,
            max_start_dist=max_start_dist,
            goal_clearance_m=goal_clearance_m,
            obstacle_half_height=obstacle_half_height,
        )
        manager.auto_switch_success = float(auto_switch_success)
        manager.max_switch_iteration = int(max_switch_iteration)
        manager.env_steps_per_iteration = int(env_steps_per_iteration)
        manager.switch_check_interval = int(switch_check_interval)
        env._dynamic_obstacle_manager = manager
    return manager


# ----------------------------------------------------------------------
# Manager event functions (wired from environment configs)
# ----------------------------------------------------------------------


def reset_dynamic_obstacles(
    env: "ManagerBasedRLEnv",
    env_ids,
    obstacle_names: list[str] | None = None,
    speed_range: tuple[float, float] = (0.3, 0.6),
    pattern: str = "patrol",
    half_extent_m: float = 2.0,
    min_start_dist: float = 1.5,
    max_start_dist: float = 3.2,
    goal_clearance_m: float = 1.0,
    obstacle_half_height: float = 0.4,
    auto_switch_success: float = 0.9,
    max_switch_iteration: int = 3003,
    env_steps_per_iteration: int = 16,
    switch_check_interval: int = 100,
):
    """Reset event: place obstacles away from spawn/goal and randomize motion."""
    if not obstacle_names:
        return
    manager = _get_or_create_dynamic_obstacle_manager(
        env,
        obstacle_names=obstacle_names,
        speed_range=speed_range,
        pattern=pattern,
        half_extent_m=half_extent_m,
        min_start_dist=min_start_dist,
        max_start_dist=max_start_dist,
        goal_clearance_m=goal_clearance_m,
        obstacle_half_height=obstacle_half_height,
        auto_switch_success=auto_switch_success,
        max_switch_iteration=max_switch_iteration,
        env_steps_per_iteration=env_steps_per_iteration,
        switch_check_interval=switch_check_interval,
    )
    manager.reset(env_ids)


def update_dynamic_obstacles(
    env: "ManagerBasedRLEnv",
    env_ids,
    obstacle_names: list[str] | None = None,
    speed_range: tuple[float, float] = (0.3, 0.6),
    pattern: str = "patrol",
    half_extent_m: float = 2.0,
    min_start_dist: float = 1.5,
    max_start_dist: float = 3.2,
    goal_clearance_m: float = 1.0,
    obstacle_half_height: float = 0.4,
    auto_switch_success: float = 0.9,
    max_switch_iteration: int = 3003,
    env_steps_per_iteration: int = 16,
    switch_check_interval: int = 100,
):
    """Interval event: advance scripted obstacle motion every environment step."""
    if not obstacle_names:
        return
    manager = _get_or_create_dynamic_obstacle_manager(
        env,
        obstacle_names=obstacle_names,
        speed_range=speed_range,
        pattern=pattern,
        half_extent_m=half_extent_m,
        min_start_dist=min_start_dist,
        max_start_dist=max_start_dist,
        goal_clearance_m=goal_clearance_m,
        obstacle_half_height=obstacle_half_height,
        auto_switch_success=auto_switch_success,
        max_switch_iteration=max_switch_iteration,
        env_steps_per_iteration=env_steps_per_iteration,
        switch_check_interval=switch_check_interval,
    )
    manager.update(env.step_dt)


# ----------------------------------------------------------------------
# Reward / termination functions
# ----------------------------------------------------------------------


def obstacle_proximity(
    env: "ManagerBasedRLEnv",
    obstacle_names: list[str] | None = None,
    penalty_radius: float = 1.2,
    falloff: float = 0.3,
) -> torch.Tensor:
    """Return an exponential proximity penalty that grows as the robot nears an obstacle.

    ``exp(-(d - penalty_radius) / falloff)`` peaks at 1.0 when the robot is at the
    penalty radius and decays smoothly beyond it, giving a strong, smooth gradient
    for close-range avoidance.
    """
    if not obstacle_names:
        return torch.zeros(env.num_envs, device=env.device)
    manager = _get_or_create_dynamic_obstacle_manager(env, obstacle_names=obstacle_names)
    if not manager.enabled:
        return torch.zeros(env.num_envs, device=env.device)
    dist = manager.min_dist_to_robot()
    falloff = max(float(falloff), 1.0e-3)
    return torch.exp(-(dist - penalty_radius) / falloff)


def obstacle_approach_penalty(
    env: "ManagerBasedRLEnv",
    obstacle_names: list[str] | None = None,
    speed_scale: float = 1.0,
) -> torch.Tensor:
    """Return a penalty proportional to how fast any obstacle is closing in on the robot.

    Uses the scripted obstacle velocity minus the robot root velocity projected onto
    the robot-to-obstacle direction; only positive (approaching) components count.
    """
    if not obstacle_names:
        return torch.zeros(env.num_envs, device=env.device)
    manager = _get_or_create_dynamic_obstacle_manager(env, obstacle_names=obstacle_names)
    if not manager.enabled:
        return torch.zeros(env.num_envs, device=env.device)
    closing = manager.closing_speed_to_robot()
    speed_scale = max(float(speed_scale), 1.0e-3)
    return (closing / speed_scale).clamp(max=2.0)


def obstacle_collision(
    env: "ManagerBasedRLEnv",
    obstacle_names: list[str] | None = None,
    collision_radius: float = 0.5,
) -> torch.Tensor:
    """Terminate when the robot root comes within the collision radius of an obstacle."""
    if not obstacle_names:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    manager = _get_or_create_dynamic_obstacle_manager(env, obstacle_names=obstacle_names)
    if not manager.enabled:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    return manager.min_dist_to_robot() < collision_radius

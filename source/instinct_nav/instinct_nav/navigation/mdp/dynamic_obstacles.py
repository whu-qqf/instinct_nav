"""Dynamic obstacle curriculum support for navigation tasks.

Stage-3 obstacle-avoidance training adds per-env scripted rigid obstacles that:

- are ray-castable by the shared depth camera (tracked mesh transforms),
- move through path interactions (cross, head-on, follow, merge, stop-and-go),
- reset ahead of the robot's facing with clearance constraints,
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
        body_half_extent: float = 0.25,
        spawn_delay_range: tuple[float, float] = (2.0, 8.0),
        spawn_interval_s: float = 0.0,
        scale_count_with_difficulty: bool = True,
        curriculum_env_fraction: float = 0.5,
        exact_curriculum_env_fraction: bool = False,
        graded_curriculum: bool = False,
        initial_difficulty: float = 0.0,
        difficulty_step: float = 0.2,
        advance_success: float = 0.75,
        demote_success: float = 0.0,
        easy_speed_range: tuple[float, float] = (0.1, 0.2),
        easy_min_start_dist: float = 3.5,
        easy_max_start_dist: float = 6.0,
        fraction_range: tuple[float, float] = (0.25, 0.9),
        reactive_spawn: bool = False,
        visible_spawn_dist: float = 5.0,
        max_closing_speed: float = 1.0,
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
        self.body_half_extent = float(body_half_extent)
        self.spawn_delay_range = (float(spawn_delay_range[0]), float(spawn_delay_range[1]))
        self.spawn_interval_s = max(0.0, float(spawn_interval_s))
        self.scale_count_with_difficulty = bool(scale_count_with_difficulty)
        self.curriculum_env_fraction = min(max(float(curriculum_env_fraction), 0.0), 1.0)
        self.exact_curriculum_env_fraction = bool(exact_curriculum_env_fraction)
        # --- Graded (capability-driven) curriculum knobs. When disabled (default) the
        # manager behaves exactly like the legacy binary iteration switch. ---
        self.graded_curriculum = bool(graded_curriculum)
        self.difficulty_step = float(difficulty_step)
        self.advance_success = float(advance_success)
        # Optional demotion threshold. 0.0 (default) => legacy monotonic-up curriculum
        # (never demotes). When >0, difficulty drops one step if the mean success rate
        # falls below this value, letting the policy consolidate at a tier it can master
        # instead of being permanently pinned at difficulty 1.0. Should be < advance_success
        # to avoid oscillation (a dead band between demote_success and advance_success).
        self.demote_success = float(demote_success)
        self.easy_speed_range = (float(easy_speed_range[0]), float(easy_speed_range[1]))
        self.easy_min_start_dist = float(easy_min_start_dist)
        self.easy_max_start_dist = float(easy_max_start_dist)
        self.fraction_range = (float(fraction_range[0]), float(fraction_range[1]))
        # --- Reactive spawn (P2): obstacles pop in at a standoff distance and
        # travel inward, so the robot always gets a visible approach phase instead
        # of a sudden close-range materialization. Default off => legacy pop-in. ---
        self.reactive_spawn = bool(reactive_spawn)
        self.visible_spawn_dist = float(visible_spawn_dist)
        self.max_closing_speed = float(max_closing_speed)
        # HardEncounterV2 is opt-in through a dedicated task config. Keep all
        # legacy tasks byte-for-byte on the old spawning and patrol path.
        self.hard_encounter_v2 = bool(getattr(env.cfg, "obstacle_hard_encounter_v2", False))
        self.hard_tier2_iteration = int(
            getattr(env.cfg, "obstacle_hard_tier2_iteration", 2500)
        )
        self.hard_mid_active_count = int(
            getattr(env.cfg, "obstacle_hard_mid_active_count", 3)
        )
        self.hard_high_active_count = int(
            getattr(env.cfg, "obstacle_hard_high_active_count", 4)
        )
        self.hard_mid_speed_range = tuple(
            getattr(env.cfg, "obstacle_hard_mid_speed_range", (0.35, 0.75))
        )
        self.hard_high_speed_range = tuple(
            getattr(env.cfg, "obstacle_hard_high_speed_range", (0.45, 1.0))
        )
        self.hard_recycle_delay_range = tuple(
            getattr(env.cfg, "obstacle_hard_recycle_delay_range", (0.4, 0.8))
        )
        # Rolling mean of the most recent spawn distances (diagnostics only).
        self._last_spawn_dist_mean = 40.0
        # Difficulty in [0, 1]; by default the easiest tier uses one obstacle and
        # the hardest uses all configured obstacles. Named ablations may keep the
        # count fixed while still grading speed, distance, behavior, and env fraction.
        self.difficulty = min(max(float(initial_difficulty), 0.0), 1.0)
        self._last_advance_check = -1
        self._last_gate_success_rate = 0.0
        self._last_gate_check_step = -1
        self._last_gate_decision = 0.0
        # Fixed per-env random draw. The curriculum-env mask is derived from it against
        # the current fraction, so annealing the fraction upward monotonically ADDS envs
        # (a fixed draw < a growing threshold) instead of reshuffling the subset.
        self._env_rand = torch.rand(self.num_envs, device=self.device)
        self._env_rank = torch.empty(self.num_envs, device=self.device, dtype=torch.long)
        self._env_rank[torch.argsort(self._env_rand)] = torch.arange(self.num_envs, device=self.device)
        # Fixed random subset of envs that join the dynamic-obstacle curriculum (sampled once
        # at startup, not per episode), so the rest keep training plain static navigation.
        self.curriculum_envs = self._curriculum_mask(self._current_fraction())
        self.terrain_lookup = _build_terrain_lookup(env)

        # In-run curriculum state: obstacles stay parked until the fixed PPO
        # iteration threshold. The switch is deliberately independent of success.
        self.enabled = False
        self.park_offset_m = 40.0
        self.auto_switch_success = 0.9
        self.max_switch_iteration = 4504
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
        # Keep the legacy per-env flag for rewards/terminations and an exact
        # per-obstacle mask for staggered spawning and delayed mask labels.
        self.spawned = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.spawned_obstacles = torch.zeros(
            self.num_envs, self.num_obstacles, device=self.device, dtype=torch.bool
        )
        self.spawn_timers = torch.zeros(self.num_envs, self.num_obstacles, device=self.device)
        self.trajectories_ready = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        # Derive history capacity from the visual delay configuration so changing
        # max_delay_depth cannot silently make mask labels use the wrong frame.
        delay_cfg = getattr(getattr(env, "delay_manager", None), "cfg", None)
        max_depth_delay = max(0, int(getattr(delay_cfg, "max_delay_depth", 0)))
        self._history_max_len = max_depth_delay + 1
        self._positions_history: list[torch.Tensor] = []
        self._spawned_history: list[torch.Tensor] = []
        self._curriculum_envs_history: list[torch.Tensor] = []
        self._velocities_history: list[torch.Tensor] = []

    # ------------------------------------------------------------------
    # graded-curriculum difficulty schedule
    # ------------------------------------------------------------------

    def _current_fraction(self) -> float:
        """Fraction of envs that carry obstacles at the current difficulty."""
        if not self.graded_curriculum:
            return self.curriculum_env_fraction
        lo, hi = self.fraction_range
        return lo + (hi - lo) * self.difficulty

    def _refresh_curriculum_envs(self) -> None:
        """Recompute the obstacle-env mask from the fixed per-env draw and current fraction."""
        self.curriculum_envs = self._curriculum_mask(self._current_fraction())

    def _curriculum_mask(self, fraction: float) -> torch.Tensor:
        """Return a fixed monotonic cohort, optionally with an exact env count."""
        fraction = min(max(float(fraction), 0.0), 1.0)
        if not self.exact_curriculum_env_fraction:
            return self._env_rand < fraction
        count = min(max(int(round(fraction * self.num_envs)), 0), self.num_envs)
        return self._env_rank < count

    def _effective_speed_range(self) -> tuple[float, float]:
        if not self.graded_curriculum:
            return self.speed_range
        d = self.difficulty
        lo = self.easy_speed_range[0] + (self.speed_range[0] - self.easy_speed_range[0]) * d
        hi = self.easy_speed_range[1] + (self.speed_range[1] - self.easy_speed_range[1]) * d
        return (lo, hi)

    def _effective_start_band(self) -> tuple[float, float]:
        if not self.graded_curriculum:
            return self.min_start_dist, self.max_start_dist
        d = self.difficulty
        mn = self.easy_min_start_dist + (self.min_start_dist - self.easy_min_start_dist) * d
        mx = self.easy_max_start_dist + (self.max_start_dist - self.easy_max_start_dist) * d
        return mn, mx

    def _effective_active_k(self) -> int:
        """Number of obstacles that actually spawn per env (rest stay parked)."""
        if not self.graded_curriculum or not self.scale_count_with_difficulty:
            return self.num_obstacles
        k = math.ceil(1 + self.difficulty * (self.num_obstacles - 1))
        return max(1, min(self.num_obstacles, int(k)))

    def _hard_stage(self) -> int:
        iteration = int(self.env.common_step_counter) // max(1, int(self.env_steps_per_iteration))
        return 1 if iteration < self.hard_tier2_iteration else 2

    def _hard_local_capacity(self) -> int:
        count = self.hard_mid_active_count if self._hard_stage() == 1 else self.hard_high_active_count
        return max(1, min(self.num_obstacles, count))

    def _hard_current_speed_range(self) -> tuple[float, float]:
        values = self.hard_mid_speed_range if self._hard_stage() == 1 else self.hard_high_speed_range
        return float(values[0]), float(values[1])

    def _behavior_weights(self) -> torch.Tensor:
        """Sampling weights for [cross, head-on, same-dir, merge, stop-and-go].

        Easy tiers use only gentle interactions (cross / same-direction); head-on and
        stop-and-go ramp in with difficulty so the hardest, least-avoidable encounters
        appear only once the policy can already handle the mild ones.
        """
        if not self.graded_curriculum:
            return torch.tensor([0.30, 0.25, 0.15, 0.20, 0.10], device=self.device)
        d = self.difficulty
        return torch.tensor(
            [0.30, 0.25 * d, 0.15 + 0.15 * (1.0 - d), 0.20, 0.10 * d], device=self.device
        )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _sample_trajectories(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample interaction trajectories ahead of the robot's facing.

        Obstacles are placed in front of the robot (along its current heading, so
        they land inside the forward depth FOV), never beside or behind it.

        Behaviors: 0 cross, 1 head-on, 2 same-direction, 3 diagonal merge,
        4 stop-and-go. Rejection sampling enforces robot/goal clearance and
        obstacle separation; unresolved candidates fall back to spaced crossings.
        """
        n = env_ids.numel()
        robot = self.env.scene["robot"]
        robot_xy = robot.data.root_pos_w[env_ids, :2]
        goal_cmd = getattr(getattr(self.env, "command_manager", None), "_terms", {}).get("robot_goal", None)
        if goal_cmd is not None and hasattr(goal_cmd, "goal_position_world"):
            goal_xy = goal_cmd.goal_position_world[env_ids, :2]
        else:
            goal_xy = robot_xy + torch.tensor([4.0, 0.0], device=self.device)

        # Forward = robot heading (base +x axis), NOT the goal direction, so
        # obstacles only ever appear ahead of where the robot is facing.
        quat = robot.data.root_quat_w[env_ids]  # [n, 4] wxyz
        qw, qx, qy, qz = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        fx = 1.0 - 2.0 * (qy * qy + qz * qz)
        fy = 2.0 * (qx * qy + qw * qz)
        forward = torch.stack([fx, fy], dim=-1)
        forward = forward / torch.norm(forward, dim=-1, keepdim=True).clamp_min(1.0e-6)
        lateral = torch.stack([-forward[:, 1], forward[:, 0]], dim=-1)

        # Graded curriculum shrinks the start band (closer/harder as difficulty rises)
        # and ramps the hardest behaviors in; both fall back to fixed legacy values.
        min_start, max_start = self._effective_start_band()
        starts = torch.zeros(n, self.num_obstacles, 2, device=self.device)
        ends = torch.zeros_like(starts)
        behaviors = torch.multinomial(
            self._behavior_weights(),
            n * self.num_obstacles, replacement=True,
        ).view(n, self.num_obstacles)
        for k in range(self.num_obstacles):
            accepted = torch.zeros(n, device=self.device, dtype=torch.bool)
            for attempt in range(12):
                # Anchor strictly ahead of the robot, in the [min_start, max_start] band.
                d_anchor = min_start + torch.rand(n, 1, device=self.device) * (
                    max_start - min_start
                )
                anchor = robot_xy + forward * d_anchor
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
                clear = (torch.norm(candidate_start - robot_xy, dim=-1) >= min_start)
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

        active_speed = (
            self.speeds[env_ids]
            * (~self.paused[env_ids]).to(self.speeds.dtype)
            * self.spawned_obstacles[env_ids].to(self.speeds.dtype)
        )
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

        # Obstacles always start hidden, parked far from the robot so depth/visibility
        # never sees them and proximity/collision terms stay ~zero. In the dynamic phase
        # they pop in ahead of the robot mid-travel once their spawn delay elapses, rather
        # than existing at episode reset.
        center = (
            self.env.scene.env_origins[env_ids][:, :2].unsqueeze(1)
            + torch.tensor([[-self.park_offset_m, 0.0]], device=self.device)
        ).expand(n, self.num_obstacles, 2)

        self.spawned[env_ids] = False
        self.spawned_obstacles[env_ids] = False
        self.trajectories_ready[env_ids] = False
        first_delay = (
            torch.rand(n, 1, device=self.device) * (self.spawn_delay_range[1] - self.spawn_delay_range[0])
            + self.spawn_delay_range[0]
        )
        stagger = torch.arange(self.num_obstacles, device=self.device, dtype=torch.float32).unsqueeze(0)
        self.spawn_timers[env_ids] = first_delay + stagger * self.spawn_interval_s
        if self.hard_encounter_v2:
            # Fill the local pool quickly after the static warm-up. Subsequent
            # obstacles are recycled with randomized delays in update().
            self.spawn_timers[env_ids] = first_delay + stagger * 0.55

        positions = torch.zeros(n, self.num_obstacles, 3, device=self.device)
        positions[:, :, :2] = center
        # Parked far outside the terrain tile; gravity is disabled so the obstacles
        # float inert in place and can never interact with the robot.
        positions[:, :, 2] = 0.5

        self.centers[env_ids] = center
        self.speeds[env_ids] = 0.0
        self.dirs[env_ids] = 0.0
        self.timers[env_ids] = 0.0
        self.positions_w[env_ids] = positions
        self._set_pose_and_velocity(env_ids, positions)

    def update(self, dt: float):
        """Advance scripted motion and pop obstacles in ahead of the robot mid-travel."""
        dt = float(dt)
        self._log_metrics()
        if not self.enabled:
            # Static-phase obstacles stay parked until the fixed iteration threshold.
            self._maybe_switch_curriculum()
            self._record_state_history()
            return

        # Graded curriculum: ratchet difficulty up as the policy proves competent.
        self._maybe_advance_difficulty()

        # --- Sudden spawn: obstacles pop in ahead of the robot once their per-env
        #     delay elapses (mid-travel), not at episode reset. Only curriculum envs
        #     ever spawn; the rest stay parked (plain static navigation). ---
        active_k = self.num_obstacles if self.hard_encounter_v2 else self._effective_active_k()
        eligible = (
            torch.arange(self.num_obstacles, device=self.device).unsqueeze(0) < active_k
        )
        unspawned = (~self.spawned_obstacles) & self.curriculum_envs.unsqueeze(1) & eligible
        if bool(unspawned.any().item()):
            self.spawn_timers = torch.where(unspawned, self.spawn_timers - dt, self.spawn_timers)
            due = unspawned & (self.spawn_timers <= 0.0)
            if bool(due.any().item()):
                if self.hard_encounter_v2:
                    self._spawn_hard_encounters(due)
                elif getattr(self.env.cfg, "obstacle_encounter_spawn", False):
                    self._spawn_encounters(due)
                elif self.spawn_interval_s <= 0.0:
                    self._spawn_envs(torch.nonzero(due.any(dim=1), as_tuple=False).reshape(-1))
                else:
                    prepare = due.any(dim=1) & ~self.trajectories_ready
                    if bool(prepare.any().item()):
                        self._prepare_trajectories(torch.nonzero(prepare, as_tuple=False).reshape(-1))
                    self._activate_obstacles(due)

        # --- Only spawned obstacles move; the rest stay parked. ---
        spawned_mask = self.spawned_obstacles

        # Stop-and-go trajectories pause for 0.5--1.5 s, then resume.
        stop_go = (self.behaviors == 4) & spawned_mask
        self.timers = torch.where(spawned_mask, self.timers - dt, self.timers)
        expire = stop_go & (self.timers <= 0.0)
        if bool(expire.any().item()):
            self.paused = torch.where(expire, ~self.paused, self.paused)
            moving_time = torch.rand_like(self.timers) * 2.0 + 1.5
            pause_time = torch.rand_like(self.timers) + 0.5
            self.timers = torch.where(expire, torch.where(self.paused, pause_time, moving_time), self.timers)

        # Integrate motion (unspawned obstacles have zero speed, so they stay parked).
        active_speed = self.speeds * (~self.paused).to(self.speeds.dtype) * spawned_mask.to(self.speeds.dtype)
        self.positions_w[:, :, :2] += self.dirs * active_speed.unsqueeze(-1) * dt
        target = torch.where(
            (torch.sum(self.dirs * (self.ends - self.starts), dim=-1) >= 0.0).unsqueeze(-1),
            self.ends, self.starts,
        )
        reached = (
            torch.sum((target - self.positions_w[:, :, :2]) * self.dirs, dim=-1) <= 0.0
        ) & spawned_mask
        if self.hard_encounter_v2 and bool(reached.any().item()):
            # Recycle completed trajectories instead of bouncing forever. This
            # keeps encounter density stable without tethering paths to the robot.
            self.spawned_obstacles[reached] = False
            self.paused[reached] = False
            self.speeds[reached] = 0.0
            self.dirs[reached] = 0.0
            recycle_delay = torch.rand_like(self.spawn_timers)
            lo, hi = self.hard_recycle_delay_range
            recycle_delay = recycle_delay * (float(hi) - float(lo)) + float(lo)
            self.spawn_timers = torch.where(reached, recycle_delay, self.spawn_timers)
            park_xy = self.env.scene.env_origins[:, :2] + torch.tensor(
                [-self.park_offset_m, 0.0], device=self.device
            )
            parked = park_xy.unsqueeze(1).expand(-1, self.num_obstacles, -1)
            self.positions_w[:, :, :2] = torch.where(
                reached.unsqueeze(-1), parked, self.positions_w[:, :, :2]
            )
            self.spawned = self.spawned_obstacles.any(dim=1)
        else:
            self.positions_w[:, :, :2] = torch.where(
                reached.unsqueeze(-1), target, self.positions_w[:, :, :2]
            )
            self.dirs = torch.where(reached.unsqueeze(-1), -self.dirs, self.dirs)

        # Refresh z so spawned obstacles track terrain height; parked ones keep their z.
        all_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        flat_pos = self.positions_w.reshape(-1, 3)
        flat_ids = all_ids.repeat_interleave(self.num_obstacles)
        z = _terrain_z_at(self.env, self.terrain_lookup, flat_pos, flat_ids).reshape(
            self.num_envs, self.num_obstacles
        )
        new_z = z + self.obstacle_half_height
        self.positions_w[:, :, 2] = torch.where(spawned_mask, new_z, self.positions_w[:, :, 2])

        self._set_pose_and_velocity(all_ids, self.positions_w)
        self._record_state_history()

    def _record_state_history(self) -> None:
        """Append the current obstacle state to the delayed-label history.

        Keeps the most recent ``_history_max_len`` position and activation snapshots
        so the mask label can index the state at the same lag as the visual packet.
        """
        self._positions_history.append(self.positions_w.clone())
        self._spawned_history.append(self.spawned_obstacles.clone())
        self._curriculum_envs_history.append(self.curriculum_envs.clone())
        effective_velocity = self.dirs * (
            self.speeds
            * (~self.paused).to(self.speeds.dtype)
            * self.spawned_obstacles.to(self.speeds.dtype)
        ).unsqueeze(-1)
        self._velocities_history.append(effective_velocity.clone())
        if len(self._positions_history) > self._history_max_len:
            self._positions_history.pop(0)
            self._spawned_history.pop(0)
            self._curriculum_envs_history.pop(0)
            self._velocities_history.pop(0)

    def _prepare_trajectories(self, env_ids: torch.Tensor) -> None:
        """Sample one episode's obstacle trajectories from the robot's current pose."""
        n = env_ids.numel()
        if n == 0:
            return
        starts, ends, behaviors = self._sample_trajectories(env_ids)
        speed_lo, speed_hi = self._effective_speed_range()

        # Reactive spawn (P2): push each pop-in point outward along the robot->start
        # direction so obstacles always appear at >= visible_spawn_dist and travel
        # inward toward the sampled interaction point. This removes the sudden
        # close-range materialization (a physically unavoidable-collision source)
        # and guarantees a visible approach phase; ends stay fixed so the encounter
        # geometry/difficulty is preserved. Speed is capped so the closing rate stays
        # reactable at the 5 Hz high-level control rate.
        if self.reactive_spawn:
            robot = self.env.scene["robot"]
            robot_xy = robot.data.root_pos_w[env_ids, :2].unsqueeze(1)  # [n, 1, 2]
            offset = starts - robot_xy  # [n, K, 2]
            dist = torch.norm(offset, dim=-1, keepdim=True).clamp_min(1.0e-3)
            unit = offset / dist
            new_dist = dist.clamp_min(self.visible_spawn_dist)
            starts = robot_xy + unit * new_dist
            speed_hi = min(speed_hi, self.max_closing_speed)
            speed_lo = min(speed_lo, speed_hi)

        speeds = (
            torch.rand(n, self.num_obstacles, device=self.device) * (speed_hi - speed_lo)
            + speed_lo
        )
        dirs = torch.nn.functional.normalize(ends - starts, dim=-1)

        # Graded curriculum only activates the first ``active_k`` obstacles; the rest are
        # parked far away (zero speed, no trajectory) so easy tiers face fewer obstacles.
        active_k = self._effective_active_k()
        if active_k < self.num_obstacles:
            park_xy = (
                self.env.scene.env_origins[env_ids][:, :2].unsqueeze(1)
                + torch.tensor([[-self.park_offset_m, 0.0]], device=self.device)
            )  # [n, 1, 2]
            starts[:, active_k:] = park_xy
            ends[:, active_k:] = park_xy
            speeds[:, active_k:] = 0.0
            dirs[:, active_k:] = 0.0

        self.starts[env_ids] = starts
        self.ends[env_ids] = ends
        self.behaviors[env_ids] = behaviors
        self.speeds[env_ids] = speeds
        self.dirs[env_ids] = dirs
        self.paused[env_ids] = False
        self.timers[env_ids] = torch.rand(n, self.num_obstacles, device=self.device) * 3.0 + 1.0
        self.trajectories_ready[env_ids] = True

    def _spawn_encounters(self, due: torch.Tensor) -> None:
        """Activate one visible crossing per env, conditioned on a broad flat corridor.

        Five slots remain available, but a timer alone cannot force an unsafe
        spawn. Rejected candidates stay parked and are retried on later steps.
        Paths are fixed at activation; no pursuit or trajectory retargeting occurs.
        """
        robot = self.env.scene["robot"]
        nearby_count = ((torch.norm(self.positions_w[..., :2] - robot.data.root_pos_w[:, None, :2], dim=-1) < 6.0)
                        & self.spawned_obstacles).sum(1)
        capacity = 1 if self.difficulty < 0.6 else 2
        eligible = due.any(1) & (nearby_count < capacity)
        ids = torch.nonzero(eligible, as_tuple=False).flatten()
        if ids.numel() == 0:
            return
        slots = due[ids].long().argmax(1)
        root = robot.data.root_pos_w[ids]
        q = robot.data.root_quat_w[ids]
        forward = torch.stack((1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2),
                               2 * (q[:, 1] * q[:, 2] + q[:, 0] * q[:, 3])), -1)
        forward = torch.nn.functional.normalize(forward, dim=-1)
        lateral = torch.stack((-forward[:, 1], forward[:, 0]), -1)
        velocity = (robot.data.root_lin_vel_w[ids, :2] * forward).sum(-1)
        lo, hi = self._effective_speed_range()
        speed = lo + torch.rand_like(velocity) * (hi - lo)
        # Match the crossing midpoint to a constant-velocity robot prediction.
        distance = torch.full_like(velocity, self.visible_spawn_dist)
        encounter_time = distance / velocity.clamp_min(0.3)
        span = speed * encounter_time
        side = torch.where(torch.rand_like(speed) < 0.5, -1.0, 1.0)
        anchor = root[:, :2] + forward * distance[:, None]
        start = anchor + lateral * (side * span)[:, None]
        end = anchor - lateral * (side * span)[:, None]
        goal = self.env.command_manager._terms["robot_goal"].goal_position_world[ids, :2]
        valid = (velocity > 0.3) & (span >= 0.7) & (span <= 2.0)
        valid &= (torch.norm(goal - anchor, dim=-1) > self.goal_clearance_m + 1.0)
        # Conservative height-field screen: accept only broad, locally flat areas.
        offsets = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], device=self.device)
        corridor = anchor[:, None, :] + lateral[:, None, :] * offsets[None, :, None]
        points = torch.cat((corridor, start[:, None], end[:, None]), dim=1)
        xyz = torch.cat((points, torch.zeros_like(points[..., :1])), dim=-1)
        heights = _terrain_z_at(self.env, self.terrain_lookup, xyz.flatten(0, 1), ids.repeat_interleave(7)).reshape(-1, 7)
        valid &= torch.isfinite(heights).all(1) & ((heights.max(1).values - heights.min(1).values) < 0.15)
        ids, slots = ids[valid], slots[valid]
        if ids.numel() == 0:
            return
        self.starts[ids, slots] = start[valid]
        self.ends[ids, slots] = end[valid]
        self.speeds[ids, slots] = speed[valid]
        self.dirs[ids, slots] = torch.nn.functional.normalize(end[valid] - start[valid], dim=-1)
        self.behaviors[ids, slots] = 0
        self.paused[ids, slots] = False
        activate = torch.zeros_like(due)
        activate[ids, slots] = True
        self._activate_obstacles(activate)
        remaining = ~self.spawned_obstacles[ids]
        self.spawn_timers[ids] = torch.where(remaining, self.spawn_timers[ids].clamp_min(self.spawn_interval_s), self.spawn_timers[ids])

    def _spawn_hard_encounters(self, due: torch.Tensor) -> None:
        """Spawn one conflict-conditioned trajectory per eligible environment.

        The pool is local and recyclable: many slots may exist, but only the
        configured 3/4 obstacles can be within 6.5 m simultaneously. Candidate
        paths are fixed in world coordinates after activation and never pursue
        the robot.
        """
        robot = self.env.scene["robot"]
        root_xy_all = robot.data.root_pos_w[:, :2]
        nearby = torch.norm(self.positions_w[..., :2] - root_xy_all[:, None], dim=-1) < 6.5
        nearby_count = (nearby & self.spawned_obstacles).sum(dim=1)
        eligible = due.any(dim=1) & (nearby_count < self._hard_local_capacity())
        ids = torch.nonzero(eligible, as_tuple=False).flatten()
        if ids.numel() == 0:
            return

        slots = due[ids].long().argmax(dim=1)
        root = robot.data.root_pos_w[ids]
        quat = robot.data.root_quat_w[ids]
        forward = torch.stack(
            (
                1.0 - 2.0 * (quat[:, 2].square() + quat[:, 3].square()),
                2.0 * (quat[:, 1] * quat[:, 2] + quat[:, 0] * quat[:, 3]),
            ),
            dim=-1,
        )
        forward = torch.nn.functional.normalize(forward, dim=-1)
        lateral = torch.stack((-forward[:, 1], forward[:, 0]), dim=-1)
        side = torch.where(torch.rand(ids.numel(), device=self.device) < 0.5, -1.0, 1.0)

        speed_lo, speed_hi = self._hard_current_speed_range()
        speed = speed_lo + torch.rand(ids.numel(), device=self.device) * (speed_hi - speed_lo)
        weights = torch.tensor([0.30, 0.25, 0.10, 0.20, 0.10, 0.05], device=self.device)
        mode = torch.multinomial(weights, ids.numel(), replacement=True)

        anchor_dist = 2.4 + torch.rand_like(speed) * 1.6
        anchor = root[:, :2] + forward * anchor_dist[:, None]
        travel_time = 2.0 + torch.rand_like(speed) * 1.5
        span = (speed * travel_time).clamp(1.4, 3.5)

        cross_start = anchor + lateral * (side * span)[:, None]
        cross_end = anchor - lateral * (side * span)[:, None]

        head_dist = 4.5 + torch.rand_like(speed) * 1.5
        head_start = root[:, :2] + forward * head_dist[:, None] + lateral * (side * 0.25)[:, None]
        head_end = root[:, :2] + forward * 0.6 - lateral * (side * 0.25)[:, None]

        same_start = root[:, :2] + forward * (3.2 + torch.rand_like(speed))[:, None]
        same_start = same_start + lateral * (side * 0.55)[:, None]
        same_end = same_start + forward * (3.5 + torch.rand_like(speed) * 1.5)[:, None]

        merge_start = anchor + lateral * (side * span)[:, None] + forward * 0.8
        merge_end = anchor - lateral * (side * 0.35)[:, None] + forward * 1.5

        # Mode 4 reuses a crossing path with stop-and-go timing. Mode 5 uses a
        # shallow diagonal whose repeated recycle yields a curved/sweeping mix
        # without per-step pursuit or retargeting.
        sweep_start = anchor + lateral * (side * span)[:, None] - forward * 0.8
        sweep_end = anchor - lateral * (side * 0.7)[:, None] + forward * 2.0

        start = cross_start
        end = cross_end
        start = torch.where((mode == 1)[:, None], head_start, start)
        end = torch.where((mode == 1)[:, None], head_end, end)
        start = torch.where((mode == 2)[:, None], same_start, start)
        end = torch.where((mode == 2)[:, None], same_end, end)
        start = torch.where((mode == 3)[:, None], merge_start, start)
        end = torch.where((mode == 3)[:, None], merge_end, end)
        start = torch.where((mode == 5)[:, None], sweep_start, start)
        end = torch.where((mode == 5)[:, None], sweep_end, end)

        # Guarantee a visible approach without creating unavoidable close-range
        # pop-ins. Moving the start outward leaves the conflict point unchanged.
        start_offset = start - root[:, :2]
        start_distance = torch.norm(start_offset, dim=-1).clamp_min(1.0e-3)
        min_spawn = 4.0
        start = root[:, :2] + start_offset * (
            start_distance.clamp_min(min_spawn) / start_distance
        )[:, None]

        goal = self.env.command_manager._terms["robot_goal"].goal_position_world[ids, :2]
        valid = torch.norm(goal - anchor, dim=-1) > self.goal_clearance_m + 0.8
        valid &= torch.norm(end - start, dim=-1) > 1.0

        # Only spawn on a broad, locally flat corridor. Invalid candidates retry
        # on a later update instead of being forced into stairs or gaps.
        offsets = torch.tensor([-1.5, -0.75, 0.0, 0.75, 1.5], device=self.device)
        corridor = anchor[:, None] + lateral[:, None] * offsets[None, :, None]
        points = torch.cat((corridor, start[:, None], end[:, None]), dim=1)
        xyz = torch.cat((points, torch.zeros_like(points[..., :1])), dim=-1)
        point_count = points.shape[1]
        heights = _terrain_z_at(
            self.env,
            self.terrain_lookup,
            xyz.flatten(0, 1),
            ids.repeat_interleave(point_count),
        ).reshape(-1, point_count)
        valid &= torch.isfinite(heights).all(dim=1)
        valid &= heights.max(dim=1).values - heights.min(dim=1).values < 0.15

        valid_ids = ids[valid]
        valid_slots = slots[valid]
        if valid_ids.numel() == 0:
            return
        self.starts[valid_ids, valid_slots] = start[valid]
        self.ends[valid_ids, valid_slots] = end[valid]
        self.speeds[valid_ids, valid_slots] = speed[valid]
        self.dirs[valid_ids, valid_slots] = torch.nn.functional.normalize(
            end[valid] - start[valid], dim=-1
        )
        self.behaviors[valid_ids, valid_slots] = mode[valid]
        self.paused[valid_ids, valid_slots] = False
        self.timers[valid_ids, valid_slots] = torch.rand(valid_ids.numel(), device=self.device) * 1.5 + 1.5
        activate = torch.zeros_like(due)
        activate[valid_ids, valid_slots] = True
        self._activate_obstacles(activate)

        # Stagger the remaining slots independently. Rejected candidates retry
        # promptly; accepted environments wait before admitting another conflict.
        lo, hi = self.hard_recycle_delay_range
        retry = torch.rand_like(self.spawn_timers[ids]) * (float(hi) - float(lo)) + float(lo)
        remaining = ~self.spawned_obstacles[ids]
        self.spawn_timers[ids] = torch.where(remaining, retry, self.spawn_timers[ids])

    def _activate_obstacles(self, due: torch.Tensor) -> None:
        """Move the selected per-env obstacle slots from parking onto their paths."""
        pairs = torch.nonzero(due, as_tuple=False)
        if pairs.numel() == 0:
            return
        env_ids = pairs[:, 0]
        obstacle_ids = pairs[:, 1]
        starts = self.starts[env_ids, obstacle_ids]

        positions = torch.zeros(pairs.shape[0], 3, device=self.device)
        positions[:, :2] = starts
        positions[:, 2] = _terrain_z_at(self.env, self.terrain_lookup, positions, env_ids) + self.obstacle_half_height
        self.positions_w[env_ids, obstacle_ids] = positions
        self.spawned_obstacles[env_ids, obstacle_ids] = True
        self.spawned = self.spawned_obstacles.any(dim=1)
        unique_env_ids = torch.unique(env_ids)
        self._set_pose_and_velocity(unique_env_ids, self.positions_w[unique_env_ids])

        robot_xy = self.env.scene["robot"].data.root_pos_w[env_ids, :2]
        spawn_dist = torch.norm(starts - robot_xy, dim=-1)
        if spawn_dist.numel() > 0:
            self._last_spawn_dist_mean = float(spawn_dist.mean().item())

    def _spawn_envs(self, env_ids: torch.Tensor) -> None:
        """Prepare and activate every difficulty-eligible obstacle at once."""
        if env_ids.numel() == 0:
            return
        self._prepare_trajectories(env_ids)
        active_k = self._effective_active_k()
        due = torch.zeros(
            self.num_envs, self.num_obstacles, device=self.device, dtype=torch.bool
        )
        due[env_ids, :active_k] = True
        self._activate_obstacles(due)

    def enable(self, env_ids: Sequence[int] | torch.Tensor | slice | None = None):
        """Switch the curriculum on: place obstacles into the action ring and start motion."""
        if self.enabled:
            return
        self.enabled = True
        if self.graded_curriculum:
            self._refresh_curriculum_envs()
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
        log["Metrics/dynamic_obstacles/difficulty"] = float(self.difficulty)
        log["Metrics/dynamic_obstacles/gate_success_rate"] = float(self._last_gate_success_rate)
        log["Metrics/dynamic_obstacles/gate_success_threshold"] = float(self.advance_success)
        log["Metrics/dynamic_obstacles/gate_check_step"] = float(self._last_gate_check_step)
        log["Metrics/dynamic_obstacles/gate_decision"] = float(self._last_gate_decision)
        goal_cmd = getattr(getattr(self.env, "command_manager", None), "_terms", {}).get("robot_goal", None)
        tracker = getattr(goal_cmd, "success_tracker", None)
        if tracker is not None:
            filled = (tracker.buffer >= 0).sum(dim=1)
            log["Metrics/dynamic_obstacles/success_window_fill_mean"] = float(filled.float().mean().item())
            log["Metrics/dynamic_obstacles/success_window_full_fraction"] = float(
                (filled >= tracker.buffer_size).float().mean().item()
            )
        log["Metrics/dynamic_obstacles/curriculum_env_frac"] = float(
            self.curriculum_envs.float().mean().item()
        )
        log["Metrics/dynamic_obstacles/spawn_dist_mean"] = float(self._last_spawn_dist_mean)
        log["Metrics/dynamic_obstacles/active_count_mean"] = float(
            self.spawned_obstacles.float().sum(dim=1)[self.curriculum_envs].mean().item()
        ) if bool(self.curriculum_envs.any().item()) else 0.0
        clean_mask_rate = getattr(self, "_last_direct_mask_clean_positive_rate", None)
        input_mask_rate = getattr(self, "_last_direct_mask_input_positive_rate", None)
        bev_token_rate = getattr(self, "_last_dynamic_bev_token_positive_rate", None)
        if clean_mask_rate is not None:
            log["Metrics/dynamic_mask/clean_positive_rate"] = float(clean_mask_rate.item())
        if input_mask_rate is not None:
            log["Metrics/dynamic_mask/input_positive_rate"] = float(input_mask_rate.item())
        if bev_token_rate is not None:
            log["Metrics/dynamic_mask/bev_token_positive_rate"] = float(bev_token_rate.item())
        sector_rate = getattr(self, "_last_dynamic_sector_risk_rate", None)
        visible_instances = getattr(self, "_last_dynamic_visible_instances", None)
        if sector_rate is not None:
            log["Metrics/dynamic_risk/sector_positive_rate"] = float(sector_rate.item())
        if visible_instances is not None:
            log["Metrics/dynamic_risk/visible_instances"] = float(visible_instances.item())
        if self.hard_encounter_v2:
            log["Metrics/dynamic_obstacles/local_capacity"] = float(self._hard_local_capacity())
            log["Metrics/dynamic_obstacles/hard_stage"] = float(self._hard_stage())
        spawned = self.spawned
        if bool(spawned.any().item()):
            log["Metrics/dynamic_obstacles/min_dist_mean"] = float(
                self.min_dist_to_robot()[spawned].mean().item()
            )
        else:
            log["Metrics/dynamic_obstacles/min_dist_mean"] = float(self.park_offset_m)

    def _mean_success_rate(self) -> float:
        """Windowed mean success rate from the robot-goal command tracker.

        The robot-goal command term keeps its rolling-window tracker in
        ``success_tracker`` (and mirrors the per-env rate into
        ``metrics["success_rate"]`` each step). ``get_success_rate`` lives on the
        tracker, not on the command term, so read the tracker directly and fall
        back to the mirrored metric.
        """
        goal_cmd = getattr(getattr(self.env, "command_manager", None), "_terms", {}).get("robot_goal", None)
        if goal_cmd is None:
            return 0.0
        tracker = getattr(goal_cmd, "success_tracker", None)
        if tracker is not None and hasattr(tracker, "get_success_rate"):
            rate = tracker.get_success_rate()
        else:
            metrics = getattr(goal_cmd, "metrics", None)
            rate = metrics.get("success_rate") if isinstance(metrics, dict) else None
        if rate is None or rate.numel() == 0:
            return 0.0
        return float(rate.mean().item())

    def _maybe_switch_curriculum(self):
        """One-way switch from the static phase to the dynamic phase.

        Legacy (non-graded) path: triggered only at the configured fixed PPO
        iteration threshold. Graded path: enabled as soon as the policy clears
        ``auto_switch_success`` (navigation is ready), with the iteration threshold
        kept as a hard fallback so obstacles always turn on eventually.
        """
        step = int(self.env.common_step_counter)
        if step < 2:
            return
        steps_per_iter = max(1, int(self.env_steps_per_iteration))
        cap_env_steps = int(self.max_switch_iteration) * steps_per_iter

        if self.graded_curriculum:
            interval = max(1, int(self.switch_check_interval))
            if step - self._last_switch_check >= interval:
                self._last_switch_check = step
                if self._mean_success_rate() >= self.auto_switch_success:
                    print(
                        f"[DynamicObstacleManager] Capability-triggered curriculum enable at "
                        f"env step {step} (success >= {self.auto_switch_success:.2f})."
                    )
                    self.enable()
                    return

        if step >= cap_env_steps:
            approx_iteration = step // steps_per_iter
            print(
                f"[DynamicObstacleManager] Switching curriculum at env step {step} "
                f"(~iteration {approx_iteration}) (fixed iteration {self.max_switch_iteration})."
            )
            self.enable()

    def _maybe_advance_difficulty(self):
        """Ratchet graded difficulty toward the tier the policy can sustain.

        Checked at most once per ``switch_check_interval`` env steps. Advances when the
        mean success rate clears ``advance_success``. When ``demote_success`` > 0 the
        curriculum is bidirectional: difficulty also drops one step if success falls
        below ``demote_success`` (a dead band in between prevents oscillation), so the
        policy is not permanently pinned at a tier it cannot master. With the default
        ``demote_success == 0.0`` the schedule is legacy monotonic-up.
        """
        if not self.graded_curriculum:
            return
        step = int(self.env.common_step_counter)
        interval = max(1, int(self.switch_check_interval))
        if step - self._last_advance_check < interval:
            return
        self._last_advance_check = step
        success = self._mean_success_rate()
        self._last_gate_success_rate = success
        self._last_gate_check_step = step
        self._last_gate_decision = 0.0
        if success >= self.advance_success and self.difficulty < 1.0:
            self.difficulty = min(1.0, self.difficulty + self.difficulty_step)
            self._last_gate_decision = 1.0
            self._refresh_curriculum_envs()
            print(
                f"[DynamicObstacleManager] Difficulty advanced to {self.difficulty:.2f} "
                f"at env step {step} (success >= {self.advance_success:.2f})."
            )
        elif self.demote_success > 0.0 and success < self.demote_success and self.difficulty > 0.0:
            self.difficulty = max(0.0, self.difficulty - self.difficulty_step)
            self._last_gate_decision = -1.0
            self._refresh_curriculum_envs()
            print(
                f"[DynamicObstacleManager] Difficulty demoted to {self.difficulty:.2f} "
                f"at env step {step} (success < {self.demote_success:.2f})."
            )

    def min_dist_to_robot(self) -> torch.Tensor:
        """Minimum XY distance from the robot root to any obstacle: [num_envs]."""
        robot = self.env.scene["robot"]
        root_xy = robot.data.root_pos_w[:, :2].unsqueeze(1)  # [num_envs, 1, 2]
        dist = torch.norm(self.positions_w[:, :, :2] - root_xy, dim=-1)  # [num_envs, K]
        dist = torch.where(self.spawned_obstacles, dist, torch.full_like(dist, float("inf")))
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

        # Use the *effective* speed actually written to the sim: a paused stop-and-go
        # obstacle has zero velocity, so it must not report its scripted speed here
        # (otherwise the robot is penalized for a phantom "approach" by a stationary
        # obstacle). Mirrors the (~paused) masking in _set_pose_and_velocity/update.
        active_speed = self.speeds * (~self.paused).to(self.speeds.dtype)
        obs_vel = self.dirs * active_speed.unsqueeze(-1)  # [num_envs, K, 2] world XY
        robot_vel = robot.data.root_lin_vel_w[:, :2].unsqueeze(1)  # [num_envs, 1, 2]
        # closing speed = -d(dist)/dt = dot(robot_vel - obs_vel, unit); positive
        # when the obstacle is approaching (distance shrinking).
        closing = torch.sum((robot_vel - obs_vel) * unit, dim=-1)  # [num_envs, K]
        closing = torch.where(self.spawned_obstacles, closing, torch.zeros_like(closing))
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
    body_half_extent: float = 0.25,
    spawn_delay_range: tuple[float, float] = (2.0, 8.0),
    spawn_interval_s: float = 0.0,
    scale_count_with_difficulty: bool = True,
    curriculum_env_fraction: float = 0.5,
    exact_curriculum_env_fraction: bool = False,
    auto_switch_success: float = 0.9,
    max_switch_iteration: int = 4504,
    env_steps_per_iteration: int = 16,
    switch_check_interval: int = 100,
    graded_curriculum: bool = False,
    initial_difficulty: float = 0.0,
    difficulty_step: float = 0.2,
    advance_success: float = 0.75,
    demote_success: float = 0.0,
    easy_speed_range: tuple[float, float] = (0.1, 0.2),
    easy_min_start_dist: float = 3.5,
    easy_max_start_dist: float = 6.0,
    fraction_range: tuple[float, float] = (0.25, 0.9),
    reactive_spawn: bool = False,
    visible_spawn_dist: float = 5.0,
    max_closing_speed: float = 1.0,
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
            body_half_extent=body_half_extent,
            spawn_delay_range=spawn_delay_range,
            spawn_interval_s=spawn_interval_s,
            scale_count_with_difficulty=scale_count_with_difficulty,
            curriculum_env_fraction=curriculum_env_fraction,
            exact_curriculum_env_fraction=exact_curriculum_env_fraction,
            graded_curriculum=graded_curriculum,
            initial_difficulty=initial_difficulty,
            difficulty_step=difficulty_step,
            advance_success=advance_success,
            demote_success=demote_success,
            easy_speed_range=easy_speed_range,
            easy_min_start_dist=easy_min_start_dist,
            easy_max_start_dist=easy_max_start_dist,
            fraction_range=fraction_range,
            reactive_spawn=reactive_spawn,
            visible_spawn_dist=visible_spawn_dist,
            max_closing_speed=max_closing_speed,
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
    body_half_extent: float = 0.25,
    spawn_delay_range: tuple[float, float] = (2.0, 8.0),
    spawn_interval_s: float = 0.0,
    scale_count_with_difficulty: bool = True,
    curriculum_env_fraction: float = 0.5,
    exact_curriculum_env_fraction: bool = False,
    auto_switch_success: float = 0.9,
    max_switch_iteration: int = 4504,
    env_steps_per_iteration: int = 16,
    switch_check_interval: int = 100,
    graded_curriculum: bool = False,
    initial_difficulty: float = 0.0,
    difficulty_step: float = 0.2,
    advance_success: float = 0.75,
    demote_success: float = 0.0,
    easy_speed_range: tuple[float, float] = (0.1, 0.2),
    easy_min_start_dist: float = 3.5,
    easy_max_start_dist: float = 6.0,
    fraction_range: tuple[float, float] = (0.25, 0.9),
    reactive_spawn: bool = False,
    visible_spawn_dist: float = 5.0,
    max_closing_speed: float = 1.0,
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
        body_half_extent=body_half_extent,
        spawn_delay_range=spawn_delay_range,
        spawn_interval_s=spawn_interval_s,
        scale_count_with_difficulty=scale_count_with_difficulty,
        curriculum_env_fraction=curriculum_env_fraction,
        exact_curriculum_env_fraction=exact_curriculum_env_fraction,
        auto_switch_success=auto_switch_success,
        max_switch_iteration=max_switch_iteration,
        env_steps_per_iteration=env_steps_per_iteration,
        switch_check_interval=switch_check_interval,
        graded_curriculum=graded_curriculum,
        initial_difficulty=initial_difficulty,
        difficulty_step=difficulty_step,
        advance_success=advance_success,
        demote_success=demote_success,
        easy_speed_range=easy_speed_range,
        easy_min_start_dist=easy_min_start_dist,
        easy_max_start_dist=easy_max_start_dist,
        fraction_range=fraction_range,
        reactive_spawn=reactive_spawn,
        visible_spawn_dist=visible_spawn_dist,
        max_closing_speed=max_closing_speed,
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
    body_half_extent: float = 0.25,
    spawn_delay_range: tuple[float, float] = (2.0, 8.0),
    spawn_interval_s: float = 0.0,
    scale_count_with_difficulty: bool = True,
    curriculum_env_fraction: float = 0.5,
    exact_curriculum_env_fraction: bool = False,
    auto_switch_success: float = 0.9,
    max_switch_iteration: int = 4504,
    env_steps_per_iteration: int = 16,
    switch_check_interval: int = 100,
    graded_curriculum: bool = False,
    initial_difficulty: float = 0.0,
    difficulty_step: float = 0.2,
    advance_success: float = 0.75,
    demote_success: float = 0.0,
    easy_speed_range: tuple[float, float] = (0.1, 0.2),
    easy_min_start_dist: float = 3.5,
    easy_max_start_dist: float = 6.0,
    fraction_range: tuple[float, float] = (0.25, 0.9),
    reactive_spawn: bool = False,
    visible_spawn_dist: float = 5.0,
    max_closing_speed: float = 1.0,
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
        body_half_extent=body_half_extent,
        spawn_delay_range=spawn_delay_range,
        spawn_interval_s=spawn_interval_s,
        scale_count_with_difficulty=scale_count_with_difficulty,
        curriculum_env_fraction=curriculum_env_fraction,
        exact_curriculum_env_fraction=exact_curriculum_env_fraction,
        auto_switch_success=auto_switch_success,
        max_switch_iteration=max_switch_iteration,
        env_steps_per_iteration=env_steps_per_iteration,
        switch_check_interval=switch_check_interval,
        graded_curriculum=graded_curriculum,
        initial_difficulty=initial_difficulty,
        difficulty_step=difficulty_step,
        advance_success=advance_success,
        demote_success=demote_success,
        easy_speed_range=easy_speed_range,
        easy_min_start_dist=easy_min_start_dist,
        easy_max_start_dist=easy_max_start_dist,
        fraction_range=fraction_range,
        reactive_spawn=reactive_spawn,
        visible_spawn_dist=visible_spawn_dist,
        max_closing_speed=max_closing_speed,
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

    ``exp(-(d - penalty_radius) / falloff)`` reaches 1.0 at the penalty radius and
    decays smoothly beyond it. Inside the radius it is clamped to 2.0 so a single
    close-pass frame cannot produce an outsized spike relative to the other reward
    terms (matching the 2.0 cap used by ``obstacle_approach_penalty``).
    """
    if not obstacle_names:
        return torch.zeros(env.num_envs, device=env.device)
    manager = _get_or_create_dynamic_obstacle_manager(env, obstacle_names=obstacle_names)
    if not manager.enabled:
        return torch.zeros(env.num_envs, device=env.device)
    dist = manager.min_dist_to_robot()
    falloff = max(float(falloff), 1.0e-3)
    proximity = torch.exp(-(dist - penalty_radius) / falloff).clamp(max=2.0)
    return proximity * manager.spawned.float()


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
    return (closing / speed_scale).clamp(max=2.0) * manager.spawned.float()


def obstacle_collision(
    env: "ManagerBasedRLEnv",
    obstacle_names: list[str] | None = None,
    collision_radius: float = 0.3,
    body_half_extent: float | None = None,
    record_failure: bool = False,
) -> torch.Tensor:
    """Terminate when the robot root nears an obstacle *surface*.

    The distance is measured from the obstacle surface (robot-to-obstacle-center
    minus the obstacle's XY half-extent), so ``collision_radius`` is the robot body
    radius at which physical contact is declared — matching the visual size of the
    obstacle cube rather than a generous center-to-center margin.

    When ``record_failure`` is set, colliding episodes are written as failures to
    the legacy SRU success tracker. NavigationEnv records the strict outcome
    separately.
    """
    if not obstacle_names:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    manager = _get_or_create_dynamic_obstacle_manager(env, obstacle_names=obstacle_names)
    if not manager.enabled:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    half_extent = manager.body_half_extent if body_half_extent is None else float(body_half_extent)
    surface_dist = manager.min_dist_to_robot() - half_extent
    termination = (surface_dist < collision_radius) & manager.spawned

    if record_failure and bool(termination.any().item()):
        goal_cmd = getattr(getattr(env, "command_manager", None), "_terms", {}).get("robot_goal", None)
        buffer = getattr(goal_cmd, "goal_reached_buffer", None) if goal_cmd is not None else None
        if buffer is not None:
            env_ids = termination.nonzero(as_tuple=False).squeeze(-1)
            buffer.add(torch.zeros(env.num_envs, dtype=torch.float, device=env.device), env_ids)

    return termination


def obstacle_time_to_collision(
    env: "ManagerBasedRLEnv",
    obstacle_names: list[str] | None = None,
    ttc_horizon: float = 2.0,
    body_half_extent: float | None = None,
) -> torch.Tensor:
    """Return a look-ahead penalty that grows as time-to-collision drops below ``ttc_horizon``.

    ``ttc = surface_distance / closing_speed`` estimates seconds until contact using the
    obstacle's approach speed. The penalty ``((horizon - ttc) / horizon)`` is 0 when the
    robot is not closing (or the obstacle is farther than ``horizon`` seconds away) and
    ramps to 1 as contact becomes imminent — rewarding the policy for slowing/turning
    *before* it enters the proximity radius, unlike the distance-only proximity term.
    """
    if not obstacle_names:
        return torch.zeros(env.num_envs, device=env.device)
    manager = _get_or_create_dynamic_obstacle_manager(env, obstacle_names=obstacle_names)
    if not manager.enabled:
        return torch.zeros(env.num_envs, device=env.device)
    half_extent = manager.body_half_extent if body_half_extent is None else float(body_half_extent)
    surface = (manager.min_dist_to_robot() - half_extent).clamp_min(0.0)
    closing = manager.closing_speed_to_robot()  # >= 0, already clamped
    horizon = max(float(ttc_horizon), 1.0e-3)
    ttc = surface / closing.clamp_min(1.0e-3)
    penalty = ((horizon - ttc) / horizon).clamp(0.0, 1.0)
    # Only count when actually approaching; a zero closing speed yields a huge ttc -> 0.
    penalty = torch.where(closing > 1.0e-3, penalty, torch.zeros_like(penalty))
    return penalty * manager.spawned.float()

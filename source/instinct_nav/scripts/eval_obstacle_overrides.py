"""Apply matched dynamic-obstacle conditions before an evaluation scene is built."""

from __future__ import annotations


def apply_obstacle_eval_overrides(
    env_cfg,
    *,
    difficulty: float | None = None,
    env_fraction: float | None = None,
    height_m: float | None = None,
) -> None:
    if difficulty is None and env_fraction is None and height_m is None:
        return

    events = env_cfg.events
    reset_term = getattr(events, "dynamic_obstacles_reset", None)
    update_term = getattr(events, "dynamic_obstacles_update", None)
    if reset_term is None or update_term is None:
        raise ValueError("Obstacle evaluation overrides require dynamic-obstacle reset and update events")

    if difficulty is not None:
        difficulty = float(difficulty)
        if not 0.0 <= difficulty <= 1.0:
            raise ValueError("--force_obstacle_difficulty must be in [0, 1]")
        env_cfg.obstacle_initial_difficulty = difficulty
        for term in (reset_term, update_term):
            term.params["initial_difficulty"] = difficulty

    if env_fraction is not None:
        env_fraction = float(env_fraction)
        if not 0.0 < env_fraction <= 1.0:
            raise ValueError("--obstacle_env_fraction must be in (0, 1]")
        env_cfg.obstacle_curriculum_env_fraction = env_fraction
        env_cfg.obstacle_fraction_range = (env_fraction, env_fraction)
        for term in (reset_term, update_term):
            term.params["curriculum_env_fraction"] = env_fraction
            term.params["fraction_range"] = (env_fraction, env_fraction)

    if height_m is not None:
        height_m = float(height_m)
        if height_m <= 0.0:
            raise ValueError("--obstacle_height_m must be positive")
        half_height = height_m / 2.0
        env_cfg.obstacle_half_height = half_height
        for term in (reset_term, update_term):
            term.params["obstacle_half_height"] = half_height
        obstacle_names = reset_term.params["obstacle_names"]
        for name in obstacle_names:
            object_cfg = getattr(env_cfg.scene, name)
            if hasattr(object_cfg.spawn, "size"):
                width, depth, _ = object_cfg.spawn.size
                object_cfg.spawn.size = (width, depth, height_m)
            elif hasattr(object_cfg.spawn, "radius") and hasattr(object_cfg.spawn, "height"):
                object_cfg.spawn.height = height_m
            else:
                raise ValueError(f"Unsupported obstacle geometry: {type(object_cfg.spawn).__name__}")
            x, y, _ = object_cfg.init_state.pos
            object_cfg.init_state.pos = (x, y, half_height)

from __future__ import annotations

from collections.abc import Sequence

import torch


def select_delayed_depth_frames(history: Sequence[torch.Tensor], time_lags: torch.Tensor | None) -> torch.Tensor:
    """Select per-env delayed depth frames from chronological noisy-depth history.

    This mirrors Isaac Lab's DelayBuffer behavior for visualization without appending
    to the policy delay buffer: if a requested lag is older than available history,
    return the latest frame for that environment.
    """
    if len(history) == 0:
        raise ValueError("Cannot select delayed depth frames from empty history.")

    latest = history[-1]
    if time_lags is None:
        return latest.detach().clone()

    lags = time_lags.to(device=latest.device, dtype=torch.long).flatten()
    if int(lags.numel()) != int(latest.shape[0]):
        raise ValueError(f"time_lags length {int(lags.numel())} does not match batch size {int(latest.shape[0])}.")

    selected = latest.detach().clone()
    history_len = len(history)
    for env_id, lag in enumerate(lags.tolist()):
        if 0 <= int(lag) < history_len:
            selected[env_id] = history[history_len - 1 - int(lag)][env_id]
    return selected


def _episode_id_value(snapshot: dict[str, object], env_id: int) -> int | None:
    episode_ids = snapshot.get("episode_ids")
    if episode_ids is None:
        return None
    value = episode_ids[env_id]
    if hasattr(value, "item"):
        return int(value.item())
    return int(value)


def select_episode_history_for_env(
    snapshots: Sequence[dict[str, object]],
    observations: Sequence[object] | None,
    env_id: int,
    reference_snapshot: dict[str, object],
) -> tuple[list[dict[str, object]], list[object]]:
    """Return only history frames from the env's current episode.

    GIF histories are collected globally by step.  If an env resets while other
    envs continue, old frames for that env can remain in the window and make the
    replay appear to teleport.  The per-env episode id tags split that history at
    reset boundaries while keeping optional observation caches aligned.
    """
    selected_snapshots = list(snapshots)
    if not selected_snapshots:
        selected_snapshots = [reference_snapshot]

    selected_observations = list(observations or [])
    if selected_observations:
        selected_observations = [None] * max(0, len(selected_snapshots) - len(selected_observations)) + selected_observations
        selected_observations = selected_observations[-len(selected_snapshots):]
    else:
        selected_observations = [None] * len(selected_snapshots)

    reference_episode_id = _episode_id_value(reference_snapshot, env_id)
    if reference_episode_id is None:
        return selected_snapshots, selected_observations

    filtered_pairs = []
    for snapshot, observation in zip(selected_snapshots, selected_observations):
        episode_id = _episode_id_value(snapshot, env_id)
        if episode_id is None or episode_id == reference_episode_id:
            filtered_pairs.append((snapshot, observation))

    if not filtered_pairs:
        filtered_pairs = [(reference_snapshot, selected_observations[-1] if selected_observations else None)]

    filtered_snapshots, filtered_observations = zip(*filtered_pairs)
    return list(filtered_snapshots), list(filtered_observations)


def rear_gravity_aligned_camera_pose(
    root_pos,
    yaw: float,
    height_m: float = 5.0,
    distance_m: float = 4.0,
    lateral_m: float = 1.5,
    target_height_m: float = 0.5,
):
    root = torch.as_tensor(root_pos, dtype=torch.float32)
    forward = torch.tensor([torch.cos(torch.tensor(yaw)), torch.sin(torch.tensor(yaw)), 0.0], dtype=torch.float32)
    left = torch.tensor([-forward[1], forward[0], 0.0], dtype=torch.float32)
    eye = root - float(distance_m) * forward - float(lateral_m) * left
    eye[2] = root[2] + float(height_m)
    target = root.clone()
    target[2] = root[2] + float(target_height_m)
    return eye, target

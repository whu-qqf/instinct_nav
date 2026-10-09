"""Training-only targets from observed obstacle identities and actual future states."""
import torch


def future_occupancy_targets(states, dones, times, envs, horizon_steps, grid_size=32,
                             size_m=6.4, radius_m=0.25):
    """State layout: robot x/y/yaw, then per-slot x/y/active/source-visible.

    Rasterize only objects visible at the source time, in that source robot frame.
    Ignore horizons crossing episode resets or any tracked slot's despawn/recycle.
    No velocities or hypothetical constant-velocity futures are used.
    """
    source = states[times, envs]
    slots = source[:, 3:].reshape(source.shape[0], -1, 4)
    tracked = (slots[..., 2] > 0.5) & (slots[..., 3] > 0.5)
    axis = size_m / 2 - (torch.arange(grid_size, device=states.device) + 0.5) * size_m / grid_size
    x = axis.view(1, 1, grid_size, 1)
    y = axis.view(1, 1, 1, grid_size)
    cos, sin = source[:, 2].cos()[:, None], source[:, 2].sin()[:, None]
    targets, valid = [], []
    for horizon in horizon_steps:
        alive = tracked.clone()
        not_reset = torch.ones_like(times, dtype=torch.bool)
        for step in range(horizon + 1):
            state = states[times + step, envs, 3:].reshape_as(slots)
            alive &= state[..., 2] > 0.5
            if step < horizon:
                not_reset &= ~dones[times + step, envs].reshape(-1).bool()
        future = states[times + horizon, envs, 3:].reshape_as(slots)
        dx = future[..., 0] - source[:, 0:1]
        dy = future[..., 1] - source[:, 1:2]
        local_x, local_y = cos * dx + sin * dy, -sin * dx + cos * dy
        disk = ((x - local_x[..., None, None]).square()
                + (y - local_y[..., None, None]).square()) <= radius_m**2
        targets.append((disk & alive[..., None, None]).any(1).float())
        valid.append(not_reset & tracked.any(1) & (~tracked | alive).all(1))
    return torch.stack(targets, 1), torch.stack(valid, 1)

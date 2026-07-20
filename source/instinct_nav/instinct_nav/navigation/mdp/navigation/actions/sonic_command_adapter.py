from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SonicCommandAdapter:
    """Convert SRU SE(2) velocity commands into SONIC planner command tensors."""

    num_envs: int
    device: str | torch.device
    max_speed: float = 2.0
    slow_walk_threshold: float = 0.05
    walk_threshold: float = 0.5

    def __post_init__(self):
        self.device = torch.device(self.device)
        self.heading = torch.zeros(self.num_envs, device=self.device)

    def reset(self, env_ids: torch.Tensor | None = None):
        self.heading = self.heading.clone()
        if env_ids is None:
            self.heading.zero_()
        else:
            self.heading[env_ids] = 0.0

    def convert(self, velocity_commands: torch.Tensor, dt: float) -> dict[str, torch.Tensor]:
        if velocity_commands.shape != (self.num_envs, 3):
            raise ValueError(
                f"Expected velocity_commands shape {(self.num_envs, 3)}, got {tuple(velocity_commands.shape)}"
            )

        commands = velocity_commands.to(self.device)
        linear_xy = commands[:, :2]
        speed = torch.linalg.norm(linear_xy, dim=1, keepdim=True).clamp(max=self.max_speed)

        direction_xy = torch.zeros_like(linear_xy)
        nonzero = speed.squeeze(-1) > 1.0e-6
        direction_xy[nonzero] = linear_xy[nonzero] / torch.linalg.norm(
            linear_xy[nonzero], dim=1, keepdim=True
        )
        movement_direction = torch.cat((direction_xy, torch.zeros(self.num_envs, 1, device=self.device)), dim=1)

        self.heading = self.heading + commands[:, 2] * dt
        facing_direction = torch.stack(
            (torch.cos(self.heading), torch.sin(self.heading), torch.zeros_like(self.heading)),
            dim=1,
        )

        mode = torch.zeros(self.num_envs, 1, dtype=torch.long, device=self.device)
        mode[(speed[:, 0] >= self.slow_walk_threshold) & (speed[:, 0] < self.walk_threshold), 0] = 1
        mode[speed[:, 0] >= self.walk_threshold, 0] = 2

        return {
            "target_vel": speed,
            "movement_direction": movement_direction,
            "facing_direction": facing_direction,
            "mode": mode,
            "height": torch.full((self.num_envs, 1), -1.0, device=self.device),
        }

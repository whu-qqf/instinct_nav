from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import CommandTerm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from .injected_velocity_command_cfg import InjectedVelocityCommandCfg


class InjectedVelocityCommand(CommandTerm):
    """Velocity command buffer written by a high-level action term."""

    cfg: InjectedVelocityCommandCfg

    def __init__(self, cfg: InjectedVelocityCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.vel_command_b

    def _update_metrics(self):
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        self.vel_command_b[env_ids] = 0.0

    def _update_command(self):
        pass

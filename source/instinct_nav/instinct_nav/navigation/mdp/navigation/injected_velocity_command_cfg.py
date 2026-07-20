from __future__ import annotations

from isaaclab.managers import CommandTermCfg
from isaaclab.utils import configclass

from .injected_velocity_command import InjectedVelocityCommand


@configclass
class InjectedVelocityCommandCfg(CommandTermCfg):
    class_type: type = InjectedVelocityCommand
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)

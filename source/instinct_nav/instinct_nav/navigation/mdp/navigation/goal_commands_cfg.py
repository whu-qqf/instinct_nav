# Copyright (c) 2022-2025, Fan Yang and Per Frivik, ETH Zurich.
# All rights reserved.
#
# SPDX-License-Identifier: MIT


from __future__ import annotations

import math
from dataclasses import MISSING
from typing import TYPE_CHECKING, Literal

from isaaclab.managers import CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg
from .goal_commands import RobotNavigationGoalCommand


"""
Base command generator.
"""

@configclass
class RobotNavigationGoalCommandCfg(CommandTermCfg):
    """Configuration for the robot goal command generator."""

    class_type: type = RobotNavigationGoalCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    robot_to_goal_line_vis: bool = True
    """If true, visualize the line from the robot to the goal."""

    sync_robot_root_to_spawn_origin: bool = False
    """If true, translate the robot root by the same delta when reset resamples env origins."""

    terrain_bank_count: int = 1
    """Number of pre-generated terrain banks. One disables bank switching."""

    terrain_bank_base_num_cols: int = 0
    """Number of terrain columns in one bank. Required when terrain banks are enabled."""

    terrain_bank_switch_interval: int = 1000
    """PPO iterations between terrain-bank switches. Ignored by success-gated banks."""

    terrain_bank_env_steps_per_iteration: int = 16
    """Environment steps collected by one PPO iteration."""

    terrain_bank_advance_success: float = 0.0
    """Mean success required to advance one bank. Values <= 0 disable success gating."""

    terrain_bank_success_check_interval: int = 100
    """Environment steps between success-gate checks."""

    terrain_bank_success_min_full_fraction: float = 0.95
    """Required fraction of environments with a full success window before advancing."""

    report_dynamic_env_success: bool = False
    """Report separate rolling success rates for static and dynamic environments."""

    reset_success_on_dynamic_enable: bool = False
    """Clear rolling success windows once when the dynamic phase starts."""

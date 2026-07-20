from __future__ import annotations

import os
from dataclasses import MISSING
from pathlib import Path

from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from .sonic_navigation_actions import SonicNavigationSE2Action

_PROJECT_ROOT = Path(__file__).resolve().parents[7]


@configclass
class SonicNavigationSE2ActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = SonicNavigationSE2Action
    low_level_decimation: int = 4
    use_raw_actions: bool = True
    scale: list[float] = [1.0, 1.0, 1.0]
    offset: list[float] = [0.0, 0.0, 0.0]
    policy_scaling: list[float] = [1.0, 1.0, 1.0]
    observation_group: str = "low_level_policy"
    low_level_position_action: ActionTermCfg = MISSING
    policy_distr_type: str = "gaussian"
    enable_low_pass_filter: bool = True
    low_pass_filter_alpha: float = 0.5
    max_speed: float = 2.0
    backend: str = "fake"
    planner_path: str | None = None
    encoder_path: str | None = None
    decoder_path: str | None = None
    motionbricks_root: str | None = None
    instinct_checkpoint_path: str | None = None
    instinct_rl_root: str | None = os.environ.get("SRU_INSTINCT_RL_ROOT", str(_PROJECT_ROOT / "external" / "instinct_rl"))
    instinct_depth_mode: str = "height_scan"

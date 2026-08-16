from __future__ import annotations

import os
from dataclasses import MISSING
from pathlib import Path

from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from .instinct_navigation_actions import InstinctFrozenNavigationSE2Action

_PROJECT_ROOT = Path(__file__).resolve().parents[7]


@configclass
class InstinctFrozenNavigationSE2ActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = InstinctFrozenNavigationSE2Action
    low_level_decimation: int = 4
    use_raw_actions: bool = True
    scale: list[float] = [1.0, 1.0, 1.0]
    offset: list[float] = [0.0, 0.0, 0.0]
    policy_scaling: list[float] = [1.0, 1.0, 1.0]
    command_min: list[float] = [0.0, 0.0, -1.0]
    command_max: list[float] = [1.0, 0.0, 1.0]
    observation_group: str = "instinct_low_level_policy"
    command_name: str = "base_velocity"
    low_level_position_action: ActionTermCfg = MISSING
    policy_distr_type: str = "gaussian"
    enable_low_pass_filter: bool = True
    low_pass_filter_alpha: float = 0.5
    checkpoint_path: str = os.environ.get(
        "SRU_G1_INSTINCT_CHECKPOINT",
        str(_PROJECT_ROOT / "artifacts" / "low_level" / "g1_parkour" / "model_30000.pt"),
    )
    instinct_rl_root: str | None = os.environ.get("SRU_INSTINCT_RL_ROOT", str(_PROJECT_ROOT / "external" / "instinct_rl"))
    backend_type: str = "checkpoint"
    actor_onnx_path: str | None = None
    depth_encoder_onnx_path: str | None = None

# Copyright (c) 2022-2025, Fan Yang and Per Frivik, ETH Zurich.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Unitree G1 SONIC navigation task registrations."""

import gymnasium as gym

from . import agents, navigation_env_cfg

gym.register(
    id="Isaac-Nav-PPO-G1-Sonic-v0",
    entry_point="instinct_nav.navigation:NavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.G1SonicNavigationEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.G1SonicNavPPOVisMapRunnerCfg,
    },
)

gym.register(
    id="Isaac-Nav-PPO-G1-Sonic-Play-v0",
    entry_point="instinct_nav.navigation:NavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.G1SonicNavigationEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.G1SonicNavPPOVisMapRunnerCfg,
    },
)

gym.register(
    id="Isaac-Nav-PPO-G1-Sonic-Dev-v0",
    entry_point="instinct_nav.navigation:NavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.G1SonicNavigationEnvCfg_DEV,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.G1SonicNavPPOVisMapRunnerDevCfg,
    },
)

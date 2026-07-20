# Copyright (c) 2022-2025, Fan Yang and Per Frivik, ETH Zurich.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Go2-specific configuration for navigation tasks."""

import gymnasium as gym

from . import agents, navigation_env_cfg

##
# Register Gym environments.
##

##############################################################################################################
# MDPO

gym.register(
    id="Isaac-Nav-MDPO-Go2-v0",
    entry_point="instinct_nav.navigation:NavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.Go2NavigationEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.Go2NavMDPOVisMapRunnerCfg,
    },
)

gym.register(
    id="Isaac-Nav-MDPO-Go2-Play-v0",
    entry_point="instinct_nav.navigation:NavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.Go2NavigationEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.Go2NavMDPOVisMapRunnerCfg,
    },
)

gym.register(
    id="Isaac-Nav-MDPO-Go2-Dev-v0",
    entry_point="instinct_nav.navigation:NavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.Go2NavigationEnvCfg_DEV,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.Go2NavMDPOVisMapRunnerCfg,
    },
)

######################################################################################
# PPO

gym.register(
    id="Isaac-Nav-PPO-Go2-v0",
    entry_point="instinct_nav.navigation:NavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.Go2NavigationEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.Go2NavPPOVisMapRunnerCfg,
    },
)

gym.register(
    id="Isaac-Nav-PPO-Go2-Play-v0",
    entry_point="instinct_nav.navigation:NavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.Go2NavigationEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.Go2NavPPOVisMapRunnerCfg,
    },
)

gym.register(
    id="Isaac-Nav-PPO-Go2-Dev-v0",
    entry_point="instinct_nav.navigation:NavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": navigation_env_cfg.Go2NavigationEnvCfg_DEV,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.Go2NavPPOVisMapRunnerCfg,
    },
)

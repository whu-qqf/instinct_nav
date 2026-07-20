from __future__ import annotations

import gymnasium as gym

from . import agents, navigation_env_cfg


def _register_navigation_task(task_id: str, env_cfg_entry_point, rsl_rl_cfg_entry_point) -> None:
    gym.register(
        id=task_id,
        entry_point="instinct_nav.navigation:NavigationEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": env_cfg_entry_point,
            "rsl_rl_cfg_entry_point": rsl_rl_cfg_entry_point,
        },
    )


# Official instinct_nav task ids.
_register_navigation_task(
    "Instinct-Nav-G1-SRU-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg,
    agents.rsl_rl_cfg.G1InstinctNavPPOVisMapRunnerCfg,
)
_register_navigation_task(
    "Instinct-Nav-G1-SRU-Dev-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_DEV,
    agents.rsl_rl_cfg.G1InstinctNavPPOVisMapRunnerDevCfg,
)
_register_navigation_task(
    "Instinct-Nav-G1-SRU-Play-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_PLAY,
    agents.rsl_rl_cfg.G1InstinctNavPPOVisMapRunnerCfg,
)

# Visibility-map fusion ablations.
_register_navigation_task(
    "Instinct-Nav-G1-SRU-Visibility-Early-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_VIS,
    agents.rsl_rl_cfg.G1InstinctNavPPOVisibilityEarlyRunnerCfg,
)
_register_navigation_task(
    "Instinct-Nav-G1-SRU-Visibility-Early-Dev-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_VIS_DEV,
    agents.rsl_rl_cfg.G1InstinctNavPPOVisibilityEarlyRunnerDevCfg,
)
_register_navigation_task(
    "Instinct-Nav-G1-SRU-Visibility-Early-Play-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_VIS_PLAY,
    agents.rsl_rl_cfg.G1InstinctNavPPOVisibilityEarlyRunnerCfg,
)
_register_navigation_task(
    "Instinct-Nav-G1-SRU-Visibility-Late-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_VIS,
    agents.rsl_rl_cfg.G1InstinctNavPPOVisibilityLateRunnerCfg,
)
_register_navigation_task(
    "Instinct-Nav-G1-SRU-Visibility-Late-Dev-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_VIS_DEV,
    agents.rsl_rl_cfg.G1InstinctNavPPOVisibilityLateRunnerDevCfg,
)
_register_navigation_task(
    "Instinct-Nav-G1-SRU-Visibility-Late-Play-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_VIS_PLAY,
    agents.rsl_rl_cfg.G1InstinctNavPPOVisibilityLateRunnerCfg,
)
_register_navigation_task(
    "Instinct-Nav-G1-SRU-Visibility-Query-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_VIS,
    agents.rsl_rl_cfg.G1InstinctNavPPOVisibilityQueryRunnerCfg,
)
_register_navigation_task(
    "Instinct-Nav-G1-SRU-Visibility-Query-Dev-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_VIS_DEV,
    agents.rsl_rl_cfg.G1InstinctNavPPOVisibilityQueryRunnerDevCfg,
)
_register_navigation_task(
    "Instinct-Nav-G1-SRU-Visibility-Query-Play-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_VIS_PLAY,
    agents.rsl_rl_cfg.G1InstinctNavPPOVisibilityQueryRunnerCfg,
)

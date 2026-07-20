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

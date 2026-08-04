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

# Scene-token fusion ablation. This uses the same local visibility memory as the
# visibility-map runs, but exposes it as fixed 3D scene tokens for query attention.
_register_navigation_task(
    "Instinct-Nav-G1-SRU-SceneToken-Query-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_SCENE_TOKEN,
    agents.rsl_rl_cfg.G1InstinctNavPPOSceneTokenQueryRunnerCfg,
)
_register_navigation_task(
    "Instinct-Nav-G1-SRU-SceneToken-Query-Dev-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_SCENE_TOKEN_DEV,
    agents.rsl_rl_cfg.G1InstinctNavPPOSceneTokenQueryRunnerDevCfg,
)
_register_navigation_task(
    "Instinct-Nav-G1-SRU-SceneToken-Query-Play-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_SCENE_TOKEN_PLAY,
    agents.rsl_rl_cfg.G1InstinctNavPPOSceneTokenQueryRunnerCfg,
)

# Scene-token v2: 9D terrain-risk tokens are provided to both actor and critic.

_register_navigation_task(
    "Instinct-Nav-G1-SRU-SceneTokenRisk-Query-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_SCENE_TOKEN_RISK,
    agents.rsl_rl_cfg.G1InstinctNavPPOSceneTokenRiskQueryRunnerCfg,
)
_register_navigation_task(
    "Instinct-Nav-G1-SRU-SceneTokenRisk-Query-Dev-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_SCENE_TOKEN_RISK_DEV,
    agents.rsl_rl_cfg.G1InstinctNavPPOSceneTokenRiskQueryRunnerDevCfg,
)
_register_navigation_task(
    "Instinct-Nav-G1-SRU-SceneTokenRisk-Query-Play-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_SCENE_TOKEN_RISK_PLAY,
    agents.rsl_rl_cfg.G1InstinctNavPPOSceneTokenRiskQueryRunnerCfg,
)


# Scene-token v3: 6D terrain-risk tokens without per-token goal-relative features.
_register_navigation_task(
    "Instinct-Nav-G1-SRU-SceneTokenRiskNoGoal-Query-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_SCENE_TOKEN_RISK_NOGOAL,
    agents.rsl_rl_cfg.G1InstinctNavPPOSceneTokenRiskNoGoalQueryRunnerCfg,
)
_register_navigation_task(
    "Instinct-Nav-G1-SRU-SceneTokenRiskNoGoal-Query-Dev-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_SCENE_TOKEN_RISK_NOGOAL_DEV,
    agents.rsl_rl_cfg.G1InstinctNavPPOSceneTokenRiskNoGoalQueryRunnerDevCfg,
)
_register_navigation_task(
    "Instinct-Nav-G1-SRU-SceneTokenRiskNoGoal-Query-Play-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_SCENE_TOKEN_RISK_NOGOAL_PLAY,
    agents.rsl_rl_cfg.G1InstinctNavPPOSceneTokenRiskNoGoalQueryRunnerCfg,
)

# Scene-token v4: reuse the frozen low-level policy camera as the high-level SRU depth/token source.
_register_navigation_task(
    "Instinct-Nav-G1-SRU-SceneTokenRiskNoGoal-SharedLowCam-Query-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_SCENE_TOKEN_RISK_NOGOAL_SHARED_LOWCAM,
    agents.rsl_rl_cfg.G1InstinctNavPPOSceneTokenRiskNoGoalSharedLowCamQueryRunnerCfg,
)
_register_navigation_task(
    "Instinct-Nav-G1-SRU-SceneTokenRiskNoGoal-SharedLowCam-Query-Dev-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_SCENE_TOKEN_RISK_NOGOAL_SHARED_LOWCAM_DEV,
    agents.rsl_rl_cfg.G1InstinctNavPPOSceneTokenRiskNoGoalSharedLowCamQueryRunnerDevCfg,
)
_register_navigation_task(
    "Instinct-Nav-G1-SRU-SceneTokenRiskNoGoal-SharedLowCam-Query-Play-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_SCENE_TOKEN_RISK_NOGOAL_SHARED_LOWCAM_PLAY,
    agents.rsl_rl_cfg.G1InstinctNavPPOSceneTokenRiskNoGoalSharedLowCamQueryRunnerCfg,
)

# Final retraining task: current SRU architecture with the official stable parkour backend.
_register_navigation_task(
    "Instinct-Nav-G1-SRU-SceneTokenRiskNoGoal-SharedLowCam-OfficialParkour-Query-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_SCENE_TOKEN_RISK_NOGOAL_SHARED_LOWCAM_OFFICIAL,
    agents.rsl_rl_cfg.G1InstinctNavPPOSceneTokenRiskNoGoalSharedLowCamOfficialRunnerCfg,
)
_register_navigation_task(
    "Instinct-Nav-G1-SRU-SceneTokenRiskNoGoal-SharedLowCam-OfficialParkour-Query-Dev-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_SCENE_TOKEN_RISK_NOGOAL_SHARED_LOWCAM_OFFICIAL_DEV,
    agents.rsl_rl_cfg.G1InstinctNavPPOSceneTokenRiskNoGoalSharedLowCamOfficialRunnerDevCfg,
)
_register_navigation_task(
    "Instinct-Nav-G1-SRU-SceneTokenRiskNoGoal-SharedLowCam-OfficialParkour-Query-Play-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_SCENE_TOKEN_RISK_NOGOAL_SHARED_LOWCAM_OFFICIAL_PLAY,
    agents.rsl_rl_cfg.G1InstinctNavPPOSceneTokenRiskNoGoalSharedLowCamOfficialRunnerCfg,
)

# Continuous confidence visibility-map ablation; legacy binary tasks stay unchanged.
_register_navigation_task(
    "Instinct-Nav-G1-SRU-SceneTokenScoreVis-SharedLowCam-OfficialParkour-Query-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_SCENE_TOKEN_SCORE_VIS_SHARED_LOWCAM_OFFICIAL,
    agents.rsl_rl_cfg.G1InstinctNavPPOSceneTokenScoreVisOfficialRunnerCfg,
)

_register_navigation_task(
    "Instinct-Nav-G1-SRU-SceneTokenScoreVis-SharedLowCam-OfficialParkour-Query-Dev-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_SCENE_TOKEN_SCORE_VIS_SHARED_LOWCAM_OFFICIAL_DEV,
    agents.rsl_rl_cfg.G1InstinctNavPPOSceneTokenScoreVisOfficialRunnerDevCfg,
)

_register_navigation_task(
    "Instinct-Nav-G1-SRU-SceneTokenScoreVis-SharedLowCam-OfficialParkour-Query-Play-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_SCENE_TOKEN_SCORE_VIS_SHARED_LOWCAM_OFFICIAL_PLAY,
    agents.rsl_rl_cfg.G1InstinctNavPPOSceneTokenScoreVisOfficialRunnerCfg,
)

# Stage-3 dynamic obstacle curriculum (ScoreVis + moving obstacles).
_register_navigation_task(
    "Instinct-Nav-G1-SRU-SceneTokenScoreVis-DynamicObstacle-OfficialParkour-Query-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_DYNAMIC_OFFICIAL,
    agents.rsl_rl_cfg.G1InstinctNavPPOSceneTokenScoreVisDynamicOfficialRunnerCfg,
)
_register_navigation_task(
    "Instinct-Nav-G1-SRU-SceneTokenScoreVis-DynamicObstacle-OfficialParkour-Query-Dev-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_DYNAMIC_OFFICIAL_DEV,
    agents.rsl_rl_cfg.G1InstinctNavPPOSceneTokenScoreVisDynamicOfficialRunnerDevCfg,
)
_register_navigation_task(
    "Instinct-Nav-G1-SRU-SceneTokenScoreVis-DynamicObstacle-OfficialParkour-Query-Play-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_DYNAMIC_OFFICIAL_PLAY,
    agents.rsl_rl_cfg.G1InstinctNavPPOSceneTokenScoreVisDynamicOfficialRunnerCfg,
)

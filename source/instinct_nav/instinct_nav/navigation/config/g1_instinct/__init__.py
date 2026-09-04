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

# Learned dynamic-map tokens: current frame queries ScoreVis history before SRU query fusion.
_register_navigation_task(
    "Instinct-Nav-G1-SRU-DynamicMapCross-DynamicObstacle-OfficialParkour-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_DYNAMIC_MAP_CROSS_OFFICIAL,
    agents.rsl_rl_cfg.G1InstinctNavPPODynamicMapCrossOfficialRunnerCfg,
)
_register_navigation_task(
    "Instinct-Nav-G1-SRU-DynamicMapCross-DynamicObstacle-OfficialParkour-Dev-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_DYNAMIC_MAP_CROSS_OFFICIAL_DEV,
    agents.rsl_rl_cfg.G1InstinctNavPPODynamicMapCrossOfficialRunnerDevCfg,
)
_register_navigation_task(
    "Instinct-Nav-G1-SRU-DynamicMapCross-DynamicObstacle-OfficialParkour-Play-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_DYNAMIC_MAP_CROSS_OFFICIAL_PLAY,
    agents.rsl_rl_cfg.G1InstinctNavPPODynamicMapCrossOfficialRunnerCfg,
)

# DynamicMapCross ablation on the expanded 40 m rich-terrain distribution.
_register_navigation_task(
    "Instinct-Nav-G1-SRU-DynamicMapCross-RichTerrain-DynamicObstacle-OfficialParkour-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_DYNAMIC_MAP_CROSS_RICH_TERRAIN_OFFICIAL,
    agents.rsl_rl_cfg.G1InstinctNavPPODynamicMapCrossOfficialRunnerCfg,
)
_register_navigation_task(
    "Instinct-Nav-G1-SRU-DynamicMapCross-RichTerrain-DynamicObstacle-OfficialParkour-Dev-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_DYNAMIC_MAP_CROSS_RICH_TERRAIN_OFFICIAL_DEV,
    agents.rsl_rl_cfg.G1InstinctNavPPODynamicMapCrossOfficialRunnerDevCfg,
)
_register_navigation_task(
    "Instinct-Nav-G1-SRU-DynamicMapCross-RichTerrain-DynamicObstacle-OfficialParkour-Play-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_DYNAMIC_MAP_CROSS_RICH_TERRAIN_OFFICIAL_PLAY,
    agents.rsl_rl_cfg.G1InstinctNavPPODynamicMapCrossOfficialRunnerCfg,
)

# rich_terrain_v4 + velocity command clipped to the frozen low-level's trained envelope
# (vx<=0.8, gaussian_clip instead of tanh) to reduce out-of-distribution overspeed -> large pitch.
_register_navigation_task(
    "Instinct-Nav-G1-SRU-DynamicMapCross-RichTerrain-SpeedClip-OfficialParkour-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_DYNAMIC_MAP_CROSS_RICH_TERRAIN_OFFICIAL_SPEEDCLIP,
    agents.rsl_rl_cfg.G1InstinctNavPPODynamicMapCrossOfficialRunnerCfg,
)
_register_navigation_task(
    "Instinct-Nav-G1-SRU-DynamicMapCross-RichTerrain-SpeedClip-OfficialParkour-Dev-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_DYNAMIC_MAP_CROSS_RICH_TERRAIN_OFFICIAL_SPEEDCLIP_DEV,
    agents.rsl_rl_cfg.G1InstinctNavPPODynamicMapCrossOfficialRunnerDevCfg,
)
_register_navigation_task(
    "Instinct-Nav-G1-SRU-DynamicMapCross-RichTerrain-SpeedClip-OfficialParkour-Play-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_DYNAMIC_MAP_CROSS_RICH_TERRAIN_OFFICIAL_SPEEDCLIP_PLAY,
    agents.rsl_rl_cfg.G1InstinctNavPPODynamicMapCrossOfficialRunnerCfg,
)

# SpeedClip + privileged stair-centerline guidance reward: teaches the robot to approach a
# staircase front-and-centered and ascend up the middle instead of mounting from an exposed
# lateral edge. Reward is 0 off stairs (no false positives on flat ground / platforms).
_register_navigation_task(
    "Instinct-Nav-G1-SRU-DynamicMapCross-RichTerrain-SpeedClip-Centerline-OfficialParkour-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_DYNAMIC_MAP_CROSS_RICH_TERRAIN_OFFICIAL_SPEEDCLIP_CENTERLINE,
    agents.rsl_rl_cfg.G1InstinctNavPPODynamicMapCrossOfficialRunnerCfg,
)
_register_navigation_task(
    "Instinct-Nav-G1-SRU-DynamicMapCross-RichTerrain-SpeedClip-Centerline-OfficialParkour-Dev-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_DYNAMIC_MAP_CROSS_RICH_TERRAIN_OFFICIAL_SPEEDCLIP_CENTERLINE_DEV,
    agents.rsl_rl_cfg.G1InstinctNavPPODynamicMapCrossOfficialRunnerDevCfg,
)
_register_navigation_task(
    "Instinct-Nav-G1-SRU-DynamicMapCross-RichTerrain-SpeedClip-Centerline-OfficialParkour-Play-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_DYNAMIC_MAP_CROSS_RICH_TERRAIN_OFFICIAL_SPEEDCLIP_CENTERLINE_PLAY,
    agents.rsl_rl_cfg.G1InstinctNavPPODynamicMapCrossOfficialRunnerCfg,
)

# Ablation: original v4 tanh velocity mapping (vx up to 1.0) + centerline reward, NO SpeedClip.
# Differs from the SpeedClip-Centerline task only in the velocity mapping -> isolates whether
# large_pitch is driven by the speed range/mapping.
_register_navigation_task(
    "Instinct-Nav-G1-SRU-DynamicMapCross-RichTerrain-Centerline-OfficialParkour-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_DYNAMIC_MAP_CROSS_RICH_TERRAIN_OFFICIAL_CENTERLINE,
    agents.rsl_rl_cfg.G1InstinctNavPPODynamicMapCrossOfficialRunnerCfg,
)
_register_navigation_task(
    "Instinct-Nav-G1-SRU-DynamicMapCross-RichTerrain-Centerline-OfficialParkour-Dev-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_DYNAMIC_MAP_CROSS_RICH_TERRAIN_OFFICIAL_CENTERLINE_DEV,
    agents.rsl_rl_cfg.G1InstinctNavPPODynamicMapCrossOfficialRunnerDevCfg,
)
_register_navigation_task(
    "Instinct-Nav-G1-SRU-DynamicMapCross-RichTerrain-Centerline-OfficialParkour-Play-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_DYNAMIC_MAP_CROSS_RICH_TERRAIN_OFFICIAL_CENTERLINE_PLAY,
    agents.rsl_rl_cfg.G1InstinctNavPPODynamicMapCrossOfficialRunnerCfg,
)

# Ablation: tanh centerline + penalty for driving toward not-yet-observed ("black")
# cells of the vis_score visibility map (discourages blind/occluded-direction motion).
_register_navigation_task(
    "Instinct-Nav-G1-SRU-DynamicMapCross-RichTerrain-Centerline-Unobserved-OfficialParkour-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_DYNAMIC_MAP_CROSS_RICH_TERRAIN_OFFICIAL_CENTERLINE_UNOBSERVED,
    agents.rsl_rl_cfg.G1InstinctNavPPODynamicMapCrossOfficialRunnerCfg,
)
_register_navigation_task(
    "Instinct-Nav-G1-SRU-DynamicMapCross-RichTerrain-Centerline-Unobserved-OfficialParkour-Dev-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_DYNAMIC_MAP_CROSS_RICH_TERRAIN_OFFICIAL_CENTERLINE_UNOBSERVED_DEV,
    agents.rsl_rl_cfg.G1InstinctNavPPODynamicMapCrossOfficialRunnerDevCfg,
)
_register_navigation_task(
    "Instinct-Nav-G1-SRU-DynamicMapCross-RichTerrain-Centerline-Unobserved-OfficialParkour-Play-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_DYNAMIC_MAP_CROSS_RICH_TERRAIN_OFFICIAL_CENTERLINE_UNOBSERVED_PLAY,
    agents.rsl_rl_cfg.G1InstinctNavPPODynamicMapCrossOfficialRunnerCfg,
)

# Ablation: Centerline-Unobserved + rebalanced avoidance reward (P0) + capability-driven
# graded obstacle curriculum (P1). Isolated via graded_curriculum flag; all other tasks
# keep the legacy binary iteration switch and inert (weight-0) TTC term.
_register_navigation_task(
    "Instinct-Nav-G1-SRU-DynamicMapCross-RichTerrain-Centerline-Unobserved-Avoid-OfficialParkour-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_DYNAMIC_MAP_CROSS_RICH_TERRAIN_OFFICIAL_CENTERLINE_UNOBSERVED_AVOID,
    agents.rsl_rl_cfg.G1InstinctNavPPODynamicMapCrossOfficialRunnerCfg,
)
_register_navigation_task(
    "Instinct-Nav-G1-SRU-DynamicMapCross-RichTerrain-Centerline-Unobserved-Avoid-OfficialParkour-Dev-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_DYNAMIC_MAP_CROSS_RICH_TERRAIN_OFFICIAL_CENTERLINE_UNOBSERVED_AVOID_DEV,
    agents.rsl_rl_cfg.G1InstinctNavPPODynamicMapCrossOfficialRunnerDevCfg,
)
_register_navigation_task(
    "Instinct-Nav-G1-SRU-DynamicMapCross-RichTerrain-Centerline-Unobserved-Avoid-OfficialParkour-Play-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_DYNAMIC_MAP_CROSS_RICH_TERRAIN_OFFICIAL_CENTERLINE_UNOBSERVED_AVOID_PLAY,
    agents.rsl_rl_cfg.G1InstinctNavPPODynamicMapCrossOfficialRunnerCfg,
)

# Same network/rewards as Centerline-Unobserved, with four pre-generated terrain
# banks switched every 1000 PPO iterations at episode-reset boundaries.
_register_navigation_task(
    "Instinct-Nav-G1-SRU-DynamicMapCross-RichTerrain-Centerline-Unobserved-Banks-OfficialParkour-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_DYNAMIC_MAP_CROSS_RICH_TERRAIN_OFFICIAL_CENTERLINE_UNOBSERVED_BANKS,
    agents.rsl_rl_cfg.G1InstinctNavPPODynamicMapCrossBanksOfficialRunnerCfg,
)
_register_navigation_task(
    "Instinct-Nav-G1-SRU-DynamicMapCross-RichTerrain-Centerline-Unobserved-Banks-OfficialParkour-Dev-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_DYNAMIC_MAP_CROSS_RICH_TERRAIN_OFFICIAL_CENTERLINE_UNOBSERVED_BANKS_DEV,
    agents.rsl_rl_cfg.G1InstinctNavPPODynamicMapCrossBanksOfficialRunnerDevCfg,
)
_register_navigation_task(
    "Instinct-Nav-G1-SRU-DynamicMapCross-RichTerrain-Centerline-Unobserved-Banks-OfficialParkour-Play-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_DYNAMIC_MAP_CROSS_RICH_TERRAIN_OFFICIAL_CENTERLINE_UNOBSERVED_BANKS_PLAY,
    agents.rsl_rl_cfg.G1InstinctNavPPODynamicMapCrossBanksOfficialRunnerCfg,
)

# Avoid ablation (rebalanced reward + graded obstacle curriculum) with four terrain
# banks switched every 1000 PPO iterations.
_register_navigation_task(
    "Instinct-Nav-G1-SRU-DynamicMapCross-RichTerrain-Centerline-Unobserved-Avoid-Banks-OfficialParkour-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_DYNAMIC_MAP_CROSS_RICH_TERRAIN_OFFICIAL_CENTERLINE_UNOBSERVED_AVOID_BANKS,
    agents.rsl_rl_cfg.G1InstinctNavPPODynamicMapCrossBanksOfficialRunnerCfg,
)
_register_navigation_task(
    "Instinct-Nav-G1-SRU-DynamicMapCross-RichTerrain-Centerline-Unobserved-Avoid-Banks-OfficialParkour-Dev-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_DYNAMIC_MAP_CROSS_RICH_TERRAIN_OFFICIAL_CENTERLINE_UNOBSERVED_AVOID_BANKS_DEV,
    agents.rsl_rl_cfg.G1InstinctNavPPODynamicMapCrossBanksOfficialRunnerDevCfg,
)
_register_navigation_task(
    "Instinct-Nav-G1-SRU-DynamicMapCross-RichTerrain-Centerline-Unobserved-Avoid-Banks-OfficialParkour-Play-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_DYNAMIC_MAP_CROSS_RICH_TERRAIN_OFFICIAL_CENTERLINE_UNOBSERVED_AVOID_BANKS_PLAY,
    agents.rsl_rl_cfg.G1InstinctNavPPODynamicMapCrossBanksOfficialRunnerCfg,
)

# AvoidReact ablation: *-Avoid- + reactive visible-approach spawn (P2) + collision-aware
# success metric/curriculum. All new behavior is flag-gated (default off) so every other
# task stays byte-identical. Reuses the same runner cfgs as the *-Avoid- family.
_register_navigation_task(
    "Instinct-Nav-G1-SRU-DynamicMapCross-RichTerrain-Centerline-Unobserved-AvoidReact-OfficialParkour-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_DYNAMIC_MAP_CROSS_RICH_TERRAIN_OFFICIAL_CENTERLINE_UNOBSERVED_AVOID_REACT,
    agents.rsl_rl_cfg.G1InstinctNavPPODynamicMapCrossOfficialRunnerCfg,
)
_register_navigation_task(
    "Instinct-Nav-G1-SRU-DynamicMapCross-RichTerrain-Centerline-Unobserved-AvoidReact-OfficialParkour-Dev-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_DYNAMIC_MAP_CROSS_RICH_TERRAIN_OFFICIAL_CENTERLINE_UNOBSERVED_AVOID_REACT_DEV,
    agents.rsl_rl_cfg.G1InstinctNavPPODynamicMapCrossOfficialRunnerDevCfg,
)
_register_navigation_task(
    "Instinct-Nav-G1-SRU-DynamicMapCross-RichTerrain-Centerline-Unobserved-AvoidReact-OfficialParkour-Play-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_DYNAMIC_MAP_CROSS_RICH_TERRAIN_OFFICIAL_CENTERLINE_UNOBSERVED_AVOID_REACT_PLAY,
    agents.rsl_rl_cfg.G1InstinctNavPPODynamicMapCrossOfficialRunnerCfg,
)
_register_navigation_task(
    "Instinct-Nav-G1-SRU-DynamicMapCross-RichTerrain-Centerline-Unobserved-AvoidReact-Banks-OfficialParkour-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_DYNAMIC_MAP_CROSS_RICH_TERRAIN_OFFICIAL_CENTERLINE_UNOBSERVED_AVOID_REACT_BANKS,
    agents.rsl_rl_cfg.G1InstinctNavPPODynamicMapCrossBanksOfficialRunnerCfg,
)
_register_navigation_task(
    "Instinct-Nav-G1-SRU-DynamicMapCross-RichTerrain-Centerline-Unobserved-AvoidReact-Banks-OfficialParkour-Dev-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_DYNAMIC_MAP_CROSS_RICH_TERRAIN_OFFICIAL_CENTERLINE_UNOBSERVED_AVOID_REACT_BANKS_DEV,
    agents.rsl_rl_cfg.G1InstinctNavPPODynamicMapCrossBanksOfficialRunnerDevCfg,
)
_register_navigation_task(
    "Instinct-Nav-G1-SRU-DynamicMapCross-RichTerrain-Centerline-Unobserved-AvoidReact-Banks-OfficialParkour-Play-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_DYNAMIC_MAP_CROSS_RICH_TERRAIN_OFFICIAL_CENTERLINE_UNOBSERVED_AVOID_REACT_BANKS_PLAY,
    agents.rsl_rl_cfg.G1InstinctNavPPODynamicMapCrossBanksOfficialRunnerCfg,
)

# AvoidReact-Banks + bidirectional (promote/demote) difficulty curriculum. Only overrides
# obstacle_demote_success/advance_success; reuses the same runner cfgs.
_register_navigation_task(
    "Instinct-Nav-G1-SRU-DynamicMapCross-RichTerrain-Centerline-Unobserved-AvoidReact-Demote-Banks-OfficialParkour-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_DYNAMIC_MAP_CROSS_RICH_TERRAIN_OFFICIAL_CENTERLINE_UNOBSERVED_AVOID_REACT_DEMOTE_BANKS,
    agents.rsl_rl_cfg.G1InstinctNavPPODynamicMapCrossBanksOfficialRunnerCfg,
)
_register_navigation_task(
    "Instinct-Nav-G1-SRU-DynamicMapCross-RichTerrain-Centerline-Unobserved-AvoidReact-Demote-Banks-OfficialParkour-Dev-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_DYNAMIC_MAP_CROSS_RICH_TERRAIN_OFFICIAL_CENTERLINE_UNOBSERVED_AVOID_REACT_DEMOTE_BANKS_DEV,
    agents.rsl_rl_cfg.G1InstinctNavPPODynamicMapCrossBanksOfficialRunnerDevCfg,
)
_register_navigation_task(
    "Instinct-Nav-G1-SRU-DynamicMapCross-RichTerrain-Centerline-Unobserved-AvoidReact-Demote-Banks-OfficialParkour-Play-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_DYNAMIC_MAP_CROSS_RICH_TERRAIN_OFFICIAL_CENTERLINE_UNOBSERVED_AVOID_REACT_DEMOTE_BANKS_PLAY,
    agents.rsl_rl_cfg.G1InstinctNavPPODynamicMapCrossBanksOfficialRunnerCfg,
)

# AvoidReact-Banks + dynamic-obstacle segmentation mask head (raw delayed depth -> mask
# CNN -> SRU feature) trained with an auxiliary BCE+Dice mask loss. Uses the dedicated
# DynamicMask runner cfg (use_dynamic_mask=True, lambda_mask=0.2).
_register_navigation_task(
    "Instinct-Nav-G1-SRU-DynamicMapCross-RichTerrain-Centerline-Unobserved-AvoidReact-DynamicMask-Banks-OfficialParkour-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_DYNAMIC_MAP_CROSS_RICH_TERRAIN_OFFICIAL_CENTERLINE_UNOBSERVED_AVOID_REACT_DYNAMIC_MASK_BANKS,
    agents.rsl_rl_cfg.G1InstinctNavPPODynamicMaskBanksOfficialRunnerCfg,
)
_register_navigation_task(
    "Instinct-Nav-G1-SRU-DynamicMapCross-RichTerrain-Centerline-Unobserved-AvoidReact-DynamicMask-Banks-OfficialParkour-Dev-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_DYNAMIC_MAP_CROSS_RICH_TERRAIN_OFFICIAL_CENTERLINE_UNOBSERVED_AVOID_REACT_DYNAMIC_MASK_BANKS_DEV,
    agents.rsl_rl_cfg.G1InstinctNavPPODynamicMaskBanksOfficialRunnerDevCfg,
)
_register_navigation_task(
    "Instinct-Nav-G1-SRU-DynamicMapCross-RichTerrain-Centerline-Unobserved-AvoidReact-DynamicMask-Banks-OfficialParkour-Play-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_DYNAMIC_MAP_CROSS_RICH_TERRAIN_OFFICIAL_CENTERLINE_UNOBSERVED_AVOID_REACT_DYNAMIC_MASK_BANKS_PLAY,
    agents.rsl_rl_cfg.G1InstinctNavPPODynamicMaskBanksOfficialRunnerCfg,
)

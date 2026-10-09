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

# Dynamic-scene eval: v3 DynamicMapCross-BEV network + current training's dynamic-obstacle
# generation logic (5 obstacles, 1.6 m tall, full difficulty, 100% envs, staggered patrol).
_register_navigation_task(
    "Instinct-Nav-G1-SRU-DynamicMapCross-RichTerrain-Centerline-Unobserved-DynamicEval-OfficialParkour-Play-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_DYNAMIC_MAP_CROSS_RICH_TERRAIN_OFFICIAL_CENTERLINE_UNOBSERVED_DYNAMIC_EVAL_PLAY,
    agents.rsl_rl_cfg.G1InstinctNavPPODynamicMapCrossOfficialRunnerCfg,
)
_register_navigation_task(
    "Instinct-Nav-G1-SRU-DynamicMapCross-RichTerrain-Centerline-Unobserved-DynamicEvalPhysical-OfficialParkour-Play-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_DYNAMIC_MAP_CROSS_RICH_TERRAIN_OFFICIAL_CENTERLINE_UNOBSERVED_DYNAMIC_EVAL_PHYSICAL_PLAY,
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

# AvoidReact-Banks + 6D DynamicMapCross tokens. Dynamic-mask pixels are
# back-projected with delayed depth into a current-only BEV presence channel.
_register_navigation_task(
    "Instinct-Nav-G1-SRU-DynamicMapCross-RichTerrain-NoCenterline-Unobserved-AvoidReact-DynamicMapMask-Banks-OfficialParkour-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_DYNAMIC_MAP_CROSS_RICH_TERRAIN_OFFICIAL_CENTERLINE_UNOBSERVED_AVOID_REACT_DYNAMIC_MAP_MASK_BANKS,
    agents.rsl_rl_cfg.G1InstinctNavPPODynamicMapMaskBanksOfficialRunnerCfg,
)
_register_navigation_task(
    "Instinct-Nav-G1-SRU-DynamicMapCross-RichTerrain-NoCenterline-Unobserved-AvoidReact-DynamicMapMask-Banks-OfficialParkour-Dev-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_DYNAMIC_MAP_CROSS_RICH_TERRAIN_OFFICIAL_CENTERLINE_UNOBSERVED_AVOID_REACT_DYNAMIC_MAP_MASK_BANKS_DEV,
    agents.rsl_rl_cfg.G1InstinctNavPPODynamicMapMaskBanksOfficialRunnerDevCfg,
)
_register_navigation_task(
    "Instinct-Nav-G1-SRU-DynamicMapCross-RichTerrain-NoCenterline-Unobserved-AvoidReact-DynamicMapMask-Banks-OfficialParkour-Play-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_DYNAMIC_MAP_CROSS_RICH_TERRAIN_OFFICIAL_CENTERLINE_UNOBSERVED_AVOID_REACT_DYNAMIC_MAP_MASK_BANKS_PLAY,
    agents.rsl_rl_cfg.G1InstinctNavPPODynamicMapMaskBanksOfficialRunnerCfg,
)

# DynamicMapMask recurrent-fusion ablation. The same 6D BEV tokens are pooled
# into a dynamic context before both SRUs while retaining late scene attention.
_register_navigation_task(
    "Instinct-Nav-G1-SRU-DynamicMapCross-RichTerrain-NoCenterline-Unobserved-AvoidReact-DynamicMapMask-RecurrentFusion-Banks-OfficialParkour-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_DYNAMIC_MAP_MASK_RECURRENT_FUSION_BANKS,
    agents.rsl_rl_cfg.G1InstinctNavPPODynamicMapMaskRecurrentFusionBanksOfficialRunnerCfg,
)
_register_navigation_task(
    "Instinct-Nav-G1-SRU-DynamicMapCross-RichTerrain-NoCenterline-Unobserved-AvoidReact-DynamicMapMask-RecurrentFusion-Banks-OfficialParkour-Dev-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_DYNAMIC_MAP_MASK_RECURRENT_FUSION_BANKS_DEV,
    agents.rsl_rl_cfg.G1InstinctNavPPODynamicMapMaskRecurrentFusionBanksOfficialRunnerDevCfg,
)
_register_navigation_task(
    "Instinct-Nav-G1-SRU-DynamicMapCross-RichTerrain-NoCenterline-Unobserved-AvoidReact-DynamicMapMask-RecurrentFusion-Banks-OfficialParkour-Play-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_DYNAMIC_MAP_MASK_RECURRENT_FUSION_BANKS_PLAY,
    agents.rsl_rl_cfg.G1InstinctNavPPODynamicMapMaskRecurrentFusionBanksOfficialRunnerCfg,
)

# AvoidReact-Banks + direct 9x16 dynamic mask observation. Simulation supplies a
# delayed, YOLO-corrupted mask; the policy consumes it without a learned mask CNN.
_register_navigation_task(
    "Instinct-Nav-G1-SRU-DynamicMapCross-RichTerrain-Centerline-Unobserved-AvoidReact-DirectMask-Banks-OfficialParkour-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_DYNAMIC_MAP_CROSS_RICH_TERRAIN_OFFICIAL_CENTERLINE_UNOBSERVED_AVOID_REACT_DIRECT_MASK_BANKS,
    agents.rsl_rl_cfg.G1InstinctNavPPODirectMaskBanksOfficialRunnerCfg,
)
_register_navigation_task(
    "Instinct-Nav-G1-SRU-DynamicMapCross-RichTerrain-Centerline-Unobserved-AvoidReact-DirectMask-Banks-OfficialParkour-Dev-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_DYNAMIC_MAP_CROSS_RICH_TERRAIN_OFFICIAL_CENTERLINE_UNOBSERVED_AVOID_REACT_DIRECT_MASK_BANKS_DEV,
    agents.rsl_rl_cfg.G1InstinctNavPPODirectMaskBanksOfficialRunnerDevCfg,
)
_register_navigation_task(
    "Instinct-Nav-G1-SRU-DynamicMapCross-RichTerrain-Centerline-Unobserved-AvoidReact-DirectMask-Banks-OfficialParkour-Play-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_DYNAMIC_MAP_CROSS_RICH_TERRAIN_OFFICIAL_CENTERLINE_UNOBSERVED_AVOID_REACT_DIRECT_MASK_BANKS_PLAY,
    agents.rsl_rl_cfg.G1InstinctNavPPODirectMaskBanksOfficialRunnerCfg,
)

# External mask/depth image features enter both SRUs; visibility BEV remains
# available through the inherited map-query branch.
_register_navigation_task(
    "Instinct-Nav-G1-SRU-RichTerrain-NoCenterline-AvoidReact-MaskCNN-Banks-OfficialParkour-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_MASK_CNN_RICH_TERRAIN_AVOID_REACT_BANKS,
    agents.rsl_rl_cfg.G1InstinctNavPPOMaskCnnBanksOfficialRunnerCfg,
)
_register_navigation_task(
    "Instinct-Nav-G1-SRU-RichTerrain-NoCenterline-AvoidReact-MaskCNN-Banks-OfficialParkour-Dev-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_MASK_CNN_RICH_TERRAIN_AVOID_REACT_BANKS_DEV,
    agents.rsl_rl_cfg.G1InstinctNavPPOMaskCnnBanksOfficialRunnerDevCfg,
)
_register_navigation_task(
    "Instinct-Nav-G1-SRU-RichTerrain-NoCenterline-AvoidReact-MaskCNN-Banks-OfficialParkour-Play-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_MASK_CNN_RICH_TERRAIN_AVOID_REACT_BANKS_PLAY,
    agents.rsl_rl_cfg.G1InstinctNavPPOMaskCnnBanksOfficialRunnerCfg,
)

_register_navigation_task(
    "Instinct-Nav-G1-SRU-MaskCNN-OldBaseline-OfficialParkour-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_MASK_CNN_OLD_BASELINE,
    agents.rsl_rl_cfg.G1InstinctNavPPOMaskCnnOldBaselineRunnerCfg,
)
_register_navigation_task(
    "Instinct-Nav-G1-SRU-MaskCNN-OldBaseline-OfficialParkour-Dev-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_MASK_CNN_OLD_BASELINE_DEV,
    agents.rsl_rl_cfg.G1InstinctNavPPOMaskCnnOldBaselineRunnerDevCfg,
)
_register_navigation_task(
    "Instinct-Nav-G1-SRU-MaskCNN-OldBaseline-OfficialParkour-Play-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_MASK_CNN_OLD_BASELINE_PLAY,
    agents.rsl_rl_cfg.G1InstinctNavPPOMaskCnnOldBaselineRunnerCfg,
)

_register_navigation_task(
    "Instinct-Nav-G1-SRU-MaskCNN-Encounter-RestoredRewards-Banks-OfficialParkour-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_MASK_CNN_ENCOUNTER_RESTORED_REWARDS,
    agents.rsl_rl_cfg.G1InstinctNavPPOMaskCnnBanksOfficialRunnerCfg,
)
_register_navigation_task(
    "Instinct-Nav-G1-SRU-MaskCNN-Encounter-RestoredRewards-Banks-OfficialParkour-Dev-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_MASK_CNN_ENCOUNTER_RESTORED_REWARDS_DEV,
    agents.rsl_rl_cfg.G1InstinctNavPPOMaskCnnBanksOfficialRunnerDevCfg,
)
_register_navigation_task(
    "Instinct-Nav-G1-SRU-MaskCNN-Encounter-RestoredRewards-Banks-OfficialParkour-Play-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_MASK_CNN_ENCOUNTER_RESTORED_REWARDS_PLAY,
    agents.rsl_rl_cfg.G1InstinctNavPPOMaskCnnBanksOfficialRunnerCfg,
)

# MaskCNN64 gated residual fusion with the restored navigation rewards. Dynamic
# obstacles start at difficulty 1 in every environment and use 1.6 m geometry.
_register_navigation_task(
    "Instinct-Nav-G1-SRU-MaskCNN64-GatedAdd-Encounter-RestoredRewards-Banks-OfficialParkour-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_MASK_CNN64_GATED_ADD,
    agents.rsl_rl_cfg.G1InstinctNavPPOMaskCnn64GatedAddBanksOfficialRunnerCfg,
)
_register_navigation_task(
    "Instinct-Nav-G1-SRU-MaskCNN64-GatedAdd-Encounter-RestoredRewards-Banks-OfficialParkour-Dev-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_MASK_CNN64_GATED_ADD_DEV,
    agents.rsl_rl_cfg.G1InstinctNavPPOMaskCnn64GatedAddBanksOfficialRunnerDevCfg,
)
_register_navigation_task(
    "Instinct-Nav-G1-SRU-MaskCNN64-GatedAdd-Encounter-RestoredRewards-Banks-OfficialParkour-Play-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_MASK_CNN64_GATED_ADD_PLAY,
    agents.rsl_rl_cfg.G1InstinctNavPPOMaskCnn64GatedAddBanksOfficialRunnerCfg,
)

# Small mask-CNN feature concatenated to the SRU, with the same hard obstacle
# setup as MaskCNN64-GatedAdd for a controlled architecture comparison.
_register_navigation_task(
    "Instinct-Nav-G1-SRU-MaskCNN32-Concat-Encounter-RestoredRewards-Banks-OfficialParkour-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_MASK_CNN32_CONCAT,
    agents.rsl_rl_cfg.G1InstinctNavPPOMaskCnn32ConcatBanksOfficialRunnerCfg,
)
_register_navigation_task(
    "Instinct-Nav-G1-SRU-MaskCNN32-Concat-Encounter-RestoredRewards-Banks-OfficialParkour-Dev-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_MASK_CNN32_CONCAT_DEV,
    agents.rsl_rl_cfg.G1InstinctNavPPOMaskCnn32ConcatBanksOfficialRunnerDevCfg,
)
_register_navigation_task(
    "Instinct-Nav-G1-SRU-MaskCNN32-Concat-Encounter-RestoredRewards-Banks-OfficialParkour-Play-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_MASK_CNN32_CONCAT_PLAY,
    agents.rsl_rl_cfg.G1InstinctNavPPOMaskCnn32ConcatBanksOfficialRunnerCfg,
)

# Identical hard encounter/bank setup, with only the mask observation and encoder removed.
_register_navigation_task(
    "Instinct-Nav-G1-SRU-NoMaskControl-Encounter-RestoredRewards-Banks-OfficialParkour-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_NOMASK_CONTROL,
    agents.rsl_rl_cfg.G1InstinctNavPPONoMaskControlBanksOfficialRunnerCfg,
)
_register_navigation_task(
    "Instinct-Nav-G1-SRU-NoMaskControl-Encounter-RestoredRewards-Banks-OfficialParkour-Dev-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_NOMASK_CONTROL_DEV,
    agents.rsl_rl_cfg.G1InstinctNavPPONoMaskControlBanksOfficialRunnerDevCfg,
)
_register_navigation_task(
    "Instinct-Nav-G1-SRU-NoMaskControl-Encounter-RestoredRewards-Banks-OfficialParkour-Play-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_NOMASK_CONTROL_PLAY,
    agents.rsl_rl_cfg.G1InstinctNavPPONoMaskControlBanksOfficialRunnerCfg,
)

# Original 5D DynamicMapCross/SRU structure. Dynamic masks are projected with
# delayed depth and rendered as current-only black/high-risk visibility cells;
# terrain banks advance only after the current bank reaches 90% success.
_register_navigation_task(
    "Instinct-Nav-G1-SRU-DynamicRiskBEV-Encounter-RestoredRewards-SuccessBanks-OfficialParkour-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_DYNAMIC_RISK_BEV_SUCCESS_BANKS,
    agents.rsl_rl_cfg.G1InstinctNavPPODynamicRiskBevSuccessBanksRunnerCfg,
)
_register_navigation_task(
    "Instinct-Nav-G1-SRU-DynamicRiskBEV-Encounter-RestoredRewards-SuccessBanks-OfficialParkour-Dev-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_DYNAMIC_RISK_BEV_SUCCESS_BANKS_DEV,
    agents.rsl_rl_cfg.G1InstinctNavPPODynamicRiskBevSuccessBanksRunnerDevCfg,
)
_register_navigation_task(
    "Instinct-Nav-G1-SRU-DynamicRiskBEV-Encounter-RestoredRewards-SuccessBanks-OfficialParkour-Play-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_DYNAMIC_RISK_BEV_SUCCESS_BANKS_PLAY,
    agents.rsl_rl_cfg.G1InstinctNavPPODynamicRiskBevSuccessBanksRunnerCfg,
)

# One continuous run on one terrain bank: mask/depth projection is active from
# iteration 0, obstacles are disabled for 5000 iterations, then enabled in a
# fixed 50% environment cohort for the remaining 2000 iterations.
_register_navigation_task(
    "Instinct-Nav-G1-SRU-DynamicRiskBEV-Static5000-Mixed50-SingleTerrain-OfficialParkour-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_DYNAMIC_RISK_BEV_STATIC5000_MIXED50_SINGLE_TERRAIN,
    agents.rsl_rl_cfg.G1InstinctNavPPODynamicRiskBevPhasedSingleTerrainRunnerCfg,
)
_register_navigation_task(
    "Instinct-Nav-G1-SRU-DynamicRiskBEV-Static5000-Mixed50-SingleTerrain-OfficialParkour-Dev-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_DYNAMIC_RISK_BEV_STATIC5000_MIXED50_SINGLE_TERRAIN_DEV,
    agents.rsl_rl_cfg.G1InstinctNavPPODynamicRiskBevPhasedSingleTerrainRunnerDevCfg,
)
_register_navigation_task(
    "Instinct-Nav-G1-SRU-DynamicRiskBEV-Static5000-Mixed50-SingleTerrain-OfficialParkour-Play-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_DYNAMIC_RISK_BEV_STATIC5000_MIXED50_SINGLE_TERRAIN_PLAY,
    agents.rsl_rl_cfg.G1InstinctNavPPODynamicRiskBevPhasedSingleTerrainRunnerCfg,
)

# Controlled two-run ablation. Both runs use one terrain set and the same
# 3000-static + 2000-mixed50 schedule; only the 5D visibility BEV differs.
_register_navigation_task(
    "Instinct-Nav-G1-SRU-Ablation-NoBEV-Static3000-Mixed50-OfficialParkour-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_ABLATION_SRU_NOBEV_STATIC3000_MIXED50,
    agents.rsl_rl_cfg.G1InstinctNavPPOAblationNoBevPhasedRunnerCfg,
)
_register_navigation_task(
    "Instinct-Nav-G1-SRU-Ablation-NoBEV-Static3000-Mixed50-OfficialParkour-Dev-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_ABLATION_SRU_NOBEV_STATIC3000_MIXED50_DEV,
    agents.rsl_rl_cfg.G1InstinctNavPPOAblationNoBevPhasedRunnerDevCfg,
)
_register_navigation_task(
    "Instinct-Nav-G1-SRU-Ablation-NoBEV-Static3000-Mixed50-OfficialParkour-Play-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_ABLATION_SRU_NOBEV_STATIC3000_MIXED50_PLAY,
    agents.rsl_rl_cfg.G1InstinctNavPPOAblationNoBevPhasedRunnerCfg,
)
_register_navigation_task(
    "Instinct-Nav-G1-SRU-Ablation-BEV-Static3000-Mixed50-OfficialParkour-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_ABLATION_SRU_BEV_STATIC3000_MIXED50,
    agents.rsl_rl_cfg.G1InstinctNavPPOAblationBevPhasedRunnerCfg,
)
_register_navigation_task(
    "Instinct-Nav-G1-SRU-Ablation-BEV-Static3000-Mixed50-OfficialParkour-Dev-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_ABLATION_SRU_BEV_STATIC3000_MIXED50_DEV,
    agents.rsl_rl_cfg.G1InstinctNavPPOAblationBevPhasedRunnerDevCfg,
)
_register_navigation_task(
    "Instinct-Nav-G1-SRU-Ablation-BEV-Static3000-Mixed50-OfficialParkour-Play-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_ABLATION_SRU_BEV_STATIC3000_MIXED50_PLAY,
    agents.rsl_rl_cfg.G1InstinctNavPPOAblationBevPhasedRunnerCfg,
)

# Human-scale terrain calibration: the 30 m tile is retained, while topology is
# densified from 15x15 at 2.0 m to 20x20 at 1.5 m and pit bridges use one cell.
_register_navigation_task(
    "Instinct-Nav-G1-SRU-Ablation-NoBEV-HumanScale20-OfficialParkour-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_ABLATION_SRU_NOBEV_HUMAN_SCALE20,
    agents.rsl_rl_cfg.G1InstinctNavPPOAblationNoBevHumanScale20RunnerCfg,
)
_register_navigation_task(
    "Instinct-Nav-G1-SRU-Ablation-NoBEV-HumanScale20-OfficialParkour-Dev-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_ABLATION_SRU_NOBEV_HUMAN_SCALE20_DEV,
    agents.rsl_rl_cfg.G1InstinctNavPPOAblationNoBevHumanScale20RunnerDevCfg,
)
_register_navigation_task(
    "Instinct-Nav-G1-SRU-Ablation-NoBEV-HumanScale20-OfficialParkour-Play-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_ABLATION_SRU_NOBEV_HUMAN_SCALE20_PLAY,
    agents.rsl_rl_cfg.G1InstinctNavPPOAblationNoBevHumanScale20RunnerCfg,
)
_register_navigation_task(
    "Instinct-Nav-G1-SRU-Ablation-BEV-HumanScale20-OfficialParkour-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_ABLATION_SRU_BEV_HUMAN_SCALE20,
    agents.rsl_rl_cfg.G1InstinctNavPPOAblationBevHumanScale20RunnerCfg,
)
_register_navigation_task(
    "Instinct-Nav-G1-SRU-Ablation-BEV-HumanScale20-OfficialParkour-Dev-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_ABLATION_SRU_BEV_HUMAN_SCALE20_DEV,
    agents.rsl_rl_cfg.G1InstinctNavPPOAblationBevHumanScale20RunnerDevCfg,
)
_register_navigation_task(
    "Instinct-Nav-G1-SRU-Ablation-BEV-HumanScale20-OfficialParkour-Play-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_ABLATION_SRU_BEV_HUMAN_SCALE20_PLAY,
    agents.rsl_rl_cfg.G1InstinctNavPPOAblationBevHumanScale20RunnerCfg,
)

# HardEncounterV2 holds the environment schedule fixed across all three
# observation ablations: 1000 static iterations, then 100% dynamic environments
# with a recyclable 3/4-obstacle local conflict pool.
_register_navigation_task(
    "Instinct-Nav-G1-SRU-Ablation-NoBEV-HardEncounterV2-OfficialParkour-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_ABLATION_SRU_NOBEV_HARD_ENCOUNTER_V2,
    agents.rsl_rl_cfg.G1InstinctNavPPOHardEncounterV2NoBevRunnerCfg,
)
_register_navigation_task(
    "Instinct-Nav-G1-SRU-Ablation-NoBEV-HardEncounterV2-OfficialParkour-Dev-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_ABLATION_SRU_NOBEV_HARD_ENCOUNTER_V2_DEV,
    agents.rsl_rl_cfg.G1InstinctNavPPOHardEncounterV2NoBevRunnerDevCfg,
)
_register_navigation_task(
    "Instinct-Nav-G1-SRU-Ablation-NoBEV-HardEncounterV2-OfficialParkour-Play-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_ABLATION_SRU_NOBEV_HARD_ENCOUNTER_V2_PLAY,
    agents.rsl_rl_cfg.G1InstinctNavPPOHardEncounterV2NoBevRunnerCfg,
)
_register_navigation_task(
    "Instinct-Nav-G1-SRU-Ablation-CurrentRiskBEV-HardEncounterV2-OfficialParkour-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_ABLATION_SRU_CURRENT_RISK_HARD_ENCOUNTER_V2,
    agents.rsl_rl_cfg.G1InstinctNavPPOHardEncounterV2CurrentRiskRunnerCfg,
)
_register_navigation_task(
    "Instinct-Nav-G1-SRU-Ablation-CurrentRiskBEV-HardEncounterV2-OfficialParkour-Dev-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_ABLATION_SRU_CURRENT_RISK_HARD_ENCOUNTER_V2_DEV,
    agents.rsl_rl_cfg.G1InstinctNavPPOHardEncounterV2CurrentRiskRunnerDevCfg,
)
_register_navigation_task(
    "Instinct-Nav-G1-SRU-Ablation-CurrentRiskBEV-HardEncounterV2-OfficialParkour-Play-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_ABLATION_SRU_CURRENT_RISK_HARD_ENCOUNTER_V2_PLAY,
    agents.rsl_rl_cfg.G1InstinctNavPPOHardEncounterV2CurrentRiskRunnerCfg,
)
_register_navigation_task(
    "Instinct-Nav-G1-SRU-Ablation-SectorRiskBEV-HardEncounterV2-OfficialParkour-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_ABLATION_SRU_SECTOR_RISK_HARD_ENCOUNTER_V2,
    agents.rsl_rl_cfg.G1InstinctNavPPOHardEncounterV2SectorRiskRunnerCfg,
)
_register_navigation_task(
    "Instinct-Nav-G1-SRU-Ablation-SectorRiskBEV-HardEncounterV2-OfficialParkour-Dev-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_ABLATION_SRU_SECTOR_RISK_HARD_ENCOUNTER_V2_DEV,
    agents.rsl_rl_cfg.G1InstinctNavPPOHardEncounterV2SectorRiskRunnerDevCfg,
)
_register_navigation_task(
    "Instinct-Nav-G1-SRU-Ablation-SectorRiskBEV-HardEncounterV2-OfficialParkour-Play-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_ABLATION_SRU_SECTOR_RISK_HARD_ENCOUNTER_V2_PLAY,
    agents.rsl_rl_cfg.G1InstinctNavPPOHardEncounterV2SectorRiskRunnerCfg,
)

# HardEncounterV3 keeps the V2 schedule but uses human-scale cylinders. The
# paired tasks isolate learned temporal-BEV query fusion from the environment.
_register_navigation_task(
    "Instinct-Nav-G1-SRU-Ablation-NoBEV-HardEncounterV3Cylinder-OfficialParkour-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_ABLATION_SRU_NOBEV_HARD_ENCOUNTER_V3_CYLINDER,
    agents.rsl_rl_cfg.G1InstinctNavPPOHardEncounterV3CylinderNoBevRunnerCfg,
)
_register_navigation_task(
    "Instinct-Nav-G1-SRU-Ablation-NoBEV-HardEncounterV3Cylinder-OfficialParkour-Dev-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_ABLATION_SRU_NOBEV_HARD_ENCOUNTER_V3_CYLINDER_DEV,
    agents.rsl_rl_cfg.G1InstinctNavPPOHardEncounterV3CylinderNoBevRunnerDevCfg,
)
_register_navigation_task(
    "Instinct-Nav-G1-SRU-Ablation-NoBEV-HardEncounterV3Cylinder-OfficialParkour-Play-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_ABLATION_SRU_NOBEV_HARD_ENCOUNTER_V3_CYLINDER_PLAY,
    agents.rsl_rl_cfg.G1InstinctNavPPOHardEncounterV3CylinderNoBevRunnerCfg,
)
_register_navigation_task(
    "Instinct-Nav-G1-SRU-LearnedBEVQuery-HardEncounterV3Cylinder-OfficialParkour-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_LEARNED_BEV_QUERY_HARD_ENCOUNTER_V3_CYLINDER,
    agents.rsl_rl_cfg.G1InstinctNavPPOLearnedBevQueryHardEncounterV3CylinderRunnerCfg,
)
_register_navigation_task(
    "Instinct-Nav-G1-SRU-LearnedBEVQuery-HardEncounterV3Cylinder-OfficialParkour-Dev-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_LEARNED_BEV_QUERY_HARD_ENCOUNTER_V3_CYLINDER_DEV,
    agents.rsl_rl_cfg.G1InstinctNavPPOLearnedBevQueryHardEncounterV3CylinderRunnerDevCfg,
)
_register_navigation_task(
    "Instinct-Nav-G1-SRU-LearnedBEVQuery-HardEncounterV3Cylinder-OfficialParkour-Play-v0",
    navigation_env_cfg.G1InstinctNavigationEnvCfg_LEARNED_BEV_QUERY_HARD_ENCOUNTER_V3_CYLINDER_PLAY,
    agents.rsl_rl_cfg.G1InstinctNavPPOLearnedBevQueryHardEncounterV3CylinderRunnerCfg,
)

for suffix, env_type, runner_type in (
    ("v0", navigation_env_cfg.G1InstinctNavigationEnvCfg_MOTION_BEV_QUERY_HARD_V3,
     agents.rsl_rl_cfg.G1InstinctNavPPOMotionBevQueryHardV3RunnerCfg),
    ("Dev-v0", navigation_env_cfg.G1InstinctNavigationEnvCfg_MOTION_BEV_QUERY_HARD_V3_DEV,
     agents.rsl_rl_cfg.G1InstinctNavPPOMotionBevQueryHardV3RunnerDevCfg),
    ("Play-v0", navigation_env_cfg.G1InstinctNavigationEnvCfg_MOTION_BEV_QUERY_HARD_V3_PLAY,
     agents.rsl_rl_cfg.G1InstinctNavPPOMotionBevQueryHardV3RunnerCfg),
):
    _register_navigation_task(f"Instinct-Nav-G1-SRU-MotionBEVQuery-HardV3-OfficialParkour-{suffix}", env_type, runner_type)

for suffix, env_type, runner_type in (
    ("v0", navigation_env_cfg.G1InstinctNavigationEnvCfg_NOBEV_REWARD_MATCHED_HARD_V3,
     agents.rsl_rl_cfg.G1InstinctNavPPONoBevRewardMatchedHardV3RunnerCfg),
    ("Dev-v0", navigation_env_cfg.G1InstinctNavigationEnvCfg_NOBEV_REWARD_MATCHED_HARD_V3_DEV,
     agents.rsl_rl_cfg.G1InstinctNavPPONoBevRewardMatchedHardV3RunnerDevCfg),
    ("Play-v0", navigation_env_cfg.G1InstinctNavigationEnvCfg_NOBEV_REWARD_MATCHED_HARD_V3_PLAY,
     agents.rsl_rl_cfg.G1InstinctNavPPONoBevRewardMatchedHardV3RunnerCfg),
):
    _register_navigation_task(f"Instinct-Nav-G1-SRU-NoBEVRewardMatched-HardV3-OfficialParkour-{suffix}", env_type, runner_type)

for suffix, env_type, runner_type in (
    ("v0", navigation_env_cfg.G1InstinctNavigationEnvCfg_MOTION_BEV_FUTURE_HARD_V3,
     agents.rsl_rl_cfg.G1InstinctNavPPOMotionBevFutureHardV3RunnerCfg),
    ("Dev-v0", navigation_env_cfg.G1InstinctNavigationEnvCfg_MOTION_BEV_FUTURE_HARD_V3_DEV,
     agents.rsl_rl_cfg.G1InstinctNavPPOMotionBevFutureHardV3RunnerDevCfg),
    ("Play-v0", navigation_env_cfg.G1InstinctNavigationEnvCfg_MOTION_BEV_FUTURE_HARD_V3_PLAY,
     agents.rsl_rl_cfg.G1InstinctNavPPOMotionBevFutureHardV3RunnerCfg),
):
    _register_navigation_task(f"Instinct-Nav-G1-SRU-MotionBEVFuture-HardV3-OfficialParkour-{suffix}", env_type, runner_type)

# AvoidReact-Banks + dynamic-obstacle segmentation mask head (raw delayed depth -> mask
for suffix, env_type, runner_type in (
    ("v0", navigation_env_cfg.G1InstinctNavigationEnvCfg_MOTION_BEV_CONVLSTM_HARD_V3,
     agents.rsl_rl_cfg.G1InstinctNavPPOMotionBevConvLSTMHardV3RunnerCfg),
    ("Dev-v0", navigation_env_cfg.G1InstinctNavigationEnvCfg_MOTION_BEV_CONVLSTM_HARD_V3_DEV,
     agents.rsl_rl_cfg.G1InstinctNavPPOMotionBevConvLSTMHardV3RunnerDevCfg),
    ("Play-v0", navigation_env_cfg.G1InstinctNavigationEnvCfg_MOTION_BEV_CONVLSTM_HARD_V3_PLAY,
     agents.rsl_rl_cfg.G1InstinctNavPPOMotionBevConvLSTMHardV3RunnerCfg),
):
    _register_navigation_task(f"Instinct-Nav-G1-SRU-MotionBEVConvLSTM-HardV3-OfficialParkour-{suffix}", env_type, runner_type)

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

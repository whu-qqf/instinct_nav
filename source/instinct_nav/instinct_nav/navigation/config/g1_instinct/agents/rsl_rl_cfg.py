from __future__ import annotations

from isaaclab.utils import configclass
from instinct_nav.navigation.config.go2.agents.rsl_rl_cfg import Go2NavPPOVisMapRunnerCfg


@configclass
class G1InstinctNavPPOVisMapRunnerCfg(Go2NavPPOVisMapRunnerCfg):
    seed = 72
    logger = "tensorboard"
    wandb_project = "isaaclab_nav_g1_instinct"
    experiment_name = "g1_instinct_navigation_ppo_vismap"

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.num_mini_batches = 4
        self.obs_groups = {"policy": ["policy"], "critic": ["critic"]}
        self.policy.use_actor_visibility_map = False
        self.policy.use_actor_visibility_query_attention = False


@configclass
class G1InstinctNavPPOVisMapRunnerDevCfg(G1InstinctNavPPOVisMapRunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 300
        self.experiment_name = "g1_instinct_navigation_ppo_vismap_dev"
        self.logger = "tensorboard"


@configclass
class G1InstinctNavPPOVisibilityEarlyRunnerCfg(G1InstinctNavPPOVisMapRunnerCfg):
    """PPO early fusion: visibility feature enters the actor SRU recurrent input."""

    def __post_init__(self):
        super().__post_init__()
        self.run_name = "visibility_map_early_fusion"
        self.policy.use_actor_visibility_map = True
        self.policy.use_actor_visibility_early_fusion = True
        self.policy.use_actor_visibility_query_attention = False


@configclass
class G1InstinctNavPPOVisibilityLateRunnerCfg(G1InstinctNavPPOVisMapRunnerCfg):
    """PPO late fusion: visibility feature is concatenated after the actor SRU."""

    def __post_init__(self):
        super().__post_init__()
        self.run_name = "visibility_map_late_fusion"
        self.policy.use_actor_visibility_map = True
        self.policy.use_actor_visibility_early_fusion = False
        self.policy.use_actor_visibility_query_attention = False


@configclass
class G1InstinctNavPPOVisibilityQueryRunnerCfg(G1InstinctNavPPOVisMapRunnerCfg):
    """PPO query fusion: actor recurrent context attends over visibility-map tokens."""

    def __post_init__(self):
        super().__post_init__()
        self.run_name = "visibility_map_query_fusion"
        self.policy.use_actor_visibility_map = True
        self.policy.use_actor_visibility_early_fusion = False
        self.policy.use_actor_visibility_query_attention = True


@configclass
class G1InstinctNavPPOVisibilityEarlyRunnerDevCfg(G1InstinctNavPPOVisibilityEarlyRunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 300
        self.experiment_name = "g1_instinct_navigation_ppo_vismap_dev"
        self.logger = "tensorboard"


@configclass
class G1InstinctNavPPOVisibilityLateRunnerDevCfg(G1InstinctNavPPOVisibilityLateRunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 300
        self.experiment_name = "g1_instinct_navigation_ppo_vismap_dev"
        self.logger = "tensorboard"


@configclass
class G1InstinctNavPPOVisibilityQueryRunnerDevCfg(G1InstinctNavPPOVisibilityQueryRunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 300
        self.experiment_name = "g1_instinct_navigation_ppo_vismap_dev"
        self.logger = "tensorboard"


@configclass
class G1InstinctNavPPOSceneTokenQueryRunnerCfg(G1InstinctNavPPOVisMapRunnerCfg):
    """PPO query fusion over fixed scene tokens derived from local visibility memory."""

    def __post_init__(self):
        super().__post_init__()
        self.run_name = "scene_token_query_fusion"
        self.policy.use_actor_visibility_map = False
        self.policy.use_actor_visibility_early_fusion = False
        self.policy.use_actor_visibility_query_attention = False
        self.policy.use_actor_scene_tokens = True
        self.policy.scene_token_num_tokens = 64
        self.policy.scene_token_dim = 6
        self.policy.scene_token_feature_dim = 64
        self.policy.scene_token_query_num_heads = 4


@configclass
class G1InstinctNavPPOSceneTokenQueryRunnerDevCfg(G1InstinctNavPPOSceneTokenQueryRunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 300
        self.experiment_name = "g1_instinct_navigation_ppo_scene_token_dev"
        self.logger = "tensorboard"


@configclass
class G1InstinctNavPPOSceneTokenRiskQueryRunnerCfg(G1InstinctNavPPOSceneTokenQueryRunnerCfg):
    """PPO query fusion over 9D actor/critic scene-risk tokens."""

    def __post_init__(self):
        super().__post_init__()
        self.run_name = "scene_token_risk_query_fusion"
        self.policy.use_actor_scene_tokens = True
        self.policy.use_critic_scene_tokens = True
        self.policy.scene_token_num_tokens = 64
        self.policy.scene_token_dim = 9
        self.policy.scene_token_feature_dim = 64
        self.policy.scene_token_query_num_heads = 4


@configclass
class G1InstinctNavPPOSceneTokenRiskQueryRunnerDevCfg(G1InstinctNavPPOSceneTokenRiskQueryRunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 300
        self.experiment_name = "g1_instinct_navigation_ppo_scene_token_risk_dev"
        self.logger = "tensorboard"


@configclass
class G1InstinctNavPPOSceneTokenRiskNoGoalQueryRunnerCfg(G1InstinctNavPPOSceneTokenQueryRunnerCfg):
    """PPO query fusion over 6D actor/critic scene-risk tokens without goal-relative features."""

    def __post_init__(self):
        super().__post_init__()
        self.run_name = "scene_token_risk_no_goal_query_fusion"
        self.policy.use_actor_scene_tokens = True
        self.policy.use_critic_scene_tokens = True
        self.policy.scene_token_num_tokens = 64
        self.policy.scene_token_dim = 6
        self.policy.scene_token_feature_dim = 64
        self.policy.scene_token_query_num_heads = 4


@configclass
class G1InstinctNavPPOSceneTokenRiskNoGoalQueryRunnerDevCfg(G1InstinctNavPPOSceneTokenRiskNoGoalQueryRunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 300
        self.experiment_name = "g1_instinct_navigation_ppo_scene_token_risk_no_goal_dev"
        self.logger = "tensorboard"

@configclass
class G1InstinctNavPPOSceneTokenRiskNoGoalSharedLowCamQueryRunnerCfg(
    G1InstinctNavPPOSceneTokenRiskNoGoalQueryRunnerCfg
):
    """Scene-token v3 runner for the shared low-level camera ablation."""

    def __post_init__(self):
        super().__post_init__()
        self.run_name = "scene_token_risk_no_goal_shared_lowcam_query_fusion"


@configclass
class G1InstinctNavPPOSceneTokenRiskNoGoalSharedLowCamQueryRunnerDevCfg(
    G1InstinctNavPPOSceneTokenRiskNoGoalSharedLowCamQueryRunnerCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 300
        self.experiment_name = "g1_instinct_navigation_ppo_scene_token_risk_no_goal_shared_lowcam_dev"
        self.logger = "tensorboard"

@configclass
class G1InstinctNavPPOSceneTokenRiskNoGoalSharedLowCamOfficialRunnerCfg(
    G1InstinctNavPPOSceneTokenRiskNoGoalSharedLowCamQueryRunnerCfg
):
    """Final SRU runner trained against the official stable parkour backend."""

    def __post_init__(self):
        super().__post_init__()
        self.run_name = "scene_token_risk_no_goal_shared_lowcam_official_parkour"


@configclass
class G1InstinctNavPPOSceneTokenRiskNoGoalSharedLowCamOfficialRunnerDevCfg(
    G1InstinctNavPPOSceneTokenRiskNoGoalSharedLowCamOfficialRunnerCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 10
        self.experiment_name = "g1_instinct_navigation_ppo_scene_token_official_parkour_dev"
        self.logger = "tensorboard"


@configclass
class G1InstinctNavPPOSceneTokenScoreVisOfficialRunnerCfg(
    G1InstinctNavPPOSceneTokenRiskNoGoalSharedLowCamOfficialRunnerCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.run_name = "scene_token_score_vis_shared_lowcam_official_parkour"


@configclass
class G1InstinctNavPPOSceneTokenScoreVisOfficialRunnerDevCfg(
    G1InstinctNavPPOSceneTokenScoreVisOfficialRunnerCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 10
        self.experiment_name = "g1_instinct_navigation_score_vis_dev"


@configclass
class G1InstinctNavPPOSceneTokenScoreVisDynamicOfficialRunnerCfg(
    G1InstinctNavPPOSceneTokenScoreVisOfficialRunnerCfg
):
    """Stage-3 dynamic obstacle curriculum runner (6000 total iterations)."""

    def __post_init__(self):
        super().__post_init__()
        self.run_name = "scene_token_score_vis_dynamic_official_parkour"
        self.max_iterations = 6000


@configclass
class G1InstinctNavPPOSceneTokenScoreVisDynamicOfficialRunnerDevCfg(
    G1InstinctNavPPOSceneTokenScoreVisDynamicOfficialRunnerCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 10
        self.experiment_name = "g1_instinct_navigation_score_vis_dynamic_dev"
        self.logger = "tensorboard"


@configclass
class G1InstinctNavPPODynamicMapCrossOfficialRunnerCfg(
    G1InstinctNavPPOSceneTokenScoreVisDynamicOfficialRunnerCfg
):
    """Learned dynamic map tokens from current-to-history cross-attention."""

    def __post_init__(self):
        super().__post_init__()
        self.run_name = "dynamic_map_cross_attention_official_parkour"
        self.policy.scene_token_dim = 5
        self.policy.use_dynamic_map_cross_attention = True


@configclass
class G1InstinctNavPPODynamicMapCrossBanksOfficialRunnerCfg(
    G1InstinctNavPPODynamicMapCrossOfficialRunnerCfg
):
    """DynamicMapCross runner with smaller PPO mini-batches for terrain-bank training."""

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.num_mini_batches = 8


@configclass
class G1InstinctNavPPODynamicMapCrossBanksOfficialRunnerDevCfg(
    G1InstinctNavPPODynamicMapCrossBanksOfficialRunnerCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 10
        self.experiment_name = "g1_instinct_navigation_dynamic_map_cross_banks_dev"
        self.logger = "tensorboard"


@configclass
class G1InstinctNavPPODynamicMapCrossOfficialRunnerDevCfg(
    G1InstinctNavPPODynamicMapCrossOfficialRunnerCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 10
        self.experiment_name = "g1_instinct_navigation_dynamic_map_cross_dev"
        self.logger = "tensorboard"


@configclass
class G1InstinctNavPPODynamicMaskBanksOfficialRunnerCfg(
    G1InstinctNavPPODynamicMapCrossBanksOfficialRunnerCfg
):
    """DynamicMapCross-Banks + dynamic-obstacle segmentation mask head + aux loss."""

    def __post_init__(self):
        super().__post_init__()
        self.run_name = "dynamic_mask_banks_official_parkour"
        self.policy.use_dynamic_mask = True
        self.policy.mask_input_dims = (1, 36, 64)
        self.policy.mask_feature_dim = 64
        self.algorithm.lambda_mask = 0.2


@configclass
class G1InstinctNavPPODynamicMaskBanksOfficialRunnerDevCfg(
    G1InstinctNavPPODynamicMaskBanksOfficialRunnerCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 10
        self.experiment_name = "g1_instinct_navigation_dynamic_mask_banks_dev"
        self.logger = "tensorboard"

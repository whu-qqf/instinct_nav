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
class G1InstinctNavPPODynamicMapMaskBanksOfficialRunnerCfg(
    G1InstinctNavPPODynamicMapCrossBanksOfficialRunnerCfg
):
    """DynamicMapCross-Banks with current dynamic BEV presence as token channel 6."""

    def __post_init__(self):
        super().__post_init__()
        self.run_name = "dynamic_map_mask_no_centerline_banks_official_parkour"
        self.policy.scene_token_dim = 6


@configclass
class G1InstinctNavPPODynamicMapMaskBanksOfficialRunnerDevCfg(
    G1InstinctNavPPODynamicMapMaskBanksOfficialRunnerCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 10
        self.experiment_name = "g1_instinct_navigation_dynamic_map_mask_no_centerline_banks_dev"
        self.logger = "tensorboard"


@configclass
class G1InstinctNavPPODynamicMapMaskRecurrentFusionBanksOfficialRunnerCfg(
    G1InstinctNavPPODynamicMapMaskBanksOfficialRunnerCfg
):
    """DynamicMapMask with dynamic BEV context fused both before and after the SRU."""

    def __post_init__(self):
        super().__post_init__()
        self.run_name = "dynamic_map_mask_recurrent_fusion_no_centerline_banks_official_parkour"
        self.policy.use_dynamic_scene_recurrent_fusion = True
        self.policy.dynamic_scene_recurrent_feature_dim = 64


@configclass
class G1InstinctNavPPODynamicMapMaskRecurrentFusionBanksOfficialRunnerDevCfg(
    G1InstinctNavPPODynamicMapMaskRecurrentFusionBanksOfficialRunnerCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 10
        self.experiment_name = "g1_instinct_navigation_dynamic_map_mask_recurrent_fusion_banks_dev"
        self.logger = "tensorboard"


@configclass
class G1InstinctNavPPODirectMaskBanksOfficialRunnerCfg(
    G1InstinctNavPPODynamicMapCrossBanksOfficialRunnerCfg
):
    """DynamicMapCross-Banks with a compact external mask fed directly to the SRU."""

    def __post_init__(self):
        super().__post_init__()
        self.run_name = "direct_mask_banks_official_parkour"
        self.policy.use_dynamic_mask = True
        self.policy.dynamic_mask_input_mode = "direct_mask"
        self.policy.mask_input_dims = (1, 9, 16)
        self.policy.mask_feature_dim = 144
        self.algorithm.lambda_mask = 0.0


@configclass
class G1InstinctNavPPODirectMaskBanksOfficialRunnerDevCfg(
    G1InstinctNavPPODirectMaskBanksOfficialRunnerCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 10
        self.experiment_name = "g1_instinct_navigation_direct_mask_banks_dev"
        self.logger = "tensorboard"


@configclass
class G1InstinctNavPPOMaskCnnBanksOfficialRunnerCfg(
    G1InstinctNavPPODynamicMapCrossBanksOfficialRunnerCfg
):
    """External mask/depth image encoder plus the inherited visibility BEV branch."""

    def __post_init__(self):
        super().__post_init__()
        self.run_name = "mask_cnn_visibility_banks_official_parkour"
        self.policy.use_actor_scene_tokens = True
        self.policy.use_critic_scene_tokens = True
        self.policy.use_dynamic_map_cross_attention = True
        self.policy.use_dynamic_scene_recurrent_fusion = False
        self.policy.use_dynamic_mask = True
        self.policy.dynamic_mask_input_mode = "encoded_mask"
        self.policy.mask_input_dims = (2, 36, 64)
        self.policy.mask_feature_dim = 128
        self.algorithm.lambda_mask = 0.0


@configclass
class G1InstinctNavPPOMaskCnnBanksOfficialRunnerDevCfg(
    G1InstinctNavPPOMaskCnnBanksOfficialRunnerCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 10
        self.experiment_name = "g1_instinct_navigation_mask_cnn_banks_dev"
        self.logger = "tensorboard"


@configclass
class G1InstinctNavPPOMaskCnn64GatedAddBanksOfficialRunnerCfg(
    G1InstinctNavPPOMaskCnnBanksOfficialRunnerCfg
):
    """Compress mask context to 64D and gate it into each 64D depth feature."""

    def __post_init__(self):
        super().__post_init__()
        self.run_name = "mask_cnn64_gated_add_visibility_banks_official_parkour"
        self.policy.mask_feature_dim = 64
        self.policy.dynamic_mask_fusion_mode = "gated_add"
        self.policy.dynamic_mask_gate_init = 0.1


@configclass
class G1InstinctNavPPOMaskCnn64GatedAddBanksOfficialRunnerDevCfg(
    G1InstinctNavPPOMaskCnn64GatedAddBanksOfficialRunnerCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 10
        self.experiment_name = "g1_instinct_navigation_mask_cnn64_gated_add_banks_dev"
        self.logger = "tensorboard"


@configclass
class G1InstinctNavPPOMaskCnn32ConcatBanksOfficialRunnerCfg(
    G1InstinctNavPPOMaskCnnBanksOfficialRunnerCfg
):
    """Append a compact 32D mask feature to the baseline SRU input."""

    def __post_init__(self):
        super().__post_init__()
        self.run_name = "mask_cnn32_concat_visibility_banks_official_parkour"
        self.max_iterations = 8000
        self.policy.mask_feature_dim = 32
        self.policy.dynamic_mask_fusion_mode = "concat"


@configclass
class G1InstinctNavPPOMaskCnn32ConcatBanksOfficialRunnerDevCfg(
    G1InstinctNavPPOMaskCnn32ConcatBanksOfficialRunnerCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 10
        self.experiment_name = "g1_instinct_navigation_mask_cnn32_concat_banks_dev"
        self.logger = "tensorboard"


@configclass
class G1InstinctNavPPONoMaskControlBanksOfficialRunnerCfg(
    G1InstinctNavPPOMaskCnn32ConcatBanksOfficialRunnerCfg
):
    """Use the MaskCNN32 PPO setup with its mask branch disabled."""

    def __post_init__(self):
        super().__post_init__()
        self.run_name = "nomask_control_hard_allenv_banks_official_parkour"
        self.policy.use_dynamic_mask = False


@configclass
class G1InstinctNavPPONoMaskControlBanksOfficialRunnerDevCfg(
    G1InstinctNavPPONoMaskControlBanksOfficialRunnerCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 10
        self.experiment_name = "g1_instinct_navigation_nomask_control_banks_dev"
        self.logger = "tensorboard"


@configclass
class G1InstinctNavPPODynamicRiskBevSuccessBanksRunnerCfg(
    G1InstinctNavPPODynamicMapCrossOfficialRunnerCfg
):
    """Original 5D DynamicMapCross SRU with mask-to-BEV risk semantics."""

    def __post_init__(self):
        super().__post_init__()
        self.run_name = "dynamic_risk_bev_success_banks_official_parkour"
        self.max_iterations = 12000
        self.algorithm.num_mini_batches = 4
        self.policy.scene_token_dim = 5
        self.policy.use_dynamic_mask = False


@configclass
class G1InstinctNavPPODynamicRiskBevSuccessBanksRunnerDevCfg(
    G1InstinctNavPPODynamicRiskBevSuccessBanksRunnerCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 10
        self.experiment_name = "g1_instinct_navigation_dynamic_risk_bev_success_banks_dev"
        self.logger = "tensorboard"


@configclass
class G1InstinctNavPPODynamicRiskBevPhasedSingleTerrainRunnerCfg(
    G1InstinctNavPPODynamicMapCrossOfficialRunnerCfg
):
    """One continuous 5000-static + 2000-mixed run with the original 5D SRU."""

    def __post_init__(self):
        super().__post_init__()
        self.run_name = "dynamic_risk_bev_static5000_mixed50_single_terrain"
        self.max_iterations = 7000
        self.algorithm.num_mini_batches = 4
        self.policy.scene_token_dim = 5
        self.policy.use_dynamic_mask = False


@configclass
class G1InstinctNavPPODynamicRiskBevPhasedSingleTerrainRunnerDevCfg(
    G1InstinctNavPPODynamicRiskBevPhasedSingleTerrainRunnerCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 10
        self.experiment_name = "g1_instinct_navigation_dynamic_risk_bev_phased_single_terrain_dev"
        self.logger = "tensorboard"


@configclass
class G1InstinctNavPPOAblationNoBevPhasedRunnerCfg(
    G1InstinctNavPPODynamicMapCrossOfficialRunnerCfg
):
    """Pure-SRU control for the static-to-dynamic BEV ablation."""

    def __post_init__(self):
        super().__post_init__()
        self.run_name = "sru_nobev_static3000_mixed50"
        self.experiment_name = "g1_sru_ablation"
        self.wandb_project = "ablation_sru"
        self.max_iterations = 5000
        self.algorithm.num_mini_batches = 4
        self.policy.use_actor_scene_tokens = False
        self.policy.use_critic_scene_tokens = False
        self.policy.use_dynamic_map_cross_attention = False
        self.policy.use_dynamic_scene_recurrent_fusion = False
        self.policy.use_dynamic_mask = False


@configclass
class G1InstinctNavPPOAblationNoBevPhasedRunnerDevCfg(
    G1InstinctNavPPOAblationNoBevPhasedRunnerCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 1
        self.logger = "tensorboard"


@configclass
class G1InstinctNavPPOAblationBevPhasedRunnerCfg(
    G1InstinctNavPPODynamicMapCrossOfficialRunnerCfg
):
    """5D visibility-BEV treatment for the static-to-dynamic ablation."""

    def __post_init__(self):
        super().__post_init__()
        self.run_name = "sru_bev_static3000_mixed50"
        self.experiment_name = "g1_sru_ablation"
        self.wandb_project = "ablation_sru"
        self.max_iterations = 5000
        self.algorithm.num_mini_batches = 4
        self.policy.scene_token_dim = 5
        self.policy.use_dynamic_mask = False


@configclass
class G1InstinctNavPPOAblationBevPhasedRunnerDevCfg(
    G1InstinctNavPPOAblationBevPhasedRunnerCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 1
        self.logger = "tensorboard"


@configclass
class G1InstinctNavPPOAblationNoBevHumanScale20RunnerCfg(
    G1InstinctNavPPOAblationNoBevPhasedRunnerCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.run_name = "sru_nobev_human_scale20_static3000_mixed50"


@configclass
class G1InstinctNavPPOAblationNoBevHumanScale20RunnerDevCfg(
    G1InstinctNavPPOAblationNoBevPhasedRunnerDevCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.run_name = "sru_nobev_human_scale20_smoke"


@configclass
class G1InstinctNavPPOAblationBevHumanScale20RunnerCfg(
    G1InstinctNavPPOAblationBevPhasedRunnerCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.run_name = "sru_bev_human_scale20_static3000_mixed50"


@configclass
class G1InstinctNavPPOAblationBevHumanScale20RunnerDevCfg(
    G1InstinctNavPPOAblationBevPhasedRunnerDevCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.run_name = "sru_bev_human_scale20_smoke"


@configclass
class G1InstinctNavPPOHardEncounterV2NoBevRunnerCfg(
    G1InstinctNavPPOAblationNoBevHumanScale20RunnerCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.run_name = "hard_v2_nobev_static1000_dynamic4000"
        self.experiment_name = "g1_sru_hard_encounter_v2"
        self.wandb_project = "ablation_sru"
        self.max_iterations = 5000


@configclass
class G1InstinctNavPPOHardEncounterV2NoBevRunnerDevCfg(
    G1InstinctNavPPOAblationNoBevHumanScale20RunnerDevCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.run_name = "hard_v2_nobev_smoke"
        self.experiment_name = "g1_sru_hard_encounter_v2_dev"


@configclass
class G1InstinctNavPPOHardEncounterV2CurrentRiskRunnerCfg(
    G1InstinctNavPPOAblationBevHumanScale20RunnerCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.run_name = "hard_v2_current_risk_static1000_dynamic4000"
        self.experiment_name = "g1_sru_hard_encounter_v2"
        self.wandb_project = "ablation_sru"
        self.max_iterations = 5000


@configclass
class G1InstinctNavPPOHardEncounterV2CurrentRiskRunnerDevCfg(
    G1InstinctNavPPOAblationBevHumanScale20RunnerDevCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.run_name = "hard_v2_current_risk_smoke"
        self.experiment_name = "g1_sru_hard_encounter_v2_dev"


@configclass
class G1InstinctNavPPOHardEncounterV2SectorRiskRunnerCfg(
    G1InstinctNavPPOHardEncounterV2CurrentRiskRunnerCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.run_name = "hard_v2_sector_risk_static1000_dynamic4000"


@configclass
class G1InstinctNavPPOHardEncounterV2SectorRiskRunnerDevCfg(
    G1InstinctNavPPOHardEncounterV2CurrentRiskRunnerDevCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.run_name = "hard_v2_sector_risk_smoke"


@configclass
class G1InstinctNavPPOHardEncounterV3CylinderNoBevRunnerCfg(
    G1InstinctNavPPOHardEncounterV2NoBevRunnerCfg
):
    """Pure-SRU control on the cylinder version of HardEncounterV2."""

    def __post_init__(self):
        super().__post_init__()
        self.run_name = "hard_v3_cylinder_nobev_static3000_dynamic3000_mixed50"
        self.experiment_name = "g1_sru_hard_encounter_v3_cylinder"
        self.max_iterations = 6000


@configclass
class G1InstinctNavPPOHardEncounterV3CylinderNoBevRunnerDevCfg(
    G1InstinctNavPPOHardEncounterV3CylinderNoBevRunnerCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.run_name = "hard_v3_cylinder_nobev_smoke"
        self.experiment_name = "g1_sru_hard_encounter_v3_cylinder_dev"
        self.max_iterations = 1
        self.logger = "tensorboard"


@configclass
class G1InstinctNavPPOLearnedBevQueryHardEncounterV3CylinderRunnerCfg(
    G1InstinctNavPPOHardEncounterV3CylinderNoBevRunnerCfg
):
    """CNN spatial tokens queried by the actor/critic SRU contexts."""

    def __post_init__(self):
        super().__post_init__()
        self.run_name = "hard_v3_cylinder_learned_bev_query_static3000_dynamic3000_mixed50"
        self.policy.use_actor_visibility_map = True
        self.policy.use_critic_visibility_map = True
        self.policy.use_actor_visibility_early_fusion = False
        self.policy.visibility_map_input_dims = (6, 32, 32)
        self.policy.visibility_map_feature_dim = 64
        self.policy.use_actor_visibility_query_attention = True
        self.policy.use_critic_visibility_query_attention = True
        self.policy.visibility_map_query_num_heads = 4


@configclass
class G1InstinctNavPPOLearnedBevQueryHardEncounterV3CylinderRunnerDevCfg(
    G1InstinctNavPPOLearnedBevQueryHardEncounterV3CylinderRunnerCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.run_name = "hard_v3_cylinder_learned_bev_query_smoke"
        self.experiment_name = "g1_sru_hard_encounter_v3_cylinder_dev"
        self.max_iterations = 1
        self.logger = "tensorboard"


@configclass
class G1InstinctNavPPOMotionBevQueryHardV3RunnerCfg(
    G1InstinctNavPPOLearnedBevQueryHardEncounterV3CylinderRunnerCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.policy.visibility_map_preserve_spatial_resolution = True
        self.experiment_name = "g1_sru_motion_bev_hard_v3"
        self.run_name = "motion_bev_query_static3000_dynamic3000_mixed50"


@configclass
class G1InstinctNavPPOMotionBevQueryHardV3RunnerDevCfg(G1InstinctNavPPOMotionBevQueryHardV3RunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "g1_sru_motion_bev_hard_v3_dev"
        self.run_name = "motion_bev_query_smoke"
        self.max_iterations = 1
        self.logger = "tensorboard"


@configclass
class G1InstinctNavPPOMotionBevFutureHardV3RunnerCfg(G1InstinctNavPPOMotionBevQueryHardV3RunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.policy.use_future_occupancy = True
        self.algorithm.lambda_future = 0.05
        self.run_name = "motion_bev_future_static3000_dynamic3000_mixed50"


@configclass
class G1InstinctNavPPOMotionBevFutureHardV3RunnerDevCfg(G1InstinctNavPPOMotionBevFutureHardV3RunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "g1_sru_motion_bev_hard_v3_dev"
        self.run_name = "motion_bev_future_smoke"
        self.max_iterations = 1
        self.logger = "tensorboard"


@configclass
class G1InstinctNavPPOMotionBevConvLSTMHardV3RunnerCfg(G1InstinctNavPPOMotionBevFutureHardV3RunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.policy.class_name = "ActorCriticBEVConvLSTM"
        self.run_name = "motion_bev_convlstm_static3000_dynamic3000_mixed50"


@configclass
class G1InstinctNavPPOMotionBevConvLSTMHardV3RunnerDevCfg(G1InstinctNavPPOMotionBevConvLSTMHardV3RunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "g1_sru_motion_bev_hard_v3_dev"
        self.run_name = "motion_bev_convlstm_smoke"
        self.max_iterations = 1
        self.logger = "tensorboard"


@configclass
class G1InstinctNavPPONoBevRewardMatchedHardV3RunnerCfg(G1InstinctNavPPOHardEncounterV3CylinderNoBevRunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "g1_sru_motion_bev_hard_v3"
        self.run_name = "nobev_reward_matched_static3000_dynamic3000_mixed50"


@configclass
class G1InstinctNavPPONoBevRewardMatchedHardV3RunnerDevCfg(G1InstinctNavPPONoBevRewardMatchedHardV3RunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "g1_sru_motion_bev_hard_v3_dev"
        self.run_name = "nobev_reward_matched_smoke"
        self.max_iterations = 1
        self.logger = "tensorboard"


@configclass
class G1InstinctNavPPOMaskCnnOldBaselineRunnerCfg(G1InstinctNavPPOMaskCnnBanksOfficialRunnerCfg):
    """Use old-baseline PPO batching while retaining the mask CNN architecture."""

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.num_mini_batches = 4
        self.run_name = "mask_cnn_old_baseline"


@configclass
class G1InstinctNavPPOMaskCnnOldBaselineRunnerDevCfg(G1InstinctNavPPOMaskCnnOldBaselineRunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 1
        self.experiment_name = "g1_instinct_navigation_mask_cnn_old_baseline_dev"
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

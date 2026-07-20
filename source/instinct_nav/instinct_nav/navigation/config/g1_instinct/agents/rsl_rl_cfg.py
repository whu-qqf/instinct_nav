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

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

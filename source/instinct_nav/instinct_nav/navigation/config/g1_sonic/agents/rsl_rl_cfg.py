# Copyright (c) 2022-2025, Fan Yang and Per Frivik, ETH Zurich.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""RSL-RL agent configurations for G1 SONIC navigation tasks."""

from isaaclab.utils import configclass

from instinct_nav.navigation.config.go2.agents.rsl_rl_cfg import Go2NavPPOVisMapRunnerCfg


@configclass
class G1SonicNavPPOVisMapRunnerCfg(Go2NavPPOVisMapRunnerCfg):
    seed = 71
    wandb_project = "isaaclab_nav_g1_sonic"
    experiment_name = "g1_sonic_navigation_ppo_vismap"

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.num_mini_batches = 8


@configclass
class G1SonicNavPPOVisMapRunnerDevCfg(G1SonicNavPPOVisMapRunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 300
        self.experiment_name = "g1_sonic_navigation_ppo_vismap_dev"
        self.logger = "tensorboard"

# Copyright (c) 2022-2025, Fan Yang and Per Frivik, ETH Zurich.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Custom navigation environment with observation delay support."""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.envs import ManagerBasedRLEnv

from instinct_nav.navigation.mdp.delay_manager import ObservationDelayManager
from instinct_nav.navigation.success_tracking import classify_navigation_success

if TYPE_CHECKING:
    from instinct_nav.navigation.navigation_env_cfg import NavigationEnvCfg


class NavigationEnv(ManagerBasedRLEnv):
    """Navigation environment with observation delay manager.

    This environment extends the base ManagerBasedRLEnv to add support for
    simulating sensor delays. The delay_manager is created before managers
    are loaded, ensuring observations can access it during initialization.
    """

    cfg: NavigationEnvCfg

    def load_managers(self):
        """Load managers with delay manager initialization.

        The delay_manager is created before other managers are loaded,
        ensuring observations can access it during initialization.
        """
        # Create the delay manager before loading other managers
        # At this point, self.num_envs and self.device are available via scene
        self.delay_manager = ObservationDelayManager(
            cfg=self.cfg.delay_cfg,
            num_envs=self.num_envs,
            device=self.device,
        )

        # Now load all other managers
        super().load_managers()

    def _reset_idx(self, env_ids):
        """Record the strict outcome separately from the legacy SRU success metric."""
        if hasattr(self, "termination_manager") and hasattr(self, "command_manager"):
            ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
            result_ids, results = classify_navigation_success(
                self.termination_manager._term_dones,
                list(self.termination_manager.active_terms),
                ids,
            )
            if result_ids.numel() > 0:
                goal = self.command_manager._terms.get("robot_goal")
                tracker = getattr(goal, "strict_success_tracker", None)
                if tracker is not None:
                    tracker.record_values(results, result_ids)
        super()._reset_idx(env_ids)

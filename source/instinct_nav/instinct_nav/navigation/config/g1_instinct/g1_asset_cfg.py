from __future__ import annotations

import copy
from pathlib import Path

from instinctlab.assets.unitree_g1 import G1_29DOF_TORSOBASE_POPSICLE_CFG, beyondmimic_action_scale

_PROJECT_ROOT = Path(__file__).resolve().parents[6]
_G1_WITH_SHOE_URDF = (
    _PROJECT_ROOT
    / "external"
    / "instinctlab"
    / "instinctlab"
    / "tasks"
    / "parkour"
    / "urdf"
    / "g1_29dof_torsoBase_popsicle_with_shoe.urdf"
)

# Local navigation copy of the G1-with-shoe robot asset used by the frozen
# InstinctLab parkour low-level policy. Importing the original parkour task cfg
# pulls in terrain and motion-reference config that navigation does not use, so
# keep this minimal asset construction inside instinct_nav.
G1_WITH_SHOE_CFG = copy.deepcopy(G1_29DOF_TORSOBASE_POPSICLE_CFG)
G1_WITH_SHOE_CFG.spawn.merge_fixed_joints = True
G1_WITH_SHOE_CFG.init_state.pos = (0.0, 0.0, 0.9)
G1_WITH_SHOE_CFG.spawn.asset_path = str(_G1_WITH_SHOE_URDF)

__all__ = ["G1_WITH_SHOE_CFG", "beyondmimic_action_scale"]

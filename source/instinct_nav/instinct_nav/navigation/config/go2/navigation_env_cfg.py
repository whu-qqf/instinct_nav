# Copyright (c) 2022-2025, Fan Yang and Per Frivik, ETH Zurich.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Go2-specific configuration for navigation environment.

Go2 is a quadruped robot from Unitree with 12 actuated joints (hip/thigh/calf × 4 legs).
It has no wheels, so the low-level policy outputs only joint position targets.

The low-level policy must be trained and exported using the standard IsaacLab Go2
velocity task before this navigation config can be used. See the handoff document at
docs/go2_low_level_policy_handoff_2026-06-21.md for the full training procedure.

Expected policy output: 12 joint position targets for [hip, thigh, calf] × 4 legs.
The rough policy expects 235-dim observations (including height_scan at 187 dims).
"""

import os

from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg, ObservationTermCfg as ObsTerm
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from instinct_nav.navigation.navigation_env_cfg import NavigationEnvCfg
import instinct_nav.navigation.mdp as mdp
from instinct_nav.navigation.assets import ISAACLAB_NAV_TASKS_ASSETS_DIR

from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG  # isort: skip

# Go2 joint names: 4 legs × 3 joints (hip, thigh, calf), no wheels
GO2_LEG_JOINT_NAMES = [".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"]

# Path to the Go2 low-level policy file.
# Rough policy: 235-dim obs (with height_scan), 12-dim action, MLP, no normalization.
GO2_LOW_LEVEL_POLICY_PATH = os.path.join(
    ISAACLAB_NAV_TASKS_ASSETS_DIR, "Policies", "locomotion", "go2", "policy_go2.pt"
)


@configclass
class Go2NavigationEnvCfg(NavigationEnvCfg):
    """Navigation environment configuration for Unitree Go2."""

    def __post_init__(self):
        super().__post_init__()

        from instinct_nav.navigation.mdp.observations import initialize_depth_noise_generator
        from instinct_nav.navigation.mdp.depth_utils.camera_config import get_camera_config

        initialize_depth_noise_generator(robot_name="go2", use_jit_precompiled=False)

        camera_config = get_camera_config("go2")
        CAMERA_RESOLUTION = camera_config.resolution

        # Robot asset: Unitree Go2
        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Camera mount on Go2 base (same prim path as other robots)
        self.scene.raycast_camera.prim_path = "{ENV_REGEX_NS}/Robot/base"
        self.scene.raycast_camera.offset.pos = (0.35, 0.0, 0.15)

        # Height scanner on Go2 base (critic scanner)
        self.scene.height_scanner_critic.prim_path = "{ENV_REGEX_NS}/Robot/base"

        # Low-level policy height scanner: matches Go2 rough velocity task training config.
        # GridPatternCfg(resolution=0.1, size=[1.6, 1.0]) → 187 height samples.
        self.scene.height_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base",
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
            ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )

        # Actions: replace AoW-D/B2W joint configs with Go2-specific ones.
        # The Go2 low-level policy outputs 12 joint position targets (hip/thigh/calf × 4 legs).
        # Go2 has no wheels, so the velocity action has zero joints.
        # Scale 0.25 matches the default Go2 velocity task action scale.
        self.actions.velocity_command.low_level_position_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=GO2_LEG_JOINT_NAMES,
            scale=0.25,
            use_default_offset=True,
        )
        self.actions.velocity_command.low_level_velocity_action = mdp.JointVelocityActionCfg(
            asset_name="robot",
            joint_names=[],
            scale=1.0,
            use_default_offset=True,
        )
        self.actions.velocity_command.low_level_policy_file = GO2_LOW_LEVEL_POLICY_PATH
        # Policy scaling: [vx, vy, omega] — tuned for Go2 velocity range.
        # Start with [1.0, 1.0, 1.0] and adjust based on observed tracking performance.
        self.actions.velocity_command.policy_scaling = [1.0, 1.0, 1.0]

        # Reward: joint acceleration penalty — uses Go2 leg joints only (no wheels)
        self.rewards.joint_acc_l2_joint.params = {
            "asset_cfg": SceneEntityCfg("robot", joint_names=GO2_LEG_JOINT_NAMES),
        }

        # Terminations: base contact — Go2 uses "base" for the main body
        self.terminations.base_contact.params = {
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base", ".*_hip", ".*_thigh"]),
            "threshold": 1.0,
        }

        # Low-pass filter: slightly different range for Go2
        self.events.randomize_low_pass_filter_alpha.params = {
            "alpha_range": (0.2, 0.7),
            "action_term": "velocity_command",
            "per_dimension": True,
            "alpha_range_vx": (0.2, 0.7),
            "alpha_range_vy": (0.2, 0.7),
            "alpha_range_omega": (0.2, 0.7),
        }

        # Low-level policy observation noise: match the Go2 velocity task training
        # distribution so the exported policy sees observations from the same distribution
        # it was trained on.
        self.observations.low_level_policy.base_lin_vel.noise = Unoise(n_min=-0.1, n_max=0.1)
        self.observations.low_level_policy.base_ang_vel.noise = Unoise(n_min=-0.2, n_max=0.2)
        self.observations.low_level_policy.projected_gravity.noise = Unoise(n_min=-0.05, n_max=0.05)
        self.observations.low_level_policy.joint_pos.noise = Unoise(n_min=-0.01, n_max=0.01)
        # joint_vel noise is already ±1.5 in both, matching

        # Add height_scan to low_level_policy observations (required by Go2 rough policy).
        # Must appear AFTER actions in the observation order to match the training layout.
        self.observations.low_level_policy.height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

        # Update low-level policy height scanner period to match low-level frequency (50Hz)
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.low_level_decimation * self.sim.dt

        # Visibility map: explicit spatial memory from depth camera
        self.observations.policy.visibility_map = ObsTerm(
            func=mdp.local_visibility_map,
            params={
                "sensor_cfg": SceneEntityCfg("raycast_camera"),
                "size_m": 6.4,
                "grid_size": 64,
                "depth_min_m": 0.5,
                "depth_max_m": 8.0,
                "pixel_stride": 2,
                "decay": 1.0,
            },
        )

        # Terrain settings (same as other robots)
        self.scene.terrain.max_init_terrain_level = 10
        self.scene.terrain.terrain_generator.difficulty_range = [0.5, 1.0]
        self.scene.terrain.terrain_generator.curriculum = False


@configclass
class Go2NavigationEnvCfg_DEV(Go2NavigationEnvCfg):
    """Development config: single environment, simpler terrain."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator.num_rows = 1
        self.scene.terrain.terrain_generator.num_cols = 30
        self.scene.terrain.max_init_terrain_level = 10
        self.scene.terrain.terrain_generator.difficulty_range = [0.4, 0.8]
        self.scene.terrain.terrain_generator.curriculum = True
        self.scene.num_envs = 1


@configclass
class Go2NavigationEnvCfg_PLAY(Go2NavigationEnvCfg):
    """Play config: small number of envs, no randomization."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 20
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 2
            self.scene.terrain.terrain_generator.num_cols = 2

        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None

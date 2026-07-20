# Copyright (c) 2022-2025, Fan Yang and Per Frivik, ETH Zurich.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Unitree G1 navigation config using a frozen SONIC-style low-level controller."""

import os
from pathlib import Path

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.sim import UsdFileCfg
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise


import instinct_nav.navigation.mdp as mdp
from instinct_nav.navigation.navigation_env_cfg import NavigationEnvCfg

_PROJECT_ROOT = Path(__file__).resolve().parents[6]
_GROOT_ROOT = Path(os.environ.get("INSTINCT_NAV_GROOT_ROOT", str(_PROJECT_ROOT / "source" / "GR00T-WholeBodyControl")))
GEAR_SONIC_DEPLOY_DIR = str(_GROOT_ROOT / "gear_sonic_deploy")
MOTIONBRICKS_ROOT = str(_GROOT_ROOT / "motionbricks")
GEAR_SONIC_G1_USD = str(_GROOT_ROOT / "gear_sonic" / "data" / "robots" / "g1" / "g1_29dof_rev_1_0" / "g1_29dof_rev_1_0.usd")
SONIC_PLANNER_ONNX = f"{GEAR_SONIC_DEPLOY_DIR}/planner/target_vel/V2/planner_sonic.onnx"
SONIC_ENCODER_ONNX = f"{GEAR_SONIC_DEPLOY_DIR}/policy/release/model_encoder.onnx"
SONIC_DECODER_ONNX = f"{GEAR_SONIC_DEPLOY_DIR}/policy/release/model_decoder.onnx"
INSTINCT_G1_CHECKPOINT = os.environ.get(
    "SRU_G1_INSTINCT_CHECKPOINT",
    str(_PROJECT_ROOT / "artifacts" / "low_level" / "g1_parkour" / "model_30000.pt"),
)
INSTINCT_RL_ROOT = os.environ.get("SRU_INSTINCT_RL_ROOT", str(_PROJECT_ROOT / "external" / "instinct_rl"))

G1_SONIC_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

G1_SONIC_DEFAULT_JOINT_POS = {
    "left_hip_pitch_joint": -0.312,
    "left_hip_roll_joint": 0.0,
    "left_hip_yaw_joint": 0.0,
    "left_knee_joint": 0.669,
    "left_ankle_pitch_joint": -0.363,
    "left_ankle_roll_joint": 0.0,
    "right_hip_pitch_joint": -0.312,
    "right_hip_roll_joint": 0.0,
    "right_hip_yaw_joint": 0.0,
    "right_knee_joint": 0.669,
    "right_ankle_pitch_joint": -0.363,
    "right_ankle_roll_joint": 0.0,
    "waist_yaw_joint": 0.0,
    "waist_roll_joint": 0.0,
    "waist_pitch_joint": 0.0,
    "left_shoulder_pitch_joint": 0.2,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": 0.6,
    "left_wrist_roll_joint": 0.0,
    "left_wrist_pitch_joint": 0.0,
    "left_wrist_yaw_joint": 0.0,
    "right_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": 0.6,
    "right_wrist_roll_joint": 0.0,
    "right_wrist_pitch_joint": 0.0,
    "right_wrist_yaw_joint": 0.0,
}

G1_SONIC_29DOF_CFG = ArticulationCfg(
    spawn=UsdFileCfg(
        usd_path=GEAR_SONIC_G1_USD,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.74),
        joint_pos=G1_SONIC_DEFAULT_JOINT_POS,
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
                "waist_yaw_joint",
                "waist_roll_joint",
                "waist_pitch_joint",
            ],
            effort_limit_sim=300,
            stiffness=120.0,
            damping=5.0,
            armature=0.01,
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit_sim=20,
            stiffness=20.0,
            damping=2.0,
            armature=0.01,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_roll_joint",
                ".*_wrist_pitch_joint",
                ".*_wrist_yaw_joint",
            ],
            effort_limit_sim=80,
            stiffness=40.0,
            damping=10.0,
            armature=0.01,
        ),
    },
)


@configclass
class G1SonicNavigationEnvCfg(NavigationEnvCfg):
    """Navigation environment for Unitree G1 with a frozen low-level locomotion prior."""

    def __post_init__(self):
        super().__post_init__()

        from instinct_nav.navigation.mdp.observations import initialize_depth_noise_generator
        from instinct_nav.navigation.mdp.depth_utils.camera_config import get_camera_config

        initialize_depth_noise_generator(robot_name="g1", use_jit_precompiled=False)
        get_camera_config("g1", use_default_fallback=True)

        self.commands.robot_goal.debug_vis = False

        self.scene.robot = G1_SONIC_29DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.raycast_camera.prim_path = "{ENV_REGEX_NS}/Robot/torso_link"
        self.scene.raycast_camera.offset.pos = (0.35, 0.0, 0.25)
        self.scene.height_scanner_critic.prim_path = "{ENV_REGEX_NS}/Robot/torso_link"
        self.scene.height_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/torso_link",
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
            ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )

        self.actions.velocity_command = mdp.SonicNavigationSE2ActionCfg(
            asset_name="robot",
            low_level_position_action=mdp.JointPositionActionCfg(
                asset_name="robot",
                joint_names=G1_SONIC_JOINT_NAMES,
                scale=1.0,
                use_default_offset=False,
            ),
            low_level_decimation=4,
            policy_scaling=[1.0, 1.0, 1.0],
            use_raw_actions=True,
            policy_distr_type="gaussian",
            enable_low_pass_filter=True,
            low_pass_filter_alpha=0.5,
            max_speed=1.2,
            backend=os.environ.get("SRU_G1_LOW_LEVEL_BACKEND", "motionbricks"),
            planner_path=SONIC_PLANNER_ONNX,
            encoder_path=SONIC_ENCODER_ONNX,
            decoder_path=SONIC_DECODER_ONNX,
            motionbricks_root=MOTIONBRICKS_ROOT,
            instinct_checkpoint_path=INSTINCT_G1_CHECKPOINT,
            instinct_rl_root=INSTINCT_RL_ROOT,
            instinct_depth_mode=os.environ.get("SRU_G1_INSTINCT_DEPTH_MODE", "strict"),
        )

        self.rewards.joint_acc_l2_joint.params = {
            "asset_cfg": SceneEntityCfg("robot", joint_names=G1_SONIC_JOINT_NAMES),
        }
        self.terminations.base_contact.params = {
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["torso_link", "pelvis"]),
            "threshold": 1.0,
        }
        self.events.randomize_low_pass_filter_alpha.params = {
            "alpha_range": (0.2, 0.7),
            "action_term": "velocity_command",
            "per_dimension": True,
            "alpha_range_vx": (0.2, 0.7),
            "alpha_range_vy": (0.2, 0.7),
            "alpha_range_omega": (0.2, 0.7),
        }

        self.observations.low_level_policy.height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.low_level_decimation * self.sim.dt

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

        self.scene.terrain.max_init_terrain_level = 10
        self.scene.terrain.terrain_generator.difficulty_range = [0.5, 1.0]
        self.scene.terrain.terrain_generator.curriculum = False


@configclass
class G1SonicNavigationEnvCfg_DEV(G1SonicNavigationEnvCfg):
    """Development config: one environment, shorter terrain grid."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator.num_rows = 1
        self.scene.terrain.terrain_generator.num_cols = 30
        self.scene.terrain.max_init_terrain_level = 10
        self.scene.terrain.terrain_generator.difficulty_range = [0.4, 0.8]
        self.scene.terrain.terrain_generator.curriculum = True
        self.scene.num_envs = 1


@configclass
class G1SonicNavigationEnvCfg_PLAY(G1SonicNavigationEnvCfg):
    """Play config: small number of environments, no randomization."""

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

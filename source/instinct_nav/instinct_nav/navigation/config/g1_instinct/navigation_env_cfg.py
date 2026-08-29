from __future__ import annotations

import copy
import math
import os
from pathlib import Path

import isaaclab.sim as sim_utils

from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.sensors.ray_caster.multi_mesh_ray_caster_cfg import MultiMeshRayCasterCfg
from isaaclab.sensors.ray_caster.patterns import PinholeCameraPatternCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import instinct_nav.navigation.mdp as nav_mdp
from instinct_nav.navigation.navigation_env_cfg import NavigationEnvCfg
from instinct_nav.terrains import RICH_MAZE_TERRAIN_CFG

from .g1_asset_cfg import G1_WITH_SHOE_CFG, beyondmimic_action_scale
from instinctlab.sensors import NoisyGroupedRayCasterCameraCfg
from instinctlab.utils.noise import CropAndResizeCfg, DepthNormalizationCfg, GaussianBlurNoiseCfg

_PROJECT_ROOT = Path(__file__).resolve().parents[6]
INSTINCT_G1_CHECKPOINT = os.environ.get(
    "SRU_G1_INSTINCT_CHECKPOINT",
    str(_PROJECT_ROOT / "artifacts" / "low_level" / "g1_parkour" / "model_30000.pt"),
)
INSTINCT_RL_ROOT = os.environ.get("SRU_INSTINCT_RL_ROOT", str(_PROJECT_ROOT / "external" / "instinct_rl"))
OFFICIAL_PARKOUR_DIR = Path(os.environ.get(
    "SRU_G1_OFFICIAL_PARKOUR_DIR",
    str(_PROJECT_ROOT / "data&model" / "checkpoints" / "parkour_onboard_preview_stair"),
))
OFFICIAL_PARKOUR_ACTOR_ONNX = str(OFFICIAL_PARKOUR_DIR / "exported" / "actor.onnx")
OFFICIAL_PARKOUR_DEPTH_ONNX = str(OFFICIAL_PARKOUR_DIR / "exported" / "0-depth_encoder.onnx")



@configclass
class G1InstinctNavigationEnvCfg(NavigationEnvCfg):
    """SRU navigation task with the exact frozen InstinctLab G1 parkour low-level policy interface."""

    def __post_init__(self):
        from instinct_nav.navigation.mdp.observations import initialize_depth_noise_generator

        super().__post_init__()
        initialize_depth_noise_generator(robot_name="g1", use_jit_precompiled=False)

        self.commands.base_velocity = nav_mdp.InjectedVelocityCommandCfg()
        self.commands.robot_goal.debug_vis = False
        self.commands.robot_goal.sync_robot_root_to_spawn_origin = True

        self.scene.robot = copy.deepcopy(G1_WITH_SHOE_CFG).replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.merge_fixed_joints = True
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.9)

        self.events.reset_base.params = {
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (-0.1, 0.1)},
            "velocity_range": {
                "x": (-0.2, 0.2),
                "y": (-0.2, 0.2),
                "z": (-0.2, 0.2),
                "roll": (-0.2, 0.2),
                "pitch": (-0.2, 0.2),
                "yaw": (-0.2, 0.2),
            },
        }
        self.events.reset_robot_joints.func = nav_mdp.reset_joints_by_offset
        self.events.reset_robot_joints.params = {"position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0)}
        self.events.randomize_camera_height = None
        self.events.randomize_action_scale = None
        self.events.reset_delay_buffer = None
        self.events.randomize_low_pass_filter_alpha = None
        self.events.push_robot = None

        # Frozen InstinctLab G1 parkour low-level policy camera.
        #
        # Keep this config local to instinct_nav instead of deep-copying
        # the original parkour task config. The low-level checkpoint expects this
        # NoisyGroupedRayCasterCamera stream: raw 64x36 depth, crop/resize to
        # 18x32, 8 delayed history frames in InstinctLowLevelPolicyCfg.
        self.scene.camera = NoisyGroupedRayCasterCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/torso_link",
            mesh_prim_paths=["/World/ground"],
            ray_alignment="yaw",
            pattern_cfg=PinholeCameraPatternCfg(
                focal_length=1.0,
                horizontal_aperture=2 * math.tan(math.radians(89.51) / 2),
                vertical_aperture=2 * math.tan(math.radians(58.29) / 2),
                width=64,
                height=36,
            ),
            debug_vis=False,
            data_types=["distance_to_image_plane"],
            update_period=0.02,
            depth_clipping_behavior="max",
            offset=NoisyGroupedRayCasterCameraCfg.OffsetCfg(
                pos=(0.0487988662332928, 0.01, 0.4378029937970051),
                rot=(0.9135367613482678, 0.004363309284746571, 0.4067366430758002, 0.0),
                convention="world",
            ),
            min_distance=0.1,
            noise_pipeline={
                "crop_and_resize": CropAndResizeCfg(crop_region=(18, 0, 16, 16)),
                "gaussian_blur": GaussianBlurNoiseCfg(kernel_size=3, sigma=1),
                "depth_normalization": DepthNormalizationCfg(
                    depth_range=(0.0, 2.5),
                    normalize=True,
                    output_range=(0.0, 1.0),
                ),
            },
            data_histories={"distance_to_image_plane_noised": 37},
        )

        self.scene.raycast_camera.prim_path = "{ENV_REGEX_NS}/Robot/torso_link"
        # Multi-bank terrain is imported as N per-slab sub-prims under /World/ground
        # (see terrains/patches.py:_create_terrain_prims_split). A base RayCaster only
        # binds the FIRST Mesh child it finds, so it would scan a single slab and
        # return bogus heights for robots on other slabs. MultiMeshRayCaster gathers
        # every child mesh under /World/ground and merges them, so it covers all slabs
        # (and is identical to a base RayCaster when the terrain is a single mesh).
        self.scene.height_scanner_critic = MultiMeshRayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/torso_link",
            offset=MultiMeshRayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
            ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(resolution=0.2, size=[10.0, 10.0]),
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )
        self.scene.height_scanner = MultiMeshRayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/torso_link",
            offset=MultiMeshRayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
            ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )

        self.actions.velocity_command = nav_mdp.InstinctFrozenNavigationSE2ActionCfg(
            asset_name="robot",
            low_level_position_action=nav_mdp.JointPositionActionCfg(
                asset_name="robot",
                joint_names=[".*"],
                scale=beyondmimic_action_scale,
                use_default_offset=True,
            ),
            low_level_decimation=4,
            policy_scaling=[1.0, 1.0, 1.0],
            command_min=[0.0, 0.0, -1.0],
            command_max=[1.0, 0.0, 1.0],
            use_raw_actions=True,
            policy_distr_type="gaussian",
            enable_low_pass_filter=True,
            low_pass_filter_alpha=0.5,
            observation_group="instinct_low_level_policy",
            command_name="base_velocity",
            checkpoint_path=INSTINCT_G1_CHECKPOINT,
            instinct_rl_root=INSTINCT_RL_ROOT,
        )

        self.observations.instinct_low_level_policy = InstinctLowLevelPolicyCfg()

        self.rewards.joint_acc_l2_joint.params = {
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
        }
        self.terminations.base_contact.params = {
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="torso_link"),
            "threshold": 1.0,
        }
        self.terminations.large_pitch_angle.func = nav_mdp.bad_orientation
        self.terminations.large_pitch_angle.params = {"limit_angle": 1.0}
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.low_level_decimation * self.sim.dt

        # Match the original SRU navigation terrain start difficulty.
        self.scene.terrain.max_init_terrain_level = 10
        self.scene.terrain.terrain_generator.difficulty_range = [0.5, 1.0]
        self.scene.terrain.terrain_generator.curriculum = False



@configclass
class InstinctLowLevelPolicyCfg(ObsGroup):
    base_ang_vel = ObsTerm(
        func=nav_mdp.base_ang_vel,
        noise=Unoise(n_min=-0.2, n_max=0.2),
        history_length=8,
        flatten_history_dim=True,
        scale=0.25,
    )
    projected_gravity = ObsTerm(
        func=nav_mdp.projected_gravity,
        noise=Unoise(n_min=-0.05, n_max=0.05),
        history_length=8,
        flatten_history_dim=True,
    )
    velocity_commands = ObsTerm(
        func=nav_mdp.generated_commands,
        history_length=8,
        flatten_history_dim=True,
        params={"command_name": "base_velocity"},
        noise=None,
    )
    joint_pos = ObsTerm(
        func=nav_mdp.joint_pos_rel,
        noise=Unoise(n_min=-0.01, n_max=0.01),
        history_length=8,
        flatten_history_dim=True,
    )
    joint_vel = ObsTerm(
        func=nav_mdp.joint_vel_rel,
        noise=Unoise(n_min=-0.5, n_max=0.5),
        scale=0.05,
        history_length=8,
        flatten_history_dim=True,
    )
    actions = ObsTerm(
        func=nav_mdp.last_low_level_action,
        params={"action_term": "velocity_command"},
        history_length=8,
        flatten_history_dim=True,
    )
    depth_image = ObsTerm(
        func=nav_mdp.delayed_visualizable_image,
        params={
            "data_type": "distance_to_image_plane_noised_history",
            "sensor_cfg": SceneEntityCfg("camera"),
            "history_skip_frames": 5,
            "num_output_frames": 8,
            "delayed_frame_ranges": (0, 1),
            "debug_vis": False,
        },
        noise=None,
    )

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = False


@configclass
class G1InstinctNavigationEnvCfg_DEV(G1InstinctNavigationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.terrain.terrain_generator.num_rows = 1
        self.scene.terrain.terrain_generator.num_cols = 8
        self.scene.terrain.max_init_terrain_level = 0
        self.scene.terrain.terrain_generator.difficulty_range = [0.0, 0.0]
        self.scene.terrain.terrain_generator.curriculum = False


@configclass
class G1InstinctNavigationEnvCfg_PLAY(G1InstinctNavigationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 20
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class G1InstinctNavigationEnvCfg_VIS(G1InstinctNavigationEnvCfg):
    """G1 navigation config with actor-side local visibility-map observation enabled."""

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.visibility_map = ObsTerm(
            func=nav_mdp.local_visibility_map,
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
        # Reuse the existing reset-delay event slot, but only reset visibility-map memory.
        self.events.reset_delay_buffer = EventTerm(func=nav_mdp.reset_visibility_map, mode="reset")


@configclass
class G1InstinctNavigationEnvCfg_VIS_DEV(G1InstinctNavigationEnvCfg_VIS):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.terrain.terrain_generator.num_rows = 1
        self.scene.terrain.terrain_generator.num_cols = 8
        self.scene.terrain.max_init_terrain_level = 0
        self.scene.terrain.terrain_generator.difficulty_range = [0.0, 0.0]
        self.scene.terrain.terrain_generator.curriculum = False


@configclass
class G1InstinctNavigationEnvCfg_VIS_PLAY(G1InstinctNavigationEnvCfg_VIS):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 20
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class G1InstinctNavigationEnvCfg_SCENE_TOKEN(G1InstinctNavigationEnvCfg):
    """G1 navigation config with actor-side scene tokens derived from local visibility memory."""

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.scene_tokens = ObsTerm(
            func=nav_mdp.scene_tokens_from_visibility_map,
            params={
                "sensor_cfg": SceneEntityCfg("raycast_camera"),
                "size_m": 6.4,
                "grid_size": 64,
                "token_grid_size": 8,
                "depth_min_m": 0.5,
                "depth_max_m": 8.0,
                "pixel_stride": 2,
                "decay": 1.0,
            },
        )
        self.events.reset_delay_buffer = EventTerm(func=nav_mdp.reset_visibility_map, mode="reset")


@configclass
class G1InstinctNavigationEnvCfg_SCENE_TOKEN_DEV(G1InstinctNavigationEnvCfg_SCENE_TOKEN):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.terrain.terrain_generator.num_rows = 1
        self.scene.terrain.terrain_generator.num_cols = 8
        self.scene.terrain.max_init_terrain_level = 0
        self.scene.terrain.terrain_generator.difficulty_range = [0.0, 0.0]
        self.scene.terrain.terrain_generator.curriculum = False


@configclass
class G1InstinctNavigationEnvCfg_SCENE_TOKEN_PLAY(G1InstinctNavigationEnvCfg_SCENE_TOKEN):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 20
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class G1InstinctNavigationEnvCfg_SCENE_TOKEN_RISK(G1InstinctNavigationEnvCfg):
    """G1 navigation config with 9D scene-risk tokens for actor and critic."""

    def __post_init__(self):
        super().__post_init__()
        scene_token_term = ObsTerm(
            func=nav_mdp.scene_tokens_risk_from_visibility_map,
            params={
                "sensor_cfg": SceneEntityCfg("raycast_camera"),
                "size_m": 6.4,
                "grid_size": 64,
                "token_grid_size": 8,
                "depth_min_m": 0.5,
                "depth_max_m": 8.0,
                "pixel_stride": 2,
                "decay": 1.0,
            },
        )
        self.observations.policy.scene_tokens = scene_token_term
        self.observations.critic.scene_tokens = scene_token_term
        self.events.reset_delay_buffer = EventTerm(func=nav_mdp.reset_visibility_map, mode="reset")


@configclass
class G1InstinctNavigationEnvCfg_SCENE_TOKEN_RISK_DEV(G1InstinctNavigationEnvCfg_SCENE_TOKEN_RISK):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.terrain.terrain_generator.num_rows = 1
        self.scene.terrain.terrain_generator.num_cols = 8
        self.scene.terrain.max_init_terrain_level = 0
        self.scene.terrain.terrain_generator.difficulty_range = [0.0, 0.0]
        self.scene.terrain.terrain_generator.curriculum = False


@configclass
class G1InstinctNavigationEnvCfg_SCENE_TOKEN_RISK_PLAY(G1InstinctNavigationEnvCfg_SCENE_TOKEN_RISK):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 20
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class G1InstinctNavigationEnvCfg_SCENE_TOKEN_RISK_NOGOAL(G1InstinctNavigationEnvCfg):
    """G1 navigation config with 6D scene-risk tokens without goal-relative features."""

    def __post_init__(self):
        super().__post_init__()
        scene_token_term = ObsTerm(
            func=nav_mdp.scene_tokens_risk_no_goal_from_visibility_map,
            params={
                "sensor_cfg": SceneEntityCfg("raycast_camera"),
                "size_m": 6.4,
                "grid_size": 64,
                "token_grid_size": 8,
                "depth_min_m": 0.5,
                "depth_max_m": 8.0,
                "pixel_stride": 2,
                "decay": 1.0,
            },
        )
        self.observations.policy.scene_tokens = scene_token_term
        self.observations.critic.scene_tokens = scene_token_term
        self.events.reset_delay_buffer = EventTerm(func=nav_mdp.reset_visibility_map, mode="reset")


@configclass
class G1InstinctNavigationEnvCfg_SCENE_TOKEN_RISK_NOGOAL_DEV(G1InstinctNavigationEnvCfg_SCENE_TOKEN_RISK_NOGOAL):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.terrain.terrain_generator.num_rows = 1
        self.scene.terrain.terrain_generator.num_cols = 8
        self.scene.terrain.max_init_terrain_level = 0
        self.scene.terrain.terrain_generator.difficulty_range = [0.0, 0.0]
        self.scene.terrain.terrain_generator.curriculum = False


@configclass
class G1InstinctNavigationEnvCfg_SCENE_TOKEN_RISK_NOGOAL_PLAY(G1InstinctNavigationEnvCfg_SCENE_TOKEN_RISK_NOGOAL):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 20
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None

@configclass
class G1InstinctNavigationEnvCfg_SCENE_TOKEN_RISK_NOGOAL_SHARED_LOWCAM(
    G1InstinctNavigationEnvCfg_SCENE_TOKEN_RISK_NOGOAL
):
    """Scene-token v3 using the frozen low-level policy camera as the only depth source.

    The low-level Instinct policy still consumes the camera's own noised/cropped
    history stream. The high-level SRU policy and critic read the same sensor's
    raw distance_to_image_plane stream, encode it with the SRU depth encoder,
    and build scene tokens from that same camera pose/intrinsics.
    """

    def __post_init__(self):
        super().__post_init__()

        shared_camera = SceneEntityCfg("camera")
        self.observations.policy.depth_image.params["sensor_cfg"] = shared_camera
        self.observations.critic.depth_image.params["sensor_cfg"] = shared_camera
        self.observations.policy.scene_tokens.params["sensor_cfg"] = shared_camera
        self.observations.critic.scene_tokens.params["sensor_cfg"] = shared_camera

        # This task intentionally uses one physical depth sensor. Removing the
        # inherited SRU raycast camera avoids creating a second camera stream.
        self.scene.raycast_camera = None

        # Keep the established SRU task unchanged; only replace its frozen low-level policy.
        self.actions.velocity_command.backend_type = "official_onnx"
        self.actions.velocity_command.actor_onnx_path = OFFICIAL_PARKOUR_ACTOR_ONNX
        self.actions.velocity_command.depth_encoder_onnx_path = OFFICIAL_PARKOUR_DEPTH_ONNX


@configclass
class G1InstinctNavigationEnvCfg_SCENE_TOKEN_RISK_NOGOAL_SHARED_LOWCAM_DEV(
    G1InstinctNavigationEnvCfg_SCENE_TOKEN_RISK_NOGOAL_SHARED_LOWCAM
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.terrain.terrain_generator.num_rows = 1
        self.scene.terrain.terrain_generator.num_cols = 8
        self.scene.terrain.max_init_terrain_level = 0
        self.scene.terrain.terrain_generator.difficulty_range = [0.0, 0.0]
        self.scene.terrain.terrain_generator.curriculum = False


@configclass
class G1InstinctNavigationEnvCfg_SCENE_TOKEN_RISK_NOGOAL_SHARED_LOWCAM_PLAY(
    G1InstinctNavigationEnvCfg_SCENE_TOKEN_RISK_NOGOAL_SHARED_LOWCAM
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 20
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None

@configclass
class G1InstinctNavigationEnvCfg_SCENE_TOKEN_RISK_NOGOAL_SHARED_LOWCAM_OFFICIAL(
    G1InstinctNavigationEnvCfg_SCENE_TOKEN_RISK_NOGOAL_SHARED_LOWCAM
):
    """Final scene-token task using the official stable parkour ONNX policy pair."""

    def __post_init__(self):
        super().__post_init__()
        action = self.actions.velocity_command
        action.backend_type = "official_onnx"
        action.actor_onnx_path = OFFICIAL_PARKOUR_ACTOR_ONNX
        action.depth_encoder_onnx_path = OFFICIAL_PARKOUR_DEPTH_ONNX


@configclass
class G1InstinctNavigationEnvCfg_SCENE_TOKEN_RISK_NOGOAL_SHARED_LOWCAM_OFFICIAL_DEV(
    G1InstinctNavigationEnvCfg_SCENE_TOKEN_RISK_NOGOAL_SHARED_LOWCAM_OFFICIAL
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 8
        self.scene.terrain.max_init_terrain_level = 0
        self.scene.terrain.terrain_generator.difficulty_range = [0.0, 0.0]
        self.scene.terrain.terrain_generator.curriculum = False


@configclass
class G1InstinctNavigationEnvCfg_SCENE_TOKEN_RISK_NOGOAL_SHARED_LOWCAM_OFFICIAL_PLAY(
    G1InstinctNavigationEnvCfg_SCENE_TOKEN_RISK_NOGOAL_SHARED_LOWCAM_OFFICIAL
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 20
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class G1InstinctNavigationEnvCfg_SCENE_TOKEN_SCORE_VIS_SHARED_LOWCAM_OFFICIAL(
    G1InstinctNavigationEnvCfg_SCENE_TOKEN_RISK_NOGOAL_SHARED_LOWCAM_OFFICIAL
):
    """Continuous visibility-confidence scene tokens with official parkour."""

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.scene_tokens.params["decay"] = 0.995
        self.observations.critic.scene_tokens.params["decay"] = 0.995


@configclass
class G1InstinctNavigationEnvCfg_SCENE_TOKEN_SCORE_VIS_SHARED_LOWCAM_OFFICIAL_DEV(
    G1InstinctNavigationEnvCfg_SCENE_TOKEN_SCORE_VIS_SHARED_LOWCAM_OFFICIAL
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 8

        self.scene.terrain.max_init_terrain_level = 0
        self.scene.terrain.terrain_generator.difficulty_range = [0.0, 0.0]
        self.scene.terrain.terrain_generator.curriculum = False

@configclass
class G1InstinctNavigationEnvCfg_SCENE_TOKEN_SCORE_VIS_SHARED_LOWCAM_OFFICIAL_PLAY(
    G1InstinctNavigationEnvCfg_SCENE_TOKEN_SCORE_VIS_SHARED_LOWCAM_OFFICIAL
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 20
        self.observations.policy.enable_corruption = False
        self.scene.env_spacing = 2.5
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class G1InstinctNavigationEnvCfg_DYNAMIC_OFFICIAL(
    G1InstinctNavigationEnvCfg_SCENE_TOKEN_SCORE_VIS_SHARED_LOWCAM_OFFICIAL
):
    """Stage-3 dynamic obstacle curriculum on top of ScoreVis shared-lowcam.

    Adds per-env scripted rigid obstacles that are ray-castable by the shared
    camera, reset away from spawn/goal, and drive obstacle-avoidance reward and
    termination. The low-level parkour policy stays frozen.
    """

    num_obstacles: int = 4
    obstacle_speed_range: tuple[float, float] = (0.3, 0.6)
    obstacle_pattern: str = "patrol"
    obstacle_half_extent_m: float = 2.0
    obstacle_min_start_dist: float = 1.5
    obstacle_max_start_dist: float = 3.2
    obstacle_goal_clearance_m: float = 1.0
    # Delay (seconds) after episode start before obstacles suddenly pop in ahead of the
    # robot. Obstacles are hidden (parked out of view) until this timer elapses, so they
    # appear mid-travel rather than existing at episode reset.
    obstacle_spawn_delay_range: tuple[float, float] = (2.0, 8.0)
    obstacle_half_height: float = 0.4
    # Physical XY half-extent of the obstacle cube (size 0.5 -> half 0.25). Collision is
    # measured from the obstacle *surface*, i.e. robot-to-obstacle-center minus this half-extent.
    obstacle_body_half_extent: float = 0.25
    # Robot body radius used for the geometric collision termination (distance from obstacle
    # surface to the robot root at which we declare a collision).
    obstacle_collision_radius: float = 0.3
    obstacle_penalty_radius: float = 0.8
    obstacle_auto_switch_success: float = 0.9
    obstacle_max_switch_iteration: int = 4504
    obstacle_env_steps_per_iteration: int = 16
    obstacle_switch_check_interval: int = 100
    # Fraction of environments that join the dynamic-obstacle curriculum once the switch
    # iteration is reached. A fixed random subset is chosen at startup (not per episode), so
    # ~half the envs keep training plain static navigation while the rest learn obstacle
    # avoidance -> prevents catastrophic forgetting and stabilizes the shared policy.
    obstacle_curriculum_env_fraction: float = 0.5
    # --- Graded (capability-driven) curriculum. Default off => legacy binary iteration
    # switch. When on, obstacles enable once nav clears obstacle_auto_switch_success and
    # difficulty ratchets up (speed/distance/count/behavior/fraction) as the policy clears
    # obstacle_advance_success. Only the *-Avoid- ablation turns this on. ---
    obstacle_graded_curriculum: bool = False
    obstacle_difficulty_step: float = 0.2
    obstacle_advance_success: float = 0.75
    obstacle_easy_speed_range: tuple[float, float] = (0.1, 0.2)
    obstacle_easy_min_start_dist: float = 3.5
    obstacle_easy_max_start_dist: float = 6.0
    obstacle_fraction_range: tuple[float, float] = (0.25, 0.9)
    # Look-ahead time-to-collision penalty horizon (seconds). Reward term weight defaults
    # to 0 on the base class (inert); the *-Avoid- ablation gives it a negative weight.
    obstacle_ttc_horizon: float = 2.0

    def __post_init__(self):
        super().__post_init__()

        # 4 obstacles per env across many envs increases the GPU narrowphase
        # contact load; enlarge the PhysX rigid patch buffer so contact reports
        # (base_contact / obstacle_collision) stay accurate.
        # RichTerrain + 4096 envs + tracked dynamic obstacle meshes can exceed the
        # default contact-patch pool during scene startup. Keep headroom for the
        # observed ~770k peak instead of allowing PhysX patch-buffer overflow.
        self.sim.physx.gpu_max_rigid_patch_count = 2**20

        obstacle_names = []
        for index in range(self.num_obstacles):
            name = f"obstacle_{index}"
            obstacle_names.append(name)
            setattr(
                self.scene,
                name,
                RigidObjectCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/Obstacle_{index}",
                    spawn=sim_utils.CuboidCfg(
                        size=(0.5, 0.5, 0.8),
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(
                            max_depenetration_velocity=1.0,
                            disable_gravity=True,
                        ),
                        mass_props=sim_utils.MassPropertiesCfg(mass=8.0),
                        # Keep the USD mesh visible to the ray caster, but disable
                        # physical contacts so scripted obstacles cannot shove the
                        # frozen parkour controller. Collision termination remains
                        # the explicit geometric check in dynamic_obstacles.py.
                        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                        physics_material=sim_utils.RigidBodyMaterialCfg(),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.35, 0.1)),
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5)),
                ),
            )

        # Let the shared camera ray-cast against the moving obstacles.
        extra_targets = [
            MultiMeshRayCasterCfg.RaycastTargetCfg(
                prim_expr=f"{{ENV_REGEX_NS}}/Obstacle_{index}",
                is_shared=False,
                track_mesh_transforms=True,
            )
            for index in range(self.num_obstacles)
        ]
        self.scene.camera.mesh_prim_paths = list(self.scene.camera.mesh_prim_paths) + extra_targets

        event_params = {
            "obstacle_names": obstacle_names,
            "speed_range": self.obstacle_speed_range,
            "pattern": self.obstacle_pattern,
            "half_extent_m": self.obstacle_half_extent_m,
            "min_start_dist": self.obstacle_min_start_dist,
            "max_start_dist": self.obstacle_max_start_dist,
            "goal_clearance_m": self.obstacle_goal_clearance_m,
            "obstacle_half_height": self.obstacle_half_height,
            "body_half_extent": self.obstacle_body_half_extent,
            "spawn_delay_range": self.obstacle_spawn_delay_range,
            "curriculum_env_fraction": self.obstacle_curriculum_env_fraction,
            "auto_switch_success": self.obstacle_auto_switch_success,
            "max_switch_iteration": self.obstacle_max_switch_iteration,
            "env_steps_per_iteration": self.obstacle_env_steps_per_iteration,
            "switch_check_interval": self.obstacle_switch_check_interval,
            "graded_curriculum": self.obstacle_graded_curriculum,
            "difficulty_step": self.obstacle_difficulty_step,
            "advance_success": self.obstacle_advance_success,
            "easy_speed_range": self.obstacle_easy_speed_range,
            "easy_min_start_dist": self.obstacle_easy_min_start_dist,
            "easy_max_start_dist": self.obstacle_easy_max_start_dist,
            "fraction_range": self.obstacle_fraction_range,
        }
        self.events.dynamic_obstacles_reset = EventTerm(
            func=nav_mdp.reset_dynamic_obstacles,
            mode="reset",
            params=dict(event_params),
        )
        self.events.dynamic_obstacles_update = EventTerm(
            func=nav_mdp.update_dynamic_obstacles,
            mode="interval",
            interval_range_s=(0.0, 0.0),
            params=dict(event_params),
        )

        self.rewards.obstacle_proximity = RewTerm(
            func=nav_mdp.obstacle_proximity,
            weight=-0.2,
            params={
                "obstacle_names": obstacle_names,
                "penalty_radius": self.obstacle_penalty_radius,
                "falloff": 0.25,
            },
        )
        self.rewards.obstacle_approach_penalty = RewTerm(
            func=nav_mdp.obstacle_approach_penalty,
            weight=-0.1,
            params={"obstacle_names": obstacle_names, "speed_scale": 1.0},
        )
        # Look-ahead TTC penalty. Weight 0 => inert on every existing task; the *-Avoid-
        # ablation raises the magnitude to teach anticipatory slowing/turning.
        self.rewards.obstacle_time_to_collision = RewTerm(
            func=nav_mdp.obstacle_time_to_collision,
            weight=0.0,
            params={"obstacle_names": obstacle_names, "ttc_horizon": self.obstacle_ttc_horizon},
        )
        self.terminations.obstacle_collision = DoneTerm(
            func=nav_mdp.obstacle_collision,
            params={
                "obstacle_names": obstacle_names,
                "collision_radius": self.obstacle_collision_radius,
                "body_half_extent": self.obstacle_body_half_extent,
            },
        )


@configclass
class G1InstinctNavigationEnvCfg_DYNAMIC_OFFICIAL_DEV(
    G1InstinctNavigationEnvCfg_DYNAMIC_OFFICIAL
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 8
        self.scene.terrain.max_init_terrain_level = 0
        self.scene.terrain.terrain_generator.difficulty_range = [0.0, 0.0]
        self.scene.terrain.terrain_generator.curriculum = False


@configclass
class G1InstinctNavigationEnvCfg_DYNAMIC_OFFICIAL_PLAY(
    G1InstinctNavigationEnvCfg_DYNAMIC_OFFICIAL
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 20
        self.observations.policy.enable_corruption = False
        self.scene.env_spacing = 2.5
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class G1InstinctNavigationEnvCfg_DYNAMIC_MAP_CROSS_OFFICIAL(G1InstinctNavigationEnvCfg_DYNAMIC_OFFICIAL):
    """Dynamic curriculum with learned current-to-history map cross-attention patches."""

    def __post_init__(self):
        super().__post_init__()
        map_patch_term = ObsTerm(
            func=nav_mdp.dynamic_map_patches_from_visibility_map,
            params={
                "sensor_cfg": SceneEntityCfg("camera"),
                "size_m": 6.4,
                "grid_size": 64,
                "token_grid_size": 8,
                "depth_min_m": 0.5,
                "depth_max_m": 8.0,
                "pixel_stride": 2,
                "decay": 0.995,
            },
        )
        self.observations.policy.scene_tokens = map_patch_term
        self.observations.critic.scene_tokens = map_patch_term


@configclass
class G1InstinctNavigationEnvCfg_DYNAMIC_MAP_CROSS_OFFICIAL_DEV(
    G1InstinctNavigationEnvCfg_DYNAMIC_MAP_CROSS_OFFICIAL
):
    def __post_init__(self):
        super().__post_init__()
        # Dev smoke exercises trajectory generation and motion immediately.
        self.obstacle_max_switch_iteration = 0
        self.events.dynamic_obstacles_reset.params["max_switch_iteration"] = 0
        self.events.dynamic_obstacles_update.params["max_switch_iteration"] = 0
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 8
        self.scene.terrain.max_init_terrain_level = 0
        self.scene.terrain.terrain_generator.difficulty_range = [0.0, 0.0]
        self.scene.terrain.terrain_generator.curriculum = False


@configclass
class G1InstinctNavigationEnvCfg_DYNAMIC_MAP_CROSS_OFFICIAL_PLAY(
    G1InstinctNavigationEnvCfg_DYNAMIC_MAP_CROSS_OFFICIAL
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 20
        self.observations.policy.enable_corruption = False
        self.scene.env_spacing = 2.5
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class G1InstinctNavigationEnvCfg_DYNAMIC_MAP_CROSS_RICH_TERRAIN_OFFICIAL(
    G1InstinctNavigationEnvCfg_DYNAMIC_MAP_CROSS_OFFICIAL
):
    """DynamicMapCross policy task on the expanded rich navigation terrain."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = copy.deepcopy(RICH_MAZE_TERRAIN_CFG)


@configclass
class G1InstinctNavigationEnvCfg_DYNAMIC_MAP_CROSS_RICH_TERRAIN_OFFICIAL_DEV(
    G1InstinctNavigationEnvCfg_DYNAMIC_MAP_CROSS_RICH_TERRAIN_OFFICIAL
):
    def __post_init__(self):
        super().__post_init__()
        self.obstacle_max_switch_iteration = 0
        self.events.dynamic_obstacles_reset.params["max_switch_iteration"] = 0
        self.events.dynamic_obstacles_update.params["max_switch_iteration"] = 0
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 8
        self.scene.terrain.max_init_terrain_level = 0
        self.scene.terrain.terrain_generator.difficulty_range = [0.0, 0.0]
        self.scene.terrain.terrain_generator.curriculum = False


@configclass
class G1InstinctNavigationEnvCfg_DYNAMIC_MAP_CROSS_RICH_TERRAIN_OFFICIAL_PLAY(
    G1InstinctNavigationEnvCfg_DYNAMIC_MAP_CROSS_RICH_TERRAIN_OFFICIAL
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 20
        self.observations.policy.enable_corruption = False
        self.scene.env_spacing = 2.5
        self.events.base_external_force_torque = None
        self.events.push_robot = None


def _apply_speed_clip_mapping(cfg) -> None:
    """Align the high-level velocity command with the frozen low-level's trained envelope.

    Two changes vs the v4 default:
    - command_max[0] (vx) 1.0 -> 0.8: never command a forward speed the InstinctLab parkour
      low-level was never trained for on stairs/gaps/boxes/slopes (it only saw vx in [0.45, 0.8]),
      which is a leading cause of out-of-distribution overspeed -> forward lurch -> large pitch.
    - policy_distr_type "gaussian" -> "gaussian_clip": drop the tanh squash for a linear+clamp
      map so a full stop (vx=0, the low-level's stand mode) and the speed ceiling are reachable
      with a finite policy mean, with uniform resolution across the walking band.
    command_min is unchanged ([0, 0, -1]); vy stays locked to 0 and wz stays [-1, 1].
    """
    cfg.actions.velocity_command.command_max = [0.8, 0.0, 1.0]
    cfg.actions.velocity_command.policy_distr_type = "gaussian_clip"


@configclass
class G1InstinctNavigationEnvCfg_DYNAMIC_MAP_CROSS_RICH_TERRAIN_OFFICIAL_SPEEDCLIP(
    G1InstinctNavigationEnvCfg_DYNAMIC_MAP_CROSS_RICH_TERRAIN_OFFICIAL
):
    """rich_terrain_v4 with the velocity command clipped to the low-level's trained envelope."""

    def __post_init__(self):
        super().__post_init__()
        _apply_speed_clip_mapping(self)


@configclass
class G1InstinctNavigationEnvCfg_DYNAMIC_MAP_CROSS_RICH_TERRAIN_OFFICIAL_SPEEDCLIP_DEV(
    G1InstinctNavigationEnvCfg_DYNAMIC_MAP_CROSS_RICH_TERRAIN_OFFICIAL_SPEEDCLIP
):
    def __post_init__(self):
        super().__post_init__()
        self.obstacle_max_switch_iteration = 0
        self.events.dynamic_obstacles_reset.params["max_switch_iteration"] = 0
        self.events.dynamic_obstacles_update.params["max_switch_iteration"] = 0
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 8
        self.scene.terrain.max_init_terrain_level = 0
        self.scene.terrain.terrain_generator.difficulty_range = [0.0, 0.0]
        self.scene.terrain.terrain_generator.curriculum = False


@configclass
class G1InstinctNavigationEnvCfg_DYNAMIC_MAP_CROSS_RICH_TERRAIN_OFFICIAL_SPEEDCLIP_PLAY(
    G1InstinctNavigationEnvCfg_DYNAMIC_MAP_CROSS_RICH_TERRAIN_OFFICIAL_SPEEDCLIP
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 20
        self.observations.policy.enable_corruption = False
        self.scene.env_spacing = 2.5
        self.events.base_external_force_torque = None
        self.events.push_robot = None


# =============================================================================
# SpeedClip + stair centerline guidance reward.
# On top of SpeedClip, enable the privileged stair-centerline shaping reward so the
# robot learns to approach a staircase front-and-centered and ascend up the middle
# instead of mounting from an exposed lateral edge (half-supported foot -> large pitch).
# The reward is 0 off stairs, so flat ground / platforms are unaffected (no false
# positives). Base RewardsCfg keeps weight 0.0, so v4 and every other task are untouched.
# =============================================================================

_STAIR_CENTERLINE_WEIGHT = 0.5


@configclass
class G1InstinctNavigationEnvCfg_DYNAMIC_MAP_CROSS_RICH_TERRAIN_OFFICIAL_SPEEDCLIP_CENTERLINE(
    G1InstinctNavigationEnvCfg_DYNAMIC_MAP_CROSS_RICH_TERRAIN_OFFICIAL_SPEEDCLIP
):
    """SpeedClip + stair centerline guidance reward."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.stair_centerline.weight = _STAIR_CENTERLINE_WEIGHT


@configclass
class G1InstinctNavigationEnvCfg_DYNAMIC_MAP_CROSS_RICH_TERRAIN_OFFICIAL_SPEEDCLIP_CENTERLINE_DEV(
    G1InstinctNavigationEnvCfg_DYNAMIC_MAP_CROSS_RICH_TERRAIN_OFFICIAL_SPEEDCLIP_DEV
):
    def __post_init__(self):
        super().__post_init__()
        self.rewards.stair_centerline.weight = _STAIR_CENTERLINE_WEIGHT


@configclass
class G1InstinctNavigationEnvCfg_DYNAMIC_MAP_CROSS_RICH_TERRAIN_OFFICIAL_SPEEDCLIP_CENTERLINE_PLAY(
    G1InstinctNavigationEnvCfg_DYNAMIC_MAP_CROSS_RICH_TERRAIN_OFFICIAL_SPEEDCLIP_PLAY
):
    def __post_init__(self):
        super().__post_init__()
        self.rewards.stair_centerline.weight = _STAIR_CENTERLINE_WEIGHT


# =============================================================================
# Original v4 tanh velocity mapping (command_max vx=1.0, policy_distr_type="gaussian")
# + stair centerline reward, WITHOUT SpeedClip. Ablation to isolate whether the
# large_pitch rate is driven by the SpeedClip mapping change or by the speed range
# itself: this run differs from SpeedClip-Centerline ONLY in the velocity mapping,
# so comparing large_pitch between the two answers "is it the speed mapping?".
# =============================================================================


@configclass
class G1InstinctNavigationEnvCfg_DYNAMIC_MAP_CROSS_RICH_TERRAIN_OFFICIAL_CENTERLINE(
    G1InstinctNavigationEnvCfg_DYNAMIC_MAP_CROSS_RICH_TERRAIN_OFFICIAL
):
    """v4 original tanh velocity mapping + stair centerline reward (no SpeedClip)."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.stair_centerline.weight = _STAIR_CENTERLINE_WEIGHT


@configclass
class G1InstinctNavigationEnvCfg_DYNAMIC_MAP_CROSS_RICH_TERRAIN_OFFICIAL_CENTERLINE_DEV(
    G1InstinctNavigationEnvCfg_DYNAMIC_MAP_CROSS_RICH_TERRAIN_OFFICIAL_DEV
):
    def __post_init__(self):
        super().__post_init__()
        self.rewards.stair_centerline.weight = _STAIR_CENTERLINE_WEIGHT


@configclass
class G1InstinctNavigationEnvCfg_DYNAMIC_MAP_CROSS_RICH_TERRAIN_OFFICIAL_CENTERLINE_PLAY(
    G1InstinctNavigationEnvCfg_DYNAMIC_MAP_CROSS_RICH_TERRAIN_OFFICIAL_PLAY
):
    def __post_init__(self):
        super().__post_init__()
        self.rewards.stair_centerline.weight = _STAIR_CENTERLINE_WEIGHT


# =============================================================================
# tanh-base centerline + "velocity into unobserved" penalty. Adds a penalty that
# discourages driving the base toward not-yet-observed ("black") cells of the
# vis_score map (occluded / outside FOV / beyond range), on top of the centerline
# guidance. Only meaningful on ScoreVis-family tasks that maintain the local
# visibility map (this DynamicMapCross base does). Weight starts modest.
# =============================================================================
_UNOBSERVED_PENALTY_WEIGHT = 0.2


@configclass
class G1InstinctNavigationEnvCfg_DYNAMIC_MAP_CROSS_RICH_TERRAIN_OFFICIAL_CENTERLINE_UNOBSERVED(
    G1InstinctNavigationEnvCfg_DYNAMIC_MAP_CROSS_RICH_TERRAIN_OFFICIAL_CENTERLINE
):
    """tanh centerline + penalize velocity pointing at unobserved vis_score cells."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.velocity_into_unobserved.weight = -_UNOBSERVED_PENALTY_WEIGHT


@configclass
class G1InstinctNavigationEnvCfg_DYNAMIC_MAP_CROSS_RICH_TERRAIN_OFFICIAL_CENTERLINE_UNOBSERVED_DEV(
    G1InstinctNavigationEnvCfg_DYNAMIC_MAP_CROSS_RICH_TERRAIN_OFFICIAL_CENTERLINE_DEV
):
    def __post_init__(self):
        super().__post_init__()
        self.rewards.velocity_into_unobserved.weight = -_UNOBSERVED_PENALTY_WEIGHT


@configclass
class G1InstinctNavigationEnvCfg_DYNAMIC_MAP_CROSS_RICH_TERRAIN_OFFICIAL_CENTERLINE_UNOBSERVED_PLAY(
    G1InstinctNavigationEnvCfg_DYNAMIC_MAP_CROSS_RICH_TERRAIN_OFFICIAL_CENTERLINE_PLAY
):
    def __post_init__(self):
        super().__post_init__()
        self.rewards.velocity_into_unobserved.weight = -_UNOBSERVED_PENALTY_WEIGHT


# =============================================================================
# tanh centerline + unobserved penalty + FAST dynamic-obstacle avoidance. Two
# changes on top of CENTERLINE_UNOBSERVED, both isolated to this ablation:
#   (P0) rebalance the avoidance reward so it is no longer ~1/180 of the goal
#        reward: raise obstacle_proximity/approach magnitudes and add a
#        look-ahead time-to-collision penalty (anticipatory slow/turn).
#   (P1) capability-driven graded curriculum: obstacles enable once nav is ready
#        and difficulty (speed/distance/count/behavior/env-fraction) ratchets up
#        as the policy clears advance_success, instead of a binary iteration switch.
# graded_curriculum defaults False on every other task, so this is fully isolated.
# =============================================================================
_AVOID_PROXIMITY_WEIGHT = 2.0
_AVOID_APPROACH_WEIGHT = 1.0
_AVOID_TTC_WEIGHT = 0.5


def _apply_avoid_rewards(cfg) -> None:
    cfg.rewards.obstacle_proximity.weight = -_AVOID_PROXIMITY_WEIGHT
    cfg.rewards.obstacle_approach_penalty.weight = -_AVOID_APPROACH_WEIGHT
    cfg.rewards.obstacle_time_to_collision.weight = -_AVOID_TTC_WEIGHT


@configclass
class G1InstinctNavigationEnvCfg_DYNAMIC_MAP_CROSS_RICH_TERRAIN_OFFICIAL_CENTERLINE_UNOBSERVED_AVOID(
    G1InstinctNavigationEnvCfg_DYNAMIC_MAP_CROSS_RICH_TERRAIN_OFFICIAL_CENTERLINE_UNOBSERVED
):
    """CENTERLINE_UNOBSERVED + rebalanced avoidance reward + graded obstacle curriculum."""

    obstacle_graded_curriculum: bool = True

    def __post_init__(self):
        super().__post_init__()
        _apply_avoid_rewards(self)


@configclass
class G1InstinctNavigationEnvCfg_DYNAMIC_MAP_CROSS_RICH_TERRAIN_OFFICIAL_CENTERLINE_UNOBSERVED_AVOID_DEV(
    G1InstinctNavigationEnvCfg_DYNAMIC_MAP_CROSS_RICH_TERRAIN_OFFICIAL_CENTERLINE_UNOBSERVED_DEV
):
    # The DEV chain already patches max_switch_iteration=0, so obstacles enable
    # immediately at difficulty 0 (single slow far obstacle) for smoke testing.
    obstacle_graded_curriculum: bool = True

    def __post_init__(self):
        super().__post_init__()
        _apply_avoid_rewards(self)


@configclass
class G1InstinctNavigationEnvCfg_DYNAMIC_MAP_CROSS_RICH_TERRAIN_OFFICIAL_CENTERLINE_UNOBSERVED_AVOID_PLAY(
    G1InstinctNavigationEnvCfg_DYNAMIC_MAP_CROSS_RICH_TERRAIN_OFFICIAL_CENTERLINE_UNOBSERVED_PLAY
):
    obstacle_graded_curriculum: bool = True

    def __post_init__(self):
        super().__post_init__()
        _apply_avoid_rewards(self)


# =============================================================================
# Pre-generated terrain banks. The network and observation shapes are unchanged:
# four independent RichTerrain column groups are cooked once at startup, then
# environments move to the scheduled group only when their episode resets.
# =============================================================================
_TERRAIN_BANK_COUNT = 4
_TERRAIN_BANK_SWITCH_INTERVAL = 1000


def _apply_terrain_banks(cfg, bank_count: int = _TERRAIN_BANK_COUNT) -> None:
    terrain_cfg = cfg.scene.terrain.terrain_generator
    base_cols = int(terrain_cfg.num_cols)
    terrain_cfg.num_cols = base_cols * bank_count
    goal_cfg = cfg.commands.robot_goal
    goal_cfg.terrain_bank_count = bank_count
    goal_cfg.terrain_bank_base_num_cols = base_cols
    goal_cfg.terrain_bank_switch_interval = _TERRAIN_BANK_SWITCH_INTERVAL
    goal_cfg.terrain_bank_env_steps_per_iteration = cfg.obstacle_env_steps_per_iteration


@configclass
class G1InstinctNavigationEnvCfg_DYNAMIC_MAP_CROSS_RICH_TERRAIN_OFFICIAL_CENTERLINE_UNOBSERVED_BANKS(
    G1InstinctNavigationEnvCfg_DYNAMIC_MAP_CROSS_RICH_TERRAIN_OFFICIAL_CENTERLINE_UNOBSERVED
):
    """Centerline + unobserved penalty with four pre-generated terrain banks."""

    obstacle_max_switch_iteration: int = 4000

    def __post_init__(self):
        super().__post_init__()
        _apply_terrain_banks(self)


@configclass
class G1InstinctNavigationEnvCfg_DYNAMIC_MAP_CROSS_RICH_TERRAIN_OFFICIAL_CENTERLINE_UNOBSERVED_BANKS_DEV(
    G1InstinctNavigationEnvCfg_DYNAMIC_MAP_CROSS_RICH_TERRAIN_OFFICIAL_CENTERLINE_UNOBSERVED_DEV
):
    def __post_init__(self):
        super().__post_init__()
        _apply_terrain_banks(self)


@configclass
class G1InstinctNavigationEnvCfg_DYNAMIC_MAP_CROSS_RICH_TERRAIN_OFFICIAL_CENTERLINE_UNOBSERVED_BANKS_PLAY(
    G1InstinctNavigationEnvCfg_DYNAMIC_MAP_CROSS_RICH_TERRAIN_OFFICIAL_CENTERLINE_UNOBSERVED_PLAY
):
    def __post_init__(self):
        super().__post_init__()
        _apply_terrain_banks(self)

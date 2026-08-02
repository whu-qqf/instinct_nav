from __future__ import annotations

import copy
import math
import os
from pathlib import Path

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.sensors.ray_caster.patterns import PinholeCameraPatternCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import instinct_nav.navigation.mdp as nav_mdp
from instinct_nav.navigation.navigation_env_cfg import NavigationEnvCfg

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
        self.scene.height_scanner_critic.prim_path = "{ENV_REGEX_NS}/Robot/torso_link"
        self.scene.height_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/torso_link",
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
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

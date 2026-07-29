from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CameraConfig:
    """Camera geometry used by the shared-lowcam training task."""

    width: int = 64
    height: int = 36
    horizontal_fov_deg: float = 89.51
    vertical_fov_deg: float = 58.29
    frame: str = "torso_link"
    # Pose of the simulated camera relative to torso_link. Real deployment must
    # calibrate the physical camera to match this as closely as possible.
    offset_pos_xyz: tuple[float, float, float] = (0.0487988662332928, 0.01, 0.4378029937970051)
    offset_quat_wxyz: tuple[float, float, float, float] = (
        0.9135367613482678,
        0.004363309284746571,
        0.4067366430758002,
        0.0,
    )
    low_level_depth_range_m: tuple[float, float] = (0.0, 2.5)
    low_level_crop_region: tuple[int, int, int, int] = (18, 0, 16, 16)
    low_level_history: int = 8


@dataclass(frozen=True)
class ObservationLayout:
    """Flat actor observation layout for the current scene-token policy."""

    proprio_dim: int = 16
    encoded_depth_dim: int = 64 * 5 * 8
    scene_token_num_tokens: int = 64
    scene_token_feature_dim: int = 6
    scene_token_dim: int = 64 * 6

    @property
    def actor_obs_dim(self) -> int:
        return self.proprio_dim + self.encoded_depth_dim + self.scene_token_dim


@dataclass(frozen=True)
class OnboardConfig:
    """Deployment constants shared by the high-level runtime."""

    repo_root: Path = Path(__file__).resolve().parents[2]
    high_level_hz: float = 5.0
    low_level_hz: float = 50.0
    command_min: tuple[float, float, float] = (-1.0, -0.6, -1.0)
    command_max: tuple[float, float, float] = (1.0, 0.6, 1.0)
    camera: CameraConfig = CameraConfig()
    layout: ObservationLayout = ObservationLayout()

    # Default artifact locations inside this repository. Override these from the
    # robot-specific launch file when deploying to another directory.
    low_level_checkpoint: Path = Path("artifacts/low_level/g1_parkour/model_30000.pt")
    depth_encoder: Path = Path("artifacts/sru/depth_encoder/vae_pretrain_new.pth")

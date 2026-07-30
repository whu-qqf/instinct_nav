from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class SensorPacket:
    """One synchronized sensor sample from the robot.

    All vectors are expected in the same body/world conventions used during
    training. The concrete robot bridge is responsible for calibration and frame
    transforms before creating this packet.
    """

    timestamp_s: float
    base_lin_vel_b: np.ndarray      # shape (3,)
    base_ang_vel_b: np.ndarray      # shape (3,)
    projected_gravity_b: np.ndarray # shape (3,)
    joint_pos: np.ndarray           # shape (num_joints,)
    joint_vel: np.ndarray           # shape (num_joints,)
    depth_m: np.ndarray             # raw depth image, expected shape (36, 64)
    goal_b: np.ndarray              # robot-body goal command, shape (4,): dir_xyz + log_dist
    reset: bool = False
    base_pos_w: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    base_yaw_w: float = 0.0


@dataclass
class VelocityCommand:
    """High-level command consumed by instinct_onboard / low-level policy."""

    vx: float
    vy: float
    wz: float
    timestamp_s: float

    def as_array(self) -> np.ndarray:
        return np.asarray([self.vx, self.vy, self.wz], dtype=np.float32)


@dataclass
class JointCommand:
    """Optional low-level output packet for integrations that bypass instinct_onboard."""

    joint_pos_target: np.ndarray
    timestamp_s: float
    feedforward_torque: Optional[np.ndarray] = None

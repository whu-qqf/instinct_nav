from __future__ import annotations

import numpy as np

from .bridges import VelocityCommandSink
from .packets import VelocityCommand


class InstinctOnboardVelocityCommandSink(VelocityCommandSink):
    """Write SRU navigation commands into an instinct_onboard node.

    `instinct_onboard`'s G1 parkour stack already exposes the exact integration
    point we need:

    - `scripts/g1_parkour.py` creates `G1ParkourNode.base_velocity_cmd`.
    - `ParkourAgent._get_base_velocity_cmd_obs()` returns that array.
    - The low-level ONNX actor then consumes it as the velocity command obs.

    A real integration can instantiate this sink with a live `G1ParkourNode` and
    call `write()` at the SRU high-level rate. The node's 50 Hz parkour loop will
    read the latest command from `base_velocity_cmd`.
    """

    def __init__(self, node, command_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)):
        if not hasattr(node, "base_velocity_cmd"):
            raise AttributeError(
                "instinct_onboard node must expose base_velocity_cmd. "
                "Use scripts/g1_parkour.py::G1ParkourNode or add the same buffer to your node."
            )
        self.node = node
        self.command_scale = np.asarray(command_scale, dtype=np.float32)

    def write(self, command: VelocityCommand) -> None:
        raw = command.as_array().astype(np.float32)
        self.node.base_velocity_cmd = raw * self.command_scale

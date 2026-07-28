from __future__ import annotations

from abc import ABC, abstractmethod

from .packets import SensorPacket, VelocityCommand


class SensorSource(ABC):
    """Robot-specific source of synchronized onboard observations."""

    @abstractmethod
    def read(self) -> SensorPacket:
        raise NotImplementedError


class VelocityCommandSink(ABC):
    """Destination for SRU velocity commands.

    A real implementation should usually publish to instinct_onboard's velocity
    command input rather than sending joint commands directly from this package.
    """

    @abstractmethod
    def write(self, command: VelocityCommand) -> None:
        raise NotImplementedError


class PrintVelocityCommandSink(VelocityCommandSink):
    """Dry-run sink used before connecting to robot control."""

    def write(self, command: VelocityCommand) -> None:
        print(
            f"[instinct_nav_onboard] t={command.timestamp_s:.3f} "
            f"vx={command.vx:+.3f} vy={command.vy:+.3f} wz={command.wz:+.3f}",
            flush=True,
        )

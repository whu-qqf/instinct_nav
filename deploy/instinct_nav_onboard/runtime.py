from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import torch

from .bridges import SensorSource, VelocityCommandSink
from .config import OnboardConfig
from .debug_dump import DebugDumpConfig, OnboardDebugDumper
from .obs_builder import HighLevelObsBuilder
from .packets import VelocityCommand


class InstinctNavOnboardRuntime:
    """High-level SRU runtime wrapper.

    This class runs the 5 Hz navigation policy and writes velocity commands to a
    sink, typically an instinct_onboard bridge. It deliberately does not command
    joints directly.
    """

    def __init__(
        self,
        cfg: OnboardConfig,
        sensor_source: SensorSource,
        command_sink: VelocityCommandSink,
        policy: torch.nn.Module,
        depth_encoder: Optional[torch.nn.Module] = None,
        device: str = "cpu",
        debug_dump_config: Optional[DebugDumpConfig] = None,
    ):
        self.cfg = cfg
        self.sensor_source = sensor_source
        self.command_sink = command_sink
        self.policy = policy.to(device).eval()
        self.depth_encoder = depth_encoder.to(device).eval() if depth_encoder is not None else None
        self.device = torch.device(device)
        self.obs_builder = HighLevelObsBuilder(cfg, device=device)
        self.debug_dumper = OnboardDebugDumper(debug_dump_config) if debug_dump_config is not None else None

    @classmethod
    def from_torchscript(
        cls,
        cfg: OnboardConfig,
        sensor_source: SensorSource,
        command_sink: VelocityCommandSink,
        policy_path: Path,
        device: str = "cpu",
    ) -> "InstinctNavOnboardRuntime":
        policy = torch.jit.load(str(policy_path), map_location=device)
        return cls(cfg, sensor_source, command_sink, policy=policy, device=device)

    def step(self) -> VelocityCommand:
        packet = self.sensor_source.read()
        encoded_depth = self._encode_depth(packet.depth_m)
        scene_tokens = self.obs_builder.build_scene_tokens_from_depth_visibility(packet.depth_m)
        obs = self.obs_builder.build_actor_obs(packet, encoded_depth=encoded_depth, scene_tokens=scene_tokens)

        with torch.no_grad():
            try:
                action = self.policy(obs, bool(packet.reset))
            except TypeError:
                action = self.policy(obs)
        action = action.reshape(-1)[:3]
        self.obs_builder.update_last_action(action)

        command = self._action_to_command(action, packet.timestamp_s)
        if self.debug_dumper is not None:
            self.debug_dumper.maybe_dump(
                packet=packet,
                encoded_depth=encoded_depth,
                scene_tokens=scene_tokens,
                actor_obs=obs,
                raw_action=action,
                command=command,
            )
        self.command_sink.write(command)
        return command

    def run_forever(self) -> None:
        period = 1.0 / self.cfg.high_level_hz
        while True:
            start = time.monotonic()
            self.step()
            elapsed = time.monotonic() - start
            time.sleep(max(0.0, period - elapsed))

    def _encode_depth(self, depth_m) -> torch.Tensor:
        if self.depth_encoder is None:
            # Wiring-only fallback. Do not use this for robot trials; the trained
            # policy expects the SRU depth encoder output distribution.
            return torch.zeros(1, self.cfg.layout.encoded_depth_dim, device=self.device)
        depth = torch.as_tensor(depth_m, device=self.device, dtype=torch.float32).reshape(1, 1, 36, 64)
        with torch.no_grad():
            encoded = self.depth_encoder(depth)
        if isinstance(encoded, tuple):
            encoded = encoded[0]
        return encoded.reshape(1, -1)

    def _action_to_command(self, action: torch.Tensor, timestamp_s: float) -> VelocityCommand:
        mins = torch.as_tensor(self.cfg.command_min, device=action.device)
        maxs = torch.as_tensor(self.cfg.command_max, device=action.device)
        command = torch.max(torch.min(action, maxs), mins).detach().cpu().numpy()
        return VelocityCommand(float(command[0]), float(command[1]), float(command[2]), timestamp_s=timestamp_s)

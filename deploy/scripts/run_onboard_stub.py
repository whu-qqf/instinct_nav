#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from instinct_nav_onboard.bridges import PrintVelocityCommandSink, SensorSource
from instinct_nav_onboard.config import OnboardConfig
from instinct_nav_onboard.packets import SensorPacket
from instinct_nav_onboard.runtime import InstinctNavOnboardRuntime
from instinct_nav_onboard.debug_dump import DebugDumpConfig


class DummySensorSource(SensorSource):
    def read(self) -> SensorPacket:
        now = time.monotonic()
        return SensorPacket(
            timestamp_s=now,
            base_lin_vel_b=np.zeros(3, dtype=np.float32),
            base_ang_vel_b=np.zeros(3, dtype=np.float32),
            projected_gravity_b=np.asarray([0.0, 0.0, -1.0], dtype=np.float32),
            joint_pos=np.zeros(29, dtype=np.float32),
            joint_vel=np.zeros(29, dtype=np.float32),
            depth_m=np.full((36, 64), 2.5, dtype=np.float32),
            goal_b=np.asarray([1.0, 0.0, 0.0, np.log1p(2.0)], dtype=np.float32),
        )


class ZeroPolicy(torch.nn.Module):
    def forward(self, obs: torch.Tensor, reset: bool = False) -> torch.Tensor:
        return torch.zeros(obs.shape[0], 3, device=obs.device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Dry-run instinct_nav onboard runtime wiring.")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--debug_dump_dir", type=str, default=None)
    parser.add_argument("--debug_dump_every_steps", type=int, default=1)
    parser.add_argument("--debug_dump_max_steps", type=int, default=1000)
    parser.add_argument("--debug_dump_stats_only", action="store_true", default=False)
    args = parser.parse_args()

    debug_cfg = None
    if args.debug_dump_dir is not None:
        debug_cfg = DebugDumpConfig(
            dump_dir=Path(args.debug_dump_dir).expanduser(),
            every_steps=args.debug_dump_every_steps,
            max_steps=args.debug_dump_max_steps,
            save_arrays=not args.debug_dump_stats_only,
        )

    runtime = InstinctNavOnboardRuntime(
        cfg=OnboardConfig(),
        sensor_source=DummySensorSource(),
        command_sink=PrintVelocityCommandSink(),
        policy=ZeroPolicy(),
        device="cpu",
        debug_dump_config=debug_cfg,
    )
    for _ in range(args.steps):
        runtime.step()
        time.sleep(1.0 / runtime.cfg.high_level_hz)


if __name__ == "__main__":
    main()

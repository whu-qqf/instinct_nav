from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .packets import SensorPacket, VelocityCommand


@dataclass(frozen=True)
class DebugDumpConfig:
    """Configuration for onboard observation/action dump files."""

    dump_dir: Path
    every_steps: int = 1
    max_steps: int = 1000
    save_arrays: bool = True


class OnboardDebugDumper:
    """Write high-level SRU inputs/outputs for sim-to-real distribution checks."""

    def __init__(self, cfg: DebugDumpConfig):
        self.cfg = cfg
        self.cfg.dump_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.cfg.dump_dir / "index.jsonl"
        self.step_count = 0
        self.saved_count = 0

    def maybe_dump(
        self,
        packet: SensorPacket,
        encoded_depth: torch.Tensor,
        scene_tokens: torch.Tensor,
        actor_obs: torch.Tensor,
        raw_action: torch.Tensor,
        command: VelocityCommand,
    ) -> None:
        self.step_count += 1
        if self.cfg.every_steps <= 0:
            return
        if self.step_count % self.cfg.every_steps != 0:
            return
        if self.saved_count >= self.cfg.max_steps:
            return

        self.saved_count += 1
        tag = f"step_{self.step_count:08d}"
        encoded_np = _tensor_to_numpy(encoded_depth).reshape(-1)
        scene_np = _tensor_to_numpy(scene_tokens).reshape(-1)
        obs_np = _tensor_to_numpy(actor_obs).reshape(-1)
        action_np = _tensor_to_numpy(raw_action).reshape(-1)[:3]
        command_np = command.as_array()

        record: dict[str, Any] = {
            "step": self.step_count,
            "saved_index": self.saved_count,
            "timestamp_s": float(packet.timestamp_s),
            "reset": bool(packet.reset),
            "file": f"{tag}.npz" if self.cfg.save_arrays else None,
            "base_lin_vel_b": _array_to_list(packet.base_lin_vel_b),
            "base_ang_vel_b": _array_to_list(packet.base_ang_vel_b),
            "projected_gravity_b": _array_to_list(packet.projected_gravity_b),
            "goal_b": _array_to_list(packet.goal_b),
            "raw_action": action_np.astype(float).tolist(),
            "command": command_np.astype(float).tolist(),
            "depth_stats": _stats(np.asarray(packet.depth_m, dtype=np.float32)),
            "encoded_depth_stats": _stats(encoded_np),
            "scene_token_stats": _stats(scene_np),
            "actor_obs_stats": _stats(obs_np),
        }

        if self.cfg.save_arrays:
            np.savez_compressed(
                self.cfg.dump_dir / f"{tag}.npz",
                timestamp_s=np.asarray(packet.timestamp_s, dtype=np.float64),
                base_lin_vel_b=np.asarray(packet.base_lin_vel_b, dtype=np.float32),
                base_ang_vel_b=np.asarray(packet.base_ang_vel_b, dtype=np.float32),
                projected_gravity_b=np.asarray(packet.projected_gravity_b, dtype=np.float32),
                joint_pos=np.asarray(packet.joint_pos, dtype=np.float32),
                joint_vel=np.asarray(packet.joint_vel, dtype=np.float32),
                depth_m=np.asarray(packet.depth_m, dtype=np.float32),
                goal_b=np.asarray(packet.goal_b, dtype=np.float32),
                encoded_depth=encoded_np.astype(np.float32),
                scene_tokens=scene_np.astype(np.float32),
                actor_obs=obs_np.astype(np.float32),
                raw_action=action_np.astype(np.float32),
                command=command_np.astype(np.float32),
            )

        with self.index_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _tensor_to_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy().astype(np.float32, copy=False)


def _array_to_list(value: np.ndarray) -> list[float]:
    return np.asarray(value, dtype=np.float32).reshape(-1).astype(float).tolist()


def _stats(value: np.ndarray) -> dict[str, float | int]:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    finite = np.isfinite(arr)
    if not finite.any():
        return {"numel": int(arr.size), "finite_ratio": 0.0, "min": float("nan"), "max": float("nan"), "mean": float("nan"), "std": float("nan")}
    finite_arr = arr[finite]
    return {
        "numel": int(arr.size),
        "finite_ratio": float(finite.mean()),
        "min": float(finite_arr.min()),
        "max": float(finite_arr.max()),
        "mean": float(finite_arr.mean()),
        "std": float(finite_arr.std()),
    }

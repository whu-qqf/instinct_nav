from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np


def projected_gravity_tilt_deg(projected_gravity_b: np.ndarray) -> float:
    """Return torso tilt from upright using body-frame projected gravity.

    Upright is ``[0, 0, -1]``.  A non-finite or near-zero vector is treated as
    invalid and returns infinity so callers can fail closed.
    """

    gravity = np.asarray(projected_gravity_b, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(gravity))
    if not np.isfinite(gravity).all() or not math.isfinite(norm) or norm < 1.0e-6:
        return math.inf
    upright_cos = float(np.clip(-gravity[2] / norm, -1.0, 1.0))
    return math.degrees(math.acos(upright_cos))


class FallSafetyMonitor:
    """Latch when torso tilt exceeds a threshold for a sustained duration."""

    def __init__(self, threshold_deg: float = 55.0, duration_s: float = 0.15):
        if not 0.0 < float(threshold_deg) < 180.0:
            raise ValueError("threshold_deg must be in (0, 180)")
        if float(duration_s) < 0.0:
            raise ValueError("duration_s must be non-negative")
        self.threshold_deg = float(threshold_deg)
        self.duration_s = float(duration_s)
        self.exceeded_since_s: float | None = None
        self.latched = False

    def update(self, projected_gravity_b: np.ndarray, now_s: float) -> tuple[bool, float]:
        tilt_deg = projected_gravity_tilt_deg(projected_gravity_b)
        if self.latched:
            return True, tilt_deg
        if not math.isfinite(tilt_deg):
            self.latched = True
            return True, tilt_deg
        if tilt_deg < self.threshold_deg:
            self.exceeded_since_s = None
            return False, tilt_deg
        if self.exceeded_since_s is None or now_s < self.exceeded_since_s:
            self.exceeded_since_s = float(now_s)
        if float(now_s) - self.exceeded_since_s >= self.duration_s:
            self.latched = True
        return self.latched, tilt_deg

    def preflight_ok(self, projected_gravity_b: np.ndarray) -> tuple[bool, float]:
        tilt_deg = projected_gravity_tilt_deg(projected_gravity_b)
        return math.isfinite(tilt_deg) and tilt_deg < self.threshold_deg, tilt_deg


class PersistentSafetyLogger:
    """Line-buffered JSONL event logger with fsync for critical events."""

    def __init__(self, log_dir: str | Path):
        directory = Path(log_dir).expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=True)
        run_stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        self.path = directory / f"safety_{run_stamp}_pid{os.getpid()}.jsonl"
        self._stream = self.path.open("a", encoding="utf-8", buffering=1)

    def write(self, event: str, *, critical: bool = False, **fields: Any) -> None:
        record = {
            "wall_time_s": time.time(),
            "monotonic_time_s": time.monotonic(),
            "event": str(event),
            **fields,
        }
        self._stream.write(json.dumps(_json_safe(record), allow_nan=False) + "\n")
        self._stream.flush()
        if critical:
            os.fsync(self._stream.fileno())

    def close(self) -> None:
        if self._stream.closed:
            return
        self._stream.flush()
        os.fsync(self._stream.fileno())
        self._stream.close()


def _json_safe(value: Any):
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value

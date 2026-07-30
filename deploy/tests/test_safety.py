import json
import math

import numpy as np

from instinct_nav_onboard.safety import FallSafetyMonitor, PersistentSafetyLogger, projected_gravity_tilt_deg


def _gravity_at_tilt(degrees: float) -> np.ndarray:
    radians = math.radians(degrees)
    return np.asarray([math.sin(radians), 0.0, -math.cos(radians)], dtype=np.float32)


def test_projected_gravity_tilt():
    assert projected_gravity_tilt_deg(np.asarray([0.0, 0.0, -1.0])) == 0.0
    assert abs(projected_gravity_tilt_deg(_gravity_at_tilt(60.0)) - 60.0) < 1.0e-4
    assert math.isinf(projected_gravity_tilt_deg(np.zeros(3)))


def test_fall_monitor_requires_sustained_tilt_and_latches():
    monitor = FallSafetyMonitor(threshold_deg=55.0, duration_s=0.15)
    assert monitor.update(_gravity_at_tilt(60.0), 1.00)[0] is False
    assert monitor.update(_gravity_at_tilt(60.0), 1.14)[0] is False
    assert monitor.update(_gravity_at_tilt(30.0), 1.15)[0] is False
    assert monitor.update(_gravity_at_tilt(60.0), 2.00)[0] is False
    assert monitor.update(_gravity_at_tilt(60.0), 2.16)[0] is True
    assert monitor.update(_gravity_at_tilt(0.0), 2.20)[0] is True


def test_fall_monitor_fails_closed_on_invalid_gravity():
    monitor = FallSafetyMonitor()
    triggered, tilt = monitor.update(np.asarray([np.nan, 0.0, -1.0]), 1.0)
    assert triggered is True
    assert math.isinf(tilt)


def test_preflight_rejects_fallen_attitude():
    monitor = FallSafetyMonitor(threshold_deg=55.0)
    assert monitor.preflight_ok(_gravity_at_tilt(20.0))[0] is True
    assert monitor.preflight_ok(_gravity_at_tilt(80.0))[0] is False


def test_persistent_logger_writes_valid_json_and_sanitizes_nonfinite(tmp_path):
    logger = PersistentSafetyLogger(tmp_path)
    logger.write("fall", critical=True, tilt_deg=math.inf, vector=np.asarray([0.0, np.nan, -1.0]))
    path = logger.path
    logger.close()
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["event"] == "fall"
    assert record["tilt_deg"] is None
    assert record["vector"] == [0.0, None, -1.0]

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("pxr", reason="observation module requires an initialized Isaac Sim Python environment")

from instinct_nav.navigation.mdp import observations


class _FakeDepthModel:
    def __call__(self, depth):
        encoded = depth.mean(dim=(-2, -1), keepdim=True).expand(-1, 2, 1, 1).contiguous()
        return encoded, depth + 0.25


class _CountingDelayManager:
    def __init__(self):
        self.calls = 0

    def compute_delayed_visual_packet(self, packet, camera_name):
        assert camera_name == "camera"
        self.calls += 1
        return packet


def test_visual_packet_advances_delay_once_per_camera_step(monkeypatch):
    monkeypatch.setattr(observations, "JIT_DEPTH_NOISE_GENERATOR", _FakeDepthModel())
    monkeypatch.setattr(observations, "use_jit", False)
    monkeypatch.setattr(observations, "_ensure_depth_noise_generator_initialized", lambda: None)

    sensor = SimpleNamespace(
        image_shape=(2, 2),
        data=SimpleNamespace(
            output={"distance_to_image_plane": torch.ones(2, 2, 2)},
            pos_w=torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            quat_w_world=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(2, -1),
            intrinsic_matrices=torch.eye(3).expand(2, -1, -1).clone(),
        ),
    )
    delay_manager = _CountingDelayManager()
    env = SimpleNamespace(
        num_envs=2,
        common_step_counter=7,
        delay_manager=delay_manager,
        scene=SimpleNamespace(sensors={"camera": sensor}),
    )
    sensor_cfg = SimpleNamespace(name="camera")

    first = observations._get_delayed_visual_packet(env, sensor_cfg)
    second = observations._get_delayed_visual_packet(env, sensor_cfg)

    assert delay_manager.calls == 1
    assert first is second
    assert torch.equal(first["encoded_depth"], torch.full((2, 2), 1.0))
    assert torch.equal(first["depth"], torch.full((2, 1, 2, 2), 1.25))
    assert torch.equal(first["camera_pos_w"], sensor.data.pos_w)

    env.common_step_counter += 1
    observations._get_delayed_visual_packet(env, sensor_cfg)
    assert delay_manager.calls == 2


def test_visibility_manager_returns_preupdate_history_and_current_bev():
    manager = observations.LocalVisibilityMapManager(
        num_envs=1,
        device="cpu",
        size_m=6.4,
        grid_size=64,
        depth_min_m=0.5,
        depth_max_m=8.0,
        pixel_stride=1,
        decay=0.995,
    )
    depth = torch.ones(1, 1, 4, 4)
    camera_pos = torch.zeros(1, 3)
    identity_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    intrinsics = torch.tensor([[[2.0, 0.0, 1.5], [0.0, 2.0, 1.5], [0.0, 0.0, 1.0]]])
    root_pos = torch.zeros(1, 3)

    updated, history_before, current_hits, current_depth = manager.update(
        depth,
        camera_pos,
        identity_quat,
        intrinsics,
        root_pos,
        identity_quat,
        return_components=True,
    )

    assert updated.shape == history_before.shape == current_hits.shape == current_depth.shape == (1, 4096)
    assert torch.count_nonzero(history_before) == 0
    assert torch.count_nonzero(current_hits) > 0
    assert torch.count_nonzero(current_depth) > 0

    _, second_history_before, second_current_hits, _ = manager.update(
        depth,
        camera_pos,
        identity_quat,
        intrinsics,
        root_pos,
        identity_quat,
        return_components=True,
    )

    assert torch.allclose(second_history_before, updated)
    assert torch.equal(second_current_hits, current_hits)

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


def test_visibility_dynamic_risk_is_current_only_black_overlay():
    manager = observations.LocalVisibilityMapManager(
        num_envs=1,
        device="cpu",
        size_m=6.4,
        grid_size=64,
    )
    manager.map.fill_(1.0)
    manager.current_dynamic_risk[:, :, 30:34, 30:34] = 1.0

    effective = manager.risk_adjusted_map
    assert torch.all(effective[:, :, 30:34, 30:34] == 0.0)
    assert torch.all(manager.map == 1.0)

    manager.reset(torch.tensor([0]))
    assert torch.count_nonzero(manager.current_dynamic_risk) == 0


def test_masked_depth_is_excluded_from_persistent_visibility():
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
    dynamic_mask = torch.ones_like(depth)
    camera_pos = torch.zeros(1, 3)
    identity_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    intrinsics = torch.tensor([[[2.0, 0.0, 1.5], [0.0, 2.0, 1.5], [0.0, 0.0, 1.0]]])

    updated, _, current_hits, _, dynamic_hits = manager.update(
        depth,
        camera_pos,
        identity_quat,
        intrinsics,
        torch.zeros(1, 3),
        identity_quat,
        return_components=True,
        pixel_mask=dynamic_mask,
        exclude_masked_pixels_from_visibility=True,
    )

    assert torch.count_nonzero(updated) == 0
    assert torch.count_nonzero(current_hits) == 0
    assert torch.count_nonzero(dynamic_hits) > 0


def test_dynamic_mask_projection_uses_spawned_obstacle_dimensions(monkeypatch):
    intrinsics = torch.tensor([[[40.0, 0.0, 32.0], [0.0, 40.0, 32.0], [0.0, 0.0, 1.0]]])
    packet = {
        "depth": torch.full((1, 1, 64, 64), 10.0),
        "intrinsics": intrinsics,
        "camera_pos_w": torch.zeros(1, 3),
        "camera_quat_w": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
    }
    monkeypatch.setattr(observations, "_get_delayed_visual_packet", lambda env, sensor_cfg: packet)
    manager = SimpleNamespace(
        positions_w=torch.tensor([[[2.0, 0.0, 0.8]]]),
        spawned_obstacles=torch.ones(1, 1, dtype=torch.bool),
        curriculum_envs=torch.ones(1, dtype=torch.bool),
        body_half_extent=0.25,
        obstacle_half_height=0.4,
    )
    env = SimpleNamespace(device="cpu", _dynamic_obstacle_manager=manager, delay_manager=None)
    sensor_cfg = SimpleNamespace(name="camera")

    short_mask = observations.dynamic_obstacle_mask_label(env, sensor_cfg).reshape(64, 64)
    manager.obstacle_half_height = 0.8
    tall_mask = observations.dynamic_obstacle_mask_label(env, sensor_cfg).reshape(64, 64)

    assert short_mask[30, 32] == 0
    assert tall_mask[30, 32] == 1
    assert tall_mask.sum() > short_mask.sum()
    assert torch.equal(
        short_mask,
        observations.dynamic_obstacle_mask_label(env, sensor_cfg, half_height_m=0.4).reshape(64, 64),
    )

    manager.body_half_extent = 0.5
    wide_mask = observations.dynamic_obstacle_mask_label(env, sensor_cfg).reshape(64, 64)
    assert wide_mask.sum() > tall_mask.sum()


def test_dynamic_sector_risk_extends_along_velocity():
    positions = torch.tensor([[[1.0, 0.0, 0.8]]])
    velocities = torch.tensor([[[1.0, 0.0]]])
    active = torch.ones(1, 1, dtype=torch.bool)
    visible = torch.ones_like(active)
    root_pos = torch.zeros(1, 3)
    root_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

    risk = observations._render_dynamic_sector_risk(
        positions,
        velocities,
        active,
        visible,
        root_pos,
        root_quat,
        size_m=6.4,
        grid_size=64,
        body_half_extent_m=0.25,
        clearance_m=0.30,
        horizon_s=1.2,
        sigma_long_m=0.15,
        sigma_lat_m=0.12,
        softness_m=0.08,
        stationary_speed_mps=0.10,
    )[0, 0]

    # Grid row 17 is x ~= +1.45 m (ahead of the obstacle); row 37 is
    # x ~= -0.55 m (behind it). The predicted sector must point forward.
    assert risk[17, 31] > 0.8
    assert risk[37, 31] < 0.1


def test_dynamic_sector_risk_requires_visibility():
    risk = observations._render_dynamic_sector_risk(
        torch.tensor([[[1.0, 0.0, 0.8]]]),
        torch.tensor([[[1.0, 0.0]]]),
        torch.ones(1, 1, dtype=torch.bool),
        torch.zeros(1, 1, dtype=torch.bool),
        torch.zeros(1, 3),
        torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        size_m=6.4,
        grid_size=64,
        body_half_extent_m=0.25,
        clearance_m=0.30,
        horizon_s=1.2,
        sigma_long_m=0.15,
        sigma_lat_m=0.12,
        softness_m=0.08,
        stationary_speed_mps=0.10,
    )

    assert torch.count_nonzero(risk) == 0

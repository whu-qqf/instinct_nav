import sys
import types

import numpy as np
import torch
import torch.nn.functional as F

sys.modules.setdefault("onnxruntime", types.SimpleNamespace(InferenceSession=object))

from instinct_nav_onboard.config import ObservationLayout, OnboardConfig
from instinct_nav_onboard.dynamic_mask import resize_binary_mask
from instinct_nav_onboard.onnx_high_level_runtime import (
    OnnxHighLevelRuntime,
    _adaptive_avg_pool2d,
    _quat_wxyz_to_matrix,
)


def _runtime_without_onnx(cfg: OnboardConfig | None = None) -> OnnxHighLevelRuntime:
    runtime = OnnxHighLevelRuntime.__new__(OnnxHighLevelRuntime)
    runtime.cfg = cfg or OnboardConfig()
    runtime.visibility = np.zeros((1, 1, 64, 64), dtype=np.float32)
    runtime.last_base_pos_w = np.zeros(3, dtype=np.float32)
    runtime.last_base_yaw_w = 0.0
    runtime.visibility_pose_valid = False
    return runtime


def test_adaptive_pool_matches_pytorch_for_training_depth_shape():
    image = np.arange(36 * 64, dtype=np.float32).reshape(36, 64)
    expected = F.adaptive_avg_pool2d(torch.from_numpy(image)[None, None], (8, 8))[0, 0].numpy()
    actual = _adaptive_avg_pool2d(image, 8, 8)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-6)


def test_visibility_warp_compensates_forward_motion():
    runtime = _runtime_without_onnx()
    runtime._warp_visibility_to_current_pose(np.zeros(3, dtype=np.float32), 0.0, size_m=6.4, grid_size=64)
    runtime.visibility[0, 0, 22, 32] = 1.0
    runtime._warp_visibility_to_current_pose(
        np.asarray([0.5, 0.0, 0.0], dtype=np.float32), 0.0, size_m=6.4, grid_size=64
    )
    occupied = np.argwhere(runtime.visibility[0, 0] > 0.5)
    assert occupied.shape == (1, 2)
    np.testing.assert_array_equal(occupied[0], np.asarray([27, 32]))


def test_camera_offset_quaternion_is_valid_rotation():
    rotation = _quat_wxyz_to_matrix(np.asarray(OnboardConfig().camera.offset_quat_wxyz, dtype=np.float32))
    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1.0e-6)
    assert np.linalg.det(rotation) > 0.999


def test_visibility_projection_matches_training_pixel_stride_two():
    runtime = _runtime_without_onnx()
    depth = np.ones((36, 64), dtype=np.float32)
    valid = np.zeros((36, 64), dtype=bool)
    valid[1, 1] = True
    odd_only = runtime._depth_to_current_visibility(depth, valid, size_m=6.4, grid_size=64)
    assert float(odd_only.sum()) == 0.0

    valid.fill(False)
    valid[2, 2] = True
    even_sample = runtime._depth_to_current_visibility(depth, valid, size_m=6.4, grid_size=64)
    assert float(even_sample.sum()) == 1.0


def test_action_transform_matches_training_tanh_before_safety_clip():
    runtime = _runtime_without_onnx()
    raw = np.asarray([3.0, -3.0, 0.25], dtype=np.float32)
    command = runtime._action_to_command(raw, timestamp_s=1.0).as_array()
    normalized = np.tanh(raw)
    mins = np.asarray(runtime.cfg.command_min, dtype=np.float32)
    maxs = np.asarray(runtime.cfg.command_max, dtype=np.float32)
    expected = mins + 0.5 * (normalized + 1.0) * (maxs - mins)
    np.testing.assert_allclose(command, expected, rtol=0.0, atol=1.0e-7)

def test_score_visibility_accumulates_repeated_hits_and_decays_when_stale():
    runtime = _runtime_without_onnx(OnboardConfig(visibility_decay=0.9))
    depth = np.full((36, 64), 2.0, dtype=np.float32)

    first_tokens = runtime.build_scene_tokens(depth).reshape(64, 6)
    first_score = first_tokens[:, 2].sum()
    second_tokens = runtime.build_scene_tokens(depth).reshape(64, 6)
    second_score = second_tokens[:, 2].sum()
    assert second_score > first_score > 0.0

    stale_depth = np.zeros((36, 64), dtype=np.float32)
    stale_tokens = runtime.build_scene_tokens(stale_depth).reshape(64, 6)
    assert stale_tokens[:, 2].sum() < second_score


def test_legacy_visibility_remains_binary_and_does_not_decay():
    runtime = _runtime_without_onnx()
    depth = np.full((36, 64), 2.0, dtype=np.float32)

    runtime.build_scene_tokens(depth)
    first_visibility = runtime.visibility.copy()
    runtime.build_scene_tokens(np.zeros((36, 64), dtype=np.float32))

    np.testing.assert_array_equal(runtime.visibility, first_visibility)


def test_dynamic_map_cross_layout_builds_history_and_current_patches():
    layout = ObservationLayout(scene_token_feature_dim=5, scene_token_dim=64 * 5)
    runtime = _runtime_without_onnx(OnboardConfig(visibility_decay=0.995, layout=layout))
    depth = np.full((36, 64), 2.0, dtype=np.float32)
    patches = runtime.build_scene_tokens(depth).reshape(64, 5)
    assert patches.shape == (64, 5)
    assert patches[:, 0].sum() > 0.0
    assert patches[:, 1].sum() > 0.0
    np.testing.assert_allclose(patches[:, 4], patches[:, 1] - patches[:, 0], atol=1.0e-6)


def test_resize_binary_mask_preserves_binary_values():
    mask = np.asarray([[0.0, 1.0], [0.6, 0.4]], dtype=np.float32)
    resized = resize_binary_mask(mask, (4, 4))
    assert resized.shape == (4, 4)
    assert set(np.unique(resized)) == {0.0, 1.0}


def test_dynamic_map_mask_projects_current_only_presence():
    layout = ObservationLayout(scene_token_feature_dim=6, scene_token_dim=64 * 6)
    cfg = OnboardConfig(visibility_decay=0.995, scene_token_layout="dynamic_map_mask", layout=layout)
    runtime = _runtime_without_onnx(cfg)
    runtime.scene_token_layout = "dynamic_map_mask"
    depth = np.full((36, 64), 2.0, dtype=np.float32)
    mask = np.zeros((36, 64), dtype=np.float32)
    mask[2:8, 2:8] = 1.0

    first = runtime.build_scene_tokens(depth, dynamic_mask=mask).reshape(64, 6)
    assert first.shape == (64, 6)
    assert first[:, 0].sum() == 0.0
    assert first[:, 1].sum() > 0.0
    assert first[:, 5].sum() > 0.0

    second = runtime.build_scene_tokens(depth, dynamic_mask=np.zeros_like(mask)).reshape(64, 6)
    assert second[:, 0].sum() > 0.0
    assert second[:, 5].sum() == 0.0


def test_dynamic_mask_requires_valid_depth_for_bev_projection():
    layout = ObservationLayout(scene_token_feature_dim=6, scene_token_dim=64 * 6)
    cfg = OnboardConfig(scene_token_layout="dynamic_map_mask", layout=layout)
    runtime = _runtime_without_onnx(cfg)
    runtime.scene_token_layout = "dynamic_map_mask"
    depth = np.zeros((36, 64), dtype=np.float32)
    mask = np.ones((36, 64), dtype=np.float32)

    tokens = runtime.build_scene_tokens(depth, dynamic_mask=mask).reshape(64, 6)
    assert tokens[:, 5].sum() == 0.0

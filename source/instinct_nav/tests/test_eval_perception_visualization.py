"""Test eval visualization without starting Isaac Sim."""
import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


def load_function(name, namespace):
    path = Path(__file__).resolve().parents[1] / "scripts" / "eval.py"
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


def test_statistics_only_eval_does_not_render_observation_panels():
    function = load_function("needs_observation_frame_cache", {})
    args = SimpleNamespace(failure_rgb_include_observations=True,
                           observation_gif_dir=None, observation_gif_limit=0)
    assert not function(args, None, None)
    assert function(args, object(), None)
    args.failure_rgb_include_observations = False
    assert not function(args, object(), None)
    assert function(args, None, object())
    args.observation_gif_dir = "output"
    args.observation_gif_limit = 10
    assert function(args, None, None)


@pytest.mark.parametrize("learned", [False, True])
def test_capture_uses_policy_packet_and_never_updates_map(learned):
    raw = torch.full((2, 2, 2, 1), 9.0)
    delayed = torch.full((2, 1, 2, 2), 2.0)
    sensor = SimpleNamespace(image_shape=(2, 2), data=SimpleNamespace(output={"distance_to_image_plane": raw}))
    term = SimpleNamespace(func=SimpleNamespace(__name__="learned_dynamic_bev_observation"),
                           params={"output_grid_size": 2}) if learned else None
    bev = torch.arange(48.0).view(2, 24)
    env = SimpleNamespace(num_envs=2, scene=SimpleNamespace(sensors={"camera": sensor}),
                          cfg=SimpleNamespace(observations=SimpleNamespace(
                              policy=SimpleNamespace(scene_tokens=None, visibility_map=term))),
                          _instinct_nav_learned_bev_cache={"bev": bev})
    wrapper = SimpleNamespace(unwrapped=env)
    ns = {
        "np": np, "DepthPoseLocalTerrainMap": type(None),
        "args_cli": SimpleNamespace(depth_pit_to_obstacle=False),
        "_resolve_depth_visualization_sensor_cfg": lambda _: SimpleNamespace(name="camera"),
        "nav_observations": SimpleNamespace(_get_noisy_depth_sample=lambda *_: {"depth": delayed}),
        "_scalar_frame_to_rgb": lambda values, **_: values.copy(),
        "_learned_bev_to_rgb": lambda values: values.copy(),
    }
    capture = load_function("_capture_observation_frame_cache", ns)
    frames = capture(wrapper)
    np.testing.assert_array_equal(frames["depth"][0], np.full((2, 2), 2.0))
    assert frames["scene_tokens"] == [None, None]
    if learned:
        np.testing.assert_array_equal(frames["visibility_obs"][0], bev[0].numpy().reshape(6, 2, 2))
    else:
        assert frames["visibility_obs"] == [None, None]


def test_dynamic_channels_are_visible_in_six_channel_panel():
    ns = {"np": np, "_label_rgb_frame": lambda image, _: image,
          "_resize_rgb_frame": lambda image, _: image}
    render = load_function("_learned_bev_to_rgb", ns)
    bev = np.zeros((6, 4, 4), dtype=np.float32)
    bev[3, 1, 2] = 1.0
    frame = render(bev)
    assert frame.shape == (8, 12, 3)
    assert frame[5, 2, 0] == 255
    assert frame[5, 2, 1] < 255


def test_snapshot_preserves_obstacle_pose_and_velocity():
    fields = ("root_pos_w", "root_quat_w", "root_lin_vel_w", "root_ang_vel_w",
              "root_lin_vel_b", "root_ang_vel_b", "joint_pos", "joint_vel", "body_state_w")
    robot = SimpleNamespace(data=SimpleNamespace(**{name: torch.zeros(2, 3) for name in fields}))
    state = torch.arange(26.0).view(2, 13)
    obstacle = SimpleNamespace(data=SimpleNamespace(root_state_w=state))
    scene = SimpleNamespace(rigid_objects={"obstacle_0": obstacle})
    env = SimpleNamespace(scene=scene, command_manager=SimpleNamespace(_terms={}))
    ns = {"torch": torch}
    capture = load_function("capture_base_contact_snapshot", ns)
    snapshot = capture(SimpleNamespace(unwrapped=env), {"robot": robot})
    torch.testing.assert_close(snapshot["obstacle_state/obstacle_0"], state)
    state.zero_()
    assert snapshot["obstacle_state/obstacle_0"].sum() > 0

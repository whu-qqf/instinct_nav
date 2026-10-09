"""Temporal BEV cache and lag tests independent of Isaac Sim."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F


def _setup(offsets):
    path = Path(__file__).resolve().parents[1] / "instinct_nav/navigation/mdp/observations.py"
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == "learned_dynamic_bev_observation")
    env = SimpleNamespace(num_envs=2, common_step_counter=0, step_dt=0.2)
    env.scene = {"robot": SimpleNamespace(data=SimpleNamespace(
        root_pos_w=torch.zeros(2, 3), root_quat_w=torch.zeros(2, 4)))}
    manager = SimpleNamespace(device="cpu", dynamic_history=None,
                              map=torch.zeros(2, 1, 8, 8),
                              current_dynamic_risk=torch.zeros(2, 1, 8, 8), calls=0)

    def update(*args, **kwargs):
        manager.calls += 1
        hit = torch.zeros(2, 8, 8)
        hit[:, 4, env.common_step_counter] = 1.0
        zero = torch.zeros(2, 64)
        return zero, zero, zero, zero, hit.flatten(1)

    manager.update = update
    packet = {"depth": torch.ones(2, 1, 2, 2), "camera_pos_w": None,
              "camera_quat_w": None, "intrinsics": None}
    ns = {"torch": torch, "F": F, "SceneEntityCfg": lambda name: SimpleNamespace(name=name),
          "_get_delayed_visual_packet": lambda *_: packet,
          "_get_or_create_visibility_map_manager": lambda *_, **__: manager,
          "dynamic_obstacle_mask_observation": lambda *_, **__: torch.zeros(2, 4)}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), ns)

    def observe(**kwargs):
        return ns["learned_dynamic_bev_observation"](
            env, SimpleNamespace(name="camera"), grid_size=8, output_grid_size=8,
            dynamic_history_offsets_s=offsets, dynamic_dilation_kernel=1,
            **kwargs,
        ).view(2, 6, 8, 8)

    return env, manager, observe


@pytest.mark.parametrize("offsets,lags", [(None, (1, 2)), ((0.4, 0.8), (2, 4))])
def test_dynamic_history_lags_and_actor_critic_cache(offsets, lags):
    env, manager, observe = _setup(offsets)
    for step in range(5):
        env.common_step_counter = step
        bev = observe()
        assert manager.calls == step + 1
        assert torch.equal(bev, observe())
        assert manager.calls == step + 1
        assert bev[:, 3, 4, step].sum() == 2
        for channel, lag in zip((4, 5), lags):
            if step >= lag:
                assert bev[:, channel, 4, step - lag].sum() == 2
                assert bev[:, channel].sum() == 2
            else:
                assert bev[:, channel].sum() == 0


@pytest.mark.parametrize("offsets", [(0.0, 0.8), (0.8, 0.4), (0.01, 0.02)])
def test_invalid_history_offsets_rejected(offsets):
    _, _, observe = _setup(offsets)
    with pytest.raises(ValueError):
        observe()


def test_long_history_is_ego_compensated_and_reset_per_environment():
    path = Path(__file__).resolve().parents[1] / "instinct_nav/navigation/mdp/observations.py"
    tree = ast.parse(path.read_text())
    names = {"LocalVisibilityMapManager", "_quat_wxyz_to_yaw"}
    nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    ns = {"torch": torch, "F": F}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), ns)
    manager = ns["LocalVisibilityMapManager"](num_envs=2, device="cpu", size_m=0.8, grid_size=8)
    manager.dynamic_history = torch.zeros(2, 4, 8, 8)
    manager.dynamic_history[:, :, 2, 3] = 1.0
    manager.pose_valid[:] = True
    roots = torch.tensor([[0.1, 0.0, 0.0], [0.1, 0.0, 0.0]])
    manager._warp_memory_to_current_frame(roots, torch.zeros(2))
    assert torch.all(manager.dynamic_history[:, :, 3, 3] == 1.0)
    assert manager.dynamic_history.sum() == 8
    manager.reset(torch.tensor([0]))
    assert manager.dynamic_history[0].sum() == 0
    assert manager.dynamic_history[1].sum() == 4


@pytest.mark.parametrize("enabled,cached_count,cached_step", [(False, 0, 7), (True, 6, 7), (True, 6, 6)])
def test_teacher_state_handles_static_initialization_and_stale_visibility(enabled, cached_count, cached_step):
    path = Path(__file__).resolve().parents[1] / "instinct_nav/navigation/mdp/observations.py"
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == "future_occupancy_teacher_state")
    robot = SimpleNamespace(data=SimpleNamespace(root_pos_w=torch.zeros(2, 3), root_quat_w=torch.zeros(2, 4)))
    scene = {"robot": robot}
    for index in range(6):
        scene[f"obstacle_{index}"] = SimpleNamespace(data=SimpleNamespace(root_pos_w=torch.ones(2, 3)))
    env = SimpleNamespace(scene=scene, num_envs=2, device="cpu", common_step_counter=7,
                          cfg=SimpleNamespace(num_obstacles=6, observations=SimpleNamespace(
                              policy=SimpleNamespace(visibility_map=SimpleNamespace(params={"size_m": 6.4})))),
                          _dynamic_obstacle_manager=SimpleNamespace(enabled=enabled,
                              spawned_obstacles=torch.ones(2, 6, dtype=torch.bool),
                              curriculum_envs=torch.ones(2, dtype=torch.bool)),
                          _instinct_nav_direct_mask_cache={"key": (cached_step,),
                              "instance_visibility": torch.ones(2, cached_count)})
    ns = {"torch": torch, "SceneEntityCfg": lambda name: SimpleNamespace(name=name),
          "_quat_wxyz_to_yaw": lambda q: torch.zeros(q.shape[0])}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), ns)
    state = ns["future_occupancy_teacher_state"](env)
    assert state.shape == (2, 27)
    slots = state[:, 3:].view(2, 6, 4)
    assert torch.all(slots[..., 2] == float(enabled))
    expected_visible = cached_count == 6 and cached_step == 7
    assert torch.all(slots[..., 3] == float(expected_visible))
    scene["obstacle_0"].data.root_pos_w.fill_(20.0)
    far_state = ns["future_occupancy_teacher_state"](env)
    assert far_state[:, 3:].view(2, 6, 4)[:, 0, 3].sum() == 0


@pytest.mark.parametrize("relative", ["scripts/train_senior.py", "source/instinct_nav/scripts/train.py"])
def test_training_wrapper_passes_teacher_without_adding_it_to_actor(relative):
    root = Path(__file__).resolve().parents[3]
    path = root / relative
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "LegacyRslRlVecEnvWrapper")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_to_legacy_obs")
    ns = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), ns)
    policy = torch.zeros(2, 8720)
    teacher = torch.zeros(2, 27)
    actor, extras = ns["_to_legacy_obs"](None, {"policy": policy, "future_teacher": teacher})
    assert actor is policy
    assert extras["observations"]["future_teacher"] is teacher
    assert extras["observations"]["critic"] is policy

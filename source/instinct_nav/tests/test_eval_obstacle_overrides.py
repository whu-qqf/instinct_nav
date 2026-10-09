from pathlib import Path
from types import SimpleNamespace
import importlib.util
import ast

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "eval_obstacle_overrides.py"
SPEC = importlib.util.spec_from_file_location("eval_obstacle_overrides", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
apply_obstacle_eval_overrides = MODULE.apply_obstacle_eval_overrides


def _config():
    names = ["obstacle_0", "obstacle_1"]
    scene = SimpleNamespace()
    for name in names:
        setattr(
            scene,
            name,
            SimpleNamespace(
                spawn=SimpleNamespace(size=(0.5, 0.5, 0.8)),
                init_state=SimpleNamespace(pos=(0.0, 0.0, 0.5)),
            ),
        )
    params = {"obstacle_names": names, "initial_difficulty": 0.0, "curriculum_env_fraction": 0.5,
              "fraction_range": (0.25, 0.9), "obstacle_half_height": 0.4}
    return SimpleNamespace(
        scene=scene,
        events=SimpleNamespace(
            dynamic_obstacles_reset=SimpleNamespace(params=params.copy()),
            dynamic_obstacles_update=SimpleNamespace(params=params.copy()),
        ),
        obstacle_initial_difficulty=0.0,
        obstacle_curriculum_env_fraction=0.5,
        obstacle_fraction_range=(0.25, 0.9),
        obstacle_half_height=0.4,
    )


def test_matches_hard_all_env_conditions_before_scene_creation():
    cfg = _config()
    apply_obstacle_eval_overrides(cfg, difficulty=1.0, env_fraction=1.0, height_m=1.6)
    assert cfg.obstacle_initial_difficulty == 1.0
    assert cfg.obstacle_fraction_range == (1.0, 1.0)
    assert cfg.obstacle_half_height == 0.8
    for name in ("dynamic_obstacles_reset", "dynamic_obstacles_update"):
        params = getattr(cfg.events, name).params
        assert params["initial_difficulty"] == 1.0
        assert params["curriculum_env_fraction"] == 1.0
        assert params["fraction_range"] == (1.0, 1.0)
        assert params["obstacle_half_height"] == 0.8
    for name in ("obstacle_0", "obstacle_1"):
        obj = getattr(cfg.scene, name)
        assert obj.spawn.size == (0.5, 0.5, 1.6)
        assert obj.init_state.pos == (0.0, 0.0, 0.8)


@pytest.mark.parametrize("kwargs", [
    {"difficulty": -0.1}, {"difficulty": 1.1},
    {"env_fraction": 0.0}, {"env_fraction": 1.1},
    {"height_m": 0.0},
])
def test_invalid_values_rejected(kwargs):
    with pytest.raises(ValueError):
        apply_obstacle_eval_overrides(_config(), **kwargs)


def test_no_override_leaves_config_unchanged():
    cfg = _config()
    apply_obstacle_eval_overrides(cfg)
    assert cfg.obstacle_half_height == 0.4
    assert cfg.scene.obstacle_0.spawn.size == (0.5, 0.5, 0.8)


@pytest.mark.parametrize("enabled", [False, True])
def test_future_eval_needs_no_teacher_and_preserves_prediction_head(enabled):
    path = MODULE_PATH.with_name("eval.py")
    tree = ast.parse(path.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "disable_future_training_supervision")
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    teacher = object()
    env_cfg = SimpleNamespace(observations=SimpleNamespace(future_teacher=teacher))
    agent_cfg = SimpleNamespace(algorithm=SimpleNamespace(lambda_future=0.05 if enabled else 0.0),
                                policy=SimpleNamespace(use_future_occupancy=enabled))
    namespace["disable_future_training_supervision"](env_cfg, agent_cfg)
    assert agent_cfg.algorithm.lambda_future == 0.0
    assert agent_cfg.policy.use_future_occupancy is enabled
    assert env_cfg.observations.future_teacher is (None if enabled else teacher)


def test_eval_terrain_seed_is_independent_and_optional():
    path = MODULE_PATH.with_name("eval.py")
    tree = ast.parse(path.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "apply_eval_terrain_seed")
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    cfg = SimpleNamespace(seed=72, scene=SimpleNamespace(
        terrain=SimpleNamespace(terrain_generator=SimpleNamespace(seed=None))))
    namespace["apply_eval_terrain_seed"](cfg, None)
    assert cfg.scene.terrain.terrain_generator.seed is None
    namespace["apply_eval_terrain_seed"](cfg, 73)
    assert cfg.scene.terrain.terrain_generator.seed == 73
    assert cfg.seed == 72
    with pytest.raises(ValueError):
        namespace["apply_eval_terrain_seed"](SimpleNamespace(scene=SimpleNamespace()), 73)


def test_cylinder_height_override_preserves_radius():
    cfg = _config()
    cfg.scene.obstacle_0.spawn = SimpleNamespace(radius=0.25, height=0.8)
    apply_obstacle_eval_overrides(cfg, height_m=1.6)
    assert cfg.scene.obstacle_0.spawn.radius == 0.25
    assert cfg.scene.obstacle_0.spawn.height == 1.6
    assert cfg.scene.obstacle_0.init_state.pos[2] == 0.8


@pytest.mark.parametrize("cylinder", [False, True])
def test_eval_conditions_record_both_geometries(cylinder):
    # Load only the pure summary function; importing eval starts Isaac Sim.
    path = MODULE_PATH.with_name("eval.py")
    tree = ast.parse(path.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "dynamic_obstacle_eval_conditions")
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    cfg = _config()
    if cylinder:
        cfg.scene.obstacle_0.spawn = SimpleNamespace(radius=0.25, height=1.6)
    fraction = SimpleNamespace(float=lambda: SimpleNamespace(mean=lambda: SimpleNamespace(item=lambda: 1.0)))
    manager = SimpleNamespace(enabled=True, difficulty=1.0, curriculum_envs=fraction,
                              obstacle_half_height=0.8, obstacle_names=["obstacle_0"],
                              num_obstacles=6, spawn_delay_range=(0.5, 1.0), spawn_interval_s=0.0)
    env = SimpleNamespace(unwrapped=SimpleNamespace(cfg=cfg, _dynamic_obstacle_manager=manager))
    result = namespace["dynamic_obstacle_eval_conditions"](env)
    assert result["enabled"] is True
    if cylinder:
        assert result["cuboid_size_m"] is None
        assert result["geometry"]["radius_m"] == 0.25
        assert result["geometry"]["height_m"] == 1.6
    else:
        assert result["cuboid_size_m"] == [0.5, 0.5, 0.8]

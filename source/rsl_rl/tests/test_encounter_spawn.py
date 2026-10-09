"""CPU checks for encounter geometry without launching Isaac Sim."""

import ast
from pathlib import Path
from types import SimpleNamespace

import torch


def test_encounter_gates_and_fixed_crossing():
    path = Path(__file__).resolve().parents[3] / "source/instinct_nav/instinct_nav/navigation/mdp/dynamic_obstacles.py"
    tree = ast.parse(path.read_text())
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_spawn_encounters")
    scope = {"torch": torch, "_terrain_z_at": lambda env, lookup, xyz, ids: torch.zeros(len(ids))}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), scope)
    root = torch.zeros(2, 3)
    data = SimpleNamespace(root_pos_w=root, root_quat_w=torch.tensor([[1., 0., 0., 0.]]).repeat(2, 1),
                           root_lin_vel_w=torch.tensor([[0.7, 0., 0.], [0., 0., 0.]]))
    goal = SimpleNamespace(goal_position_world=torch.tensor([[12., 0., 0.]]).repeat(2, 1))
    manager = SimpleNamespace(
        env=SimpleNamespace(scene={"robot": SimpleNamespace(data=data)},
                            command_manager=SimpleNamespace(_terms={"robot_goal": goal})),
        positions_w=torch.full((2, 5, 3), -100.), spawned_obstacles=torch.zeros(2, 5, dtype=torch.bool),
        starts=torch.zeros(2, 5, 2), ends=torch.zeros(2, 5, 2), dirs=torch.zeros(2, 5, 2),
        speeds=torch.zeros(2, 5), behaviors=torch.zeros(2, 5, dtype=torch.long),
        paused=torch.zeros(2, 5, dtype=torch.bool), spawn_timers=torch.zeros(2, 5),
        difficulty=0., visible_spawn_dist=5., goal_clearance_m=1., device="cpu",
        terrain_lookup=None, spawn_interval_s=1.5, _effective_speed_range=lambda: (0.15, 0.15),
    )
    activated = []
    manager._activate_obstacles = lambda due: activated.append(due.clone())
    scope["_spawn_encounters"](manager, torch.ones(2, 5, dtype=torch.bool))
    assert len(activated) == 1
    assert activated[0].sum() == 1 and activated[0][0, 0]
    assert torch.allclose((manager.starts[0, 0] + manager.ends[0, 0]) / 2, torch.tensor([5., 0.]))
    assert manager.spawn_timers[0].min() >= 1.5
    manager.positions_w[0, 0] = torch.tensor([4., 0., 0.])
    manager.spawned_obstacles[0, 0] = True
    scope["_spawn_encounters"](manager, torch.ones(2, 5, dtype=torch.bool))
    assert len(activated) == 1

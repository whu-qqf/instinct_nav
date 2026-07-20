import math
import importlib.util
import sys
from pathlib import Path

import torch


def _load_adapter_class():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "instinct_nav"
        / "navigation"
        / "mdp"
        / "navigation"
        / "actions"
        / "sonic_command_adapter.py"
    )
    spec = importlib.util.spec_from_file_location("sonic_command_adapter", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.SonicCommandAdapter


SonicCommandAdapter = _load_adapter_class()


def test_zero_command_maps_to_idle():
    adapter = SonicCommandAdapter(num_envs=2, device="cpu")
    command = adapter.convert(torch.zeros(2, 3), dt=0.1)

    assert command["target_vel"].shape == (2, 1)
    assert torch.allclose(command["target_vel"], torch.zeros(2, 1))
    assert torch.equal(command["mode"], torch.zeros(2, 1, dtype=torch.long))
    assert torch.allclose(command["movement_direction"], torch.zeros(2, 3))
    assert torch.allclose(
        command["facing_direction"],
        torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
    )
    assert torch.allclose(command["height"], torch.full((2, 1), -1.0))


def test_forward_command_sets_speed_and_direction():
    adapter = SonicCommandAdapter(num_envs=1, device="cpu")
    command = adapter.convert(torch.tensor([[0.6, 0.0, 0.0]]), dt=0.1)

    assert torch.allclose(command["target_vel"], torch.tensor([[0.6]]))
    assert torch.equal(command["mode"], torch.tensor([[2]], dtype=torch.long))
    assert torch.allclose(command["movement_direction"], torch.tensor([[1.0, 0.0, 0.0]]), atol=1e-6)


def test_strafe_command_sets_lateral_direction():
    adapter = SonicCommandAdapter(num_envs=1, device="cpu")
    command = adapter.convert(torch.tensor([[0.0, 0.4, 0.0]]), dt=0.1)

    assert torch.allclose(command["target_vel"], torch.tensor([[0.4]]))
    assert torch.equal(command["mode"], torch.tensor([[1]], dtype=torch.long))
    assert torch.allclose(command["movement_direction"], torch.tensor([[0.0, 1.0, 0.0]]), atol=1e-6)


def test_yaw_command_integrates_facing_direction():
    adapter = SonicCommandAdapter(num_envs=1, device="cpu")
    command = adapter.convert(torch.tensor([[0.0, 0.0, math.pi]]), dt=0.5)

    assert torch.allclose(command["facing_direction"], torch.tensor([[0.0, 1.0, 0.0]]), atol=1e-5)


def test_speed_is_clipped():
    adapter = SonicCommandAdapter(num_envs=1, device="cpu", max_speed=1.2)
    command = adapter.convert(torch.tensor([[3.0, 4.0, 0.0]]), dt=0.1)

    assert torch.allclose(command["target_vel"], torch.tensor([[1.2]]))
    assert torch.allclose(command["movement_direction"], torch.tensor([[0.6, 0.8, 0.0]]), atol=1e-6)

import importlib.util
import sys
from pathlib import Path

import pytest
import torch


def _load_backend_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "instinct_nav"
        / "navigation"
        / "mdp"
        / "navigation"
        / "actions"
        / "sonic_backend.py"
    )
    spec = importlib.util.spec_from_file_location("sonic_backend_contract", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_instinct_backend_rejects_height_scan_depth_substitution():
    module = _load_backend_module()
    backend = object.__new__(module.InstinctFrozenG1Backend)
    backend.num_envs = 1
    backend.device = torch.device("cpu")
    backend.depth_shape = (8, 18, 32)

    with pytest.raises(ValueError, match="depth_image"):
        backend._extract_depth_image(
            commands={},
            low_level_observations=torch.zeros(1, 99 + 187),
        )


def test_instinct_backend_accepts_exact_depth_history_shape():
    module = _load_backend_module()
    backend = object.__new__(module.InstinctFrozenG1Backend)
    backend.num_envs = 2
    backend.device = torch.device("cpu")
    backend.depth_shape = (8, 18, 32)
    depth = torch.zeros(2, 8, 18, 32)

    assert backend._extract_depth_image({"depth_image": depth}, torch.zeros(2, 99)).shape == (2, 8, 18, 32)

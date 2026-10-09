import importlib.util
from pathlib import Path

import torch


spec = importlib.util.spec_from_file_location(
    "success_tracking",
    Path(__file__).parents[1] / "instinct_nav/navigation/success_tracking.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
classify_navigation_success = module.classify_navigation_success
should_advance_terrain_bank = module.should_advance_terrain_bank


def test_exactly_once_with_failure_precedence():
    names = ["time_out", "base_contact", "large_pitch_angle", "early_termination"]
    terms = torch.tensor(
        [
            [False, False, False, True],
            [False, True, False, True],
            [False, False, True, False],
            [True, False, False, False],
            [False, False, False, False],
        ]
    )
    ids, success = classify_navigation_success(terms, names, torch.arange(5))
    assert ids.tolist() == [0, 1, 2, 3]
    assert success.tolist() == [1.0, 0.0, 0.0, 0.0]


def test_subset_preserves_environment_ids():
    names = ["early_termination"]
    terms = torch.tensor([[False], [True], [False], [True]])
    ids, success = classify_navigation_success(terms, names, torch.tensor([3, 1]))
    assert ids.tolist() == [3, 1]
    assert success.tolist() == [1.0, 1.0]


def test_terrain_bank_gate_requires_success_and_full_window():
    assert not should_advance_terrain_bank(0.899, 1.0, 0.9, 0.95, 0, 4)
    assert not should_advance_terrain_bank(0.95, 0.949, 0.9, 0.95, 0, 4)
    assert should_advance_terrain_bank(0.9, 0.95, 0.9, 0.95, 0, 4)


def test_terrain_bank_gate_stops_at_last_bank():
    assert not should_advance_terrain_bank(1.0, 1.0, 0.9, 0.95, 3, 4)

"""Exactly-once navigation episode outcome classification."""

from __future__ import annotations

import torch


PHYSICAL_FAILURE_TERMS = (
    "obstacle_collision",
    "base_contact",
    "large_pitch_angle",
    "terrain_fall",
)


def should_advance_terrain_bank(
    success_rate: float,
    full_fraction: float,
    success_threshold: float,
    min_full_fraction: float,
    current_bank: int,
    bank_count: int,
) -> bool:
    """Return whether a sequential terrain curriculum may advance one bank."""
    return (
        current_bank < bank_count - 1
        and full_fraction >= min_full_fraction
        and success_rate >= success_threshold
    )


def classify_navigation_success(
    term_dones: torch.Tensor,
    term_names: list[str],
    env_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return terminal env ids and exclusive success values for one reset batch."""
    selected = term_dones[env_ids]
    finished = selected.any(dim=1)
    finished_ids = env_ids[finished]
    selected = selected[finished]
    if finished_ids.numel() == 0:
        return finished_ids, torch.empty(0, device=term_dones.device, dtype=torch.float32)

    name_to_index = {name: index for index, name in enumerate(term_names)}
    early_index = name_to_index.get("early_termination")
    success = (
        selected[:, early_index].clone()
        if early_index is not None
        else torch.zeros(finished_ids.numel(), device=term_dones.device, dtype=torch.bool)
    )
    for name in PHYSICAL_FAILURE_TERMS:
        index = name_to_index.get(name)
        if index is not None:
            success &= ~selected[:, index]
    return finished_ids, success.float()

"""Assignment locking helpers for unindexed motif windows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class LockedAssignmentWindow:
    motif_index: int
    token_start: int
    token_stop: int
    n_tokens: int


def should_lock_assignments(
    *,
    step_idx: int,
    lock_step: int | None,
    already_locked: bool,
    has_assignments: bool,
) -> bool:
    if already_locked or lock_step is None or not has_assignments:
        return False
    return step_idx >= lock_step


def validate_contiguous_assignment_windows(
    assignments: list[Any],
    atom_to_token_map: torch.Tensor,
) -> list[LockedAssignmentWindow]:
    windows: list[LockedAssignmentWindow] = []
    for assignment in assignments:
        ca_idx = assignment.sample_atom_indices.detach().cpu().long()
        if ca_idx.numel() == 0:
            raise ValueError(
                f"motif_unindexing assignment for motif {assignment.motif_index} has no CA atoms"
            )
        tokens = atom_to_token_map[ca_idx].detach().cpu().long()
        if tokens.numel() != torch.unique(tokens).numel():
            raise ValueError(
                f"motif_unindexing assignment for motif {assignment.motif_index} reuses tokens"
            )
        tokens_sorted = torch.sort(tokens).values
        expected = torch.arange(
            int(tokens_sorted[0].item()),
            int(tokens_sorted[0].item()) + int(tokens_sorted.numel()),
            dtype=tokens_sorted.dtype,
        )
        if not torch.equal(tokens_sorted, expected):
            raise ValueError(
                "motif_unindexing locked assignment for motif "
                f"{assignment.motif_index} is not a contiguous token window: "
                f"tokens={tokens_sorted.tolist()}"
            )
        windows.append(
            LockedAssignmentWindow(
                motif_index=int(assignment.motif_index),
                token_start=int(tokens_sorted[0].item()),
                token_stop=int(tokens_sorted[-1].item()) + 1,
                n_tokens=int(tokens_sorted.numel()),
            )
        )
    return windows

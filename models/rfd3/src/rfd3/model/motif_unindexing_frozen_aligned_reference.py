"""Frozen scaffold-frame reference helpers for unindexed motif promotion.

This module supports the Option A cache experiment: once an unindexed motif
assignment has activated, align the original motif reference into the current
scaffold frame exactly once, then reuse that frozen aligned reference for both
runtime promoted features and the static cache rebuild.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from rfd3.model.floating_motif_projection import kabsch_align_all_atom


@dataclass
class FrozenAlignedReferenceState:
    token_mask: torch.Tensor
    projected_atom_mask: torch.Tensor
    reference_pos: torch.Tensor
    frozen_step: int

    def tensors_for(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.token_mask.to(device=device, dtype=torch.bool),
            self.projected_atom_mask.to(device=device, dtype=torch.bool),
            self.reference_pos.to(device=device, dtype=dtype),
        )


def build_frozen_aligned_reference_state(
    assignments: list[Any],
    atom_to_token_map: torch.Tensor,
    xyz: torch.Tensor,
    *,
    n_atoms: int,
    n_tokens: int,
    step_idx: int,
) -> FrozenAlignedReferenceState:
    token_mask = torch.zeros(n_tokens, dtype=torch.bool)
    projected_atom_mask = torch.zeros(n_atoms, dtype=torch.bool)
    reference_pos = torch.full((n_atoms, 3), float("nan"), dtype=torch.float32)

    for assignment in assignments:
        ca_idx = assignment.sample_atom_indices.detach().cpu().long()
        tokens = atom_to_token_map[ca_idx].long()
        token_mask[tokens] = True

        atom_idx = (
            assignment.projection_sample_atom_indices
            if assignment.projection_sample_atom_indices is not None
            else assignment.sample_atom_indices
        ).detach().cpu().long()
        reference_xyz = (
            assignment.projection_reference_xyz
            if assignment.projection_reference_xyz is not None
            else assignment.reference_xyz
        ).detach().cpu().float()

        current = xyz.index_select(dim=-2, index=atom_idx.to(xyz.device))
        reference = reference_xyz.to(device=xyz.device, dtype=xyz.dtype)
        reference = reference.unsqueeze(0).expand(current.shape[0], -1, -1)
        aligned = kabsch_align_all_atom(reference, current).detach()
        if aligned.ndim == 3:
            aligned = aligned[0]

        projected_atom_mask[atom_idx] = True
        reference_pos[atom_idx] = aligned.detach().cpu().float()

    return FrozenAlignedReferenceState(
        token_mask=token_mask,
        projected_atom_mask=projected_atom_mask,
        reference_pos=reference_pos,
        frozen_step=step_idx,
    )

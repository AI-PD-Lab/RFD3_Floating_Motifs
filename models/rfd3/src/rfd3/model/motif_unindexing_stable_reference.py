"""Stable-reference feature helpers for unindexed motif promotion.

This keeps the model-facing motif reference in the original input coordinate
frame.  Kabsch alignment remains an external projection operation; it is not
used to rewrite ref_pos/motif_pos every diffusion step.
"""

from __future__ import annotations

from typing import Any

import torch


def stable_reference_feature_tensors(
    assignments: list[Any],
    atom_to_token_map: torch.Tensor,
    *,
    n_atoms: int,
    n_tokens: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return token/atom masks plus original-frame motif reference positions."""

    token_mask = torch.zeros(n_tokens, dtype=torch.bool, device=device)
    projected_atom_mask = torch.zeros(n_atoms, dtype=torch.bool, device=device)
    reference_pos = torch.full(
        (n_atoms, 3),
        float("nan"),
        dtype=dtype,
        device=device,
    )

    for assignment in assignments:
        ca_idx = assignment.sample_atom_indices.detach().cpu().long()
        tokens = atom_to_token_map[ca_idx].to(device=device, dtype=torch.long)
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
        ).detach().float()

        projected_atom_mask[atom_idx.to(device=device)] = True
        reference_pos[atom_idx.to(device=device)] = reference_xyz.to(
            device=device,
            dtype=reference_pos.dtype,
        )

    return token_mask, projected_atom_mask, reference_pos

"""Build standard atom masks from the RFD3 feature dict `f`.

All masks have shape [L] (number of atoms in the flat atom array).

RFD3 atom representation is atom14 + CCD atoms, where:
  - Virtual atoms (V0–V8) fill unused atom14 slots; identified by is_virtual / element == "VX"
  - No explicit hydrogen atoms in atom14 (all non-virtual atoms are heavy atoms)
  - is_motif_atom_with_fixed_coord  → fixed-coordinate atoms (motif / target chain)
  - ~is_motif_atom_with_fixed_coord → generated/diffused atoms (binder / de novo chain)

Binder / target split:
  Generated atoms ≈ binder  (not fixed, not virtual)
  Fixed atoms     ≈ target  (fixed coord, not virtual)
In PPI design, the newly designed chain is diffused (generated) and the input
structure is fixed (target), making this mapping natural and correct.
"""
from __future__ import annotations

from typing import Any

import torch

from rfd3.potentials.config import PotentialsConfig


def build_masks(f: dict, config: PotentialsConfig) -> dict[str, torch.Tensor]:
    """Adapt RFD3 feature dict `f` into the standard mask set for guidance.

    Returns a dict of [L] boolean tensors.
    """
    fixed_source = f["is_motif_atom_with_fixed_coord"]
    L = fixed_source.shape[0]
    device = _get_device(fixed_source)

    is_virtual = _get_bool_feature(f, "is_virtual", L, device)
    is_fixed = _as_bool_tensor(fixed_source, L, device)
    is_backbone = _get_bool_feature(f, "is_backbone", L, device)
    is_ca = _get_bool_feature(f, "is_ca", L, device)
    is_hydrogen = _infer_hydrogen_mask(f, L, device)

    real_atom_mask = ~is_virtual
    virtual_atom_mask = is_virtual
    fixed_atom_mask = is_fixed
    generated_atom_mask = ~is_fixed

    # Base selection from include_atoms
    include = config.include_atoms
    if include == "all":
        base_mask = torch.ones(L, dtype=torch.bool, device=device)
    elif include == "real":
        base_mask = real_atom_mask.clone()
    elif include == "real_heavy":
        base_mask = real_atom_mask & ~is_hydrogen
    elif include == "backbone":
        base_mask = is_backbone.clone()
    elif include == "CA":
        base_mask = is_ca.clone()
    else:
        base_mask = real_atom_mask.clone()

    # potential_atom_mask: atoms that are inputs to potential functions
    potential_atom_mask = base_mask.clone()
    if config.exclude_virtual_atoms:
        potential_atom_mask = potential_atom_mask & real_atom_mask

    # Binder ~= generated atoms; target ~= fixed-coordinate atoms.  Both masks
    # honor include_atoms/potential_atom_mask so CA/backbone/heavy settings are
    # applied consistently to binder_ROG and interface contacts.
    binder_atom_mask = generated_atom_mask & potential_atom_mask
    target_atom_mask = fixed_atom_mask & potential_atom_mask

    # guide_atom_mask: atoms that receive coordinate updates from guidance
    # Fixed atoms may contribute to potentials (e.g. interface_ncontacts uses
    # target atoms for distance computation) but must NOT receive gradient updates.
    guide_atom_mask = base_mask.clone()
    if config.exclude_virtual_atoms:
        guide_atom_mask = guide_atom_mask & real_atom_mask
    if config.exclude_fixed_atoms:
        guide_atom_mask = guide_atom_mask & ~fixed_atom_mask
    if config.guide_only_generated:
        guide_atom_mask = guide_atom_mask & generated_atom_mask

    return {
        "real_atom_mask": real_atom_mask,
        "virtual_atom_mask": virtual_atom_mask,
        "fixed_atom_mask": fixed_atom_mask,
        "generated_atom_mask": generated_atom_mask,
        "diffused_atom_mask": generated_atom_mask,  # alias
        "binder_atom_mask": binder_atom_mask,
        "target_atom_mask": target_atom_mask,
        # Ligand mask is a placeholder; RFD3 does not expose a separate ligand flag
        # in the feature dict — callers can override via a custom mask builder.
        "ligand_atom_mask": torch.zeros(L, dtype=torch.bool, device=device),
        "backbone_atom_mask": is_backbone,
        "ca_atom_mask": is_ca,
        "potential_atom_mask": potential_atom_mask,
        "guide_atom_mask": guide_atom_mask,
    }


def _get_device(value: Any) -> torch.device:
    if isinstance(value, torch.Tensor):
        return value.device
    return torch.device("cpu")


def _get_bool_feature(
    f: dict,
    key: str,
    L: int,
    device: torch.device,
) -> torch.Tensor:
    return _as_bool_tensor(
        f.get(key, torch.zeros(L, dtype=torch.bool, device=device)),
        L,
        device,
    )


def _as_bool_tensor(value: Any, L: int, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.to(device=device)
    else:
        tensor = torch.as_tensor(value, device=device)
    tensor = tensor.bool()
    if tensor.shape[0] != L:
        raise ValueError(f"Expected atom mask length {L}, got shape {tuple(tensor.shape)}")
    return tensor


def _infer_hydrogen_mask(f: dict, L: int, device: torch.device) -> torch.Tensor:
    """Infer hydrogens from available RFD3 features.

    RFD3 usually has atom14 heavy atoms plus virtual atoms, but atomized ligands
    can expose hydrogen atoms.  The most reliable runtime feature is
    ref_element, which may be integer atomic numbers or one-hot encoded.
    """
    if "is_hydrogen" in f:
        return _as_bool_tensor(f["is_hydrogen"], L, device)

    ref_element = f.get("ref_element")
    if ref_element is None:
        return torch.zeros(L, dtype=torch.bool, device=device)

    if not isinstance(ref_element, torch.Tensor):
        ref_element = torch.as_tensor(ref_element, device=device)
    else:
        ref_element = ref_element.to(device=device)

    if ref_element.ndim == 1:
        return ref_element.long() == 1
    if ref_element.ndim == 2 and ref_element.shape[-1] > 1:
        return ref_element.argmax(dim=-1).long() == 1

    return torch.zeros(L, dtype=torch.bool, device=device)

"""Hetero pseudo-symmetry masks and projection helpers.

The normal symmetry sampler projects a full asymmetric unit onto every copy.
Hetero pseudo-symmetry deliberately does less: it can reuse the same symmetric
initialization, but after initialization only selected oligomer-interface atoms
are regularized.  Motif atoms and nearby motif-contact residues always win mask
priority and are never overwritten here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from rfd3.inference.symmetry.atom_array import FIXED_ENTITY_ID
from rfd3.inference.symmetry.symmetry_utils import apply_symmetry_to_xyz_atomwise


@dataclass
class HeteroPseudoSymmetryConfig:
    post_init_symmetry: Literal["interface_only", "initialization_only"] = (
        "interface_only"
    )
    projection_enabled: bool = True
    hard: bool = False
    weight: float = 1.0
    start_step: int = 0
    stop_after: int | None = None
    schedule: Literal["constant", "linear_decay"] = "constant"
    interface_distance_cutoff: float = 8.0
    interface_sequence_buffer: int = 2
    interface_include_sidechains: bool = True
    motif_contact_exclusion_enabled: bool = True
    motif_contact_distance_cutoff: float = 8.0
    motif_contact_sequence_buffer: int = 1
    support_enabled: bool = True
    support_distance_cutoff: float = 12.0
    support_weight: float = 0.3
    support_sequence_buffer: int = 2
    debug: bool = False


def should_apply_hetero_projection(
    config: HeteroPseudoSymmetryConfig, step_idx: int
) -> bool:
    if config.post_init_symmetry == "initialization_only":
        return False
    if not config.projection_enabled:
        return False
    if step_idx < config.start_step:
        return False
    if config.stop_after is not None and step_idx > config.stop_after:
        return False
    return True


def hetero_projection_weight(
    config: HeteroPseudoSymmetryConfig, step_idx: int
) -> float:
    if config.hard:
        return 1.0
    weight = float(config.weight)
    if config.schedule == "linear_decay" and config.stop_after is not None:
        span = max(config.stop_after - config.start_step, 1)
        progress = min(max(step_idx - config.start_step, 0), span) / span
        weight *= 1.0 - progress
    return max(0.0, min(1.0, weight))


def apply_hetero_pseudo_symmetry(
    X_L: torch.Tensor,
    f: dict,
    config: HeteroPseudoSymmetryConfig,
    step_idx: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float | bool]]:
    """Project only interface/support atoms toward the normal symmetric copy."""

    if not should_apply_hetero_projection(config, step_idx):
        return X_L, {"active": False}

    masks = build_hetero_pseudo_symmetry_masks(X_L, f, config)
    interface_weight = hetero_projection_weight(config, step_idx)
    support_weight = 0.0 if config.hard else interface_weight * config.support_weight

    sym_feats = {k: v for k, v in f.items() if "sym" in k}
    sym_projected = apply_symmetry_to_xyz_atomwise(
        X_L.clone(), sym_feats, partial_diffusion=("partial_t" in f)
    )

    projected = X_L
    if interface_weight > 0.0 and masks["interface_mask"].any():
        projected = _blend_masked(
            projected,
            sym_projected,
            masks["interface_mask"],
            interface_weight,
        )
    if support_weight > 0.0 and masks["support_mask"].any():
        projected = _blend_masked(
            projected,
            sym_projected,
            masks["support_mask"],
            support_weight,
        )

    debug = {
        "active": True,
        "hard": config.hard,
        "interface_weight": interface_weight,
        "support_weight": support_weight,
        "motif_atoms": masks["motif_mask"].sum(dim=-1),
        "motif_contact_atoms": masks["motif_contact_mask"].sum(dim=-1),
        "interface_atoms": masks["interface_mask"].sum(dim=-1),
        "support_atoms": masks["support_mask"].sum(dim=-1),
    }
    return projected.detach(), debug


def build_hetero_pseudo_symmetry_masks(
    X_L: torch.Tensor,
    f: dict,
    config: HeteroPseudoSymmetryConfig,
) -> dict[str, torch.Tensor]:
    """Build [D, L] masks with priority:

    motif > motif-contact exclusion > oligomer interface > support > free.
    """

    squeeze = X_L.ndim == 2
    X = X_L.unsqueeze(0) if squeeze else X_L
    D, L, _ = X.shape
    device = X.device

    real_mask = ~_bool_feature(f, "is_virtual", L, device)
    motif_mask_1d = _motif_mask(f, L, device) & real_mask
    selectable_detection = real_mask & ~motif_mask_1d
    if not config.interface_include_sidechains:
        backbone = _bool_feature(f, "is_backbone", L, device)
        ca = _bool_feature(f, "is_ca", L, device)
        selectable_detection &= backbone | ca

    sym_entity_id = f["sym_entity_id"].to(device=device)
    sym_transform_id = f["sym_transform_id"].to(device=device)
    atom_to_token = f.get("atom_to_token_map")
    atom_to_token = (
        atom_to_token.to(device=device).long()
        if isinstance(atom_to_token, torch.Tensor)
        else torch.arange(L, device=device)
    )

    motif_contact = _motif_contact_mask(
        X,
        motif_mask_1d,
        real_mask,
        sym_entity_id,
        sym_transform_id,
        atom_to_token,
        config,
    )

    interface = torch.zeros((D, L), dtype=torch.bool, device=device)
    for entity_id in torch.unique(sym_entity_id).tolist():
        if int(entity_id) == FIXED_ENTITY_ID:
            continue
        entity_mask = sym_entity_id == int(entity_id)
        transform_ids = torch.unique(sym_transform_id[entity_mask]).tolist()
        for transform_id in transform_ids:
            chain_mask = entity_mask & (sym_transform_id == int(transform_id))
            candidates = chain_mask & selectable_detection
            others = entity_mask & (sym_transform_id != int(transform_id)) & selectable_detection
            if not candidates.any() or not others.any():
                continue
            distances = torch.cdist(X[:, candidates, :], X[:, others, :])
            contact_atoms = distances.amin(dim=-1) <= config.interface_distance_cutoff
            candidate_idx = torch.where(candidates)[0]
            interface[:, candidate_idx] |= contact_atoms

    interface = _expand_atom_mask_by_token_buffer(
        interface,
        atom_to_token,
        sym_entity_id,
        sym_transform_id,
        buffer=config.interface_sequence_buffer,
    )
    interface &= real_mask.unsqueeze(0)
    interface &= ~motif_mask_1d.unsqueeze(0)
    interface &= ~motif_contact

    support = torch.zeros_like(interface)
    if config.support_enabled and interface.any():
        support = _support_mask_from_interface(
            X,
            interface,
            real_mask,
            sym_entity_id,
            sym_transform_id,
            config.support_distance_cutoff,
        )
        support = _expand_atom_mask_by_token_buffer(
            support,
            atom_to_token,
            sym_entity_id,
            sym_transform_id,
            buffer=config.support_sequence_buffer,
        )
        support &= real_mask.unsqueeze(0)
        support &= ~motif_mask_1d.unsqueeze(0)
        support &= ~motif_contact
        support &= ~interface

    motif_b = motif_mask_1d.unsqueeze(0).expand(D, -1)
    if squeeze:
        interface = interface.squeeze(0)
        support = support.squeeze(0)
        motif_contact = motif_contact.squeeze(0)
        motif_b = motif_b.squeeze(0)

    return {
        "motif_mask": motif_b,
        "motif_contact_mask": motif_contact,
        "interface_mask": interface,
        "support_mask": support,
    }


def _motif_contact_mask(
    X: torch.Tensor,
    motif_mask_1d: torch.Tensor,
    real_mask: torch.Tensor,
    sym_entity_id: torch.Tensor,
    sym_transform_id: torch.Tensor,
    atom_to_token: torch.Tensor,
    config: HeteroPseudoSymmetryConfig,
) -> torch.Tensor:
    D, L, _ = X.shape
    motif_contact = torch.zeros((D, L), dtype=torch.bool, device=X.device)
    if not config.motif_contact_exclusion_enabled or not motif_mask_1d.any():
        return motif_contact

    for entity_id in torch.unique(sym_entity_id).tolist():
        if int(entity_id) == FIXED_ENTITY_ID:
            continue
        entity_mask = sym_entity_id == int(entity_id)
        for transform_id in torch.unique(sym_transform_id[entity_mask]).tolist():
            chain_mask = entity_mask & (sym_transform_id == int(transform_id))
            scaffold = chain_mask & real_mask & ~motif_mask_1d
            motif = chain_mask & motif_mask_1d
            if not scaffold.any() or not motif.any():
                continue
            distances = torch.cdist(X[:, scaffold, :], X[:, motif, :])
            contact_atoms = distances.amin(dim=-1) <= config.motif_contact_distance_cutoff
            scaffold_idx = torch.where(scaffold)[0]
            motif_contact[:, scaffold_idx] |= contact_atoms

    return _expand_atom_mask_by_token_buffer(
        motif_contact,
        atom_to_token,
        sym_entity_id,
        sym_transform_id,
        buffer=config.motif_contact_sequence_buffer,
    )


def _support_mask_from_interface(
    X: torch.Tensor,
    interface: torch.Tensor,
    real_mask: torch.Tensor,
    sym_entity_id: torch.Tensor,
    sym_transform_id: torch.Tensor,
    cutoff: float,
) -> torch.Tensor:
    support = torch.zeros_like(interface)
    for entity_id in torch.unique(sym_entity_id).tolist():
        if int(entity_id) == FIXED_ENTITY_ID:
            continue
        entity_mask = sym_entity_id == int(entity_id)
        for transform_id in torch.unique(sym_transform_id[entity_mask]).tolist():
            chain_mask = entity_mask & (sym_transform_id == int(transform_id))
            candidates = chain_mask & real_mask
            candidate_idx = torch.where(candidates)[0]
            if candidate_idx.numel() == 0:
                continue
            for batch_idx in range(X.shape[0]):
                interface_idx = torch.where(interface[batch_idx] & chain_mask)[0]
                if interface_idx.numel() == 0:
                    continue
                distances = torch.cdist(
                    X[batch_idx : batch_idx + 1, candidate_idx, :],
                    X[batch_idx : batch_idx + 1, interface_idx, :],
                ).squeeze(0)
                support[batch_idx, candidate_idx] |= distances.amin(dim=-1) <= cutoff
    return support


def _expand_atom_mask_by_token_buffer(
    mask: torch.Tensor,
    atom_to_token: torch.Tensor,
    sym_entity_id: torch.Tensor,
    sym_transform_id: torch.Tensor,
    buffer: int,
) -> torch.Tensor:
    if buffer <= 0 or not mask.any():
        return mask

    squeeze = mask.ndim == 1
    expanded = mask.unsqueeze(0).clone() if squeeze else mask.clone()
    for batch_idx in range(expanded.shape[0]):
        active_idx = torch.where(expanded[batch_idx])[0]
        for idx in active_idx.tolist():
            token = atom_to_token[idx]
            chain = (sym_entity_id == sym_entity_id[idx]) & (
                sym_transform_id == sym_transform_id[idx]
            )
            token_delta = (atom_to_token - token).abs() <= buffer
            expanded[batch_idx] |= chain & token_delta
    return expanded.squeeze(0) if squeeze else expanded


def _blend_masked(
    current: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    weight: float,
) -> torch.Tensor:
    if mask.ndim == 1:
        mask = mask.unsqueeze(0).expand(current.shape[0], -1)
    blended = current * (1.0 - weight) + target * weight
    return torch.where(mask[..., None], blended, current)


def _motif_mask(f: dict, L: int, device: torch.device) -> torch.Tensor:
    mask = _bool_feature(f, "is_motif_atom_with_fixed_coord", L, device)
    mask |= _bool_feature(f, "is_motif_atom_with_fixed_seq", L, device)
    mask |= _bool_feature(f, "is_motif_atom_unindexed", L, device)
    mask |= _bool_feature(f, "is_motif_atom", L, device)
    return mask


def _bool_feature(
    f: dict,
    key: str,
    L: int,
    device: torch.device,
) -> torch.Tensor:
    value = f.get(key)
    if value is None:
        return torch.zeros(L, dtype=torch.bool, device=device)
    if isinstance(value, torch.Tensor):
        tensor = value.to(device=device)
    else:
        tensor = torch.as_tensor(value, device=device)
    tensor = tensor.bool()
    if tensor.shape[0] != L:
        raise ValueError(f"Expected {key} to have length {L}, got {tuple(tensor.shape)}")
    return tensor

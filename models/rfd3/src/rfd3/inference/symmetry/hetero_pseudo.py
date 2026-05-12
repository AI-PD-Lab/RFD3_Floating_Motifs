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
    recenter_enabled: bool = True
    motif_follow_scaffold_frame: bool = True
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

    X_work = (
        _center_hetero_xyz(X_L, f, partial_diffusion=("partial_t" in f))
        if config.recenter_enabled
        else X_L
    )
    masks = build_hetero_pseudo_symmetry_masks(X_work, f, config)
    interface_weight = hetero_projection_weight(config, step_idx)
    support_weight = 0.0 if config.hard else interface_weight * config.support_weight

    sym_projected = _apply_hetero_symmetry_to_xyz_scaffoldwise(
        X_work.clone(),
        f,
        motif_mask=masks["motif_mask"],
    )

    projected = X_work
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
    if config.motif_follow_scaffold_frame and masks["motif_mask"].any():
        projected, motif_frame_updates = _apply_scaffold_frame_to_motifs(
            X_work,
            projected,
            f,
            motif_mask=masks["motif_mask"],
            frame_mask=masks["interface_mask"] | masks["support_mask"],
        )
    else:
        motif_frame_updates = torch.zeros(
            X_work.shape[0] if X_work.ndim == 3 else 1,
            dtype=torch.long,
            device=X_work.device,
        )

    debug = {
        "active": True,
        "hard": config.hard,
        "recentered": config.recenter_enabled,
        "motif_follow_scaffold_frame": config.motif_follow_scaffold_frame,
        "interface_weight": interface_weight,
        "support_weight": support_weight,
        "motif_atoms": masks["motif_mask"].sum(dim=-1),
        "motif_frame_updates": motif_frame_updates,
        "motif_contact_atoms": masks["motif_contact_mask"].sum(dim=-1),
        "interface_atoms": masks["interface_mask"].sum(dim=-1),
        "support_atoms": masks["support_mask"].sum(dim=-1),
    }
    return projected.detach(), debug


def _center_hetero_xyz(
    X_L: torch.Tensor,
    f: dict,
    partial_diffusion: bool = False,
) -> torch.Tensor:
    """Recenter hetero coordinates without copying any ASU atoms across copies."""
    if partial_diffusion:
        return X_L

    squeeze = X_L.ndim == 2
    X = X_L.unsqueeze(0) if squeeze else X_L
    L = X.shape[-2]
    device = X.device
    sym_entity_id = f["sym_entity_id"].to(device=device)
    fixed_motif_mask = sym_entity_id == FIXED_ENTITY_ID
    movable = ~fixed_motif_mask
    if not movable.any():
        return X_L

    centered = X.clone()
    centered[:, movable, :] = centered[:, movable, :] - centered[
        :, movable, :
    ].mean(dim=1, keepdim=True)
    return centered.squeeze(0) if squeeze else centered


def _apply_hetero_symmetry_to_xyz_scaffoldwise(
    X_L: torch.Tensor,
    f: dict,
    motif_mask: torch.Tensor,
) -> torch.Tensor:
    """Hetero-only symmetry target that tolerates different motif lengths.

    The normal atomwise projector assumes every symmetric copy has exactly the
    same atom count. Heterotypic SymMotif copies can differ in motif length, so
    here we only build symmetry targets for non-motif real atoms by their order
    within each copy. This keeps normal symmetry untouched and gives the
    hetero interface/support masks a compatible target tensor.
    """

    squeeze = X_L.ndim == 2
    X = X_L.unsqueeze(0) if squeeze else X_L
    D, L, _ = X.shape
    device = X.device

    sym_entity_id = f["sym_entity_id"].to(device=device)
    sym_transform_id = f["sym_transform_id"].to(device=device)
    is_sym_asu = f["is_sym_asu"].to(device=device).bool()
    real_mask = ~_bool_feature(f, "is_virtual", L, device)
    motif_1d = motif_mask.any(dim=0) if motif_mask.ndim == 2 else motif_mask

    sym_transforms = {
        int(k): v
        for k, v in f["sym_transform"].items()
        if int(k) != FIXED_ENTITY_ID
    }

    sym_X = X.clone()
    scaffold = real_mask & ~motif_1d
    unique_entity_id = torch.unique(sym_entity_id)
    unique_entity_id = unique_entity_id[unique_entity_id != FIXED_ENTITY_ID]
    for entity_id in unique_entity_id.tolist():
        entity_mask = sym_entity_id == int(entity_id)
        asu_mask = entity_mask & is_sym_asu & scaffold
        if not asu_mask.any():
            continue
        asu_idx = torch.where(asu_mask)[0]
        transform_ids = torch.unique(sym_transform_id[entity_mask]).tolist()
        for target_id in transform_ids:
            target_id = int(target_id)
            target_mask = entity_mask & (sym_transform_id == target_id) & scaffold
            if not target_mask.any() or target_id not in sym_transforms:
                continue
            target_idx = torch.where(target_mask)[0]
            n_match = min(asu_idx.numel(), target_idx.numel())
            if n_match == 0:
                continue
            asu_xyz = X[:, asu_idx[:n_match], :]
            R, T = sym_transforms[target_id]
            projected = torch.einsum(
                "blc,cd->bld", asu_xyz, R.to(device=device, dtype=asu_xyz.dtype)
            ) + T.to(device=device, dtype=asu_xyz.dtype)
            sym_X[:, target_idx[:n_match], :] = projected

    return sym_X.squeeze(0) if squeeze else sym_X


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


def _apply_scaffold_frame_to_motifs(
    before: torch.Tensor,
    after: torch.Tensor,
    f: dict,
    motif_mask: torch.Tensor,
    frame_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Move each hetero motif by the rigid frame implied by its scaffold copy.

    Groups atoms by sym_transform_id (instance index) rather than by
    (sym_entity_id, sym_transform_id).  This allows scaffold atoms and motif
    atoms that belong to the SAME instance but to DIFFERENT entities (e.g.
    heterotypic SymMotif designs where the binder is entity 0 and the scaffold
    placeholder is entity 1) to be correctly linked: the scaffold frame drives
    the motif even when they carry different entity_ids.
    """

    squeeze = before.ndim == 2
    X0 = before.unsqueeze(0) if squeeze else before
    X1 = after.unsqueeze(0) if squeeze else after
    motif_b = motif_mask.unsqueeze(0) if motif_mask.ndim == 1 else motif_mask
    frame_b = frame_mask.unsqueeze(0) if frame_mask.ndim == 1 else frame_mask
    D, L, _ = X0.shape
    device = X0.device

    real_mask = ~_bool_feature(f, "is_virtual", L, device)
    sym_entity_id = f["sym_entity_id"].to(device=device)
    sym_transform_id = f["sym_transform_id"].to(device=device)
    updated = X1.clone()
    update_counts = torch.zeros(D, dtype=torch.long, device=device)

    non_fixed_mask = sym_entity_id != FIXED_ENTITY_ID
    for transform_id in torch.unique(sym_transform_id[non_fixed_mask]).tolist():
        transform_mask = (sym_transform_id == int(transform_id)) & non_fixed_mask
        motif_1d = transform_mask & real_mask & motif_b.any(dim=0)
        if not motif_1d.any():
            continue

        # Use scaffold atoms in this instance (same transform_id) as anchors.
        # These may be in a different entity than the motif atoms, which is the
        # common case for heterotypic SymMotif designs.
        scaffold_mask = transform_mask & real_mask & ~motif_b.any(dim=0)
        default_anchor = scaffold_mask
        for batch_idx in range(D):
            anchors = scaffold_mask & frame_b[batch_idx]
            if not anchors.any():
                anchors = default_anchor
            if not anchors.any():
                continue
            moved = (
                X1[batch_idx, anchors, :] - X0[batch_idx, anchors, :]
            ).norm(dim=-1) > 1e-6
            if not bool(moved.any().item()):
                continue

            R, T = _rigid_transform(
                X0[batch_idx, anchors, :],
                X1[batch_idx, anchors, :],
            )
            motif_idx = torch.where(motif_1d)[0]
            updated[batch_idx, motif_idx, :] = (
                X0[batch_idx, motif_idx, :] @ R + T
            )
            update_counts[batch_idx] += 1

    return (updated.squeeze(0) if squeeze else updated), update_counts


def _rigid_transform(
    source: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    original_dtype = source.dtype
    device_type = source.device.type
    with torch.autocast(device_type=device_type, enabled=False):
        source32 = source.float()
        target32 = target.float()
        source_centroid = source32.mean(dim=0)
        target_centroid = target32.mean(dim=0)
        if source32.shape[0] < 3:
            R = torch.eye(3, device=source.device, dtype=torch.float32)
            T = target_centroid - source_centroid
            return R.to(original_dtype), T.to(original_dtype)

        source_centered = source32 - source_centroid
        target_centered = target32 - target_centroid
        H = source_centered.transpose(-1, -2) @ target_centered
        U, _, Vh = torch.linalg.svd(H)
        R = U @ Vh
        if torch.det(R) < 0:
            U = U.clone()
            U[:, -1] *= -1
            R = U @ Vh
        T = target_centroid - source_centroid @ R
    return R.to(original_dtype), T.to(original_dtype)


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

import logging
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from biotite import structure as struc

logger = logging.getLogger(__name__)

FLOATING_MOTIF_REFERENCE_ANNOTATIONS = (
    "floating_motif_reference_x",
    "floating_motif_reference_y",
    "floating_motif_reference_z",
)

# Annotation that groups atoms into named rigid-body super-motifs.
# Atoms with the same non-empty value are Kabsch-aligned together as one rigid body.
SUPERMOTIF_ID_ANNOTATION = "supermotif_id"

_WARNED_INSUFFICIENT_MOTIFS: set[tuple[str, int]] = set()


@dataclass
class FloatingMotifReference:
    """Reference geometry for one independently floating contig motif segment.

    RFD3 samples atom-level coordinates as a flattened [D, L, 3] tensor. The
    sample indices below are therefore atom indices in that same convention.
    """

    sample_atom_indices: torch.Tensor
    reference_xyz: torch.Tensor
    reference_atom_mask: torch.Tensor
    source_components: tuple[str, ...] = ()


def annotate_floating_motif_reference_coords(atom_array: struc.AtomArray):
    """Store pre-noising coordinates for later inference-time motif projection."""

    coord = np.asarray(atom_array.coord, dtype=np.float32)
    for axis, annotation in enumerate(FLOATING_MOTIF_REFERENCE_ANNOTATIONS):
        atom_array.set_annotation(annotation, coord[:, axis].copy())
    return atom_array


def should_project_floating_motifs(
    step_idx, enabled=False, project_every=1, burn_in=0, stop_after=None
) -> bool:
    if project_every < 1:
        raise ValueError("floating_motif_project_every must be >= 1")
    if not enabled:
        return False
    if step_idx < burn_in:
        return False
    if stop_after is not None and step_idx > stop_after:
        return False
    return ((step_idx - burn_in) % project_every) == 0


def kabsch_align_all_atom(reference_xyz, mobile_xyz, mask=None, eps=1e-8):
    """Fit reference_xyz onto mobile_xyz using all valid all-atom points."""

    squeeze = reference_xyz.ndim == 2
    if squeeze:
        reference_xyz = reference_xyz.unsqueeze(0)
        mobile_xyz = mobile_xyz.unsqueeze(0)
        if mask is not None:
            mask = mask.unsqueeze(0)

    original_dtype = reference_xyz.dtype
    device_type = reference_xyz.device.type
    with torch.autocast(device_type=device_type, enabled=False):
        ref32 = reference_xyz.float()
        mob32 = mobile_xyz.float()

        finite_mask = torch.isfinite(ref32).all(dim=-1) & torch.isfinite(mob32).all(
            dim=-1
        )
        if mask is None:
            mask = finite_mask
        else:
            mask = mask.bool() & finite_mask

        mask32 = mask.float()
        denom = mask32.sum(dim=-1, keepdim=True).clamp_min(eps)

        ref_centroid = (ref32 * mask32[..., None]).sum(dim=-2) / denom
        mob_centroid = (mob32 * mask32[..., None]).sum(dim=-2) / denom

        ref_centered = ref32 - ref_centroid[..., None, :]
        mob_centered = mob32 - mob_centroid[..., None, :]

        ref_centered_masked = ref_centered * mask32[..., None]
        mob_centered_masked = mob_centered * mask32[..., None]

        H = ref_centered_masked.transpose(-1, -2) @ mob_centered_masked
        U, _, Vh = torch.linalg.svd(H)

        R = U @ Vh
        det = torch.det(R)
        needs_flip = det < 0
        if needs_flip.any():
            U_fixed = U.clone()
            U_fixed[needs_flip, :, -1] *= -1
            R = U_fixed @ Vh

        aligned = ref_centered @ R + mob_centroid[..., None, :]
    aligned = aligned.to(original_dtype)
    return aligned.squeeze(0) if squeeze else aligned


def build_floating_motif_references_from_contigs(
    contig_map: Any = None,
    input_pdb_features: Any = None,
    sample_features: dict[str, Any] | None = None,
) -> list[FloatingMotifReference]:
    """Build contig motif references from the inference AtomArray.

    Super-motifs (atoms sharing a non-empty SUPERMOTIF_ID_ANNOTATION value) are
    collected first and aligned together as one rigid body each.  All remaining
    contig motif atoms are then split into independent contiguous segments and
    aligned independently, preserving the original behaviour.
    """

    del contig_map, input_pdb_features
    sample_features = sample_features or {}
    atom_array = sample_features.get("atom_array")
    if (
        atom_array is None
        or "src_component" not in atom_array.get_annotation_categories()
    ):
        return []

    ref_xyz = _reference_coords_from_atom_array(atom_array)
    src_components = np.asarray(atom_array.src_component).astype(str)
    is_contig_motif = _get_contig_motif_atom_mask(atom_array, src_components)
    if not np.any(is_contig_motif):
        return []

    references: list[FloatingMotifReference] = []
    supermotif_src_components: set[str] = set()

    # --- 1. Super-motifs: group non-connected parts into single rigid bodies ---
    if SUPERMOTIF_ID_ANNOTATION in atom_array.get_annotation_categories():
        supermotif_ids = np.asarray(
            atom_array.get_annotation(SUPERMOTIF_ID_ANNOTATION)
        ).astype(str)
        unique_sm_ids = sorted({s for s in supermotif_ids if s})

        for sm_id in unique_sm_ids:
            sm_mask = (supermotif_ids == sm_id) & is_contig_motif
            sm_atom_indices = np.where(sm_mask)[0].astype(np.int64)
            if sm_atom_indices.size == 0:
                continue

            references.append(
                FloatingMotifReference(
                    sample_atom_indices=torch.as_tensor(sm_atom_indices, dtype=torch.long),
                    reference_xyz=torch.as_tensor(ref_xyz[sm_atom_indices]).detach().clone(),
                    reference_atom_mask=torch.as_tensor(
                        _get_reference_atom_mask(atom_array, ref_xyz, sm_atom_indices)
                    ).bool(),
                    source_components=tuple(
                        dict.fromkeys(str(c) for c in src_components[sm_atom_indices])
                    ),
                )
            )
            supermotif_src_components.update(str(c) for c in src_components[sm_mask])

    # --- 2. Remaining atoms: independent contiguous segments (original behaviour) ---
    if supermotif_src_components:
        remaining_motif_mask = is_contig_motif & ~np.isin(
            src_components, list(supermotif_src_components)
        )
    else:
        remaining_motif_mask = is_contig_motif

    for atom_indices in _iter_contiguous_motif_atom_segments(
        atom_array, src_components, remaining_motif_mask
    ):
        atom_indices_np = np.asarray(atom_indices, dtype=np.int64)
        references.append(
            FloatingMotifReference(
                sample_atom_indices=torch.as_tensor(atom_indices_np, dtype=torch.long),
                reference_xyz=torch.as_tensor(ref_xyz[atom_indices_np]).detach().clone(),
                reference_atom_mask=torch.as_tensor(
                    _get_reference_atom_mask(atom_array, ref_xyz, atom_indices_np)
                ).bool(),
                source_components=tuple(
                    dict.fromkeys(str(x) for x in src_components[atom_indices_np])
                ),
            )
        )
    return references


_PROJ_CALL_COUNT = 0


def project_floating_motifs_all_atom(xyz, floating_motif_refs, alpha: float = 1.0):
    """Project each contig motif independently after a denoising update.

    This is an inference-time approximation of floating-anchor diffusion behavior.
    It rigidly projects sampled contig motifs after normal RFD3 denoising; it is
    not true FADiff training and does not alter the model architecture.

    ``alpha`` controls how much of the Kabsch-aligned reference to blend in:
    1.0 (default) is the original hard replacement; 0.0 leaves coordinates
    unchanged.  Values in between give a linear blend
    ``alpha * aligned + (1 - alpha) * current``, which lets the motif geometry
    relax gradually toward the end of the post-activation window.
    """
    # ── TEMPORARY DIAGNOSTIC ──────────────────────────────────────────────────
    global _PROJ_CALL_COUNT
    _PROJ_CALL_COUNT += 1
    logger.info(
        "[proj_diag] call=%d alpha=%.6f n_refs=%d xyz_shape=%s",
        _PROJ_CALL_COUNT,
        alpha,
        len(floating_motif_refs) if floating_motif_refs else 0,
        tuple(xyz.shape),
    )
    # ─────────────────────────────────────────────────────────────────────────

    if not floating_motif_refs:
        return xyz

    squeeze = xyz.ndim == 2
    xyz_batched = xyz.unsqueeze(0) if squeeze else xyz
    projected = xyz_batched.clone()

    for motif_ref in floating_motif_refs:
        atom_idx = motif_ref.sample_atom_indices.to(
            device=xyz_batched.device, dtype=torch.long
        )
        if atom_idx.numel() == 0:
            continue

        current = xyz_batched.index_select(dim=-2, index=atom_idx)
        reference = motif_ref.reference_xyz.to(
            device=xyz_batched.device, dtype=xyz_batched.dtype
        )
        reference = reference.unsqueeze(0).expand(current.shape[0], -1, -1)
        ref_mask = motif_ref.reference_atom_mask.to(device=xyz_batched.device).bool()
        ref_mask = ref_mask.unsqueeze(0).expand(current.shape[0], -1)
        valid_mask = (
            ref_mask
            & torch.isfinite(reference).all(dim=-1)
            & torch.isfinite(current).all(dim=-1)
        )
        valid_counts = valid_mask.sum(dim=-1)
        valid_batches = valid_counts >= 3

        if not bool(valid_batches.all().item()):
            _warn_insufficient_atoms_once(motif_ref, valid_batches)
        if not bool(valid_batches.any().item()):
            continue

        aligned = current.clone()
        aligned_valid = kabsch_align_all_atom(
            reference[valid_batches],
            current[valid_batches],
            valid_mask[valid_batches],
        )
        aligned[valid_batches] = aligned_valid

        replace_mask = valid_mask & torch.isfinite(aligned).all(dim=-1)
        motif_current = projected.index_select(dim=-2, index=atom_idx)
        blended = alpha * aligned + (1.0 - alpha) * motif_current
        # ── TEMPORARY DIAGNOSTIC ─────────────────────────────────────────────
        logger.info(
            "[proj_diag] blend: (aligned-blended).abs().max()=%.6e  "
            "(motif_current-blended).abs().max()=%.6e  "
            "dtypes: aligned=%s blended=%s motif_current=%s  "
            "replace_mask_true=%d/%d",
            float((aligned - blended).abs().max().item()),
            float((motif_current - blended).abs().max().item()),
            str(aligned.dtype),
            str(blended.dtype),
            str(motif_current.dtype),
            int(replace_mask.sum().item()),
            int(replace_mask.numel()),
        )
        # ─────────────────────────────────────────────────────────────────────
        motif_current = torch.where(replace_mask[..., None], blended, motif_current)
        projected.scatter_(
            dim=-2,
            index=atom_idx.view(1, -1, 1).expand(projected.shape[0], -1, 3),
            src=motif_current,
        )

    return projected.squeeze(0) if squeeze else projected


def remove_floating_motif_atoms_from_fixed_mask(mask, floating_motif_refs):
    if not floating_motif_refs:
        return mask
    updated = mask.clone()
    for motif_ref in floating_motif_refs:
        atom_idx = motif_ref.sample_atom_indices.to(
            device=updated.device, dtype=torch.long
        )
        updated[..., atom_idx] = False
    return updated


def _reference_coords_from_atom_array(atom_array):
    if all(
        annotation in atom_array.get_annotation_categories()
        for annotation in FLOATING_MOTIF_REFERENCE_ANNOTATIONS
    ):
        return np.stack(
            [
                atom_array.get_annotation(annotation)
                for annotation in FLOATING_MOTIF_REFERENCE_ANNOTATIONS
            ],
            axis=-1,
        ).astype(np.float32)
    return np.asarray(atom_array.coord, dtype=np.float32)


def _get_contig_motif_atom_mask(atom_array, src_components):
    motif_mask = np.asarray([bool(c) and c[0].isalpha() for c in src_components])
    if "is_motif_atom_unindexed" in atom_array.get_annotation_categories():
        is_unindexed = atom_array.is_motif_atom_unindexed.astype(bool)
        if "is_motif_atom_unindexed_floating_motif" in atom_array.get_annotation_categories():
            is_unindexed &= ~atom_array.is_motif_atom_unindexed_floating_motif.astype(
                bool
            )
        motif_mask &= ~is_unindexed
    if "is_ligand" in atom_array.get_annotation_categories():
        motif_mask &= ~atom_array.is_ligand.astype(bool)
    return motif_mask


def _get_reference_atom_mask(atom_array, ref_xyz, atom_indices):
    mask = np.isfinite(ref_xyz[atom_indices]).all(axis=-1)
    if "occupancy" in atom_array.get_annotation_categories():
        mask &= atom_array.occupancy[atom_indices] > 0.0
    if "is_virtual" in atom_array.get_annotation_categories():
        mask &= ~atom_array.is_virtual[atom_indices].astype(bool)
    if "element" in atom_array.get_annotation_categories():
        mask &= atom_array.element[atom_indices] != "VX"
    return mask


def _iter_contiguous_motif_atom_segments(atom_array, src_components, is_contig_motif):
    starts = struc.get_residue_starts(atom_array, add_exclusive_stop=True)
    segment: list[int] = []
    prev_component = None

    for start, end in zip(starts[:-1], starts[1:]):
        atom_indices = list(range(start, end))
        is_motif_token = bool(np.any(is_contig_motif[start:end]))
        if not is_motif_token:
            if segment:
                yield segment
                segment = []
                prev_component = None
            continue

        component = str(src_components[start])
        if segment and not _are_consecutive_components(prev_component, component):
            yield segment
            segment = []
        segment.extend(atom_indices)
        prev_component = component

    if segment:
        yield segment


def _are_consecutive_components(previous, current):
    prev_parsed = _parse_component(previous)
    cur_parsed = _parse_component(current)
    if prev_parsed is None or cur_parsed is None:
        return previous == current
    prev_chain, prev_resid = prev_parsed
    cur_chain, cur_resid = cur_parsed
    return prev_chain == cur_chain and cur_resid == prev_resid + 1


def _parse_component(component):
    if component is None:
        return None
    match = re.match(r"^([^0-9-]+)(-?\d+)", str(component))
    if match is None:
        return None
    return match.group(1), int(match.group(2))


def _warn_insufficient_atoms_once(motif_ref, valid_batches):
    motif_name = ",".join(motif_ref.source_components) or "<unknown>"
    for batch_idx in torch.where(~valid_batches)[0].detach().cpu().tolist():
        key = (motif_name, int(batch_idx))
        if key not in _WARNED_INSUFFICIENT_MOTIFS:
            logger.warning(
                "Skipping floating motif projection for motif %s batch %d: fewer than 3 valid atoms.",
                motif_name,
                batch_idx,
            )
            _WARNED_INSUFFICIENT_MOTIFS.add(key)

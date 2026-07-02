"""Core guidance computation: run potentials, backprop, apply gradient.

Public API: compute_potential_guidance()

Three apply_mode strategies:

token_translation (preferred, stable):
    Atom gradients are averaged per residue/token, then broadcast back so every
    atom in a residue receives the same rigid translation.  This is the mode that
    most closely mirrors RFdiffusion1's frame-based guidance and avoids internal
    distortions within residues.

atom (direct all-atom):
    Raw atom-level gradients are applied after masking and clipping.  This allows
    per-atom distortions (e.g. sidechain repositioning) but is less stable.
    Use conservative guide_scale (0.05-0.1) and guide_clip_rms (0.005-0.01).

hybrid (blend):
    guidance = token_component + atom_guidance_fraction * (atom_grad - token_component)
    atom_guidance_fraction=0.0 → pure token_translation
    atom_guidance_fraction=1.0 → pure atom mode (before clipping)
    The internal_component carries within-token deformations (bond angle / torsion
    relaxation) on top of the rigid token translation.
"""

from __future__ import annotations

import torch

from rfd3.potentials.manager import PotentialManager


def compute_potential_guidance(
    xyz_t: torch.Tensor,  # [D, L, 3]   current atom coordinates
    potential_manager: PotentialManager,
    t: float,  # current noise level
    T: float,  # maximum noise level (first step)
    masks: dict[str, torch.Tensor],
    metadata: dict,
    apply_mode: str,
    atom_guidance_fraction: float,
) -> tuple[torch.Tensor, dict]:
    """Compute a coordinate perturbation from external potentials.

    Returns:
        guidance  – [D, L, 3] tensor to add to xyz_t (same dtype/device)
        debug_dict – populated only when PotentialManager.debug is True
    """
    if potential_manager.is_empty():
        return torch.zeros_like(xyz_t), {}

    masks = {
        key: value.to(device=xyz_t.device, dtype=torch.bool)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in masks.items()
    }
    base_guide_atom_mask = masks["guide_atom_mask"].to(
        device=xyz_t.device, dtype=torch.bool
    )  # [L]
    atom_to_token_map = metadata["atom_to_token_map"].to(
        device=xyz_t.device, dtype=torch.long
    )  # [L]
    n_tokens = int(metadata.get("n_tokens", atom_to_token_map.max().item() + 1))

    if atom_to_token_map.shape[0] != xyz_t.shape[1]:
        raise ValueError(
            "atom_to_token_map length must match xyz_t atom dimension: "
            f"{atom_to_token_map.shape[0]} != {xyz_t.shape[1]}"
        )
    if base_guide_atom_mask.shape[0] != xyz_t.shape[1]:
        raise ValueError(
            "guide_atom_mask length must match xyz_t atom dimension: "
            f"{base_guide_atom_mask.shape[0]} != {xyz_t.shape[1]}"
        )

    total_guidance = torch.zeros_like(xyz_t)
    total_potential_value = xyz_t.new_zeros(())
    raw_grad_rms_values: list[float] = []
    potential_debug: list[dict] = []
    guide_atom_mask = base_guide_atom_mask

    for potential in potential_manager.potentials:
        potential_value, atom_grad = _potential_atom_gradient(
            potential=potential,
            xyz_t=xyz_t,
            masks=masks,
            metadata=metadata,
        )
        if atom_grad is None:
            if potential_manager.debug:
                skipped_debug = {
                    "type": type(potential).__name__,
                    "value": round(float(potential_value.detach()), 6),
                    "skipped": True,
                    "reason": getattr(
                        potential, "skip_reason", "no_coordinate_gradient"
                    )
                    or "no_coordinate_gradient",
                }
                skip_detail = getattr(potential, "skip_detail", None)
                if skip_detail is not None:
                    skipped_debug["detail"] = skip_detail
                potential_debug.append(skipped_debug)
            continue
        total_potential_value = total_potential_value + potential_value.detach()

        guide_atom_mask = base_guide_atom_mask
        if hasattr(potential, "guide_atom_mask"):
            guide_atom_mask = potential.guide_atom_mask(
                masks=masks,
                metadata=metadata,
                device=xyz_t.device,
            )
            guide_atom_mask = guide_atom_mask.to(device=xyz_t.device, dtype=torch.bool)

        if hasattr(potential, "transform_atom_gradient"):
            atom_grad = potential.transform_atom_gradient(
                atom_grad=atom_grad,
                masks=masks,
                metadata=metadata,
                xyz=xyz_t,
            )

        atom_grad = torch.nan_to_num(atom_grad, nan=0.0, posinf=0.0, neginf=0.0)
        if hasattr(potential, "instance_guide_masks"):
            instance_masks = potential.instance_guide_masks(
                masks=masks,
                metadata=metadata,
                device=xyz_t.device,
            )
        else:
            instance_masks = None
        if instance_masks:
            guidance, instance_debug = _apply_instance_guidance(
                atom_grad=atom_grad,
                instance_masks=instance_masks,
                base_guide_atom_mask=base_guide_atom_mask,
                atom_to_token_map=atom_to_token_map,
                n_tokens=n_tokens,
                potential=potential,
                potential_manager=potential_manager,
                t=t,
                T=T,
                apply_mode=apply_mode,
                atom_guidance_fraction=atom_guidance_fraction,
            )
            total_guidance = total_guidance + guidance
            raw_grad_rms_values.extend(
                item["raw_atom_grad_rms"] for item in instance_debug
            )
            if potential_manager.debug:
                potential_debug.append(
                    {
                        "type": type(potential).__name__,
                        "value": round(float(potential_value.detach()), 6),
                        "instances": instance_debug,
                    }
                )
            continue

        atom_grad = atom_grad * guide_atom_mask[None, :, None].to(dtype=atom_grad.dtype)
        raw_grad_rms = _rms_over_mask(atom_grad, guide_atom_mask)
        raw_grad_rms_values.append(raw_grad_rms)

        potential_apply_mode = getattr(potential, "apply_mode", apply_mode)
        potential_atom_fraction = float(
            getattr(potential, "atom_guidance_fraction", atom_guidance_fraction)
        )
        guidance_unscaled = _apply_guidance_mode(
            atom_grad=atom_grad,
            atom_to_token_map=atom_to_token_map,
            n_tokens=n_tokens,
            guide_atom_mask=guide_atom_mask,
            apply_mode=potential_apply_mode,
            atom_guidance_fraction=potential_atom_fraction,
        )

        clip_factor = 1.0
        clip_rms = potential_manager.get_potential_guide_clip_rms(potential)
        if bool(guide_atom_mask.any()):
            rms = _rms_over_mask(guidance_unscaled, guide_atom_mask)
            if clip_rms == 0.0:
                clip_factor = 0.0
                guidance_unscaled = torch.zeros_like(guidance_unscaled)
            elif rms > 1e-12:
                clip_factor = min(1.0, clip_rms / float(rms))
                guidance_unscaled = guidance_unscaled * clip_factor

        scale = potential_manager.get_potential_guide_scale(potential, t, T)
        guidance = guidance_unscaled * scale
        total_guidance = total_guidance + guidance

        if potential_manager.debug:
            potential_debug.append(
                {
                    "type": type(potential).__name__,
                    "value": round(float(potential_value.detach()), 6),
                    "guide_scale": round(scale, 6),
                    "guide_decay": getattr(
                        potential, "guide_decay", potential_manager.guide_decay
                    ),
                    "guide_clip_rms": round(clip_rms, 6),
                    "clip_factor": round(clip_factor, 6),
                    "apply_mode": potential_apply_mode,
                    "raw_atom_grad_rms": round(raw_grad_rms, 6),
                    "final_guidance_rms": round(
                        _rms_over_mask(guidance, guide_atom_mask), 6
                    ),
                }
            )

    guidance = total_guidance

    # ── debug ─────────────────────────────────────────────────────────────────
    debug_dict: dict = {}
    if potential_manager.debug:
        debug_dict = {
            "t": round(t, 4),
            "potential_value": round(float(total_potential_value), 6),
            "raw_atom_grad_rms": round(max(raw_grad_rms_values, default=0.0), 6),
            "final_guidance_rms": round(_rms_over_mask(guidance, guide_atom_mask), 6),
            "n_guided_atoms": int(guide_atom_mask.sum().item()),
            "potentials": potential_debug,
        }

    return guidance, debug_dict


def _apply_instance_guidance(
    atom_grad: torch.Tensor,
    instance_masks: list[torch.Tensor],
    base_guide_atom_mask: torch.Tensor,
    atom_to_token_map: torch.Tensor,
    n_tokens: int,
    potential,
    potential_manager: PotentialManager,
    t: float,
    T: float,
    apply_mode: str,
    atom_guidance_fraction: float,
) -> tuple[torch.Tensor, list[dict]]:
    """Apply one potential as independent masked instances.

    The scalar potential may be computed as a sum, but each instance gets its
    own mask, guidance-mode reduction, clipping, and scale application.  This is
    important for hetero pseudo-symmetry where motif instances are independent.
    """

    total_guidance = torch.zeros_like(atom_grad)
    debug_items: list[dict] = []
    potential_apply_mode = getattr(potential, "apply_mode", apply_mode)
    potential_atom_fraction = float(
        getattr(potential, "atom_guidance_fraction", atom_guidance_fraction)
    )
    clip_rms = potential_manager.get_potential_guide_clip_rms(potential)
    scale = potential_manager.get_potential_guide_scale(potential, t, T)

    for instance_idx, instance_mask in enumerate(instance_masks):
        instance_mask = instance_mask.to(
            device=atom_grad.device, dtype=torch.bool
        ) & base_guide_atom_mask
        if not bool(instance_mask.any()):
            continue
        instance_grad = atom_grad * instance_mask[None, :, None].to(
            dtype=atom_grad.dtype
        )
        raw_grad_rms = _rms_over_mask(instance_grad, instance_mask)
        guidance_unscaled = _apply_guidance_mode(
            atom_grad=instance_grad,
            atom_to_token_map=atom_to_token_map,
            n_tokens=n_tokens,
            guide_atom_mask=instance_mask,
            apply_mode=potential_apply_mode,
            atom_guidance_fraction=potential_atom_fraction,
        )

        clip_factor = 1.0
        rms = _rms_over_mask(guidance_unscaled, instance_mask)
        if clip_rms == 0.0:
            clip_factor = 0.0
            guidance_unscaled = torch.zeros_like(guidance_unscaled)
        elif rms > 1e-12:
            clip_factor = min(1.0, clip_rms / float(rms))
            guidance_unscaled = guidance_unscaled * clip_factor

        guidance = guidance_unscaled * scale
        total_guidance = total_guidance + guidance
        debug_items.append(
            {
                "instance": instance_idx,
                "n_guided_atoms": int(instance_mask.sum().item()),
                "guide_scale": round(scale, 6),
                "guide_clip_rms": round(clip_rms, 6),
                "clip_factor": round(clip_factor, 6),
                "apply_mode": potential_apply_mode,
                "raw_atom_grad_rms": round(raw_grad_rms, 6),
                "final_guidance_rms": round(
                    _rms_over_mask(guidance, instance_mask), 6
                ),
            }
        )

    return total_guidance, debug_items


def _potential_atom_gradient(
    potential,
    xyz_t: torch.Tensor,
    masks: dict[str, torch.Tensor],
    metadata: dict,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Compute one potential and return its atom-level gradient."""
    with torch.enable_grad():
        xyz_for_grad = xyz_t.detach().clone().requires_grad_(True)
        potential_value = potential.compute(
            xyz=xyz_for_grad,
            masks=masks,
            metadata=metadata,
        )
        if not potential_value.requires_grad:
            return potential_value, None
        potential_value.backward()
        if xyz_for_grad.grad is None:
            return potential_value, None
        return potential_value, xyz_for_grad.grad.clone()


def _apply_guidance_mode(
    atom_grad: torch.Tensor,
    atom_to_token_map: torch.Tensor,
    n_tokens: int,
    guide_atom_mask: torch.Tensor,
    apply_mode: str,
    atom_guidance_fraction: float,
) -> torch.Tensor:
    """Reduce raw atom gradients according to the configured application mode."""
    if apply_mode == "token_translation":
        return _token_translation(
            atom_grad, atom_to_token_map, n_tokens, guide_atom_mask
        )

    if apply_mode == "atom":
        return atom_grad

    if apply_mode == "hybrid":
        token_component = _token_translation(
            atom_grad, atom_to_token_map, n_tokens, guide_atom_mask
        )
        internal_component = atom_grad - token_component
        return token_component + atom_guidance_fraction * internal_component

    raise ValueError(f"Unknown apply_mode: {apply_mode!r}")


def _token_translation(
    atom_grad: torch.Tensor,  # [D, L, 3]
    atom_to_token_map: torch.Tensor,  # [L]   int64
    n_tokens: int,
    guide_atom_mask: torch.Tensor,  # [L]   bool
) -> torch.Tensor:
    """Per-token mean of guided atom gradients, broadcast back to atoms.

    Only guided atoms (guide_atom_mask=True) contribute to the token average.
    Tokens with no guided atoms receive a zero vector.
    All atoms in the same token get the same vector (rigid body translation).
    """
    D, L, _ = atom_grad.shape
    device, dtype = atom_grad.device, atom_grad.dtype

    # Zero out non-guided atoms before averaging
    atom_to_token_map = atom_to_token_map.to(device=device, dtype=torch.long)
    guide_atom_mask = guide_atom_mask.to(device=device, dtype=torch.bool)

    masked_grad = atom_grad * guide_atom_mask[None, :, None].to(dtype=dtype)

    idx_3d = atom_to_token_map[None, :, None].expand(D, L, 3)  # [D, L, 3]
    token_sum = torch.zeros(D, n_tokens, 3, device=device, dtype=dtype)
    token_sum.scatter_add_(1, idx_3d, masked_grad)

    # Count guided atoms per token
    idx_1d = atom_to_token_map[None, :].expand(D, L)  # [D, L]
    token_count = torch.zeros(D, n_tokens, device=device, dtype=dtype)
    token_count.scatter_add_(
        1,
        idx_1d,
        guide_atom_mask[None, :].to(dtype=dtype).expand(D, L),
    )
    token_has_guided = token_count > 0  # [D, n_tokens] — before clamp
    token_count = token_count.clamp(min=1.0)

    token_mean = token_sum / token_count[:, :, None]  # [D, I, 3]

    # Broadcast: each atom gets its token's mean translation
    token_broadcast = token_mean[:, atom_to_token_map, :]  # [D, L, 3]

    # Zero only tokens with no guided atoms; all atoms in a guided token move together
    # (masking by guide_atom_mask here would displace only CA and leave N/C/O fixed,
    # producing impossible bond lengths)
    token_has_guided_per_atom = token_has_guided[:, atom_to_token_map]  # [D, L]
    token_broadcast = token_broadcast * token_has_guided_per_atom[:, :, None].to(dtype=dtype)

    return token_broadcast


def _rms_over_mask(
    tensor: torch.Tensor,  # [D, L, 3]
    mask: torch.Tensor,  # [L]   bool
) -> float:
    mask = mask.to(device=tensor.device, dtype=torch.bool)
    if not bool(mask.any()):
        return 0.0
    guided = tensor[:, mask, :]  # [D, n, 3]
    sq_norms = guided.pow(2).sum(-1)  # [D, n]
    return float(sq_norms.mean().sqrt())

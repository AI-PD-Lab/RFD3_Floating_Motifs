"""Adapter layer: connects the potential system to the RFD3 sampler state.

RFD3 tensor / metadata mapping
───────────────────────────────
xyz_t             ← X_L                        shape [D, L, 3]
atom_to_token_map ← f["atom_to_token_map"]     shape [L]  int64
real_atom_mask    ← ~f["is_virtual"]           shape [L]  bool
virtual_atom_mask ← f["is_virtual"]            shape [L]  bool
fixed_atom_mask   ← f["is_motif_atom_with_fixed_coord"]   shape [L]  bool
generated_mask    ← ~f["is_motif_atom_with_fixed_coord"]  shape [L]  bool
motif_mask       ← any motif conditioning, not only fixed coordinates
backbone_mask     ← f["is_backbone"]           shape [L]  bool
ca_mask           ← f["is_ca"]                 shape [L]  bool
binder_mask       ← generated_mask & real_mask (generated non-virtual atoms)
target_mask       ← fixed_mask    & real_mask  (fixed non-virtual atoms)

All mask building is isolated in masks.py.  This file only owns the adapter
that connects sampler state to the guidance pipeline.
"""
from __future__ import annotations

import logging
import sys
from collections.abc import Mapping

import torch

from rfd3.potentials.config import PotentialsConfig
from rfd3.potentials.guidance import compute_potential_guidance
from rfd3.potentials.manager import PotentialManager
from rfd3.potentials.masks import build_masks
from rfd3.potentials.parsing import parse_potentials

logger = logging.getLogger(__name__)

_FLOATING_MOTIF_REFERENCE_KEYS = (
    "floating_motif_reference_x",
    "floating_motif_reference_y",
    "floating_motif_reference_z",
)


class RFD3PotentialAdapter:
    """Apply external potential guidance at each reverse-diffusion step.

    Usage (inside the sampler loop, after the normal coordinate update):

        if potential_adapter is not None:
            X_L = potential_adapter.apply(X_L, t=t_hat, T=noise_schedule[0])
    """

    def __init__(
        self,
        config: PotentialsConfig,
        f: dict,
        input_pos: torch.Tensor | None = None,
        floating_motif_refs=None,
    ):
        self.config = config

        # Always register floating motif atoms as explicit_motif so motif_atom_mask
        # is correct regardless of select_fixed_atoms / select_unfixed_sequence choices.
        f = _ensure_floating_motifs_in_motif_mask(f, floating_motif_refs)

        # Build masks once (they depend only on f, which is static across steps)
        self.masks = build_masks(f, config)

        atom_to_token_map = f["atom_to_token_map"]
        self.metadata: dict = {
            "atom_to_token_map": atom_to_token_map,
            "n_tokens": int(atom_to_token_map.max().item()) + 1,
        }
        for key in (
            "ref_pos",
            "motif_pos",
            "is_motif_atom_with_fixed_seq",
            "sym_entity_id",
            "sym_transform_id",
            "is_sym_asu",
            "sym_transform",
        ):
            if key in f:
                self.metadata[key] = f[key]
        _add_floating_motif_reference_metadata(self.metadata, f)
        _add_floating_motif_reference_metadata_from_refs(
            self.metadata,
            floating_motif_refs,
            n_atoms=int(atom_to_token_map.shape[0]),
        )
        _add_motif_atom_res_id_metadata(
            self.metadata,
            floating_motif_refs,
            n_atoms=int(atom_to_token_map.shape[0]),
        )
        if input_pos is not None:
            if input_pos.ndim == 3:
                self.metadata["input_pos"] = input_pos[0].detach()
            else:
                self.metadata["input_pos"] = input_pos.detach()

        potentials = parse_potentials(config.guiding_potentials)
        self.manager = PotentialManager(
            potentials=potentials,
            guide_scale=config.guide_scale,
            guide_decay=config.guide_decay,
            guide_clip_rms=config.guide_clip_rms,
            debug=config.debug,
        )
        print(
            "[potentials] adapter initialized "
            f"enabled={self.enabled()} mode={config.apply_mode} "
            f"n_potentials={len(potentials)} "
            f"n_guided_atoms={int(self.masks['guide_atom_mask'].sum().item())} "
            f"n_motif_atoms={int(self.masks['motif_atom_mask'].sum().item())} "
            f"has_floating_ref={'floating_motif_reference_pos' in self.metadata}",
            file=sys.stderr,
            flush=True,
        )

        # Max noise level T captured on the first apply() call (= noise_schedule[0])
        self._T: float | None = None

    # ── public interface ──────────────────────────────────────────────────────

    def enabled(self) -> bool:
        return self.config.enabled and not self.manager.is_empty()

    def should_apply(self, step_idx: int | None) -> bool:
        """Return whether potentials should run at this denoising step."""
        if not self.enabled():
            return False
        if step_idx is None:
            return True
        if step_idx < self.config.guide_start_step:
            return False
        if (
            self.config.guide_stop_after is not None
            and step_idx > self.config.guide_stop_after
        ):
            return False
        return True

    def apply(
        self,
        X_L: torch.Tensor,          # [D, L, 3]  current coordinates
        t: torch.Tensor | float,    # current noise level scalar
        T: float | None = None,     # max noise level (pass noise_schedule[0])
        step_idx: int | None = None,  # denoising step index
    ) -> torch.Tensor:
        """Compute and add potential guidance to X_L.

        Returns a detached tensor so the sampler's grad assertions remain valid.
        The no-potential path (disabled or empty) returns X_L unchanged.
        """
        if not self.should_apply(step_idx):
            return X_L

        t_float = float(t.item() if isinstance(t, torch.Tensor) else t)
        if T is not None:
            self._T = float(T)
        elif self._T is None:
            self._T = t_float

        guidance, debug_dict = compute_potential_guidance(
            xyz_t=X_L,
            potential_manager=self.manager,
            t=t_float,
            T=self._T,
            masks=self.masks,
            metadata=self.metadata,
            apply_mode=self.config.apply_mode,
            atom_guidance_fraction=self.config.atom_guidance_fraction,
        )

        if self.config.debug and debug_dict:
            logger.info("[potentials] %s", debug_dict)
            print(f"[potentials] {debug_dict}", file=sys.stderr, flush=True)

        # detach() ensures X_L.requires_grad stays False, satisfying the
        # sampler's assertion: assert not X_L.requires_grad
        return (X_L + guidance).detach()

    # ── RFD3 state extraction helpers (for completeness / future extension) ──

    @staticmethod
    def extract_xyz(X_L: torch.Tensor) -> torch.Tensor:
        """Identity: X_L IS the coordinate tensor in RFD3."""
        return X_L

    @staticmethod
    def extract_metadata(f: dict) -> dict:
        atom_to_token_map = f["atom_to_token_map"]
        metadata = {
            "atom_to_token_map": atom_to_token_map,
            "n_tokens": int(atom_to_token_map.max().item()) + 1,
        }
        for key in (
            "ref_pos",
            "motif_pos",
            "is_motif_atom_with_fixed_seq",
            "sym_entity_id",
            "sym_transform_id",
            "is_sym_asu",
            "sym_transform",
        ):
            if key in f:
                metadata[key] = f[key]
        _add_floating_motif_reference_metadata(metadata, f)
        return metadata

    @staticmethod
    def add_guidance_to_xyz_next(
        xyz_next: torch.Tensor,
        guidance: torch.Tensor,
    ) -> torch.Tensor:
        """Add guidance to coordinates and detach (keeps sampler assertions intact)."""
        return (xyz_next + guidance).detach()


def build_potential_adapter(
    potentials_config: dict | PotentialsConfig,
    f: dict,
    input_pos: torch.Tensor | None = None,
    floating_motif_refs=None,
) -> RFD3PotentialAdapter | None:
    """Build an adapter from a config dict (or PotentialsConfig) and feature dict.

    Returns None when potentials are disabled or guiding_potentials is empty,
    so the sampler hot-path avoids any overhead.
    """
    plain_config = _to_plain_config(potentials_config)

    if isinstance(plain_config, dict):
        cfg = PotentialsConfig(**plain_config)
    elif isinstance(plain_config, PotentialsConfig):
        cfg = plain_config
    else:
        raise TypeError(
            f"Expected dict or PotentialsConfig, got {type(potentials_config)}"
        )

    if not cfg.enabled or not cfg.guiding_potentials:
        return None

    return RFD3PotentialAdapter(
        cfg,
        f,
        input_pos=input_pos,
        floating_motif_refs=floating_motif_refs,
    )


def _add_floating_motif_reference_metadata(metadata: dict, f: dict) -> None:
    """Carry pre-origin motif coordinates preserved by input parsing."""
    if "floating_motif_reference_pos" in f:
        metadata["floating_motif_reference_pos"] = f["floating_motif_reference_pos"]
        return

    if all(key in f for key in _FLOATING_MOTIF_REFERENCE_KEYS):
        metadata["floating_motif_reference_pos"] = torch.stack(
            [torch.as_tensor(f[key]) for key in _FLOATING_MOTIF_REFERENCE_KEYS],
            dim=-1,
        )


def _add_floating_motif_reference_metadata_from_refs(
    metadata: dict,
    floating_motif_refs,
    n_atoms: int,
) -> None:
    """Use the same motif references as the Kabsch projector, when available."""
    if not floating_motif_refs:
        return

    reference_pos = torch.full((n_atoms, 3), float("nan"), dtype=torch.float32)
    for motif_ref in floating_motif_refs:
        atom_idx = motif_ref.sample_atom_indices.detach().cpu().long()
        reference_xyz = motif_ref.reference_xyz.detach().cpu().float()
        reference_mask = motif_ref.reference_atom_mask.detach().cpu().bool()
        if atom_idx.numel() == 0:
            continue
        valid = (
            reference_mask
            & torch.isfinite(reference_xyz).all(dim=-1)
            & (atom_idx >= 0)
            & (atom_idx < n_atoms)
        )
        if not bool(valid.any()):
            continue
        reference_pos[atom_idx[valid]] = reference_xyz[valid]

    if torch.isfinite(reference_pos).any():
        metadata["floating_motif_reference_pos"] = reference_pos


def _add_motif_atom_res_id_metadata(
    metadata: dict,
    floating_motif_refs,
    n_atoms: int,
) -> None:
    """Carry each motif atom's original residue id (from src_component) into
    metadata as a length-L int tensor ('motif_atom_res_id', -1 where unknown).

    Lets a potential's offline align step (align=True) restrict an external
    align/receptor chain to exactly the contig motif's residues. Without it,
    MinimalOverlapPotential / TargetAnchorDistance / MotifPairAxisDot load the
    WHOLE align_chain_id chain and require it to match the contig motif block's
    atom count -- which fails whenever the align chain (e.g. a full-length
    receptor-template chain) spans residues the contig trimmed away.
    """
    if not floating_motif_refs:
        return

    res_id = torch.full((n_atoms,), -1, dtype=torch.long)
    for motif_ref in floating_motif_refs:
        src_res = getattr(motif_ref, "source_res_ids", None)
        if src_res is None:
            continue
        atom_idx = motif_ref.sample_atom_indices.detach().cpu().long()
        src_res = src_res.detach().cpu().long()
        n = min(atom_idx.numel(), src_res.numel())
        if n == 0:
            continue
        atom_idx = atom_idx[:n]
        src_res = src_res[:n]
        valid = (atom_idx >= 0) & (atom_idx < n_atoms) & (src_res >= 0)
        if not bool(valid.any()):
            continue
        res_id[atom_idx[valid]] = src_res[valid]

    if bool((res_id >= 0).any()):
        metadata["motif_atom_res_id"] = res_id


def _ensure_floating_motifs_in_motif_mask(
    f: dict,
    floating_motif_refs,
) -> dict:
    """Always register floating motif atoms via the explicit_motif path of motif_atom_mask.

    masks.py builds motif_atom_mask = is_fixed | is_fixed_seq | is_unindexed | explicit_motif.
    Using floating_motif_refs (built from src_component, the same source as the Kabsch
    projector) as the authoritative motif-atom source guarantees correct motif block
    detection regardless of which combination of select_fixed_atoms and
    select_unfixed_sequence the user chose — including the partial case where only some
    motif residues have unfixed sequence.

    Any existing is_motif_atom in f is preserved via OR so no prior signal is lost.
    """
    if not floating_motif_refs:
        return f

    n_atoms_tensor = f.get("atom_to_token_map")
    if n_atoms_tensor is None:
        return f
    n_atoms = int(n_atoms_tensor.shape[0])

    motif_atom = torch.zeros(n_atoms, dtype=torch.bool)
    for motif_ref in floating_motif_refs:
        atom_idx = motif_ref.sample_atom_indices.long()
        valid = (atom_idx >= 0) & (atom_idx < n_atoms)
        motif_atom[atom_idx[valid]] = True

    if not bool(motif_atom.any()):
        return f

    existing = f.get("is_motif_atom")
    if existing is not None:
        motif_atom = motif_atom | torch.as_tensor(existing, dtype=torch.bool)

    augmented = dict(f)
    augmented["is_motif_atom"] = motif_atom
    return augmented


def _to_plain_config(config):
    """Accept plain dicts, dataclasses, and OmegaConf DictConfig without coupling."""
    if isinstance(config, PotentialsConfig):
        return config
    try:
        from omegaconf import DictConfig, ListConfig, OmegaConf
    except ImportError:
        pass
    else:
        if isinstance(config, (DictConfig, ListConfig)):
            return OmegaConf.to_container(config, resolve=True)
    if isinstance(config, Mapping):
        return dict(config)
    return config

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


class RFD3PotentialAdapter:
    """Apply external potential guidance at each reverse-diffusion step.

    Usage (inside the sampler loop, after the normal coordinate update):

        if potential_adapter is not None:
            X_L = potential_adapter.apply(X_L, t=t_hat, T=noise_schedule[0])
    """

    def __init__(self, config: PotentialsConfig, f: dict):
        self.config = config

        # Build masks once (they depend only on f, which is static across steps)
        self.masks = build_masks(f, config)

        atom_to_token_map = f["atom_to_token_map"]
        self.metadata: dict = {
            "atom_to_token_map": atom_to_token_map,
            "n_tokens": int(atom_to_token_map.max().item()) + 1,
        }
        for key in ("ref_pos", "motif_pos", "is_motif_atom_with_fixed_seq"):
            if key in f:
                self.metadata[key] = f[key]

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
            f"n_guided_atoms={int(self.masks['guide_atom_mask'].sum().item())}",
            file=sys.stderr,
            flush=True,
        )

        # Max noise level T captured on the first apply() call (= noise_schedule[0])
        self._T: float | None = None

    # ── public interface ──────────────────────────────────────────────────────

    def enabled(self) -> bool:
        return self.config.enabled and not self.manager.is_empty()

    def apply(
        self,
        X_L: torch.Tensor,          # [D, L, 3]  current coordinates
        t: torch.Tensor | float,    # current noise level scalar
        T: float | None = None,     # max noise level (pass noise_schedule[0])
    ) -> torch.Tensor:
        """Compute and add potential guidance to X_L.

        Returns a detached tensor so the sampler's grad assertions remain valid.
        The no-potential path (disabled or empty) returns X_L unchanged.
        """
        if not self.enabled():
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
        for key in ("ref_pos", "motif_pos", "is_motif_atom_with_fixed_seq"):
            if key in f:
                metadata[key] = f[key]
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

    return RFD3PotentialAdapter(cfg, f)


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

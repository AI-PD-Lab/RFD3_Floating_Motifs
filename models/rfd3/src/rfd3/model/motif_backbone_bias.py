"""Backbone-bias-only motif guidance for SymKabschPot inference.

During diffusion: a gradient bias pulls contiguous scaffold CA windows toward
the internal backbone geometry of the reference motif.  No Kabsch paste, no
token-type modification — the model diffuses freely with only a soft shape prior.

After diffusion: a single hard paste places the Kabsch-aligned reference
backbone (N, CA, C, O) onto the best-matching scaffold window — replicating
the process_unindexed_outputs postprocessing step but operating on X_L
directly inside the sampler, before PDB writing.

This is the most "unindexing-like" approach: the model never sees [0,1,0]
token types and there is no per-step hard projection; the postprocessing
handles motif placement cleanly at the end.
"""

from __future__ import annotations

import logging
import math
import sys
from dataclasses import dataclass, field
from typing import Any

import torch

from rfd3.model.floating_motif_projection import (
    FloatingMotifReference,
    kabsch_align_all_atom,
    project_floating_motifs_all_atom,
)
from rfd3.model.motif_unindexing import (
    ExternalMotifReference,
    MotifAssignment,
    MotifUnindexingConfig,
    _candidate_ca_groups,
    _clip_rms,
    _fmt,
    _ordered_ca_atom_indices,
    _rmsd,
    _sample_atom_names,
    load_motif_references,
)

_logger = logging.getLogger(__name__)


class _BiasLogger:
    def info(self, message: str, *args) -> None:
        text = message % args if args else message
        _logger.info(text)
        print(text, file=sys.stderr, flush=True)


logger = _BiasLogger()


@dataclass
class MotifBackboneBiasConfig:
    enabled: bool = False
    motif_pdbs: list[str] = field(default_factory=list)
    use_unindexed_motifs: bool = False
    loss_weight: float = 1.0
    bias_clip_rms: float | None = None
    update_frequency: int = 1
    allow_overlap: bool = False
    postprocess_paste: bool = True
    debug: bool = False
    debug_frequency: int = 5


class MotifBackboneBiasController:
    """Backbone gradient bias throughout diffusion, single hard paste at the end."""

    # Backbone-only: shape signal without sidechain noise.
    BACKBONE_ATOM_NAMES = ("N", "CA", "C", "O")

    def __init__(
        self,
        config: MotifBackboneBiasConfig,
        f: dict[str, Any],
        sample_features: dict[str, Any] | None = None,
    ):
        self.config = config
        _mu_cfg = MotifUnindexingConfig(
            motif_pdbs=list(config.motif_pdbs),
            use_unindexed_motifs=config.use_unindexed_motifs,
        )
        self.motifs: list[ExternalMotifReference] = load_motif_references(
            _mu_cfg, sample_features
        )
        self.ca_atom_indices = _ordered_ca_atom_indices(f)
        self.candidate_groups = _candidate_ca_groups(f, self.ca_atom_indices)
        self.atom_to_token_map = (
            torch.as_tensor(f["atom_to_token_map"], dtype=torch.long).detach().cpu()
        )
        self.sample_atom_names = _sample_atom_names(sample_features)
        self.assignments: list[MotifAssignment] = []
        self.last_update_step: int | None = None

        if self.config.debug:
            logger.info(
                "[motif_backbone_bias] init source=%s n_motifs=%d motif_lengths=%s "
                "n_ca=%d candidate_groups=%s loss_weight=%.4f bias_clip_rms=%s "
                "update_frequency=%d postprocess_paste=%s",
                "unindexed_motifs" if config.use_unindexed_motifs else "motif_pdbs",
                len(self.motifs),
                [f"{m.name or m.path}:{m.n_residues}" for m in self.motifs],
                int(self.ca_atom_indices.numel()),
                [int(g.numel()) for g in self.candidate_groups],
                config.loss_weight,
                _fmt(config.bias_clip_rms),
                config.update_frequency,
                config.postprocess_paste,
            )

    def enabled(self) -> bool:
        return self.config.enabled and bool(self.motifs)

    def apply_bias(self, xyz: torch.Tensor, step_idx: int) -> torch.Tensor:
        """Gradient bias toward backbone shape of each motif — runs every step."""
        if self._should_update(step_idx) or not self.assignments:
            self.assignments = self._find_assignments(xyz.detach())
            self.last_update_step = step_idx
            if self.config.debug and step_idx % self.config.debug_frequency == 0:
                self._log_assignments(step_idx)

        if not self.assignments:
            return xyz

        return self._apply_gradient_bias(xyz, step_idx)

    def postprocess_X_L(self, X_L: torch.Tensor) -> torch.Tensor:
        """Single hard paste after diffusion: Kabsch-align reference backbone onto
        the best-matching scaffold window and overwrite those positions in X_L."""
        if not self.config.postprocess_paste:
            return X_L

        # Re-score on the final X_L to get the definitive best window.
        final_assignments = self._find_assignments(X_L.detach())
        if not final_assignments:
            if self.config.debug:
                logger.info("[motif_backbone_bias] postprocess: no assignments found")
            return X_L

        refs = self._assignments_to_floating_refs(final_assignments)
        if not refs:
            return X_L

        result = project_floating_motifs_all_atom(X_L, refs)

        if self.config.debug:
            logger.info(
                "[motif_backbone_bias] postprocess: pasted %d motif backbone(s)",
                len(refs),
            )
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_assignments(self, xyz: torch.Tensor) -> list[MotifAssignment]:
        candidates_by_motif = [
            self._score_candidates(xyz, motif)
            for motif in self.motifs
        ]
        used: set[int] = set()
        assignments: list[MotifAssignment] = []
        for motif_idx, (motif, candidates) in enumerate(
            zip(self.motifs, candidates_by_motif)
        ):
            for rmsd, ca_idx in candidates:
                atom_set = {int(i) for i in ca_idx.tolist()}
                if not self.config.allow_overlap and atom_set & used:
                    continue
                proj_idx, proj_xyz = self._backbone_reference(motif, ca_idx)
                assignments.append(
                    MotifAssignment(
                        motif_index=motif_idx,
                        sample_atom_indices=ca_idx.long(),
                        reference_xyz=motif.ca_xyz.float(),
                        rmsd=float(rmsd),
                        projection_sample_atom_indices=proj_idx,
                        projection_reference_xyz=proj_xyz,
                    )
                )
                used |= atom_set
                break
        return assignments

    def _score_candidates(
        self,
        xyz: torch.Tensor,
        motif: ExternalMotifReference,
    ) -> list[tuple[float, torch.Tensor]]:
        scored = []
        for ca_idx in self._candidate_windows(motif.n_residues):
            proj_idx, proj_xyz = self._backbone_reference(motif, ca_idx)
            current = xyz.index_select(dim=-2, index=proj_idx.to(xyz.device))
            reference = proj_xyz.to(device=xyz.device, dtype=xyz.dtype)
            reference = reference.unsqueeze(0).expand(current.shape[0], -1, -1)
            aligned = kabsch_align_all_atom(reference, current).detach()
            score = float(_rmsd(current, aligned).mean().item())
            if math.isfinite(score):
                scored.append((score, ca_idx.detach().cpu().long()))
        scored.sort(key=lambda x: x[0])
        return scored

    def _candidate_windows(self, n_residues: int) -> list[torch.Tensor]:
        windows = []
        for group in self.candidate_groups:
            if group.numel() < n_residues:
                continue
            for start in range(0, int(group.numel()) - n_residues + 1):
                windows.append(group[start : start + n_residues])
        return windows

    def _backbone_reference(
        self,
        motif: ExternalMotifReference,
        ca_atom_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (sample_atom_indices, reference_xyz) for backbone atoms only."""
        if not motif.residue_atoms or self.sample_atom_names is None:
            return ca_atom_indices.detach().cpu().long(), motif.ca_xyz.detach().cpu().float()

        ca_idx = ca_atom_indices.detach().cpu().long()
        token_ids = self.atom_to_token_map[ca_idx].tolist()
        sample_indices: list[int] = []
        reference_xyz: list[torch.Tensor] = []

        for residue_i, token_id in enumerate(token_ids):
            if residue_i >= len(motif.residue_atoms):
                break
            ref_atoms = motif.residue_atoms[residue_i]
            token_atom_idx = torch.where(self.atom_to_token_map == int(token_id))[0]
            sample_by_name = {
                str(self.sample_atom_names[int(idx)]): int(idx)
                for idx in token_atom_idx.tolist()
            }
            for atom_name in self.BACKBONE_ATOM_NAMES:
                if atom_name not in ref_atoms or atom_name not in sample_by_name:
                    continue
                sample_indices.append(sample_by_name[atom_name])
                reference_xyz.append(ref_atoms[atom_name])

        if len(sample_indices) < 3:
            return ca_idx, motif.ca_xyz.detach().cpu().float()
        return (
            torch.tensor(sample_indices, dtype=torch.long),
            torch.stack(reference_xyz).float(),
        )

    def _apply_gradient_bias(self, xyz: torch.Tensor, step_idx: int) -> torch.Tensor:
        with torch.enable_grad():
            x = xyz.detach().clone().requires_grad_(True)
            losses = []
            for assignment in self.assignments:
                atom_idx = (
                    assignment.projection_sample_atom_indices
                    if assignment.projection_sample_atom_indices is not None
                    else assignment.sample_atom_indices
                ).to(device=x.device, dtype=torch.long)
                ref_xyz = (
                    assignment.projection_reference_xyz
                    if assignment.projection_reference_xyz is not None
                    else assignment.reference_xyz
                ).to(device=x.device, dtype=x.dtype)

                current = x.index_select(dim=-2, index=atom_idx)
                reference = ref_xyz.unsqueeze(0).expand(current.shape[0], -1, -1)
                aligned = kabsch_align_all_atom(reference, current).detach()
                losses.append((current - aligned).pow(2).sum(dim=-1).mean())

            if not losses:
                return xyz
            loss = torch.stack(losses).mean()
            objective = -self.config.loss_weight * loss
            grad = torch.autograd.grad(objective, x)[0]
            unclipped_rms = torch.sqrt(grad.pow(2).sum(dim=-1).mean()).detach()
            grad = _clip_rms(grad, self.config.bias_clip_rms)
            clipped_rms = torch.sqrt(grad.pow(2).sum(dim=-1).mean()).detach()

        if (
            self.config.debug
            and self.config.debug_frequency > 0
            and step_idx % self.config.debug_frequency == 0
        ):
            logger.info(
                "[motif_backbone_bias] step=%d bias_mse=%.6f grad_rms=%.6f "
                "unclipped_grad_rms=%.6f",
                step_idx,
                float(loss.detach().item()),
                float(clipped_rms.item()),
                float(unclipped_rms.item()),
            )
        return (xyz + grad).detach()

    def _assignments_to_floating_refs(
        self, assignments: list[MotifAssignment]
    ) -> list[FloatingMotifReference]:
        refs = []
        for assignment in assignments:
            atom_idx = (
                assignment.projection_sample_atom_indices
                if assignment.projection_sample_atom_indices is not None
                else assignment.sample_atom_indices
            ).detach().cpu().long()
            ref_xyz = (
                assignment.projection_reference_xyz
                if assignment.projection_reference_xyz is not None
                else assignment.reference_xyz
            ).detach().cpu().float()
            refs.append(
                FloatingMotifReference(
                    sample_atom_indices=atom_idx,
                    reference_xyz=ref_xyz,
                    reference_atom_mask=torch.ones(atom_idx.shape[0], dtype=torch.bool),
                    source_components=(
                        f"backbone_bias_motif_{assignment.motif_index}",
                    ),
                )
            )
        return refs

    def _should_update(self, step_idx: int) -> bool:
        if self.last_update_step is None:
            return True
        return (step_idx - self.last_update_step) >= self.config.update_frequency

    def _log_assignments(self, step_idx: int) -> None:
        logger.info(
            "[motif_backbone_bias] step=%d assignments=%d/%d",
            step_idx,
            len(self.assignments),
            len(self.motifs),
        )
        for assignment in self.assignments:
            motif = self.motifs[assignment.motif_index]
            ca_idx = assignment.sample_atom_indices.detach().cpu()
            n_proj = (
                0
                if assignment.projection_sample_atom_indices is None
                else int(assignment.projection_sample_atom_indices.numel())
            )
            logger.info(
                "[motif_backbone_bias] step=%d motif=%s assigned_ca=%d-%d "
                "backbone_atoms=%d rmsd=%.4f",
                step_idx,
                motif.name or motif.path,
                int(ca_idx[0]),
                int(ca_idx[-1]),
                n_proj,
                assignment.rmsd,
            )


def build_motif_backbone_bias_controller(
    config: dict | MotifBackboneBiasConfig | None,
    f: dict[str, Any],
    sample_features: dict[str, Any] | None = None,
) -> MotifBackboneBiasController | None:
    cfg = _coerce_config(config)
    if not cfg.enabled:
        return None
    if cfg.loss_weight < 0.0:
        raise ValueError("motif_backbone_bias.loss_weight must be non-negative")
    if cfg.bias_clip_rms is not None and cfg.bias_clip_rms <= 0.0:
        raise ValueError("motif_backbone_bias.bias_clip_rms must be positive or null")
    if cfg.update_frequency < 1:
        raise ValueError("motif_backbone_bias.update_frequency must be >= 1")
    if cfg.debug_frequency < 1:
        raise ValueError("motif_backbone_bias.debug_frequency must be >= 1")
    controller = MotifBackboneBiasController(cfg, f, sample_features=sample_features)
    return controller if controller.enabled() else None


def _coerce_config(config) -> MotifBackboneBiasConfig:
    if config is None:
        return MotifBackboneBiasConfig()
    if isinstance(config, MotifBackboneBiasConfig):
        return config
    try:
        from omegaconf import DictConfig, ListConfig, OmegaConf
    except ImportError:
        pass
    else:
        if isinstance(config, (DictConfig, ListConfig)):
            config = OmegaConf.to_container(config, resolve=True)
    if isinstance(config, dict):
        return MotifBackboneBiasConfig(**config)
    raise TypeError(
        f"motif_backbone_bias must be a dict or MotifBackboneBiasConfig, "
        f"got {type(config)}"
    )

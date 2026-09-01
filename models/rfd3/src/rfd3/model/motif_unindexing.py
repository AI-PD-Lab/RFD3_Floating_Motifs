"""External-reference motif unindexing for SymKabschPot inference.

This module intentionally does not participate in the legacy unindexed-token
input path.  Motif PDBs are loaded as external CA-coordinate references only;
no motif atoms are inserted into the generated atom array.
"""

from __future__ import annotations

import logging
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from rfd3.model.floating_motif_projection import (
    FloatingMotifReference,
    kabsch_align_all_atom,
)
from rfd3.model.motif_unindexing_assignment_lock import (
    LockedAssignmentWindow,
    should_lock_assignments,
    validate_contiguous_assignment_windows,
)
from rfd3.model.motif_unindexing_frozen_aligned_reference import (
    FrozenAlignedReferenceState,
    build_frozen_aligned_reference_state,
)
from rfd3.model.motif_unindexing_stable_reference import (
    stable_reference_feature_tensors,
)

_logger = logging.getLogger(__name__)


class _MotifUnindexingLogger:
    def info(self, message: str, *args) -> None:
        text = message % args if args else message
        _logger.info(text)
        print(text, file=sys.stderr, flush=True)


logger = _MotifUnindexingLogger()


@dataclass
class MotifUnindexingConfig:
    enabled: bool = False
    motif_pdbs: list[str] = field(default_factory=list)
    use_unindexed_motifs: bool = False
    target_length: int | None = None
    update_frequency: int = 1
    loss_weight: float = 1.0
    bias_clip_rms: float | None = None
    activation_threshold: float = 2.0
    post_activation_guidance_steps: int = 20
    post_activation_stop_after: int | None = None
    allow_overlap: bool = False
    promote_to_motif_on_activation: bool = False
    floating_project_on_activation: bool = False
    bias_stop_after: int | None = None
    stop_bias_on_activation: bool = False
    promote_immediately_on_assignment: bool = False
    assignment_lock_step: int | None = None
    stable_reference_promotion: bool = False
    frozen_aligned_reference_promotion: bool = False
    activate_immediately_on_assignment: bool = False
    rebuild_static_cache_after_assignment_lock: bool = False
    rebuild_static_cache_after_assignment_lock_mask_only: bool = False
    rebuild_static_cache_after_assignment_lock_motif_pos_only: bool = False
    rebuild_static_cache_after_immediate_activation: bool = False
    rebuild_static_cache_after_activation: bool = False
    rebuild_static_cache_after_frozen_alignment: bool = False
    rebuild_static_cache_at_projection_stop: bool = False
    projection_atom_names: list[str] = field(
        default_factory=lambda: ["N", "CA", "C", "O", "CB"]
    )
    boundary_distance_bias_weight: float = 1.0
    boundary_distance_target: float = 3.8
    boundary_distance_bias_clip_rms: float | None = None
    # Soft Kabsch: replace hard binary paste with a linearly-decaying blend.
    # Disabled by default; when disabled projection_alpha() always returns 1.0,
    # preserving bit-identical behaviour with prior code.
    soft_kabsch_enabled: bool = False
    soft_kabsch_ramp_start_step: int | None = None
    soft_kabsch_alpha_min: float = 0.0
    debug: bool = False
    debug_frequency: int = 1


@dataclass
class ExternalMotifReference:
    path: str
    ca_xyz: torch.Tensor
    residue_atoms: list[dict[str, torch.Tensor]] = field(default_factory=list)
    name: str | None = None

    @property
    def n_residues(self) -> int:
        return int(self.ca_xyz.shape[0])


@dataclass
class MotifAssignment:
    motif_index: int
    sample_atom_indices: torch.Tensor
    reference_xyz: torch.Tensor
    rmsd: float
    projection_sample_atom_indices: torch.Tensor | None = None
    projection_reference_xyz: torch.Tensor | None = None


class MotifUnindexingController:
    """Find motif-like generated regions, bias them, then enable projection."""

    def __init__(
        self,
        config: MotifUnindexingConfig,
        f: dict[str, Any],
        sample_features: dict[str, Any] | None = None,
    ):
        self.config = config
        self.motifs = load_motif_references(config, sample_features)
        self.ca_atom_indices = _ordered_ca_atom_indices(f)
        self.candidate_groups = _candidate_ca_groups(f, self.ca_atom_indices)
        self.atom_to_token_map = torch.as_tensor(
            f["atom_to_token_map"], dtype=torch.long
        ).detach().cpu()
        self.sample_atom_names = _sample_atom_names(sample_features)
        self.assignments: list[MotifAssignment] = []
        self.assignments_locked = False
        self.assignment_locked_step: int | None = None
        self.locked_assignment_windows: list[LockedAssignmentWindow] = []
        self.activated_step: int | None = None
        self.last_update_step: int | None = None
        self._logged_post_activation_stop = False
        self._logged_feature_promotion = False
        self._logged_projection_stop_cache_ready = False
        self._frozen_aligned_reference: FrozenAlignedReferenceState | None = None
        self._last_candidate_counts: list[int] = []
        self._last_best_raw_rmsds: list[float | None] = []
        if self.config.debug:
            logger.info(
                "[motif_unindexing] init source=%s n_motifs=%d motif_lengths=%s "
                "n_ca=%d candidate_groups=%s update_frequency=%d loss_weight=%.4f "
                "bias_clip_rms=%s activation_threshold=%.4f "
                "post_activation_guidance_steps=%d post_activation_stop_after=%s "
                "allow_overlap=%s promote_to_motif_on_activation=%s "
                "promote_immediately_on_assignment=%s "
                "assignment_lock_step=%s "
                "stable_reference_promotion=%s "
                "frozen_aligned_reference_promotion=%s "
                "activate_immediately_on_assignment=%s "
                "rebuild_static_cache_after_assignment_lock=%s "
                "rebuild_static_cache_after_assignment_lock_mask_only=%s "
                "rebuild_static_cache_after_immediate_activation=%s "
                "rebuild_static_cache_after_activation=%s "
                "rebuild_static_cache_after_frozen_alignment=%s "
                "rebuild_static_cache_at_projection_stop=%s "
                "floating_project_on_activation=%s bias_stop_after=%s "
                "projection_atom_names=%s boundary_distance_bias_weight=%.4f "
                "boundary_distance_target=%.4f boundary_distance_bias_clip_rms=%s "
                "soft_kabsch_enabled=%s soft_kabsch_ramp_start_step=%s "
                "soft_kabsch_alpha_min=%.4f",
                "unindexed_motifs" if config.use_unindexed_motifs else "motif_pdbs",
                len(self.motifs),
                [
                    f"{motif.name or motif.path}:{motif.n_residues}"
                    for motif in self.motifs
                ],
                int(self.ca_atom_indices.numel()),
                [int(group.numel()) for group in self.candidate_groups],
                config.update_frequency,
                config.loss_weight,
                _fmt(config.bias_clip_rms),
                config.activation_threshold,
                config.post_activation_guidance_steps,
                "none"
                if config.post_activation_stop_after is None
                else str(config.post_activation_stop_after),
                config.allow_overlap,
                config.promote_to_motif_on_activation,
                config.promote_immediately_on_assignment,
                "none" if config.assignment_lock_step is None else str(config.assignment_lock_step),
                config.stable_reference_promotion,
                config.frozen_aligned_reference_promotion,
                config.activate_immediately_on_assignment,
                config.rebuild_static_cache_after_assignment_lock,
                config.rebuild_static_cache_after_assignment_lock_mask_only,
                config.rebuild_static_cache_after_immediate_activation,
                config.rebuild_static_cache_after_activation,
                config.rebuild_static_cache_after_frozen_alignment,
                config.rebuild_static_cache_at_projection_stop,
                config.floating_project_on_activation,
                "none" if config.bias_stop_after is None else str(config.bias_stop_after),
                config.projection_atom_names,
                config.boundary_distance_bias_weight,
                config.boundary_distance_target,
                _fmt(config.boundary_distance_bias_clip_rms),
                config.soft_kabsch_enabled,
                "none" if config.soft_kabsch_ramp_start_step is None else str(config.soft_kabsch_ramp_start_step),
                config.soft_kabsch_alpha_min,
            )

    def enabled(self) -> bool:
        return self.config.enabled and bool(self.motifs)

    def apply_pre_activation_bias(
        self,
        xyz: torch.Tensor,
        step_idx: int,
    ) -> torch.Tensor:
        if not self.enabled():
            return xyz
        if self.config.bias_stop_after is not None and step_idx > self.config.bias_stop_after:
            return xyz

        if self.activated_step is not None:
            if self.config.stop_bias_on_activation:
                return xyz
            # Post-activation: use frozen assignments as a soft restoring force.
            if not self.assignments:
                return xyz
            return self._apply_bias(xyz, step_idx)

        if (not self.assignments_locked) and (
            self._should_update_assignments(step_idx) or not self.assignments
        ):
            self.assignments = self.find_assignments(xyz.detach())
            self.last_update_step = step_idx
            self._debug_log_assignments(step_idx)
            if should_lock_assignments(
                step_idx=step_idx,
                lock_step=self.config.assignment_lock_step,
                already_locked=self.assignments_locked,
                has_assignments=bool(self.assignments),
            ):
                self._lock_assignments(step_idx)

        if not self.assignments:
            self._debug_log(step_idx, "no assignments available")
            return xyz

        if self.config.activate_immediately_on_assignment:
            self.activated_step = step_idx
            if self.config.debug:
                logger.info(
                    "[motif_unindexing] immediately activated at step %s "
                    "after assignment",
                    step_idx,
                )
            return xyz

        rmsd = self.current_mean_rmsd(xyz.detach())
        self._debug_log(
            step_idx,
            f"mean_rmsd={_fmt(rmsd)} activation_threshold={self.config.activation_threshold:.4f}",
        )
        if rmsd is not None and rmsd <= self.config.activation_threshold:
            self.activated_step = step_idx
            if self.config.debug:
                logger.info(
                    "[motif_unindexing] activated at step %s rmsd=%.4f",
                    step_idx,
                    rmsd,
                )
            return xyz

        return self._apply_bias(xyz, step_idx)

    def _build_floating_refs(self) -> list[FloatingMotifReference]:
        """Build FloatingMotifReferences from current assignments (no activation gate)."""
        refs = []
        for assignment in self.assignments:
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
            refs.append(
                FloatingMotifReference(
                    sample_atom_indices=atom_idx,
                    reference_xyz=reference_xyz,
                    reference_atom_mask=torch.ones(atom_idx.shape[0], dtype=torch.bool),
                    source_components=(f"external_motif_{assignment.motif_index}",),
                )
            )
        return refs

    def active_floating_motif_refs(self, step_idx: int) -> list[FloatingMotifReference]:
        if self.activated_step is None:
            return []
        if not self.is_post_activation_active(step_idx):
            return []
        return self._build_floating_refs()

    def promoted_feature_dict(
        self,
        f: dict[str, Any],
        xyz: torch.Tensor,
        step_idx: int,
    ) -> dict[str, Any]:
        """Expose activated assignments as motif-like runtime features.

        This intentionally leaves fixed-sequence and fixed-coordinate masks
        unchanged so existing select_unfixed_sequence/select_fixed_atoms choices
        are preserved.  The extra signal is the same token-type motif class that
        indexed motifs provide at feature-build time, plus aligned reference
        coordinates for the atoms we can match by name.
        """

        immediate = self.config.promote_immediately_on_assignment
        on_activation = self.config.promote_to_motif_on_activation
        if not immediate and not on_activation:
            return f
        if not self.assignments:
            return f
        if on_activation and not immediate and self.activated_step is None:
            return f

        promoted = dict(f)
        device = self._feature_device(f)
        n_atoms = int(self.atom_to_token_map.shape[0])
        n_tokens = int(self.atom_to_token_map.max().item()) + 1
        if (
            self.config.frozen_aligned_reference_promotion
            and self.activated_step is not None
        ):
            if self._frozen_aligned_reference is None:
                self._frozen_aligned_reference = build_frozen_aligned_reference_state(
                    self.assignments,
                    self.atom_to_token_map,
                    xyz,
                    n_atoms=n_atoms,
                    n_tokens=n_tokens,
                    step_idx=step_idx,
                )
                if self.config.debug:
                    logger.info(
                        "[motif_unindexing] step=%s froze_aligned_reference "
                        "activated_step=%s n_tokens=%d n_backbone_atoms=%d",
                        step_idx,
                        self.activated_step,
                        int(self._frozen_aligned_reference.token_mask.sum().item()),
                        int(self._frozen_aligned_reference.projected_atom_mask.sum().item()),
                    )
            token_mask, projected_atom_mask, reference_pos = (
                self._frozen_aligned_reference.tensors_for(
                    device=device,
                    dtype=xyz.dtype,
                )
            )
            reference_frame = "frozen_aligned"
        elif self.config.stable_reference_promotion:
            token_mask, projected_atom_mask, reference_pos = (
                stable_reference_feature_tensors(
                    self.assignments,
                    self.atom_to_token_map,
                    n_atoms=n_atoms,
                    n_tokens=n_tokens,
                    device=device,
                    dtype=xyz.dtype,
                )
            )
            reference_frame = "stable_input"
        else:
            token_mask = torch.zeros(n_tokens, dtype=torch.bool, device=device)
            projected_atom_mask = torch.zeros(n_atoms, dtype=torch.bool, device=device)
            reference_pos = torch.full(
                (n_atoms, 3),
                float("nan"),
                dtype=xyz.dtype,
                device=xyz.device,
            )

            for assignment in self.assignments:
                ca_idx = assignment.sample_atom_indices.detach().cpu().long()
                tokens = self.atom_to_token_map[ca_idx].to(device=device, dtype=torch.long)
                token_mask[tokens] = True
                atom_idx, aligned = self._aligned_assignment_reference(assignment, xyz)
                projected_atom_mask[atom_idx.to(device=device)] = True
                reference_pos[atom_idx.to(device=xyz.device)] = aligned.to(
                    dtype=reference_pos.dtype
                )
            reference_frame = "kabsch_aligned"

        all_token_atoms = token_mask[self.atom_to_token_map.to(device=device)]
        promoted["is_motif_atom"] = _or_bool_feature(
            promoted.get("is_motif_atom"),
            all_token_atoms,
            device=device,
        )
        promoted["ref_motif_token_type"] = _promote_token_type(
            promoted.get("ref_motif_token_type"),
            token_mask,
            device=device,
        )
        promoted["floating_motif_reference_pos"] = _with_reference_positions(
            promoted.get("floating_motif_reference_pos"),
            reference_pos,
            device=device,
        )
        promoted["motif_pos"] = _with_reference_positions(
            promoted.get("motif_pos"),
            reference_pos,
            device=device,
            fill_value=0.0,
        )
        promoted["ref_pos"] = _with_reference_positions(
            promoted.get("ref_pos"),
            reference_pos,
            device=device,
            fill_value=0.0,
        )
        if "ref_mask" in promoted:
            promoted["ref_mask"] = _or_bool_feature(
                promoted["ref_mask"],
                projected_atom_mask,
                device=device,
            ).to(promoted["ref_mask"].dtype)

        # Activate ref_pos_embedder for matched backbone atoms, exactly as
        # floating motifs do: coord stays unfixed (free to diffuse + Kabsch),
        # but sequence-level reference geometry reaches the pairwise encoder.
        promoted["is_motif_atom_with_fixed_seq"] = _or_bool_feature(
            promoted.get("is_motif_atom_with_fixed_seq"),
            projected_atom_mask,
            device=device,
        )

        if self.config.debug and not self._logged_feature_promotion:
            logger.info(
                "[motif_unindexing] step=%s promoted_to_motif n_tokens=%d "
                "n_token_atoms=%d n_backbone_atoms_with_fixed_seq=%d "
                "reference_frame=%s",
                step_idx,
                int(token_mask.sum().item()),
                int(all_token_atoms.sum().item()),
                int(projected_atom_mask.sum().item()),
                reference_frame,
            )
            self._logged_feature_promotion = True
        return promoted

    def should_rebuild_static_cache_from_promoted_step(self, step_idx: int) -> bool:
        return self.static_cache_rebuild_reason(step_idx) is not None

    def static_cache_rebuild_reason(self, step_idx: int) -> str | None:
        """Return the opt-in cache rebuild mode active at this step.

        By default f_step carries Kabsch-aligned ref_pos in the current scaffold
        frame.  With frozen_aligned_reference_promotion=True, the first
        post-activation f_step freezes that aligned reference and reuses it for
        both runtime features and the optional cache rebuild.
        """
        has_promoted_features = (
            self.config.promote_immediately_on_assignment
            or self.config.promote_to_motif_on_activation
        )
        if not has_promoted_features:
            return None
        # projection_stop fires at a specific step and always replaces any prior
        # cache — check it first so it takes priority over the continuous
        # lock-time reasons that would otherwise mask it.
        if (
            self.config.rebuild_static_cache_at_projection_stop
            and self.config.floating_project_on_activation
            and self._is_final_projection_step(step_idx)
        ):
            if self.config.debug and not self._logged_projection_stop_cache_ready:
                logger.info(
                    "[motif_unindexing] step=%s projection_stop_cache_ready "
                    "activated_step=%s post_activation_stop_after=%s",
                    step_idx,
                    self.activated_step,
                    self.config.post_activation_stop_after,
                )
                self._logged_projection_stop_cache_ready = True
            return "projection_stop"
        if (
            self.config.rebuild_static_cache_after_assignment_lock_motif_pos_only
            and self.assignments_locked
            and self.activated_step is not None
        ):
            # Delay until the first Kabsch projection has fired so that the
            # projected atom positions are actually close to the reference
            # distances being encoded in P_LL.  Rebuilding at lock time (before
            # activation) encodes distances for atoms that are still thousands of
            # Å away, which causes the model to fight the scaffold coherence.
            return "assignment_lock_motif_pos_only"
        if (
            self.config.rebuild_static_cache_after_assignment_lock_mask_only
            and self.assignments_locked
        ):
            return "assignment_lock_mask_only"
        if (
            self.config.rebuild_static_cache_after_assignment_lock
            and self.assignments_locked
        ):
            return "assignment_lock"
        if (
            self.config.rebuild_static_cache_after_frozen_alignment
            and self.config.frozen_aligned_reference_promotion
            and self.config.floating_project_on_activation
            and self.activated_step is not None
            and self._frozen_aligned_reference is not None
        ):
            return "frozen_aligned_after_activation"
        if (
            self.config.rebuild_static_cache_after_activation
            and self.config.floating_project_on_activation
            and self.activated_step is not None
        ):
            return "after_activation"
        if (
            self.config.rebuild_static_cache_after_immediate_activation
            and self.config.activate_immediately_on_assignment
            and self.config.promote_immediately_on_assignment
            and self.activated_step is not None
        ):
            return "immediate_activation"
        return None

    def _is_final_projection_step(self, step_idx: int) -> bool:
        if self.activated_step is None:
            return False
        if self.config.post_activation_stop_after is not None:
            # Rebuild one model step before the configured projection stop.
            # With post_activation_stop_after=160, this fires at step 159.
            return step_idx == max(self.activated_step, self.config.post_activation_stop_after - 1)
        if self.config.post_activation_guidance_steps <= 0:
            return False
        if not self.is_post_activation_active(step_idx):
            return False
        return (
            step_idx - self.activated_step
        ) >= self.config.post_activation_guidance_steps - 1

    def feature_dict_for_initializer(
        self,
        f: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Build a feature dict for promoted_initializer_outputs using the
        motif's original reference coordinates (NOT Kabsch-aligned to current
        noisy positions).

        This is used to rebuild the static C_L / _sl_cached / _sm_cached once,
        so the model's pairwise embedding cache encodes the actual motif
        backbone geometry rather than noise-based Kabsch-aligned positions.

        Returns None if no promote flag is set or no assignments exist; the
        caller should fall back to the original initializer_outputs.
        """
        if not (
            self.config.promote_immediately_on_assignment
            or self.config.promote_to_motif_on_activation
        ):
            return None
        if not self.assignments:
            return None

        f_init = dict(f)
        device = self._feature_device(f)
        n_atoms = int(self.atom_to_token_map.shape[0])
        n_tokens = int(self.atom_to_token_map.max().item()) + 1
        token_mask = torch.zeros(n_tokens, dtype=torch.bool, device=device)
        projected_atom_mask = torch.zeros(n_atoms, dtype=torch.bool, device=device)
        ref_pos = torch.full(
            (n_atoms, 3),
            float("nan"),
            dtype=torch.float32,
            device=device,
        )

        for assignment in self.assignments:
            ca_idx = assignment.sample_atom_indices.detach().cpu().long()
            tokens = self.atom_to_token_map[ca_idx].to(device=device, dtype=torch.long)
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

            projected_atom_mask[atom_idx.to(device=device)] = True
            ref_pos[atom_idx.to(device=device)] = reference_xyz.to(
                device=device, dtype=ref_pos.dtype
            )

        f_init["ref_pos"] = _with_reference_positions(
            f_init.get("ref_pos"),
            ref_pos,
            device=device,
            fill_value=0.0,
        )
        f_init["is_motif_atom_with_fixed_seq"] = _or_bool_feature(
            f_init.get("is_motif_atom_with_fixed_seq"),
            projected_atom_mask,
            device=device,
        )
        f_init["ref_motif_token_type"] = _promote_token_type(
            f_init.get("ref_motif_token_type"),
            token_mask,
            device=device,
        )
        all_token_atoms = token_mask[self.atom_to_token_map.to(device=device)]
        f_init["is_motif_atom"] = _or_bool_feature(
            f_init.get("is_motif_atom"),
            all_token_atoms,
            device=device,
        )
        if "ref_mask" in f_init:
            f_init["ref_mask"] = _or_bool_feature(
                f_init["ref_mask"],
                projected_atom_mask,
                device=device,
            ).to(f_init["ref_mask"].dtype)
        return f_init

    def feature_dict_for_mask_only_initializer(
        self,
        f: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Build static-cache features with motif identity but no coordinates."""
        if not self.assignments:
            return None

        f_init = dict(f)
        device = self._feature_device(f)
        n_atoms = int(self.atom_to_token_map.shape[0])
        n_tokens = int(self.atom_to_token_map.max().item()) + 1
        token_mask = torch.zeros(n_tokens, dtype=torch.bool, device=device)
        projected_atom_mask = torch.zeros(n_atoms, dtype=torch.bool, device=device)

        for assignment in self.assignments:
            ca_idx = assignment.sample_atom_indices.detach().cpu().long()
            tokens = self.atom_to_token_map[ca_idx].to(device=device, dtype=torch.long)
            token_mask[tokens] = True
            atom_idx = (
                assignment.projection_sample_atom_indices
                if assignment.projection_sample_atom_indices is not None
                else assignment.sample_atom_indices
            ).detach().cpu().long()
            projected_atom_mask[atom_idx.to(device=device)] = True

        all_token_atoms = token_mask[self.atom_to_token_map.to(device=device)]
        f_init["ref_motif_token_type"] = _promote_token_type(
            f_init.get("ref_motif_token_type"),
            token_mask,
            device=device,
        )
        f_init["is_motif_atom"] = _or_bool_feature(
            f_init.get("is_motif_atom"),
            all_token_atoms,
            device=device,
        )
        f_init["is_motif_atom_with_fixed_seq"] = _or_bool_feature(
            f_init.get("is_motif_atom_with_fixed_seq"),
            projected_atom_mask,
            device=device,
        )
        if self.config.debug:
            logger.info(
                "[motif_unindexing] mask_only_initializer_features "
                "n_tokens=%d n_token_atoms=%d n_backbone_atoms_with_fixed_seq=%d",
                int(token_mask.sum().item()),
                int(all_token_atoms.sum().item()),
                int(projected_atom_mask.sum().item()),
            )
        return f_init

    def feature_dict_for_motif_pos_only_initializer(
        self,
        f: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Build cache features encoding only intra-motif pairwise distances.

        Writes motif_pos (original reference coords) and a motif_id tensor
        (integer, -1 for non-motif) so the SinusoidalDistEmbed fires for
        intra-motif atom pairs.  encoders.py uses motif_id >= 0 as the gate
        instead of is_motif_atom_with_fixed_coord, so C_L is built without the
        "fixed-coord" atom encoding — consistent with f_step at runtime.

        is_motif_atom_with_fixed_seq is intentionally NOT set: that would activate
        the ref_pos_embedder with the scaffold's generic ref_pos rather than real
        motif backbone geometry, polluting the pairwise cache with wrong geometry.

        ref_pos is intentionally left untouched: the PositionPairDistEmbedder encodes
        3-D direction vectors that depend on the global coordinate frame, which is
        inconsistent with the scaffold orientation mid-run.  Distances via motif_pos
        are frame-invariant and safe to bake in from the original reference.
        """
        if not self.assignments:
            return None

        f_init = dict(f)
        device = self._feature_device(f)
        n_atoms = int(self.atom_to_token_map.shape[0])
        n_tokens = int(self.atom_to_token_map.max().item()) + 1
        token_mask = torch.zeros(n_tokens, dtype=torch.bool, device=device)
        projected_atom_mask = torch.zeros(n_atoms, dtype=torch.bool, device=device)
        motif_id = torch.full((n_atoms,), -1, dtype=torch.long, device=device)
        motif_pos = torch.full(
            (n_atoms, 3), float("nan"), dtype=torch.float32, device=device
        )

        for assignment in self.assignments:
            ca_idx = assignment.sample_atom_indices.detach().cpu().long()
            tokens = self.atom_to_token_map[ca_idx].to(device=device, dtype=torch.long)
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

            atom_idx_dev = atom_idx.to(device=device)
            projected_atom_mask[atom_idx_dev] = True
            motif_id[atom_idx_dev] = assignment.motif_index
            motif_pos[atom_idx_dev] = reference_xyz.to(
                device=device, dtype=motif_pos.dtype
            )

        all_token_atoms = token_mask[self.atom_to_token_map.to(device=device)]

        f_init["motif_pos"] = _with_reference_positions(
            f_init.get("motif_pos"),
            motif_pos,
            device=device,
            fill_value=0.0,
        )
        # motif_id gates SinusoidalDistEmbed to intra-motif pairs only.
        # encoders.py uses motif_id >= 0 as the valid_mask gate instead of
        # is_motif_atom_with_fixed_coord, so the C_L atom embeddings are NOT
        # encoded as "fixed-coord" atoms.  This keeps the cache-time C_L
        # consistent with f_step (where is_motif_atom_with_fixed_coord is
        # always False for unindexed motifs).
        f_init["motif_id"] = motif_id
        # is_motif_atom_with_fixed_seq is intentionally NOT set here: activating
        # the ref_pos_embedder with the scaffold's generic ref_pos (not the real
        # motif backbone geometry) would encode wrong intra-residue geometry into
        # P_LL, violating the "motif_pos distances only" contract.
        f_init["ref_motif_token_type"] = _promote_token_type(
            f_init.get("ref_motif_token_type"),
            token_mask,
            device=device,
        )
        f_init["is_motif_atom"] = _or_bool_feature(
            f_init.get("is_motif_atom"),
            all_token_atoms,
            device=device,
        )
        if "ref_mask" in f_init:
            f_init["ref_mask"] = _or_bool_feature(
                f_init["ref_mask"],
                projected_atom_mask,
                device=device,
            ).to(f_init["ref_mask"].dtype)
        if self.config.debug:
            logger.info(
                "[motif_unindexing] motif_pos_only_initializer_features "
                "n_motifs=%d n_tokens=%d n_token_atoms=%d n_backbone_atoms=%d",
                len(self.assignments),
                int(token_mask.sum().item()),
                int(all_token_atoms.sum().item()),
                int(projected_atom_mask.sum().item()),
            )
        return f_init

    def is_post_activation_active(self, step_idx: int) -> bool:
        if self.activated_step is None:
            return False
        if self.config.post_activation_stop_after is not None:
            is_active = step_idx <= self.config.post_activation_stop_after
            stop_description = (
                f"post_activation_stop_after={self.config.post_activation_stop_after}"
            )
        else:
            is_active = (
                step_idx - self.activated_step
            ) < self.config.post_activation_guidance_steps
            stop_description = (
                "post_activation_guidance_steps="
                f"{self.config.post_activation_guidance_steps}"
            )
        if (
            self.config.debug
            and not is_active
            and not self._logged_post_activation_stop
            and step_idx >= self.activated_step
        ):
            logger.info(
                "[motif_unindexing] post-activation projection stopped at step %s "
                "(activated_step=%s, %s)",
                step_idx,
                self.activated_step,
                stop_description,
            )
            self._logged_post_activation_stop = True
        return is_active

    def log_diagnostics(self, xyz: torch.Tensor, step_idx: int) -> None:
        """Log per-step motif RMSD and CA–CA chain-break diagnostics.

        Active only when ``debug=True``, ``activated_step`` is set, and the
        current step is a multiple of ``debug_frequency``.  Designed to be
        called unconditionally from the sampler; it is a no-op otherwise.
        """
        if not self.config.debug:
            return
        if self.activated_step is None:
            return
        if not self._should_debug(step_idx):
            return

        # --- Per-assignment RMSD (post-projection, Kabsch-aligned) ---
        alpha = self.projection_alpha(step_idx)
        for assignment in self.assignments:
            atom_idx = (
                assignment.projection_sample_atom_indices
                if assignment.projection_sample_atom_indices is not None
                else assignment.sample_atom_indices
            ).to(device=xyz.device, dtype=torch.long)
            reference_xyz = (
                assignment.projection_reference_xyz
                if assignment.projection_reference_xyz is not None
                else assignment.reference_xyz
            ).to(device=xyz.device, dtype=xyz.dtype)
            current = xyz.index_select(dim=-2, index=atom_idx)
            reference = reference_xyz.unsqueeze(0).expand(current.shape[0], -1, -1)
            aligned = kabsch_align_all_atom(reference, current).detach()
            rmsd_val = float(_rmsd(current, aligned).mean().item())
            motif = self.motifs[assignment.motif_index]
            logger.info(
                "[motif_unindexing] step=%s diag motif=%s rmsd=%.4f alpha=%.4f",
                step_idx,
                motif.name or motif.path,
                rmsd_val,
                alpha,
            )

        # --- CA–CA chain-break detection ---
        xyz_cpu = xyz.detach().cpu().float()
        # Use first batch element for diagnostics (batch dim may be 1 or absent)
        xyz_2d = xyz_cpu[0] if xyz_cpu.ndim == 3 else xyz_cpu
        for group in self.candidate_groups:
            ca_indices = group.tolist()
            for i in range(len(ca_indices) - 1):
                idx_a = ca_indices[i]
                idx_b = ca_indices[i + 1]
                pos_a = xyz_2d[idx_a]
                pos_b = xyz_2d[idx_b]
                dist = float(torch.linalg.norm(pos_b - pos_a).item())
                if dist > 4.5:
                    logger.info(
                        "[motif_unindexing] step=%s chain_break "
                        "ca_atom_indices=%d-%d distance=%.3f",
                        step_idx,
                        idx_a,
                        idx_b,
                        dist,
                    )

    def projection_alpha(self, step_idx: int) -> float:
        """Return the Kabsch blend weight for this diffusion step.

        Returns 1.0 (full hard Kabsch) unless ``soft_kabsch_enabled=True`` and
        the current step is within the configured ramp window.  The ramp runs
        linearly from 1.0 at ``soft_kabsch_ramp_start_step`` down to
        ``soft_kabsch_alpha_min`` at the final active projection step.

        Step indices count upward from 0 (high noise) to ~num_timesteps-2
        (low noise / structured), so the ramp alpha *decreases* as step_idx
        increases toward the end of the projection window.
        """
        if not self.config.soft_kabsch_enabled:
            return 1.0
        if self.activated_step is None:
            return 1.0
        ramp_start = self.config.soft_kabsch_ramp_start_step
        if ramp_start is None or step_idx < ramp_start:
            return 1.0
        # Determine the last step at which projection is active (inclusive).
        if self.config.post_activation_stop_after is not None:
            ramp_end = self.config.post_activation_stop_after
        else:
            ramp_end = self.activated_step + self.config.post_activation_guidance_steps - 1
        if ramp_end <= ramp_start:
            return float(self.config.soft_kabsch_alpha_min)
        progress = (step_idx - ramp_start) / (ramp_end - ramp_start)
        progress = max(0.0, min(1.0, progress))
        return 1.0 + (self.config.soft_kabsch_alpha_min - 1.0) * progress

    def apply_boundary_distance_bias(
        self,
        xyz: torch.Tensor,
        step_idx: int,
    ) -> torch.Tensor:
        """Pull scaffold CA atoms adjacent to projected motifs to CA-CA distance.

        The projected motif boundary positions are detached from the loss, so
        this nudges only the neighboring scaffold atoms and does not suppress
        rigid-body motion of the floating motif itself.
        """
        if (
            not self.assignments
            or self.activated_step is None
            or not self.is_post_activation_active(step_idx)
            or self.config.boundary_distance_bias_weight <= 0.0
        ):
            return xyz

        boundary_pairs = self._boundary_distance_pairs()
        if not boundary_pairs:
            return xyz

        with torch.enable_grad():
            x = xyz.detach().clone().requires_grad_(True)
            losses = []
            outside_indices = torch.tensor(
                [outside_idx for outside_idx, _ in boundary_pairs],
                device=x.device,
                dtype=torch.long,
            )
            for outside_idx, motif_idx in boundary_pairs:
                outside = x.index_select(
                    dim=-2,
                    index=torch.tensor([outside_idx], device=x.device, dtype=torch.long),
                )
                motif_boundary = x.index_select(
                    dim=-2,
                    index=torch.tensor([motif_idx], device=x.device, dtype=torch.long),
                ).detach()
                distance = torch.linalg.norm(outside - motif_boundary, dim=-1)
                target = torch.as_tensor(
                    self.config.boundary_distance_target,
                    dtype=x.dtype,
                    device=x.device,
                )
                losses.append((distance - target).pow(2).mean())
            if not losses:
                return xyz
            loss = torch.stack(losses).mean()
            objective = -self.config.boundary_distance_bias_weight * loss
            grad = torch.autograd.grad(objective, x, retain_graph=False)[0]
            grad_mask = torch.zeros(x.shape[-2], dtype=torch.bool, device=x.device)
            grad_mask[outside_indices] = True
            grad = grad * grad_mask.view(*([1] * (grad.ndim - 2)), -1, 1)
            unclipped_grad_rms = torch.sqrt(grad.pow(2).sum(dim=-1).mean()).detach()
            clip_rms = (
                self.config.boundary_distance_bias_clip_rms
                if self.config.boundary_distance_bias_clip_rms is not None
                else self.config.bias_clip_rms
            )
            grad = _clip_rms(grad, clip_rms)
            clipped_grad_rms = torch.sqrt(grad.pow(2).sum(dim=-1).mean()).detach()

        if self._should_debug(step_idx):
            logger.info(
                "[motif_unindexing] step=%s boundary_distance_mse=%.6f "
                "boundary_pairs=%d grad_rms=%.6f unclipped_grad_rms=%.6f",
                step_idx,
                float(loss.detach().item()),
                len(boundary_pairs),
                float(clipped_grad_rms.item()),
                float(unclipped_grad_rms.item()),
            )
        return (xyz + grad).detach()

    def find_assignments(self, xyz: torch.Tensor) -> list[MotifAssignment]:
        candidates_by_motif = [
            self._score_candidates_for_motif(xyz, motif_idx, motif)
            for motif_idx, motif in enumerate(self.motifs)
        ]
        self._last_candidate_counts = [len(candidates) for candidates in candidates_by_motif]
        self._last_best_raw_rmsds = [
            float(candidates[0][0]) if candidates else None
            for candidates in candidates_by_motif
        ]
        used: set[int] = set()
        assignments: list[MotifAssignment] = []
        for motif_idx, candidates in enumerate(candidates_by_motif):
            for rmsd, atom_idx in candidates:
                atom_set = set(int(i) for i in atom_idx.detach().cpu().tolist())
                if not self.config.allow_overlap and atom_set & used:
                    continue
                motif = self.motifs[motif_idx]
                assignment = MotifAssignment(
                    motif_index=motif_idx,
                    sample_atom_indices=atom_idx.detach().cpu().long(),
                    reference_xyz=motif.ca_xyz.detach().cpu().float(),
                    rmsd=float(rmsd),
                )
                (
                    assignment.projection_sample_atom_indices,
                    assignment.projection_reference_xyz,
                ) = self._projection_reference_for_ca_indices(motif, atom_idx)
                assignments.append(assignment)
                used |= atom_set
                break
        return assignments

    def current_mean_rmsd(self, xyz: torch.Tensor) -> float | None:
        rmsds = []
        for assignment in self.assignments:
            atom_idx = (
                assignment.projection_sample_atom_indices
                if assignment.projection_sample_atom_indices is not None
                else assignment.sample_atom_indices
            )
            reference_xyz = (
                assignment.projection_reference_xyz
                if assignment.projection_reference_xyz is not None
                else assignment.reference_xyz
            )
            atom_idx = atom_idx.to(
                device=xyz.device, dtype=torch.long
            )
            current = xyz.index_select(dim=-2, index=atom_idx)
            reference = reference_xyz.to(
                device=xyz.device, dtype=xyz.dtype
            )
            reference = reference.unsqueeze(0).expand(current.shape[0], -1, -1)
            aligned = kabsch_align_all_atom(reference, current).detach()
            rmsds.append(_rmsd(current, aligned).mean())
        if not rmsds:
            return None
        return float(torch.stack(rmsds).mean().item())

    def _score_candidates_for_motif(
        self,
        xyz: torch.Tensor,
        motif_idx: int,
        motif: ExternalMotifReference,
    ) -> list[tuple[float, torch.Tensor]]:
        del motif_idx
        scored = []
        for atom_idx in self._candidate_windows(motif.n_residues):
            score_atom_idx, score_reference = self._projection_reference_for_ca_indices(
                motif,
                atom_idx,
            )
            current = xyz.index_select(dim=-2, index=score_atom_idx.to(xyz.device))
            reference = score_reference.to(device=xyz.device, dtype=xyz.dtype)
            reference = reference.unsqueeze(0).expand(current.shape[0], -1, -1)
            aligned = kabsch_align_all_atom(reference, current).detach()
            score = float(_rmsd(current, aligned).mean().item())
            if math.isfinite(score):
                scored.append((score, atom_idx.detach().cpu().long()))
        scored.sort(key=lambda item: item[0])
        return scored

    def _candidate_windows(self, n_residues: int) -> list[torch.Tensor]:
        if n_residues < 1:
            return []
        windows = []
        for group in self.candidate_groups:
            if group.numel() < n_residues:
                continue
            for start in range(0, int(group.numel()) - n_residues + 1):
                windows.append(group[start : start + n_residues])
        return windows

    def _boundary_distance_pairs(self) -> list[tuple[int, int]]:
        pairs: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        for assignment in self.assignments:
            assigned = {
                int(atom_idx)
                for atom_idx in assignment.sample_atom_indices.detach().cpu().tolist()
            }
            if not assigned:
                continue
            for group in self.candidate_groups:
                group_atoms = [int(atom_idx) for atom_idx in group.tolist()]
                positions = [
                    idx
                    for idx, atom_idx in enumerate(group_atoms)
                    if atom_idx in assigned
                ]
                if len(positions) != len(assigned):
                    continue
                first_pos = min(positions)
                last_pos = max(positions)
                if positions != list(range(first_pos, last_pos + 1)):
                    continue
                first_ca = group_atoms[first_pos]
                last_ca = group_atoms[last_pos]
                if first_pos > 0:
                    pair = (group_atoms[first_pos - 1], first_ca)
                    if pair not in seen:
                        pairs.append(pair)
                        seen.add(pair)
                if last_pos + 1 < len(group_atoms):
                    pair = (group_atoms[last_pos + 1], last_ca)
                    if pair not in seen:
                        pairs.append(pair)
                        seen.add(pair)
                break
        return pairs

    def _apply_bias(self, xyz: torch.Tensor, step_idx: int) -> torch.Tensor:
        with torch.enable_grad():
            x = xyz.detach().clone().requires_grad_(True)
            losses = []
            for assignment in self.assignments:
                atom_idx = (
                    assignment.projection_sample_atom_indices
                    if assignment.projection_sample_atom_indices is not None
                    else assignment.sample_atom_indices
                )
                reference_xyz = (
                    assignment.projection_reference_xyz
                    if assignment.projection_reference_xyz is not None
                    else assignment.reference_xyz
                )
                atom_idx = atom_idx.to(
                    device=x.device, dtype=torch.long
                )
                current = x.index_select(dim=-2, index=atom_idx)
                reference = reference_xyz.to(
                    device=x.device, dtype=x.dtype
                )
                reference = reference.unsqueeze(0).expand(current.shape[0], -1, -1)
                aligned = kabsch_align_all_atom(reference, current).detach()
                losses.append((current - aligned).pow(2).sum(dim=-1).mean())
            if not losses:
                return xyz
            loss = torch.stack(losses).mean()
            objective = -self.config.loss_weight * loss
            grad = torch.autograd.grad(objective, x, retain_graph=False)[0]
            unclipped_grad_rms = torch.sqrt(grad.pow(2).sum(dim=-1).mean()).detach()
            grad = _clip_rms(grad, self.config.bias_clip_rms)
            clipped_grad_rms = torch.sqrt(grad.pow(2).sum(dim=-1).mean()).detach()
        if self._should_debug(step_idx):
            logger.info(
                "[motif_unindexing] step=%s bias_mse=%.6f grad_rms=%.6f "
                "unclipped_grad_rms=%.6f",
                step_idx,
                float(loss.detach().item()),
                float(clipped_grad_rms.item()),
                float(unclipped_grad_rms.item()),
            )
        return (xyz + grad).detach()

    def _should_update_assignments(self, step_idx: int) -> bool:
        if self.last_update_step is None:
            return True
        return (
            step_idx - self.last_update_step
        ) >= self.config.update_frequency

    def _should_debug(self, step_idx: int) -> bool:
        return (
            self.config.debug
            and self.config.debug_frequency > 0
            and step_idx % self.config.debug_frequency == 0
        )

    def _debug_log(self, step_idx: int, message: str) -> None:
        if self._should_debug(step_idx):
            logger.info("[motif_unindexing] step=%s %s", step_idx, message)

    def _lock_assignments(self, step_idx: int) -> None:
        self.locked_assignment_windows = validate_contiguous_assignment_windows(
            self.assignments,
            self.atom_to_token_map,
        )
        self.assignments_locked = True
        self.assignment_locked_step = step_idx
        if self.config.debug:
            windows = [
                (
                    window.motif_index,
                    window.token_start,
                    window.token_stop,
                    window.n_tokens,
                )
                for window in self.locked_assignment_windows
            ]
            logger.info(
                "[motif_unindexing] step=%s assignments_locked windows=%s",
                step_idx,
                windows,
            )

    def _debug_log_assignments(self, step_idx: int) -> None:
        if not self._should_debug(step_idx):
            return
        logger.info(
            "[motif_unindexing] step=%s assignment_update candidate_counts=%s "
            "best_raw_rmsds=%s assigned=%d/%d",
            step_idx,
            self._last_candidate_counts,
            [_fmt(v) for v in self._last_best_raw_rmsds],
            len(self.assignments),
            len(self.motifs),
        )
        for assignment in self.assignments:
            motif = self.motifs[assignment.motif_index]
            atom_idx = assignment.sample_atom_indices.detach().cpu()
            logger.info(
                "[motif_unindexing] step=%s motif=%s index=%d n_ca=%d "
                "assigned_atom_range=%d-%d rmsd=%.4f",
                step_idx,
                motif.name or motif.path,
                assignment.motif_index,
                motif.n_residues,
                int(atom_idx[0]),
                int(atom_idx[-1]),
                assignment.rmsd,
            )
            proj_atoms = (
                0
                if assignment.projection_sample_atom_indices is None
                else int(assignment.projection_sample_atom_indices.numel())
            )
            logger.info(
                "[motif_unindexing] step=%s motif=%s projection_atoms=%d",
                step_idx,
                motif.name or motif.path,
                proj_atoms,
            )

    def _projection_reference_for_ca_indices(
        self,
        motif: ExternalMotifReference,
        ca_atom_indices: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if not motif.residue_atoms or self.sample_atom_names is None:
            return ca_atom_indices.detach().cpu().long(), motif.ca_xyz.detach().cpu().float()

        ca_idx = ca_atom_indices.detach().cpu().long()
        token_ids = self.atom_to_token_map[ca_idx].tolist()
        sample_indices: list[int] = []
        reference_xyz: list[torch.Tensor] = []
        wanted = tuple(self.config.projection_atom_names)

        for residue_i, token_id in enumerate(token_ids):
            if residue_i >= len(motif.residue_atoms):
                break
            ref_atoms = motif.residue_atoms[residue_i]
            token_atom_idx = torch.where(self.atom_to_token_map == int(token_id))[0]
            sample_by_name = {
                str(self.sample_atom_names[int(idx)]): int(idx)
                for idx in token_atom_idx.tolist()
            }
            for atom_name in wanted:
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

    def _aligned_assignment_reference(
        self,
        assignment: MotifAssignment,
        xyz: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        atom_idx = (
            assignment.projection_sample_atom_indices
            if assignment.projection_sample_atom_indices is not None
            else assignment.sample_atom_indices
        )
        reference_xyz = (
            assignment.projection_reference_xyz
            if assignment.projection_reference_xyz is not None
            else assignment.reference_xyz
        )
        atom_idx = atom_idx.detach().cpu().long()
        current = xyz.index_select(dim=-2, index=atom_idx.to(xyz.device))
        reference = reference_xyz.to(device=xyz.device, dtype=xyz.dtype)
        reference = reference.unsqueeze(0).expand(current.shape[0], -1, -1)
        aligned = kabsch_align_all_atom(reference, current).detach()
        if aligned.ndim == 3:
            aligned = aligned[0]
        return atom_idx, aligned

    @staticmethod
    def _feature_device(f: dict[str, Any]) -> torch.device:
        for value in f.values():
            if torch.is_tensor(value):
                return value.device
        return torch.device("cpu")


def build_motif_unindexing_controller(
    config: dict | MotifUnindexingConfig | None,
    f: dict[str, Any],
    sample_features: dict[str, Any] | None = None,
) -> MotifUnindexingController | None:
    cfg = _coerce_config(config)
    if not cfg.enabled:
        return None
    if cfg.update_frequency < 1:
        raise ValueError("motif_unindexing.update_frequency must be >= 1")
    if cfg.loss_weight < 0.0:
        raise ValueError("motif_unindexing.loss_weight must be non-negative")
    if cfg.bias_clip_rms is not None and cfg.bias_clip_rms <= 0.0:
        raise ValueError("motif_unindexing.bias_clip_rms must be positive or null")
    if cfg.boundary_distance_bias_weight < 0.0:
        raise ValueError(
            "motif_unindexing.boundary_distance_bias_weight must be non-negative"
        )
    if cfg.boundary_distance_target <= 0.0:
        raise ValueError("motif_unindexing.boundary_distance_target must be positive")
    if (
        cfg.boundary_distance_bias_clip_rms is not None
        and cfg.boundary_distance_bias_clip_rms <= 0.0
    ):
        raise ValueError(
            "motif_unindexing.boundary_distance_bias_clip_rms must be positive or null"
        )
    if cfg.activation_threshold < 0.0:
        raise ValueError("motif_unindexing.activation_threshold must be non-negative")
    if cfg.post_activation_guidance_steps < 0:
        raise ValueError(
            "motif_unindexing.post_activation_guidance_steps must be non-negative"
        )
    if cfg.post_activation_stop_after is not None and cfg.post_activation_stop_after < 0:
        raise ValueError(
            "motif_unindexing.post_activation_stop_after must be non-negative or null"
        )
    if cfg.assignment_lock_step is not None and cfg.assignment_lock_step < 0:
        raise ValueError(
            "motif_unindexing.assignment_lock_step must be non-negative or null"
        )
    if cfg.bias_stop_after is not None and cfg.bias_stop_after < 0:
        raise ValueError("motif_unindexing.bias_stop_after must be non-negative or null")
    if cfg.debug_frequency < 1:
        raise ValueError("motif_unindexing.debug_frequency must be >= 1")
    if not (0.0 <= cfg.soft_kabsch_alpha_min <= 1.0):
        raise ValueError(
            "motif_unindexing.soft_kabsch_alpha_min must be in [0.0, 1.0], "
            f"got {cfg.soft_kabsch_alpha_min}"
        )
    if cfg.soft_kabsch_ramp_start_step is not None and cfg.soft_kabsch_ramp_start_step < 0:
        raise ValueError(
            "motif_unindexing.soft_kabsch_ramp_start_step must be non-negative if set"
        )
    controller = MotifUnindexingController(cfg, f, sample_features=sample_features)
    return controller if controller.enabled() else None


def load_motif_references(
    config: MotifUnindexingConfig,
    sample_features: dict[str, Any] | None = None,
) -> list[ExternalMotifReference]:
    refs = [
        load_external_motif_reference(path) for path in config.motif_pdbs
    ]
    if config.use_unindexed_motifs:
        refs.extend(load_unindexed_motif_references_from_sample(sample_features))
    return refs


def load_external_motif_reference(path: str) -> ExternalMotifReference:
    residue_atoms = _load_pdb_residue_atoms(path)
    ca_xyz = _ca_xyz_from_residue_atoms(residue_atoms, source=str(path))
    if ca_xyz.shape[0] < 3:
        raise ValueError(
            f"motif_unindexing motif {path!r} must contain at least 3 CA atoms"
        )
    return ExternalMotifReference(
        path=path,
        ca_xyz=ca_xyz,
        residue_atoms=residue_atoms,
    )


def load_unindexed_motif_references_from_sample(
    sample_features: dict[str, Any] | None,
) -> list[ExternalMotifReference]:
    if sample_features is None:
        raise ValueError(
            "motif_unindexing.use_unindexed_motifs=True requires sample_features"
        )
    specification = sample_features.get("specification") or {}
    input_path = specification.get("input")
    motifs = specification.get("motifs") or {}
    unindexed_motifs = specification.get("unindexed_motifs") or []
    if not input_path:
        raise ValueError(
            "motif_unindexing.use_unindexed_motifs=True requires an input PDB in the JSON"
        )
    if not motifs or not unindexed_motifs:
        raise ValueError(
            "motif_unindexing.use_unindexed_motifs=True requires JSON motifs and unindexed_motifs"
        )

    from rfd3.inference.parsing import InputSelection
    from rfd3.utils.inference import inference_load_

    atom_array = inference_load_(input_path)["atom_array"]
    refs = []
    for motif_name in unindexed_motifs:
        if motif_name not in motifs:
            raise ValueError(
                f"motif_unindexing unindexed_motif {motif_name!r} is not in JSON motifs"
            )
        selection = InputSelection.from_any(motifs[motif_name], atom_array=atom_array)
        motif_atoms = atom_array[selection.get_mask()]
        residue_atoms = _residue_atoms_from_atom_array(motif_atoms)
        ca_xyz = _ca_xyz_from_residue_atoms(
            residue_atoms,
            source=f"{input_path}:{motif_name}",
        )
        if ca_xyz.shape[0] < 3:
            raise ValueError(
                f"motif_unindexing motif {motif_name!r} from {input_path!r} "
                "must contain at least 3 selected CA atoms"
            )
        refs.append(
            ExternalMotifReference(
                path=str(input_path),
                ca_xyz=ca_xyz,
                residue_atoms=residue_atoms,
                name=str(motif_name),
            )
        )
    return refs


def _load_pdb_residue_atoms(path: str) -> list[dict[str, torch.Tensor]]:
    residues: list[dict[str, torch.Tensor]] = []
    current_key = None
    current_atoms: dict[str, torch.Tensor] = {}
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            atom_name = line[12:16].strip()
            chain_id = line[21].strip()
            res_id = line[22:26].strip()
            ins_code = line[26].strip()
            key = (chain_id, res_id, ins_code)
            if current_key is not None and key != current_key:
                residues.append(current_atoms)
                current_atoms = {}
            current_key = key
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue
            current_atoms[atom_name] = torch.tensor((x, y, z), dtype=torch.float32)
    if current_key is not None:
        residues.append(current_atoms)
    if not residues:
        raise ValueError(f"motif_unindexing motif {path!r} contains no atoms")
    return residues


def _residue_atoms_from_atom_array(atom_array) -> list[dict[str, torch.Tensor]]:
    residues: list[dict[str, torch.Tensor]] = []
    current_key = None
    current_atoms: dict[str, torch.Tensor] = {}
    chain_ids = getattr(atom_array, "chain_id")
    res_ids = getattr(atom_array, "res_id")
    atom_names = getattr(atom_array, "atom_name")
    coords = getattr(atom_array, "coord")
    for idx in range(len(atom_array)):
        key = (str(chain_ids[idx]), int(res_ids[idx]))
        if current_key is not None and key != current_key:
            residues.append(current_atoms)
            current_atoms = {}
        current_key = key
        current_atoms[str(atom_names[idx])] = torch.tensor(
            coords[idx], dtype=torch.float32
        )
    if current_key is not None:
        residues.append(current_atoms)
    return residues


def _ca_xyz_from_residue_atoms(
    residue_atoms: list[dict[str, torch.Tensor]],
    *,
    source: str,
) -> torch.Tensor:
    coords = [atoms["CA"] for atoms in residue_atoms if "CA" in atoms]
    if not coords:
        raise ValueError(f"motif_unindexing motif {source!r} contains no CA atoms")
    return torch.stack(coords).float()


def _ordered_ca_atom_indices(f: dict[str, Any]) -> torch.Tensor:
    is_ca = torch.as_tensor(f["is_ca"], dtype=torch.bool).detach().cpu()
    atom_to_token_map = torch.as_tensor(
        f["atom_to_token_map"], dtype=torch.long
    ).detach().cpu()
    atom_idx = torch.where(is_ca)[0].long()
    order = torch.argsort(atom_to_token_map[atom_idx], stable=True)
    return atom_idx[order]


def _sample_atom_names(sample_features: dict[str, Any] | None) -> list[str] | None:
    if not sample_features:
        return None
    atom_array = sample_features.get("atom_array")
    if atom_array is None or not hasattr(atom_array, "atom_name"):
        return None
    return [str(name) for name in atom_array.atom_name]


def _candidate_ca_groups(
    f: dict[str, Any],
    ca_atom_indices: torch.Tensor,
) -> list[torch.Tensor]:
    if ca_atom_indices.numel() == 0:
        return []
    atom_to_token_map = torch.as_tensor(
        f["atom_to_token_map"], dtype=torch.long
    ).detach().cpu()
    sym_transform = f.get("sym_transform_id")
    sym_transform = (
        torch.as_tensor(sym_transform, dtype=torch.long).detach().cpu()
        if sym_transform is not None
        else None
    )

    groups: list[list[int]] = []
    current: list[int] = []
    prev_token = None
    prev_transform = None
    for atom_idx in ca_atom_indices.tolist():
        token = int(atom_to_token_map[atom_idx].item())
        transform = (
            int(sym_transform[atom_idx].item()) if sym_transform is not None else None
        )
        starts_new = (
            prev_token is not None
            and (token != prev_token + 1 or transform != prev_transform)
        )
        if starts_new and current:
            groups.append(current)
            current = []
        current.append(int(atom_idx))
        prev_token = token
        prev_transform = transform
    if current:
        groups.append(current)
    return [torch.tensor(group, dtype=torch.long) for group in groups]


def _rmsd(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.sqrt((a - b).pow(2).sum(dim=-1).mean(dim=-1).clamp_min(1e-12))


def _fmt(value: float | None) -> str:
    return "none" if value is None else f"{float(value):.4f}"


def _clip_rms(grad: torch.Tensor, clip_rms: float | None) -> torch.Tensor:
    if clip_rms is None:
        return grad
    grad_rms = torch.sqrt(grad.pow(2).sum(dim=-1).mean()).clamp_min(1e-12)
    scale = min(1.0, float(clip_rms) / float(grad_rms.detach().item()))
    return grad * scale


def _or_bool_feature(
    current: Any,
    mask: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    mask = mask.to(device=device, dtype=torch.bool)
    if current is None:
        return mask.clone()
    return torch.as_tensor(current, device=device).bool().clone() | mask


def _promote_token_type(
    current: Any,
    token_mask: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    token_mask = token_mask.to(device=device, dtype=torch.bool)
    if current is None:
        token_type = torch.zeros(
            token_mask.shape[0],
            3,
            dtype=torch.float32,
            device=device,
        )
        token_type[:, 0] = 1.0
    else:
        token_type = torch.as_tensor(current, device=device).clone()
    token_type[token_mask] = torch.as_tensor(
        [0, 1, 0],
        dtype=token_type.dtype,
        device=device,
    )
    return token_type


def _with_reference_positions(
    current: Any,
    reference_pos: torch.Tensor,
    *,
    device: torch.device,
    fill_value: float = float("nan"),
) -> torch.Tensor:
    valid = torch.isfinite(reference_pos).all(dim=-1)
    valid_device = valid.to(device=device)
    if current is None:
        out = torch.full(
            reference_pos.shape,
            fill_value,
            dtype=reference_pos.dtype,
            device=device,
        )
    else:
        out = torch.as_tensor(current, device=device).clone()
    out[valid_device] = reference_pos.to(device=device, dtype=out.dtype)[valid_device]
    return out


def _coerce_config(config) -> MotifUnindexingConfig:
    if config is None:
        return MotifUnindexingConfig()
    if isinstance(config, MotifUnindexingConfig):
        return config
    try:
        from omegaconf import DictConfig, ListConfig, OmegaConf
    except ImportError:
        pass
    else:
        if isinstance(config, (DictConfig, ListConfig)):
            config = OmegaConf.to_container(config, resolve=True)
    if isinstance(config, dict):
        return MotifUnindexingConfig(**config)
    raise TypeError(
        "motif_unindexing must be a dict or MotifUnindexingConfig, "
        f"got {type(config)}"
    )

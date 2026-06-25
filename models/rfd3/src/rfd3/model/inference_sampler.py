import inspect
import logging
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Literal

import torch
from jaxtyping import Float
from rfd3.inference.symmetry.hetero_pseudo import (
    HeteroPseudoSymmetryConfig,
    _center_hetero_xyz,
    apply_hetero_pseudo_symmetry,
)
from rfd3.inference.symmetry.symmetry_utils import apply_symmetry_to_xyz_atomwise
from rfd3.model.cfg_utils import strip_X
from rfd3.model.floating_motif_projection import (
    project_floating_motifs_all_atom,
    remove_floating_motif_atoms_from_fixed_mask,
    should_project_floating_motifs,
)
from rfd3.model.motif_unindexing import build_motif_unindexing_controller
from rfd3.model.motif_backbone_bias import build_motif_backbone_bias_controller

from foundry.common import exists
from foundry.utils.alignment import weighted_rigid_align
from foundry.utils.ddp import RankedLogger
from foundry.utils.rotation_augmentation import (
    rot_vec_mul,
    uniform_random_rotation,
)

logging.basicConfig(level=logging.INFO)
ranked_logger = RankedLogger(__name__, rank_zero_only=True)


@dataclass(kw_only=True)
class SampleDiffusionConfig:
    kind: Literal["default", "symmetry", "hetero_symmetry"] = "default"

    # Standard EDM args
    num_timesteps: int = 200
    min_t: int = 0
    max_t: int = 1
    sigma_data: int = 16
    s_min: float = 4e-4
    s_max: int = 160
    p: int = 7
    gamma_0: float = 0.6
    gamma_min: float = 1.0
    noise_scale: float = 1.003
    step_scale: float = 1.5
    solver: Literal["af3"] = "af3"

    # RFD3 / design args
    center_option: str = "all"
    s_trans: float = 1.0
    s_jitter_origin: float = 0.0
    fraction_of_steps_to_fix_motif: float = 0.0
    skip_few_diffusion_steps: bool = False
    allow_realignment: bool = False
    insert_motif_at_end: bool = True
    use_classifier_free_guidance: bool = False
    cfg_scale: float = 2.0
    cfg_t_max: float | None = None

    # Inference-only approximation of independently floating contig motifs.
    floating_motif_project: bool = False
    floating_motif_project_every: int = 1
    floating_motif_burn_in: int = 0
    floating_motif_stop_after: int | None = None

    # Recycling
    n_recycle: int | None = None  # Override model default n_recycle for inference

    # External differentiable potentials (disabled by default; no overhead when empty)
    potentials: dict = field(default_factory=dict)

    # External-reference motif unindexing.  Disabled by default and independent
    # of the legacy unindexed-token input path.
    motif_unindexing: dict = field(default_factory=dict)

    # Backbone-bias-only motif guidance: soft gradient pull during diffusion,
    # single hard backbone paste after the loop.  No Kabsch per-step, no
    # token-type modification.
    motif_backbone_bias: dict = field(default_factory=dict)

    # Normal symmetry remains true homomeric symmetry.  By default it follows
    # the existing sym_step_frac schedule; this optional step cutoff is an
    # explicit inference-time stop for full monomer projection.
    full_symmetry_stop_after: int | None = None

    # Hetero pseudo-symmetry: disabled unless inference_sampler.kind is
    # "hetero_symmetry".  These top-level fields keep compatibility with the
    # existing sampler config filtering.
    hetero_post_init_symmetry: Literal[
        "interface_only", "initialization_only"
    ] = "interface_only"
    hetero_projection_enabled: bool = True
    hetero_projection_hard: bool = False
    hetero_projection_weight: float = 1.0
    hetero_projection_start_step: int = 0
    hetero_projection_stop_after: int | None = None
    hetero_projection_schedule: Literal["constant", "linear_decay"] = "constant"
    hetero_interface_distance_cutoff: float = 8.0
    hetero_interface_sequence_buffer: int = 2
    hetero_interface_include_sidechains: bool = True
    hetero_motif_contact_exclusion_enabled: bool = True
    hetero_motif_contact_distance_cutoff: float = 8.0
    hetero_motif_contact_sequence_buffer: int = 1
    hetero_support_enabled: bool = True
    hetero_support_distance_cutoff: float = 12.0
    hetero_support_weight: float = 0.3
    hetero_support_sequence_buffer: int = 2
    hetero_recenter_enabled: bool = True
    hetero_motif_follow_scaffold_frame: bool = True
    hetero_debug: bool = False
    hetero_diagnostics_interval: int = 0
    hetero_require_per_copy_floating_motifs: bool = True
    hetero_init_floating_motifs_from_reference: bool = True


class SampleDiffusionWithMotif(SampleDiffusionConfig):
    """Diffusion sampler that supports optional motif alignment."""

    def _construct_inference_noise_schedule(
        self, device: torch.device, partial_t: float = None
    ) -> torch.Tensor:
        """Constructs a noise schedule for use during inference.

        The inference noise schedule is defined in the AF-3 supplement as:

            t_hat = sigma_data * (s_max**(1/p) + t * (s_min**(1/p) - s_max**(1/p)))**p

        Returns:
            torch.Tensor: A tensor representing the noise schedule `t_hat`.

        Reference:
            AlphaFold 3 Supplement, Section 3.7.1.
        """
        # Create a linearly spaced tensor of timesteps between min_t and max_t
        t = torch.linspace(self.min_t, self.max_t, self.num_timesteps, device=device)

        # Construct the noise schedule, using the formula provided in the reference
        t_hat = (
            self.sigma_data
            * (
                (self.s_max) ** (1 / self.p)
                + t * (self.s_min ** (1 / self.p) - self.s_max ** (1 / self.p))
            )
            ** self.p
        )

        if partial_t is not None:
            # For now, partial t is a global parameter
            partial_t = float(partial_t.mean())
            noise_schedule = t_hat
            ranked_logger.info("Using partial diffusion with t={}".format(partial_t))

            # Debug the noise schedule filtering
            original_schedule_len = len(noise_schedule)
            original_max = noise_schedule.max().item()
            original_min = noise_schedule.min().item()

            noise_schedule = noise_schedule[noise_schedule <= partial_t]

            new_schedule_len = len(noise_schedule)
            if new_schedule_len > 0:
                new_max = noise_schedule.max().item()
                new_min = noise_schedule.min().item()
                ranked_logger.info(
                    f"Noise schedule: {original_schedule_len} → {new_schedule_len} steps"
                )
                ranked_logger.info(
                    f"Original range: [{original_min:.3f}, {original_max:.3f}]"
                )
                ranked_logger.info(f"Filtered range: [{new_min:.3f}, {new_max:.3f}]")
            else:
                ranked_logger.warning(
                    f"No noise schedule steps found with t <= {partial_t}!"
                )
                ranked_logger.info(
                    f"Original schedule range: [{original_min:.3f}, {original_max:.3f}]"
                )
                # Fallback to smallest available step
                noise_schedule_original = self._construct_inference_noise_schedule(
                    device=device
                )
                noise_schedule = noise_schedule_original[-1:]  # Just use the final step
                ranked_logger.info(
                    f"Using fallback: final step with t={noise_schedule[0].item():.6f}"
                )
        else:
            noise_schedule = t_hat

        return noise_schedule

    def _get_initial_structure(
        self,
        c0: torch.Tensor,
        D: int,
        L: int,
        coord_atom_lvl_to_be_noised: torch.Tensor,
        is_motif_atom_with_fixed_coord,
    ) -> torch.Tensor:
        noise = c0 * torch.normal(mean=0.0, std=1.0, size=(D, L, 3), device=c0.device)
        noise[..., is_motif_atom_with_fixed_coord, :] = 0  # Zero out noise going in
        X_L = noise + coord_atom_lvl_to_be_noised
        return X_L

    def sample_diffusion_like_af3(
        self,
        *,
        f: dict[str, Any],
        diffusion_module: torch.nn.Module,
        diffusion_batch_size: int,
        coord_atom_lvl_to_be_noised: Float[torch.Tensor, "D L 3"],
        initializer_outputs,
        initializer_fn=None,
        ref_initializer_outputs: dict[str, Any] | None,
        f_ref: dict[str, Any] | None,
        floating_motif_refs=None,
        sample_features: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        # Motif setup to recenter the motif at every step
        is_motif_atom_with_fixed_coord = f["is_motif_atom_with_fixed_coord"]
        fixed_coord_noise_mask = (
            remove_floating_motif_atoms_from_fixed_mask(
                is_motif_atom_with_fixed_coord, floating_motif_refs
            )
            if self.floating_motif_project
            else is_motif_atom_with_fixed_coord
        )
        f_diffusion = (
            _with_floating_motifs_unfixed_for_diffusion(f, floating_motif_refs)
            if self.floating_motif_project
            else f
        )

        # Book-keeping
        noise_schedule = self._construct_inference_noise_schedule(
            device=coord_atom_lvl_to_be_noised.device,
            partial_t=f.get("partial_t", None),
        )

        L = f["ref_element"].shape[0]
        D = diffusion_batch_size

        X_L = self._get_initial_structure(
            c0=noise_schedule[0],
            D=D,
            L=L,
            coord_atom_lvl_to_be_noised=coord_atom_lvl_to_be_noised.clone(),
            is_motif_atom_with_fixed_coord=fixed_coord_noise_mask,
        )  # (D, L, 3)

        motif_unindexing_controller = build_motif_unindexing_controller(
            self.motif_unindexing,
            f,
            sample_features=sample_features,
        )
        ranked_logger.info(
            "[motif_unindexing] sampler_config enabled=%s controller=%s",
            bool((self.motif_unindexing or {}).get("enabled", False)),
            "built" if motif_unindexing_controller is not None else "disabled",
        )
        promoted_initializer_outputs = None

        motif_backbone_bias_controller = build_motif_backbone_bias_controller(
            self.motif_backbone_bias, f, sample_features=sample_features
        )

        # Build the potential adapter once (masks are static across steps)
        potential_adapter = None
        if self.potentials:
            from rfd3.potentials.integration import build_potential_adapter

            potential_adapter = build_potential_adapter(
                self.potentials,
                f,
                input_pos=coord_atom_lvl_to_be_noised,
                floating_motif_refs=floating_motif_refs,
            )
            if potential_adapter is not None:
                ranked_logger.info(
                    f"[potentials] enabled - mode={potential_adapter.config.apply_mode}, "
                    f"n_potentials={len(potential_adapter.manager.potentials)}"
                )

        if self.s_jitter_origin > 0.0:
            X_L[:, fixed_coord_noise_mask, :] += torch.normal(
                mean=0.0,
                std=self.s_jitter_origin,
                size=(D, 1, 3),
                device=X_L.device,
            )

        X_noisy_L_traj = []
        X_denoised_L_traj = []
        sequence_entropy_traj = []
        t_hats = []

        threshold_step = (len(noise_schedule) - 1) * self.fraction_of_steps_to_fix_motif

        for step_num, (c_t_minus_1, c_t) in enumerate(
            zip(noise_schedule, noise_schedule[1:])
        ):
            # Assert no grads on X_L
            assert not torch.is_grad_enabled(), "Computation graph should not be active"
            assert not X_L.requires_grad, "X_L should not require gradients"

            # Apply a random rotation and translation to the structure
            if self.allow_realignment:
                X_L, _ = centre_random_augment_around_motif(
                    X_L,
                    coord_atom_lvl_to_be_noised,
                    is_motif_atom_with_fixed_coord,
                    center_option=self.center_option,
                    # If centering_affects_motif is True, the model's predictions from (step_num-1) might affect the motif
                    centering_affects_motif=(max(step_num - 1, 0)) >= threshold_step,
                    # If keeping the motif position wrt the origin fixed, we can't do translational augmentation
                    # We want to keep this position fixed in the interval where the model is not allowed to change it
                    s_trans=self.s_trans if step_num >= threshold_step else 0.0,
                )

            # Update gamma & step scale
            gamma = self.gamma_0 if c_t > self.gamma_min else 0
            step_scale = self.step_scale

            # Compute the value of t_hat
            t_hat = c_t_minus_1 * (gamma + 1)

            f_step = f_diffusion
            initializer_outputs_step = initializer_outputs
            if motif_unindexing_controller is not None:
                f_step = motif_unindexing_controller.promoted_feature_dict(
                    f_diffusion,
                    X_L.detach(),
                    step_num,
                )
                cache_rebuild_reason = motif_unindexing_controller.static_cache_rebuild_reason(step_num)
                if (
                    motif_unindexing_controller.config.rebuild_static_cache_at_projection_stop
                    and 158 <= step_num <= 161
                ):
                    ranked_logger.info(
                        "[motif_unindexing] step=%s projection_stop_cache_trace reason=%s promoted=%s initializer_fn=%s cache_built=%s activated_step=%s post_activation_stop_after=%s",
                        step_num,
                        cache_rebuild_reason,
                        f_step is not f_diffusion,
                        initializer_fn is not None,
                        promoted_initializer_outputs is not None,
                        motif_unindexing_controller.activated_step,
                        motif_unindexing_controller.config.post_activation_stop_after,
                    )
                if cache_rebuild_reason is not None and promoted_initializer_outputs is None:
                    ranked_logger.info(
                        "[motif_unindexing] step=%s static_cache_rebuild_requested reason=%s promoted=%s initializer_fn=%s activated_step=%s post_activation_stop_after=%s",
                        step_num,
                        cache_rebuild_reason,
                        f_step is not f_diffusion,
                        initializer_fn is not None,
                        motif_unindexing_controller.activated_step,
                        motif_unindexing_controller.config.post_activation_stop_after,
                    )
                if f_step is not f_diffusion and initializer_fn is not None:
                    # Rebuild the static pairwise cache (C_L / _sl_cached / _sm_cached)
                    # once for one of the explicit cache-promotion modes.
                    #
                    # The immediate-activation experiment must use f_step, not
                    # feature_dict_for_initializer: f_step contains Kabsch-aligned
                    # ref_pos in the same frame used by runtime conditioning.
                    if promoted_initializer_outputs is None:
                        if cache_rebuild_reason is not None:
                            ranked_logger.info(
                                "[motif_unindexing] step=%s rebuilding_static_cache_from_promoted_step reason=%s activated_step=%s post_activation_stop_after=%s",
                                step_num,
                                cache_rebuild_reason,
                                motif_unindexing_controller.activated_step,
                                motif_unindexing_controller.config.post_activation_stop_after,
                            )
                            if cache_rebuild_reason == "assignment_lock_motif_pos_only":
                                f_init = motif_unindexing_controller.feature_dict_for_motif_pos_only_initializer(
                                    f_diffusion
                                )
                                promoted_initializer_outputs = initializer_fn(
                                    f_init if f_init is not None else f_step
                                )
                            elif cache_rebuild_reason == "assignment_lock_mask_only":
                                f_init = motif_unindexing_controller.feature_dict_for_mask_only_initializer(
                                    f_diffusion
                                )
                                promoted_initializer_outputs = initializer_fn(
                                    f_init if f_init is not None else f_step
                                )
                            else:
                                promoted_initializer_outputs = initializer_fn(f_step)
                        elif (
                            motif_unindexing_controller.config.promote_to_motif_on_activation
                            and motif_unindexing_controller.activated_step is not None
                        ):
                            f_init = motif_unindexing_controller.feature_dict_for_initializer(
                                f_diffusion
                            )
                            if f_init is not None:
                                promoted_initializer_outputs = initializer_fn(f_init)
                            else:
                                promoted_initializer_outputs = initializer_fn(f_step)
                    if promoted_initializer_outputs is not None:
                        initializer_outputs_step = promoted_initializer_outputs

            # Noise the coordinates with scaled Gaussian noise
            epsilon_L = (
                self.noise_scale
                * torch.sqrt(torch.square(t_hat) - torch.square(c_t_minus_1))
                * torch.normal(mean=0.0, std=1.0, size=X_L.shape, device=X_L.device)
            )
            epsilon_L[..., fixed_coord_noise_mask, :] = (
                0  # No noise injection for fixed atoms
            )
            X_noisy_L = X_L + epsilon_L

            # Denoise the coordinates
            # Handle chunked mode vs standard mode
            if "chunked_pairwise_embedder" in initializer_outputs_step:
                # Chunked mode: explicitly provide P_LL=None
                tic = time.time()
                chunked_embedder = initializer_outputs_step[
                    "chunked_pairwise_embedder"
                ]  # Don't pop, just get
                other_outputs = {
                    k: v
                    for k, v in initializer_outputs_step.items()
                    if k != "chunked_pairwise_embedder"
                }
                outs = diffusion_module(
                    X_noisy_L=X_noisy_L,
                    t=t_hat.tile(D),
                    f=f_step,
                    P_LL=None,  # Not used in chunked mode
                    chunked_pairwise_embedder=chunked_embedder,
                    initializer_outputs=other_outputs,
                    n_recycle=self.n_recycle,
                    **other_outputs,
                )
                toc = time.time()
                ranked_logger.info(
                    f"[chunked] step {step_num}: {(toc - tic)*1000:.1f} ms"
                )
            else:
                # Standard mode: P_LL is included in initializer_outputs
                outs = diffusion_module(
                    X_noisy_L=X_noisy_L,
                    t=t_hat.tile(D),
                    f=f_step,
                    n_recycle=self.n_recycle,
                    **initializer_outputs_step,
                )

            X_denoised_L = outs["X_L"] if "X_L" in outs else outs

            # Compute the delta between the noisy and denoised coordinates, scaled by t_hat
            delta_L = (
                X_noisy_L - X_denoised_L
            ) / t_hat  # gradient of x wrt. t at x_t_hat
            d_t = c_t - t_hat

            if self.use_classifier_free_guidance and (
                self.cfg_t_max is None or c_t > self.cfg_t_max
            ):
                X_noisy_L_stripped = strip_X(X_noisy_L, f_ref)

                # unconditional forward pass
                outs_ref = diffusion_module(
                    X_noisy_L=X_noisy_L_stripped,  # modify X
                    t=t_hat.tile(D),
                    f=f_ref,  # modified f
                    n_recycle=self.n_recycle,
                    **ref_initializer_outputs,
                )

                X_denoised_L_stripped = outs_ref["X_L"]

                delta_L_ref = (
                    X_noisy_L_stripped - X_denoised_L_stripped
                ) / t_hat  # gradient of x wrt. t at x_t_hat

                # pad delta_L_ref with zeros to match delta_L (for the unindexed atoms)
                if delta_L_ref.shape[1] < delta_L.shape[1]:
                    delta_L_ref = torch.cat(
                        [
                            delta_L_ref,
                            torch.zeros_like(delta_L[:, delta_L_ref.shape[1] :, :]),
                        ],
                        dim=1,
                    )

                # apply CFG
                delta_L = delta_L + (self.cfg_scale - 1) * (delta_L - delta_L_ref)

            if exists(outs.get("sequence_logits_I")):
                # Compute confidence
                p = torch.softmax(
                    outs["sequence_logits_I"], dim=-1
                ).cpu()  # shape (D, L, 32)
                seq_entropy = -torch.sum(
                    p * torch.log(p + 1e-10), dim=-1
                )  # shape (D, L,)
                sequence_entropy_traj.append(seq_entropy)

            # Update the coordinates, scaled by the step size
            X_L = X_noisy_L + step_scale * d_t * delta_L

            # potential guidance hook
            # Applied after the normal sampler step, before X_L is stored.
            # torch.enable_grad() is used internally; we exit before the next
            # iteration so the grad-disabled assertions at the top of the loop
            # still pass.  The returned X_L is always detached.
            if potential_adapter is not None:
                X_L = potential_adapter.apply(
                    X_L,
                    t=t_hat,
                    T=float(noise_schedule[0]),
                    step_idx=step_num,
                )
            if motif_unindexing_controller is not None:
                X_L = motif_unindexing_controller.apply_pre_activation_bias(
                    X_L,
                    step_num,
                )
            if should_project_floating_motifs(
                step_num,
                enabled=self.floating_motif_project,
                project_every=self.floating_motif_project_every,
                burn_in=self.floating_motif_burn_in,
                stop_after=self.floating_motif_stop_after,
            ):
                X_L = project_floating_motifs_all_atom(X_L, floating_motif_refs)
            if motif_unindexing_controller is not None:
                dynamic_refs = motif_unindexing_controller.active_floating_motif_refs(
                    step_num
                )
                # ── TEMPORARY DIAGNOSTIC ─────────────────────────────────────
                ranked_logger.info(
                    "[sampler_diag] step=%d activated_step=%s n_dynamic_refs=%d "
                    "is_post_active=%s alpha=%.4f",
                    step_num,
                    motif_unindexing_controller.activated_step,
                    len(dynamic_refs),
                    motif_unindexing_controller.is_post_activation_active(step_num),
                    motif_unindexing_controller.projection_alpha(step_num),
                )
                # ─────────────────────────────────────────────────────────────
                if dynamic_refs:
                    X_L = project_floating_motifs_all_atom(
                        X_L,
                        dynamic_refs,
                        alpha=motif_unindexing_controller.projection_alpha(step_num),
                    )
                X_L = motif_unindexing_controller.apply_boundary_distance_bias(
                    X_L,
                    step_num,
                )
                motif_unindexing_controller.log_diagnostics(X_L, step_num)
            if motif_backbone_bias_controller is not None:
                X_L = motif_backbone_bias_controller.apply_bias(X_L, step_num)

            # Append the results to the trajectory (for visualization of the diffusion process)
            X_noisy_L_scaled = (
                self.sigma_data * X_noisy_L / torch.sqrt(t_hat**2 + self.sigma_data**2)
            )  # Save noisy traj as scaled inputs
            X_noisy_L_traj.append(X_noisy_L_scaled)
            X_denoised_L_traj.append(X_denoised_L)
            t_hats.append(t_hat)

        if motif_backbone_bias_controller is not None:
            X_L = motif_backbone_bias_controller.postprocess_X_L(X_L)

        if torch.any(is_motif_atom_with_fixed_coord) and self.allow_realignment:
            # Insert the gt motif at the end
            X_L, _ = centre_random_augment_around_motif(
                X_L,
                coord_atom_lvl_to_be_noised,
                is_motif_atom_with_fixed_coord,
                reinsert_motif=self.insert_motif_at_end,
            )

            # Align prediction to original motif
            X_L = weighted_rigid_align(
                coord_atom_lvl_to_be_noised,
                X_L,
                X_exists_L=is_motif_atom_with_fixed_coord,
            )

        return dict(
            X_L=X_L,  # (D, L, 3)
            X_noisy_L_traj=X_noisy_L_traj,  # list[Tensor[D, L, 3]]
            X_denoised_L_traj=X_denoised_L_traj,  # list[Tensor[D, L, 3]]
            t_hats=t_hats,  # list[Tensor[D]], where D is shared across all diffusion batches
            sequence_logits_I=outs.get("sequence_logits_I"),  # (D, I, 32)
            sequence_indices_I=outs.get("sequence_indices_I"),  # (D, I, 32)
            sequence_entropy_traj=sequence_entropy_traj,  # list[Tensor[D, I]]
        )


class SampleDiffusionWithSymmetry(SampleDiffusionWithMotif):
    """
    This class is a wrapper around the SampleDiffusionWithMotif class.
    It is used to sample diffusion with symmetry.
    """

    def __init__(self, sym_step_frac: float = 0.9, **kwargs):
        assert (
            kwargs.get("gamma_0") > 0.5
        ), "gamma_0 must be greater than 0.5 for symmetry sampling"
        self.sym_step_frac = sym_step_frac
        super().__init__(**kwargs)

    def post_step_hook(self, X_L: torch.Tensor, f: dict) -> torch.Tensor:
        """Called after every diffusion step (after Kabsch paste). No-op by default."""
        return X_L

    def log_step_diagnostics(
        self,
        step_num: int,
        stage: str,
        X_L: torch.Tensor,
        f: dict,
    ) -> None:
        del step_num, stage, X_L, f

    def apply_symmetry_to_X_L(self, X_L, f):
        # check that we are doing symmetric inference

        assert "sym_transform" in f.keys(), "Symmetry transform not found in f"

        # update symmetric frames to correct for change in global frame
        symmetry_feats = {k: v for k, v in f.items() if "sym" in k}

        # apply symmetry frame shift to X_L
        X_L = apply_symmetry_to_xyz_atomwise(
            X_L, symmetry_feats, partial_diffusion=("partial_t" in f)
        )

        return X_L

    def should_apply_full_symmetry(self, step_num, c_t, gamma_min_sym):
        if (
            self.full_symmetry_stop_after is not None
            and step_num > self.full_symmetry_stop_after
        ):
            return False
        return c_t > gamma_min_sym

    def apply_post_denoise_symmetry(self, outs, f, step_num, c_t, gamma_min_sym):
        # Preserve the existing homomeric behavior: denoised coordinates are
        # projected before the EDM update, and a second projection below keeps
        # post-guidance coordinates homomeric before Kabsch motif paste.
        if "X_L" in outs and self.should_apply_full_symmetry(
            step_num, c_t, gamma_min_sym
        ):
            outs["X_L"] = self.apply_symmetry_to_X_L(outs["X_L"], f)
        return outs

    def apply_post_update_symmetry(self, X_L, f, step_num, c_t, gamma_min_sym):
        if self.should_apply_full_symmetry(step_num, c_t, gamma_min_sym):
            return self.apply_symmetry_to_X_L(X_L, f)
        return X_L

    def sample_diffusion_like_af3(
        self,
        *,
        f: dict[str, Any],
        diffusion_module: torch.nn.Module,
        diffusion_batch_size: int,
        coord_atom_lvl_to_be_noised: Float[torch.Tensor, "D L 3"],
        initializer_outputs,
        initializer_fn=None,
        ref_initializer_outputs: dict[str, Any] | None,
        f_ref: dict[str, Any] | None,
        floating_motif_refs=None,
        sample_features: dict[str, Any] | None = None,
        **_,
    ) -> dict[str, Any]:
        # Motif setup to recenter the motif at every step
        is_motif_atom_with_fixed_coord = f["is_motif_atom_with_fixed_coord"]
        fixed_coord_noise_mask = (
            remove_floating_motif_atoms_from_fixed_mask(
                is_motif_atom_with_fixed_coord, floating_motif_refs
            )
            if self.floating_motif_project
            else is_motif_atom_with_fixed_coord
        )
        f_diffusion = (
            _with_floating_motifs_unfixed_for_diffusion(f, floating_motif_refs)
            if self.floating_motif_project
            else f
        )
        # Book-keeping
        noise_schedule = self._construct_inference_noise_schedule(
            device=coord_atom_lvl_to_be_noised.device,
            partial_t=f.get("partial_t", None),
        )

        L = f["ref_element"].shape[0]
        D = diffusion_batch_size
        X_L = self._get_initial_structure(
            c0=noise_schedule[0],
            D=D,
            L=L,
            coord_atom_lvl_to_be_noised=coord_atom_lvl_to_be_noised.clone(),
            is_motif_atom_with_fixed_coord=fixed_coord_noise_mask,
        )  # (D, L, 3)

        motif_unindexing_controller = build_motif_unindexing_controller(
            self.motif_unindexing,
            f,
            sample_features=sample_features,
        )
        ranked_logger.info(
            "[motif_unindexing] sampler_config enabled=%s controller=%s",
            bool((self.motif_unindexing or {}).get("enabled", False)),
            "built" if motif_unindexing_controller is not None else "disabled",
        )
        promoted_initializer_outputs = None

        motif_backbone_bias_controller = build_motif_backbone_bias_controller(
            self.motif_backbone_bias, f, sample_features=sample_features
        )

        # Build the potential adapter once (masks are static across steps)
        potential_adapter = None
        if self.potentials:
            from rfd3.potentials.integration import build_potential_adapter

            potential_adapter = build_potential_adapter(
                self.potentials,
                f,
                input_pos=coord_atom_lvl_to_be_noised,
                floating_motif_refs=floating_motif_refs,
            )
            if potential_adapter is not None:
                ranked_logger.info(
                    f"[potentials] enabled (symmetry sampler) - "
                    f"mode={potential_adapter.config.apply_mode}, "
                    f"n_potentials={len(potential_adapter.manager.potentials)}"
                )

        X_noisy_L_traj = []
        X_denoised_L_traj = []
        sequence_entropy_traj = []
        t_hats = []

        # symmetrize X_L until the step gamma = gamma_min_sym
        gamma_min_sym_idx = min(
            int(len(noise_schedule) * self.sym_step_frac), len(noise_schedule) - 1
        )
        gamma_min_sym = noise_schedule[gamma_min_sym_idx]

        ranked_logger.info(f"gamma_min_sym: {gamma_min_sym}")
        ranked_logger.info(f"gamma_min: {self.gamma_min}")
        for step_num, (c_t_minus_1, c_t) in enumerate(
            zip(noise_schedule, noise_schedule[1:])
        ):
            # Assert no grads on X_L
            assert not torch.is_grad_enabled(), "Computation graph should not be active"
            assert not X_L.requires_grad, "X_L should not require gradients"

            # Apply a random rotation and translation to the structure
            if self.allow_realignment:
                X_L, R = centre_random_augment_around_motif(
                    X_L,
                    coord_atom_lvl_to_be_noised,
                    is_motif_atom_with_fixed_coord,
                )

            # Update gamma & step scale
            gamma = self.gamma_0 if c_t > self.gamma_min else 0
            step_scale = self.step_scale

            # Compute the value of t_hat
            t_hat = c_t_minus_1 * (gamma + 1)

            f_step = f_diffusion
            initializer_outputs_step = initializer_outputs
            if motif_unindexing_controller is not None:
                f_step = motif_unindexing_controller.promoted_feature_dict(
                    f_diffusion,
                    X_L.detach(),
                    step_num,
                )
                cache_rebuild_reason = motif_unindexing_controller.static_cache_rebuild_reason(step_num)
                if (
                    motif_unindexing_controller.config.rebuild_static_cache_at_projection_stop
                    and 158 <= step_num <= 161
                ):
                    ranked_logger.info(
                        "[motif_unindexing] step=%s projection_stop_cache_trace reason=%s promoted=%s initializer_fn=%s cache_built=%s activated_step=%s post_activation_stop_after=%s",
                        step_num,
                        cache_rebuild_reason,
                        f_step is not f_diffusion,
                        initializer_fn is not None,
                        promoted_initializer_outputs is not None,
                        motif_unindexing_controller.activated_step,
                        motif_unindexing_controller.config.post_activation_stop_after,
                    )
                if cache_rebuild_reason is not None and promoted_initializer_outputs is None:
                    ranked_logger.info(
                        "[motif_unindexing] step=%s static_cache_rebuild_requested reason=%s promoted=%s initializer_fn=%s activated_step=%s post_activation_stop_after=%s",
                        step_num,
                        cache_rebuild_reason,
                        f_step is not f_diffusion,
                        initializer_fn is not None,
                        motif_unindexing_controller.activated_step,
                        motif_unindexing_controller.config.post_activation_stop_after,
                    )
                if f_step is not f_diffusion and initializer_fn is not None:
                    if promoted_initializer_outputs is None:
                        if cache_rebuild_reason is not None:
                            ranked_logger.info(
                                "[motif_unindexing] step=%s rebuilding_static_cache_from_promoted_step reason=%s activated_step=%s post_activation_stop_after=%s",
                                step_num,
                                cache_rebuild_reason,
                                motif_unindexing_controller.activated_step,
                                motif_unindexing_controller.config.post_activation_stop_after,
                            )
                            if cache_rebuild_reason == "assignment_lock_motif_pos_only":
                                f_init = motif_unindexing_controller.feature_dict_for_motif_pos_only_initializer(
                                    f_diffusion
                                )
                                promoted_initializer_outputs = initializer_fn(
                                    f_init if f_init is not None else f_step
                                )
                            elif cache_rebuild_reason == "assignment_lock_mask_only":
                                f_init = motif_unindexing_controller.feature_dict_for_mask_only_initializer(
                                    f_diffusion
                                )
                                promoted_initializer_outputs = initializer_fn(
                                    f_init if f_init is not None else f_step
                                )
                            else:
                                promoted_initializer_outputs = initializer_fn(f_step)
                        elif (
                            motif_unindexing_controller.config.promote_to_motif_on_activation
                            and motif_unindexing_controller.activated_step is not None
                        ):
                            f_init = motif_unindexing_controller.feature_dict_for_initializer(
                                f_diffusion
                            )
                            if f_init is not None:
                                promoted_initializer_outputs = initializer_fn(f_init)
                            else:
                                promoted_initializer_outputs = initializer_fn(f_step)
                    if promoted_initializer_outputs is not None:
                        initializer_outputs_step = promoted_initializer_outputs

            # Noise the coordinates with scaled Gaussian noise
            epsilon_L = (
                self.noise_scale
                * torch.sqrt(torch.square(t_hat) - torch.square(c_t_minus_1))
                * torch.normal(mean=0.0, std=1.0, size=X_L.shape, device=X_L.device)
            )
            epsilon_L[..., fixed_coord_noise_mask, :] = (
                0  # No noise injection for fixed atoms
            )

            # NOTE: no symmetry applied to the noisy structure
            X_noisy_L = X_L + epsilon_L
            self.log_step_diagnostics(step_num, "noisy", X_noisy_L, f)

            # Denoise the coordinates
            # Handle chunked mode vs standard mode (same as default sampler)
            if "chunked_pairwise_embedder" in initializer_outputs_step:
                # Chunked mode: explicitly provide P_LL=None
                tic = time.time()
                chunked_embedder = initializer_outputs_step[
                    "chunked_pairwise_embedder"
                ]  # Don't pop, just get
                other_outputs = {
                    k: v
                    for k, v in initializer_outputs_step.items()
                    if k != "chunked_pairwise_embedder"
                }
                outs = diffusion_module(
                    X_noisy_L=X_noisy_L,
                    t=t_hat.tile(D),
                    f=f_step,
                    P_LL=None,  # Not used in chunked mode
                    chunked_pairwise_embedder=chunked_embedder,
                    initializer_outputs=other_outputs,
                    n_recycle=self.n_recycle,
                    **other_outputs,
                )
                toc = time.time()
                ranked_logger.info(
                    f"[chunked] step {step_num}: {(toc - tic)*1000:.1f} ms"
                )
            else:
                # Standard mode: P_LL is included in initializer_outputs
                outs = diffusion_module(
                    X_noisy_L=X_noisy_L,
                    t=t_hat.tile(D),
                    f=f_step,
                    n_recycle=self.n_recycle,
                    **initializer_outputs_step,
                )
            outs = self.apply_post_denoise_symmetry(
                outs, f, step_num, c_t, gamma_min_sym
            )

            X_denoised_L = outs["X_L"] if "X_L" in outs else outs
            self.log_step_diagnostics(step_num, "denoised", X_denoised_L, f)

            # Compute the delta between the noisy and denoised coordinates, scaled by t_hat
            delta_L = (
                X_noisy_L - X_denoised_L
            ) / t_hat  # gradient of x wrt. t at x_t_hat
            d_t = c_t - t_hat

            # NOTE: no classifier-free guidance for symmetry

            if exists(outs.get("sequence_logits_I")):
                # Compute confidence
                p = torch.softmax(
                    outs["sequence_logits_I"], dim=-1
                ).cpu()  # shape (D, L, 32)
                seq_entropy = -torch.sum(
                    p * torch.log(p + 1e-10), dim=-1
                )  # shape (D, L,)
                sequence_entropy_traj.append(seq_entropy)

            # Update the coordinates, scaled by the step size
            # delta_L should be symmetric
            X_L = X_noisy_L + step_scale * d_t * delta_L

            # potential guidance hook
            if potential_adapter is not None:
                X_L = potential_adapter.apply(
                    X_L,
                    t=t_hat,
                    T=float(noise_schedule[0]),
                    step_idx=step_num,
                )
            if motif_unindexing_controller is not None:
                X_L = motif_unindexing_controller.apply_pre_activation_bias(
                    X_L,
                    step_num,
                )
            self.log_step_diagnostics(step_num, "post_ode", X_L, f)
            X_L = self.apply_post_update_symmetry(
                X_L, f, step_num, c_t, gamma_min_sym
            )
            self.log_step_diagnostics(step_num, "post_hetero", X_L, f)
            if should_project_floating_motifs(
                step_num,
                enabled=self.floating_motif_project,
                project_every=self.floating_motif_project_every,
                burn_in=self.floating_motif_burn_in,
                stop_after=self.floating_motif_stop_after,
            ):
                X_L = project_floating_motifs_all_atom(X_L, floating_motif_refs)
            if motif_unindexing_controller is not None:
                dynamic_refs = motif_unindexing_controller.active_floating_motif_refs(
                    step_num
                )
                if dynamic_refs:
                    X_L = project_floating_motifs_all_atom(
                        X_L,
                        dynamic_refs,
                        alpha=motif_unindexing_controller.projection_alpha(step_num),
                    )
                    X_L = self.apply_post_update_symmetry(
                        X_L, f, step_num, c_t, gamma_min_sym
                    )
                X_L = motif_unindexing_controller.apply_boundary_distance_bias(
                    X_L,
                    step_num,
                )
                motif_unindexing_controller.log_diagnostics(X_L, step_num)
            if motif_backbone_bias_controller is not None:
                X_L = motif_backbone_bias_controller.apply_bias(X_L, step_num)
            self.log_step_diagnostics(step_num, "post_kabsch", X_L, f)
            X_L = self.post_step_hook(X_L, f)
            self.log_step_diagnostics(step_num, "post_hook", X_L, f)

            # Append the results to the trajectory (for visualization of the diffusion process)
            X_noisy_L_scaled = (
                self.sigma_data * X_noisy_L / torch.sqrt(t_hat**2 + self.sigma_data**2)
            )  # Save noisy traj as scaled inputs
            X_noisy_L_traj.append(X_noisy_L_scaled)
            X_denoised_L_traj.append(X_denoised_L)
            t_hats.append(t_hat)

        if motif_backbone_bias_controller is not None:
            X_L = motif_backbone_bias_controller.postprocess_X_L(X_L)

        if torch.any(is_motif_atom_with_fixed_coord) and self.allow_realignment:
            # Insert the gt motif at the end
            X_L, R = centre_random_augment_around_motif(
                X_L,
                coord_atom_lvl_to_be_noised,
                is_motif_atom_with_fixed_coord,
                reinsert_motif=self.insert_motif_at_end,
            )

            # apply symmetry frame shift to X_L
            X_L = self.apply_symmetry_to_X_L(X_L, f)

            # Align prediction to original motif
            X_L = weighted_rigid_align(
                coord_atom_lvl_to_be_noised,
                X_L,
                X_exists_L=is_motif_atom_with_fixed_coord,
            )

        return dict(
            X_L=X_L,  # (D, L, 3)
            X_noisy_L_traj=X_noisy_L_traj,  # list[Tensor[D, L, 3]]
            X_denoised_L_traj=X_denoised_L_traj,  # list[Tensor[D, L, 3]]
            t_hats=t_hats,  # list[Tensor[D]], where D is shared across all diffusion batches
            sequence_logits_I=outs.get("sequence_logits_I"),  # (D, I, 32)
            sequence_indices_I=outs.get("sequence_indices_I"),  # (D, I, 32)
            sequence_entropy_traj=sequence_entropy_traj,  # list[Tensor[D, I]]
        )


class SampleDiffusionWithHeteroPseudoSymmetry(SampleDiffusionWithSymmetry):
    """Symmetric initialization with optional interface-only post-init symmetry.

    This sampler intentionally bypasses full-monomer projection after
    initialization.  Motif Kabsch projection and potentials are inherited from
    the standard sampler order: denoise, potential guidance, hetero interface
    projection, then floating motif Kabsch paste.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hetero_config = HeteroPseudoSymmetryConfig(
            post_init_symmetry=self.hetero_post_init_symmetry,
            projection_enabled=self.hetero_projection_enabled,
            hard=self.hetero_projection_hard,
            weight=self.hetero_projection_weight,
            start_step=self.hetero_projection_start_step,
            stop_after=self.hetero_projection_stop_after,
            schedule=self.hetero_projection_schedule,
            interface_distance_cutoff=self.hetero_interface_distance_cutoff,
            interface_sequence_buffer=self.hetero_interface_sequence_buffer,
            interface_include_sidechains=self.hetero_interface_include_sidechains,
            motif_contact_exclusion_enabled=(
                self.hetero_motif_contact_exclusion_enabled
            ),
            motif_contact_distance_cutoff=(
                self.hetero_motif_contact_distance_cutoff
            ),
            motif_contact_sequence_buffer=self.hetero_motif_contact_sequence_buffer,
            support_enabled=self.hetero_support_enabled,
            support_distance_cutoff=self.hetero_support_distance_cutoff,
            support_weight=self.hetero_support_weight,
            support_sequence_buffer=self.hetero_support_sequence_buffer,
            recenter_enabled=self.hetero_recenter_enabled,
            motif_follow_scaffold_frame=self.hetero_motif_follow_scaffold_frame,
            debug=self.hetero_debug,
        )
        ranked_logger.info(
            f"[hetero_symmetry] mode={self.hetero_config.post_init_symmetry} "
            f"projection_enabled={self.hetero_config.projection_enabled} "
            f"hard={self.hetero_config.hard} "
            f"weight={self.hetero_config.weight:.3f} "
            f"support_weight={self.hetero_config.support_weight:.3f}"
        )
        if self.hetero_debug or self.hetero_diagnostics_interval > 0:
            _emit_hetero_debug(
                "[hetero_debug_init] "
                f"post_init={self.hetero_config.post_init_symmetry} "
                f"projection_enabled={self.hetero_config.projection_enabled} "
                f"hard={self.hetero_config.hard} "
                f"weight={self.hetero_config.weight:.3f} "
                f"stop_after={self.hetero_config.stop_after} "
                f"recenter={self.hetero_config.recenter_enabled} "
                "motif_follow_scaffold_frame="
                f"{self.hetero_config.motif_follow_scaffold_frame} "
                f"diag_interval={self.hetero_diagnostics_interval}"
            )

    def apply_symmetry_to_X_L(self, X_L, f):
        del f
        return X_L

    def apply_post_denoise_symmetry(self, outs, f, step_num, c_t, gamma_min_sym):
        del f, step_num, c_t, gamma_min_sym
        return outs

    def post_step_hook(self, X_L: torch.Tensor, f: dict) -> torch.Tensor:
        """Recenter all movable atoms after Kabsch paste to prevent COM drift."""
        return _center_hetero_xyz(X_L, f, partial_diffusion=("partial_t" in f))

    def log_step_diagnostics(
        self,
        step_num: int,
        stage: str,
        X_L: torch.Tensor,
        f: dict,
    ) -> None:
        if self.hetero_diagnostics_interval <= 0:
            return
        if step_num >= 3 and step_num % self.hetero_diagnostics_interval != 0:
            return
        stats = _coordinate_diagnostics(X_L, f)
        _emit_hetero_debug(
            "[hetero_diag] "
            f"step={step_num:03d} stage={stage} "
            f"all_rog={stats['all_rog']:.2f} "
            f"scaffold_rog={stats['scaffold_rog']:.2f} "
            f"motif_rog={stats['motif_rog']:.2f} "
            f"all_max_r={stats['all_max_r']:.2f} "
            f"scaffold_max_r={stats['scaffold_max_r']:.2f} "
            f"motif_max_r={stats['motif_max_r']:.2f} "
            f"all_com_norm={stats['all_com_norm']:.2f} "
            f"scaffold_com_norm={stats['scaffold_com_norm']:.2f} "
            f"motif_com_norm={stats['motif_com_norm']:.2f}"
        )

    def apply_post_update_symmetry(self, X_L, f, step_num, c_t, gamma_min_sym):
        del c_t, gamma_min_sym
        X_L, debug = apply_hetero_pseudo_symmetry(
            X_L, f, self.hetero_config, step_num
        )
        if self.hetero_config.debug:
            _emit_hetero_debug(f"[hetero_symmetry] step={step_num} {debug}")
        return X_L

    def _get_initial_structure(
        self,
        c0: torch.Tensor,
        D: int,
        L: int,
        coord_atom_lvl_to_be_noised: torch.Tensor,
        is_motif_atom_with_fixed_coord,
    ) -> torch.Tensor:
        X_L = super()._get_initial_structure(
            c0=c0,
            D=D,
            L=L,
            coord_atom_lvl_to_be_noised=coord_atom_lvl_to_be_noised,
            is_motif_atom_with_fixed_coord=is_motif_atom_with_fixed_coord,
        )
        floating_motif_refs = getattr(
            self, "_hetero_initial_floating_motif_refs", None
        )
        if (
            self.hetero_init_floating_motifs_from_reference
            and floating_motif_refs
        ):
            X_L = _insert_floating_motif_reference_coords(X_L, floating_motif_refs)
        return X_L

    def sample_diffusion_like_af3(self, *, f, floating_motif_refs=None, **kwargs):
        self._validate_hetero_floating_motifs(f, floating_motif_refs)
        self._hetero_initial_floating_motif_refs = floating_motif_refs
        try:
            return super().sample_diffusion_like_af3(
                f=f, floating_motif_refs=floating_motif_refs, **kwargs
            )
        finally:
            self._hetero_initial_floating_motif_refs = None

    def _validate_hetero_floating_motifs(self, f, floating_motif_refs):
        if (
            not self.floating_motif_project
            or not self.hetero_require_per_copy_floating_motifs
            or not floating_motif_refs
            or "sym_transform_id" not in f
        ):
            return
        transform_ids = f["sym_transform_id"]
        n_copies = int(torch.unique(transform_ids).numel())
        if n_copies > 1 and len(floating_motif_refs) == 1:
            raise ValueError(
                "hetero_symmetry with floating_motif_project=True received one "
                f"floating motif reference for {n_copies} symmetry copies. "
                "Provide one independently defined motif per copy, or set "
                "hetero_require_per_copy_floating_motifs=False to bypass this "
                "validation."
            )


class ConditionalDiffusionSampler:
    """
    Conditional diffusion sampler, chooses at construction time which sampler to use,
    then forwards `sample_diffusion_like_af3` to the chosen sampler.
    If you write a new sampler, you best add it to the registry below
    and inference_sampler.kind in inference_engine config.
    """

    _registry = {
        "default": SampleDiffusionWithMotif,
        "symmetry": SampleDiffusionWithSymmetry,
        "hetero_symmetry": SampleDiffusionWithHeteroPseudoSymmetry,
    }

    def __init__(self, kind="default", **kwargs):
        ranked_logger.info(
            f"Initializing ConditionalDiffusionSampler with kind: {kind}"
        )
        try:
            SamplerCls = self._registry[kind]
            # remove kwargs that the sampler cannot take
            init_args = self.get_class_init_args(SamplerCls)
            kwargs = {k: v for k, v in kwargs.items() if k in init_args}
        except KeyError:
            raise ValueError(
                f"Invalid sampler kind: {kind}, must be one of {list(self._registry.keys())}"
            )
        self.sampler = SamplerCls(**kwargs)

    def sample_diffusion_like_af3(self, **kwargs):
        return self.sampler.sample_diffusion_like_af3(**kwargs)

    def get_class_init_args(self, cls):
        arg_names = []
        if hasattr(cls, "__init__") and callable(cls.__init__):
            for p_cls in cls.__mro__:
                if "__init__" in p_cls.__dict__ and p_cls is not object:
                    signature = inspect.signature(p_cls.__init__)
                    arg_names.extend(
                        [param.name for param in signature.parameters.values()]
                    )
        return arg_names


def _insert_floating_motif_reference_coords(X_L, floating_motif_refs):
    """Initialize hetero floating motifs at their per-copy reference positions."""
    if not floating_motif_refs:
        return X_L

    X_out = X_L.clone()
    for motif_ref in floating_motif_refs:
        atom_idx = motif_ref.sample_atom_indices.to(
            device=X_out.device, dtype=torch.long
        )
        if atom_idx.numel() == 0:
            continue
        reference = motif_ref.reference_xyz.to(
            device=X_out.device, dtype=X_out.dtype
        )
        valid = (
            motif_ref.reference_atom_mask.to(device=X_out.device).bool()
            & torch.isfinite(reference).all(dim=-1)
        )
        if not valid.any():
            continue
        current = X_out.index_select(dim=-2, index=atom_idx)
        reference = reference.unsqueeze(0).expand(current.shape[0], -1, -1)
        valid = valid.unsqueeze(0).expand(current.shape[0], -1)
        updated = torch.where(valid[..., None], reference, current)
        X_out.scatter_(
            dim=-2,
            index=atom_idx.view(1, -1, 1).expand(X_out.shape[0], -1, 3),
            src=updated,
        )
    return X_out


def _with_floating_motifs_unfixed_for_diffusion(f, floating_motif_refs):
    """Let floating motifs denoise while preserving the original conditioning dict."""
    if not floating_motif_refs or "is_motif_atom_with_fixed_coord" not in f:
        return f

    fixed_atom_mask = remove_floating_motif_atoms_from_fixed_mask(
        f["is_motif_atom_with_fixed_coord"], floating_motif_refs
    )
    f_diffusion = dict(f)
    f_diffusion["is_motif_atom_with_fixed_coord"] = fixed_atom_mask

    if (
        "is_motif_token_with_fully_fixed_coord" in f
        and "atom_to_token_map" in f
    ):
        f_diffusion["is_motif_token_with_fully_fixed_coord"] = (
            _token_fully_fixed_coord_mask(
                fixed_atom_mask,
                f["atom_to_token_map"],
                f["is_motif_token_with_fully_fixed_coord"],
            )
        )
    return f_diffusion


def _token_fully_fixed_coord_mask(
    fixed_atom_mask: torch.Tensor,
    atom_to_token_map: torch.Tensor,
    template_token_mask: torch.Tensor,
) -> torch.Tensor:
    token_mask = template_token_mask.clone()
    atom_to_token_map = atom_to_token_map.to(
        device=fixed_atom_mask.device, dtype=torch.long
    )
    for token_idx in torch.unique(atom_to_token_map).tolist():
        token_atoms = atom_to_token_map == int(token_idx)
        token_mask[int(token_idx)] = bool(fixed_atom_mask[token_atoms].all().item())
    return token_mask


def _emit_hetero_debug(message: str) -> None:
    """Emit hetero diagnostics even when rank-filtered logging is suppressed."""
    ranked_logger.info(message)
    print(message, file=sys.stderr, flush=True)


def _coordinate_diagnostics(X_L: torch.Tensor, f: dict) -> dict[str, float]:
    X = X_L.detach()
    if X.ndim == 2:
        X = X.unsqueeze(0)
    X = X.float()
    L = X.shape[-2]
    device = X.device
    real_mask = ~_feature_bool(f, "is_virtual", L, device)
    motif_mask = (
        _feature_bool(f, "is_motif_atom_with_fixed_coord", L, device)
        | _feature_bool(f, "is_motif_atom_with_fixed_seq", L, device)
        | _feature_bool(f, "is_motif_atom_unindexed", L, device)
        | _feature_bool(f, "is_motif_atom", L, device)
    ) & real_mask
    scaffold_mask = real_mask & ~motif_mask
    stats = {}
    stats.update(_coordinate_subset_stats(X, real_mask, "all"))
    stats.update(_coordinate_subset_stats(X, scaffold_mask, "scaffold"))
    stats.update(_coordinate_subset_stats(X, motif_mask, "motif"))
    return stats


def _coordinate_subset_stats(
    X: torch.Tensor,
    mask: torch.Tensor,
    prefix: str,
) -> dict[str, float]:
    if not bool(mask.any().item()):
        return {
            f"{prefix}_rog": float("nan"),
            f"{prefix}_max_r": float("nan"),
            f"{prefix}_com_norm": float("nan"),
        }
    coords = X[:, mask, :]
    com = coords.mean(dim=1, keepdim=True)
    centered = coords - com
    radius = torch.linalg.norm(centered, dim=-1)
    return {
        f"{prefix}_rog": torch.sqrt((radius.square()).mean()).item(),
        f"{prefix}_max_r": radius.max().item(),
        f"{prefix}_com_norm": torch.linalg.norm(com.squeeze(1), dim=-1).mean().item(),
    }


def _feature_bool(
    f: dict,
    key: str,
    L: int,
    device: torch.device,
) -> torch.Tensor:
    value = f.get(key)
    if value is None:
        return torch.zeros(L, dtype=torch.bool, device=device)
    if isinstance(value, torch.Tensor):
        value = value.to(device=device)
    else:
        value = torch.as_tensor(value, device=device)
    return value.bool()


def centre_random_augment_around_motif(
    X_L: torch.Tensor,  # (D, L, 3) noisy diffused coordinates
    coord_atom_lvl_to_be_noised: torch.Tensor,  # (D, L, 3) original coordinates
    is_motif_atom_with_fixed_coord: torch.Tensor,  # (D, L) indices in original coordinates to be kept constant
    s_trans: float = 1.0,
    center_option: str = "all",
    centering_affects_motif: bool = True,
    reinsert_motif=True,
):
    D, L, _ = X_L.shape

    if reinsert_motif and torch.any(is_motif_atom_with_fixed_coord):
        # ... Align original coordinates to the prediction
        coords_with_gt_aligned = weighted_rigid_align(
            X_L[..., is_motif_atom_with_fixed_coord, :],
            coord_atom_lvl_to_be_noised[..., is_motif_atom_with_fixed_coord, :],
        )

        # ... Insert original coordinates into X_L
        X_L[..., is_motif_atom_with_fixed_coord, :] = coords_with_gt_aligned

    # ... Centering
    if torch.any(is_motif_atom_with_fixed_coord):
        if center_option == "motif":
            center = torch.mean(
                X_L[..., is_motif_atom_with_fixed_coord, :], dim=-2, keepdim=True
            )  # (D, 1, 3) - COM of motif atoms
        elif center_option == "diffuse":
            center = torch.mean(
                X_L[..., ~is_motif_atom_with_fixed_coord, :], dim=-2, keepdim=True
            )  # (D, 1, 3) - COM of diffused atoms

        else:
            center = torch.mean(X_L, dim=-2, keepdim=True)
    else:
        center = torch.mean(X_L, dim=-2, keepdim=True)

    # ... Center
    if centering_affects_motif:
        X_L = X_L - center
    else:
        X_L[..., ~is_motif_atom_with_fixed_coord, :] = (
            X_L[..., ~is_motif_atom_with_fixed_coord, :] - center
        )

    # ... Random augmentation
    R = uniform_random_rotation((D,)).to(X_L.device)
    noise = (
        torch.normal(mean=0, std=1, size=(D, 1, 3), device=X_L.device) * s_trans
    )  # (D, 1, 3)
    X_L = rot_vec_mul(R[:, None], X_L) + noise

    return X_L, R

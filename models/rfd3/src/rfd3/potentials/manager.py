"""PotentialManager: aggregates potentials and computes timestep-decayed guide scale."""
from __future__ import annotations

import math

import torch

from rfd3.potentials.potentials import BasePotential


class PotentialManager:
    """Holds active potentials, sums them, and computes the time-decayed guide scale.

    Guide scale decay schedules (t and T are noise levels):
        constant   → guide_scale
        sqrt       → guide_scale * sqrt(t / T)
        linear     → guide_scale * (t / T)
        quadratic  → guide_scale * (t / T)^2
        cubic      → guide_scale * (t / T)^3
        quartic    → guide_scale * (t / T)^4
        exponential→ guide_scale * normalized exp curve from 0 to 1
        cosine     → guide_scale * half-cosine ramp from 0 to 1
        inverse_*  → guide_scale * (1 - matching non-inverse decay)

    Since t decreases from T → 0 over the diffusion trajectory, non-inverse
    schedules reduce guidance as the structure converges toward a clean sample.
    Inverse schedules increase guidance toward the end of the trajectory.
    """

    def __init__(
        self,
        potentials: list[BasePotential],
        guide_scale: float,
        guide_decay: str,
        guide_clip_rms: float,
        debug: bool = False,
    ):
        self.potentials = potentials
        self.guide_scale = guide_scale
        self.guide_decay = guide_decay
        self.guide_clip_rms = guide_clip_rms
        self.debug = debug

    def is_empty(self) -> bool:
        return len(self.potentials) == 0

    def compute_all_potentials(
        self,
        xyz: torch.Tensor,
        masks: dict[str, torch.Tensor],
        metadata: dict,
    ) -> torch.Tensor:
        """Sum all potential values into a single scalar for backprop.

        Uses explicit tensor accumulation (not Python's sum()) to ensure the
        grad_fn chain is preserved even for a single-potential list.
        """
        if self.is_empty():
            return xyz.new_zeros(())
        total: torch.Tensor | None = None
        for p in self.potentials:
            v = p.compute(xyz, masks, metadata)
            total = v if total is None else total + v
        return total  # type: ignore[return-value]

    def get_guide_scale(self, t: float, T: float) -> float:
        """Return time-decayed guide scale for current noise level t."""
        return self.get_potential_guide_scale(None, t, T)

    def get_potential_guide_scale(
        self,
        potential: BasePotential | None,
        t: float,
        T: float,
    ) -> float:
        """Return guide scale for a potential, honoring per-potential overrides."""
        guide_scale = float(
            getattr(potential, "guide_scale", self.guide_scale)
            if potential is not None
            else self.guide_scale
        )
        guide_decay = (
            getattr(potential, "guide_decay", self.guide_decay)
            if potential is not None
            else self.guide_decay
        )
        if T <= 0.0:
            return guide_scale
        ratio = max(0.0, min(1.0, t / T))  # 1 at first step, ~0 at last step
        decay = str(guide_decay)
        inverse = decay.startswith("inverse_")
        if inverse:
            decay = decay.removeprefix("inverse_")

        if decay == "constant":
            return guide_scale

        if decay == "sqrt":
            fraction = math.sqrt(ratio)
        elif decay == "linear":
            fraction = ratio
        elif decay == "quadratic":
            fraction = ratio**2
        elif decay == "cubic":
            fraction = ratio**3
        elif decay == "quartic":
            fraction = ratio**4
        elif decay == "exponential":
            beta = 5.0
            fraction = (math.exp(beta * ratio) - 1.0) / (math.exp(beta) - 1.0)
        elif decay == "cosine":
            fraction = 0.5 - 0.5 * math.cos(math.pi * ratio)
        else:
            return self.guide_scale

        if inverse:
            fraction = 1.0 - fraction
        return guide_scale * fraction

    def get_potential_guide_clip_rms(self, potential: BasePotential) -> float:
        """Return RMS clip threshold for a potential."""
        return float(getattr(potential, "guide_clip_rms", self.guide_clip_rms))

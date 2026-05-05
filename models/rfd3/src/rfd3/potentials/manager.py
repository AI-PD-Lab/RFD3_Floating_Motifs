"""PotentialManager: aggregates potentials and computes timestep-decayed guide scale."""
from __future__ import annotations

import torch

from rfd3.potentials.potentials import BasePotential


class PotentialManager:
    """Holds active potentials, sums them, and computes the time-decayed guide scale.

    Guide scale decay schedules (t and T are noise levels):
        constant   → guide_scale
        linear     → guide_scale * (t / T)
        quadratic  → guide_scale * (t / T)^2
        cubic      → guide_scale * (t / T)^3

    Since t decreases from T → 0 over the diffusion trajectory, all non-constant
    schedules reduce guidance as the structure converges toward a clean sample.
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
        if T <= 0.0:
            return self.guide_scale
        ratio = t / T  # in [0, 1]; 1 at first step, ~0 at last step
        if self.guide_decay == "constant":
            return self.guide_scale
        elif self.guide_decay == "linear":
            return self.guide_scale * ratio
        elif self.guide_decay == "quadratic":
            return self.guide_scale * (ratio**2)
        elif self.guide_decay == "cubic":
            return self.guide_scale * (ratio**3)
        else:
            return self.guide_scale

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PotentialsConfig:
    """Configuration for the external potential guidance system.

    Intended to be nested under the inference sampler config:

        inference_sampler:
          potentials:
            enabled: false
            apply_mode: token_translation
            ...
    """

    enabled: bool = False

    # How atom gradients are reduced / applied
    apply_mode: str = "token_translation"  # "token_translation" | "atom" | "hybrid"

    # Guide scale and decay schedule
    guide_scale: float = 0.25
    guide_decay: str = "quadratic"  # "constant" | "linear" | "quadratic" | "cubic"

    # RMS clip threshold (Angstrom) — applied before scale
    guide_clip_rms: float = 0.02

    # For hybrid mode: fraction of raw atom gradient beyond token translation to include
    # 0.0 == pure token_translation; 1.0 ~= pure atom mode
    atom_guidance_fraction: float = 0.25

    # Which atoms are used as input to potentials AND receive guidance
    include_atoms: str = "real_heavy"  # "all" | "real" | "real_heavy" | "backbone" | "CA"

    # Safety defaults — fixed and virtual atoms never move
    exclude_fixed_atoms: bool = True
    exclude_virtual_atoms: bool = True
    guide_only_generated: bool = True

    debug: bool = False

    # List of potential specifications (dict or RFD1-style string)
    guiding_potentials: list[Any] = field(default_factory=list)

    def __post_init__(self) -> None:
        valid_modes = ("token_translation", "atom", "hybrid")
        if self.apply_mode not in valid_modes:
            raise ValueError(
                f"apply_mode must be one of {valid_modes}, got {self.apply_mode!r}"
            )
        valid_decays = ("constant", "linear", "quadratic", "cubic")
        if self.guide_decay not in valid_decays:
            raise ValueError(
                f"guide_decay must be one of {valid_decays}, got {self.guide_decay!r}"
            )
        valid_include = ("all", "real", "real_heavy", "backbone", "CA")
        if self.include_atoms not in valid_include:
            raise ValueError(
                f"include_atoms must be one of {valid_include}, got {self.include_atoms!r}"
            )
        if not 0.0 <= float(self.atom_guidance_fraction) <= 1.0:
            raise ValueError(
                "atom_guidance_fraction must be in [0, 1], "
                f"got {self.atom_guidance_fraction!r}"
            )
        if self.guide_clip_rms < 0.0:
            raise ValueError(
                f"guide_clip_rms must be non-negative, got {self.guide_clip_rms!r}"
            )

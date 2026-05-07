"""Parse potential specifications from YAML config or RFdiffusion1-style strings."""
from __future__ import annotations

from collections.abc import Mapping

from rfd3.potentials.potentials import POTENTIAL_REGISTRY, BasePotential


GUIDANCE_OVERRIDE_KEYS = {
    "guide_scale",
    "guide_decay",
    "guide_clip_rms",
    "apply_mode",
    "atom_guidance_fraction",
}

VALID_GUIDE_DECAYS = (
    "constant",
    "sqrt",
    "linear",
    "quadratic",
    "cubic",
    "quartic",
    "exponential",
    "cosine",
    "inverse_sqrt",
    "inverse_linear",
    "inverse_quadratic",
    "inverse_cubic",
    "inverse_quartic",
    "inverse_exponential",
    "inverse_cosine",
)
VALID_APPLY_MODES = ("token_translation", "atom", "hybrid")


def parse_potentials(guiding_potentials: list | None) -> list[BasePotential]:
    """Parse a list of potential specs into BasePotential instances.

    Each entry is either:
    - dict:  {"type": "binder_ROG", "weight": 1.0, ...}
    - str:   "type:binder_ROG,weight:1.0,r_0:8.0"
    """
    if not guiding_potentials:
        return []
    return [_parse_one(spec) for spec in guiding_potentials]


def _parse_one(spec) -> BasePotential:
    plain_spec = _to_plain_container(spec)
    if isinstance(plain_spec, Mapping):
        return _parse_dict(dict(plain_spec))
    if isinstance(spec, str):
        return _parse_string(spec)
    raise ValueError(f"Unsupported potential spec type {type(spec)}: {spec!r}")


def _parse_dict(spec: dict) -> BasePotential:
    pot_type = spec.pop("type", None)
    if pot_type is None:
        raise ValueError(f"Potential spec missing 'type' key: {spec!r}")
    if pot_type not in POTENTIAL_REGISTRY:
        raise ValueError(
            f"Unknown potential type {pot_type!r}. "
            f"Available: {sorted(POTENTIAL_REGISTRY)}"
        )
    cls = POTENTIAL_REGISTRY[pot_type]
    guidance_overrides = {
        k: _coerce(spec.pop(k))
        for k in list(spec)
        if k in GUIDANCE_OVERRIDE_KEYS
    }
    _validate_guidance_overrides(guidance_overrides)
    kwargs = {k: _coerce(v) for k, v in spec.items()}
    potential = cls(**kwargs)
    for key, value in guidance_overrides.items():
        setattr(potential, key, value)
    return potential


def _validate_guidance_overrides(overrides: dict) -> None:
    guide_decay = overrides.get("guide_decay")
    if guide_decay is not None and guide_decay not in VALID_GUIDE_DECAYS:
        raise ValueError(
            f"guide_decay must be one of {VALID_GUIDE_DECAYS}, got {guide_decay!r}"
        )
    apply_mode = overrides.get("apply_mode")
    if apply_mode is not None and apply_mode not in VALID_APPLY_MODES:
        raise ValueError(
            f"apply_mode must be one of {VALID_APPLY_MODES}, got {apply_mode!r}"
        )
    guide_clip_rms = overrides.get("guide_clip_rms")
    if guide_clip_rms is not None and float(guide_clip_rms) < 0.0:
        raise ValueError(
            f"guide_clip_rms must be non-negative, got {guide_clip_rms!r}"
        )
    atom_guidance_fraction = overrides.get("atom_guidance_fraction")
    if atom_guidance_fraction is not None and not (
        0.0 <= float(atom_guidance_fraction) <= 1.0
    ):
        raise ValueError(
            "atom_guidance_fraction must be in [0, 1], "
            f"got {atom_guidance_fraction!r}"
        )


def _parse_string(spec: str) -> BasePotential:
    """Parse "type:binder_ROG,weight:1.0,r_0:8.0" into a potential."""
    parts = spec.split(",")
    d: dict = {}
    for part in parts:
        k, _, v = part.partition(":")
        if not k or not v:
            raise ValueError(f"Malformed potential spec segment {part!r} in {spec!r}")
        d[k.strip()] = v.strip()
    return _parse_dict(d)


def _coerce(v):
    """Try to convert a string value to int or float; leave as-is if not possible."""
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v
    if isinstance(v, str) and v.lower() in ("true", "false"):
        return v.lower() == "true"
    try:
        return int(v)
    except (ValueError, TypeError):
        pass
    try:
        return float(v)
    except (ValueError, TypeError):
        pass
    return v


def _to_plain_container(value):
    try:
        from omegaconf import DictConfig, ListConfig, OmegaConf
    except ImportError:
        return value
    if isinstance(value, (DictConfig, ListConfig)):
        return OmegaConf.to_container(value, resolve=True)
    return value

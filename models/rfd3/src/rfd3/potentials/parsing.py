"""Parse potential specifications from YAML config or RFdiffusion1-style strings."""
from __future__ import annotations

from collections.abc import Mapping

from rfd3.potentials.potentials import POTENTIAL_REGISTRY, BasePotential


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
    kwargs = {k: _coerce(v) for k, v in spec.items()}
    return cls(**kwargs)


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

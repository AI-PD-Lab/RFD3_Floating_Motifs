"""Shared ASU helpers for symmetry-aware potentials. Group-agnostic (Cn, Dn, ...).

Why every symmetry potential needs this
---------------------------------------
`apply_symmetry_to_xyz_atomwise` (symmetry_utils.py:384) rebuilds EVERY copy from the
ASU on each symmetrised step:

    asu_xyz = X_L[:, entity_asu_mask, :]
    sym_X_L[:, this_subunit, :] = asu_xyz @ R + t

Consequences for a potential:

  1. Gradient applied to a non-ASU copy is discarded wholesale at the next paste.
     Only the ASU's gradient survives.
  2. Between pastes the non-ASU copies' coordinates are NOT exact symmetry images --
     they are pre-paste noise. Reading per-copy quantities off them measures the wrong
     thing (observed: four C4 "copies" reporting axis dots +0.98 .. -0.49 in one step,
     for a design whose output was symmetric to 0.001).

So: derive the quantity from the ASU, GENERATE the other copies with the stored
`sym_transform`, and restrict guide mask + gradient to the ASU.

Note none of the fork's own `SymmetryAware*` potentials do this -- `is_sym_asu` is never
referenced in potentials.py -- so they all spread gradient over copies that get
overwritten.

Group-agnostic notes
--------------------
* `sym_transforms` returns whatever transforms the run declares, so Cn, Dn and
  `input_defined` all work without special-casing.
* Directions transform by rotation alone; points need rotation AND translation.
  Use `transform_directions` / `transform_points` accordingly.
* Do NOT assume every copy shares a quantity. Under Cn about z all copies share the
  z-component of any direction, but under Dn the perpendicular 2-folds flip it. Score
  per-copy terms over ALL generated copies; for Cn they are equal and it costs nothing.
"""
from __future__ import annotations

import torch

import rfd3.potentials.potentials as P

FIXED_TRANSFORM_ID = -1


def subunit_masks(masks, metadata, device):
    """One boolean atom mask per symmetry subunit (motif atoms)."""
    out = []
    for local_blocks in P._symmetry_subunit_motif_blocks(masks, metadata, device):
        if not local_blocks:
            continue
        m = torch.zeros_like(local_blocks[0], dtype=torch.bool, device=device)
        for b in local_blocks:
            m |= b
        out.append(m)
    return out


def asu_mask(masks, metadata, device):
    """Mask of the ASU subunit's motif atoms -- the only copy whose gradient survives.

    Returns (mask, reason). `reason` is None on success, else a short diagnostic and
    the mask falls back to the first subunit (better than silently scoring noise).
    """
    subs = subunit_masks(masks, metadata, device)
    if not subs:
        return None, "no_motif_blocks"
    is_asu = metadata.get("is_sym_asu")
    if is_asu is not None:
        is_asu = is_asu.to(device=device, dtype=torch.bool)
        for m in subs:
            if bool((m & is_asu).any()):
                return m, None
    return subs[0], "is_sym_asu_absent_using_first_subunit"


def sym_transforms(metadata, device, dtype):
    """[(R, t), ...] for every symmetry copy, excluding the fixed/non-symmetrised one.

    Works for any point group the run declares -- the transforms are read from the
    run's own `sym_transform`, not reconstructed from an assumed group.
    """
    st = metadata.get("sym_transform")
    if not st:
        return None
    out = []
    for k in sorted(st.keys(), key=lambda x: int(x)):
        if int(k) == FIXED_TRANSFORM_ID:
            continue
        entry = st[k]
        R = torch.as_tensor(entry[0], device=device, dtype=dtype).reshape(3, 3)
        try:
            t = torch.as_tensor(entry[1], device=device, dtype=dtype).reshape(3)
        except Exception:
            t = torch.zeros(3, device=device, dtype=dtype)
        out.append((R, t))
    return out or None


def transform_directions(v, transforms):
    """v [D,3] unit directions -> list of [D,3], one per copy. Rotation only."""
    return [torch.einsum("dj,jk->dk", v, R) for R, _ in transforms]


def transform_points(pts, transforms):
    """pts [D,N,3] -> list of [D,N,3], one per copy. Rotation AND translation,
    matching apply_symmetry_to_xyz_atomwise's `asu_xyz @ R + t`."""
    return [torch.einsum("dnj,jk->dnk", pts, R) + t for R, t in transforms]


def asu_blocks_or_all(masks, metadata, device):
    """Gradient/guide target: the ASU block alone, or all blocks if unmarked."""
    m, _ = asu_mask(masks, metadata, device)
    if m is not None:
        return [m]
    return P._symmetry_aware_all_blocks(masks, metadata, device)


def pair_indices(n, neighbor_only):
    """Copy pairs to score. `neighbor_only` presumes a cyclic ordering (Cn); for
    D groups and other non-cyclic layouts use all pairs."""
    if n < 2:
        return []
    if neighbor_only:
        return [(0, 1)] if n == 2 else [(i, (i + 1) % n) for i in range(n)]
    return [(i, j) for i in range(n) for j in range(i + 1, n)]

"""Append the receptor chain(s) attached to floating motifs onto the final
output structure, for runs that configure `minimal_overlap`/`minimal_clashes`
(rfd3.potentials.potentials.MinimalOverlapPotential) with
`motif_chains[*].include_in_output: true`.

RFD3 never diffuses the receptor itself -- see MinimalOverlapPotential's
docstring: it only ever loads a reference point cloud once and transports it
every step with the motif's own Kabsch rotation. `project_floating_motifs_all_atom`
does a hard replace at the default alpha=1.0, so the final motif atoms in the
output ARE the original reference motif geometry, rigidly transformed by the
last Kabsch fit RFD3 performed. That means re-fitting the motif's own stored
reference coordinates (`floating_motif_reference_x/y/z`, preserved as AtomArray
annotations through to the final output) onto its final output coordinates
recovers that exact transform with no approximation -- so the full-atom
receptor can be pasted in at output time without threading any new state
through the diffusion loop itself.

Which atoms belong to which motif block is resolved from
`prediction_metadata["diffused_index_map"]` (already written into every
design's output .json), NOT from any per-atom AtomArray annotation. Two
annotation-based heuristics were tried and both failed empirically on a real
run: `is_motif_atom_with_fixed_seq` only marks the sequence-fixed hotspot
subset of a motif (fragmenting one ~60-residue block into ~9 scattered
islands), and `floating_motif_reference_x/y/z` being non-NaN turned out to
cover essentially the WHOLE chain, not just the motif. `_motif_distance_blocks`
in potentials.py -- the function actually used by every potential during
diffusion -- gets this right via a fourth mask term (`is_motif_atom`) that
isn't available as a raw AtomArray annotation at output time (confirmed: it's
absent from the parsed AtomArray, so it must only be set later, in the
tensor-feature-dict construction the diffusion loop has access to but this
module doesn't). `diffused_index_map` sidesteps the whole problem: it's
RFD3's own record of original-contig-identifier -> output-identifier, written
by the engine itself, so grouping by it is exact rather than reconstructed.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import torch
import biotite.structure as struc
import biotite.structure.io as strucio
from biotite.structure import AtomArray

from rfd3.constants import (
    OPTIONAL_CONDITIONING_VALUES,
    REQUIRED_CONDITIONING_ANNOTATION_VALUES,
)
from rfd3.potentials.potentials import _kabsch_ref_to_current_rotation


def _log(msg: str) -> None:
    # plain print to stderr, not the `logging` module -- mirrors
    # MinimalOverlapPotential's own debug_log mechanism (potentials.py),
    # which is proven to reach SLURM .err logs in this environment; a prior
    # version of this module used logging.getLogger(__name__) and NONE of
    # its messages ever appeared in a real run's log, root cause not yet
    # understood, so don't depend on it here.
    print(f"[receptor_grafting] {msg}", file=sys.stderr, flush=True)

_FLOATING_MOTIF_REFERENCE_ANNOTATIONS = (
    "floating_motif_reference_x",
    "floating_motif_reference_y",
    "floating_motif_reference_z",
)

# Largest fraction of the design's own atoms a single "motif block" is allowed
# to cover before we assume the block-detection heuristic below has gone wrong
# (e.g. degenerated to "the whole structure is one block") and bail out rather
# than risk grafting a receptor onto a nonsensical block.
_MAX_PLAUSIBLE_BLOCK_FRACTION = 0.5

# Smallest atom count a single referenced motif block is allowed to have
# before we assume something is malformed in diffused_index_map (a real
# floating motif here is on the order of hundreds of atoms; a genuine block
# would not plausibly be this small) -- defensive backstop, kept even though
# diffused_index_map is authoritative rather than reconstructed.
_MIN_PLAUSIBLE_BLOCK_ATOMS = 50

_CHAIN_ID_ALPHABET = [chr(c) for c in range(ord("A"), ord("Z") + 1)] + [
    chr(c) for c in range(ord("a"), ord("z") + 1)
]


def graft_receptor_chains(
    atom_array: AtomArray,
    guiding_potentials: list[dict] | None,
    diffused_index_map: dict[str, str] | None,
) -> AtomArray:
    """Return `atom_array` with configured receptor chains appended.

    No-op (returns `atom_array` unchanged) if no `minimal_overlap`/
    `minimal_clashes` potential with `include_in_output: true` entries is
    configured, or if grafting isn't safe to do automatically -- in the
    latter case this logs a warning and skips rather than raising, so a
    surprise here never fails an otherwise-successful design run.

    `diffused_index_map` (from `prediction_metadata["diffused_index_map"]`)
    maps original contig identifiers (e.g. "A1", "D63") to output identifiers
    (e.g. "A1", "A257") -- grouping its keys by original chain, in first-
    appearance (contig) order, gives exactly the motif blocks
    `motif_chains[*].motif_index` is meant to index into.
    """
    _log(
        f"called with guiding_potentials={type(guiding_potentials).__name__} "
        f"len={len(guiding_potentials) if guiding_potentials is not None else 0} "
        f"types={[type(p).__name__ for p in (guiding_potentials or [])]}"
    )
    specs = _collect_motif_chain_specs(guiding_potentials)
    _log(f"{len(specs)} motif_chains spec(s) with include_in_output=true found")
    if not specs:
        return atom_array

    categories = set(atom_array.get_annotation_categories())
    if not all(a in categories for a in _FLOATING_MOTIF_REFERENCE_ANNOTATIONS):
        _log(
            "floating_motif_reference_x/y/z annotations not found on the output "
            "structure -- skipping receptor grafting. This should not happen "
            "for indexed floating motifs; if you see this, something upstream "
            "is stripping annotations that are normally preserved through to "
            "RFD3Output."
        )
        return atom_array

    if not diffused_index_map:
        _log(
            "diffused_index_map is missing or empty in prediction_metadata -- "
            "skipping receptor grafting (can't reliably determine motif block "
            "boundaries without it)."
        )
        return atom_array

    blocks = _motif_blocks_from_diffused_index_map(atom_array, diffused_index_map)
    block_orig_resids = _motif_block_orig_resids(diffused_index_map)
    _log(f"detected {len(blocks)} motif block(s), sizes {[len(b) for b in blocks]}")
    if blocks and max(len(b) for b in blocks) > _MAX_PLAUSIBLE_BLOCK_FRACTION * len(
        atom_array
    ):
        _log(
            "the largest detected motif block covers more than "
            f"{_MAX_PLAUSIBLE_BLOCK_FRACTION:.0%} of the structure's atoms -- "
            "this doesn't look like a real floating-motif block, skipping "
            "receptor grafting rather than risk attaching a receptor to the "
            "wrong atoms."
        )
        return atom_array

    result = atom_array
    used_chains = set(np.unique(result.chain_id).tolist())
    for spec in specs:
        motif_i = spec["motif_index"]
        if not (0 <= motif_i < len(blocks)):
            _log(
                f"motif_index {motif_i} out of range ({len(blocks)} motif "
                "blocks detected on the output structure) -- skipping this "
                "entry"
            )
            continue

        block_idx = blocks[motif_i]
        # Narrow to backbone atoms (N, CA, C, O) before anything else. Only
        # a scattered hotspot subset of a motif's sequence is held fixed
        # (is_motif_atom_with_fixed_seq) -- RFD3 is free to redesign the rest,
        # so the OUTPUT structure's own per-block atom count genuinely varies
        # design-to-design (different amino acids have different heavy-atom
        # counts). Backbone atoms are the one atom set that's invariant to
        # sequence identity, so restricting to them here is what makes this
        # module's own Kabsch fit and (when align=True) its offline alignment
        # correspondence reliable for *any* motif/receptor pair regardless of
        # how much of the sequence ends up redesigned -- unlike
        # MinimalOverlapPotential's own `ref_xyz_i`, which stays fixed because
        # it's built from a static input-time template, not the final output.
        block_idx = block_idx[_select_atoms(atom_array, "backbone")[block_idx]]
        if len(block_idx) < _MIN_PLAUSIBLE_BLOCK_ATOMS:
            _log(
                f"motif_index {motif_i} resolved to a backbone block of only "
                f"{len(block_idx)} atoms (< {_MIN_PLAUSIBLE_BLOCK_ATOMS}) -- "
                "this doesn't look like a real floating-motif block, skipping "
                "this entry rather than risk a degenerate Kabsch fit on a "
                "scattered fragment"
            )
            continue

        ref_xyz = np.stack(
            [
                np.asarray(atom_array.floating_motif_reference_x)[block_idx],
                np.asarray(atom_array.floating_motif_reference_y)[block_idx],
                np.asarray(atom_array.floating_motif_reference_z)[block_idx],
            ],
            axis=-1,
        )
        cur_xyz = np.asarray(atom_array.coord)[block_idx]

        # ref_xyz is already backbone-only (see above), so it IS the
        # correspondence target for the offline alignment fit -- no separate
        # selection needed.
        ca_ref_xyz = ref_xyz if spec.get("align") else None

        motif_res_ids = (
            block_orig_resids[motif_i]
            if 0 <= motif_i < len(block_orig_resids)
            else None
        )
        receptor_aa = _transport_receptor_chain(
            spec, ref_xyz, cur_xyz, ca_ref_xyz, motif_res_ids
        )
        if receptor_aa is None:
            continue

        receptor_aa = _assign_fresh_chain_id(receptor_aa, spec["chain_id"], used_chains)
        used_chains.add(str(receptor_aa.chain_id[0]))
        result = _harmonize_and_concatenate(result, receptor_aa)
        _log(
            f"appended {spec['chain_pdb']} chain {spec['chain_id']} "
            f"({receptor_aa.array_length()} atoms) as chain "
            f"{receptor_aa.chain_id[0]}, attached to motif block {motif_i}"
        )

    return result


def _collect_motif_chain_specs(guiding_potentials: list[dict] | None) -> list[dict]:
    specs = []
    for pot in guiding_potentials or []:
        if not isinstance(pot, dict):
            continue
        if pot.get("type") not in ("minimal_overlap", "minimal_clashes"):
            continue
        for spec in pot.get("motif_chains") or []:
            if spec.get("include_in_output"):
                specs.append(spec)
    return specs


_CHAIN_RESID_RE = re.compile(r"^([A-Za-z]+)(\d+)$")


def _parse_chain_resid(token: str) -> tuple[str, int] | None:
    m = _CHAIN_RESID_RE.match(token)
    if not m:
        return None
    return m.group(1), int(m.group(2))


def _motif_blocks_from_diffused_index_map(
    atom_array: AtomArray, diffused_index_map: dict[str, str]
) -> list[np.ndarray]:
    """Group output atom indices into motif blocks using RFD3's own
    input->output identifier map, instead of any per-atom annotation.

    Keys are original {chain}{res_id} identifiers (e.g. "A1", "D63") in
    contig order; values are the corresponding OUTPUT {chain}{res_id}
    identifiers after merging into the single generated chain. Residues with
    no entry (e.g. the auto-gap linker) are correctly excluded by simply not
    appearing. Grouping keys by original chain, in first-appearance order,
    gives blocks in the same order `motif_chains[*].motif_index` expects
    (contig order) -- confirmed on a real design: original chains "A" then
    "D", 63 residues each, both landing in output chain "A".
    """
    order: list[str] = []
    out_pairs_by_orig_chain: dict[str, list[tuple[str, int]]] = {}
    for orig_key, out_key in diffused_index_map.items():
        parsed_orig = _parse_chain_resid(orig_key)
        parsed_out = _parse_chain_resid(out_key)
        if parsed_orig is None or parsed_out is None:
            continue
        orig_chain = parsed_orig[0]
        if orig_chain not in out_pairs_by_orig_chain:
            out_pairs_by_orig_chain[orig_chain] = []
            order.append(orig_chain)
        out_pairs_by_orig_chain[orig_chain].append(parsed_out)

    chain_id_arr = np.asarray(atom_array.chain_id)
    res_id_arr = np.asarray(atom_array.res_id)

    blocks = []
    for orig_chain in order:
        resids_by_out_chain: dict[str, set[int]] = {}
        for out_chain, out_resid in out_pairs_by_orig_chain[orig_chain]:
            resids_by_out_chain.setdefault(out_chain, set()).add(out_resid)

        mask = np.zeros(atom_array.array_length(), dtype=bool)
        for out_chain, resids in resids_by_out_chain.items():
            mask |= (chain_id_arr == out_chain) & np.isin(res_id_arr, list(resids))
        blocks.append(np.flatnonzero(mask))
    return blocks


def _motif_block_orig_resids(
    diffused_index_map: dict[str, str]
) -> list[set[int]]:
    """Original residue ids per motif block, grouped by ORIGINAL chain in
    first-appearance (contig) order -- index-parallel to
    `_motif_blocks_from_diffused_index_map` (same grouping, same order).

    These are the residues the contig actually kept. Restricting `align_chain_id`
    to them lets the offline alignment fit succeed when the align chain (e.g. a
    full-length receptor-template chain) spans residues the contig trimmed away;
    without it the whole-chain count won't match the motif block and the
    alignment correction is silently skipped, grafting the receptor in the
    chain_pdb's raw (un-re-registered) frame.
    """
    order: list[str] = []
    resids_by_orig_chain: dict[str, set[int]] = {}
    for orig_key in diffused_index_map:
        parsed = _parse_chain_resid(orig_key)
        if parsed is None:
            continue
        orig_chain, orig_resid = parsed
        if orig_chain not in resids_by_orig_chain:
            resids_by_orig_chain[orig_chain] = set()
            order.append(orig_chain)
        resids_by_orig_chain[orig_chain].add(orig_resid)
    return [resids_by_orig_chain[chain] for chain in order]


def _select_atoms(atom_array: AtomArray, selection: str) -> np.ndarray:
    """Boolean atom mask for 'CA' / 'backbone' / 'heavy' / 'all', mirroring
    potentials.py's `_load_chain_atoms` selection semantics exactly -- needed
    so `align_atom_selection` picks out the same atom count/order on both the
    align_chain_id side (raw chain_pdb file) and the design's own motif block
    reference, which is required for the two point clouds to correspond 1:1.
    """
    atom_name = np.asarray(atom_array.atom_name)
    if selection == "CA":
        return atom_name == "CA"
    if selection == "backbone":
        return np.isin(atom_name, ["N", "CA", "C", "O"])
    if selection == "heavy":
        element = atom_array.element
        if element is None or getattr(element, "size", 0) == 0:
            element = struc.infer_elements(atom_array)
        # OXT is a terminus-only atom present in essentially any raw PDB but
        # never modeled by RFD3's own fixed per-residue atom template --
        # excluded here for the same reason as potentials.py's
        # `_load_chain_atoms`, so this stays 1:1-comparable for any future
        # receptor/motif pair without needing hand-cleaned input files.
        return (np.asarray(element) != "H") & (atom_name != "OXT")
    if selection == "all":
        return np.ones(atom_array.array_length(), dtype=bool)
    raise ValueError(
        f"receptor_grafting align_atom_selection must be one of 'CA', "
        f"'backbone', 'heavy', 'all', got {selection!r}"
    )


def _transport_receptor_chain(
    spec: dict,
    ref_xyz: np.ndarray,
    cur_xyz: np.ndarray,
    ca_ref_xyz: np.ndarray | None = None,
    motif_res_ids: set[int] | None = None,
) -> AtomArray | None:
    """Mirrors MinimalOverlapPotential's own `align`/`align_chain_id` offline
    registration (potentials.py's `_resolve_chain_ref_xyz`): by default
    (``align`` unset) the chain_pdb's raw coordinates are assumed to already
    live in the same frame as `floating_motif_reference_x/y/z` -- true only
    when the design used no origin-shifting input strategy (e.g.
    `infer_ori_strategy: com` recenters the whole parsed structure, but never
    touches an externally, separately-loaded chain_pdb file). When that
    assumption doesn't hold, `align: true` + `align_chain_id` (the motif's own
    chain letter in the same chain_pdb file) recovers the correction the same
    way the potential does: Kabsch-fit align_chain_id's raw atoms onto the
    design's own CA motif reference, then apply that same rotation+translation
    to the receptor chain before the per-step rigid transport below.
    """
    chain_pdb = spec["chain_pdb"]
    chain_id = str(spec["chain_id"])
    output_atom_selection = spec.get("output_atom_selection", "heavy")

    path = Path(chain_pdb)
    if not path.is_file():
        _log(f"{chain_pdb} not found -- skipping")
        return None

    full = strucio.load_structure(str(path), model=1)
    chain_mask = full.chain_id == chain_id

    if output_atom_selection == "heavy":
        element = full.element
        if element is None or getattr(element, "size", 0) == 0:
            full = full.copy()
            full.element = struc.infer_elements(full)
            element = full.element
        chain_mask = chain_mask & (np.asarray(element) != "H")
    elif output_atom_selection == "backbone":
        chain_mask = chain_mask & np.isin(full.atom_name, ["N", "CA", "C", "O"])
    elif output_atom_selection == "CA":
        chain_mask = chain_mask & (full.atom_name == "CA")
    elif output_atom_selection != "all":
        raise ValueError(
            "receptor_grafting output_atom_selection must be one of 'heavy', "
            f"'backbone', 'CA', 'all', got {output_atom_selection!r}"
        )

    receptor_aa = full[chain_mask].copy()
    if receptor_aa.array_length() == 0:
        _log(
            f"no atoms found for chain {chain_id!r} in {chain_pdb} with "
            f"output_atom_selection={output_atom_selection!r} -- skipping"
        )
        return None

    ref_xyz_t = torch.as_tensor(ref_xyz, dtype=torch.float32)
    cur_xyz_t = torch.as_tensor(cur_xyz, dtype=torch.float32).unsqueeze(0)
    rotation = _kabsch_ref_to_current_rotation(ref_xyz_t, cur_xyz_t)
    if rotation is None:
        _log(
            f"degenerate Kabsch fit for the motif attached to chain "
            f"{chain_id!r} in {chain_pdb} -- skipping"
        )
        return None

    ref_centered = ref_xyz_t - ref_xyz_t.mean(dim=0)
    cur_centered = cur_xyz_t[0] - cur_xyz_t[0].mean(dim=0)
    fitted = ref_centered @ rotation[0]
    fit_rmsd = torch.sqrt(((fitted - cur_centered) ** 2).sum(dim=-1).mean()).item()
    _log(
        f"Kabsch fit for motif attached to chain {chain_id!r}: n_atoms="
        f"{ref_xyz.shape[0]} fit_rmsd={fit_rmsd:.3f} A "
        f"ref_extent={ref_xyz.std(axis=0).round(2).tolist()} "
        f"cur_extent={cur_xyz.std(axis=0).round(2).tolist()}"
    )

    receptor_coords = torch.as_tensor(receptor_aa.coord, dtype=torch.float32)

    if spec.get("align"):
        align_chain_id = spec.get("align_chain_id")
        # Always "backbone" here, regardless of `align_atom_selection` in the
        # shared config -- that field also feeds MinimalOverlapPotential's
        # own alignment, whose ref_xyz_i is stable under "heavy" because it
        # comes from a fixed input-time template. This module's ca_ref_xyz
        # comes from the final OUTPUT structure instead (see block_idx
        # narrowing above), where atom composition varies with whatever
        # sequence RFD3 redesigned -- backbone is the one selection that's
        # guaranteed 1:1-comparable no matter what.
        align_sel = "backbone"
        if not align_chain_id:
            _log(
                f"align=True for chain {chain_id!r} but align_chain_id is "
                "missing -- skipping alignment correction, using chain_pdb's "
                "raw frame as-is (likely wrong if the file needs re-registration)"
            )
        elif ca_ref_xyz is None or ca_ref_xyz.shape[0] == 0:
            _log(
                f"align=True for chain {chain_id!r} but no "
                f"{align_sel!r} atoms found in the design's own motif block "
                "reference -- skipping alignment correction"
            )
        else:
            chain_mask_align = np.asarray(full.chain_id) == align_chain_id
            align_mask = chain_mask_align & _select_atoms(full, align_sel)
            # Restrict the align chain to exactly the contig motif's residues.
            # ca_ref_xyz is the motif block (only the residues the contig kept);
            # an align chain that carries extra residues (e.g. a full-length
            # receptor-template chain) would otherwise fail the 1:1 count check
            # below and silently skip the alignment correction. Matches on
            # res_id, which is shared because align_chain_id and the motif come
            # from the same source numbering.
            if motif_res_ids is not None:
                align_mask = align_mask & np.isin(
                    np.asarray(full.res_id), sorted(motif_res_ids)
                )
            align_xyz_np = np.asarray(full.coord)[align_mask]
            if align_xyz_np.shape[0] != ca_ref_xyz.shape[0]:
                _log(
                    f"align_chain_id={align_chain_id!r} has "
                    f"{align_xyz_np.shape[0]} {align_sel!r} atoms in "
                    f"{chain_pdb} but the design's own motif reference has "
                    f"{ca_ref_xyz.shape[0]} -- skipping alignment correction "
                    "(need a 1:1 atom correspondence)"
                )
            else:
                align_xyz_t = torch.as_tensor(align_xyz_np, dtype=torch.float32)
                ca_ref_xyz_t = torch.as_tensor(
                    ca_ref_xyz, dtype=torch.float32
                ).unsqueeze(0)
                R_align = _kabsch_ref_to_current_rotation(align_xyz_t, ca_ref_xyz_t)
                if R_align is None:
                    _log(
                        f"degenerate offline alignment fit for "
                        f"align_chain_id={align_chain_id!r} -- skipping "
                        "alignment correction"
                    )
                else:
                    align_com = align_xyz_t.mean(dim=0)
                    ref_ca_mean = ca_ref_xyz_t[0].mean(dim=0)
                    receptor_coords = (
                        receptor_coords - align_com
                    ) @ R_align[0] + ref_ca_mean
                    _log(
                        f"applied offline alignment for chain {chain_id!r} "
                        f"via align_chain_id={align_chain_id!r} "
                        f"({align_xyz_np.shape[0]} atoms)"
                    )

    ref_com = torch.as_tensor(ref_xyz.mean(axis=0), dtype=torch.float32)
    cur_com = torch.as_tensor(cur_xyz.mean(axis=0), dtype=torch.float32)
    transported = (receptor_coords - ref_com) @ rotation[0] + cur_com
    receptor_aa.coord = transported.numpy()

    receptor_aa.set_annotation(
        "is_grafted_receptor", np.ones(receptor_aa.array_length(), dtype=bool)
    )
    return receptor_aa


def _assign_fresh_chain_id(
    receptor_aa: AtomArray, original_chain_id: str, used_chains: set[str]
) -> AtomArray:
    candidates = [original_chain_id] + [
        c for c in _CHAIN_ID_ALPHABET if c != original_chain_id
    ]
    for candidate in candidates:
        if candidate not in used_chains:
            receptor_aa.set_annotation(
                "chain_id",
                np.full(receptor_aa.array_length(), candidate),
            )
            return receptor_aa
    raise RuntimeError("receptor_grafting: ran out of chain ids to assign")


def _harmonize_and_concatenate(
    atom_array: AtomArray, receptor_aa: AtomArray
) -> AtomArray:
    """Fill in missing annotation categories on whichever side lacks them
    before concatenating, mirroring the pattern already used to append
    floating motifs as new chains at parse time
    (inference/input_parsing.py's `_append_motifs_as_chains`)."""
    all_defaults = {
        **REQUIRED_CONDITIONING_ANNOTATION_VALUES,
        **OPTIONAL_CONDITIONING_VALUES,
        "is_grafted_receptor": False,
    }
    for annot, default in all_defaults.items():
        dtype = bool if annot == "is_grafted_receptor" else (
            np.float64 if isinstance(default, float) else int
        )
        if (
            annot in atom_array.get_annotation_categories()
            and annot not in receptor_aa.get_annotation_categories()
        ):
            receptor_aa.set_annotation(
                annot, np.full(receptor_aa.array_length(), default, dtype=dtype)
            )
        elif (
            annot in receptor_aa.get_annotation_categories()
            and annot not in atom_array.get_annotation_categories()
        ):
            atom_array.set_annotation(
                annot, np.full(atom_array.array_length(), default, dtype=dtype)
            )
    return struc.concatenate([atom_array, receptor_aa])

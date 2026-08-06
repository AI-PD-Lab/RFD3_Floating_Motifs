"""Tests for MinimalOverlapPotential (registered as minimal_overlap / minimal_clashes).

No GPU or checkpoint needed -- these construct synthetic xyz/masks/metadata
dicts directly against the BasePotential.compute contract, the same way the
other potentials in potentials.py are exercised internally, plus a couple of
tiny structures written on the fly via biotite to test the PDB-chain loader
and the offline (align=True) Kabsch pre-alignment path.
"""

from __future__ import annotations

import biotite.structure as struc
import biotite.structure.io as strucio
import pytest
import torch

from rfd3.potentials.potentials import (
    POTENTIAL_REGISTRY,
    MinimalOverlapPotential,
    _kabsch_ref_to_current_rotation,
    _load_chain_atoms,
    _pairwise_clash_overlap,
)

torch.manual_seed(0)


# --------------------------------------------------------------------------
# Helpers to build synthetic structures / RFD3-style masks+metadata
# --------------------------------------------------------------------------


def _write_pdb(tmp_path, name, chains):
    """chains: dict[chain_id -> Nx3 array-like]. Writes CA-only atoms."""
    n_total = sum(len(coords) for coords in chains.values())
    arr = struc.AtomArray(n_total)
    i = 0
    for chain_id, coords in chains.items():
        for res_i, xyz in enumerate(coords):
            arr.coord[i] = xyz
            arr.chain_id[i] = chain_id
            arr.res_id[i] = res_i + 1
            arr.res_name[i] = "GLY"
            arr.atom_name[i] = "CA"
            arr.element[i] = "C"
            arr.hetero[i] = False
            i += 1
    path = tmp_path / name
    strucio.save_structure(str(path), arr)
    return str(path)


def _two_motif_scene(ref0, ref1, current0, current1, extra_ca=None):
    """Build xyz/masks/metadata for two 4-atom motif blocks (tokens with a gap
    between them so `_motif_distance_blocks` doesn't merge them into one run).
    """
    ref = torch.cat([ref0, ref1], dim=0)
    current = torch.cat([current0, current1], dim=0)
    n_motif = ref.shape[0]
    if extra_ca is not None:
        current = torch.cat([current, extra_ca], dim=0)
        n_extra = extra_ca.shape[0]
    else:
        n_extra = 0
    n_total = n_motif + n_extra

    atom_to_token = torch.cat(
        [
            torch.arange(0, ref0.shape[0]),
            torch.arange(10, 10 + ref1.shape[0]),
            torch.arange(100, 100 + n_extra),
        ]
    ).long()

    motif_atom_mask = torch.zeros(n_total, dtype=torch.bool)
    motif_atom_mask[:n_motif] = True
    real_atom_mask = torch.ones(n_total, dtype=torch.bool)
    ca_atom_mask = torch.ones(n_total, dtype=torch.bool)

    input_pos = torch.full((n_total, 3), float("nan"))
    input_pos[:n_motif] = ref

    xyz = current.unsqueeze(0)  # [D=1, L, 3]
    masks = {
        "motif_atom_mask": motif_atom_mask,
        "real_atom_mask": real_atom_mask,
        "ca_atom_mask": ca_atom_mask,
        "potential_atom_mask": real_atom_mask,
    }
    metadata = {
        "atom_to_token_map": atom_to_token,
        "input_pos": input_pos,
    }
    return xyz, masks, metadata


def _tetra(offset):
    """A small non-degenerate 4-point rigid cluster, translated by offset."""
    base = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    return base + torch.tensor(offset)


def _asymmetric_tetra(offset):
    """Like `_tetra`, but deliberately irregular.

    `_tetra`'s corner-simplex shape is symmetric under permuting x/y/z, which
    gives its centred Gram matrix (ref_centred.T @ ref_centred) a REPEATED
    eigenvalue. Kabsch's rotation comes from `torch.linalg.svd(covariance)`,
    and covariance's singular values equal that Gram matrix's eigenvalues
    *regardless of which rotation was applied* (covariance = Gram @ R, R
    orthogonal) -- so any exact-fit Kabsch call on `_tetra` hits PyTorch's
    well-known SVD-backward instability for repeated singular values, giving
    NaN gradients on the very first `autograd.grad` call. Forward-only tests
    never touch that backward pass, so `_tetra` is fine for them; anything
    that differentiates through `_kabsch_ref_to_current_rotation` needs a
    shape with a non-degenerate Gram spectrum instead.
    """
    base = torch.tensor(
        [[0.0, 0.0, 0.0], [1.3, 0.0, 0.0], [0.0, 1.7, 0.0], [0.2, 0.3, 2.1]]
    )
    return base + torch.tensor(offset)


FLIP_Z = torch.tensor([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])
IDENTITY = torch.eye(3)


def _rigid_transform(points, ref_com, rotation, target_com):
    return (points - ref_com) @ rotation + target_com


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------


def test_registered_under_both_names():
    assert POTENTIAL_REGISTRY["minimal_overlap"] is MinimalOverlapPotential
    assert POTENTIAL_REGISTRY["minimal_clashes"] is MinimalOverlapPotential


# --------------------------------------------------------------------------
# Constructor validation
# --------------------------------------------------------------------------


def test_requires_motif_chains():
    with pytest.raises(ValueError, match="motif_chains"):
        MinimalOverlapPotential(weight=1.0)
    with pytest.raises(ValueError, match="motif_chains"):
        MinimalOverlapPotential(weight=1.0, motif_chains=[])


def test_requires_chain_pdb_and_chain_id():
    with pytest.raises(ValueError):
        MinimalOverlapPotential(weight=1.0, motif_chains=[{"motif_index": 0}])


def test_align_requires_align_chain_id():
    with pytest.raises(ValueError, match="align_chain_id"):
        MinimalOverlapPotential(
            weight=1.0,
            motif_chains=[
                {
                    "motif_index": 0,
                    "chain_pdb": "unused.pdb",
                    "chain_id": "R",
                    "align": True,
                }
            ],
        )


def test_bad_protein_atom_filter_rejected():
    with pytest.raises(ValueError, match="protein_atom_filter"):
        MinimalOverlapPotential(
            weight=1.0,
            motif_chains=[{"motif_index": 0, "chain_pdb": "x.pdb", "chain_id": "R"}],
            protein_atom_filter="nonsense",
        )


# --------------------------------------------------------------------------
# _load_chain_atoms
# --------------------------------------------------------------------------


def test_load_chain_atoms_filters_chain_and_atom_selection(tmp_path):
    chain_a = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
    chain_b = [[10.0, 0.0, 0.0], [11.0, 0.0, 0.0]]
    pdb_path = _write_pdb(tmp_path, "chains.pdb", {"A": chain_a, "B": chain_b})

    loaded_b = _load_chain_atoms(pdb_path, "B", "CA")
    assert loaded_b.shape == (2, 3)
    assert torch.allclose(
        loaded_b.sort(dim=0).values, torch.tensor(chain_b).sort(dim=0).values, atol=1e-3
    )

    loaded_a = _load_chain_atoms(pdb_path, "A", "CA")
    assert loaded_a.shape == (3, 3)


def test_load_chain_atoms_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        _load_chain_atoms("/nonexistent/path/does_not_exist.pdb", "A", "CA")


def test_load_chain_atoms_missing_chain_raises(tmp_path):
    pdb_path = _write_pdb(tmp_path, "one_chain.pdb", {"A": [[0.0, 0.0, 0.0]]})
    with pytest.raises(ValueError, match="no atoms found"):
        _load_chain_atoms(pdb_path, "Z", "CA")


# --------------------------------------------------------------------------
# _pairwise_clash_overlap sanity
# --------------------------------------------------------------------------


def test_pairwise_clash_overlap_zero_when_far_apart():
    a = torch.zeros(1, 2, 3)
    b = torch.zeros(1, 2, 3) + torch.tensor([100.0, 0.0, 0.0])
    assert float(_pairwise_clash_overlap(a, b, clash_distance=4.0)) == 0.0


def test_pairwise_clash_overlap_positive_when_overlapping():
    a = torch.zeros(1, 1, 3)
    b = torch.zeros(1, 1, 3) + torch.tensor([1.0, 0.0, 0.0])  # 1 A apart, cutoff 4 A
    val = float(_pairwise_clash_overlap(a, b, clash_distance=4.0))
    assert val == pytest.approx((4.0 - 1.0) ** 2, abs=1e-4)


# --------------------------------------------------------------------------
# compute(): orientation controls overlap, value matches an independent
# recomputation of the same transform + clash formula
# --------------------------------------------------------------------------


def _build_two_motif_potential(motif1_rotation, chainA_ref, chainB_ref, shape_fn=_tetra):
    """Two 4-atom motifs 8 A apart (COMs), motif0 always unrotated, motif1's
    rotation is the variable under test. Chains are injected directly into the
    lazy cache so this is a pure geometry/transform test, no file I/O.
    """
    ref0 = shape_fn([0.0, 0.0, 0.0])
    ref1 = shape_fn([0.0, 0.0, 0.0])  # same local shape, doesn't matter where
    current_com0 = torch.tensor([0.0, 0.0, 0.0])
    current_com1 = torch.tensor([8.0, 0.0, 0.0])

    current0 = _rigid_transform(ref0, ref0.mean(0), IDENTITY, current_com0)
    current1 = _rigid_transform(ref1, ref1.mean(0), motif1_rotation, current_com1)

    xyz, masks, metadata = _two_motif_scene(ref0, ref1, current0, current1)

    pot = MinimalOverlapPotential(
        weight=1.0,
        motif_chains=[
            {"motif_index": 0, "chain_pdb": "unused0.pdb", "chain_id": "R0"},
            {"motif_index": 1, "chain_pdb": "unused1.pdb", "chain_id": "R1"},
        ],
        clash_distance=4.0,
    )
    pot._chain_ref_xyz[0] = chainA_ref
    pot._chain_ref_xyz[1] = chainB_ref
    return pot, xyz, masks, metadata, ref0, ref1, current_com0, current_com1


def test_reference_orientation_does_not_clash():
    chainA_ref = _tetra([2.5, 0.0, 0.0])  # motif0's chain points toward +x (away from origin)
    chainB_ref = _tetra([2.5, 0.0, 0.0])  # motif1's chain also points +x -> away from motif0
    pot, xyz, masks, metadata, *_ = _build_two_motif_potential(IDENTITY, chainA_ref, chainB_ref)
    value = pot.compute(xyz, masks, metadata)
    assert float(value) == pytest.approx(0.0, abs=1e-4)


def test_flipped_orientation_clashes_and_is_worse_than_reference():
    chainA_ref = _tetra([2.5, 0.0, 0.0])
    chainB_ref = _tetra([2.5, 0.0, 0.0])

    pot_ref, xyz_ref, masks_ref, meta_ref, *_ = _build_two_motif_potential(
        IDENTITY, chainA_ref, chainB_ref
    )
    value_ref = float(pot_ref.compute(xyz_ref, masks_ref, meta_ref))

    pot_flip, xyz_flip, masks_flip, meta_flip, *_ = _build_two_motif_potential(
        FLIP_Z, chainA_ref, chainB_ref
    )
    value_flip = float(pot_flip.compute(xyz_flip, masks_flip, meta_flip))

    # Flipping motif1 180 deg turns its chain to point back toward motif0's
    # chain -> strictly more overlap (more negative potential) than the
    # reference (non-clashing) orientation.
    assert value_flip < value_ref - 1e-6
    assert value_ref == pytest.approx(0.0, abs=1e-4)
    assert value_flip < 0.0


def test_compute_matches_independent_transform_recomputation():
    chainA_ref = _tetra([2.5, 0.0, 0.0])
    chainB_ref = _tetra([2.5, 0.0, 0.0])
    pot, xyz, masks, metadata, ref0, ref1, com0, com1 = _build_two_motif_potential(
        FLIP_Z, chainA_ref, chainB_ref
    )
    value = float(pot.compute(xyz, masks, metadata))

    # Independently recompute expected transported chain positions and the
    # expected potential value, without reusing any of the class's internals
    # beyond the two shared primitives it documents itself as built from.
    # compute() sums THREE clash terms: chain0-vs-chain1, chain0-vs-motif1's
    # own atoms, and chain1-vs-motif0's own atoms (protein_atom_filter='CA'
    # matches every atom here, so "the rest of the structure" for each chain
    # is exactly the other motif's own 4 atoms) -- replicate all three.
    R0 = _kabsch_ref_to_current_rotation(ref0, xyz[:, :4, :])[0]
    R1 = _kabsch_ref_to_current_rotation(ref1, xyz[:, 4:8, :])[0]
    chainA_expected = ((chainA_ref - ref0.mean(0)) @ R0 + com0).unsqueeze(0)
    chainB_expected = ((chainB_ref - ref1.mean(0)) @ R1 + com1).unsqueeze(0)
    motif0_atoms = xyz[:, :4, :]
    motif1_atoms = xyz[:, 4:8, :]
    expected_overlap = float(
        _pairwise_clash_overlap(chainA_expected, chainB_expected, clash_distance=4.0)
        + _pairwise_clash_overlap(chainA_expected, motif1_atoms, clash_distance=4.0)
        + _pairwise_clash_overlap(chainB_expected, motif0_atoms, clash_distance=4.0)
    )
    assert value == pytest.approx(-expected_overlap, abs=1e-4)


# --------------------------------------------------------------------------
# align=True: automatic offline Kabsch pre-alignment onto the design's own
# reference frame, from a chain living in a *different* structure.
# --------------------------------------------------------------------------


def test_align_true_recovers_chain_in_design_reference_frame(tmp_path):
    # The design's own motif reference (frame "D") and the TRUE, ground-truth
    # receptor point cloud in that same design frame -- what align=True should
    # recover, given only the file below (whose frame is unrelated to "D").
    ref_xyz_i = _tetra([0.0, 0.0, 0.0])
    receptor_design_frame = _tetra([5.0, 0.0, 0.0])

    # A *different* structure ("crystal structure") holding the same physical
    # assembly (motif + receptor) in an arbitrary other pose: apply one shared
    # rigid transform to both, exactly as a real alternate crystal form would.
    known_rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )  # 90 deg about z, a proper rotation
    known_translation = torch.tensor([100.0, -50.0, 7.0])
    motif_in_file = ref_xyz_i @ known_rotation + known_translation
    receptor_in_file = receptor_design_frame @ known_rotation + known_translation

    pdb_path = _write_pdb(
        tmp_path,
        "external_complex.pdb",
        {"M": motif_in_file.tolist(), "R": receptor_in_file.tolist()},
    )

    pot = MinimalOverlapPotential(
        weight=1.0,
        motif_chains=[
            {
                "motif_index": 0,
                "chain_pdb": pdb_path,
                "chain_id": "R",
                "align": True,
                "align_chain_id": "M",
            }
        ],
    )
    resolved = pot._resolve_chain_ref_xyz(0, pot.motif_chains[0], ref_xyz_i, "cpu", torch.float32)

    # The offline Kabsch fit should invert known_rotation/known_translation
    # and recover the receptor's true design-frame coordinates.
    assert torch.allclose(resolved, receptor_design_frame, atol=1e-3)


def test_align_true_atom_count_mismatch_raises(tmp_path):
    ref_xyz_i = _tetra([0.0, 0.0, 0.0])  # 4 atoms
    pdb_path = _write_pdb(
        tmp_path,
        "mismatch.pdb",
        {"M": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], "R": [[5.0, 0.0, 0.0]]},  # only 2 atoms
    )
    pot = MinimalOverlapPotential(
        weight=1.0,
        motif_chains=[
            {
                "motif_index": 0,
                "chain_pdb": pdb_path,
                "chain_id": "R",
                "align": True,
                "align_chain_id": "M",
            }
        ],
    )
    with pytest.raises(ValueError, match="atoms"):
        pot._resolve_chain_ref_xyz(0, pot.motif_chains[0], ref_xyz_i, "cpu", torch.float32)


# --------------------------------------------------------------------------
# Edge cases: skip gracefully rather than raising mid-run
# --------------------------------------------------------------------------


def test_out_of_range_motif_index_returns_zero():
    ref0 = _tetra([0.0, 0.0, 0.0])
    ref1 = _tetra([0.0, 0.0, 0.0])
    current0 = _rigid_transform(ref0, ref0.mean(0), IDENTITY, torch.zeros(3))
    current1 = _rigid_transform(ref1, ref1.mean(0), IDENTITY, torch.tensor([8.0, 0.0, 0.0]))
    xyz, masks, metadata = _two_motif_scene(ref0, ref1, current0, current1)

    pot = MinimalOverlapPotential(
        weight=1.0,
        motif_chains=[{"motif_index": 5, "chain_pdb": "unused.pdb", "chain_id": "R"}],
    )
    value = pot.compute(xyz, masks, metadata)
    assert float(value) == 0.0
    assert pot.skip_reason == "motif_chains_0_index_out_of_range"


def test_no_motif_blocks_returns_zero():
    xyz = torch.zeros(1, 4, 3)
    masks = {
        "motif_atom_mask": torch.zeros(4, dtype=torch.bool),
        "real_atom_mask": torch.ones(4, dtype=torch.bool),
        "ca_atom_mask": torch.ones(4, dtype=torch.bool),
    }
    metadata = {"atom_to_token_map": torch.arange(4)}
    pot = MinimalOverlapPotential(
        weight=1.0,
        motif_chains=[{"motif_index": 0, "chain_pdb": "unused.pdb", "chain_id": "R"}],
    )
    value = pot.compute(xyz, masks, metadata)
    assert float(value) == 0.0
    assert pot.skip_reason == "no_motif_blocks"


# --------------------------------------------------------------------------
# Rotation-only gradient ascent actually reduces the clash
# --------------------------------------------------------------------------


def _rigidify(xyz, ref0, ref1):
    """Snap each motif block back onto an exact rigid image of its reference
    shape (Kabsch-fit rotation + current COM). Mirrors what the real pipeline's
    `floating_motif_project` does after every guidance step (see MotifDistance's
    docstring: "a subsequent Kabsch projection can then restore the original
    rigid all-atom motif geometry") -- without it, a rotation-FIELD update
    (a first-order/linearised approximation of a true rotation, applied here as
    a literal position delta) accumulates non-rigid shape distortion over many
    steps, eventually collapsing a block into a near-degenerate (singular)
    point cloud.
    """
    new_xyz = xyz.clone()
    for ref, sl in ((ref0, slice(0, 4)), (ref1, slice(4, 8))):
        block = xyz[:, sl, :]
        R = _kabsch_ref_to_current_rotation(ref, block)
        com = block.mean(dim=1, keepdim=True)
        rigid = (ref - ref.mean(0)).unsqueeze(0) @ R + com
        new_xyz[:, sl, :] = rigid
    return new_xyz


def test_gradient_ascent_reduces_overlap():
    # Uses _asymmetric_tetra for the motif shapes (not _tetra) -- this test is
    # the only one that differentiates through the Kabsch fit, and _tetra's
    # symmetric Gram spectrum triggers SVD-backward NaNs (see
    # _asymmetric_tetra's docstring).
    chainA_ref = _tetra([2.5, 0.0, 0.0])
    chainB_ref = _tetra([2.5, 0.0, 0.0])
    pot, xyz, masks, metadata, ref0, ref1, _com0, _com1 = _build_two_motif_potential(
        FLIP_Z, chainA_ref, chainB_ref, shape_fn=_asymmetric_tetra
    )

    xyz = xyz.clone().requires_grad_(True)
    values = []
    # The real guidance pipeline RMS-clips each step (guide_clip_rms), decays
    # guide_scale over the trajectory (guide_decay), AND re-rigidifies motifs
    # after each step (floating_motif_project) -- mirror all three, otherwise
    # the rotation-FIELD update (a linear approximation applied as a literal
    # position delta) slowly distorts the tetrahedra into a degenerate shape.
    initial_step_rms = 0.05
    decay = 0.92
    guide_mask = pot.guide_atom_mask(masks, metadata, xyz.device)
    for step in range(40):
        value = pot.compute(xyz, masks, metadata)
        values.append(float(value.detach()))
        (grad,) = torch.autograd.grad(value, xyz)
        atom_grad = grad * guide_mask[None, :, None].to(dtype=grad.dtype)
        update = pot.transform_atom_gradient(atom_grad, masks, metadata, xyz.detach())
        rms = update[:, guide_mask, :].pow(2).mean().sqrt().clamp_min(1e-8)
        target_step_rms = initial_step_rms * (decay**step)
        normalized_update = update * (target_step_rms / rms)
        with torch.no_grad():
            xyz = _rigidify(xyz + normalized_update, ref0, ref1).requires_grad_(True)

    # Gradient ASCENT on `value` (= -weight*overlap) must drive overlap down:
    # the sequence of `value`s should trend up (toward 0), ending well above
    # where it started, and each step must genuinely be legal (no NaNs/Infs).
    assert all(torch.isfinite(torch.tensor(values)))
    assert values[-1] > values[0] + 1e-3
    # Rotation-only projection must not have translated either motif's COM.
    com0_before = xyz[:, :4, :].mean(dim=1)
    assert torch.allclose(com0_before, torch.tensor([[0.0, 0.0, 0.0]]), atol=1e-3)

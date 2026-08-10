"""Tests for TargetAnchorDistance (rfd3.potentials.potentials) -- a harmonic
restraint on the distance between one specific atom on each of two RECEPTOR
chains (e.g. the membrane-proximal residue of each ectodomain), transported
every step by the same per-motif Kabsch rotation that tracks the attached
floating motif's own atoms.

Unlike test_potentials.py, this file uses biotite (to build small synthetic
chain_pdb files on disk), mirroring test_receptor_grafting.py's conventions.
"""

import numpy as np
import pytest
import torch
import biotite.structure as struc
import biotite.structure.io as strucio

from rfd3.potentials.config import PotentialsConfig
from rfd3.potentials.masks import build_masks
from rfd3.potentials.potentials import TargetAnchorDistance, _project_gradient_to_rigid_body

# ─── shared helpers ──────────────────────────────────────────────────────────


def _random_rotation(rng):
    a = rng.normal(size=(3, 3))
    q, r = np.linalg.qr(a)
    d = np.sign(np.diag(r))
    q = q * d
    if np.linalg.det(q) < 0:
        q[:, -1] *= -1
    return q


def _save_chain_pdb(tmp_path, name, entries):
    """entries: list of (chain_id, res_id, atom_name, coord)."""
    n = len(entries)
    arr = struc.AtomArray(n)
    arr.coord = np.asarray([e[3] for e in entries], dtype=np.float32)
    arr.chain_id = np.array([e[0] for e in entries])
    arr.res_id = np.array([e[1] for e in entries])
    arr.res_name = np.full(n, "GLY")
    arr.atom_name = np.array([e[2] for e in entries])
    arr.element = np.full(n, "C")
    arr.hetero = np.zeros(n, dtype=bool)
    path = tmp_path / f"{name}.pdb"
    strucio.save_structure(str(path), arr)
    return str(path)


def _two_motif_setup():
    """Two 4-atom motif blocks (a token gap between them creates two distinct
    blocks), mirroring test_potentials.py's own _make_radial_orientation_setup
    convention (metadata["motif_pos"] supplies the reference geometry)."""
    n_atoms = 8
    f = {
        "atom_to_token_map": torch.tensor([0, 1, 2, 3, 5, 6, 7, 8]),
        "is_motif_atom_with_fixed_coord": torch.ones(n_atoms, dtype=torch.bool),
        "is_virtual": torch.zeros(n_atoms, dtype=torch.bool),
        "is_ca": torch.ones(n_atoms, dtype=torch.bool),
        "is_backbone": torch.ones(n_atoms, dtype=torch.bool),
    }
    config = PotentialsConfig(
        enabled=True,
        include_atoms="real",
        exclude_fixed_atoms=False,
        exclude_virtual_atoms=True,
    )
    masks = build_masks(f, config)

    motif0_ref = torch.tensor(
        [[-5.0, 1.0, 0.0], [-5.0, -1.0, 0.0], [-5.0, 0.0, 1.0], [-5.0, 0.0, -1.0]]
    )
    motif1_ref = torch.tensor(
        [[5.0, 1.0, 0.0], [5.0, -1.0, 0.0], [5.0, 0.0, 1.0], [5.0, 0.0, -1.0]]
    )
    ref_pos = torch.cat([motif0_ref, motif1_ref], dim=0)
    metadata = {"atom_to_token_map": f["atom_to_token_map"], "motif_pos": ref_pos}
    return masks, metadata, motif0_ref, motif1_ref


# ─── validation ──────────────────────────────────────────────────────────────


@pytest.mark.fast
def test_target_anchor_distance_requires_exactly_two_motif_chains():
    with pytest.raises(ValueError, match="exactly 2"):
        TargetAnchorDistance(weight=1.0, motif_chains=[{"motif_index": 0}])


@pytest.mark.fast
def test_target_anchor_distance_requires_target_residue():
    with pytest.raises(ValueError, match="target_residue"):
        TargetAnchorDistance(
            weight=1.0,
            motif_chains=[
                {"motif_index": 0, "chain_pdb": "x.pdb", "chain_id": "B"},
                {"motif_index": 1, "chain_pdb": "x.pdb", "chain_id": "C", "target_residue": 1},
            ],
        )


@pytest.mark.fast
def test_target_anchor_distance_align_requires_align_chain_id():
    with pytest.raises(ValueError, match="align_chain_id"):
        TargetAnchorDistance(
            weight=1.0,
            motif_chains=[
                {
                    "motif_index": 0, "chain_pdb": "x.pdb", "chain_id": "B",
                    "target_residue": 1, "align": True,
                },
                {"motif_index": 1, "chain_pdb": "x.pdb", "chain_id": "C", "target_residue": 1},
            ],
        )


# ─── correctness: transport via the motif's own Kabsch fit ─────────────────


@pytest.mark.fast
def test_target_anchor_distance_transports_receptor_anchor_with_motif(tmp_path):
    """The receptor anchor for each motif must move with that motif's own
    per-step rigid transform: anchor_current = (anchor_ref - motif_ref_com) @
    R + motif_current_com -- exactly mirroring MinimalOverlapPotential's own
    per-chain transport, just for one point instead of a full cloud."""
    rng = np.random.default_rng(0)
    masks, metadata, motif0_ref, motif1_ref = _two_motif_setup()

    r0 = _random_rotation(rng)
    r1 = _random_rotation(rng)
    com0_new = np.array([50.0, -20.0, 5.0])
    com1_new = np.array([-30.0, 40.0, -10.0])
    motif0_cur = (motif0_ref.numpy() - motif0_ref.numpy().mean(axis=0)) @ r0 + com0_new
    motif1_cur = (motif1_ref.numpy() - motif1_ref.numpy().mean(axis=0)) @ r1 + com1_new
    xyz = torch.as_tensor(
        np.concatenate([motif0_cur, motif1_cur], axis=0), dtype=torch.float32
    ).unsqueeze(0)

    # motif0's receptor: anchor sits at a fixed lever-arm offset from motif0's
    # own reference COM, no align needed (already in the design's own frame).
    lever0 = np.array([3.0, 0.0, 0.0])
    anchor0_ref_frame = motif0_ref.numpy().mean(axis=0) + lever0
    chain_pdb0 = _save_chain_pdb(
        tmp_path, "receptor0", [("B", 500, "CA", anchor0_ref_frame)]
    )

    # motif1's receptor: anchor + an align_chain_id copy of motif1's own
    # geometry, both expressed in a DIFFERENT raw frame (rigid shift+rotation
    # away from the design's own reference frame) -- align=True must recover
    # the correct anchor position regardless.
    lever1 = np.array([0.0, 4.0, 0.0])
    anchor1_ref_frame = motif1_ref.numpy().mean(axis=0) + lever1
    r_shift = _random_rotation(rng)
    t_shift = np.array([100.0, -50.0, 25.0])
    motif1_com = motif1_ref.numpy().mean(axis=0)
    raw_motif1 = (motif1_ref.numpy() - motif1_com) @ r_shift + motif1_com + t_shift
    raw_anchor1 = (anchor1_ref_frame - motif1_com) @ r_shift + motif1_com + t_shift
    chain_pdb1_entries = [("C", 610, "CA", raw_anchor1)] + [
        ("M", i + 1, "CA", raw_motif1[i]) for i in range(4)
    ]
    chain_pdb1 = _save_chain_pdb(tmp_path, "receptor1", chain_pdb1_entries)

    pot = TargetAnchorDistance(
        weight=1.0,
        target_distance=0.0,
        motif_chains=[
            {"motif_index": 0, "chain_pdb": chain_pdb0, "chain_id": "B", "target_residue": 500},
            {
                "motif_index": 1, "chain_pdb": chain_pdb1, "chain_id": "C", "target_residue": 610,
                "align": True, "align_chain_id": "M", "align_atom_selection": "CA",
            },
        ],
    )

    value = pot.compute(xyz, masks, metadata)
    assert pot.skip_reason is None

    expected_anchor0 = (lever0) @ r0 + com0_new
    expected_anchor1 = (lever1) @ r1 + com1_new
    expected_dist = np.linalg.norm(expected_anchor0 - expected_anchor1)
    expected_value = -1.0 * (expected_dist - 0.0) ** 2
    assert torch.allclose(value, torch.tensor(float(expected_value)), atol=1e-2), (
        value.item(), expected_value
    )


@pytest.mark.fast
def test_target_anchor_distance_align_false_is_wrong_when_frame_shifted(tmp_path):
    """Sanity guard: with align=False, a raw anchor point from a genuinely
    shifted frame must NOT happen to give the same (correct) answer -- proves
    the align=True test above is actually exercising the correction, not
    passing by coincidence."""
    rng = np.random.default_rng(1)
    masks, metadata, motif0_ref, motif1_ref = _two_motif_setup()

    com0_new = np.array([0.0, 0.0, 0.0])
    com1_new = np.array([20.0, 0.0, 0.0])
    xyz = torch.cat([motif0_ref, motif1_ref + torch.tensor([15.0, 0.0, 0.0])], dim=0).unsqueeze(0)

    anchor0_ref_frame = motif0_ref.numpy().mean(axis=0) + np.array([3.0, 0.0, 0.0])
    chain_pdb0 = _save_chain_pdb(tmp_path, "r0", [("B", 1, "CA", anchor0_ref_frame)])

    t_shift = np.array([100.0, -50.0, 25.0])  # large, deliberately never corrected
    anchor1_raw_shifted = motif1_ref.numpy().mean(axis=0) + np.array([0.0, 4.0, 0.0]) + t_shift
    chain_pdb1 = _save_chain_pdb(tmp_path, "r1", [("C", 1, "CA", anchor1_raw_shifted)])

    pot = TargetAnchorDistance(
        weight=1.0,
        target_distance=0.0,
        motif_chains=[
            {"motif_index": 0, "chain_pdb": chain_pdb0, "chain_id": "B", "target_residue": 1},
            {"motif_index": 1, "chain_pdb": chain_pdb1, "chain_id": "C", "target_residue": 1, "align": False},
        ],
    )
    value = pot.compute(xyz, masks, metadata)
    # the t_shift is huge and never corrected -- the resulting distance should
    # be wildly larger than any plausible "correct" answer (~tens of A).
    implied_dist = (-value.item()) ** 0.5
    assert implied_dist > 50.0, f"expected align=False to be measurably wrong, got dist={implied_dist}"


# ─── gradient projection: translation / rotation degrees of freedom ────────


@pytest.mark.fast
def test_target_anchor_distance_translation_only_moves_block_rigidly(tmp_path):
    """allow_rotation=False must give every atom in a motif block the exact
    same displacement (pure rigid translation)."""
    masks, metadata, motif0_ref, motif1_ref = _two_motif_setup()
    # A nontrivial (non-identity) rigid transform per block -- using the
    # reference geometry as-is for "current" would make the Kabsch fit's
    # covariance matrix exactly symmetric (repeated singular values), a known
    # SVD gradient singularity (NaN) unrelated to TargetAnchorDistance itself.
    rng = np.random.default_rng(2)
    r0, r1 = _random_rotation(rng), _random_rotation(rng)
    motif0_cur = (motif0_ref.numpy() - motif0_ref.numpy().mean(axis=0)) @ r0 + np.array([1.0, 2.0, 3.0])
    motif1_cur = (motif1_ref.numpy() - motif1_ref.numpy().mean(axis=0)) @ r1 + np.array([-2.0, 1.0, 0.0])
    xyz = torch.as_tensor(
        np.concatenate([motif0_cur, motif1_cur], axis=0), dtype=torch.float32
    ).unsqueeze(0).requires_grad_(True)

    lever0 = np.array([3.0, 0.0, 0.0])
    anchor0 = motif0_ref.numpy().mean(axis=0) + lever0
    lever1 = np.array([0.0, 0.0, 3.0])
    anchor1 = motif1_ref.numpy().mean(axis=0) + lever1
    chain_pdb0 = _save_chain_pdb(tmp_path, "ra", [("B", 1, "CA", anchor0)])
    chain_pdb1 = _save_chain_pdb(tmp_path, "rb", [("C", 1, "CA", anchor1)])

    pot = TargetAnchorDistance(
        weight=1.0,
        target_distance=5.0,
        allow_translation=True,
        allow_rotation=False,
        motif_chains=[
            {"motif_index": 0, "chain_pdb": chain_pdb0, "chain_id": "B", "target_residue": 1},
            {"motif_index": 1, "chain_pdb": chain_pdb1, "chain_id": "C", "target_residue": 1},
        ],
    )
    value = pot.compute(xyz, masks, metadata)
    value.backward()
    atom_grad = xyz.grad.clone()
    transformed = pot.transform_atom_gradient(atom_grad, masks, metadata, xyz.detach())

    assert not torch.isnan(transformed).any()
    block0 = transformed[0, :4, :]
    block1 = transformed[0, 4:, :]
    for i in range(1, 4):
        assert torch.allclose(block0[0], block0[i], atol=1e-6)
        assert torch.allclose(block1[0], block1[i], atol=1e-6)
    assert transformed.abs().sum().item() > 0.0


@pytest.mark.fast
def test_target_anchor_distance_allow_rotation_gives_nonuniform_displacement(tmp_path):
    """allow_rotation=True should let atoms within a block move differently
    from each other (translation-at-COM + torque about COM)."""
    masks, metadata, motif0_ref, motif1_ref = _two_motif_setup()
    # Nontrivial transform, not identity -- see comment in the translation-only
    # test above (identity current==reference degenerates the Kabsch SVD
    # gradient to NaN, which would make the "not equal" assertion below pass
    # spuriously rather than for the intended reason).
    rng = np.random.default_rng(3)
    r0, r1 = _random_rotation(rng), _random_rotation(rng)
    motif0_cur = (motif0_ref.numpy() - motif0_ref.numpy().mean(axis=0)) @ r0 + np.array([4.0, -1.0, 2.0])
    motif1_cur = (motif1_ref.numpy() - motif1_ref.numpy().mean(axis=0)) @ r1 + np.array([0.0, 3.0, -2.0])
    xyz = torch.as_tensor(
        np.concatenate([motif0_cur, motif1_cur], axis=0), dtype=torch.float32
    ).unsqueeze(0).requires_grad_(True)

    lever0 = np.array([3.0, 2.0, 0.0])  # off-axis lever so rotation has an effect
    anchor0 = motif0_ref.numpy().mean(axis=0) + lever0
    anchor1 = motif1_ref.numpy().mean(axis=0) + np.array([0.0, 0.0, 3.0])
    chain_pdb0 = _save_chain_pdb(tmp_path, "rc", [("B", 1, "CA", anchor0)])
    chain_pdb1 = _save_chain_pdb(tmp_path, "rd", [("C", 1, "CA", anchor1)])

    pot = TargetAnchorDistance(
        weight=1.0,
        target_distance=5.0,
        allow_translation=True,
        allow_rotation=True,
        motif_chains=[
            {"motif_index": 0, "chain_pdb": chain_pdb0, "chain_id": "B", "target_residue": 1},
            {"motif_index": 1, "chain_pdb": chain_pdb1, "chain_id": "C", "target_residue": 1},
        ],
    )
    value = pot.compute(xyz, masks, metadata)
    value.backward()
    atom_grad = xyz.grad.clone()
    transformed = pot.transform_atom_gradient(atom_grad, masks, metadata, xyz.detach())

    assert not torch.isnan(transformed).any()
    block0 = transformed[0, :4, :]
    assert not torch.allclose(block0[0], block0[2], atol=1e-6)


@pytest.mark.fast
def test_target_anchor_distance_guide_atom_mask_covers_both_blocks(tmp_path):
    masks, metadata, motif0_ref, motif1_ref = _two_motif_setup()
    chain_pdb0 = _save_chain_pdb(tmp_path, "re", [("B", 1, "CA", np.zeros(3))])
    chain_pdb1 = _save_chain_pdb(tmp_path, "rf", [("C", 1, "CA", np.zeros(3))])
    pot = TargetAnchorDistance(
        weight=1.0,
        target_distance=5.0,
        motif_chains=[
            {"motif_index": 0, "chain_pdb": chain_pdb0, "chain_id": "B", "target_residue": 1},
            {"motif_index": 1, "chain_pdb": chain_pdb1, "chain_id": "C", "target_residue": 1},
        ],
    )
    guide_mask = pot.guide_atom_mask(masks, metadata, torch.device("cpu"))
    assert guide_mask.sum().item() == 8
    assert torch.all(guide_mask)

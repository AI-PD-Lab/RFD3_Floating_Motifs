"""Tests for MotifPairAxisDot's motif_chains input mode -- an alternative to
supplying motif_axes directly: two residues per motif (receptor_residues)
define the axis vector, resolved internally from chain_pdb (optionally
aligned into the motif's own body frame), instead of a hand-computed
[x, y, z] unit vector.

Uses biotite (to build small synthetic chain_pdb files), like
test_target_anchor_distance.py.
"""

import numpy as np
import pytest
import torch
import biotite.structure as struc
import biotite.structure.io as strucio

from rfd3.potentials.config import PotentialsConfig
from rfd3.potentials.masks import build_masks
from rfd3.potentials.potentials import MotifPairAxisDot


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
    """Two 4-atom motif blocks (mirrors test_target_anchor_distance.py's own
    _two_motif_setup): a token gap creates two distinct blocks, and
    metadata["motif_pos"] supplies the reference geometry."""
    n_atoms = 8
    f = {
        "atom_to_token_map": torch.tensor([0, 1, 2, 3, 5, 6, 7, 8]),
        "is_motif_atom_with_fixed_coord": torch.ones(n_atoms, dtype=torch.bool),
        "is_virtual": torch.zeros(n_atoms, dtype=torch.bool),
        "is_ca": torch.ones(n_atoms, dtype=torch.bool),
        "is_backbone": torch.ones(n_atoms, dtype=torch.bool),
    }
    config = PotentialsConfig(
        enabled=True, include_atoms="real", exclude_fixed_atoms=False, exclude_virtual_atoms=True,
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


@pytest.mark.fast
def test_motif_pair_axis_dot_requires_exactly_one_input_mode():
    with pytest.raises(ValueError, match="exactly one"):
        MotifPairAxisDot(weight=1.0)
    with pytest.raises(ValueError, match="exactly one"):
        MotifPairAxisDot(
            weight=1.0,
            motif_axes=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            motif_chains=[{"motif_index": 0}, {"motif_index": 1}],
        )


@pytest.mark.fast
def test_motif_pair_axis_dot_motif_chains_derives_motif_i_j():
    """motif_i/motif_j are redundant with motif_chains' own motif_index
    values and must not need to be given -- and if a NON-default pair (e.g.
    swapped/other block indices) is used, derivation must pick THOSE values
    up, not silently fall back to 0/1."""
    pot = MotifPairAxisDot(
        weight=1.0,
        motif_chains=[
            {"motif_index": 3, "chain_pdb": "x.pdb", "chain_id": "B", "receptor_residues": [1, 2]},
            {"motif_index": 5, "chain_pdb": "x.pdb", "chain_id": "C", "receptor_residues": [1, 2]},
        ],
    )
    assert pot.motif_i == 3
    assert pot.motif_j == 5


@pytest.mark.fast
def test_motif_pair_axis_dot_motif_chains_rejects_mismatched_explicit_motif_i_j():
    with pytest.raises(ValueError, match="motif_index values"):
        MotifPairAxisDot(
            weight=1.0,
            motif_i=0,
            motif_j=99,  # does not match either motif_chains entry
            motif_chains=[
                {"motif_index": 0, "chain_pdb": "x.pdb", "chain_id": "B", "receptor_residues": [1, 2]},
                {"motif_index": 1, "chain_pdb": "x.pdb", "chain_id": "C", "receptor_residues": [1, 2]},
            ],
        )


@pytest.mark.fast
def test_motif_pair_axis_dot_motif_chains_requires_receptor_residues():
    with pytest.raises(ValueError, match="receptor_residues"):
        MotifPairAxisDot(
            weight=1.0,
            motif_chains=[
                {"motif_index": 0, "chain_pdb": "x.pdb", "chain_id": "B"},
                {"motif_index": 1, "chain_pdb": "x.pdb", "chain_id": "C", "receptor_residues": [1, 2]},
            ],
        )


@pytest.mark.fast
def test_motif_pair_axis_dot_motif_chains_matches_equivalent_motif_axes(tmp_path):
    """Two residues (no align needed -- raw frame == body frame here) must
    give the same dot product as directly supplying the equivalent unit
    vectors via motif_axes."""
    masks, metadata, motif0_ref, motif1_ref = _two_motif_setup()
    rng = np.random.default_rng(0)
    r0, r1 = _random_rotation(rng), _random_rotation(rng)
    motif0_cur = (motif0_ref.numpy() - motif0_ref.numpy().mean(axis=0)) @ r0 + np.array([1.0, 2.0, 3.0])
    motif1_cur = (motif1_ref.numpy() - motif1_ref.numpy().mean(axis=0)) @ r1 + np.array([-2.0, 1.0, 0.0])
    xyz = torch.as_tensor(
        np.concatenate([motif0_cur, motif1_cur], axis=0), dtype=torch.float32
    ).unsqueeze(0)

    raw_axis0 = np.array([3.0, 0.0, 0.0])  # proximal at motif0 ref COM, distal offset by this
    raw_axis1 = np.array([0.0, 4.0, 0.0])
    proximal0 = motif0_ref.numpy().mean(axis=0)
    distal0 = proximal0 + raw_axis0
    proximal1 = motif1_ref.numpy().mean(axis=0)
    distal1 = proximal1 + raw_axis1

    chain_pdb0 = _save_chain_pdb(
        tmp_path, "c0", [("B", 1, "CA", proximal0), ("B", 2, "CA", distal0)]
    )
    chain_pdb1 = _save_chain_pdb(
        tmp_path, "c1", [("C", 1, "CA", proximal1), ("C", 2, "CA", distal1)]
    )

    pot_chains = MotifPairAxisDot(
        weight=1.0,
        target_dot=1.0,
        motif_chains=[
            {"motif_index": 0, "chain_pdb": chain_pdb0, "chain_id": "B", "receptor_residues": [1, 2]},
            {"motif_index": 1, "chain_pdb": chain_pdb1, "chain_id": "C", "receptor_residues": [1, 2]},
        ],
    )
    pot_axes = MotifPairAxisDot(
        weight=1.0,
        target_dot=1.0,
        motif_axes=[list(raw_axis0 / np.linalg.norm(raw_axis0)), list(raw_axis1 / np.linalg.norm(raw_axis1))],
    )

    value_chains = pot_chains.compute(xyz, masks, metadata)
    value_axes = pot_axes.compute(xyz, masks, metadata)
    assert pot_chains.skip_reason is None
    assert torch.allclose(value_chains, value_axes, atol=1e-4)


@pytest.mark.fast
def test_motif_pair_axis_dot_motif_chains_align_recovers_frame_shift(tmp_path):
    """With align=True, a receptor axis measured in a rotated raw frame must
    still land correctly in the motif's own body frame -- align=False in the
    same scenario must NOT (proves this test exercises the correction)."""
    masks, metadata, motif0_ref, motif1_ref = _two_motif_setup()
    rng = np.random.default_rng(1)
    r0, r1 = _random_rotation(rng), _random_rotation(rng)
    motif0_cur = (motif0_ref.numpy() - motif0_ref.numpy().mean(axis=0)) @ r0 + np.array([0.0, 0.0, 0.0])
    motif1_cur = (motif1_ref.numpy() - motif1_ref.numpy().mean(axis=0)) @ r1 + np.array([20.0, 0.0, 0.0])
    xyz = torch.as_tensor(
        np.concatenate([motif0_cur, motif1_cur], axis=0), dtype=torch.float32
    ).unsqueeze(0)

    # motif0's receptor axis measured in a raw frame rotated away from the
    # design's own reference frame -- align=True must recover it via
    # align_chain_id ("A"), a copy of motif0's own reference geometry
    # expressed in that same raw frame.
    r_shift = _random_rotation(rng)
    motif0_com = motif0_ref.numpy().mean(axis=0)
    raw_motif0 = (motif0_ref.numpy() - motif0_com) @ r_shift + motif0_com
    true_axis0_body_frame = np.array([1.0, 1.0, 0.0])
    true_axis0_body_frame /= np.linalg.norm(true_axis0_body_frame)
    proximal0_body = motif0_com
    distal0_body = motif0_com + true_axis0_body_frame
    # raw-frame positions of proximal/distal: apply the SAME r_shift used for
    # the motif's own reference copy (align_chain_id) so both are expressed
    # in the one shared "raw" coordinate system.
    proximal0_raw = (proximal0_body - motif0_com) @ r_shift + motif0_com
    distal0_raw = (distal0_body - motif0_com) @ r_shift + motif0_com

    chain_pdb0_entries = [
        ("B", 1, "CA", proximal0_raw),
        ("B", 2, "CA", distal0_raw),
    ] + [("A", i + 1, "CA", raw_motif0[i]) for i in range(4)]
    chain_pdb0 = _save_chain_pdb(tmp_path, "c0align", chain_pdb0_entries)

    chain_pdb1 = _save_chain_pdb(
        tmp_path, "c1align",
        [("C", 1, "CA", motif1_ref.numpy().mean(axis=0)), ("C", 2, "CA", motif1_ref.numpy().mean(axis=0) + np.array([0, 0, 1.0]))],
    )

    pot_aligned = MotifPairAxisDot(
        weight=1.0,
        motif_chains=[
            {
                "motif_index": 0, "chain_pdb": chain_pdb0, "chain_id": "B",
                "receptor_residues": [1, 2], "align": True, "align_chain_id": "A",
                "align_atom_selection": "CA",
            },
            {"motif_index": 1, "chain_pdb": chain_pdb1, "chain_id": "C", "receptor_residues": [1, 2]},
        ],
    )
    axis0_aligned = pot_aligned._resolve_axis(0, 0, torch.as_tensor(motif0_ref.numpy(), dtype=torch.float32), "cpu", torch.float32)
    assert torch.allclose(
        axis0_aligned, torch.as_tensor(true_axis0_body_frame, dtype=torch.float32), atol=1e-3
    )

    pot_unaligned = MotifPairAxisDot(
        weight=1.0,
        motif_chains=[
            {"motif_index": 0, "chain_pdb": chain_pdb0, "chain_id": "B", "receptor_residues": [1, 2], "align": False},
            {"motif_index": 1, "chain_pdb": chain_pdb1, "chain_id": "C", "receptor_residues": [1, 2]},
        ],
    )
    axis0_unaligned = pot_unaligned._resolve_axis(0, 0, torch.as_tensor(motif0_ref.numpy(), dtype=torch.float32), "cpu", torch.float32)
    assert not torch.allclose(
        axis0_unaligned, torch.as_tensor(true_axis0_body_frame, dtype=torch.float32), atol=1e-2
    )

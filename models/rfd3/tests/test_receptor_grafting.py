"""Tests for rfd3.utils.receptor_grafting -- appending the receptor chain(s)
attached to floating motifs onto the final output structure, for runs that
configure `minimal_overlap`/`minimal_clashes` with
`motif_chains[*].include_in_output: true`.

These tests are self-contained: they build small synthetic AtomArrays and
PDB files rather than requiring a real design output or checkpoint.
"""

import numpy as np
import pytest
import biotite.structure as struc
import biotite.structure.io as strucio

import rfd3.utils.receptor_grafting as rg

# ─── shared helpers ──────────────────────────────────────────────────────────


def _random_rotation(rng):
    a = rng.normal(size=(3, 3))
    q, r = np.linalg.qr(a)
    d = np.sign(np.diag(r))
    q = q * d
    if np.linalg.det(q) < 0:
        q[:, -1] *= -1
    return q


def _make_atom_array(coords, chain_id, res_names, atom_names, elements):
    n = len(coords)
    arr = struc.AtomArray(n)
    arr.coord = np.asarray(coords, dtype=np.float32)
    arr.chain_id = np.full(n, chain_id)
    arr.res_id = np.arange(1, n + 1)
    arr.res_name = np.asarray(res_names)
    arr.atom_name = np.asarray(atom_names)
    arr.element = np.asarray(elements)
    arr.hetero = np.zeros(n, dtype=bool)
    return arr


def _build_receptor_pdb(tmp_path, name, coords):
    aa = _make_atom_array(
        coords, name, ["GLY"] * len(coords), ["CA"] * len(coords), ["C"] * len(coords)
    )
    path = tmp_path / f"fake_receptor_{name}.pdb"
    strucio.save_structure(str(path), aa)
    return str(path)


def _two_motif_scene(rng):
    """Mirrors the real EGFR-EGFR case: output chain "A" holds motif0
    (res_id 1-60), an auto-gap linker (res_id 61-72, no diffused_index_map
    entry), then motif1 (res_id 73-132). is_motif_atom_with_fixed_seq only
    covers a scattered hotspot subset (real behavior -- only a hotspot subset
    of a motif's sequence is held fixed, the rest can be redesigned) and
    floating_motif_reference_x/y/z is populated for the WHOLE chain, not just
    the motif (also real, confirmed behavior) -- neither should matter since
    block detection is driven entirely by diffused_index_map.
    """
    n_per_motif = 60
    motif0_ref = rng.normal(scale=5, size=(n_per_motif, 3))
    motif1_ref = rng.normal(scale=5, size=(n_per_motif, 3)) + np.array([100.0, 0, 0])

    r0 = _random_rotation(rng)
    r1 = _random_rotation(rng)
    com0_new = np.array([50.0, -20.0, 5.0])
    com1_new = np.array([-30.0, 40.0, -10.0])
    motif0_cur = (motif0_ref - motif0_ref.mean(axis=0)) @ r0 + com0_new
    motif1_cur = (motif1_ref - motif1_ref.mean(axis=0)) @ r1 + com1_new

    n_linker = 12
    linker_coords = rng.normal(scale=5.0, size=(n_linker, 3))
    linker_ref = rng.normal(scale=5.0, size=(n_linker, 3))  # populated but meaningless, as in reality

    n_total = 2 * n_per_motif + n_linker
    coords = np.concatenate([motif0_cur, linker_coords, motif1_cur], axis=0)
    ref_full = np.concatenate([motif0_ref, linker_ref, motif1_ref], axis=0)

    design_aa = struc.AtomArray(n_total)
    design_aa.coord = coords.astype(np.float32)
    design_aa.chain_id = np.full(n_total, "A")
    design_aa.res_id = np.arange(1, n_total + 1)
    design_aa.res_name = np.full(n_total, "GLY")
    design_aa.atom_name = np.full(n_total, "CA")
    design_aa.element = np.full(n_total, "C")
    design_aa.hetero = np.zeros(n_total, dtype=bool)

    # is_motif_atom_with_fixed_seq: scattered hotspot subset only -- real
    # behavior, deliberately NOT the whole block, to prove it's unused for
    # block detection.
    is_fixed_seq = np.zeros(n_total, dtype=bool)
    is_fixed_seq[0:n_per_motif:5] = True
    is_fixed_seq[n_per_motif + n_linker : n_per_motif + n_linker + n_per_motif : 5] = True
    design_aa.set_annotation("is_motif_atom_with_fixed_seq", is_fixed_seq)
    design_aa.set_annotation("is_motif_atom_with_fixed_coord", np.zeros(n_total, dtype=bool))
    design_aa.set_annotation("is_motif_atom_unindexed", np.zeros(n_total, dtype=bool))

    # floating_motif_reference_x/y/z populated for the WHOLE chain (real
    # behavior) -- values only need to be CORRECT at the true motif atoms;
    # the rest are never read once diffused_index_map picks the right ones.
    design_aa.set_annotation("floating_motif_reference_x", ref_full[:, 0])
    design_aa.set_annotation("floating_motif_reference_y", ref_full[:, 1])
    design_aa.set_annotation("floating_motif_reference_z", ref_full[:, 2])

    diffused_index_map = {}
    for i in range(1, n_per_motif + 1):
        diffused_index_map[f"A{i}"] = f"A{i}"
    motif1_out_start = n_per_motif + n_linker + 1
    for i in range(1, n_per_motif + 1):
        diffused_index_map[f"D{i}"] = f"A{motif1_out_start + i - 1}"

    return design_aa, diffused_index_map, motif0_ref, motif1_ref, com0_new, com1_new, r0, r1


# ─── graft_receptor_chains: block detection + rigid transport ───────────────


@pytest.mark.fast
def test_graft_receptor_chains_recovers_exact_rigid_transform(tmp_path):
    """Each receptor chain should land exactly where its own motif's Kabsch
    transform (reference -> current) says it should, recovered purely from
    `floating_motif_reference_x/y/z` + `diffused_index_map` -- no knowledge of
    R0/R1/com0_new/com1_new is threaded into graft_receptor_chains itself."""
    rng = np.random.default_rng(0)
    design_aa, diffused_index_map, motif0_ref, motif1_ref, com0_new, com1_new, r0, r1 = (
        _two_motif_scene(rng)
    )

    receptor0_ref = np.array(
        [[10.0, 0.0, 0.0], [11.0, 1.0, 0.0], [12.0, 0.0, 1.0], [13.0, -1.0, 0.5], [14.0, 0.5, -0.5]],
        dtype=np.float64,
    )
    receptor1_ref = np.array(
        [[-10.0, 2.0, 1.0], [-11.0, 3.0, 0.0], [-12.0, 1.0, 2.0], [-13.0, 0.0, 1.5], [-14.0, -0.5, 0.5]],
        dtype=np.float64,
    )
    receptor0_pdb = _build_receptor_pdb(tmp_path, "B", receptor0_ref)
    receptor1_pdb = _build_receptor_pdb(tmp_path, "C", receptor1_ref)

    guiding_potentials = [
        {
            "type": "minimal_overlap",
            "weight": 1.0,
            "motif_chains": [
                {"motif_index": 0, "chain_pdb": receptor0_pdb, "chain_id": "B", "include_in_output": True},
                {"motif_index": 1, "chain_pdb": receptor1_pdb, "chain_id": "C", "include_in_output": True},
            ],
        }
    ]

    result = rg.graft_receptor_chains(design_aa, guiding_potentials, diffused_index_map)

    new_chains = set(np.unique(result.chain_id)) - {"A"}
    assert new_chains == {"B", "C"}

    grafted_b = result[result.chain_id == "B"]
    grafted_c = result[result.chain_id == "C"]

    expected_b = (receptor0_ref - motif0_ref.mean(axis=0)) @ r0 + com0_new
    expected_c = (receptor1_ref - motif1_ref.mean(axis=0)) @ r1 + com1_new

    err_b = np.abs(np.asarray(grafted_b.coord, dtype=np.float64) - expected_b).max()
    err_c = np.abs(np.asarray(grafted_c.coord, dtype=np.float64) - expected_c).max()
    assert err_b < 1e-3, f"chain B transform mismatch, err {err_b}"
    assert err_c < 1e-3, f"chain C transform mismatch, err {err_c}"


@pytest.mark.fast
def test_graft_receptor_chains_noop_without_minimal_overlap(tmp_path):
    rng = np.random.default_rng(0)
    design_aa, diffused_index_map, *_ = _two_motif_scene(rng)
    out = rg.graft_receptor_chains(design_aa, [{"type": "motif_distance"}], diffused_index_map)
    assert out is design_aa


@pytest.mark.fast
def test_graft_receptor_chains_noop_without_reference_annotation(tmp_path):
    rng = np.random.default_rng(0)
    design_aa, diffused_index_map, *_ = _two_motif_scene(rng)
    receptor_pdb = _build_receptor_pdb(tmp_path, "B", np.zeros((5, 3)))
    guiding_potentials = [
        {
            "type": "minimal_overlap",
            "motif_chains": [
                {"motif_index": 0, "chain_pdb": receptor_pdb, "chain_id": "B", "include_in_output": True}
            ],
        }
    ]
    no_ref_aa = design_aa.copy()
    no_ref_aa.del_annotation("floating_motif_reference_x")
    out = rg.graft_receptor_chains(no_ref_aa, guiding_potentials, diffused_index_map)
    assert out is no_ref_aa


@pytest.mark.fast
def test_graft_receptor_chains_noop_without_diffused_index_map(tmp_path):
    rng = np.random.default_rng(0)
    design_aa, _diffused_index_map, *_ = _two_motif_scene(rng)
    receptor_pdb = _build_receptor_pdb(tmp_path, "B", np.zeros((5, 3)))
    guiding_potentials = [
        {
            "type": "minimal_overlap",
            "motif_chains": [
                {"motif_index": 0, "chain_pdb": receptor_pdb, "chain_id": "B", "include_in_output": True}
            ],
        }
    ]
    out = rg.graft_receptor_chains(design_aa, guiding_potentials, None)
    assert out is design_aa


# ─── align / align_chain_id offline registration ────────────────────────────


@pytest.mark.fast
def test_transport_receptor_chain_align_recovers_frame_shift(tmp_path):
    """Reproduces the real bug: chain_pdb's raw frame differs from the
    design's floating_motif_reference frame by a rigid offset (RFD3's
    infer_ori_strategy: com origin-shift recenters the whole PARSED
    structure, but never touches an externally, separately-loaded chain_pdb
    file). align=True + align_chain_id must recover and cancel that offset;
    align=False must NOT (proving the bug is real, not just that align=True
    happens to also work by coincidence)."""
    rng = np.random.default_rng(1)
    n_motif = 63
    n_receptor = 40

    # raw chain_pdb file: chain "A" (motif's own residues, unshifted/raw
    # frame) + chain "B" (the receptor, physically near chain A in this raw
    # frame -- e.g. a real bound-complex PDB).
    raw_chain_a = rng.normal(scale=8.0, size=(n_motif, 3))
    raw_chain_b = (
        raw_chain_a.mean(axis=0) + np.array([12.0, 0.0, 0.0]) + rng.normal(scale=3.0, size=(n_receptor, 3))
    )
    chain_a_aa = _make_atom_array(raw_chain_a, "A", ["GLY"] * n_motif, ["CA"] * n_motif, ["C"] * n_motif)
    chain_b_aa = _make_atom_array(
        raw_chain_b, "B", ["GLY"] * n_receptor, ["CA"] * n_receptor, ["C"] * n_receptor
    )
    combined = chain_a_aa + chain_b_aa
    chain_pdb_path = str(tmp_path / "fake_complex_align_test.pdb")
    strucio.save_structure(chain_pdb_path, combined)

    # simulate RFD3's internal origin-shift: design's own reference frame for
    # this motif block is raw_chain_a rigidly transformed by (r_shift, a big
    # translation) -- exactly what infer_ori_strategy: com does to the whole
    # parsed structure, while chain_pdb (loaded separately, raw) never sees it.
    r_shift = _random_rotation(rng)
    t_shift = np.array([5.453, -37.539, 1.374])  # matches the magnitude of the real measured offset
    align_com0 = raw_chain_a.mean(axis=0)
    ref_xyz = (raw_chain_a - align_com0) @ r_shift + align_com0 + t_shift

    # simulate the per-step diffusion transform: the design's own motif moves
    # from ref_xyz to some current position via another independent rigid
    # transform.
    r_step = _random_rotation(rng)
    cur_com = np.array([100.0, -60.0, 30.0])
    cur_xyz = (ref_xyz - ref_xyz.mean(axis=0)) @ r_step + cur_com

    # expected: raw_chain_b correctly registered into the design's frame via
    # align (r_shift, translation), then transported by the same per-step
    # transform (r_step, cur_com) that moved the motif from ref_xyz to cur_xyz.
    b_in_design_frame = (raw_chain_b - align_com0) @ r_shift + align_com0 + t_shift
    expected_b_final = (b_in_design_frame - ref_xyz.mean(axis=0)) @ r_step + cur_com

    spec_aligned = {
        "motif_index": 0,
        "chain_pdb": chain_pdb_path,
        "chain_id": "B",
        "align": True,
        "align_chain_id": "A",
        "align_atom_selection": "CA",
        "output_atom_selection": "CA",
    }
    receptor_aligned = rg._transport_receptor_chain(spec_aligned, ref_xyz, cur_xyz, ref_xyz)
    err_aligned = np.abs(np.asarray(receptor_aligned.coord, dtype=np.float64) - expected_b_final).max()
    assert err_aligned < 1e-2, f"align=True transform mismatch, err {err_aligned}"

    spec_unaligned = dict(spec_aligned)
    spec_unaligned["align"] = False
    receptor_unaligned = rg._transport_receptor_chain(spec_unaligned, ref_xyz, cur_xyz, None)
    err_unaligned = np.abs(np.asarray(receptor_unaligned.coord, dtype=np.float64) - expected_b_final).max()
    assert err_unaligned > 5.0, (
        "expected align=False to be measurably wrong in this frame-shifted scenario -- "
        f"got err {err_unaligned}, which would mean the bug this test guards isn't reproduced"
    )


# ─── heavy-atom selection excludes OXT (generic terminal-atom fix) ─────────


@pytest.mark.fast
def test_select_atoms_heavy_excludes_oxt(tmp_path):
    """OXT (the C-terminal carboxylate oxygen) is present on essentially any
    raw PDB's terminal residue but never modeled by RFD3's own fixed
    per-residue atom template. 'heavy' must exclude it so align_atom_selection
    stays 1:1-comparable with the design's own motif reference for any future
    receptor pair, without requiring hand-cleaned input files."""
    coords = np.array(
        [[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0], [3.0, 0, 0], [4.0, 0, 0], [5.0, 0, 0], [6.0, 0, 0]]
    )
    arr = struc.AtomArray(7)
    arr.coord = coords.astype(np.float32)
    arr.chain_id = np.full(7, "A")
    arr.res_id = np.array([1, 1, 1, 1, 1, 1, 1])
    arr.res_name = np.full(7, "SER")
    arr.atom_name = np.array(["N", "CA", "C", "O", "CB", "OG", "OXT"])
    arr.element = np.array(["N", "C", "C", "O", "C", "O", "O"])
    arr.hetero = np.zeros(7, dtype=bool)

    heavy_mask = rg._select_atoms(arr, "heavy")
    assert heavy_mask.sum() == 6, "OXT must be excluded from 'heavy'"
    assert not heavy_mask[list(arr.atom_name).index("OXT")]


# ─── backbone-only correspondence: robust to sequence redesign ────────────


@pytest.mark.fast
def test_select_block_atoms_backbone_is_invariant_to_residue_identity(tmp_path):
    """Only a scattered hotspot subset of a motif's sequence is held fixed
    during diffusion -- the rest can be redesigned, so the OUTPUT structure's
    own per-block heavy-atom count varies design-to-design (different amino
    acids have different atom counts). Backbone atoms (N, CA, C, O) are the
    one atom set invariant to identity, which is what graft_receptor_chains
    relies on for a stable Kabsch/alignment correspondence regardless of how
    much sequence gets redesigned."""
    # Two "designs" of the same 2-residue motif with DIFFERENT amino acids at
    # each position (GLY: 4 heavy atoms: N,CA,C,O; TRP: 14 heavy atoms) --
    # backbone-only selection must give the same atom COUNT for both.
    def make_two_residue_chain(res_names_and_sidechains):
        atom_names = []
        elements = []
        res_ids = []
        for i, (res_name, sidechain_atoms) in enumerate(res_names_and_sidechains, start=1):
            names = ["N", "CA", "C", "O"] + sidechain_atoms
            atom_names.extend(names)
            elements.extend(["N", "C", "C", "O"] + ["C"] * len(sidechain_atoms))
            res_ids.extend([i] * len(names))
        n = len(atom_names)
        arr = struc.AtomArray(n)
        arr.coord = np.zeros((n, 3), dtype=np.float32)
        arr.chain_id = np.full(n, "A")
        arr.res_id = np.array(res_ids)
        arr.res_name = np.full(n, "XXX")
        arr.atom_name = np.array(atom_names)
        arr.element = np.array(elements)
        arr.hetero = np.zeros(n, dtype=bool)
        return arr

    gly_gly = make_two_residue_chain([("GLY", []), ("GLY", [])])
    trp_trp = make_two_residue_chain(
        [("TRP", ["CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"]), ("TRP", ["CB"])]
    )

    assert gly_gly.array_length() == 8  # 2 * 4 backbone-only atoms
    assert trp_trp.array_length() > gly_gly.array_length()  # larger due to sidechains

    gly_backbone = rg._select_atoms(gly_gly, "backbone")
    trp_backbone = rg._select_atoms(trp_trp, "backbone")
    assert gly_backbone.sum() == 8  # unaffected, already all-backbone
    assert trp_backbone.sum() == 8  # SAME count despite the much larger residue

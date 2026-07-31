"""Tests for the Supermotif feature.

Supermotifs group non-connected motif fragments into a single rigid body for
Kabsch alignment during floating motif projection, preserving inter-fragment geometry.
"""
import numpy as np
import pytest
import torch
from biotite import structure as struc

from rfd3.model.floating_motif_projection import (
    FLOATING_MOTIF_REFERENCE_ANNOTATIONS,
    SUPERMOTIF_ID_ANNOTATION,
    FloatingMotifReference,
    build_floating_motif_references_from_contigs,
    project_floating_motifs_all_atom,
)

_ATOM_NAMES = ["N", "CA", "C", "O"]
_N = len(_ATOM_NAMES)  # atoms per residue


def _random_rotation(dtype=torch.float32):
    q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=dtype))
    if torch.det(q) < 0:
        q[:, -1] *= -1
    return q


def _make_atom_array(residues, include_supermotif_annotation=True):
    """Build a Biotite AtomArray for supermotif unit tests.

    Args:
        residues: list of (chain_id, res_id, src_component, supermotif_id).
            src_component starting with a letter  → contig motif atom.
            src_component starting with a digit   → non-motif (excluded from refs).
            supermotif_id non-empty string        → atom belongs to that supermotif.
            supermotif_id == ""                   → not part of any supermotif.
        include_supermotif_annotation: if False, omit the SUPERMOTIF_ID_ANNOTATION
            entirely (tests backward-compatibility path).
    """
    atoms, coords, src_comps, sm_ids = [], [], [], []
    chain_ids, res_ids, atom_name_list, element_list = [], [], [], []

    for res_idx, (chain_id, res_id, src_component, sm_id) in enumerate(residues):
        for atom_idx, atom_name in enumerate(_ATOM_NAMES):
            coord = np.array(
                [float(res_idx * 5), float(atom_idx), float(res_idx + atom_idx)],
                dtype=np.float32,
            )
            atoms.append(
                struc.Atom(np.zeros(3, dtype=np.float32), res_name="ALA", res_id=res_id)
            )
            coords.append(coord)
            src_comps.append(src_component)
            sm_ids.append(sm_id)
            chain_ids.append(chain_id)
            res_ids.append(res_id)
            atom_name_list.append(atom_name)
            element_list.append(atom_name[0])

    atom_array = struc.array(atoms)
    atom_array.chain_id = np.asarray(chain_ids)
    atom_array.res_id = np.asarray(res_ids)
    coords_np = np.asarray(coords, dtype=np.float32)

    atom_array.set_annotation("atom_name", np.asarray(atom_name_list))
    atom_array.set_annotation("element", np.asarray(element_list))
    atom_array.set_annotation("occupancy", np.ones(len(atom_array), dtype=np.float32))
    atom_array.set_annotation("src_component", np.asarray(src_comps))
    atom_array.set_annotation(
        "is_motif_atom_unindexed", np.zeros(len(atom_array), dtype=bool)
    )
    if include_supermotif_annotation:
        atom_array.set_annotation(SUPERMOTIF_ID_ANNOTATION, np.asarray(sm_ids))
    for axis, annotation in enumerate(FLOATING_MOTIF_REFERENCE_ANNOTATIONS):
        atom_array.set_annotation(annotation, coords_np[:, axis])

    return atom_array, coords_np


# ---------------------------------------------------------------------------
# build_floating_motif_references_from_contigs — supermotif grouping
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_build_supermotif_groups_noncontiguous_fragments_into_single_reference():
    """Non-contiguous motif fragments sharing a supermotif_id become one FloatingMotifReference."""
    residues = [
        ("A", 1, "A1", "sm1"),
        ("A", 2, "A2", "sm1"),
        ("A", 3, "3", ""),  # non-motif spacer (numeric src_component)
        ("A", 5, "A5", "sm1"),
        ("A", 6, "A6", "sm1"),
    ]
    atom_array, _ = _make_atom_array(residues)
    refs = build_floating_motif_references_from_contigs(
        sample_features={"atom_array": atom_array}
    )

    assert len(refs) == 1, "All sm1 atoms should collapse into a single reference"
    assert refs[0].source_components == ("A1", "A2", "A5", "A6")
    assert refs[0].sample_atom_indices.numel() == 4 * _N  # 4 motif residues × 4 atoms


@pytest.mark.fast
def test_build_supermotif_sample_atom_indices_are_correct():
    """sample_atom_indices must point to the correct positions in the full atom array."""
    residues = [
        ("A", 1, "A1", "sm1"),
        ("A", 2, "A2", "sm1"),
        ("A", 3, "3", ""),
        ("A", 5, "A5", "sm1"),
        ("A", 6, "A6", "sm1"),
    ]
    atom_array, _ = _make_atom_array(residues)
    refs = build_floating_motif_references_from_contigs(
        sample_features={"atom_array": atom_array}
    )

    # A1→0-3, A2→4-7, spacer→8-11 (excluded), A5→12-15, A6→16-19
    expected = list(range(0, 8)) + list(range(12, 20))
    assert sorted(refs[0].sample_atom_indices.tolist()) == expected


@pytest.mark.fast
def test_build_supermotif_reference_xyz_matches_stored_coords():
    """reference_xyz must be read from FLOATING_MOTIF_REFERENCE_ANNOTATIONS, not atom_array.coord."""
    residues = [
        ("A", 1, "A1", "sm1"),
        ("A", 2, "A2", "sm1"),
        ("A", 3, "3", ""),
        ("A", 5, "A5", "sm1"),
        ("A", 6, "A6", "sm1"),
    ]
    atom_array, coords_np = _make_atom_array(residues)
    refs = build_floating_motif_references_from_contigs(
        sample_features={"atom_array": atom_array}
    )

    indices = sorted(refs[0].sample_atom_indices.tolist())
    expected_xyz = torch.from_numpy(coords_np[indices])
    assert torch.allclose(refs[0].reference_xyz, expected_xyz)


@pytest.mark.fast
def test_build_supermotif_plus_remaining_independent_segment():
    """Atoms not in any supermotif are still split into independent contiguous segment refs."""
    residues = [
        ("A", 1, "A1", "sm1"),
        ("A", 2, "A2", "sm1"),
        ("A", 3, "3", ""),
        ("A", 5, "A5", ""),  # independent contig motif, consecutive with A6
        ("A", 6, "A6", ""),
    ]
    atom_array, _ = _make_atom_array(residues)
    refs = build_floating_motif_references_from_contigs(
        sample_features={"atom_array": atom_array}
    )

    assert len(refs) == 2
    assert refs[0].source_components == ("A1", "A2")   # supermotif first
    assert refs[1].source_components == ("A5", "A6")   # independent segment second


@pytest.mark.fast
def test_build_supermotif_atoms_do_not_appear_in_independent_segments():
    """Atoms consumed by a supermotif must be absent from any independent segment ref."""
    residues = [
        ("A", 1, "A1", "sm1"),
        ("A", 2, "A2", "sm1"),
        ("A", 3, "3", ""),
        ("A", 5, "A5", "sm1"),  # non-contiguous but in sm1
        ("A", 6, "A6", "sm1"),
        ("A", 7, "7", ""),
        ("A", 8, "A8", ""),  # independent motif segment
        ("A", 9, "A9", ""),
    ]
    atom_array, _ = _make_atom_array(residues)
    refs = build_floating_motif_references_from_contigs(
        sample_features={"atom_array": atom_array}
    )

    assert len(refs) == 2, "Expected 1 supermotif ref + 1 independent ref"
    sm_indices = set(refs[0].sample_atom_indices.tolist())
    ind_indices = set(refs[1].sample_atom_indices.tolist())

    assert sm_indices.isdisjoint(ind_indices), "Supermotif and independent refs must not share atoms"
    assert refs[1].source_components == ("A8", "A9")


@pytest.mark.fast
def test_build_two_supermotifs_produce_separate_references():
    """Each distinct non-empty supermotif_id produces its own reference, sorted alphabetically."""
    residues = [
        ("A", 1, "A1", "sm1"),
        ("A", 2, "A2", "sm1"),
        ("A", 3, "3", ""),
        ("A", 5, "A5", "sm2"),
        ("A", 6, "A6", "sm2"),
    ]
    atom_array, _ = _make_atom_array(residues)
    refs = build_floating_motif_references_from_contigs(
        sample_features={"atom_array": atom_array}
    )

    assert len(refs) == 2
    assert set(refs[0].source_components) == {"A1", "A2"}  # sm1 first (alphabetical)
    assert set(refs[1].source_components) == {"A5", "A6"}  # sm2 second


@pytest.mark.fast
def test_build_empty_supermotif_id_falls_back_to_independent_segments():
    """All-empty supermotif_id values are treated as no supermotif — original behaviour."""
    residues = [
        ("A", 1, "A1", ""),
        ("A", 2, "A2", ""),
        ("A", 3, "3", ""),
        ("A", 5, "A5", ""),
        ("A", 6, "A6", ""),
    ]
    atom_array, _ = _make_atom_array(residues)
    refs = build_floating_motif_references_from_contigs(
        sample_features={"atom_array": atom_array}
    )

    assert len(refs) == 2
    assert refs[0].source_components == ("A1", "A2")
    assert refs[1].source_components == ("A5", "A6")


@pytest.mark.fast
def test_build_no_supermotif_annotation_falls_back_to_independent_segments():
    """Without SUPERMOTIF_ID_ANNOTATION the original independent-segment behaviour is preserved."""
    residues = [
        ("A", 1, "A1", ""),
        ("A", 2, "A2", ""),
        ("A", 3, "3", ""),
        ("A", 5, "A5", ""),
        ("A", 6, "A6", ""),
    ]
    atom_array, _ = _make_atom_array(residues, include_supermotif_annotation=False)
    refs = build_floating_motif_references_from_contigs(
        sample_features={"atom_array": atom_array}
    )

    assert len(refs) == 2
    assert refs[0].source_components == ("A1", "A2")
    assert refs[1].source_components == ("A5", "A6")


@pytest.mark.fast
def test_build_supermotif_ignores_non_contig_motif_atoms():
    """A supermotif_id on a non-motif atom (numeric src_component) produces no reference."""
    residues = [
        ("A", 3, "3", "sm1"),  # supermotif tag on a non-motif atom → must be ignored
    ]
    atom_array, _ = _make_atom_array(residues)
    refs = build_floating_motif_references_from_contigs(
        sample_features={"atom_array": atom_array}
    )

    assert refs == [], "Non-contig-motif atoms with supermotif_id must not produce references"


@pytest.mark.fast
def test_build_supermotif_reference_atom_mask_excludes_zero_occupancy():
    """Atoms with occupancy == 0 must be masked False in reference_atom_mask."""
    residues = [
        ("A", 1, "A1", "sm1"),
        ("A", 2, "A2", "sm1"),
    ]
    atom_array, _ = _make_atom_array(residues)
    occ = atom_array.occupancy.copy()
    occ[0] = 0.0  # mask first atom
    atom_array.set_annotation("occupancy", occ)

    refs = build_floating_motif_references_from_contigs(
        sample_features={"atom_array": atom_array}
    )

    assert len(refs) == 1
    ref = refs[0]
    assert ref.reference_atom_mask.shape[0] == 2 * _N
    assert not ref.reference_atom_mask[0].item(), "Zero-occupancy atom must be masked out"
    assert ref.reference_atom_mask[1:].all().item(), "Remaining atoms must stay in mask"


# ---------------------------------------------------------------------------
# Rigid-body projection behaviour
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_supermotif_projection_preserves_inter_fragment_centroid_distance():
    """Single-rigid-body Kabsch preserves the reference centroid distance between fragments."""
    # Fragment A near origin, fragment B exactly 10 Å away in X.
    frag_a_ref = torch.arange(15, dtype=torch.float32).reshape(5, 3)
    frag_b_ref = frag_a_ref + torch.tensor([10.0, 0.0, 0.0])
    ref_dist = torch.dist(frag_a_ref.mean(0), frag_b_ref.mean(0))  # 10.0

    # Drift each fragment by a large, *different* translation (independent drift).
    xyz = torch.cat(
        [
            frag_a_ref + torch.tensor([-30.0, 0.0, 0.0]),
            torch.zeros(4, 3),  # non-motif spacer
            frag_b_ref + torch.tensor([40.0, 0.0, 0.0]),
        ],
        dim=0,
    )

    supermotif_ref = FloatingMotifReference(
        sample_atom_indices=torch.cat([torch.arange(0, 5), torch.arange(9, 14)]),
        reference_xyz=torch.cat([frag_a_ref, frag_b_ref], dim=0),
        reference_atom_mask=torch.ones(10, dtype=torch.bool),
        source_components=("A1", "A5"),
    )
    projected = project_floating_motifs_all_atom(xyz, [supermotif_ref])
    proj_dist = torch.dist(projected[:5].mean(0), projected[9:14].mean(0))

    # A single rigid body maps both fragments by the same R and t, so rotation
    # preserves all pairwise distances.
    assert torch.abs(proj_dist - ref_dist) < 1e-4


@pytest.mark.fast
def test_supermotif_vs_independent_alignment_inter_fragment_distance():
    """Independent per-fragment Kabsch does not preserve inter-fragment geometry under drift."""
    frag_a_ref = torch.arange(15, dtype=torch.float32).reshape(5, 3)
    frag_b_ref = frag_a_ref + torch.tensor([10.0, 0.0, 0.0])
    ref_dist = torch.dist(frag_a_ref.mean(0), frag_b_ref.mean(0))  # 10.0

    drift_a = torch.tensor([-30.0, 0.0, 0.0])
    drift_b = torch.tensor([40.0, 0.0, 0.0])
    xyz = torch.cat(
        [frag_a_ref + drift_a, torch.zeros(4, 3), frag_b_ref + drift_b], dim=0
    )

    # --- supermotif: single rigid body → distance preserved ---
    supermotif_ref = FloatingMotifReference(
        sample_atom_indices=torch.cat([torch.arange(0, 5), torch.arange(9, 14)]),
        reference_xyz=torch.cat([frag_a_ref, frag_b_ref], dim=0),
        reference_atom_mask=torch.ones(10, dtype=torch.bool),
        source_components=("A1", "A5"),
    )
    projected_sm = project_floating_motifs_all_atom(xyz, [supermotif_ref])
    sm_dist = torch.dist(projected_sm[:5].mean(0), projected_sm[9:14].mean(0))
    assert torch.abs(sm_dist - ref_dist) < 1e-4

    # --- independent: each fragment snapped to its own drifted frame → distance drifts ---
    # proj centroid A ≈ (frag_a + drift_a).mean, proj centroid B ≈ (frag_b + drift_b).mean
    # ind_dist ≈ |drift_a - drift_b - [10,0,0]| = |[-30,-0,0]-[40,0,0]-[10,0,0]| = 80
    independent_refs = [
        FloatingMotifReference(
            sample_atom_indices=torch.arange(0, 5),
            reference_xyz=frag_a_ref,
            reference_atom_mask=torch.ones(5, dtype=torch.bool),
            source_components=("A1",),
        ),
        FloatingMotifReference(
            sample_atom_indices=torch.arange(9, 14),
            reference_xyz=frag_b_ref,
            reference_atom_mask=torch.ones(5, dtype=torch.bool),
            source_components=("A5",),
        ),
    ]
    projected_ind = project_floating_motifs_all_atom(xyz, independent_refs)
    ind_dist = torch.dist(projected_ind[:5].mean(0), projected_ind[9:14].mean(0))
    assert torch.abs(ind_dist - ref_dist) > 5.0, (
        "Independent alignment with large drift must produce a different inter-fragment distance"
    )


@pytest.mark.fast
def test_supermotif_projection_does_not_alter_non_motif_atoms():
    """Atoms outside the supermotif reference must be left unchanged after projection."""
    torch.manual_seed(5)
    frag_a_ref = torch.randn(5, 3)
    frag_b_ref = frag_a_ref + torch.tensor([10.0, 0.0, 0.0])
    non_motif = torch.randn(4, 3)

    rot = _random_rotation()
    xyz = torch.cat([frag_a_ref @ rot, non_motif, frag_b_ref @ rot], dim=0)

    supermotif_ref = FloatingMotifReference(
        sample_atom_indices=torch.cat([torch.arange(0, 5), torch.arange(9, 14)]),
        reference_xyz=torch.cat([frag_a_ref, frag_b_ref], dim=0),
        reference_atom_mask=torch.ones(10, dtype=torch.bool),
        source_components=("A1", "A5"),
    )
    projected = project_floating_motifs_all_atom(xyz, [supermotif_ref])

    assert torch.allclose(projected[5:9], non_motif), "Non-motif atoms must not be modified"


@pytest.mark.fast
def test_supermotif_projection_recovers_rigid_transform():
    """When the target is an exact rigid-body transform of the reference, projection recovers it."""
    torch.manual_seed(99)
    frag_a_ref = torch.randn(6, 3)
    frag_b_ref = torch.randn(5, 3) + 15.0

    rot = _random_rotation()
    trans = torch.tensor([3.0, -2.0, 1.5])
    frag_a_target = frag_a_ref @ rot + trans
    frag_b_target = frag_b_ref @ rot + trans
    non_motif = torch.randn(4, 3)

    xyz = torch.cat([frag_a_target, non_motif, frag_b_target], dim=0)

    supermotif_ref = FloatingMotifReference(
        sample_atom_indices=torch.cat([torch.arange(0, 6), torch.arange(10, 15)]),
        reference_xyz=torch.cat([frag_a_ref, frag_b_ref], dim=0),
        reference_atom_mask=torch.ones(11, dtype=torch.bool),
        source_components=("A1", "A5"),
    )
    projected = project_floating_motifs_all_atom(xyz, [supermotif_ref])

    assert torch.allclose(projected[:6], frag_a_target, atol=1e-4)
    assert torch.allclose(projected[10:15], frag_b_target, atol=1e-4)
    assert torch.allclose(projected[6:10], non_motif)


# ---------------------------------------------------------------------------
# build_floating_motif_references_from_contigs — supermotif ordering
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_build_supermotif_refs_come_before_independent_refs():
    """Supermotif references are prepended; independent segment refs follow."""
    residues = [
        ("A", 1, "A1", ""),   # independent
        ("A", 2, "A2", ""),
        ("A", 3, "3", ""),
        ("A", 5, "A5", "sm1"),  # supermotif
        ("A", 6, "A6", "sm1"),
    ]
    atom_array, _ = _make_atom_array(residues)
    refs = build_floating_motif_references_from_contigs(
        sample_features={"atom_array": atom_array}
    )

    assert len(refs) == 2
    # Supermotif ref must be first regardless of position in the array
    assert set(refs[0].source_components) == {"A5", "A6"}  # sm1
    assert set(refs[1].source_components) == {"A1", "A2"}  # independent


@pytest.mark.fast
def test_build_three_supermotifs_sorted_alphabetically():
    """Multiple supermotifs are emitted in alphabetical order of their names."""
    residues = [
        ("A", 1, "A1", "gamma"),
        ("A", 2, "A2", "alpha"),
        ("A", 3, "A3", "beta"),
    ]
    atom_array, _ = _make_atom_array(residues)
    refs = build_floating_motif_references_from_contigs(
        sample_features={"atom_array": atom_array}
    )

    assert len(refs) == 3
    names = [refs[i].source_components[0] for i in range(3)]
    assert names == ["A2", "A3", "A1"], (
        "References should follow alphabetical supermotif order: alpha→A2, beta→A3, gamma→A1"
    )

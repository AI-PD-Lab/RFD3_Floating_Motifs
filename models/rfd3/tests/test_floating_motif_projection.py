import numpy as np
import pytest
import torch
from biotite import structure as struc
from rfd3.model.floating_motif_projection import (
    FLOATING_MOTIF_REFERENCE_ANNOTATIONS,
    FloatingMotifReference,
    build_floating_motif_references_from_contigs,
    kabsch_align_all_atom,
    project_floating_motifs_all_atom,
    should_project_floating_motifs,
)


def _random_rotation(dtype=torch.float64):
    q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=dtype))
    if torch.det(q) < 0:
        q[:, -1] *= -1
    return q


@pytest.mark.fast
def test_kabsch_align_all_atom_recovers_random_transform():
    torch.manual_seed(0)
    ref = torch.randn(32, 3, dtype=torch.float64)
    rot = _random_rotation()
    trans = torch.tensor([3.0, -2.0, 0.5], dtype=torch.float64)
    mob = ref @ rot + trans

    aligned = kabsch_align_all_atom(ref, mob)
    rmsd = torch.sqrt(torch.mean(torch.sum((aligned - mob) ** 2, dim=-1)))

    assert rmsd.item() < 1e-4


@pytest.mark.fast
def test_project_floating_motifs_all_atom_independent_segments():
    torch.manual_seed(1)
    ref1 = torch.randn(6, 3)
    ref2 = torch.randn(5, 3) + 20.0
    non_motif = torch.randn(4, 3)

    target1 = ref1 @ _random_rotation(dtype=torch.float32) + torch.tensor(
        [1.0, 2.0, 3.0]
    )
    target2 = ref2 @ _random_rotation(dtype=torch.float32) + torch.tensor(
        [-5.0, 0.5, 7.0]
    )
    xyz = torch.cat([target1, non_motif, target2], dim=0)

    refs = [
        FloatingMotifReference(
            sample_atom_indices=torch.arange(0, 6),
            reference_xyz=ref1,
            reference_atom_mask=torch.ones(6, dtype=torch.bool),
            source_components=("A1", "A2"),
        ),
        FloatingMotifReference(
            sample_atom_indices=torch.arange(10, 15),
            reference_xyz=ref2,
            reference_atom_mask=torch.ones(5, dtype=torch.bool),
            source_components=("A5", "A6"),
        ),
    ]

    projected = project_floating_motifs_all_atom(xyz, refs)

    assert torch.allclose(projected[:6], target1, atol=1e-4)
    assert torch.allclose(projected[6:10], non_motif)
    assert torch.allclose(projected[10:15], target2, atol=1e-4)


@pytest.mark.fast
def test_should_project_floating_motifs_schedule():
    assert not should_project_floating_motifs(20, enabled=False)
    assert not should_project_floating_motifs(19, enabled=True, burn_in=20)
    assert should_project_floating_motifs(20, enabled=True, burn_in=20, project_every=5)
    assert not should_project_floating_motifs(
        21, enabled=True, burn_in=20, project_every=5
    )
    assert should_project_floating_motifs(25, enabled=True, burn_in=20, project_every=5)
    assert not should_project_floating_motifs(
        31, enabled=True, burn_in=20, project_every=5, stop_after=30
    )
    with pytest.raises(ValueError):
        should_project_floating_motifs(0, enabled=True, project_every=0)


@pytest.mark.fast
def test_build_floating_motif_references_from_contigs_independent_segments():
    atom_array, original_coords = _make_segmented_atom_array()
    refs = build_floating_motif_references_from_contigs(
        sample_features={"atom_array": atom_array}
    )

    assert len(refs) == 2
    assert refs[0].source_components == ("A1", "A2")
    assert refs[1].source_components == ("A5", "A6")
    assert torch.allclose(refs[0].reference_xyz, torch.from_numpy(original_coords[:8]))
    assert torch.allclose(
        refs[1].reference_xyz, torch.from_numpy(original_coords[12:20])
    )


def _make_segmented_atom_array():
    atoms = []
    original_coords = []
    src_components = []
    atom_names = []
    elements = []

    residues = [
        ("A", 1, "A1"),
        ("A", 2, "A2"),
        ("A", 3, "3"),
        ("A", 5, "A5"),
        ("A", 6, "A6"),
    ]
    names = ["N", "CA", "C", "O"]
    chain_ids = []
    res_ids = []
    for res_idx, (chain_id, res_id, src_component) in enumerate(residues):
        for atom_idx, atom_name in enumerate(names):
            coord = np.array([res_idx, atom_idx, res_idx + atom_idx], dtype=np.float32)
            atoms.append(
                struc.Atom(
                    np.zeros(3, dtype=np.float32),
                    res_name="ALA",
                    res_id=res_id,
                )
            )
            original_coords.append(coord)
            src_components.append(src_component)
            chain_ids.append(chain_id)
            res_ids.append(res_id)
            atom_names.append(atom_name)
            elements.append(atom_name[0])

    atom_array = struc.array(atoms)
    atom_array.chain_id = np.asarray(chain_ids)
    atom_array.res_id = np.asarray(res_ids)
    original_coords = np.asarray(original_coords, dtype=np.float32)
    atom_array.set_annotation("atom_name", np.asarray(atom_names))
    atom_array.set_annotation("element", np.asarray(elements))
    atom_array.set_annotation("occupancy", np.ones(len(atom_array), dtype=np.float32))
    atom_array.set_annotation("src_component", np.asarray(src_components))
    atom_array.set_annotation(
        "is_motif_atom_unindexed", np.zeros(len(atom_array), dtype=bool)
    )
    for axis, annotation in enumerate(FLOATING_MOTIF_REFERENCE_ANNOTATIONS):
        atom_array.set_annotation(annotation, original_coords[:, axis])
    return atom_array, original_coords

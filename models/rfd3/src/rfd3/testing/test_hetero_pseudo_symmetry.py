import torch

from rfd3.inference.symmetry.hetero_pseudo import (
    HeteroPseudoSymmetryConfig,
    apply_hetero_pseudo_symmetry,
    build_hetero_pseudo_symmetry_masks,
)
from rfd3.inference.symmetry.symmetry_utils import apply_symmetry_to_xyz_atomwise
from rfd3.model.inference_sampler import (
    SampleDiffusionWithHeteroPseudoSymmetry,
    SampleDiffusionWithSymmetry,
)


def _toy_c2_features():
    device = torch.device("cpu")
    return {
        "ref_element": torch.ones(6, dtype=torch.long, device=device) * 6,
        "is_virtual": torch.zeros(6, dtype=torch.bool, device=device),
        "is_backbone": torch.ones(6, dtype=torch.bool, device=device),
        "is_ca": torch.ones(6, dtype=torch.bool, device=device),
        "is_motif_atom_with_fixed_coord": torch.tensor(
            [True, False, False, True, False, False], device=device
        ),
        "is_motif_atom_with_fixed_seq": torch.zeros(
            6, dtype=torch.bool, device=device
        ),
        "is_motif_atom_unindexed": torch.zeros(6, dtype=torch.bool, device=device),
        "atom_to_token_map": torch.arange(6, device=device),
        "sym_entity_id": torch.zeros(6, dtype=torch.long, device=device),
        "sym_transform_id": torch.tensor([0, 0, 0, 1, 1, 1], device=device),
        "is_sym_asu": torch.tensor(
            [True, True, True, False, False, False], device=device
        ),
        "sym_transform": {
            0: (torch.eye(3, device=device), torch.zeros(3, device=device)),
            1: (
                torch.eye(3, device=device),
                torch.tensor([10.0, 0.0, 0.0], device=device),
            ),
        },
    }


def _toy_c2_xyz():
    return torch.tensor(
        [
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [5.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [11.0, 0.0, 0.0],
                [5.4, 0.0, 0.0],
            ]
        ]
    )


def test_initialization_only_skips_post_init_projection():
    f = _toy_c2_features()
    xyz = _toy_c2_xyz()
    config = HeteroPseudoSymmetryConfig(
        post_init_symmetry="initialization_only",
        interface_distance_cutoff=1.0,
    )

    projected, debug = apply_hetero_pseudo_symmetry(xyz, f, config, step_idx=0)

    assert debug["active"] is False
    assert torch.equal(projected, xyz)


def test_interface_projection_respects_motif_and_contact_priority():
    f = _toy_c2_features()
    xyz = _toy_c2_xyz()
    config = HeteroPseudoSymmetryConfig(
        hard=True,
        recenter_enabled=False,
        interface_distance_cutoff=1.0,
        interface_sequence_buffer=0,
        motif_contact_distance_cutoff=2.0,
        motif_contact_sequence_buffer=0,
        support_enabled=False,
        motif_follow_scaffold_frame=False,
    )

    masks = build_hetero_pseudo_symmetry_masks(xyz, f, config)
    projected, debug = apply_hetero_pseudo_symmetry(xyz, f, config, step_idx=0)

    assert debug["active"] is True
    assert masks["motif_mask"].tolist() == [[True, False, False, True, False, False]]
    assert masks["motif_contact_mask"].tolist() == [
        [False, True, False, False, True, False]
    ]
    assert masks["interface_mask"].tolist() == [
        [False, False, True, False, False, True]
    ]
    assert torch.equal(projected[:, [0, 1, 3, 4]], xyz[:, [0, 1, 3, 4]])
    assert torch.equal(
        projected[:, [2, 5]],
        torch.tensor([[[5.0, 0.0, 0.0], [15.0, 0.0, 0.0]]]),
    )


def test_hetero_recenter_moves_whole_assembly_without_copying_motifs():
    f = _toy_c2_features()
    xyz = _toy_c2_xyz() + torch.tensor([100.0, 200.0, -50.0])
    config = HeteroPseudoSymmetryConfig(
        weight=0.0,
        support_enabled=False,
        motif_contact_exclusion_enabled=False,
    )

    projected, debug = apply_hetero_pseudo_symmetry(xyz, f, config, step_idx=0)

    assert debug["recentered"] is True
    assert torch.allclose(projected.mean(dim=1), torch.zeros(1, 3), atol=1e-5)
    assert torch.allclose(
        projected[:, 3] - projected[:, 0],
        xyz[:, 3] - xyz[:, 0],
    )


def test_hetero_projection_allows_unequal_motif_lengths():
    device = torch.device("cpu")
    # copy 0: 2 motif atoms + 4 scaffold atoms
    # copy 1: 3 motif atoms + 4 scaffold atoms
    f = {
        "ref_element": torch.ones(13, dtype=torch.long, device=device) * 6,
        "is_virtual": torch.zeros(13, dtype=torch.bool, device=device),
        "is_backbone": torch.ones(13, dtype=torch.bool, device=device),
        "is_ca": torch.ones(13, dtype=torch.bool, device=device),
        "is_motif_atom_with_fixed_coord": torch.zeros(
            13, dtype=torch.bool, device=device
        ),
        "is_motif_atom_with_fixed_seq": torch.tensor(
            [True, True, False, False, False, False,
             True, True, True, False, False, False, False],
            device=device,
        ),
        "is_motif_atom_unindexed": torch.zeros(13, dtype=torch.bool, device=device),
        "atom_to_token_map": torch.arange(13, device=device),
        "sym_entity_id": torch.zeros(13, dtype=torch.long, device=device),
        "sym_transform_id": torch.tensor([0] * 6 + [1] * 7, device=device),
        "is_sym_asu": torch.tensor([True] * 6 + [False] * 7, device=device),
        "sym_transform": {
            0: (torch.eye(3, device=device), torch.zeros(3, device=device)),
            1: (
                torch.eye(3, device=device),
                torch.tensor([10.0, 0.0, 0.0], device=device),
            ),
        },
    }
    xyz = torch.randn(1, 13, 3, device=device)
    config = HeteroPseudoSymmetryConfig(
        hard=True,
        interface_distance_cutoff=999.0,
        interface_sequence_buffer=0,
        motif_contact_exclusion_enabled=False,
        support_enabled=False,
    )

    projected, debug = apply_hetero_pseudo_symmetry(xyz, f, config, step_idx=0)

    assert projected.shape == xyz.shape
    assert debug["active"] is True
    assert debug["interface_atoms"].item() == 8


def test_normal_and_hetero_sampler_projection_modes_are_distinct():
    f = _toy_c2_features()
    xyz = _toy_c2_xyz()
    normal = SampleDiffusionWithSymmetry(gamma_0=0.6)
    hetero = SampleDiffusionWithHeteroPseudoSymmetry(
        gamma_0=0.6,
        hetero_post_init_symmetry="initialization_only",
    )

    normal_projected = normal.apply_post_update_symmetry(
        xyz.clone(), f, step_num=0, c_t=torch.tensor(10.0), gamma_min_sym=1.0
    )
    hetero_projected = hetero.apply_post_update_symmetry(
        xyz.clone(), f, step_num=0, c_t=torch.tensor(10.0), gamma_min_sym=1.0
    )

    assert torch.equal(normal_projected, apply_symmetry_to_xyz_atomwise(xyz.clone(), f))
    assert torch.equal(hetero_projected, xyz)


def test_hetero_motifs_follow_scaffold_frame_without_copying_other_motifs():
    f = _toy_c2_features()
    xyz = _toy_c2_xyz()
    config = HeteroPseudoSymmetryConfig(
        hard=True,
        recenter_enabled=False,
        interface_distance_cutoff=1.0,
        interface_sequence_buffer=0,
        motif_contact_exclusion_enabled=False,
        support_enabled=False,
    )

    projected, debug = apply_hetero_pseudo_symmetry(xyz, f, config, step_idx=0)
    motif_shift = projected[:, 5] - xyz[:, 5]

    assert debug["motif_frame_updates"].item() == 1
    assert torch.equal(projected[:, [1, 2]], xyz[:, [1, 2]])
    assert torch.allclose(projected[:, 3], xyz[:, 3] + motif_shift)

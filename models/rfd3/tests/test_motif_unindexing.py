import torch

from rfd3.model.motif_unindexing import (
    MotifAssignment,
    MotifUnindexingConfig,
    build_motif_unindexing_controller,
)


def _write_ca_pdb(path, coords):
    lines = []
    for i, (x, y, z) in enumerate(coords, start=1):
        lines.append(
            f"ATOM  {i:5d}  CA  ALA A{i:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           C\n"
        )
    path.write_text("".join(lines), encoding="utf-8")


def _write_backbone_pdb(path, residues):
    lines = []
    serial = 1
    for res_i, atoms in enumerate(residues, start=1):
        for atom_name, (x, y, z) in atoms.items():
            lines.append(
                f"ATOM  {serial:5d} {atom_name:>4s} ALA A{res_i:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           C\n"
            )
            serial += 1
    path.write_text("".join(lines), encoding="utf-8")


def _features(n_atoms=8):
    return {
        "atom_to_token_map": torch.arange(n_atoms, dtype=torch.long),
        "is_ca": torch.ones(n_atoms, dtype=torch.bool),
    }


def _backbone_features(n_residues=3):
    atom_to_token_map = []
    is_ca = []
    for token in range(n_residues):
        atom_to_token_map.extend([token] * 5)
        is_ca.extend([False, True, False, False, False])
    return {
        "atom_to_token_map": torch.tensor(atom_to_token_map, dtype=torch.long),
        "is_ca": torch.tensor(is_ca, dtype=torch.bool),
    }


class _AtomArrayStub:
    def __init__(self, atom_names):
        self.atom_name = atom_names


def test_motif_unindexing_assigns_best_non_overlapping_regions(tmp_path):
    motif_a = tmp_path / "motif_a.pdb"
    motif_b = tmp_path / "motif_b.pdb"
    _write_ca_pdb(motif_a, [(0, 0, 0), (1, 0, 0), (2, 0, 0)])
    _write_ca_pdb(motif_b, [(0, 0, 0), (0, 1, 0), (0, 2, 0)])

    xyz = torch.tensor(
        [
            [
                [10.0, 10.0, 0.0],
                [11.0, 10.0, 0.0],
                [12.0, 10.0, 0.0],
                [30.0, 30.0, 0.0],
                [30.0, 31.0, 0.0],
                [30.0, 32.0, 0.0],
                [50.0, 50.0, 0.0],
                [60.0, 50.0, 0.0],
            ]
        ]
    )
    controller = build_motif_unindexing_controller(
        MotifUnindexingConfig(
            enabled=True,
            motif_pdbs=[str(motif_a), str(motif_b)],
            allow_overlap=False,
        ),
        _features(),
    )

    assignments = controller.find_assignments(xyz)

    assert len(assignments) == 2
    assert assignments[0].sample_atom_indices.tolist() == [0, 1, 2]
    assert assignments[1].sample_atom_indices.tolist() == [3, 4, 5]
    assert set(assignments[0].sample_atom_indices.tolist()).isdisjoint(
        assignments[1].sample_atom_indices.tolist()
    )


def test_motif_unindexing_bias_reduces_selected_region_loss(tmp_path):
    motif = tmp_path / "motif.pdb"
    _write_ca_pdb(motif, [(0, 0, 0), (1, 0, 0), (2, 0, 0)])
    xyz = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.5, 0.0],
                [2.0, 0.0, 0.0],
                [9.0, 9.0, 9.0],
            ]
        ]
    )
    controller = build_motif_unindexing_controller(
        MotifUnindexingConfig(
            enabled=True,
            motif_pdbs=[str(motif)],
            loss_weight=0.25,
            activation_threshold=0.0,
        ),
        _features(n_atoms=4),
    )
    controller.assignments = controller.find_assignments(xyz)
    before = controller.current_mean_rmsd(xyz)

    updated = controller.apply_pre_activation_bias(xyz, step_idx=0)
    controller.assignments = controller.find_assignments(updated)
    after = controller.current_mean_rmsd(updated)

    assert after < before


def test_motif_unindexing_activation_uses_floating_motif_refs(tmp_path):
    motif = tmp_path / "motif.pdb"
    _write_ca_pdb(motif, [(0, 0, 0), (1, 0, 0), (2, 0, 0)])
    xyz = torch.tensor([[[5.0, 0, 0], [6.0, 0, 0], [7.0, 0, 0]]])
    controller = build_motif_unindexing_controller(
        MotifUnindexingConfig(
            enabled=True,
            motif_pdbs=[str(motif)],
            activation_threshold=0.1,
            post_activation_guidance_steps=2,
        ),
        _features(n_atoms=3),
    )

    controller.apply_pre_activation_bias(xyz, step_idx=0)

    refs_step_0 = controller.active_floating_motif_refs(step_idx=0)
    refs_step_2 = controller.active_floating_motif_refs(step_idx=2)
    assert len(refs_step_0) == 1
    assert refs_step_0[0].sample_atom_indices.tolist() == [0, 1, 2]
    assert refs_step_2 == []


def test_motif_unindexing_dynamic_refs_use_matched_backbone_atoms(tmp_path):
    motif = tmp_path / "motif_backbone.pdb"
    residues = []
    for i in range(3):
        x = float(i)
        residues.append(
            {
                "N": (x - 0.2, 0.0, 0.0),
                "CA": (x, 0.0, 0.0),
                "C": (x + 0.2, 0.0, 0.0),
                "O": (x + 0.3, 0.1, 0.0),
                "CB": (x, 0.4, 0.0),
            }
        )
    _write_backbone_pdb(motif, residues)

    atom_names = ["N", "CA", "C", "O", "CB"] * 3
    xyz = torch.tensor(
        [
            [
                [-0.2, 5.0, 0.0],
                [0.0, 5.0, 0.0],
                [0.2, 5.0, 0.0],
                [0.3, 5.1, 0.0],
                [0.0, 5.4, 0.0],
                [0.8, 5.0, 0.0],
                [1.0, 5.0, 0.0],
                [1.2, 5.0, 0.0],
                [1.3, 5.1, 0.0],
                [1.0, 5.4, 0.0],
                [1.8, 5.0, 0.0],
                [2.0, 5.0, 0.0],
                [2.2, 5.0, 0.0],
                [2.3, 5.1, 0.0],
                [2.0, 5.4, 0.0],
            ]
        ]
    )
    controller = build_motif_unindexing_controller(
        MotifUnindexingConfig(
            enabled=True,
            motif_pdbs=[str(motif)],
            activation_threshold=0.1,
            post_activation_guidance_steps=2,
        ),
        _backbone_features(),
        sample_features={"atom_array": _AtomArrayStub(atom_names)},
    )

    controller.apply_pre_activation_bias(xyz, step_idx=0)
    refs = controller.active_floating_motif_refs(step_idx=0)

    assert len(refs) == 1
    assert refs[0].sample_atom_indices.tolist() == list(range(15))
    assert refs[0].reference_xyz.shape == (15, 3)


def test_motif_unindexing_promotes_activated_assignments_to_motif_features(tmp_path):
    motif = tmp_path / "motif_backbone_promote.pdb"
    residues = []
    for i in range(3):
        x = float(i)
        residues.append(
            {
                "N": (x - 0.2, 0.0, 0.0),
                "CA": (x, 0.0, 0.0),
                "C": (x + 0.2, 0.0, 0.0),
                "O": (x + 0.3, 0.1, 0.0),
                "CB": (x, 0.4, 0.0),
            }
        )
    _write_backbone_pdb(motif, residues)

    atom_names = ["N", "CA", "C", "O", "CB"] * 3
    xyz = torch.tensor(
        [
            [
                residues[res_i][atom_name]
                for res_i in range(3)
                for atom_name in ["N", "CA", "C", "O", "CB"]
            ]
        ],
        dtype=torch.float32,
    )
    controller = build_motif_unindexing_controller(
        MotifUnindexingConfig(
            enabled=True,
            motif_pdbs=[str(motif)],
            activation_threshold=0.1,
            promote_to_motif_on_activation=True,
        ),
        _backbone_features(),
        sample_features={"atom_array": _AtomArrayStub(atom_names)},
    )
    controller.apply_pre_activation_bias(xyz, step_idx=0)
    f = {
        **_backbone_features(),
        "ref_motif_token_type": torch.tensor(
            [[1, 0, 0], [1, 0, 0], [1, 0, 0]],
            dtype=torch.float32,
        ),
        "is_motif_atom": torch.zeros(15, dtype=torch.bool),
        "is_motif_atom_with_fixed_seq": torch.zeros(15, dtype=torch.bool),
        "is_motif_atom_with_fixed_coord": torch.zeros(15, dtype=torch.bool),
        "ref_pos": torch.zeros(15, 3, dtype=torch.float32),
        "motif_pos": torch.zeros(15, 3, dtype=torch.float32),
        "ref_mask": torch.zeros(15, dtype=torch.bool),
    }

    promoted = controller.promoted_feature_dict(f, xyz, step_idx=1)

    assert promoted["ref_motif_token_type"].tolist() == [
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0],
    ]
    assert promoted["is_motif_atom"].all()
    assert not promoted["is_motif_atom_with_fixed_coord"].any()
    assert promoted["ref_mask"].all()
    assert torch.isfinite(promoted["floating_motif_reference_pos"]).all()


def test_motif_unindexing_matching_uses_matched_backbone_atoms(tmp_path):
    motif = tmp_path / "motif_backbone_match.pdb"
    residues = []
    for i in range(3):
        x = float(i)
        residues.append(
            {
                "N": (x - 0.2, 0.0, 0.0),
                "CA": (x, 0.0, 0.0),
                "C": (x + 0.2, 0.0, 0.0),
                "O": (x + 0.3, 0.1, 0.0),
                "CB": (x, 0.4, 0.0),
            }
        )
    _write_backbone_pdb(motif, residues)

    atom_names = ["N", "CA", "C", "O", "CB"] * 4
    generated_residues = []
    bad_first_residue = {
        "N": (-5.0, 9.0, 0.0),
        "CA": (0.0, 5.0, 0.0),
        "C": (5.0, 9.0, 0.0),
        "O": (6.0, 9.0, 0.0),
        "CB": (0.0, 15.0, 0.0),
    }
    generated_residues.append(bad_first_residue)
    for motif_residue in residues:
        generated_residues.append(
            {
                atom_name: (xyz[0] + 1.0, xyz[1] + 5.0, xyz[2])
                for atom_name, xyz in motif_residue.items()
            }
        )
    xyz = torch.tensor(
        [
            [
                generated_residues[res_i][atom_name]
                for res_i in range(4)
                for atom_name in ["N", "CA", "C", "O", "CB"]
            ]
        ],
        dtype=torch.float32,
    )
    controller = build_motif_unindexing_controller(
        MotifUnindexingConfig(
            enabled=True,
            motif_pdbs=[str(motif)],
        ),
        _backbone_features(n_residues=4),
        sample_features={"atom_array": _AtomArrayStub(atom_names)},
    )

    assignments = controller.find_assignments(xyz)

    assert assignments[0].sample_atom_indices.tolist() == [6, 11, 16]


def test_floating_project_on_activation_respects_timeout(tmp_path):
    """floating_project_on_activation must still honor projection stop gates."""
    motif = tmp_path / "motif.pdb"
    _write_ca_pdb(motif, [(0, 0, 0), (1, 0, 0), (2, 0, 0)])
    xyz = torch.zeros(1, 3, 3)
    controller = build_motif_unindexing_controller(
        MotifUnindexingConfig(
            enabled=True,
            motif_pdbs=[str(motif)],
            post_activation_guidance_steps=2,
            floating_project_on_activation=True,
        ),
        _features(n_atoms=3),
    )
    controller.activated_step = 0
    controller.assignments = controller.find_assignments(xyz)

    assert len(controller.active_floating_motif_refs(step_idx=1)) == 1
    assert len(controller.active_floating_motif_refs(step_idx=2)) == 0


def test_floating_project_on_activation_off_still_respects_timeout(tmp_path):
    """Without the flag the existing guidance-steps timeout still fires."""
    motif = tmp_path / "motif.pdb"
    _write_ca_pdb(motif, [(0, 0, 0), (1, 0, 0), (2, 0, 0)])
    xyz = torch.zeros(1, 3, 3)
    controller = build_motif_unindexing_controller(
        MotifUnindexingConfig(
            enabled=True,
            motif_pdbs=[str(motif)],
            post_activation_guidance_steps=2,
            floating_project_on_activation=False,
        ),
        _features(n_atoms=3),
    )
    controller.activated_step = 0
    controller.assignments = controller.find_assignments(xyz)

    assert len(controller.active_floating_motif_refs(step_idx=1)) == 1
    assert len(controller.active_floating_motif_refs(step_idx=2)) == 0


def test_promoted_feature_dict_is_noop_before_activation(tmp_path):
    """promoted_feature_dict must return the original dict when not yet activated."""
    motif = tmp_path / "motif.pdb"
    _write_ca_pdb(motif, [(0, 0, 0), (1, 0, 0), (2, 0, 0)])
    xyz = torch.zeros(1, 3, 3)
    controller = build_motif_unindexing_controller(
        MotifUnindexingConfig(
            enabled=True,
            motif_pdbs=[str(motif)],
            activation_threshold=0.0,
            promote_to_motif_on_activation=True,
        ),
        _features(n_atoms=3),
    )
    # activated_step is still None; no call to apply_pre_activation_bias yet
    assert controller.activated_step is None

    f = _features(n_atoms=3)
    result = controller.promoted_feature_dict(f, xyz, step_idx=0)

    assert result is f


def test_promoted_feature_dict_is_noop_when_flag_disabled(tmp_path):
    """promoted_feature_dict must be a no-op when promote_to_motif_on_activation=False."""
    motif = tmp_path / "motif.pdb"
    _write_ca_pdb(motif, [(0, 0, 0), (1, 0, 0), (2, 0, 0)])
    xyz = torch.zeros(1, 3, 3)
    controller = build_motif_unindexing_controller(
        MotifUnindexingConfig(
            enabled=True,
            motif_pdbs=[str(motif)],
            activation_threshold=100.0,
            promote_to_motif_on_activation=False,
        ),
        _features(n_atoms=3),
    )
    # Force activation so we can confirm the flag is what gates it
    controller.assignments = controller.find_assignments(xyz)
    controller.activated_step = 0

    f = _features(n_atoms=3)
    result = controller.promoted_feature_dict(f, xyz, step_idx=1)

    assert result is f


def test_post_activation_stop_after_is_absolute(tmp_path):
    """post_activation_stop_after is an absolute step ceiling, not relative."""
    motif = tmp_path / "motif.pdb"
    _write_ca_pdb(motif, [(0, 0, 0), (1, 0, 0), (2, 0, 0)])
    controller = build_motif_unindexing_controller(
        MotifUnindexingConfig(
            enabled=True,
            motif_pdbs=[str(motif)],
            post_activation_stop_after=5,
            post_activation_guidance_steps=200,
        ),
        _features(n_atoms=3),
    )
    # Force activation at step 3 so we can check the absolute ceiling
    controller.activated_step = 3

    assert controller.is_post_activation_active(step_idx=5)
    assert not controller.is_post_activation_active(step_idx=6)


def test_bias_continues_post_activation(tmp_path):
    """Bias keeps applying after activation when bias_stop_after is not set."""
    motif = tmp_path / "motif.pdb"
    _write_ca_pdb(motif, [(0, 0, 0), (1, 0, 0), (2, 0, 0)])
    xyz = torch.zeros(1, 3, 3)
    controller = build_motif_unindexing_controller(
        MotifUnindexingConfig(
            enabled=True,
            motif_pdbs=[str(motif)],
        ),
        _features(n_atoms=3),
    )
    controller.assignments = controller.find_assignments(xyz)
    controller.activated_step = 0

    result = controller.apply_pre_activation_bias(xyz.clone(), step_idx=50)
    assert not torch.equal(result, xyz), "bias should still modify xyz after activation"


def test_bias_stop_after_disables_bias(tmp_path):
    """bias_stop_after is an absolute ceiling for the gradient bias."""
    motif = tmp_path / "motif.pdb"
    _write_ca_pdb(motif, [(0, 0, 0), (1, 0, 0), (2, 0, 0)])
    xyz = torch.zeros(1, 3, 3)
    controller = build_motif_unindexing_controller(
        MotifUnindexingConfig(
            enabled=True,
            motif_pdbs=[str(motif)],
            bias_stop_after=10,
        ),
        _features(n_atoms=3),
    )
    controller.assignments = controller.find_assignments(xyz)
    controller.activated_step = 0

    result_before = controller.apply_pre_activation_bias(xyz.clone(), step_idx=10)
    result_after = controller.apply_pre_activation_bias(xyz.clone(), step_idx=11)
    assert not torch.equal(result_before, xyz), "bias should run at step <= bias_stop_after"
    assert torch.equal(result_after, xyz), "bias should be disabled after bias_stop_after"


def test_immediate_assignment_activation_is_opt_in(tmp_path):
    motif = tmp_path / "motif.pdb"
    _write_ca_pdb(motif, [(0, 0, 0), (1, 0, 0), (2, 0, 0)])
    xyz = torch.tensor(
        [[[10.0, 10.0, 0.0], [11.0, 10.5, 0.0], [12.0, 10.0, 0.0]]]
    )

    baseline = build_motif_unindexing_controller(
        MotifUnindexingConfig(
            enabled=True,
            motif_pdbs=[str(motif)],
            promote_immediately_on_assignment=True,
            activation_threshold=0.0,
        ),
        _features(n_atoms=3),
    )
    baseline.apply_pre_activation_bias(xyz, step_idx=0)
    assert baseline.activated_step is None

    immediate = build_motif_unindexing_controller(
        MotifUnindexingConfig(
            enabled=True,
            motif_pdbs=[str(motif)],
            promote_immediately_on_assignment=True,
            activate_immediately_on_assignment=True,
            rebuild_static_cache_after_immediate_activation=True,
            activation_threshold=0.0,
        ),
        _features(n_atoms=3),
    )
    immediate.apply_pre_activation_bias(xyz, step_idx=0)

    assert immediate.activated_step == 0
    assert immediate.should_rebuild_static_cache_from_promoted_step(step_idx=0)


def test_delayed_static_cache_rebuild_waits_for_regular_activation(tmp_path):
    motif = tmp_path / "motif.pdb"
    _write_ca_pdb(motif, [(0, 0, 0), (1, 0, 0), (2, 0, 0)])
    xyz = torch.tensor(
        [[[10.0, 10.0, 0.0], [11.0, 10.5, 0.0], [12.0, 10.0, 0.0]]]
    )
    controller = build_motif_unindexing_controller(
        MotifUnindexingConfig(
            enabled=True,
            motif_pdbs=[str(motif)],
            promote_immediately_on_assignment=True,
            floating_project_on_activation=True,
            rebuild_static_cache_after_activation=True,
            activation_threshold=0.0,
        ),
        _features(n_atoms=3),
    )

    controller.apply_pre_activation_bias(xyz, step_idx=0)
    assert controller.activated_step is None
    assert not controller.should_rebuild_static_cache_from_promoted_step(step_idx=0)

    controller.activated_step = 3
    assert controller.should_rebuild_static_cache_from_promoted_step(step_idx=4)


def test_stop_step_static_cache_rebuild_waits_until_final_projection_step(tmp_path):
    motif = tmp_path / "motif.pdb"
    _write_ca_pdb(motif, [(0, 0, 0), (1, 0, 0), (2, 0, 0)])
    xyz = torch.zeros(1, 3, 3)
    controller = build_motif_unindexing_controller(
        MotifUnindexingConfig(
            enabled=True,
            motif_pdbs=[str(motif)],
            promote_immediately_on_assignment=True,
            floating_project_on_activation=True,
            rebuild_static_cache_at_projection_stop=True,
            post_activation_stop_after=160,
        ),
        _features(n_atoms=3),
    )
    controller.assignments = controller.find_assignments(xyz)
    controller.activated_step = 48

    assert not controller.should_rebuild_static_cache_from_promoted_step(step_idx=158)
    assert controller.should_rebuild_static_cache_from_promoted_step(step_idx=159)
    assert not controller.should_rebuild_static_cache_from_promoted_step(step_idx=160)
    assert not controller.should_rebuild_static_cache_from_promoted_step(step_idx=161)


def test_ca_only_promotion_when_no_residue_atoms(tmp_path):
    """CA-only motif (no backbone atoms) should still populate ref_motif_token_type."""
    motif = tmp_path / "motif_ca.pdb"
    _write_ca_pdb(motif, [(0, 0, 0), (1, 0, 0), (2, 0, 0)])
    # xyz exactly at motif positions → RMSD ≈ 0 → immediate activation
    xyz = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]])
    controller = build_motif_unindexing_controller(
        MotifUnindexingConfig(
            enabled=True,
            motif_pdbs=[str(motif)],
            activation_threshold=0.1,
            promote_to_motif_on_activation=True,
        ),
        _features(n_atoms=3),
        sample_features=None,
    )
    controller.apply_pre_activation_bias(xyz, step_idx=0)
    assert controller.activated_step == 0

    f = {
        **_features(n_atoms=3),
        "ref_motif_token_type": torch.tensor(
            [[1, 0, 0]] * 3, dtype=torch.float32
        ),
        "is_motif_atom": torch.zeros(3, dtype=torch.bool),
        "ref_mask": torch.zeros(3, dtype=torch.bool),
    }
    promoted = controller.promoted_feature_dict(f, xyz, step_idx=1)

    assert (promoted["ref_motif_token_type"][:, 1] == 1).all(), (
        "motif-class column should be set for all 3 tokens"
    )
    assert promoted["is_motif_atom"].all()
    assert promoted["ref_mask"].all()


def test_boundary_distance_bias_moves_only_adjacent_scaffold_ca(tmp_path):
    motif = tmp_path / "motif.pdb"
    _write_ca_pdb(motif, [(0, 0, 0), (1, 0, 0), (2, 0, 0)])
    xyz = torch.tensor(
        [[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [11.0, 0.0, 0.0], [12.0, 0.0, 0.0], [30.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )
    controller = build_motif_unindexing_controller(
        MotifUnindexingConfig(
            enabled=True,
            motif_pdbs=[str(motif)],
            boundary_distance_bias_weight=0.1,
            boundary_distance_target=3.8,
        ),
        _features(n_atoms=5),
    )
    controller.assignments = [
        MotifAssignment(
            motif_index=0,
            sample_atom_indices=torch.tensor([1, 2, 3], dtype=torch.long),
            reference_xyz=torch.tensor(
                [(0, 0, 0), (1, 0, 0), (2, 0, 0)], dtype=torch.float32
            ),
            rmsd=0.0,
        )
    ]
    controller.activated_step = 0

    updated = controller.apply_boundary_distance_bias(xyz, step_idx=0)

    assert torch.equal(updated[:, 1:4], xyz[:, 1:4])
    assert torch.equal(updated[:, 2:3], xyz[:, 2:3])
    before_left = torch.linalg.norm(xyz[:, 0] - xyz[:, 1])
    after_left = torch.linalg.norm(updated[:, 0] - updated[:, 1])
    before_right = torch.linalg.norm(xyz[:, 4] - xyz[:, 3])
    after_right = torch.linalg.norm(updated[:, 4] - updated[:, 3])
    assert abs(after_left.item() - 3.8) < abs(before_left.item() - 3.8)
    assert abs(after_right.item() - 3.8) < abs(before_right.item() - 3.8)
    assert not torch.equal(updated[:, 0], xyz[:, 0])
    assert not torch.equal(updated[:, 4], xyz[:, 4])

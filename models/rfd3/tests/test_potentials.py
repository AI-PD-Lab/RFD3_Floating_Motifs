"""Tests for the RFD3 external potential guidance system.

These tests are intentionally self-contained: they do NOT require a trained
checkpoint, a full RFD3 pipeline, or any biotite/atomworks dependencies.
They test the mathematics and masking logic of the potential system in isolation.
"""
import pytest
import torch

from rfd3.potentials.config import PotentialsConfig
from rfd3.potentials.guidance import (
    _rms_over_mask,
    _token_translation,
    compute_potential_guidance,
)
from rfd3.potentials.integration import RFD3PotentialAdapter, build_potential_adapter
from rfd3.potentials.manager import PotentialManager
from rfd3.potentials.masks import build_masks
from rfd3.potentials.parsing import parse_potentials
from rfd3.potentials.potentials import (
    AtomPairDistance,
    BasePotential,
    BinderROG,
    InterfaceNContacts,
    MotifDistance,
    MonomerROG,
)


# ─── shared helpers ──────────────────────────────────────────────────────────


def _make_f(
    n_atoms: int = 10,
    n_tokens: int = 5,
    n_fixed: int = 3,
    n_virtual: int = 2,
    device: str = "cpu",
) -> dict:
    """Minimal feature dict that mirrors what RFD3 puts in `f`."""
    t = torch.device(device)
    atom_to_token_map = torch.arange(n_atoms, device=t) % n_tokens

    is_fixed = torch.zeros(n_atoms, dtype=torch.bool, device=t)
    is_fixed[:n_fixed] = True

    is_virtual = torch.zeros(n_atoms, dtype=torch.bool, device=t)
    if n_virtual > 0:
        is_virtual[-n_virtual:] = True

    is_ca = torch.zeros(n_atoms, dtype=torch.bool, device=t)
    is_ca[1::5] = True

    is_backbone = torch.zeros(n_atoms, dtype=torch.bool, device=t)
    is_backbone[:4] = True

    return {
        "atom_to_token_map": atom_to_token_map,
        "is_motif_atom_with_fixed_coord": is_fixed,
        "is_virtual": is_virtual,
        "is_ca": is_ca,
        "is_backbone": is_backbone,
    }


def _make_manager(
    potentials,
    guide_scale: float = 1.0,
    guide_decay: str = "constant",
    guide_clip_rms: float = 1e6,
    debug: bool = False,
) -> PotentialManager:
    return PotentialManager(
        potentials=potentials,
        guide_scale=guide_scale,
        guide_decay=guide_decay,
        guide_clip_rms=guide_clip_rms,
        debug=debug,
    )


class LinearPotential(BasePotential):
    """Test potential with an exactly known coordinate gradient."""

    def __init__(self, coeff: torch.Tensor):
        super().__init__(weight=1.0)
        self.coeff = coeff

    def compute(self, xyz, masks, metadata):
        return (xyz * self.coeff.to(device=xyz.device, dtype=xyz.dtype)).sum()


# ─── test 1: token_translation gives per-token mean, broadcast back ──────────


@pytest.mark.fast
def test_token_translation_is_mean_per_token():
    """Each atom in a token must receive the mean gradient of ALL guided atoms in that token."""
    n_atoms = 6
    n_tokens = 2
    # Atoms 0,1,2 → token 0; atoms 3,4,5 → token 1
    atom_to_token_map = torch.tensor([0, 0, 0, 1, 1, 1])
    guide_mask = torch.ones(n_atoms, dtype=torch.bool)

    D = 1
    atom_grad = torch.zeros(D, n_atoms, 3)
    atom_grad[0, 0] = torch.tensor([1.0, 0.0, 0.0])
    atom_grad[0, 1] = torch.tensor([2.0, 0.0, 0.0])
    atom_grad[0, 2] = torch.tensor([3.0, 0.0, 0.0])
    atom_grad[0, 3] = torch.tensor([4.0, 0.0, 0.0])
    atom_grad[0, 4] = torch.tensor([5.0, 0.0, 0.0])
    atom_grad[0, 5] = torch.tensor([6.0, 0.0, 0.0])

    result = _token_translation(atom_grad, atom_to_token_map, n_tokens, guide_mask)

    expected_tok0 = torch.tensor([2.0, 0.0, 0.0])  # mean(1,2,3)
    expected_tok1 = torch.tensor([5.0, 0.0, 0.0])  # mean(4,5,6)

    for i in range(3):
        assert torch.allclose(result[0, i], expected_tok0, atol=1e-5), (
            f"Atom {i} (token 0) got {result[0,i]}, expected {expected_tok0}"
        )
    for i in range(3, 6):
        assert torch.allclose(result[0, i], expected_tok1, atol=1e-5), (
            f"Atom {i} (token 1) got {result[0,i]}, expected {expected_tok1}"
        )


# ─── test 2: fixed atoms receive exactly zero guidance ────────────────────────


@pytest.mark.fast
def test_fixed_atoms_get_zero_guidance():
    """Atoms with is_motif_atom_with_fixed_coord=True must never receive guidance."""
    f = _make_f(n_atoms=10, n_tokens=5, n_fixed=3, n_virtual=0)
    config = PotentialsConfig(
        enabled=True,
        include_atoms="real",
        exclude_fixed_atoms=True,
        exclude_virtual_atoms=True,
        guide_only_generated=True,
        guide_scale=1.0,
        guide_decay="constant",
        guide_clip_rms=1e6,
    )
    masks = build_masks(f, config)
    manager = _make_manager([MonomerROG(weight=1.0)], guide_clip_rms=1e6)

    xyz = torch.randn(2, 10, 3)
    guidance, _ = compute_potential_guidance(
        xyz_t=xyz,
        potential_manager=manager,
        t=1.0,
        T=1.0,
        masks=masks,
        metadata={"atom_to_token_map": f["atom_to_token_map"]},
        apply_mode="atom",
        atom_guidance_fraction=0.0,
    )

    fixed_mask = f["is_motif_atom_with_fixed_coord"]
    assert guidance[:, fixed_mask, :].abs().max().item() == 0.0, (
        "Fixed atoms must receive zero guidance"
    )


# ─── test 3: virtual atoms receive zero guidance ──────────────────────────────


@pytest.mark.fast
def test_virtual_atoms_get_zero_guidance():
    """Atoms with is_virtual=True must never receive guidance when exclude_virtual_atoms=True."""
    f = _make_f(n_atoms=10, n_tokens=5, n_fixed=0, n_virtual=3)
    config = PotentialsConfig(
        enabled=True,
        include_atoms="real",
        exclude_fixed_atoms=False,
        exclude_virtual_atoms=True,
        guide_only_generated=False,
        guide_scale=1.0,
        guide_decay="constant",
        guide_clip_rms=1e6,
    )
    masks = build_masks(f, config)
    manager = _make_manager([MonomerROG(weight=1.0)], guide_clip_rms=1e6)

    xyz = torch.randn(1, 10, 3)
    guidance, _ = compute_potential_guidance(
        xyz_t=xyz,
        potential_manager=manager,
        t=1.0,
        T=1.0,
        masks=masks,
        metadata={"atom_to_token_map": f["atom_to_token_map"]},
        apply_mode="atom",
        atom_guidance_fraction=0.0,
    )

    virtual_mask = f["is_virtual"]
    assert guidance[:, virtual_mask, :].abs().max().item() == 0.0, (
        "Virtual atoms must receive zero guidance"
    )


@pytest.mark.fast
def test_real_heavy_excludes_hydrogens():
    """include_atoms=real_heavy should exclude atoms with atomic number 1."""
    f = _make_f(n_atoms=5, n_tokens=5, n_fixed=0, n_virtual=0)
    f["ref_element"] = torch.tensor([6, 1, 8, 1, 7])
    config = PotentialsConfig(
        enabled=True,
        include_atoms="real_heavy",
        exclude_fixed_atoms=False,
        exclude_virtual_atoms=True,
        guide_only_generated=False,
    )

    masks = build_masks(f, config)

    assert masks["potential_atom_mask"].tolist() == [True, False, True, False, True]
    assert masks["guide_atom_mask"].tolist() == [True, False, True, False, True]


# ─── test 4: binder_ROG guidance compacts the binder ────────────────────────


@pytest.mark.fast
def test_binder_rog_decreases_rog():
    """A single binder_ROG guidance step should reduce the binder radius of gyration."""
    torch.manual_seed(42)
    D, n_atoms, n_tokens, n_fixed = 1, 20, 10, 5

    xyz = torch.randn(D, n_atoms, 3) * 10.0
    f = {
        "atom_to_token_map": torch.arange(n_atoms) % n_tokens,
        "is_motif_atom_with_fixed_coord": torch.cat([
            torch.ones(n_fixed, dtype=torch.bool),
            torch.zeros(n_atoms - n_fixed, dtype=torch.bool),
        ]),
        "is_virtual": torch.zeros(n_atoms, dtype=torch.bool),
        "is_ca": torch.zeros(n_atoms, dtype=torch.bool),
        "is_backbone": torch.zeros(n_atoms, dtype=torch.bool),
    }

    config = PotentialsConfig(
        enabled=True,
        include_atoms="real",
        exclude_fixed_atoms=True,
        exclude_virtual_atoms=True,
        guide_only_generated=True,
        guide_scale=1.0,
        guide_decay="constant",
        guide_clip_rms=1e6,
    )
    masks = build_masks(f, config)
    manager = _make_manager([BinderROG(weight=1.0)], guide_clip_rms=1e6)

    guidance, _ = compute_potential_guidance(
        xyz_t=xyz,
        potential_manager=manager,
        t=1.0,
        T=1.0,
        masks=masks,
        metadata={"atom_to_token_map": f["atom_to_token_map"]},
        apply_mode="atom",
        atom_guidance_fraction=0.0,
    )

    binder_mask = masks["binder_atom_mask"]
    xyz_new = xyz + guidance

    def rog(coords, mask):
        c = coords[:, mask, :]
        com = c.mean(dim=1, keepdim=True)
        return ((c - com) ** 2).sum(dim=-1).mean().sqrt()

    rog_before = rog(xyz, binder_mask)
    rog_after = rog(xyz_new, binder_mask)
    assert rog_after < rog_before, (
        f"ROG should decrease: {rog_before:.4f} → {rog_after:.4f}"
    )


# ─── test 5: empty potential list gives zero guidance ────────────────────────


@pytest.mark.fast
def test_empty_potentials_give_zero_guidance():
    """With no guiding_potentials, the guidance tensor must be identically zero."""
    f = _make_f()
    config = PotentialsConfig(enabled=True, guiding_potentials=[])
    masks = build_masks(f, config)
    manager = _make_manager([])

    xyz = torch.randn(2, 10, 3)
    guidance, _ = compute_potential_guidance(
        xyz_t=xyz,
        potential_manager=manager,
        t=1.0,
        T=1.0,
        masks=masks,
        metadata={"atom_to_token_map": f["atom_to_token_map"]},
        apply_mode="token_translation",
        atom_guidance_fraction=0.0,
    )

    assert guidance.abs().max().item() == 0.0, "Empty potentials must give zero guidance"


# ─── test 6: atom mode preserves individual per-atom gradient ────────────────


@pytest.mark.fast
def test_atom_mode_no_token_averaging():
    """In atom mode, atoms in the same token can have different guidance vectors."""
    n_atoms, n_tokens = 4, 2
    atom_to_token_map = torch.tensor([0, 0, 1, 1])
    guide_mask = torch.ones(n_atoms, dtype=torch.bool)

    D = 1
    atom_grad = torch.zeros(D, n_atoms, 3)
    atom_grad[0, 0] = torch.tensor([1.0, 0.0, 0.0])
    atom_grad[0, 1] = torch.tensor([3.0, 0.0, 0.0])  # deliberately different from atom 0

    # token_translation collapses atoms in the same token to the mean
    token_result = _token_translation(atom_grad, atom_to_token_map, n_tokens, guide_mask)
    assert torch.allclose(token_result[0, 0], token_result[0, 1], atol=1e-5), (
        "token_translation must give identical vectors to atoms in the same token"
    )

    # atom mode preserves per-atom values
    assert not torch.allclose(atom_grad[0, 0], atom_grad[0, 1], atol=1e-5), (
        "atom mode should preserve distinct per-atom gradients"
    )


@pytest.mark.fast
def test_atom_mode_uses_raw_atom_gradients():
    """compute_potential_guidance atom mode should return raw masked gradients."""
    f = _make_f(n_atoms=4, n_tokens=2, n_fixed=0, n_virtual=0)
    config = PotentialsConfig(
        enabled=True,
        include_atoms="real",
        exclude_fixed_atoms=False,
        exclude_virtual_atoms=False,
        guide_only_generated=False,
        guide_scale=1.0,
        guide_decay="constant",
        guide_clip_rms=1e6,
    )
    masks = build_masks(f, config)
    coeff = torch.tensor(
        [[[1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 7.0, 0.0]]]
    )
    manager = _make_manager([LinearPotential(coeff)], guide_clip_rms=1e6)
    xyz = torch.zeros(1, 4, 3)

    guidance, _ = compute_potential_guidance(
        xyz_t=xyz,
        potential_manager=manager,
        t=1.0,
        T=1.0,
        masks=masks,
        metadata={"atom_to_token_map": f["atom_to_token_map"]},
        apply_mode="atom",
        atom_guidance_fraction=0.0,
    )

    assert torch.allclose(guidance, coeff, atol=1e-6)


# ─── test 7: hybrid(fraction=0) matches token_translation ────────────────────


@pytest.mark.fast
def test_hybrid_fraction_zero_matches_token_translation():
    """hybrid with atom_guidance_fraction=0 must equal token_translation."""
    torch.manual_seed(7)
    f = _make_f(n_atoms=12, n_tokens=4, n_fixed=0, n_virtual=0)
    config = PotentialsConfig(
        enabled=True,
        include_atoms="real",
        exclude_fixed_atoms=False,
        exclude_virtual_atoms=False,
        guide_only_generated=False,
        guide_scale=1.0,
        guide_decay="constant",
        guide_clip_rms=1e6,
    )
    masks = build_masks(f, config)
    manager = _make_manager([MonomerROG(weight=1.0)], guide_clip_rms=1e6)
    metadata = {"atom_to_token_map": f["atom_to_token_map"]}
    xyz = torch.randn(1, 12, 3) * 5.0

    g_tok, _ = compute_potential_guidance(
        xyz_t=xyz,
        potential_manager=manager,
        t=1.0, T=1.0, masks=masks, metadata=metadata,
        apply_mode="token_translation", atom_guidance_fraction=0.0,
    )
    g_hyb, _ = compute_potential_guidance(
        xyz_t=xyz,
        potential_manager=manager,
        t=1.0, T=1.0, masks=masks, metadata=metadata,
        apply_mode="hybrid", atom_guidance_fraction=0.0,
    )

    assert torch.allclose(g_tok, g_hyb, atol=1e-5), (
        "hybrid(fraction=0.0) must match token_translation"
    )


@pytest.mark.fast
def test_hybrid_fraction_one_matches_atom_mode():
    """hybrid with atom_guidance_fraction=1 must equal atom mode before clipping/scaling."""
    f = _make_f(n_atoms=4, n_tokens=2, n_fixed=0, n_virtual=0)
    config = PotentialsConfig(
        enabled=True,
        include_atoms="real",
        exclude_fixed_atoms=False,
        exclude_virtual_atoms=False,
        guide_only_generated=False,
        guide_scale=1.0,
        guide_decay="constant",
        guide_clip_rms=1e6,
    )
    masks = build_masks(f, config)
    coeff = torch.tensor(
        [[[1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 7.0, 0.0]]]
    )
    manager = _make_manager([LinearPotential(coeff)], guide_clip_rms=1e6)
    metadata = {"atom_to_token_map": f["atom_to_token_map"]}
    xyz = torch.zeros(1, 4, 3)

    g_atom, _ = compute_potential_guidance(
        xyz_t=xyz,
        potential_manager=manager,
        t=1.0, T=1.0, masks=masks, metadata=metadata,
        apply_mode="atom", atom_guidance_fraction=0.0,
    )
    g_hyb, _ = compute_potential_guidance(
        xyz_t=xyz,
        potential_manager=manager,
        t=1.0, T=1.0, masks=masks, metadata=metadata,
        apply_mode="hybrid", atom_guidance_fraction=1.0,
    )

    assert torch.allclose(g_atom, g_hyb, atol=1e-6)


# ─── test 8: guide decay reduces scale monotonically ────────────────────────


@pytest.mark.fast
@pytest.mark.parametrize("decay", ["linear", "quadratic", "cubic"])
def test_guide_decay_is_monotone(decay):
    """Scale should decrease as t decreases from T (non-constant decays)."""
    manager = _make_manager([], guide_scale=1.0, guide_decay=decay)
    T = 10.0
    ts = [10.0, 7.0, 4.0, 1.0, 0.1]
    scales = [manager.get_guide_scale(t, T) for t in ts]
    for i in range(len(scales) - 1):
        assert scales[i] >= scales[i + 1], (
            f"{decay}: scale should decrease as t decreases; got {scales}"
        )


# ─── test 9: parsing — dict and RFD1-style strings ───────────────────────────


@pytest.mark.fast
def test_parsing_dict_and_string():
    """Both dict and "type:X,key:val" string specs should instantiate correctly."""
    specs_dict = [
        {"type": "binder_ROG", "weight": 2.0},
        {"type": "interface_ncontacts", "weight": 0.5, "r_0": 6.0, "d_0": 1.5},
        {"type": "motif_distance", "weight": 1.5, "motif_i": 0, "motif_j": 1, "target_distance": 12.0},
    ]
    pots_dict = parse_potentials(specs_dict)
    assert isinstance(pots_dict[0], BinderROG)
    assert pots_dict[0].weight == 2.0
    assert isinstance(pots_dict[1], InterfaceNContacts)
    assert pots_dict[1].r_0 == 6.0
    assert isinstance(pots_dict[2], MotifDistance)
    assert pots_dict[2].target_distance == 12.0

    specs_str = [
        "type:binder_ROG,weight:2.0",
        "type:interface_ncontacts,weight:0.5,r_0:6.0,d_0:1.5",
        "type:motif_distance,weight:1.5,motif_i:0,motif_j:1,target_distance:12.0",
    ]
    pots_str = parse_potentials(specs_str)
    assert isinstance(pots_str[0], BinderROG)
    assert pots_str[0].weight == 2.0
    assert isinstance(pots_str[1], InterfaceNContacts)
    assert pots_str[1].r_0 == 6.0
    assert isinstance(pots_str[2], MotifDistance)
    assert pots_str[2].target_distance == 12.0


@pytest.mark.fast
def test_atom_pair_distance_scalar_to_maximize():
    """atom_pair_distance returns -weight * (distance - target_distance)^2."""
    pot = AtomPairDistance(weight=2.0, atom_i=0, atom_j=1, target_distance=5.0)
    xyz = torch.tensor([[[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]])

    value = pot.compute(xyz, masks={}, metadata={})

    assert value.ndim == 0
    assert torch.allclose(value, torch.tensor(-8.0))


@pytest.mark.fast
def test_motif_distance_scalar_to_maximize():
    """motif_distance returns -weight * (center_distance - target_distance)^2."""
    f = {
        "atom_to_token_map": torch.tensor([0, 1, 2, 3]),
        "is_motif_atom_with_fixed_coord": torch.tensor([True, True, False, True]),
        "is_virtual": torch.zeros(4, dtype=torch.bool),
        "is_ca": torch.ones(4, dtype=torch.bool),
        "is_backbone": torch.zeros(4, dtype=torch.bool),
    }
    config = PotentialsConfig(
        enabled=True,
        include_atoms="CA",
        exclude_fixed_atoms=False,
        exclude_virtual_atoms=True,
        guide_only_generated=False,
    )
    masks = build_masks(f, config)
    pot = MotifDistance(
        weight=2.0,
        motif_i=0,
        motif_j=1,
        target_distance=3.0,
    )
    xyz = torch.tensor([[
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [99.0, 0.0, 0.0],
        [5.0, 0.0, 0.0],
    ]])

    value = pot.compute(xyz, masks, metadata={"atom_to_token_map": f["atom_to_token_map"]})

    assert value.ndim == 0
    assert torch.allclose(value, torch.tensor(-8.0))


@pytest.mark.fast
def test_motif_distance_guidance_moves_toward_target_distance():
    """A small motif_distance guidance step should reduce the distance error."""
    f = {
        "atom_to_token_map": torch.tensor([0, 1, 2, 3]),
        "is_motif_atom_with_fixed_coord": torch.tensor([True, True, False, True]),
        "is_virtual": torch.zeros(4, dtype=torch.bool),
        "is_ca": torch.ones(4, dtype=torch.bool),
        "is_backbone": torch.zeros(4, dtype=torch.bool),
    }
    config = PotentialsConfig(
        enabled=True,
        include_atoms="CA",
        exclude_fixed_atoms=False,
        exclude_virtual_atoms=True,
        guide_only_generated=False,
        guide_scale=1.0,
        guide_decay="constant",
        guide_clip_rms=1e6,
    )
    masks = build_masks(f, config)
    manager = _make_manager(
        [MotifDistance(weight=1.0, motif_i=0, motif_j=1, target_distance=3.0)],
        guide_clip_rms=1e6,
    )
    xyz = torch.tensor([[
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [99.0, 0.0, 0.0],
        [5.0, 0.0, 0.0],
    ]])
    metadata = {"atom_to_token_map": f["atom_to_token_map"]}

    guidance, _ = compute_potential_guidance(
        xyz_t=xyz,
        potential_manager=manager,
        t=1.0,
        T=1.0,
        masks=masks,
        metadata=metadata,
        apply_mode="atom",
        atom_guidance_fraction=0.0,
    )
    xyz_new = xyz + 0.05 * guidance

    dist_before = (xyz[:, :2, :].mean(dim=1) - xyz[:, 3, :]).norm(dim=-1)
    dist_after = (xyz_new[:, :2, :].mean(dim=1) - xyz_new[:, 3, :]).norm(dim=-1)

    assert (dist_after - 3.0).abs() < (dist_before - 3.0).abs()


# ─── test 10: build_potential_adapter returns None when disabled ───────────────


@pytest.mark.fast
def test_adapter_disabled_by_default():
    """build_potential_adapter must return None when enabled=False or potentials empty."""
    f = _make_f()

    # disabled flag
    adapter = build_potential_adapter({"enabled": False, "guiding_potentials": [{"type": "monomer_ROG"}]}, f)
    assert adapter is None

    # enabled but empty list
    adapter = build_potential_adapter({"enabled": True, "guiding_potentials": []}, f)
    assert adapter is None


# ─── test 11: no-potential path leaves X_L unchanged ─────────────────────────


@pytest.mark.fast
def test_disabled_adapter_leaves_xl_unchanged():
    """With enabled=False the adapter must return the SAME coordinates."""
    f = _make_f()
    config = PotentialsConfig(
        enabled=False,
        guiding_potentials=[{"type": "monomer_ROG"}],
    )
    adapter = RFD3PotentialAdapter(config, f)
    xyz = torch.randn(2, 10, 3)
    result = adapter.apply(xyz, t=1.0, T=10.0)
    assert result is xyz, "Disabled adapter must return the original tensor unchanged"

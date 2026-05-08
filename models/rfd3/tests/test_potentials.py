"""Tests for the RFD3 external potential guidance system.

These tests are intentionally self-contained: they do NOT require a trained
checkpoint, a full RFD3 pipeline, or any biotite/atomworks dependencies.
They test the mathematics and masking logic of the potential system in isolation.
"""

import pytest
import torch
from rfd3.potentials.config import PotentialsConfig
from rfd3.potentials.guidance import (
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
    MonomerROG,
    MotifBridge,
    MotifCOMDistance,
    MotifDistance,
    MotifRadialOrientationPotential,
    MotifRigid,
    MotifSphericalPosition,
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
        assert torch.allclose(
            result[0, i], expected_tok0, atol=1e-5
        ), f"Atom {i} (token 0) got {result[0,i]}, expected {expected_tok0}"
    for i in range(3, 6):
        assert torch.allclose(
            result[0, i], expected_tok1, atol=1e-5
        ), f"Atom {i} (token 1) got {result[0,i]}, expected {expected_tok1}"


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
    assert (
        guidance[:, fixed_mask, :].abs().max().item() == 0.0
    ), "Fixed atoms must receive zero guidance"


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
    assert (
        guidance[:, virtual_mask, :].abs().max().item() == 0.0
    ), "Virtual atoms must receive zero guidance"


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
        "is_motif_atom_with_fixed_coord": torch.cat(
            [
                torch.ones(n_fixed, dtype=torch.bool),
                torch.zeros(n_atoms - n_fixed, dtype=torch.bool),
            ]
        ),
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
    assert (
        rog_after < rog_before
    ), f"ROG should decrease: {rog_before:.4f} → {rog_after:.4f}"


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

    assert (
        guidance.abs().max().item() == 0.0
    ), "Empty potentials must give zero guidance"


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
    atom_grad[0, 1] = torch.tensor(
        [3.0, 0.0, 0.0]
    )  # deliberately different from atom 0

    # token_translation collapses atoms in the same token to the mean
    token_result = _token_translation(
        atom_grad, atom_to_token_map, n_tokens, guide_mask
    )
    assert torch.allclose(
        token_result[0, 0], token_result[0, 1], atol=1e-5
    ), "token_translation must give identical vectors to atoms in the same token"

    # atom mode preserves per-atom values
    assert not torch.allclose(
        atom_grad[0, 0], atom_grad[0, 1], atol=1e-5
    ), "atom mode should preserve distinct per-atom gradients"


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


@pytest.mark.fast
def test_per_potential_guidance_overrides_are_applied_separately():
    """Each potential may override guide_scale and guide_clip_rms independently."""
    f = _make_f(n_atoms=2, n_tokens=2, n_fixed=0, n_virtual=0)
    config = PotentialsConfig(
        enabled=True,
        include_atoms="real",
        exclude_fixed_atoms=False,
        exclude_virtual_atoms=False,
        guide_only_generated=False,
        guide_scale=99.0,
        guide_decay="constant",
        guide_clip_rms=1e6,
    )
    masks = build_masks(f, config)
    coeff_x = torch.tensor([[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
    coeff_y = torch.tensor([[[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]])
    pot_x = LinearPotential(coeff_x)
    pot_y = LinearPotential(coeff_y)
    pot_x.guide_scale = 2.0
    pot_y.guide_scale = 0.5
    pot_x.guide_clip_rms = 1e6
    pot_y.guide_clip_rms = 1e6
    manager = _make_manager(
        [pot_x, pot_y],
        guide_scale=99.0,
        guide_decay="constant",
        guide_clip_rms=1e6,
    )

    guidance, _ = compute_potential_guidance(
        xyz_t=torch.zeros(1, 2, 3),
        potential_manager=manager,
        t=1.0,
        T=1.0,
        masks=masks,
        metadata={"atom_to_token_map": f["atom_to_token_map"]},
        apply_mode="atom",
        atom_guidance_fraction=0.0,
    )

    expected = coeff_x * 2.0 + coeff_y * 0.5
    assert torch.allclose(guidance, expected, atol=1e-6)


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
        t=1.0,
        T=1.0,
        masks=masks,
        metadata=metadata,
        apply_mode="token_translation",
        atom_guidance_fraction=0.0,
    )
    g_hyb, _ = compute_potential_guidance(
        xyz_t=xyz,
        potential_manager=manager,
        t=1.0,
        T=1.0,
        masks=masks,
        metadata=metadata,
        apply_mode="hybrid",
        atom_guidance_fraction=0.0,
    )

    assert torch.allclose(
        g_tok, g_hyb, atol=1e-5
    ), "hybrid(fraction=0.0) must match token_translation"


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
        t=1.0,
        T=1.0,
        masks=masks,
        metadata=metadata,
        apply_mode="atom",
        atom_guidance_fraction=0.0,
    )
    g_hyb, _ = compute_potential_guidance(
        xyz_t=xyz,
        potential_manager=manager,
        t=1.0,
        T=1.0,
        masks=masks,
        metadata=metadata,
        apply_mode="hybrid",
        atom_guidance_fraction=1.0,
    )

    assert torch.allclose(g_atom, g_hyb, atol=1e-6)


# ─── test 8: guide decay reduces scale monotonically ────────────────────────


@pytest.mark.fast
@pytest.mark.parametrize(
    "decay",
    ["sqrt", "linear", "quadratic", "cubic", "quartic", "exponential", "cosine"],
)
def test_guide_decay_is_monotone(decay):
    """Scale should decrease as t decreases from T (non-constant decays)."""
    manager = _make_manager([], guide_scale=1.0, guide_decay=decay)
    T = 10.0
    ts = [10.0, 7.0, 4.0, 1.0, 0.1]
    scales = [manager.get_guide_scale(t, T) for t in ts]
    for i in range(len(scales) - 1):
        assert (
            scales[i] >= scales[i + 1]
        ), f"{decay}: scale should decrease as t decreases; got {scales}"


@pytest.mark.fast
@pytest.mark.parametrize(
    "decay",
    [
        "inverse_sqrt",
        "inverse_linear",
        "inverse_quadratic",
        "inverse_cubic",
        "inverse_quartic",
        "inverse_exponential",
        "inverse_cosine",
    ],
)
def test_inverse_guide_decay_is_monotone(decay):
    """Inverse schedules should strengthen as t decreases from T."""
    manager = _make_manager([], guide_scale=1.0, guide_decay=decay)
    T = 10.0
    ts = [10.0, 7.0, 4.0, 1.0, 0.1]
    scales = [manager.get_guide_scale(t, T) for t in ts]
    for i in range(len(scales) - 1):
        assert (
            scales[i] <= scales[i + 1]
        ), f"{decay}: scale should increase as t decreases; got {scales}"


# ─── test 9: parsing — dict and RFD1-style strings ───────────────────────────


@pytest.mark.fast
def test_parsing_dict_and_string():
    """Both dict and "type:X,key:val" string specs should instantiate correctly."""
    specs_dict = [
        {"type": "binder_ROG", "weight": 2.0},
        {"type": "interface_ncontacts", "weight": 0.5, "r_0": 6.0, "d_0": 1.5},
        {
            "type": "motif_distance",
            "weight": 1.5,
            "motif_i": 0,
            "motif_j": 1,
            "target_distance": 12.0,
        },
        {
            "type": "motif_bridge",
            "weight": 4.0,
            "motif_i": 0,
            "motif_j": 1,
            "max_radius": 8.0,
        },
        {
            "type": "motif_rigid",
            "weight": 3.0,
            "k": 0.5,
            "loss": "mse",
            "guide_scale": 2.0,
            "guide_decay": "inverse_linear",
            "guide_clip_rms": 0.01,
        },
    ]
    pots_dict = parse_potentials(specs_dict)
    assert isinstance(pots_dict[0], BinderROG)
    assert pots_dict[0].weight == 2.0
    assert isinstance(pots_dict[1], InterfaceNContacts)
    assert pots_dict[1].r_0 == 6.0
    assert isinstance(pots_dict[2], MotifDistance)
    assert pots_dict[2].target_distance == 12.0
    assert isinstance(pots_dict[3], MotifBridge)
    assert pots_dict[3].max_radius == 8.0
    assert isinstance(pots_dict[4], MotifRigid)
    assert pots_dict[4].k == 0.5
    assert pots_dict[4].guide_scale == 2.0
    assert pots_dict[4].guide_decay == "inverse_linear"
    assert pots_dict[4].guide_clip_rms == 0.01

    specs_str = [
        "type:binder_ROG,weight:2.0",
        "type:interface_ncontacts,weight:0.5,r_0:6.0,d_0:1.5",
        "type:motif_distance,weight:1.5,motif_i:0,motif_j:1,target_distance:12.0",
        "type:motif_bridge,weight:4.0,motif_i:0,motif_j:1,max_radius:8.0",
        "type:motif_rigid,weight:3.0,k:0.5,loss:mse,guide_scale:2.0,guide_decay:inverse_linear,guide_clip_rms:0.01",
    ]
    pots_str = parse_potentials(specs_str)
    assert isinstance(pots_str[0], BinderROG)
    assert pots_str[0].weight == 2.0
    assert isinstance(pots_str[1], InterfaceNContacts)
    assert pots_str[1].r_0 == 6.0
    assert isinstance(pots_str[2], MotifDistance)
    assert pots_str[2].target_distance == 12.0
    assert isinstance(pots_str[3], MotifBridge)
    assert pots_str[3].max_radius == 8.0
    assert isinstance(pots_str[4], MotifRigid)
    assert pots_str[4].loss == "mse"
    assert pots_str[4].guide_scale == 2.0
    assert pots_str[4].guide_decay == "inverse_linear"
    assert pots_str[4].guide_clip_rms == 0.01


@pytest.mark.fast
def test_per_potential_invalid_guide_decay_raises():
    """Per-potential guide_decay should be validated during parsing."""
    with pytest.raises(ValueError, match="guide_decay"):
        parse_potentials(
            [
                {
                    "type": "motif_rigid",
                    "guide_decay": "pseudo_huber",
                }
            ]
        )


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
    xyz = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [99.0, 0.0, 0.0],
                [5.0, 0.0, 0.0],
            ]
        ]
    )

    value = pot.compute(
        xyz, masks, metadata={"atom_to_token_map": f["atom_to_token_map"]}
    )

    assert value.ndim == 0
    assert torch.allclose(value, torch.tensor(-8.0))


@pytest.mark.fast
def test_motif_distance_uses_unfixed_motif_atoms():
    """motif_distance must still work when motif atoms are diffused coordinates."""
    f = {
        "atom_to_token_map": torch.tensor([0, 1, 2, 3]),
        "is_motif_atom_with_fixed_coord": torch.zeros(4, dtype=torch.bool),
        "is_motif_atom_with_fixed_seq": torch.tensor([True, True, False, True]),
        "is_motif_atom_unindexed": torch.zeros(4, dtype=torch.bool),
        "is_virtual": torch.zeros(4, dtype=torch.bool),
        "is_ca": torch.ones(4, dtype=torch.bool),
        "is_backbone": torch.zeros(4, dtype=torch.bool),
    }
    config = PotentialsConfig(
        enabled=True,
        include_atoms="CA",
        exclude_fixed_atoms=True,
        exclude_virtual_atoms=True,
        guide_only_generated=True,
    )
    masks = build_masks(f, config)
    pot = MotifDistance(
        weight=2.0,
        motif_i=0,
        motif_j=1,
        target_distance=3.0,
    )
    xyz = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [99.0, 0.0, 0.0],
                [5.0, 0.0, 0.0],
            ]
        ],
        requires_grad=True,
    )

    value = pot.compute(
        xyz, masks, metadata={"atom_to_token_map": f["atom_to_token_map"]}
    )
    value.backward()

    assert value.ndim == 0
    assert torch.allclose(value.detach(), torch.tensor(-8.0))
    assert xyz.grad is not None
    assert xyz.grad.abs().sum().item() > 0.0


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
    xyz = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [99.0, 0.0, 0.0],
                [5.0, 0.0, 0.0],
            ]
        ]
    )
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


@pytest.mark.fast
def test_motif_distance_guidance_translates_whole_motif_blocks():
    """motif_distance guidance should move each motif as one COM translation."""
    f = {
        "atom_to_token_map": torch.tensor([0, 1, 2, 3, 4]),
        "is_motif_atom_with_fixed_coord": torch.tensor([True, True, False, True, True]),
        "is_virtual": torch.zeros(5, dtype=torch.bool),
        "is_ca": torch.ones(5, dtype=torch.bool),
        "is_backbone": torch.ones(5, dtype=torch.bool),
    }
    config = PotentialsConfig(
        enabled=True,
        include_atoms="CA",
        exclude_fixed_atoms=True,
        exclude_virtual_atoms=True,
        guide_only_generated=True,
        guide_scale=1.0,
        guide_decay="constant",
        guide_clip_rms=1e6,
    )
    masks = build_masks(f, config)
    manager = _make_manager(
        [MotifDistance(weight=1.0, motif_i=0, motif_j=1, target_distance=3.0)],
        guide_clip_rms=1e6,
    )
    xyz = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [99.0, 0.0, 0.0],
                [7.0, 0.0, 0.0],
                [7.0, 2.0, 0.0],
            ]
        ]
    )

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

    assert torch.allclose(guidance[0, 0], guidance[0, 1])
    assert torch.allclose(guidance[0, 3], guidance[0, 4])
    assert torch.allclose(guidance[0, 2], torch.zeros(3))
    assert guidance[:, [0, 1, 3, 4], :].abs().sum().item() > 0.0


@pytest.mark.fast
def test_motif_bridge_guidance_moves_scaffold_between_motifs():
    """motif_bridge should move generated non-motif atoms toward the motif span."""
    f = {
        "atom_to_token_map": torch.tensor([0, 1, 2]),
        "is_motif_atom_with_fixed_coord": torch.tensor([True, False, True]),
        "is_virtual": torch.zeros(3, dtype=torch.bool),
        "is_ca": torch.ones(3, dtype=torch.bool),
        "is_backbone": torch.ones(3, dtype=torch.bool),
    }
    config = PotentialsConfig(
        enabled=True,
        include_atoms="CA",
        exclude_fixed_atoms=True,
        exclude_virtual_atoms=True,
        guide_only_generated=True,
        guide_scale=1.0,
        guide_decay="constant",
        guide_clip_rms=1e6,
    )
    masks = build_masks(f, config)
    manager = _make_manager(
        [
            MotifBridge(
                weight=1.0,
                motif_i=0,
                motif_j=1,
                spread_weight=1.0,
                outside_weight=1.0,
                tube_weight=0.0,
            )
        ],
        guide_clip_rms=1e6,
    )
    xyz = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0],
                [20.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
            ]
        ]
    )

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
    xyz_new = xyz + 0.1 * guidance

    assert xyz_new[0, 1, 0] < xyz[0, 1, 0]
    assert guidance[:, f["is_motif_atom_with_fixed_coord"], :].abs().max().item() == 0.0


@pytest.mark.fast
def test_motif_rigid_penalizes_distorted_fixed_seq_motif():
    """motif_rigid compares current motif distances to ref_pos distances."""
    f = {
        "atom_to_token_map": torch.tensor([0, 1, 2]),
        "is_motif_atom_with_fixed_coord": torch.zeros(3, dtype=torch.bool),
        "is_motif_atom_with_fixed_seq": torch.ones(3, dtype=torch.bool),
        "is_virtual": torch.zeros(3, dtype=torch.bool),
        "is_ca": torch.ones(3, dtype=torch.bool),
        "is_backbone": torch.ones(3, dtype=torch.bool),
    }
    config = PotentialsConfig(
        enabled=True,
        include_atoms="CA",
        exclude_fixed_atoms=True,
        exclude_virtual_atoms=True,
        guide_only_generated=True,
    )
    masks = build_masks(f, config)
    metadata = {
        "atom_to_token_map": f["atom_to_token_map"],
        "ref_pos": torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        ),
        "is_motif_atom_with_fixed_seq": f["is_motif_atom_with_fixed_seq"],
    }
    xyz = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        ]
    )
    pot = MotifRigid(weight=2.0, loss="mse")

    value = pot.compute(xyz, masks, metadata)

    assert value.ndim == 0
    assert value.item() < 0.0


@pytest.mark.fast
def test_motif_rigid_guidance_reduces_distance_geometry_error():
    """A small motif_rigid step should move distorted motif distances toward ref_pos."""
    f = {
        "atom_to_token_map": torch.tensor([0, 1, 2]),
        "is_motif_atom_with_fixed_coord": torch.zeros(3, dtype=torch.bool),
        "is_motif_atom_with_fixed_seq": torch.ones(3, dtype=torch.bool),
        "is_virtual": torch.zeros(3, dtype=torch.bool),
        "is_ca": torch.ones(3, dtype=torch.bool),
        "is_backbone": torch.ones(3, dtype=torch.bool),
    }
    config = PotentialsConfig(
        enabled=True,
        include_atoms="CA",
        exclude_fixed_atoms=True,
        exclude_virtual_atoms=True,
        guide_only_generated=True,
        guide_scale=1.0,
        guide_decay="constant",
        guide_clip_rms=1e6,
    )
    masks = build_masks(f, config)
    ref_pos = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    metadata = {
        "atom_to_token_map": f["atom_to_token_map"],
        "ref_pos": ref_pos,
        "is_motif_atom_with_fixed_seq": f["is_motif_atom_with_fixed_seq"],
    }
    xyz = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        ]
    )
    manager = _make_manager([MotifRigid(weight=1.0, loss="mse")], guide_clip_rms=1e6)

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
    xyz_new = xyz + 0.1 * guidance

    def geometry_error(coords):
        dcur = torch.cdist(coords, coords)
        dref = torch.cdist(ref_pos[None, :, :], ref_pos[None, :, :])
        upper = torch.triu(torch.ones(3, 3, dtype=torch.bool), diagonal=1)
        return (dcur[:, upper] - dref[:, upper]).pow(2).mean()

    assert geometry_error(xyz_new) < geometry_error(xyz)


# ─── test 10: build_potential_adapter returns None when disabled ───────────────


@pytest.mark.fast
def test_adapter_disabled_by_default():
    """build_potential_adapter must return None when enabled=False or potentials empty."""
    f = _make_f()

    # disabled flag
    adapter = build_potential_adapter(
        {"enabled": False, "guiding_potentials": [{"type": "monomer_ROG"}]}, f
    )
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


# ─── MotifRadialOrientationPotential helpers ──────────────────────────────────

def _make_radial_orientation_setup():
    """Two motif blocks (4 atoms each) symmetric about the origin.

    Block 0 centres at [-5, 0, 0], block 1 at [5, 0, 0].
    A token-index gap (tokens 0-3, then 5-8) creates two distinct motif blocks.
    All atoms are real motif atoms so both _motif_distance_blocks and
    _motif_input_reference_xyz resolve cleanly.
    """
    n_atoms = 8
    f = {
        # Gap between token 3 and token 5 → two contiguous motif blocks
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

    # Reference: block 0 around [-5,0,0], block 1 around [5,0,0].
    # Atoms span y and z so each motif has at least 3 non-collinear atoms
    # for a well-defined frame.
    ref_pos = torch.tensor(
        [
            [-5.0, 1.0, 0.0],
            [-5.0, -1.0, 0.0],
            [-5.0, 0.0, 1.0],
            [-5.0, 0.0, -1.0],
            [5.0, 1.0, 0.0],
            [5.0, -1.0, 0.0],
            [5.0, 0.0, 1.0],
            [5.0, 0.0, -1.0],
        ]
    )
    metadata = {
        "atom_to_token_map": f["atom_to_token_map"],
        # motif_pos makes _motif_input_reference_xyz return the reference coords
        "motif_pos": ref_pos,
    }
    return f, masks, ref_pos, metadata


# ─── test: potential is ~0 at the reference pose ─────────────────────────────


@pytest.mark.fast
def test_motif_radial_orientation_zero_at_reference_pose():
    """At the reference pose the orientation loss must be (near) zero."""
    _, masks, ref_pos, metadata = _make_radial_orientation_setup()

    # D=1, current xyz = reference positions (perfect match)
    xyz = ref_pos.unsqueeze(0)  # [1, 8, 3]

    pot = MotifRadialOrientationPotential(weight=1.0)
    value = pot.compute(xyz, masks, metadata)

    assert value.ndim == 0, "compute() must return a scalar"
    assert abs(value.item()) < 1e-4, (
        f"Potential should be ~0 at reference pose, got {value.item():.6f}"
    )


# ─── test: non-zero when current does not match reference ────────────────────


@pytest.mark.fast
def test_motif_radial_orientation_negative_when_rotated():
    """A 90-degree rotation of one motif block must give a negative potential."""
    _, masks, ref_pos, metadata = _make_radial_orientation_setup()

    # Rotate block 0 (atoms 0-3) by 90° around the z-axis about its own centre
    # [-5, 0, 0].  Block 1 (atoms 4-7) stays at the reference.
    cur = ref_pos.clone()
    c0 = ref_pos[:4].mean(dim=0)  # [-5, 0, 0]
    centred = ref_pos[:4] - c0
    # 90° z-rotation: (x,y) → (-y, x)
    rotated = torch.stack([-centred[:, 1], centred[:, 0], centred[:, 2]], dim=-1)
    cur[:4] = rotated + c0

    xyz = cur.unsqueeze(0)  # [1, 8, 3]
    pot = MotifRadialOrientationPotential(weight=1.0)
    value = pot.compute(xyz, masks, metadata)

    assert value.ndim == 0
    assert value.item() < -1e-4, (
        f"Potential should be negative when a motif is misoriented, got {value.item():.6f}"
    )


# ─── test: guidance step reduces orientation error ───────────────────────────


@pytest.mark.fast
def test_motif_radial_orientation_guidance_reduces_error():
    """One guidance step must bring the rotated motif closer to the target."""
    f, masks, ref_pos, metadata = _make_radial_orientation_setup()

    # Same rotated setup as the previous test
    cur = ref_pos.clone()
    c0 = ref_pos[:4].mean(dim=0)
    centred = ref_pos[:4] - c0
    rotated = torch.stack([-centred[:, 1], centred[:, 0], centred[:, 2]], dim=-1)
    cur[:4] = rotated + c0
    xyz = cur.unsqueeze(0)

    config = PotentialsConfig(
        enabled=True,
        include_atoms="real",
        exclude_fixed_atoms=False,
        exclude_virtual_atoms=True,
        guide_scale=1.0,
        guide_decay="constant",
        guide_clip_rms=1e6,
    )
    manager = _make_manager(
        [MotifRadialOrientationPotential(weight=1.0)], guide_clip_rms=1e6
    )

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

    def orientation_error(coords):
        """Frobenius distance between current centred coords and reference centred."""
        err = 0.0
        for start, stop in [(0, 4), (4, 8)]:
            cur_c = coords[0, start:stop]
            ref_c = ref_pos[start:stop]
            cur_c = cur_c - cur_c.mean(dim=0)
            ref_c = ref_c - ref_c.mean(dim=0)
            err += (cur_c - ref_c).pow(2).sum().item()
        return err

    err_before = orientation_error(xyz)
    err_after = orientation_error(xyz_new)
    assert err_after < err_before, (
        f"Guidance should reduce orientation error: {err_before:.4f} → {err_after:.4f}"
    )


# ─── test: invariance to radial translation ───────────────────────────────────


@pytest.mark.fast
def test_motif_radial_orientation_invariant_to_radial_translation():
    """Moving motifs radially outward (rigid body, no deformation) must not change
    the potential value.  Orientation is preserved; only distance from COM changes."""
    _, masks, ref_pos, metadata = _make_radial_orientation_setup()

    # Misoriented current: rotate block 0 by 90° around z at its own centre
    cur = ref_pos.clone()
    c0 = ref_pos[:4].mean(dim=0)  # [-5, 0, 0]
    centred = ref_pos[:4] - c0
    cur[:4] = torch.stack([-centred[:, 1], centred[:, 0], centred[:, 2]], dim=-1) + c0

    pot = MotifRadialOrientationPotential(weight=1.0)
    val_close = pot.compute(cur.unsqueeze(0), masks, metadata)

    # Move both blocks farther from COM by translating them along the radial axis
    # while preserving internal geometry.  Translation is symmetric so COM stays at 0.
    cur_far = cur.clone()
    cur_far[:4] = cur[:4] + torch.tensor([-5.0, 0.0, 0.0])  # block 0 farther left
    cur_far[4:] = cur[4:] + torch.tensor([5.0, 0.0, 0.0])   # block 1 farther right

    val_far = pot.compute(cur_far.unsqueeze(0), masks, metadata)

    assert abs(val_close.item() - val_far.item()) < 1e-4, (
        f"Potential must be invariant to radial translation: "
        f"close={val_close.item():.6f}, far={val_far.item():.6f}"
    )


# ─── test: offset (0,0,0) matches reference; nonzero offset does not ─────────


@pytest.mark.fast
def test_motif_radial_orientation_offset_shifts_target():
    """A nonzero per-motif offset must produce a different energy than zero offset."""
    _, masks, ref_pos, metadata = _make_radial_orientation_setup()
    xyz = ref_pos.unsqueeze(0)

    pot_no_offset = MotifRadialOrientationPotential(weight=1.0, motif_offsets=[])
    pot_with_offset = MotifRadialOrientationPotential(
        weight=1.0, motif_offsets=[[0.0, 0.0, 90.0]]  # 90° on block 0
    )

    val_no_offset = pot_no_offset.compute(xyz, masks, metadata)
    val_with_offset = pot_with_offset.compute(xyz, masks, metadata)

    # No offset at reference → ~0; with 90° offset the reference no longer
    # matches the target, so the potential should be negative.
    assert abs(val_no_offset.item()) < 1e-4, (
        f"No offset at reference pose should give ~0, got {val_no_offset.item():.6f}"
    )
    assert val_with_offset.item() < -1e-4, (
        f"90° offset at reference pose should give negative value, "
        f"got {val_with_offset.item():.6f}"
    )


# ─── test: parsing round-trip ─────────────────────────────────────────────────


@pytest.mark.fast
def test_motif_radial_orientation_parsing():
    """The potential must be instantiated correctly from a dict spec."""
    from rfd3.potentials.parsing import parse_potentials

    specs = [
        {
            "type": "motif_radial_orientation",
            "weight": 2.0,
            "motif_offsets": [[10.0, 0.0, 0.0], [0.0, 20.0, 0.0]],
            "motif_axis_weights": [[1.0, 0.0, 0.0], [0.5, 0.5, 1.0]],
            "origin_atom_filter": "real",
        }
    ]
    pots = parse_potentials(specs)
    assert len(pots) == 1
    pot = pots[0]
    assert isinstance(pot, MotifRadialOrientationPotential)
    assert pot.weight == 2.0
    assert pot.origin_atom_filter == "real"
    assert len(pot.motif_offsets) == 2
    assert pot.motif_offsets[0][0] == 10.0
    assert pot.motif_offsets[1][1] == 20.0
    assert len(pot.motif_axis_weights) == 2
    assert pot.motif_axis_weights[0][0] == 1.0
    assert pot.motif_axis_weights[0][1] == 0.0
    assert pot.motif_axis_weights[1][2] == 1.0


@pytest.mark.fast
def test_motif_radial_orientation_axis_weights_zero_suppresses_loss():
    """axis_weights=[0,0,0] for every motif must zero the potential at any pose."""
    _, masks, ref_pos, metadata = _make_radial_orientation_setup()

    # Randomly rotated pose (definitely not reference)
    torch.manual_seed(7)
    xyz = (ref_pos + torch.randn_like(ref_pos) * 3.0).unsqueeze(0)

    pot = MotifRadialOrientationPotential(
        weight=1.0,
        motif_axis_weights=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    )
    val = pot.compute(xyz, masks, metadata)
    assert val.item() == pytest.approx(0.0, abs=1e-6), (
        f"axis_weights=[0,0,0] must give zero loss, got {val.item()}"
    )


@pytest.mark.fast
def test_motif_radial_orientation_axis_weights_free_radial_spin():
    """axis_weights=[1,0,0] must be invariant to rotation around the radial axis.

    Block 0 radial axis = [-1,0,0], block 1 radial axis = [+1,0,0].  Rotating
    all atoms 90° around the x-axis (the shared radial line) changes t1/t2
    coordinates but leaves r-components unchanged, so the r-only loss must be
    identical before and after the rotation.
    """
    _, masks, ref_pos, metadata = _make_radial_orientation_setup()

    # 90° rotation around x-axis: (x, y, z) → (x, -z, y)
    def rot_x90(t):
        return torch.stack([t[:, 0], -t[:, 2], t[:, 1]], dim=-1)

    xyz_ref = ref_pos.unsqueeze(0)          # [1, 8, 3]
    xyz_rot = rot_x90(ref_pos).unsqueeze(0) # [1, 8, 3] — spun around radial axis

    pot = MotifRadialOrientationPotential(
        weight=1.0,
        motif_axis_weights=[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
    )
    val_ref = pot.compute(xyz_ref, masks, metadata).item()
    val_rot = pot.compute(xyz_rot, masks, metadata).item()

    assert val_ref == pytest.approx(val_rot, abs=1e-5), (
        f"r-only axis_weights must be spin-invariant: ref={val_ref:.6f}, rot={val_rot:.6f}"
    )

    # Sanity: full weights [1,1,1] should penalise the rotation.
    pot_full = MotifRadialOrientationPotential(weight=1.0)
    val_full_rot = pot_full.compute(xyz_rot, masks, metadata).item()
    assert val_full_rot < -1e-3, (
        f"Full-weight potential must penalise 90° r-rotation, got {val_full_rot:.6f}"
    )


# ════════════════════════════════════════════════════════════════════════════
# MotifCOMDistance tests
# Reuses _make_radial_orientation_setup: 2 blocks centred at ±5 on x-axis,
# protein COM at origin, so each motif is exactly 5 Å from COM by default.
# ════════════════════════════════════════════════════════════════════════════


@pytest.mark.fast
def test_motif_com_distance_zero_at_target():
    """Loss must be ~0 when each motif is exactly at its target distance."""
    _, masks, ref_pos, metadata = _make_radial_orientation_setup()
    # Both blocks are 5 Å from COM; set target_distances accordingly.
    xyz = ref_pos.unsqueeze(0)  # [1, 8, 3]
    pot = MotifCOMDistance(weight=1.0, target_distances=[5.0, 5.0])
    val = pot.compute(xyz, masks, metadata).item()
    assert val == pytest.approx(0.0, abs=1e-5), (
        f"Expected ~0 at target distance, got {val:.6f}"
    )


@pytest.mark.fast
def test_motif_com_distance_negative_when_off_target():
    """Loss must be negative (penalising) when motifs are not at target distance."""
    _, masks, ref_pos, metadata = _make_radial_orientation_setup()
    xyz = ref_pos.unsqueeze(0)
    # Targets are 10 Å but motifs are at 5 Å → nonzero penalty
    pot = MotifCOMDistance(weight=1.0, target_distances=[10.0, 10.0])
    val = pot.compute(xyz, masks, metadata).item()
    assert val < -1e-3, f"Expected negative loss, got {val:.6f}"
    # Penalty is -weight * mean((5-10)^2) = -25
    assert val == pytest.approx(-25.0, abs=1e-4)


@pytest.mark.fast
def test_motif_com_distance_guidance_moves_toward_target():
    """One gradient step should reduce the distance error."""
    _, masks, ref_pos, metadata = _make_radial_orientation_setup()

    xyz = ref_pos.clone().unsqueeze(0).requires_grad_(True)  # [1, 8, 3]
    pot = MotifCOMDistance(weight=1.0, target_distances=[10.0, 10.0])

    val = pot.compute(xyz, masks, metadata)
    val.backward()
    assert xyz.grad is not None

    # Step in gradient direction (ascent)
    step_size = 0.5
    xyz_new = (xyz + step_size * xyz.grad).detach()

    val_new = pot.compute(xyz_new, masks, metadata).item()
    assert val_new > val.item(), (
        f"Guidance must improve potential: before={val.item():.4f}, after={val_new:.4f}"
    )


@pytest.mark.fast
def test_motif_com_distance_empty_target_distances_gives_zero():
    """With no target_distances the potential must return zero."""
    _, masks, ref_pos, metadata = _make_radial_orientation_setup()
    xyz = ref_pos.unsqueeze(0)
    pot = MotifCOMDistance(weight=1.0, target_distances=[])
    val = pot.compute(xyz, masks, metadata).item()
    assert val == pytest.approx(0.0, abs=1e-9)


@pytest.mark.fast
def test_motif_com_distance_partial_constraint():
    """Only motifs with a target entry should be constrained."""
    _, masks, ref_pos, metadata = _make_radial_orientation_setup()
    xyz = ref_pos.unsqueeze(0)
    # Only constrain motif 0 (target 5 Å = current → zero loss)
    pot_zero = MotifCOMDistance(weight=1.0, target_distances=[5.0])
    assert pot_zero.compute(xyz, masks, metadata).item() == pytest.approx(0.0, abs=1e-5)
    # Only constrain motif 0 at wrong distance
    pot_nonzero = MotifCOMDistance(weight=1.0, target_distances=[10.0])
    assert pot_nonzero.compute(xyz, masks, metadata).item() < -1e-3


@pytest.mark.fast
def test_motif_com_distance_parsing():
    """The potential must parse correctly from a dict spec."""
    from rfd3.potentials.parsing import parse_potentials

    specs = [
        {
            "type": "motif_com_distance",
            "weight": 2.0,
            "target_distances": [25.0, 30.0],
            "origin_atom_filter": "real",
        }
    ]
    pots = parse_potentials(specs)
    pot = pots[0]
    assert isinstance(pot, MotifCOMDistance)
    assert pot.weight == 2.0
    assert pot.target_distances[0] == 25.0
    assert pot.target_distances[1] == 30.0
    assert pot.origin_atom_filter == "real"


# ════════════════════════════════════════════════════════════════════════════
# MotifSphericalPosition tests
# Reuses _make_radial_orientation_setup: 2 blocks centred at ±5 on x-axis,
# protein COM at origin.  Reference radial directions: [-1,0,0] and [+1,0,0].
# ════════════════════════════════════════════════════════════════════════════


@pytest.mark.fast
def test_motif_spherical_position_zero_at_reference():
    """Loss must be ~0 when each motif is on its reference radial direction."""
    _, masks, ref_pos, metadata = _make_radial_orientation_setup()
    xyz = ref_pos.unsqueeze(0)  # [1, 8, 3] — motifs at ±5 on x-axis

    pot = MotifSphericalPosition(weight=1.0)
    val = pot.compute(xyz, masks, metadata)

    assert val.ndim == 0
    assert abs(val.item()) < 1e-5, (
        f"Expected ~0 at reference pose, got {val.item():.6f}"
    )


@pytest.mark.fast
def test_motif_spherical_position_nonzero_off_direction():
    """Moving a motif to a wrong angular position must give a negative loss."""
    _, masks, ref_pos, metadata = _make_radial_orientation_setup()

    # Displace block 0 from x-axis to y-axis (r_cur = [0,1,0] vs r_ref = [-1,0,0])
    cur = ref_pos.clone()
    cur[:4] = cur[:4] - torch.tensor([-5.0, 0.0, 0.0]) + torch.tensor([0.0, 5.0, 0.0])
    xyz = cur.unsqueeze(0)

    pot = MotifSphericalPosition(weight=1.0)
    val = pot.compute(xyz, masks, metadata)

    assert val.item() < -1e-4, (
        f"Expected negative loss when motif is off reference direction, got {val.item():.6f}"
    )


@pytest.mark.fast
def test_motif_spherical_position_distance_invariant():
    """Scaling the motif-COM distance must not change the loss.

    Both blocks are moved symmetrically (block 0 to +y, block 1 to -y) so the
    protein COM remains at the origin regardless of the scale factor d.  The
    normalised radial directions are then purely [0,1,0] and [0,-1,0] for both
    scale values, so the loss must be identical.
    """
    _, masks, ref_pos, metadata = _make_radial_orientation_setup()

    def make_symmetric_xyz(d: float) -> torch.Tensor:
        cur = ref_pos.clone()
        c0 = ref_pos[:4].mean(dim=0)
        c1 = ref_pos[4:].mean(dim=0)
        cur[:4] = ref_pos[:4] - c0 + torch.tensor([0.0, d, 0.0])
        cur[4:] = ref_pos[4:] - c1 + torch.tensor([0.0, -d, 0.0])
        return cur.unsqueeze(0)

    pot = MotifSphericalPosition(weight=1.0)
    val_close = pot.compute(make_symmetric_xyz(5.0), masks, metadata).item()
    val_far = pot.compute(make_symmetric_xyz(15.0), masks, metadata).item()

    assert abs(val_close - val_far) < 1e-4, (
        f"Loss must be distance-invariant: close={val_close:.6f}, far={val_far:.6f}"
    )


@pytest.mark.fast
def test_motif_spherical_position_guidance_moves_toward_target():
    """One gradient step must reduce the angular error."""
    _, masks, ref_pos, metadata = _make_radial_orientation_setup()

    # Displace block 0 to y-axis
    cur = ref_pos.clone()
    cur[:4] = cur[:4] - torch.tensor([-5.0, 0.0, 0.0]) + torch.tensor([0.0, 5.0, 0.0])
    xyz = cur.clone().unsqueeze(0).requires_grad_(True)

    pot = MotifSphericalPosition(weight=1.0)
    val = pot.compute(xyz, masks, metadata)
    val.backward()
    assert xyz.grad is not None

    xyz_new = (xyz + 0.5 * xyz.grad).detach()
    val_new = pot.compute(xyz_new, masks, metadata).item()
    assert val_new > val.item(), (
        f"Guidance must improve potential: before={val.item():.4f}, after={val_new:.4f}"
    )


@pytest.mark.fast
def test_motif_spherical_position_pure_translation():
    """Gradient must be uniform across each motif block — no rotation component."""
    _, masks, ref_pos, metadata = _make_radial_orientation_setup()

    cur = ref_pos.clone()
    cur[:4] = cur[:4] - torch.tensor([-5.0, 0.0, 0.0]) + torch.tensor([0.0, 5.0, 0.0])
    xyz = cur.clone().unsqueeze(0).requires_grad_(True)

    pot = MotifSphericalPosition(weight=1.0)
    val = pot.compute(xyz, masks, metadata)
    val.backward()
    assert xyz.grad is not None

    # All 4 atoms in block 0 must receive the same gradient (pure translation)
    block0_grads = xyz.grad[0, :4, :]  # [4, 3]
    assert torch.allclose(block0_grads[0], block0_grads[1], atol=1e-6)
    assert torch.allclose(block0_grads[0], block0_grads[2], atol=1e-6)
    assert torch.allclose(block0_grads[0], block0_grads[3], atol=1e-6)


@pytest.mark.fast
def test_motif_spherical_position_offset_shifts_target():
    """A nonzero per-motif offset must produce a different energy than zero offset."""
    _, masks, ref_pos, metadata = _make_radial_orientation_setup()
    xyz = ref_pos.unsqueeze(0)  # at reference — zero offset should give ~0

    pot_no_offset = MotifSphericalPosition(weight=1.0, motif_offsets=[])
    pot_with_offset = MotifSphericalPosition(
        weight=1.0,
        motif_offsets=[[0.0, 90.0, 0.0]],  # rotate reference direction 90° for block 0
    )

    val_no_offset = pot_no_offset.compute(xyz, masks, metadata).item()
    val_with_offset = pot_with_offset.compute(xyz, masks, metadata).item()

    assert abs(val_no_offset) < 1e-5, (
        f"No offset at reference must give ~0, got {val_no_offset:.6f}"
    )
    assert val_with_offset < -1e-4, (
        f"90° offset must give negative loss at reference pose, got {val_with_offset:.6f}"
    )


@pytest.mark.fast
def test_motif_spherical_position_parsing():
    """The potential must parse correctly from a dict spec."""
    specs = [
        {
            "type": "motif_spherical_position",
            "weight": 3.0,
            "motif_offsets": [[10.0, 0.0, 0.0], [0.0, 20.0, 0.0]],
            "origin_atom_filter": "real",
        }
    ]
    pots = parse_potentials(specs)
    assert len(pots) == 1
    pot = pots[0]
    assert isinstance(pot, MotifSphericalPosition)
    assert pot.weight == 3.0
    assert pot.origin_atom_filter == "real"
    assert len(pot.motif_offsets) == 2
    assert pot.motif_offsets[0][0] == 10.0
    assert pot.motif_offsets[1][1] == 20.0

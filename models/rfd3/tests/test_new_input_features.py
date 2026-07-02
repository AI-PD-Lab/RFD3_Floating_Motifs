"""
Tests for non_fixed_contig and motifs input fields added to DesignInputSpecification.

Tests verify:
  1. non_fixed_contig atoms are floating (is_motif_atom_with_fixed_coord=0)
  2. regular contig atoms remain fixed when non_fixed_contig is also present
  3. select_fixed_atoms overrides non_fixed_contig (highest priority)
  4. motifs are appended as separate chains, floating by default
  5. Kabsch eligibility: NFC and motif atoms have alphabetic src_component and
     is_motif_atom_unindexed=0, so _get_contig_motif_atom_mask includes them
  6. contig / non_fixed_contig / motifs overlap raises an error
  7. SymmetryConfig accepts mode and instances fields
"""

import numpy as np
import pytest
from atomworks.io.tools.inference import components_to_atom_array
from biotite.structure import get_residue_starts
from rfd3.inference.input_parsing import DesignInputSpecification, resolve_auto_length
from rfd3.inference.symmetry.symmetry_utils import SymmetryConfig
from rfd3.model.floating_motif_projection import (
    _get_contig_motif_atom_mask,
    build_floating_motif_references_from_contigs,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_two_chain_input():
    """Return a simple 2-chain protein (A: 20 residues, B: 15 residues)."""
    components = [
        {"seq": "ACDEFGHIKLMNPQRSTVWY", "chain_type": "polypeptide(l)",
         "is_polymer": True, "chain_id": "A"},
        {"seq": "ACDEFGHIKLMNPQR", "chain_type": "polypeptide(l)",
         "is_polymer": True, "chain_id": "B"},
    ]
    aa = components_to_atom_array(components)
    rng = np.random.default_rng(0)
    aa.coord = rng.standard_normal((len(aa), 3)).astype(np.float32) * 10
    return aa


def _build(aa, **kwargs):
    """Build a DesignInputSpecification with atom_array_input and return the atom array."""
    spec = DesignInputSpecification(atom_array_input=aa, **kwargs)
    return spec.build()


# ---------------------------------------------------------------------------
# non_fixed_contig tests
# ---------------------------------------------------------------------------

@pytest.mark.fast
def test_nfc_atoms_are_unfixed():
    """Atoms selected by non_fixed_contig must have is_motif_atom_with_fixed_coord=0."""
    aa_in = _make_two_chain_input()
    result = _build(aa_in, non_fixed_contig="A1-10,50")

    src = np.asarray(result.src_component).astype(str)
    nfc_atoms = np.array([
        s.startswith("A") and s[1:].isdigit() and 1 <= int(s[1:]) <= 10
        for s in src
    ])

    assert nfc_atoms.sum() > 0, "No NFC atoms found in result"

    fixed_coord = result.is_motif_atom_with_fixed_coord.astype(bool)
    assert not np.any(fixed_coord[nfc_atoms]), \
        "NFC atoms should have is_motif_atom_with_fixed_coord=0"


@pytest.mark.fast
def test_nfc_atoms_have_zeroed_coords():
    """NFC atoms must be zeroed out in the built array (they float from origin)."""
    aa_in = _make_two_chain_input()
    result = _build(aa_in, non_fixed_contig="A1-10,50")

    src = np.asarray(result.src_component).astype(str)
    nfc_atoms = np.array([
        s.startswith("A") and s[1:].isdigit() and 1 <= int(s[1:]) <= 10
        for s in src
    ])
    assert nfc_atoms.sum() > 0
    nfc_coords = result.coord[nfc_atoms]
    assert np.allclose(nfc_coords, 0.0), \
        "NFC atom coordinates should be zeroed (floating from origin)"


@pytest.mark.fast
def test_nfc_kabsch_eligible():
    """NFC atoms must pass the _get_contig_motif_atom_mask check (Kabsch eligible)."""
    aa_in = _make_two_chain_input()
    result = _build(aa_in, non_fixed_contig="A1-10,50")

    src = np.asarray(result.src_component).astype(str)
    kabsch_mask = _get_contig_motif_atom_mask(result, src)

    nfc_atoms = np.array([
        s.startswith("A") and s[1:].isdigit() and 1 <= int(s[1:]) <= 10
        for s in src
    ])
    assert nfc_atoms.sum() > 0
    assert np.all(kabsch_mask[nfc_atoms]), \
        "NFC atoms must be Kabsch-eligible (alphabetic src_component, not unindexed)"


@pytest.mark.fast
def test_nfc_reference_coords_stored():
    """Floating motif reference annotations must be present and non-zero for NFC atoms."""
    from rfd3.model.floating_motif_projection import FLOATING_MOTIF_REFERENCE_ANNOTATIONS

    aa_in = _make_two_chain_input()
    result = _build(aa_in, non_fixed_contig="A1-10,50")

    src = np.asarray(result.src_component).astype(str)
    nfc_atoms = np.array([
        s.startswith("A") and s[1:].isdigit() and 1 <= int(s[1:]) <= 10
        for s in src
    ])
    assert nfc_atoms.sum() > 0
    for annot in FLOATING_MOTIF_REFERENCE_ANNOTATIONS:
        assert annot in result.get_annotation_categories(), \
            f"Floating motif annotation '{annot}' missing"
        ref_vals = result.get_annotation(annot)[nfc_atoms]
        assert not np.allclose(ref_vals, 0.0), \
            f"Reference coord annotation '{annot}' should store PDB coords, not zeros"


@pytest.mark.fast
def test_sfa_overrides_nfc():
    """Explicit select_fixed_atoms=True must fix all atoms including NFC residues."""
    aa_in = _make_two_chain_input()
    result = _build(
        aa_in,
        non_fixed_contig="A1-10,50",
        select_fixed_atoms="A1-10",
    )
    src = np.asarray(result.src_component).astype(str)
    input_atoms = np.array([
        s.startswith("A") and s[1:].isdigit() and 1 <= int(s[1:]) <= 10
        for s in src
    ])
    fixed_coord = result.is_motif_atom_with_fixed_coord.astype(bool)
    assert np.all(fixed_coord[input_atoms]), \
        "Explicit select_fixed_atoms must override NFC and fix those atoms"


@pytest.mark.fast
def test_nfc_only_no_contig():
    """non_fixed_contig without any regular contig should build successfully."""
    aa_in = _make_two_chain_input()
    result = _build(aa_in, non_fixed_contig="A1-10,50")

    src = np.asarray(result.src_component).astype(str)
    nfc_atoms = np.array([
        s.startswith("A") and s[1:].isdigit() and 1 <= int(s[1:]) <= 10
        for s in src
    ])
    assert nfc_atoms.sum() > 0
    fixed_coord = result.is_motif_atom_with_fixed_coord.astype(bool)
    assert not np.any(fixed_coord[nfc_atoms]), \
        "NFC-only atoms should be floating"


# ---------------------------------------------------------------------------
# motifs tests
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_auto_length_default_ellipsoid_example():
    """length='auto' default should match the 50 A by 20 A ellipsoid estimate."""
    info = resolve_auto_length(potentials=None)
    assert info["median"] == 322
    assert info["length"] == "258-386"
    assert info["distance"] == 50.0
    assert info["radius"] == 20.0


@pytest.mark.fast
def test_auto_length_uses_potential_distance_and_bridge_radius():
    """Auto length should read motif-distance and bridge-radius potential fields."""
    info = resolve_auto_length(
        potentials={
            "guiding_potentials": [
                {"type": "motif_distance", "target_distance": 60.0},
                {"type": "motif_bridge", "max_radius": 15.0},
            ]
        }
    )
    # V = 4/3*pi*30*15*15 = 28274.3 A^3; /130 = 217.5 residues.
    assert info["median"] == 217
    assert info["length"] == "174-260"
    assert info["distance"] == 60.0
    assert info["radius"] == 15.0


@pytest.mark.fast
def test_design_spec_length_auto_becomes_regular_range():
    """The literal length token 'auto' should canonicalize before normal expansion."""
    aa_in = _make_two_chain_input()
    spec = DesignInputSpecification(
        atom_array_input=aa_in,
        length="auto",
        auto_length_potentials={
            "guiding_potentials": [
                {"type": "motif_distance", "target_distance": 60.0},
                {"type": "motif_bridge", "max_radius": 15.0},
            ]
        },
    )
    assert spec.length == "174-260"
    assert spec.extra["auto_length"]["median"] == 217


@pytest.mark.fast
def test_contig_auto_token_fills_remaining_length():
    """A contig-level auto token should fill total length minus motif lengths."""
    aa_in = _make_two_chain_input()
    spec = DesignInputSpecification(
        atom_array_input=aa_in,
        contig="A1-5,auto,B1-5",
        length="30",
        select_fixed_atoms=False,
    )
    result, metadata = spec.build(return_metadata=True)
    assert len(get_residue_starts(result)) == 30
    assert metadata["extra"]["contig_auto"]["length"] == "30"
    assert metadata["extra"]["contig_auto"]["length_min"] == 30
    assert metadata["extra"]["contig_auto"]["length_max"] == 30
    assert metadata["extra"]["contig_auto"]["fixed_budget"] == 10
    assert metadata["extra"]["contig_auto"]["auto_lengths"] == ["20"]


@pytest.mark.fast
def test_contig_auto_token_subtracts_max_of_other_ranges():
    """Other scaffold ranges should count by their maximum when resolving contig auto."""
    aa_in = _make_two_chain_input()
    spec = DesignInputSpecification(
        atom_array_input=aa_in,
        contig="A1-5,10-15,auto,B1-5",
        length="40",
        select_fixed_atoms=False,
    )
    result, metadata = spec.build(return_metadata=True)
    assert len(get_residue_starts(result)) == 40
    assert metadata["extra"]["contig_auto"]["fixed_budget"] == 25
    assert metadata["extra"]["contig_auto"]["auto_lengths"] == ["15"]


@pytest.mark.fast
def test_contig_auto_token_becomes_range_with_length_range():
    """With ranged top-level length, contig auto should resolve to a range."""
    aa_in = _make_two_chain_input()
    spec = DesignInputSpecification(
        atom_array_input=aa_in,
        contig="A1-5,auto,B1-5",
        length="30-40",
        select_fixed_atoms=False,
    )
    result, metadata = spec.build(return_metadata=True)
    n_res = len(get_residue_starts(result))
    assert 30 <= n_res <= 40
    assert metadata["extra"]["contig_auto"]["length"] == "30-40"
    assert metadata["extra"]["contig_auto"]["length_min"] == 30
    assert metadata["extra"]["contig_auto"]["length_max"] == 40
    assert metadata["extra"]["contig_auto"]["fixed_budget"] == 10
    assert metadata["extra"]["contig_auto"]["auto_lengths"] == ["20-30"]
    assert metadata["extra"]["contig_auto"]["resolved_contig"] == "A1-5,20-30,B1-5"


@pytest.mark.fast
def test_contig_auto_range_subtracts_max_of_other_ranges_from_both_bounds():
    """Ranged contig auto should subtract other range maxima from both bounds."""
    aa_in = _make_two_chain_input()
    spec = DesignInputSpecification(
        atom_array_input=aa_in,
        contig="A1-5,10-15,auto,B1-5",
        length="40-50",
        select_fixed_atoms=False,
    )
    result, metadata = spec.build(return_metadata=True)
    n_res = len(get_residue_starts(result))
    assert 40 <= n_res <= 50
    assert metadata["extra"]["contig_auto"]["fixed_budget"] == 25
    assert metadata["extra"]["contig_auto"]["auto_lengths"] == ["15-25"]
    assert (
        metadata["extra"]["contig_auto"]["resolved_contig"]
        == "A1-5,10-15,15-25,B1-5"
    )


@pytest.mark.fast
def test_motif_appended_as_separate_chain():
    """motifs must be appended as additional chains separate from the scaffold chain."""
    aa_in = _make_two_chain_input()
    result = _build(
        aa_in,
        contig="50",
        motifs={"b_motif": "B1-10"},
        sequence_unrestrained_motifs=["b_motif"],
    )

    chains = np.unique(result.chain_id)
    assert len(chains) >= 2, "Motif should be on a separate chain"

    # The motif chain should have src_component starting with B
    src = np.asarray(result.src_component).astype(str)
    motif_atoms = np.array([
        s.startswith("B") and s[1:].isdigit() and 1 <= int(s[1:]) <= 10
        for s in src
    ])
    assert motif_atoms.sum() > 0, "Motif atoms (B1-10) must be present in result"


@pytest.mark.fast
def test_motif_atoms_are_unfixed():
    """Motif atoms must default to is_motif_atom_with_fixed_coord=0."""
    aa_in = _make_two_chain_input()
    result = _build(aa_in, contig="50", motifs={"b_motif": "B1-10"})

    src = np.asarray(result.src_component).astype(str)
    motif_atoms = np.array([
        s.startswith("B") and s[1:].isdigit() and 1 <= int(s[1:]) <= 10
        for s in src
    ])
    fixed_coord = result.is_motif_atom_with_fixed_coord.astype(bool)
    assert not np.any(fixed_coord[motif_atoms]), \
        "Motif atoms must be floating by default"


@pytest.mark.fast
def test_motif_kabsch_eligible():
    """Motif atoms must be Kabsch-eligible (alphabetic src_component, not unindexed)."""
    aa_in = _make_two_chain_input()
    result = _build(aa_in, contig="50", motifs={"b_motif": "B1-10"})

    src = np.asarray(result.src_component).astype(str)
    motif_atoms = np.array([
        s.startswith("B") and s[1:].isdigit() and 1 <= int(s[1:]) <= 10
        for s in src
    ])
    kabsch_mask = _get_contig_motif_atom_mask(result, src)
    assert np.all(kabsch_mask[motif_atoms]), \
        "Motif atoms must pass Kabsch eligibility check"


@pytest.mark.fast
def test_motif_sfa_override():
    """Explicit select_fixed_atoms can fix motif atoms (highest priority)."""
    aa_in = _make_two_chain_input()
    result = _build(
        aa_in,
        contig="50",
        motifs={"b_motif": "B1-10"},
        select_fixed_atoms="B1-10",
    )
    src = np.asarray(result.src_component).astype(str)
    motif_atoms = np.array([
        s.startswith("B") and s[1:].isdigit() and 1 <= int(s[1:]) <= 10
        for s in src
    ])
    fixed_coord = result.is_motif_atom_with_fixed_coord.astype(bool)
    assert np.all(fixed_coord[motif_atoms]), \
        "Explicit select_fixed_atoms must override default motif unfixing"


@pytest.mark.fast
def test_multiple_motifs_separate_chains():
    """Multiple motifs must each appear on their own separate chain."""
    aa_in = _make_two_chain_input()
    result = _build(
        aa_in,
        contig="30",
        motifs={"a_motif": "A1-5", "b_motif": "B1-5"},
        sequence_unrestrained_motifs=["a_motif", "b_motif"],
    )
    chains = np.unique(result.chain_id)
    # scaffold (A) + motif1 chain + motif2 chain = at least 3 chains
    assert len(chains) >= 3, \
        f"Two motifs should produce at least 3 chains, got {chains}"


@pytest.mark.fast
def test_unindexed_motif_inlines_on_main_chain():
    """unindexed_motifs should inline on the main chain without trainer cleanup."""
    aa_in = _make_two_chain_input()
    result = _build(
        aa_in,
        length="30",
        motifs={"b_motif": "B1-5"},
        unindexed_motifs=["b_motif"],
    )

    chains = np.unique(result.chain_id)
    assert len(chains) == 1, f"unindexed_motifs should stay on the main chain, got {chains}"

    src = np.asarray(result.src_component).astype(str)
    motif_atoms = np.array([
        s.startswith("B") and s[1:].isdigit() and 1 <= int(s[1:]) <= 5
        for s in src
    ])
    assert motif_atoms.sum() > 0, "Unindexed motif atoms must be present in result"
    assert not np.any(result.is_motif_atom_with_fixed_coord.astype(bool)[motif_atoms]), \
        "Unindexed motif atoms must remain floating by default"
    assert not np.any(result.is_motif_atom_unindexed.astype(bool)[motif_atoms]), \
        "Active unindexed_motifs should no longer use the true unindex pathway"
    assert np.all(_get_contig_motif_atom_mask(result, src)[motif_atoms]), \
        "Unindexed motif atoms must remain Kabsch-eligible"
    assert np.all(result.is_motif_atom_with_fixed_seq.astype(bool)[motif_atoms]), \
        "Unindexed motif sequence should remain fixed by default"
    assert len(get_residue_starts(result)) == 30, \
        "Length-only unindexed_motifs should resolve into a single chain of the requested total length"


@pytest.mark.fast
def test_multiple_unindexed_motifs_share_main_chain():
    """Multiple unindexed_motifs should all inline on the same main chain."""
    aa_in = _make_two_chain_input()
    result = _build(
        aa_in,
        length="20",
        motifs={"a_motif": "A1-5", "b_motif": "B1-5"},
        unindexed_motifs=["a_motif", "b_motif"],
    )

    chains = np.unique(result.chain_id)
    assert len(chains) == 1, f"All unindexed motifs should stay on one main chain, got {chains}"


@pytest.mark.fast
def test_unindexed_motif_conflicts_with_sequence_unrestrained():
    """A motif name must not be used in both unindexed_motifs and sequence_unrestrained_motifs."""
    aa_in = _make_two_chain_input()
    with pytest.raises(ValueError, match="unindexed_motifs"):
        _build(
            aa_in,
            length="20",
            motifs={"b_motif": "B1-5"},
            unindexed_motifs=["b_motif"],
            sequence_unrestrained_motifs=["b_motif"],
        )


@pytest.mark.fast
def test_unindexed_motifs_work_with_regular_unindex():
    """unindexed_motifs should coexist with the existing unindex path."""
    aa_in = _make_two_chain_input()
    result = _build(
        aa_in,
        length="20",
        motifs={"b_motif": "B1-5"},
        unindexed_motifs=["b_motif"],
        unindex="A10-12",
        select_fixed_atoms="A10-12",
    )

    src = np.asarray(result.src_component).astype(str)
    motif_atoms = np.array([
        s.startswith("B") and s[1:].isdigit() and 1 <= int(s[1:]) <= 5
        for s in src
    ])
    assert motif_atoms.sum() > 0, "Unindexed motif atoms must still be present when unindex is used"
    assert np.all(_get_contig_motif_atom_mask(result, src)[motif_atoms]), \
        "Unindexed motif atoms must remain Kabsch-eligible when unindex is also present"
    assert not np.any(result.is_motif_atom_unindexed.astype(bool)[motif_atoms]), \
        "unindexed_motifs should stay off the true unindex pathway even when regular unindex is used"
    assert result.is_motif_atom_unindexed.astype(bool).sum() > 0, \
        "Regular unindex path should still contribute unindexed atoms"


@pytest.mark.fast
def test_unindexed_motif_select_unfixed_sequence_override():
    """select_unfixed_sequence should still be able to unfix unindexed motif sequence."""
    aa_in = _make_two_chain_input()
    result = _build(
        aa_in,
        length="20",
        motifs={"b_motif": "B1-5"},
        unindexed_motifs=["b_motif"],
        select_unfixed_sequence="B1-5",
    )

    src = np.asarray(result.src_component).astype(str)
    motif_atoms = np.array([
        s.startswith("B") and s[1:].isdigit() and 1 <= int(s[1:]) <= 5
        for s in src
    ])
    assert motif_atoms.sum() > 0
    assert not np.any(result.is_motif_atom_with_fixed_seq.astype(bool)[motif_atoms]), \
        "select_unfixed_sequence must override the default fixed sequence for unindexed motifs"


@pytest.mark.fast
def test_unindexed_motif_total_length_is_preserved():
    """unindexed_motifs should consume length budget rather than append extra residues."""
    aa_in = _make_two_chain_input()
    spec = DesignInputSpecification(
        atom_array_input=aa_in,
        length="30",
        motifs={"b_motif": "B1-5"},
        unindexed_motifs=["b_motif"],
    )
    result, metadata = spec.build(return_metadata=True)
    assert len(get_residue_starts(result)) == 30
    assert "B1" in metadata["extra"]["sampled_contig"]


@pytest.mark.fast
def test_sym_motif_instance_includes_unindexed_motifs():
    """SymMotif instance builds should also inline unindexed_motifs on the main chain."""
    aa_in = _make_two_chain_input()
    spec = DesignInputSpecification(
        atom_array_input=aa_in,
        contig="SymMotif,10",
        motifs={"a_motif": "A1-5", "tail": "B1-3"},
        unindexed_motifs=["tail"],
        symmetry={
            "id": "C2",
            "mode": "heterotypic",
            "instances": {"0": ["a_motif"], "1": ["a_motif"]},
            "is_symmetric_motif": False,
        },
    )
    result = spec._build_sym_motif_instance(0, aa_in)

    chains = np.unique(result.chain_id)
    assert len(chains) == 1, f"SymMotif instance should remain one chain before symmetry, got {chains}"

    src = np.asarray(result.src_component).astype(str)
    tail_atoms = np.array([
        s.startswith("B") and s[1:].isdigit() and 1 <= int(s[1:]) <= 3
        for s in src
    ])
    assert tail_atoms.sum() > 0, "Unindexed motif atoms should be present in SymMotif instance build"
    assert not np.any(result.is_motif_atom_unindexed.astype(bool)[tail_atoms]), \
        "Active unindexed_motifs should inline in SymMotif builds without using the true unindex pathway"


# ---------------------------------------------------------------------------
# Overlap / exclusivity tests
# ---------------------------------------------------------------------------

@pytest.mark.fast
def test_contig_nfc_mutually_exclusive():
    """contig and non_fixed_contig must not both be provided (mutually exclusive)."""
    aa_in = _make_two_chain_input()
    with pytest.raises(ValueError, match="mutually exclusive"):
        _build(aa_in, contig="A1-10,50", non_fixed_contig="A11-20")


@pytest.mark.fast
def test_motif_name_in_contig_inlines_atoms():
    """A motif name token in contig must be resolved inline on the main chain (not a separate chain)."""
    aa_in = _make_two_chain_input()
    result = _build(aa_in, contig="b_motif,50", motifs={"b_motif": "B1-5"})

    # Only one chain — the motif is inlined, not on a separate chain
    chains = np.unique(result.chain_id)
    assert len(chains) == 1, f"Motif-name-in-contig should inline atoms; got chains: {chains}"

    # B1-5 atoms must be floating
    src = np.asarray(result.src_component).astype(str)
    motif_atoms = np.array([
        s.startswith("B") and s[1:].isdigit() and 1 <= int(s[1:]) <= 5
        for s in src
    ])
    assert motif_atoms.sum() > 0, "Motif atoms (B1-5) must be present in result"
    fixed_coord = result.is_motif_atom_with_fixed_coord.astype(bool)
    assert not np.any(fixed_coord[motif_atoms]), \
        "Motif atoms inlined via contig must be floating (is_motif_atom_with_fixed_coord=0)"


@pytest.mark.fast
def test_contig_motif_overlap_raises():
    """contig direct PDB residues and motifs selecting the same residues must raise ValueError."""
    aa_in = _make_two_chain_input()
    with pytest.raises(ValueError, match="overlap"):
        _build(aa_in, contig="A1-10,50", motifs={"dup": "A1-5"})


@pytest.mark.fast
def test_missing_input_with_nfc_raises():
    """non_fixed_contig without input must raise ValueError."""
    with pytest.raises(ValueError):
        DesignInputSpecification(non_fixed_contig="A1-10", length="50")


# ---------------------------------------------------------------------------
# SymmetryConfig hetero mode tests
# ---------------------------------------------------------------------------

@pytest.mark.fast
def test_symmetry_config_accepts_heterotypic_mode():
    """SymmetryConfig must accept heterotypic mode and instances without error."""
    cfg = SymmetryConfig(
        id="C2",
        mode="heterotypic",
        instances={"0": ["motif_fgfr"], "1": ["motif_her2"]},
    )
    assert cfg.mode == "heterotypic"
    assert cfg.instances == {"0": ["motif_fgfr"], "1": ["motif_her2"]}


@pytest.mark.fast
def test_symmetry_config_accepts_independent_mode():
    """SymmetryConfig independent mode instance format (reserved) is accepted."""
    cfg = SymmetryConfig(
        mode="independent",
        instances={
            "0": {"contig": "A1-59,200"},
            "1": {"contig": "B1-59,200"},
        },
    )
    assert cfg.mode == "independent"
    assert "contig" in cfg.instances["0"]


@pytest.mark.fast
def test_symmetry_config_extra_fields_allowed():
    """SymmetryConfig must tolerate extra fields (extra='allow') for forward compat."""
    cfg = SymmetryConfig(id="C3", future_param="future_value")
    assert cfg.id == "C3"

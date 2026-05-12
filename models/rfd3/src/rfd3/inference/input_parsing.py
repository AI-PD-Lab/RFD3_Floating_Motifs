import copy
import json
import logging
import os
import random
import re
import time
import warnings
from contextlib import contextmanager
from os import PathLike
from typing import Any, Dict, List, Optional, Union

import numpy as np
from atomworks.constants import STANDARD_AA, STANDARD_DNA, STANDARD_RNA
from atomworks.io.parser import parse_atom_array
from atomworks.io.utils.bonds import get_inferred_polymer_bonds

# from atomworks.ml.datasets.datasets import BaseDataset
from atomworks.ml.transforms.base import TransformedDict
from atomworks.ml.utils.token import (
    get_token_starts,
)
from biotite import structure as struc
from biotite.structure import AtomArray, BondList, get_residue_starts
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)
from rfd3.constants import (
    INFERENCE_ANNOTATIONS,
    OPTIONAL_CONDITIONING_VALUES,
    REQUIRED_CONDITIONING_ANNOTATION_VALUES,
    REQUIRED_INFERENCE_ANNOTATIONS,
)
from rfd3.inference.legacy_input_parsing import (
    _check_has_backbone_connections_to_nonstandard_residues,
    create_atom_array_from_design_specification_legacy,
)
from rfd3.inference.parsing import InputSelection
from rfd3.inference.symmetry.symmetry_utils import (
    SymmetryConfig,
    center_symmetric_src_atom_array,
    make_symmetric_atom_array,
)
from rfd3.inference.symmetry.atom_array import (
    add_src_sym_component_annotations,
    add_sym_annotations,
    fix_3D_sym_motif_annotations,
    get_symmetry_unit,
)
from rfd3.inference.symmetry.checks import check_symmetry_config
from rfd3.inference.symmetry.frames import get_symmetry_frames_from_symmetry_id
from rfd3.model.floating_motif_projection import (
    FLOATING_MOTIF_REFERENCE_ANNOTATIONS,
    annotate_floating_motif_reference_coords,
)
from rfd3.transforms.conditioning_base import (
    check_has_required_conditioning_annotations,
    convert_existing_annotations_to_bool,
    get_motif_features,
    set_default_conditioning_annotations,
)
from rfd3.transforms.util_transforms import assign_types_
from rfd3.utils.inference import (
    extract_ligand_array,
    inference_load_,
    set_com,
    set_common_annotations,
    set_indices,
)

from foundry.common import exists
from foundry.utils.components import (
    fetch_mask_from_idx,
    get_design_pattern_with_constraints,
    get_motif_components_and_breaks,
)
from foundry.utils.ddp import RankedLogger

logging.basicConfig(level=logging.DEBUG)

logger = RankedLogger(__name__, rank_zero_only=True)


_SCAFFOLD_CONTIG_TOKEN_RE = re.compile(r"^\d+(?:-\d+)?$")
UNINDEXED_FLOATING_MOTIF_ANNOTATION = "is_motif_atom_unindexed_floating_motif"
# Keep the original true-unindex implementation on disk for later experiments, but
# route active unindexed_motifs through inline sampled placement for now.
UNINDEXED_MOTIFS_USE_LEGACY_TRUE_UNINDEX = False


def _input_selection_from_contig_with_placeholders(
    contig: str,
    atom_array: AtomArray,
    motif_names: set[str] | None = None,
) -> InputSelection:
    """Parse only concrete PDB tokens from a contig with placeholders."""
    motif_names = motif_names or set()
    direct_parts = []
    for token in contig.split(","):
        token = token.strip()
        if not token:
            continue
        if (
            token == "SymMotif"
            or token in motif_names
            or _SCAFFOLD_CONTIG_TOKEN_RE.match(token)
        ):
            continue
        direct_parts.append(token)

    if not direct_parts:
        return InputSelection(
            raw=contig,
            data={},
            mask=np.zeros(len(atom_array), dtype=bool),
            tokens=None,
        )

    selection = InputSelection.from_any(
        ",".join(direct_parts),
        atom_array=atom_array,
    )
    return selection.model_copy(update={"raw": contig})


def _get_tokens_from_selection_data(selection: InputSelection, atom_array: AtomArray):
    data = {k: v for k, v in selection.data.items() if v}
    if not data:
        return {}
    return InputSelection.from_any(dict(data), atom_array=atom_array).get_tokens(atom_array)


def _shift_floating_motif_reference_coords(atom_array, shift):
    if shift is None or not all(
        annotation in atom_array.get_annotation_categories()
        for annotation in FLOATING_MOTIF_REFERENCE_ANNOTATIONS
    ):
        return atom_array

    shift = np.asarray(shift, dtype=np.float32)
    for axis, annotation in enumerate(FLOATING_MOTIF_REFERENCE_ANNOTATIONS):
        values = atom_array.get_annotation(annotation).astype(np.float32, copy=True)
        values -= shift[axis]
        atom_array.set_annotation(annotation, values)
    return atom_array


def _infer_uniform_coordinate_shift(before, after):
    finite = np.isfinite(before).all(axis=-1) & np.isfinite(after).all(axis=-1)
    if not np.any(finite):
        return None
    return np.median(before[finite] - after[finite], axis=0)


#################################################################################
# Custom infer_ori functions
#################################################################################


class LegacySpecification(BaseModel):
    """Legacy specification for compatibility with legacy input parsing."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="allow",
    )

    def build(self, *args, **kwargs):
        """Build atom array using legacy input parsing."""
        atom_array = create_atom_array_from_design_specification_legacy(
            **self.model_dump(),
        )
        return atom_array, self.model_dump()

    def to_pipeline_input(self, example_id):
        atom_array, spec_dict = self.build(return_metadata=True)

        # ... Forward into
        data = prepare_pipeline_input_from_atom_array(atom_array)
        data["example_id"] = example_id

        # ... Wrap up with additional features
        if "extra" not in spec_dict:
            spec_dict["extra"] = {}
        spec_dict["extra"]["example_id"] = example_id
        data["specification"] = spec_dict
        return data


# ========================================================================
# Input specification
# ========================================================================


class DesignInputSpecification(BaseModel):
    """Validated and parsed input specification before resolution."""

    model_config = ConfigDict(
        hide_input_in_errors=False,
        arbitrary_types_allowed=True,
        validate_assignment=False,
        str_strip_whitespace=True,
        str_min_length=1,
        extra="forbid",
    )
    # fmt: off
    # ========================================================================
    # Data inputs, motif generation & selection
    # ========================================================================
    # Data inputs
    atom_array_input: Optional[AtomArray] = Field(None, description="Loaded atom array", exclude=True)
    input: Optional[str] =  Field(None, description="Path to input PDB/CIF file")
    # Motif selection from input file
    contig:  Optional[InputSelection] = Field(None, description="Contig specification string (e.g. 'A1-10,B1-5')")
    unindex: Optional[InputSelection] = Field(None,
        description="Unindexed components selection. Components to fix in the generated structure without specifying sequence index. "\
        "Components must not overlap with `contig` argument. "\
        "E.g. 'A15-20,B6-10' or dict. We recommend specifying unindexed residues as a contig string, "\
        "then using select_fixed_atoms will subset the atoms to the specified atoms")
    non_fixed_contig: Optional[InputSelection] = Field(None,
        description="Contig of atoms from input that are included in the design but NOT fixed in 3D space. "
        "Atoms are still Kabsch-aligned during floating-motif projection (alphabetic src_component, not unindexed). "
        "Default is is_motif_atom_with_fixed_coord=False for selected atoms; select_fixed_atoms overrides. "
        "Format identical to 'contig' (e.g. 'A11-20,B1-5'). Must not overlap with 'contig', 'unindex', or 'motifs'.")
    motifs: Optional[Dict[str, str]] = Field(None,
        description="Named floating motif definitions. "
        "Format: {'motif_name': 'contig_str', ...}. A motif is only included in the design if: "
        "(a) its name appears as a token in 'contig', "
        "(b) its name appears in 'sequence_unrestrained_motifs', or "
        "(c) it is assigned via a 'SymMotif' placeholder in a symmetry instances dict. "
        "Atoms are Kabsch-aligned but coordinate-unfixed by default; select_fixed_atoms overrides.")
    sequence_unrestrained_motifs: Optional[List[str]] = Field(None,
        description="Motif names (from the 'motifs' dict) to append as separate floating chains "
        "when their position in the scaffold chain is not constrained. "
        "Only motifs listed here are appended; motifs referenced by name in 'contig' or via "
        "'SymMotif' are placed inline and must NOT also appear here.")
    unindexed_motifs: Optional[List[str]] = Field(None,
        description="Motif names (from the 'motifs' dict) to include on the main chain without "
        "explicitly placing them in 'contig'. These motifs are inserted into a hidden sampled "
        "inline layout before diffusion, so the model sees a single contiguous chain while the "
        "motifs remain floating/Kabsch-aligned and internally conserved. Names listed here must "
        "NOT also appear in 'contig', 'sequence_unrestrained_motifs', or be assigned via "
        "'SymMotif'.")
    # Extra args:
    length:  Optional[str] = Field(None, description="Length range as 'min-max' or int. Constrains length of contig if provided")
    ligand:  Optional[str] = Field(None, description="Ligand name or index to include in design.")
    allow_ligand_on_existing_chain: bool = Field(False, description="If True, suppress the error when a ligand shares a chain ID with the built atom array. Use with caution — chain ID is leaked to the model.")
    cif_parser_args: Optional[Dict[str, Any]] = Field(None, description="CIF parser arguments")
    extra: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Extra metadata to include in output (useful for logging additional info in metadata)")
    dialect: int = Field(2, description="RFdiffusion3 input dialect. 1: legacy, 2: release.")

    # ========================================================================
    # Conditioning
    # ========================================================================
    # Sequence and coordinate conditioning
    select_fixed_atoms: Optional[InputSelection] = Field(None,
        description='''Atoms to fix coordinates for. Examples:
        - True (default when inputs provided): All atoms pulled from the input are fixed in 3d space
        - False: All atoms pulled from the input are unfixed in 3d space
        - ContigStr: Components to fix in 3d space, e.g. "A1-10,B1-3" fixes residues 1-10 in chain A and residues 1-3 in chain B.
        - {"A1": "N,CA,C,O,CB,CG", "A2-10": "BKBN"} fixes backbone and CB for residues 1 and 2, and all atoms for residues 3-10 in chain A.
    '''.replace('\t\t', '\t')
    )
    select_unfixed_sequence: Optional[InputSelection] = Field(None, description='''Components to unfix sequence for. 
        - True (default when inputs provided): All atoms from the input have fixed sequences by default.
        - False: All atoms pulled from the input have diffused sequences by default.
        - ContigStr: Components to unfix sequence for, e.g. "A5-10,B1-3" unfixes sequence for residues 5-10 in chain A and residues 1-3 in chain B.
        - Dictionary: Allowed but not recommended.
        NOTE: Excludes ligands (ligands / DNA always has fixed sequence).
    '''.replace('\t\t', '\t')
    )
    # Assignments of conditioning annotations
    # RASA accessibilty
    select_buried: Optional[InputSelection] = Field(None, description="Selection of RASA buried conditioning")
    select_partially_buried: Optional[InputSelection] = Field(None, description="Selection of RASA partially buried conditioning")
    select_exposed: Optional[InputSelection] = Field(None, description="Selection of RASA exposed conditioning")
    # Hotspots & Hbonds
    select_hbond_acceptor: Optional[InputSelection] = Field(None, description="Atom-wise hydrogen bond acceptor")
    select_hbond_donor: Optional[InputSelection] = Field(None, description="Atom-wise hydrogen bond donor")
    select_hotspots: Optional[InputSelection] = Field(None, description="Atom-level or token-level hotspots for PPI")
    redesign_motif_sidechains: Union[bool, str] = Field(False, 
        description="Perform fixed-backbone sequence design on when 'contig' is provided. Changes the default behaviour when not using `select_fixed_atoms`."
    )

    # ========================================================================
    # Global conditioning & symmetry
    # ========================================================================
    # Symmetry
    symmetry: Optional[SymmetryConfig] = Field(None, description="Symmetry specification, see docs/symmetry.md")
    # Centering & COM guidance
    ori_token: Optional[list[float]] = Field(None, description="Origin coordinates")
    infer_ori_strategy: Optional[str] = Field(None, description="Strategy for inferring origin; `com` or `hotspots`")
    # Additional global conditioning
    plddt_enhanced: Optional[bool] = Field(True, description="Enable pLDDT enhancement")
    is_non_loopy: Optional[bool] = Field(None, description="Non-loopy conditioning")
    # Partial diffusion
    partial_t: Optional[float] = Field(None, ge=0.0, description="Angstroms of noise to add for partial diffusion (None turns off partial diffusion), t <= 15 recommended.")
    # fmt: on

    # ========================================================================
    # Properties
    # ========================================================================

    @property
    def is_partial_diffusion(self) -> bool:
        """Whether partial diffusion is enabled."""
        return exists(self.partial_t)

    # ========================================================================
    # Loading / saving
    # ========================================================================

    @classmethod
    def from_json(cls, path):
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)

    @classmethod
    def from_rfd3_out(cls, path: str):
        """Load from path to rfd3 outputs, either .cif, .cif.gz, .json or denoised / noisy trajectory files"""
        path = path.replace(".cif.gz", ".cif").replace(".cif", ".json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Output file not found at {path}")
        with open(path, "r") as f:
            data = json.load(f)
        if "input_specification" in data:
            spec_args = data["input_specification"]
            return cls(**spec_args)
        else:
            raise ValueError(f"No input specification found in json output: {path}")

    def get_dict_to_save(self, exclude_extra: bool = False) -> dict:
        # Returns dictionary for saving (reproducible) outputs to json
        return self.model_dump(
            exclude_defaults=True,
            exclude={"atom_array_input"} | set({"extra"} if exclude_extra else {}),
        )

    # ========================================================================
    # Pre-Validation / canonicalization
    # ========================================================================

    @model_validator(mode="before")
    @classmethod
    def validate_input_schema(cls, data: dict) -> dict:
        if not (
            exists(data.get("input"))
            or exists(data.get("contig"))
            or exists(data.get("non_fixed_contig"))
            or exists(data.get("length"))
        ):
            raise ValueError(
                "Either 'input' or 'contig' / 'non_fixed_contig' / 'length' must be provided."
            )

        # SymMotif placeholder requires symmetry.instances to be defined
        contig_raw = data.get("contig")
        if isinstance(contig_raw, str) and "SymMotif" in [t.strip() for t in contig_raw.split(",")]:
            sym = data.get("symmetry") or {}
            if isinstance(sym, dict):
                sym_id = sym.get("id")
                sym_instances = sym.get("instances")
                sym_mode = sym.get("is_symmetric_motif", True)
            else:
                sym_id = getattr(sym, "id", None)
                sym_instances = getattr(sym, "instances", None)
                sym_mode = getattr(sym, "is_symmetric_motif", True)
            if not sym_id:
                raise ValueError("'SymMotif' placeholder requires 'symmetry.id' to be defined.")
            if not sym_instances:
                raise ValueError("'SymMotif' placeholder requires 'symmetry.instances' to be defined.")
            if sym_mode:
                raise ValueError(
                    "'SymMotif' requires 'symmetry.is_symmetric_motif: false' — "
                    "each instance builds its own motif from the ASU."
                )
            if not exists(data.get("motifs")):
                raise ValueError("'SymMotif' placeholder requires 'motifs' to be defined.")

        # contig and non_fixed_contig are mutually exclusive — combining them leaves the
        # design position of NFC atoms undefined relative to the scaffold.
        if exists(data.get("contig")) and exists(data.get("non_fixed_contig")):
            raise ValueError(
                "'contig' and 'non_fixed_contig' are mutually exclusive. "
                "Use 'contig' (which may reference motif names from the 'motifs' dict) to "
                "position-define the full design chain. Use 'non_fixed_contig' alone when "
                "you want a fully floating design pattern with no position constraint."
            )

        # non_fixed_contig and motifs require an input PDB
        for field in ("non_fixed_contig", "motifs"):
            if exists(data.get(field)) and not (
                exists(data.get("input")) or exists(data.get("atom_array_input"))
            ):
                raise ValueError(
                    f"'{field}' requires 'input' (or 'atom_array_input') to be provided."
                )

        # unused input check
        if exists(data.get("input")) and not (
            (
                exists(data.get("contig"))
                or exists(data.get("non_fixed_contig"))
                or exists(data.get("unindex"))
                or exists(data.get("motifs"))
                or exists(data.get("ligand"))
            )
            or exists(data.get("partial_t"))
        ):
            raise ValueError("Input provided but unused in composition specification.")

        if not exists(data.get("partial_t")):
            # non-partial diffusion checks
            if exists(data.get("unindex")) and not (
                exists(data.get("contig"))
                or exists(data.get("non_fixed_contig"))
                or exists(data.get("length"))
            ):
                raise ValueError(
                    "Unindex provided but neither a length nor contig/non_fixed_contig was specified."
                )
        else:
            # partial diffusion checks
            if exists(data.get("length")):
                raise ValueError(
                    "Length argument must not be provided during partial diffusion."
                )
            if not (exists(data.get("input")) or exists(data.get("atom_array_input"))):
                raise ValueError(
                    "Partial diffusion requires input file or input atom array."
                )

        return data

    @model_validator(mode="before")
    @classmethod
    def canonicalize(cls, data: dict) -> dict:
        # Canonicalize length argument
        data["length"] = str(data["length"]) if exists(data.get("length")) else None

        # Normalize input to str
        data["input"] = str(data["input"]) if exists(data.get("input")) else None
        return data

    @model_validator(mode="before")
    @classmethod
    def load_input(cls, data: dict) -> dict:
        with validator_context("load_input"):
            # ... Find provided selections (InputSelection-typed fields)
            selections = [
                # Motif
                "contig",
                "non_fixed_contig",
                "unindex",
                # Aux
                "select_fixed_atoms",
                "select_unfixed_sequence",
                # Conditioning
                "select_buried",
                "select_partially_buried",
                "select_exposed",
                "select_hbond_acceptor",
                "select_hbond_donor",
                "select_hotspots",
            ]
            selections = [s for s in selections if s in data]

            # ... Early return if no input file provided / atom array input
            if not exists(data.get("input")) and not exists(
                data.get("atom_array_input")
            ):
                if selections:
                    raise ValueError(
                        "Atom array input must be provided before parsing selections: {}".format(
                            selections
                        )
                    )
                return data

            # ... Load atom array from input file if provided
            if exists(data.get("input")):
                if exists(data.get("atom_array_input")):
                    raise ValueError(
                        "Both 'input' and 'atom_array_input' provided; please provide only one."
                    )
                atom_array = inference_load_(
                    data["input"], cif_parser_args=data.get("cif_parser_args")
                )["atom_array"]

                # Center for symmetric design
                if exists(data.get("symmetry")) and data["symmetry"].get("id"):
                    atom_array = center_symmetric_src_atom_array(atom_array)

                if "atom_id" in atom_array.get_annotation_categories():
                    atom_array.del_annotation("atom_id")

                data["atom_array_input"] = atom_array

            atom_array = data["atom_array_input"]

            # ... Set defaults if not provided
            if not exists(data.get("select_fixed_atoms")):
                if exists(data.get("non_fixed_contig")):
                    # NFC standalone: nothing fixed by default.
                    # select_fixed_atoms can re-fix specific atoms at highest priority.
                    data["select_fixed_atoms"] = InputSelection.from_any(
                        False, atom_array=atom_array
                    )
                elif exists(data.get("motifs")):
                    # Motifs present: fix only direct PDB residue tokens in contig.
                    # Strip both motif-name tokens (floating by the motifs unfix block)
                    # and scaffold-count tokens (pure digits like "200" — they have no
                    # corresponding atoms in the input PDB and must not reach InputSelection,
                    # which could accidentally match a PDB residue with that number).
                    contig_raw = data.get("contig")
                    if exists(contig_raw) and isinstance(contig_raw, str):
                        import re as _re
                        _pdb_token = _re.compile(r"^[A-Za-z]\d")
                        motif_keys = set(data["motifs"].keys())
                        direct_parts = [
                            t.strip()
                            for t in contig_raw.split(",")
                            if t.strip() not in motif_keys
                            and _pdb_token.match(t.strip())
                        ]
                        sfa_value = ",".join(direct_parts) if direct_parts else False
                    else:
                        sfa_value = contig_raw or False
                    data["select_fixed_atoms"] = InputSelection.from_any(
                        sfa_value, atom_array=atom_array
                    )
                else:
                    data["select_fixed_atoms"] = InputSelection.from_any(
                        True, atom_array=atom_array
                    )
            if not exists(data.get("select_unfixed_sequence")):
                data["select_unfixed_sequence"] = InputSelection.from_any(
                    False, atom_array=atom_array
                )

            # Validate sequence_unrestrained_motifs: names must all be in motifs
            if exists(data.get("sequence_unrestrained_motifs")):
                if not isinstance(data["sequence_unrestrained_motifs"], list):
                    raise ValueError(
                        "'sequence_unrestrained_motifs' must be a list of motif name strings."
                    )
                motifs_keys = set(data.get("motifs") or {})
                for name in data["sequence_unrestrained_motifs"]:
                    if not isinstance(name, str):
                        raise ValueError(
                            f"'sequence_unrestrained_motifs' entries must be strings, got {type(name)}."
                        )
                    if name not in motifs_keys:
                        raise ValueError(
                            f"'sequence_unrestrained_motifs' entry '{name}' not found in 'motifs' dict."
                        )

            if exists(data.get("unindexed_motifs")):
                if not isinstance(data["unindexed_motifs"], list):
                    raise ValueError(
                        "'unindexed_motifs' must be a list of motif name strings."
                    )
                motifs_keys = set(data.get("motifs") or {})
                for name in data["unindexed_motifs"]:
                    if not isinstance(name, str):
                        raise ValueError(
                            f"'unindexed_motifs' entries must be strings, got {type(name)}."
                        )
                    if name not in motifs_keys:
                        raise ValueError(
                            f"'unindexed_motifs' entry '{name}' not found in 'motifs' dict."
                        )

                if exists(data.get("sequence_unrestrained_motifs")):
                    overlap = set(data["unindexed_motifs"]) & set(
                        data["sequence_unrestrained_motifs"]
                    )
                    if overlap:
                        raise ValueError(
                            f"Motif names must not appear in both 'unindexed_motifs' and "
                            f"'sequence_unrestrained_motifs': {sorted(overlap)}"
                        )

                contig_raw = data.get("contig")
                if isinstance(contig_raw, str):
                    contig_tokens = {t.strip() for t in contig_raw.split(",") if t.strip()}
                    overlap = set(data["unindexed_motifs"]) & contig_tokens
                    if overlap:
                        raise ValueError(
                            f"Motif names must not appear in both 'contig' and "
                            f"'unindexed_motifs': {sorted(overlap)}"
                        )

                    if "SymMotif" in contig_tokens:
                        sym = data.get("symmetry") or {}
                        sym_instances = (
                            sym.get("instances")
                            if isinstance(sym, dict)
                            else getattr(sym, "instances", None)
                        ) or {}
                        sym_motif_names = set()
                        if isinstance(sym_instances, dict):
                            for value in sym_instances.values():
                                if isinstance(value, list):
                                    sym_motif_names.update(
                                        v for v in value if isinstance(v, str)
                                    )
                        overlap = set(data["unindexed_motifs"]) & sym_motif_names
                        if overlap:
                            raise ValueError(
                                f"Motif names assigned via 'SymMotif' must not also appear in "
                                f"'unindexed_motifs': {sorted(overlap)}"
                            )

            # Validate motifs dict: each value must be a parseable contig string
            if exists(data.get("motifs")):
                if not isinstance(data["motifs"], dict):
                    raise ValueError(
                        f"'motifs' must be a dict of {{name: contig_str}}, got {type(data['motifs'])}."
                    )
                for motif_name, motif_contig in data["motifs"].items():
                    if not isinstance(motif_contig, str):
                        raise ValueError(
                            f"Motif '{motif_name}' contig must be a string, got {type(motif_contig)}."
                        )
                    try:
                        InputSelection.from_any(motif_contig, atom_array=atom_array)
                    except Exception as e:
                        raise ValueError(
                            f"Invalid contig string for motif '{motif_name}': {e}"
                        ) from e

            # Coerce selections
            for sele in selections:
                if sele in ["contig", "non_fixed_contig", "unindexed_breaks"]:
                    if exists(data[sele]) and not isinstance(data[sele], str):
                        raise ValueError(
                            f"{sele} selection must be a string or None, got {type(data[sele])} instead."
                        )
                if not isinstance(data.get(sele), InputSelection):
                    if sele == "contig" and isinstance(data.get(sele), str):
                        data[sele] = _input_selection_from_contig_with_placeholders(
                            data[sele],
                            atom_array=atom_array,
                            motif_names=set(data.get("motifs") or {}),
                        )
                    else:
                        data[sele] = InputSelection.from_any(
                            data[sele], atom_array=atom_array
                        )
        return data

    # ========================================================================
    # Post-Validation
    # ========================================================================

    @model_validator(mode="after")
    def assert_exclusivity(self):
        with validator_context("assert_exclusivity"):
            # ... Assert indexed and unindexed do not overlap
            if exists(self.contig) and exists(self.unindex):
                indexed_set = set(self.contig.keys())
                unindexed_set = set(self.unindex.keys())
                overlap = indexed_set & unindexed_set
                if overlap:
                    raise ValueError(
                        f"Indexed and unindexed components must not overlap, got: {overlap}"
                    )

            # ... Assert contig / non_fixed_contig / motifs are disjoint (atom-level)
            if exists(self.atom_array_input):
                _sele_masks: dict[str, np.ndarray] = {}
                if exists(self.contig):
                    _sele_masks["contig"] = self.contig.get_mask()
                if exists(self.non_fixed_contig):
                    _sele_masks["non_fixed_contig"] = self.non_fixed_contig.get_mask()
                if exists(self.motifs):
                    for mname, mcontig in self.motifs.items():
                        _sele_masks[f"motifs.{mname}"] = InputSelection.from_any(
                            mcontig, atom_array=self.atom_array_input
                        ).get_mask()
                _names = list(_sele_masks.keys())
                _masks = list(_sele_masks.values())
                for i in range(len(_names)):
                    for j in range(i + 1, len(_names)):
                        if np.any(_masks[i] & _masks[j]):
                            raise ValueError(
                                f"Selections '{_names[i]}' and '{_names[j]}' overlap; "
                                "contig, non_fixed_contig, and motifs must be disjoint."
                            )

            # ... Assert mutual exclusivity of rasa binning
            exclusive_sets = [
                ("Motifs", ("contig", "unindex")),
                (
                    "RASA",
                    ("select_buried", "select_partially_buried", "select_exposed"),
                ),
            ]

            for name, excl_set in exclusive_sets:
                masks = [getattr(self, field, None) for field in excl_set]
                masks = [m.get_mask() for m in masks if m is not None]
                if not masks:
                    continue
                mask_sum = np.zeros_like(masks[0], dtype=int)
                for m in masks:
                    if m is not None:
                        mask_sum += m.astype(int)
                if np.any(mask_sum > 1):
                    raise ValueError(
                        f"Selections for `{name}` must be mutually exclusive, got overlapping selections: {excl_set}. Mask sum: {mask_sum}"
                    )

        return self

    @model_validator(mode="after")
    def attempt_expansion(self):
        if self.is_partial_diffusion and exists(self.contig):
            contig = self.contig
            length = self.length
            try:
                get_design_pattern_with_constraints(contig.raw, length=length)
            except Exception as e:
                raise ValueError(f"Failed to expand contig ({contig.raw}): {e}")
        return self

    @model_validator(mode="after")
    def _assign_types_to_input(self):
        """Assign conditioning annotations to the input atom array"""
        aa = self.atom_array_input
        if not exists(aa):
            return self

        # ... Selections and their annotation values
        selection_fields = {
            # field name:         (annotation name, assigned value, non-selected value)
            "select_fixed_atoms": ("is_motif_atom_with_fixed_coord", True, False),
            "select_unfixed_sequence": ("is_motif_atom_with_fixed_seq", False, True),
            "unindex": ("is_motif_atom_unindexed", True, False),
            "select_hotspots": ("is_atom_level_hotspot", True, False),
            "select_hbond_acceptor": ("active_acceptor", True, False),
            "select_hbond_donor": ("active_donor", True, False),
            "select_buried": ("rasa_bin", 0, 3),
            "select_partially_buried": ("rasa_bin", 1, 3),
            "select_exposed": ("rasa_bin", 2, 3),
        }
        selection_fields = {
            k: v for k, v in selection_fields.items() if exists(getattr(self, k, None))
        }

        # ... Init global
        [
            aa.set_annotation(name, np.full(aa.array_length(), val, dtype=int))
            for name, val in REQUIRED_CONDITIONING_ANNOTATION_VALUES.items()
        ]
        aa.set_annotation(
            UNINDEXED_FLOATING_MOTIF_ANNOTATION,
            np.zeros(aa.array_length(), dtype=int),
        )

        # Priority unfix: NFC and named motifs default to floating (is_motif_atom_with_fixed_coord=0).
        # This runs AFTER global init (which sets all to True) but BEFORE apply_selections,
        # so that an explicit select_fixed_atoms can still override these atoms to True.
        if exists(self.non_fixed_contig):
            nfc_mask = self.non_fixed_contig.get_mask()
            aa.is_motif_atom_with_fixed_coord[nfc_mask] = 0

        if exists(self.motifs):
            for _mname, _mcontig in self.motifs.items():
                _msele = InputSelection.from_any(_mcontig, atom_array=aa)
                aa.is_motif_atom_with_fixed_coord[_msele.get_mask()] = 0

        if exists(self.unindexed_motifs) and UNINDEXED_MOTIFS_USE_LEGACY_TRUE_UNINDEX:
            for _mname in self.unindexed_motifs:
                _mcontig = self.motifs[_mname]
                _msele = InputSelection.from_any(_mcontig, atom_array=aa)
                _mask = _msele.get_mask()
                aa.is_motif_atom_unindexed[_mask] = 1
                aa.get_annotation(UNINDEXED_FLOATING_MOTIF_ANNOTATION)[_mask] = 1
        elif exists(self.unindexed_motifs):
            for _mname in self.unindexed_motifs:
                _mcontig = self.motifs[_mname]
                _msele = InputSelection.from_any(_mcontig, atom_array=aa)
                _mask = _msele.get_mask()
                aa.get_annotation(UNINDEXED_FLOATING_MOTIF_ANNOTATION)[_mask] = 1

        # Application of selections to each token fn;
        def apply_selections(start, end):
            chain_id = aa.chain_id[start]
            res_id = aa.res_id[start]

            # Assign all select fields to atom array annotations.
            for selection_name, (
                annotation_name,
                set_value,
                default_value,
            ) in selection_fields.items():
                # ... Get input values
                selection = getattr(self, selection_name)

                # Important line: selects from data dictionary based on src chain & res_id (Not name!)
                atom_names_sele = selection.get(f"{chain_id}{res_id}")

                if atom_names_sele is None:
                    continue
                mask = np.isin(aa.atom_name[start:end], atom_names_sele)
                if annotation_name in aa.get_annotation_categories():
                    # ... Set only mask overridden features if exists in atom array
                    aa.get_annotation(annotation_name)[start:end] = np.where(
                        mask, set_value, default_value
                    ).astype(np.int_)
                    # ).astype(int)
                else:
                    # ... Otherwise, set the entire annotation and use defaults for unselected
                    mask_aa = np.zeros(aa.array_length(), dtype=bool)
                    mask_aa[start:end] = mask
                    annotation_values = np.where(
                        mask_aa,
                        set_value,
                        default_value,
                    ).astype(np.int_)
                    aa.set_annotation(annotation_name, annotation_values)

        # ... Set default assignments per-token based on whether redesigning
        starts = get_residue_starts(aa, add_exclusive_stop=True)
        for start, end in zip(starts[:-1], starts[1:]):
            # ... Relax sequence and sidechains
            if aa.res_name[start] in STANDARD_AA and self.redesign_motif_sidechains:
                is_bkbn = np.isin(aa.atom_name[start:end], ["N", "CA", "C", "O"])
                aa.is_motif_atom_with_fixed_coord[start:end] = is_bkbn.astype(int)
                aa.is_motif_atom_with_fixed_seq[start:end] = np.full_like(
                    is_bkbn, False, dtype=int
                )

            # ... Apply selections on top
            apply_selections(start, end)

        return self

    # ========================================================================
    # Building
    # ========================================================================

    def build(self, return_metadata=False):
        """Main build pipeline."""
        atom_array_input_annotated = copy.deepcopy(self.atom_array_input)
        atom_array = self._build_init(atom_array_input_annotated)

        # Apply post-processing
        atom_array = self._append_ligand(atom_array, atom_array_input_annotated)
        atom_array = self._apply_symmetry(atom_array, atom_array_input_annotated)
        atom_array = self._mark_unindexed_named_motifs(
            atom_array, atom_array_input_annotated
        )
        atom_array = annotate_floating_motif_reference_coords(atom_array)

        # Apply globals to all tokens (including diffused)
        atom_array = self._set_origin(atom_array)
        atom_array = self._apply_globals(atom_array)

        # Final validation and cleanup
        check_has_required_conditioning_annotations(
            atom_array, required=REQUIRED_INFERENCE_ANNOTATIONS
        )
        convert_existing_annotations_to_bool(atom_array)

        # ... Route return type
        if not return_metadata:
            return copy.deepcopy(atom_array)
        else:
            metadata = self.get_dict_to_save()
            metadata["extra"] = metadata.get("extra", {}) | {
                "num_tokens_in": len(get_token_starts(atom_array)),
                "num_residues_in": len(get_residue_starts(atom_array)),
                "num_chains": len(np.unique(atom_array.chain_id)),
                "num_atoms": len(atom_array),
                "num_residues": len(
                    np.unique(list(zip(atom_array.chain_id, atom_array.res_id)))
                ),
            }
            return copy.deepcopy(atom_array), metadata

    # ============================================================================
    # Building functions
    # ============================================================================

    def _build_init(self, atom_array_input_annotated):
        # Build the design pattern. contig and non_fixed_contig are mutually exclusive.
        # contig may embed motif-name tokens (keys in self.motifs): those are resolved to
        # their PDB residue strings and inlined in the main chain at that position.
        # Motifs NOT referenced by name in contig can either be added as true unindexed
        # motifs (position-free within the main chain) or appended as separate chains.
        indexed_tokens: dict = {}
        _referenced_motifs: set = set()

        if exists(self.contig):
            # Collect direct PDB tokens. Placeholder and scaffold tokens are retained
            # in self.contig.raw for design resolution but omitted from selection.data.
            indexed_tokens.update(
                _get_tokens_from_selection_data(self.contig, atom_array_input_annotated)
            )
            # Resolve motif-name and SymMotif tokens. SymMotif uses instance 0's motif
            # for the main chain build; _apply_symmetry rebuilds each instance.
            resolved_parts = []
            for _tok in self.contig.raw.split(","):
                _tok = _tok.strip()
                if _tok == "SymMotif":
                    # Resolve to instance 0's motif for the main chain build.
                    # Validated in validate_input_schema: instances["0"] and motifs exist.
                    _sym_motif_name = (self.symmetry.instances.get("0") or [])[0]
                    _referenced_motifs.add(_sym_motif_name)
                    _msele = InputSelection.from_any(
                        self.motifs[_sym_motif_name], atom_array=atom_array_input_annotated
                    )
                    indexed_tokens.update(_msele.get_tokens(atom_array_input_annotated))
                    resolved_parts.append(self.motifs[_sym_motif_name])
                elif exists(self.motifs) and _tok in self.motifs:
                    _referenced_motifs.add(_tok)
                    _msele = InputSelection.from_any(
                        self.motifs[_tok], atom_array=atom_array_input_annotated
                    )
                    indexed_tokens.update(
                        _msele.get_tokens(atom_array_input_annotated)
                    )
                    resolved_parts.append(self.motifs[_tok])
                else:
                    resolved_parts.append(_tok)
            _design_contig = ",".join(resolved_parts)

        elif exists(self.non_fixed_contig):
            _design_contig = self.non_fixed_contig.raw
            indexed_tokens.update(
                self.non_fixed_contig.get_tokens(atom_array_input_annotated)
            )

        else:
            _design_contig = None

        unindexed_tokens = (
            self.unindex.get_tokens(atom_array_input_annotated)
            if exists(self.unindex)
            else {}
        )
        # Subset to only fixed coordindate atoms
        unindexed_tokens = {
            k: tok[tok.is_motif_atom_with_fixed_coord.astype(bool)]
            for k, tok in unindexed_tokens.items()
        }
        unindexed_components, unindexed_breaks = self.break_unindexed(self.unindex)
        inline_motif_tokens = {}
        inline_motif_components = []
        if UNINDEXED_MOTIFS_USE_LEGACY_TRUE_UNINDEX:
            (
                unindexed_motif_tokens,
                unindexed_motif_components,
                unindexed_motif_breaks,
            ) = self._get_unindexed_named_motif_payload(
                atom_array_input_annotated, excluded_motifs=_referenced_motifs
            )
            unindexed_tokens.update(unindexed_motif_tokens)
            unindexed_components = unindexed_motif_components + unindexed_components
            unindexed_breaks = unindexed_motif_breaks + unindexed_breaks
        else:
            (
                inline_motif_tokens,
                inline_motif_components,
            ) = self._get_inline_named_motif_payload(
                atom_array_input_annotated, excluded_motifs=_referenced_motifs
            )
            indexed_tokens.update(inline_motif_tokens)

        if not self.is_partial_diffusion:
            # ... Sample the contig string
            effective_length = self._adjust_length_for_inline_motifs(
                self.length, inline_motif_components
            )
            if exists(_design_contig) or exists(self.length):
                components_to_accumulate = get_design_pattern_with_constraints(
                    _design_contig if _design_contig else effective_length,
                    length=effective_length,
                )
            else:
                components_to_accumulate = []

            if inline_motif_components:
                components_to_accumulate = self._insert_inline_named_motifs(
                    components_to_accumulate,
                    inline_motif_components,
                )

            self.extra["sampled_contig"] = ",".join(
                [str(x) for x in components_to_accumulate]
            )

            # ... Include unindexed components in accumulation
            unindexed_breaks = [None] * len(components_to_accumulate) + unindexed_breaks
            components_to_accumulate += unindexed_components

            # ... Accumulate from scratch
            atom_array = accumulate_components(
                components_to_accumulate,
                indexed_tokens=indexed_tokens,
                unindexed_tokens=unindexed_tokens,
                atom_array_accum=[],
                unindexed_breaks=unindexed_breaks,
                start_chain="A",
                start_resid=1,
            )
        else:
            # ... Set common annotations
            atom_array_in = assign_types_(copy.deepcopy(atom_array_input_annotated))
            atom_array_in = set_common_annotations(
                atom_array_in, set_src_component_to_res_name=False
            )

            # ... Override motif annotations from pipeline
            zeros = np.zeros(atom_array_in.array_length(), dtype=int)
            atom_array_in.is_motif_atom_unindexed = (
                zeros  # reset unindexed annotation since those are copied already.
            )
            atom_array_in.is_motif_atom_with_fixed_coord = (
                self.select_fixed_atoms.get_mask().astype(int)
                if exists(self.select_fixed_atoms)
                else zeros
            )
            atom_array_in.is_motif_atom_with_fixed_seq = (
                ~self.select_unfixed_sequence.get_mask()
                if exists(self.select_unfixed_sequence)
                else zeros
            ).astype(int)

            # ... Subset to residues only
            atom_array_in = atom_array_in[atom_array_in.is_protein]

            # ... Set chain ID for unindexed residues as whatever the input has
            start_resid = np.max(atom_array_in.res_id) + 1
            start_chain = atom_array_in.chain_id[0]

            # ... Accumulate from input
            components_to_accumulate = unindexed_components
            atom_array = accumulate_components(
                # No accumulation of components
                components_to_accumulate=components_to_accumulate,
                indexed_tokens={},
                # Append all inputs to unindexed tokens
                unindexed_tokens=unindexed_tokens,
                atom_array_accum=[atom_array_in],
                start_chain=start_chain,
                start_resid=start_resid,
                unindexed_breaks=unindexed_breaks,
            )

        # Append motifs listed in sequence_unrestrained_motifs as separate floating chains.
        # Motifs referenced inline (via contig name-tokens or SymMotif) are already in the
        # main chain; they must not also appear in sequence_unrestrained_motifs.
        if exists(self.sequence_unrestrained_motifs) and not self.is_partial_diffusion:
            to_append = {
                k: self.motifs[k]
                for k in self.sequence_unrestrained_motifs
                if k in (self.motifs or {}) and k not in _referenced_motifs
            }
            if to_append:
                atom_array = self._append_named_motifs(
                    atom_array, atom_array_input_annotated, motifs_to_append=to_append
                )

        return atom_array

    def _mark_unindexed_named_motifs(self, atom_array, atom_array_input_annotated):
        """Mark built atoms originating from unindexed_motifs for Kabsch eligibility."""
        if not exists(self.unindexed_motifs):
            return atom_array

        atom_array.set_annotation(
            UNINDEXED_FLOATING_MOTIF_ANNOTATION,
            np.zeros(atom_array.array_length(), dtype=int),
        )
        src = np.asarray(atom_array.src_component).astype(str)
        marked = atom_array.get_annotation(UNINDEXED_FLOATING_MOTIF_ANNOTATION)

        for motif_name in self.unindexed_motifs:
            motif_tokens = InputSelection.from_any(
                self.motifs[motif_name], atom_array=atom_array_input_annotated
            ).get_tokens(atom_array_input_annotated)
            if not motif_tokens:
                continue
            motif_components = set(motif_tokens.keys())
            marked[np.isin(src, list(motif_components))] = 1

        atom_array.set_annotation(UNINDEXED_FLOATING_MOTIF_ANNOTATION, marked)
        return atom_array

    def _get_inline_named_motif_payload(
        self, atom_array_input_annotated, excluded_motifs: Optional[set[str]] = None
    ) -> tuple[dict, list[list[str]]]:
        """Return inline motif tokens/components for hidden sampled placement.

        This is the active unindexed_motifs path: motifs are inserted into a sampled
        inline layout before diffusion, rather than diffused as appended true-unindex
        guideposts and cleaned up afterward.
        """
        if not exists(self.unindexed_motifs):
            return {}, []

        excluded_motifs = excluded_motifs or set()
        indexed_tokens = {}
        inline_components = []
        for motif_name in self.unindexed_motifs:
            if motif_name in excluded_motifs:
                continue
            motif_contig_str = self.motifs[motif_name]
            motif_tokens = InputSelection.from_any(
                motif_contig_str, atom_array=atom_array_input_annotated
            ).get_tokens(atom_array_input_annotated)
            indexed_tokens.update(motif_tokens)
            inline_components.append(
                get_design_pattern_with_constraints(motif_contig_str)
            )
        return indexed_tokens, inline_components

    def _insert_inline_named_motifs(
        self,
        components_to_accumulate: list,
        inline_motif_components: list[list[str]],
    ) -> list:
        """Insert unindexed_motifs into a hidden sampled inline layout.

        The base design pattern is first sampled as usual, then expanded to
        residue-level free components so the motif segments can be inserted at
        random sequence slots without the user specifying an explicit contig.
        """
        if not inline_motif_components:
            return components_to_accumulate

        expanded_segments = self._expand_components_for_inline_insertion(
            components_to_accumulate
        )
        n_slots = sum(len(segment) + 1 for segment in expanded_segments)
        if n_slots <= 0:
            expanded_segments = [[]]
            n_slots = 1

        slot_indices = sorted(random.choices(range(n_slots), k=len(inline_motif_components)))
        motif_iter = iter(inline_motif_components)
        slot_iter = iter(slot_indices)
        next_slot = next(slot_iter, None)
        global_slot = 0
        rebuilt_segments = []

        for segment in expanded_segments:
            rebuilt = []
            for local_slot in range(len(segment) + 1):
                while next_slot == global_slot:
                    rebuilt.extend(next(motif_iter))
                    next_slot = next(slot_iter, None)
                if local_slot < len(segment):
                    rebuilt.append(segment[local_slot])
                global_slot += 1
            rebuilt_segments.append(rebuilt)

        return self._compress_inline_components(rebuilt_segments)

    @staticmethod
    def _adjust_length_for_inline_motifs(
        length: Optional[str], inline_motif_components: list[list[str]]
    ) -> Optional[str]:
        if not exists(length) or not inline_motif_components:
            return length

        motif_len = sum(len(components) for components in inline_motif_components)
        if "-" in length:
            length_min, length_max = map(int, str(length).split("-"))
            length_min -= motif_len
            length_max -= motif_len
            if length_min < 0 or length_max < 0 or length_min > length_max:
                raise ValueError(
                    "Inline unindexed_motifs exceed the available length budget."
                )
            return f"{length_min}-{length_max}"

        adjusted = int(length) - motif_len
        if adjusted < 0:
            raise ValueError(
                "Inline unindexed_motifs exceed the available length budget."
            )
        return str(adjusted)

    @staticmethod
    def _expand_components_for_inline_insertion(components_to_accumulate: list) -> list[list]:
        segments = [[]]
        for component in components_to_accumulate:
            comp = str(component)
            if comp == "/0":
                segments.append([])
                continue
            if comp and comp[0].isdigit():
                suffix = comp[-1] if comp[-1].isalpha() else ""
                count = int(comp[:-1] if suffix else comp)
                unit = f"1{suffix}" if suffix else 1
                segments[-1].extend([unit] * count)
            else:
                segments[-1].append(component)
        return segments

    @staticmethod
    def _compress_inline_components(segments: list[list]) -> list:
        components = []
        for seg_idx, segment in enumerate(segments):
            run_suffix = None
            run_count = 0
            for component in segment:
                comp = str(component)
                if comp.startswith("1") and (len(comp) == 1 or comp[1:].isalpha()):
                    suffix = comp[1:] if len(comp) > 1 else ""
                    if run_suffix == suffix:
                        run_count += 1
                    else:
                        if run_count:
                            components.append(f"{run_count}{run_suffix}" if run_suffix else run_count)
                        run_suffix = suffix
                        run_count = 1
                    continue

                if run_count:
                    components.append(f"{run_count}{run_suffix}" if run_suffix else run_count)
                    run_suffix = None
                    run_count = 0
                components.append(component)

            if run_count:
                components.append(f"{run_count}{run_suffix}" if run_suffix else run_count)
            if seg_idx < len(segments) - 1:
                components.append("/0")
        return components

    def _get_unindexed_named_motif_payload(
        self, atom_array_input_annotated, excluded_motifs: Optional[set[str]] = None
    ) -> tuple[dict, list, list]:
        """Return true-unindexed motif tokens/components/breaks for named motifs.

        These motifs follow the regular unindex pathway for position-free sequence placement,
        but retain a dedicated annotation so floating motif projection still treats them as
        rigid floating motifs.
        """
        if not exists(self.unindexed_motifs):
            return {}, [], []

        excluded_motifs = excluded_motifs or set()
        components = []
        breaks = []
        unindexed_tokens = {}
        for motif_name in self.unindexed_motifs:
            if motif_name in excluded_motifs:
                continue
            motif_contig_str = self.motifs[motif_name]
            motif_components, motif_breaks = get_motif_components_and_breaks(
                motif_contig_str
            )
            components.extend(motif_components)
            breaks.extend(motif_breaks)
            motif_tokens = InputSelection.from_any(
                motif_contig_str, atom_array=atom_array_input_annotated
            ).get_tokens(atom_array_input_annotated)
            unindexed_tokens.update(motif_tokens)
        return unindexed_tokens, components, breaks

    # ============================================================================
    # Auxiliary functions
    # ============================================================================

    def _append_named_motifs(
        self, atom_array, atom_array_input_annotated, motifs_to_append=None
    ):
        """Append each entry in motifs_to_append as a separate chain.

        Each motif is built via accumulate_components using alphabetic src_component
        (so Kabsch alignment picks it up). is_motif_atom_with_fixed_coord is already
        set to False in _assign_types_to_input; create_motif_residue preserves it.
        Defaults to self.motifs when motifs_to_append is None.
        """
        if motifs_to_append is None:
            motifs_to_append = self.motifs
        used_chains = set(np.unique(atom_array.chain_id))

        for motif_name, motif_contig_str in motifs_to_append.items():
            # Parse contig → components + tokens
            motif_components = get_design_pattern_with_constraints(motif_contig_str)
            motif_tokens = InputSelection.from_any(
                motif_contig_str, atom_array=atom_array_input_annotated
            ).get_tokens(atom_array_input_annotated)

            next_chain = _next_chain_id(used_chains)
            used_chains.add(next_chain)

            motif_aa = accumulate_components(
                motif_components,
                indexed_tokens=motif_tokens,
                unindexed_tokens={},
                atom_array_accum=[],
                unindexed_breaks=[None] * len(motif_components),
                start_chain=next_chain,
                start_resid=1,
            )

            # Harmonize annotations so struc.concatenate keeps them all
            all_defaults = {
                **REQUIRED_CONDITIONING_ANNOTATION_VALUES,
                **OPTIONAL_CONDITIONING_VALUES,
            }
            for annot, default in all_defaults.items():
                _dtype = np.float64 if isinstance(default, float) else int
                if (
                    annot in atom_array.get_annotation_categories()
                    and annot not in motif_aa.get_annotation_categories()
                ):
                    motif_aa.set_annotation(
                        annot,
                        np.full(motif_aa.array_length(), default, dtype=_dtype),
                    )
                elif (
                    annot in motif_aa.get_annotation_categories()
                    and annot not in atom_array.get_annotation_categories()
                ):
                    atom_array.set_annotation(
                        annot,
                        np.full(atom_array.array_length(), default, dtype=_dtype),
                    )

            atom_array = struc.concatenate([atom_array, motif_aa])
            logger.info(
                f"Appended motif '{motif_name}' ({motif_contig_str}) as chain {next_chain} "
                f"({motif_aa.array_length()} atoms, floating)."
            )

        return atom_array

    @staticmethod
    def break_unindexed(unindex: InputSelection):
        if not exists(unindex):
            return [], []

        # ... If original type was string, use that
        if isinstance(unindex.raw, str):
            unindexed_string = unindex.raw
        elif isinstance(unindex.raw, dict):
            unindexed_string = ",".join(unindex.raw.keys())
        else:
            logger.info(
                "`Unindex` provided as non-string, separate keys in dictionary will be considered separate contiguous components"
            )
            unindexed_string = ",".join(unindex.keys())

        # ... Break expected unindexed contig string
        unindexed_components, breaks = get_motif_components_and_breaks(unindexed_string)

        return unindexed_components, breaks

    # ============================================================================
    # Setter functions
    # ============================================================================

    def _append_ligand(self, atom_array, atom_array_input_annotated):
        """Append ligand if specified."""
        if exists(self.ligand):
            ligand_array = extract_ligand_array(
                atom_array_input_annotated,
                self.ligand,
                fixed_atoms={},
                set_defaults=False,
                additional_annotations=set(
                    list(atom_array.get_annotation_categories())
                    + list(atom_array_input_annotated.get_annotation_categories())
                ),
            )
            # Validate chain assignments — chain ID is leaked to the model
            # so collisions are a significant deviation from convention.
            ligand_chains = np.unique(ligand_array.chain_id)
            existing_chains = set(np.unique(atom_array.chain_id))
            overlapping = sorted(existing_chains & set(ligand_chains))
            if not self.allow_ligand_on_existing_chain:
                if overlapping:
                    raise ValueError(
                        f"Ligand chain(s) {overlapping} overlap with existing "
                        f"chain(s) {sorted(existing_chains)}. Place ligands on "
                        f"separate chains or set 'allow_ligand_on_existing_chain: "
                        f"true' to restore the old behaviour."
                    )
                # Multiple ligands must each be on their own chain.
                for chain in ligand_chains:
                    n_residues = len(
                        np.unique(ligand_array.res_id[ligand_array.chain_id == chain])
                    )
                    if n_residues > 1:
                        raise ValueError(
                            f"Multiple ligand residues on chain {chain}. Each "
                            f"ligand must be on its own chain, or set "
                            f"'allow_ligand_on_existing_chain: true' to restore "
                            f"the old behaviour."
                        )
            if self.allow_ligand_on_existing_chain:
                # Legacy behaviour: offset from protein max to avoid clashes.
                ligand_array.res_id = (
                    ligand_array.res_id
                    - np.min(ligand_array.res_id)
                    + np.max(atom_array.res_id)
                    + 1
                )
            else:
                # Reset ligand res_id to start from 1 per chain, matching
                # the convention AF3 uses in its output CIF files.
                for chain in ligand_chains:
                    mask = ligand_array.chain_id == chain
                    ligand_array.res_id[mask] = 1
            # Harmonize conditioning annotations before concatenation: biotite's
            # concatenate only preserves annotations present in ALL arrays (set
            # intersection), so mismatched optional conditioning annotations
            # (e.g. is_atom_level_hotspot) get silently dropped.
            for annot, default in OPTIONAL_CONDITIONING_VALUES.items():
                if (
                    annot in ligand_array.get_annotation_categories()
                    and annot not in atom_array.get_annotation_categories()
                ):
                    atom_array.set_annotation(
                        annot, np.full(atom_array.array_length(), default)
                    )
                elif (
                    annot in atom_array.get_annotation_categories()
                    and annot not in ligand_array.get_annotation_categories()
                ):
                    ligand_array.set_annotation(
                        annot, np.full(ligand_array.array_length(), default)
                    )
            atom_array = atom_array + ligand_array
        return atom_array

    def _apply_symmetry(self, atom_array, atom_array_input_annotated):
        """Apply symmetry transformation if specified."""
        if not (exists(self.symmetry) and self.symmetry.id):
            return atom_array
        if self._has_sym_motif:
            return self._apply_sym_motif_symmetry(atom_array, atom_array_input_annotated)
        return make_symmetric_atom_array(
            atom_array,
            self.symmetry,
            sm=self.ligand,
            src_atom_array=atom_array_input_annotated,
        )

    @property
    def _has_sym_motif(self) -> bool:
        """True when the contig contains a SymMotif placeholder token."""
        return (
            exists(self.contig)
            and "SymMotif" in [t.strip() for t in self.contig.raw.split(",")]
        )

    def _apply_sym_motif_symmetry(self, atom_array_instance0, atom_array_input_annotated):
        """Build one chain per symmetric instance with its specific motif, then apply frames.

        atom_array_instance0 is used only to derive symmetry config; it is then discarded
        and all instances (including 0) are rebuilt by _build_sym_motif_instance so that
        each carries its own motif atoms with correct annotations.
        """
        if self.ligand:
            raise NotImplementedError(
                "Ligands combined with SymMotif symmetry are not yet supported."
            )

        sym_conf = check_symmetry_config(
            atom_array_instance0,
            self.symmetry,
            sm=None,
            has_dist_cond=False,
            src_atom_array=atom_array_input_annotated,
        )
        frames = get_symmetry_frames_from_symmetry_id(sym_conf)

        symmetry_unit_list = []
        for transform_id, frame in enumerate(frames):
            instance_aa = self._build_sym_motif_instance(
                transform_id, atom_array_input_annotated
            )
            instance_aa = add_sym_annotations(instance_aa, sym_conf)

            # Motif atoms are already at their native PDB positions (which for a
            # C2-symmetric complex are already in the correct frame for each instance).
            # get_symmetry_unit would apply the C2 frame on top of that, double-rotating
            # motif atoms back to instance-0's position.  Save and restore them so only
            # the sym_transform annotations are affected, not the coordinates.
            _is_motif_atom = np.asarray(
                [bool(c) and c[0].isalpha() for c in instance_aa.src_component]
            )
            _saved_motif_coords = instance_aa.coord[_is_motif_atom].copy()

            sym_unit = get_symmetry_unit(instance_aa, transform_id, frame)

            sym_unit.coord[_is_motif_atom] = _saved_motif_coords
            symmetry_unit_list.append(sym_unit)

        result = struc.concatenate(symmetry_unit_list)
        if {"_is_motif", "_is_indexed_motif"}.issubset(
            result.get_annotation_categories()
        ):
            result = fix_3D_sym_motif_annotations(result)
        result = add_src_sym_component_annotations(result)
        return result

    def _build_sym_motif_instance(self, instance_idx: int, atom_array_input_annotated):
        """Build one instance's chain with SymMotif resolved to that instance's motif.

        Falls back to instance 0's motif if this instance_idx is not in instances.
        """
        instances = self.symmetry.instances or {}
        motif_names = instances.get(str(instance_idx)) or instances.get("0") or []
        if not motif_names:
            raise ValueError(
                f"SymMotif: no motif defined for instance {instance_idx} and no fallback at '0'."
            )
        motif_name = motif_names[0]
        if motif_name not in (self.motifs or {}):
            raise ValueError(
                f"SymMotif instance {instance_idx}: motif '{motif_name}' not in motifs dict."
            )
        motif_contig_str = self.motifs[motif_name]

        # Resolve the contig with this instance's motif substituted for SymMotif.
        resolved_parts = []
        indexed_tokens: dict = {}
        referenced_motifs = {motif_name}
        for _tok in self.contig.raw.split(","):
            _tok = _tok.strip()
            if _tok == "SymMotif":
                resolved_parts.append(motif_contig_str)
                _msele = InputSelection.from_any(
                    motif_contig_str, atom_array=atom_array_input_annotated
                )
                indexed_tokens.update(_msele.get_tokens(atom_array_input_annotated))
            elif exists(self.motifs) and _tok in self.motifs:
                referenced_motifs.add(_tok)
                resolved_parts.append(self.motifs[_tok])
                _msele = InputSelection.from_any(
                    self.motifs[_tok], atom_array=atom_array_input_annotated
                )
                indexed_tokens.update(_msele.get_tokens(atom_array_input_annotated))
            else:
                resolved_parts.append(_tok)

        resolved_contig = ",".join(resolved_parts)
        if UNINDEXED_MOTIFS_USE_LEGACY_TRUE_UNINDEX:
            components = get_design_pattern_with_constraints(
                resolved_contig, length=self.length
            )
            (
                unindexed_motif_tokens,
                unindexed_motif_components,
                unindexed_motif_breaks,
            ) = self._get_unindexed_named_motif_payload(
                atom_array_input_annotated, excluded_motifs=referenced_motifs
            )
            components += unindexed_motif_components
        else:
            (
                inline_motif_tokens,
                inline_motif_components,
            ) = self._get_inline_named_motif_payload(
                atom_array_input_annotated, excluded_motifs=referenced_motifs
            )
            effective_length = self._adjust_length_for_inline_motifs(
                self.length, inline_motif_components
            )
            components = get_design_pattern_with_constraints(
                resolved_contig, length=effective_length
            )
            indexed_tokens.update(inline_motif_tokens)
            components = self._insert_inline_named_motifs(
                components,
                inline_motif_components,
            )
            unindexed_motif_tokens = {}
            unindexed_motif_breaks = []
        return accumulate_components(
            components,
            indexed_tokens=indexed_tokens,
            unindexed_tokens=unindexed_motif_tokens,
            atom_array_accum=[],
            unindexed_breaks=([None] * (len(components) - len(unindexed_motif_breaks)))
            + unindexed_motif_breaks,
            start_chain="A",
            start_resid=1,
        )

    def _set_origin(self, atom_array):
        """Set origin token and initialize coordinates."""
        if self.is_partial_diffusion:
            # Partial diffusion: use COM, keep all coordinates
            if exists(self.symmetry) and self.symmetry.id:
                # For symmetric structures, avoid COM centering that would collapse chains
                logger.info(
                    "Partial diffusion with symmetry: skipping COM centering to preserve chain spacing"
                )
            else:
                coord_before_origin = atom_array.coord.copy()
                atom_array = set_com(
                    atom_array, ori_token=None, infer_ori_strategy="com"
                )
                atom_array = _shift_floating_motif_reference_coords(
                    atom_array,
                    _infer_uniform_coordinate_shift(
                        coord_before_origin, atom_array.coord
                    ),
                )
        else:
            # Standard: set ori token, zero out diffused atoms
            coord_before_origin = atom_array.coord.copy()
            atom_array = set_com(
                atom_array,
                ori_token=self.ori_token,
                infer_ori_strategy=self.infer_ori_strategy,
            )
            atom_array = _shift_floating_motif_reference_coords(
                atom_array,
                _infer_uniform_coordinate_shift(coord_before_origin, atom_array.coord),
            )
            # Diffused atoms are always initialized at origin during regular diffusion (all information removed)
            atom_array.coord[
                ~atom_array.is_motif_atom_with_fixed_coord.astype(bool)
            ] = 0.0
        return atom_array

    def _apply_globals(self, atom_array):
        # Temperature conditioning
        if exists(self.is_non_loopy):
            is_non_loopy_annot = np.zeros(atom_array.array_length(), dtype=int)
            is_motif_token = get_motif_features(atom_array)["is_motif_token"]
            diffused_region_mask = ~(is_motif_token.astype(bool))
            if exists(self.is_non_loopy):
                is_non_loopy_annot[diffused_region_mask] = (
                    1 if self.is_non_loopy else -1
                )
            atom_array.set_annotation("is_non_loopy", is_non_loopy_annot)
            atom_array.set_annotation("is_non_loopy_atom_level", is_non_loopy_annot)
        else:
            zeros = np.zeros(atom_array.array_length(), dtype=int)
            atom_array.set_annotation("is_non_loopy", zeros)
            atom_array.set_annotation("is_non_loopy_atom_level", zeros)

        if self.plddt_enhanced:
            atom_array.set_annotation(
                "ref_plddt", np.full((atom_array.array_length(),), True, dtype=int)
            )

        # Partial diffusion time annotation
        if self.is_partial_diffusion:
            atom_array.set_annotation(
                "partial_t", np.full(atom_array.shape[0], self.partial_t, dtype=float)
            )
        return atom_array

    @classmethod
    def safe_init(cls, **spec_kwargs):
        if spec_kwargs.get("dialect", 2) < 2:
            warn = (
                "Using dialect==1, which is deprecated and will be removed in future releases. "
                "Please update your input specification to dialect=2 and use the new schema if possible"
            )
            warnings.warn(warn, DeprecationWarning)
            logger.warning(warn)
            return LegacySpecification(**spec_kwargs)
        else:
            return cls(**spec_kwargs)

    def to_pipeline_input(self, example_id):
        atom_array, spec_dict = self.build(return_metadata=True)

        # ... Forward into
        data = prepare_pipeline_input_from_atom_array(atom_array)
        data["example_id"] = example_id

        # ... Wrap up with additional features
        if "extra" not in spec_dict:
            spec_dict["extra"] = {}
        spec_dict["extra"]["example_id"] = example_id
        data["specification"] = spec_dict
        return data


# ============================================================================
# APIs and utils
# ============================================================================


def prepare_pipeline_input_from_atom_array(  # see atomworks.ml.datasets.parsers.base.load_example_from_metadata_row
    atom_array_orig,
) -> dict:
    """
    Load or create an example from a metadata dictionary.
    If the file path is not provided in the metadata dictionary, create a spoofed CIF file based on the length.
    Args:
        atom_array_orig: Atom array instantiated with conditioning annotations

    Returns:
        dict: A dictionary containing the parsed row data and additional loaded CIF data.
    """
    _start_parse_time = time.time()
    # HACK: Set empty bond graph:
    if atom_array_orig.bonds is None:
        atom_array_orig.bonds = BondList(atom_array_orig.array_length())

    # Temporary spoof of chain IDs to ensure duplicates aren't dropped:
    result_dict = parse_atom_array(
        atom_array_orig,
        remove_ccds=[],
        fix_arginines=False,
        add_missing_atoms=False,
        extra_fields=INFERENCE_ANNOTATIONS,
        build_assembly=None,
        hydrogen_policy="remove",
    )
    atom_array = result_dict["asym_unit"][0]

    # HACK: Set iid information manually
    # We currently do not preserve this information from the input,
    # if you want these we'd need to remove the spoofing here
    check_has_required_conditioning_annotations(
        atom_array, required=REQUIRED_INFERENCE_ANNOTATIONS
    )
    atom_array = convert_existing_annotations_to_bool(atom_array)
    atom_array.set_annotation("chain_iid", [f"{c}_1" for c in atom_array.chain_id])
    atom_array.set_annotation("pn_unit_iid", [f"{c}_1" for c in atom_array.pn_unit_id])

    # Ensure motif annotations are removed
    atom_array.del_annotation(
        "is_motif_token"
    ) if "is_motif_token" in atom_array.get_annotation_categories() else None
    atom_array.del_annotation(
        "is_motif_atom"
    ) if "is_motif_atom" in atom_array.get_annotation_categories() else None

    data = {
        "atom_array": atom_array,  # First model
        "chain_info": result_dict["chain_info"],
        "ligand_info": result_dict["ligand_info"],
        "metadata": result_dict["metadata"],
    }
    _stop_parse_time = time.time()
    data = TransformedDict(data)
    return data


def create_atom_array_from_design_specification(
    **spec_kwargs,
) -> tuple[AtomArray, dict]:
    if int(spec_kwargs.get("dialect", 2)) < 2:
        warn = (
            "Using dialect==1, which is deprecated and will be removed in future releases. "
            "Please update your input specification to dialect=2 and use the new schema if possible"
        )
        warnings.warn(warn, DeprecationWarning)
        logger.warning(warn)
        atom_array = create_atom_array_from_design_specification_legacy(**spec_kwargs)
        return atom_array, {}

    # Create input specfication and build
    spec = DesignInputSpecification(**spec_kwargs)
    atom_array, metadata = spec.build(return_metadata=True)
    return atom_array, metadata


@contextmanager
def validator_context(validator_name: str, data: dict = None):
    """Context manager for validator execution with logging."""
    logger.debug(f"Starting validator: {validator_name}")
    try:
        yield
        logger.debug(f"✓ Completed validator: {validator_name}")
    except Exception as e:
        logger.error(
            f"✗ Failed in validator: {validator_name}\n"
            f"  Error: {str(e)}\n"
            f"  Error type: {type(e).__name__}"
        )
        raise e


def _next_chain_id(used_chains: set) -> str:
    """Return the first single- then double-letter chain ID not in used_chains."""
    import string
    for c in string.ascii_uppercase:
        if c not in used_chains:
            return c
    for c1 in string.ascii_uppercase:
        for c2 in string.ascii_uppercase:
            cc = c1 + c2
            if cc not in used_chains:
                return cc
    raise RuntimeError("Exhausted all chain IDs.")


def create_diffused_residues(n, additional_annotations=None):
    if n <= 0:
        raise ValueError(f"Negative/null residue count ({n}) not allowed.")

    atoms = []
    [
        atoms.extend(
            [
                struc.Atom(
                    np.array([0.0, 0.0, 0.0], dtype=np.float32),
                    res_name="ALA",
                    res_id=idx,
                )
                for _ in range(5)
            ]
        )
        for idx in range(1, n + 1)
    ]
    array = struc.array(atoms)
    array.set_annotation(
        "element", np.array(["N", "C", "C", "O", "C"] * n, dtype="<U2")
    )
    array.set_annotation(
        "atom_name", np.array(["N", "CA", "C", "O", "CB"] * n, dtype="<U2")
    )
    array = set_default_conditioning_annotations(
        array, motif=False, additional=additional_annotations
    )
    array = set_common_annotations(array)
    return array


def create_motif_residue(
    token,
    strip_sidechains_by_default: bool,
):
    extra_annotations = {}
    if UNINDEXED_FLOATING_MOTIF_ANNOTATION in token.get_annotation_categories():
        extra_annotations[UNINDEXED_FLOATING_MOTIF_ANNOTATION] = token.get_annotation(
            UNINDEXED_FLOATING_MOTIF_ANNOTATION
        ).copy()

    if strip_sidechains_by_default and token.res_name in STANDARD_AA:
        n_atoms = token.shape[0]
        diffuse_oxygen = False
        if n_atoms < 3:
            raise ValueError(
                f"Not enough data for {src_chain}{src_resid} in input atom array."
            )
        if n_atoms == 3:
            # Handle cases with N, CA, C only;
            token = token + create_o_atoms(token.copy())
            diffuse_oxygen = True  # flag oxygen for generation

        # Subset to the first 4 atoms (N, CA, C, O) only
        token = token[np.isin(token.atom_name, ["N", "CA", "C", "O"])]

        # exactly N, CA, C, O but no CB. Place CB onto idealized position and conver to ALA
        # Sequence name ALA ensures the padded atoms to be diffused from the fixed backbone
        # are placed on the CB so as to not leak the identity of the residue.
        token = token + create_cb_atoms(token.copy())

        # Sequence name must be set to ALA such that the central atom is correctly CB
        token.res_name = np.full_like(token.res_name, "ALA", dtype=token.res_name.dtype)
        token.set_annotation(
            "is_motif_atom_with_fixed_coord",
            np.where(
                np.arange(token.shape[0], dtype=int) < (4 - int(diffuse_oxygen)),
                token.is_motif_atom_with_fixed_coord,
                0,
            ),
        )

    check_has_required_conditioning_annotations(token)
    token = set_common_annotations(token)
    for annot, values in extra_annotations.items():
        if len(values) == token.shape[0]:
            token.set_annotation(annot, values)
    token.set_annotation("res_id", np.full(token.shape[0], 1))  # Reset to 1

    return token


def _polymer_link_atoms_for_residue(res_name: str) -> set[frozenset[str]]:
    """
    Return the atom-name pairs that represent canonical polymerization atoms for a residue.

    Only standard AA/DNA/RNA residues are treated as having canonical backbone links; PTMs
    and other chem comp types fall back to nonstandard handling.
    """
    if res_name in STANDARD_AA:
        return {frozenset({"C", "N"})}
    if res_name in STANDARD_DNA or res_name in STANDARD_RNA:
        return {frozenset({"O3'", "P"}), frozenset({"O3*", "P"})}  # allow legacy O3*
    return set()


def _is_standard_polymer_backbone_bond(atom_a: struc.Atom, atom_b: struc.Atom) -> bool:
    if atom_a.chain_id != atom_b.chain_id:
        return False
    if abs(atom_a.res_id - atom_b.res_id) != 1:
        return False
    atom_pair = frozenset({atom_a.atom_name, atom_b.atom_name})

    pairs_a = _polymer_link_atoms_for_residue(atom_a.res_name)
    pairs_b = _polymer_link_atoms_for_residue(atom_b.res_name)
    shared_pairs = pairs_a & pairs_b
    return atom_pair in shared_pairs


def _is_polymer_backbone_like(atom_a: struc.Atom, atom_b: struc.Atom) -> bool:
    """
    Broader backbone check that treats canonical polymer atoms (C/N for peptide,
    O3'/O3*–P for nucleic) as backbone links even when the residue itself is
    non-standard (e.g., PTR/SEP).
    """
    if atom_a.chain_id != atom_b.chain_id:
        return False
    if abs(atom_a.res_id - atom_b.res_id) != 1:
        return False
    atom_pair = frozenset({atom_a.atom_name, atom_b.atom_name})
    return atom_pair in {
        frozenset({"C", "N"}),
        frozenset({"O3'", "P"}),
        frozenset({"O3*", "P"}),
    }


def _restore_component_bonds(
    atom_array_accum: struc.AtomArray,
    src_atom_array: Optional[struc.AtomArray],
    source_to_accum_idx: Dict[int, int],
    source_idx_to_component: Dict[int, str],
    unindexed_components: set[str],
) -> struc.AtomArray:
    """
    Rehydrate bonds from the input structure onto the accumulated array.

    - Replays bonds from `src_atom_array` using the provided source→accum mappings.
    - Skips canonical polymer backbone bonds (peptide/nucleic) that are reconstructed elsewhere.
    - Protects unindexed components by disallowing cross-residue bonds that involve
      an unindexed residue (except standard backbone).
    - Emits warnings when a source bond cannot be remapped because one endpoint was
      dropped during accumulation.
    """
    if atom_array_accum.bonds is None:
        atom_array_accum.bonds = struc.BondList(atom_array_accum.array_length())

    if (
        src_atom_array is None
        or not hasattr(src_atom_array, "bonds")
        or src_atom_array.bonds is None
        or not source_to_accum_idx
    ):
        return atom_array_accum

    bonds_to_add: List[List[int]] = []
    seen_pairs: set[tuple[int, int]] = set()
    src_bonds = np.asarray(src_atom_array.bonds.as_array(), dtype=np.int64)

    def _is_unindexed_source(idx: int) -> bool:
        component = source_idx_to_component.get(idx)
        return component in unindexed_components if component is not None else False

    def _fmt_atom(atom: struc.Atom) -> str:
        # Use a readable residue/atom separator to avoid names running together
        return f"{atom.chain_id}{atom.res_id}:{atom.res_name}_{atom.atom_name}"

    for atom_i_idx, atom_j_idx, bond_type in src_bonds:
        atom_i_idx = int(atom_i_idx)
        atom_j_idx = int(atom_j_idx)
        bond_type = int(bond_type)
        mapped_i = source_to_accum_idx.get(atom_i_idx)
        mapped_j = source_to_accum_idx.get(atom_j_idx)

        atom_i = src_atom_array[atom_i_idx]
        atom_j = src_atom_array[atom_j_idx]

        # If we only have one side of the bond, assert if the mapped atom is from an
        # unindexed component and the bond would connect across residues.
        if mapped_i is None or mapped_j is None:
            if _is_standard_polymer_backbone_bond(
                atom_i, atom_j
            ) or _is_polymer_backbone_like(atom_i, atom_j):
                continue
            if mapped_i is not None and _is_unindexed_source(atom_i_idx):
                if (
                    atom_i.chain_id != atom_j.chain_id
                    or atom_i.res_id != atom_j.res_id
                    or not (
                        _is_standard_polymer_backbone_bond(atom_i, atom_j)
                        or _is_polymer_backbone_like(atom_i, atom_j)
                    )
                ):
                    raise AssertionError(
                        f"Unsupported bond between unindexed component {atom_i.chain_id}{atom_i.res_id} "
                        f"and omitted residue {atom_j.chain_id}{atom_j.res_id}."
                    )
            if mapped_j is not None and _is_unindexed_source(atom_j_idx):
                if (
                    atom_i.chain_id != atom_j.chain_id
                    or atom_i.res_id != atom_j.res_id
                    or not (
                        _is_standard_polymer_backbone_bond(atom_i, atom_j)
                        or _is_polymer_backbone_like(atom_i, atom_j)
                    )
                ):
                    raise AssertionError(
                        f"Unsupported bond between unindexed component {atom_j.chain_id}{atom_j.res_id} "
                        f"and omitted residue {atom_i.chain_id}{atom_i.res_id}."
                    )
            # Only warn when we retained one side of a cross-residue/chain linkage
            # (e.g., glycan partner missing), not for missing intra-residue atoms.
            if (mapped_i is not None or mapped_j is not None) and (
                atom_i.chain_id != atom_j.chain_id or atom_i.res_id != atom_j.res_id
            ):
                logger.warning(
                    (
                        "Skipping non-backbone bond from source structure between %s and %s (type %d): "
                        "one atom is not present in accumulated components. "
                        "Bond cannot be inferred automatically; set it manually if needed."
                    )
                    % (_fmt_atom(atom_i), _fmt_atom(atom_j), bond_type)
                )
            continue

        # Do not connect unindexed residues to anything else for now.
        comp_i = source_idx_to_component.get(atom_i_idx)
        comp_j = source_idx_to_component.get(atom_j_idx)
        if (comp_i in unindexed_components or comp_j in unindexed_components) and (
            atom_i.chain_id != atom_j.chain_id or atom_i.res_id != atom_j.res_id
        ):
            if not (
                _is_standard_polymer_backbone_bond(atom_i, atom_j)
                or _is_polymer_backbone_like(atom_i, atom_j)
            ):
                raise AssertionError(
                    "Bonds involving unindexed residues are not yet supported."
                )
            continue

        if _is_standard_polymer_backbone_bond(
            atom_i, atom_j
        ) or _is_polymer_backbone_like(atom_i, atom_j):
            continue

        pair = (min(mapped_i, mapped_j), max(mapped_i, mapped_j))
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        bonds_to_add.append([mapped_i, mapped_j, bond_type])

    bond_array = (
        np.array(bonds_to_add, dtype=np.int64)
        if bonds_to_add
        else np.empty((0, 3), dtype=np.int64)
    )
    new_bonds = struc.BondList(atom_array_accum.array_length(), bond_array)
    atom_array_accum.bonds = atom_array_accum.bonds.merge(new_bonds)
    return atom_array_accum


def _add_backbone_bonds_for_nonstandard_residues(
    atom_array_accum: struc.AtomArray,
) -> struc.AtomArray:
    """
    Add backbone/polymer bonds for cases where at least one residue is non-standard.

    Uses `atomworks.io.utils.bonds.get_inferred_polymer_bonds`, which consults CCD
    chem-comp metadata to decide the correct polymerization atoms (C/N, CG/N, etc.).
    Only bonds involving at least one non-standard residue are added; standard
    AA/DNA/RNA pairs are assumed to already carry their backbone bonds.
    """
    if atom_array_accum.bonds is None:
        atom_array_accum.bonds = struc.BondList(atom_array_accum.array_length())

    unindexed_mask = (
        atom_array_accum.get_annotation("is_motif_atom_unindexed")
        if "is_motif_atom_unindexed" in atom_array_accum.get_annotation_categories()
        else np.zeros(atom_array_accum.array_length(), dtype=bool)
    )

    existing_pairs = {
        (min(a, b), max(a, b)) for a, b, _ in atom_array_accum.bonds.as_array()
    }
    bonds_to_add: List[List[int]] = []

    inferred_bonds, _ = get_inferred_polymer_bonds(atom_array_accum)
    for atom_i_idx, atom_j_idx, bond_type in inferred_bonds:
        atom_i_idx = int(atom_i_idx)
        atom_j_idx = int(atom_j_idx)

        # Do not connect unindexed residues across residue boundaries
        if (unindexed_mask[atom_i_idx] or unindexed_mask[atom_j_idx]) and (
            atom_array_accum.chain_id[atom_i_idx]
            != atom_array_accum.chain_id[atom_j_idx]
            or atom_array_accum.res_id[atom_i_idx]
            != atom_array_accum.res_id[atom_j_idx]
        ):
            continue

        # Only synthesize bonds when at least one residue is non-standard; standard
        # backbone bonds should already exist.
        if _is_standard_polymer_backbone_bond(
            atom_array_accum[atom_i_idx], atom_array_accum[atom_j_idx]
        ):
            continue

        pair = (min(atom_i_idx, atom_j_idx), max(atom_i_idx, atom_j_idx))
        if pair in existing_pairs:
            continue
        existing_pairs.add(pair)
        bonds_to_add.append([pair[0], pair[1], int(bond_type)])

    if bonds_to_add:
        new_bonds = struc.BondList(
            atom_array_accum.array_length(), np.array(bonds_to_add, dtype=np.int64)
        )
        atom_array_accum.bonds = atom_array_accum.bonds.merge(new_bonds)
    return atom_array_accum


def _sort_bonds(atom_array_accum: struc.AtomArray) -> struc.AtomArray:
    """Sort bonds deterministically by atom indices then bond type."""
    bonds_arr = atom_array_accum.bonds.as_array().copy()
    # ensure lower index first
    swap_mask = bonds_arr[:, 0] > bonds_arr[:, 1]
    bonds_arr[swap_mask, :2] = bonds_arr[swap_mask][:, [1, 0]]
    order = np.lexsort((bonds_arr[:, 2], bonds_arr[:, 1], bonds_arr[:, 0]))
    bonds_arr = bonds_arr[order]
    atom_array_accum.bonds = struc.BondList(
        atom_array_accum.array_length(), bonds_arr.astype(np.int64)
    )
    return atom_array_accum


def accumulate_components(
    components_to_accumulate: List[Union[str, int]],
    *,
    # Tokens from input
    indexed_tokens: Dict[str, AtomArray],
    unindexed_tokens: Dict[str, AtomArray],
    # Additional parameters
    atom_array_accum=[],
    start_chain: str = "A",
    start_resid: int = 1,
    unindexed_breaks: Optional[List[bool]] = [],
    src_atom_array: Optional[AtomArray] = None,
    strip_sidechains_by_default: bool = False,
    **kwargs,
) -> AtomArray:
    # ... Create list of components
    assert (
        x := (set(list(indexed_tokens.keys()) + list(unindexed_tokens.keys())))
    ).issubset(
        (y := set(components_to_accumulate))
    ), "Unindexed and indexed set {} is not subset of components to accumulate {}".format(
        x, y
    )
    all_tokens = indexed_tokens | unindexed_tokens
    all_annots = []
    [
        all_annots.extend(list(tok.get_annotation_categories()))
        for tok in all_tokens.values()
    ]
    all_annots = set(all_annots)
    atom_array_accum = [] if atom_array_accum is None else atom_array_accum
    unindexed_breaks = (
        [None] * len(components_to_accumulate)
        if unindexed_breaks is None
        else unindexed_breaks
    )

    # ... For-loop accum variables
    unindexed_components_started = (
        False  # once one unindexed component is added, stop adding diffused residues
    )
    chain = start_chain
    res_id = start_resid
    molecule_id = 0
    source_to_accum_idx: Dict[int, int] = {}
    source_idx_to_component: Dict[int, str] = {}
    unindexed_component_names = set(unindexed_tokens.keys())
    current_accum_idx = sum(len(arr) for arr in atom_array_accum)

    # ... Insert contig information one- by one-
    assert len(components_to_accumulate) == len(
        unindexed_breaks
    ), "Mismatch in number of components to accumulate and breaks"
    for component, is_break in zip(components_to_accumulate, unindexed_breaks):
        src_indices = None
        if exists(is_break) and is_break:
            if not unindexed_components_started:
                chain = start_chain
                res_id = start_resid
                unindexed_components_started = True

        if component == "/0":
            # Reset iterators on next chain
            chain = chr(ord(chain) + 1)
            molecule_id += 1
            res_id = 1
            continue

        # ... Create array to insert
        if str(component)[0].isalpha():  # motif (e.g. "A22")
            n = 1

            # ... Fetch the motif residue
            token = all_tokens[component]
            if src_atom_array is not None:
                src_mask = fetch_mask_from_idx(component, atom_array=src_atom_array)
                src_indices = np.where(src_mask)[0]
                # try:
                # except ComponentValidationError as e:
                #     src_indices = None
                #     print(e)

            # ... Ensure motif residues are set properly
            token = create_motif_residue(
                token, strip_sidechains_by_default=strip_sidechains_by_default
            )

            # ... Insert breakpoint when break clause is met
            if exists(is_break) and is_break:
                token.set_annotation(
                    "is_motif_atom_unindexed_motif_breakpoint",
                    np.ones(token.shape[0], dtype=int),
                )
            else:
                token.set_annotation(
                    "is_motif_atom_unindexed_motif_breakpoint",
                    np.zeros(token.shape[0], dtype=int),
                )
        else:
            ## foundry components update sends P for protein tokens
            n = int(component[:-1])
            # ... Skip if none or unindexed
            if n == 0 or unindexed_components_started:
                res_id += n
                continue

            # ... Create diffused residues
            token = create_diffused_residues(n, all_annots)

        # ... Set index of insertion
        token = set_indices(
            array=token,
            chain=chain,
            res_id_start=res_id,
            molecule_id=molecule_id,
            component=component,
        )

        assert (
            len(get_token_starts(token)) == n
        ), f"Mismatch in number of residues: expected {n}, got {len(get_token_starts(token))} in \n{token}"

        if (
            src_atom_array is not None
            and str(component)[0].isalpha()
            and src_indices is not None
            and len(src_indices) == len(token)
        ):
            for i, src_idx in enumerate(src_indices):
                source_to_accum_idx[int(src_idx)] = current_accum_idx + i
                source_idx_to_component[int(src_idx)] = str(component)

        # ... Insert & Increment residue ID
        atom_array_accum.append(token)
        res_id += n
        current_accum_idx += len(token)

    # ... Concatenate all components
    atom_array_accum = struc.concatenate(atom_array_accum)
    atom_array_accum.set_annotation("pn_unit_iid", atom_array_accum.chain_id)
    should_restore_bonds = (
        src_atom_array is not None
        and bool(source_to_accum_idx)
        and _check_has_backbone_connections_to_nonstandard_residues(
            atom_array_accum, src_atom_array
        )
    )
    if should_restore_bonds:
        atom_array_accum = _restore_component_bonds(
            atom_array_accum=atom_array_accum,
            src_atom_array=src_atom_array,
            source_to_accum_idx=source_to_accum_idx,
            source_idx_to_component=source_idx_to_component,
            unindexed_components=unindexed_component_names,
        )
        atom_array_accum = _add_backbone_bonds_for_nonstandard_residues(
            atom_array_accum=atom_array_accum
        )
        atom_array_accum = _sort_bonds(atom_array_accum)

    # Reset res_id for unindexed residues to avoid duplicates (ridiculously long lines of code, cleanup later)
    if np.any(atom_array_accum.is_motif_atom_unindexed.astype(bool)) and not np.all(
        atom_array_accum.is_motif_atom_unindexed.astype(bool)
    ):
        max_id = np.max(
            atom_array_accum[
                ~atom_array_accum.is_motif_atom_unindexed.astype(bool)
            ].res_id
        )
        min_id_udx = np.min(
            atom_array_accum[
                atom_array_accum.is_motif_atom_unindexed.astype(bool)
            ].res_id
        )
        atom_array_accum.res_id[
            atom_array_accum.is_motif_atom_unindexed.astype(bool)
        ] += max_id - min_id_udx + 1

    # ... Bonds
    if atom_array_accum.bonds is None:
        atom_array_accum.bonds = BondList(atom_array_accum.array_length())
    return atom_array_accum


def ensure_input_is_abspath(args: Dict[str, Any], path: PathLike | None):
    """
    Ensures the input source is an absolute path if exists, if not it will convert

    args:
        args: Inference specification for atom array
        path: None or file to which the input is relative to.
    """
    if isinstance(args, str):
        raise ValueError(
            "Expected args to be a dictionary, got a string: {}. If you are using an input JSON ensure it contains dictionaries of arguments".format(
                args
            )
        )
    if "input" not in args or not exists(args["input"]):
        return args
    input = str(args["input"])
    if not os.path.isabs(input):
        if path is None:
            raise ValueError(
                "Input path is relative, but no base path was provided to resolve it against."
            )
        input = os.path.abspath(os.path.join(os.path.dirname(str(path)), input))
        logger.info(
            f"Input source path is relative, converted to absolute path: {input}"
        )
        args["input"] = input
    return args

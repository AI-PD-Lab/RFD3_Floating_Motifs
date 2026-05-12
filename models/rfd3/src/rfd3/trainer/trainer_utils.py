from collections import Counter, OrderedDict
import re

import numpy as np
import torch
from atomworks.ml.encoding_definitions import AF3SequenceEncoding
from atomworks.ml.utils.token import (
    get_token_starts,
    spread_token_wise,
)
from biotite.structure import concatenate, infer_elements
from jaxtyping import Float, Int
from rfd3.constants import (
    ATOM14_ATOM_NAMES,
    VIRTUAL_ATOM_ELEMENT_NAME,
    association_schemes,
    association_schemes_stripped,
)
from rfd3.utils.io import (
    build_stack_from_atom_array_and_batched_coords,
)
from scipy.optimize import linear_sum_assignment

from foundry.common import exists
from foundry.utils.ddp import RankedLogger

global_logger = RankedLogger(__name__, rank_zero_only=False)
UNINDEXED_FLOATING_MOTIF_ANNOTATION = "is_motif_atom_unindexed_floating_motif"

#######################################################################
# Pythonic Helper functions
#######################################################################


def _remap_outputs(
    xyz: Float[torch.Tensor, "D L 3"], mapping: Int[torch.Tensor, "D L"]
) -> Float[torch.Tensor, "D L 3"]:
    """Helper function to remap outputs using a mapping tensor."""
    for i in range(xyz.shape[0]):
        xyz[i, mapping[i]] = xyz[i].clone()
    return xyz


def _reorder_dict(d: dict) -> OrderedDict:
    """
    Reorders keys in the dictionary to ensure 'metrics' and 'specification' are last (in that order if both present).
    """
    ordered = OrderedDict()
    first_keys = ["task", "diffused_index_map"]
    last_keys = ["metrics", "specification", "inference_sampler"]
    # First
    for k in first_keys:
        if k in d:
            ordered[k] = d[k]
    # Middle
    for k in d:
        if k not in last_keys and k not in first_keys:
            ordered[k] = d[k]
    # Last
    for k in last_keys:
        if k in d:
            ordered[k] = d[k]
    return ordered


#######################################################################
# Biotite-related helper functions
#######################################################################


def _build_atom_array_stack(
    coords,
    src_atom_array,
    sequence_indices,
    sequence_logits,
    allow_sequence_outputs=True,
    read_sequence_from_sequence_head=True,
    association_scheme: str = "atom14",
):
    """
    Wraps around build_atom_array_and_batched_coords to also include additional modifications to atom array
    """
    atom_array_stack = build_stack_from_atom_array_and_batched_coords(
        coords, src_atom_array.copy()
    )

    # ... Spoof empty sequences to alanines
    atom_array_stack.res_name[
        atom_array_stack.is_protein & (atom_array_stack.res_name == "UNK")
    ] = "ALA"

    # ... Add sequence if available
    if allow_sequence_outputs:
        array_list = []
        if read_sequence_from_sequence_head and exists(sequence_logits):
            sequence_encoding = AF3SequenceEncoding()
            for i, (atom_array, seq_indices, seq_logits) in enumerate(
                zip(atom_array_stack, sequence_indices, sequence_logits)
            ):
                # Set residue names
                diffused_mask = ~atom_array.is_motif_atom_with_fixed_seq
                three_letter_sequence = sequence_encoding.decode(
                    seq_indices.cpu().numpy().astype(int)
                )  # [I]

                atom_array.res_name[diffused_mask] = three_letter_sequence[
                    atom_array.token_id
                ][diffused_mask]  # [L]

                # Set bfactor column as entropy of sequence logits
                p = torch.softmax(seq_logits, dim=-1).cpu().numpy()  # shape (L, 32)
                res_entropy = -np.sum(p * np.log(p + 1e-10), axis=-1)  # shape (L,)
                atom_array.b_factor = spread_token_wise(atom_array, res_entropy)
                array_list.append(atom_array.copy())
        else:
            # This automatically deletes virtual atoms and assigns resname, atom name, and elements
            for atom_array in atom_array_stack:
                atom_array = _readout_seq_from_struc(
                    atom_array, association_scheme=association_scheme
                )
                array_list.append(atom_array)

    # Return as list
    atom_array_stack = array_list

    return atom_array_stack


def _cleanup_virtual_atoms_and_assign_atom_name_elements(
    atom_array, association_scheme: str = "atom14"
):
    ## remove virtual atoms based on predicted residue and assign correct atom name and elements
    ret_mask = []
    atom_names = []
    # This is used to indicate which residue is unidentified, probably due to an invalid structure.
    # This is different from the ref_mask, which is used to delete virtual atoms, but this one is used to assign UNK resname for invalid residues.
    invalid_mask = []

    # ... Iterate through each residue.
    # Here we iterate through res_id instead of token_id to avoid some atomization cases or something else.
    res_ids = atom_array.res_id
    res_start_indices = np.concatenate(
        [[0], np.where(res_ids[1:] != res_ids[:-1])[0] + 1]
    )
    res_end_indices = np.concatenate([res_start_indices[1:], [len(res_ids)]])
    warning_issued = False
    for start, end in zip(res_start_indices, res_end_indices):
        res_array = atom_array[start:end]

        is_seq_known = all(
            np.array(res_array.is_motif_atom_with_fixed_seq, dtype=bool)
        ) or all(np.array(res_array.is_motif_atom_unindexed, dtype=bool))

        # ... If sequence is known for the original atom array, just skip
        if is_seq_known:
            ret_mask += [True] * len(res_array)
            invalid_mask += [False] * len(res_array)
            res_name = res_array[0].res_name
            atom_names += res_array.gt_atom_name.tolist()
            continue

        # ... If sequence is unknown for the original atom array, use the predicted / inferred sequence
        res_name = res_array[0].res_name
        if res_name not in association_schemes[association_scheme]:
            global_logger.warning(
                "Model predicted non-protein sequence for diffused residue. Cannot clean up outputs. Assigning unknown residue token."
            )
            warning_issued = True
            ret_mask += [True] * len(res_array)
            invalid_mask += [True] * len(res_array)
            atom_names += res_array.atom_name.tolist()
            continue

        scheme = association_schemes[association_scheme][res_name]
        ret_mask += [True if item is not None else False for item in scheme]
        atom_names += [item.strip() if item is not None else "VX" for item in scheme]
        invalid_mask += [False] * len(scheme)

    if len(atom_names) != atom_array.array_length():
        global_logger.warning(
            f"{atom_names=}\n{atom_array.atom_name=}\nAtom names length {len(atom_names)} does not match original array length {atom_array.array_length()}."
            "\nCould not cleanup atom array!!!"
        )
        if not warning_issued:
            raise ValueError("Atom names length does not match original array length. ")
        return atom_array
    atom_array.atom_name = atom_names
    atom_array.element = np.where(
        atom_array.element == VIRTUAL_ATOM_ELEMENT_NAME,
        infer_elements(atom_names),
        atom_array.element,
    )
    atom_array.res_name[invalid_mask] = np.array(["UNK"] * sum(invalid_mask))
    return atom_array[ret_mask]


def _readout_seq_from_struc(
    atom_array, central_atom="CB", threshold=0.5, association_scheme: str = "atom14"
):
    cur_atom_array_list = []

    # Iterate through each residue
    res_ids = atom_array.res_id
    res_start_indices = np.concatenate(
        [[0], np.where(res_ids[1:] != res_ids[:-1])[0] + 1]
    )
    res_end_indices = np.concatenate([res_start_indices[1:], [len(res_ids)]])

    for start, end in zip(res_start_indices, res_end_indices):
        # ... Check if the current residue is after padding (seq unknown):
        cur_res_atom_array = atom_array[start:end]
        is_seq_known = all(
            np.array(cur_res_atom_array.is_motif_atom_with_fixed_seq, dtype=bool)
        )

        # Here it assumes that every non-protein part has its sequence shown (not padded)
        if not is_seq_known:
            # For Glycine: it doesn't have CB, so set the virtual atom as CA.
            # The current way to handle this is to check if predicted CA and CB are too close, because in the case of glycine and we pad virtual atoms based on CB, CB's coords are set as CA.
            # There might be a better way to do this.
            CA_coord = cur_res_atom_array.coord[cur_res_atom_array.atom_name == "CA"]
            CB_coord = cur_res_atom_array.coord[cur_res_atom_array.atom_name == "CB"]
            if np.linalg.norm(CA_coord - CB_coord) < threshold:
                cur_central_atom = "CA"
            else:
                cur_central_atom = central_atom

            central_mask = cur_res_atom_array.atom_name == cur_central_atom

            # ... Calculate the distance to the central atom
            central_coord = cur_res_atom_array.coord[central_mask][
                0
            ]  # Should only have one central atom anyway
            dists = np.linalg.norm(cur_res_atom_array.coord - central_coord, axis=-1)

            # ... Select virtual atom by the distance. Shouldn't count the central atom itself.
            is_virtual = (dists < threshold) & ~central_mask

            # ... Throw away virtual atoms
            cur_res_atom_array_wo_virtual = cur_res_atom_array[~is_virtual]
            cur_pred_res_atom_names = (
                cur_res_atom_array_wo_virtual.atom_name
            )  # e.g. [N, CA, C, O, CB, V6, V2]

            # ... Iterate over the possible restypes and find the matched one if there is any
            has_restype_assigned = False
            for restype, atom_names in association_schemes_stripped[
                association_scheme
            ].items():
                atom_names = np.array(atom_names)

                # Shouldn't match these two
                if restype in ["UNK", "MSK"]:
                    continue

                # ... Find the index of virtual atom names in the standard atom14 names
                atom_name_idx_in_atom14_scheme = np.array(
                    [
                        np.where(ATOM14_ATOM_NAMES == atom_name)[0][0]
                        for atom_name in cur_pred_res_atom_names
                    ]
                )  # five backbone atoms + some virtual atoms, returning e.g. [0, 1, 2, 3, 4, 11, 7]
                atom14_scheme_mask = np.zeros_like(ATOM14_ATOM_NAMES, dtype=bool)
                atom14_scheme_mask[atom_name_idx_in_atom14_scheme] = True

                # ... Find the matched restype by checking if all the non-None posititons and None positions match
                # This is designed to keep virtual atoms and doesn't assign the atom names for now, which will be handled later.
                if all(x is not None for x in atom_names[atom14_scheme_mask]) and all(
                    x is None for x in atom_names[~atom14_scheme_mask]
                ):
                    cur_res_atom_array.res_name = np.array(
                        [restype] * len(cur_res_atom_array)
                    )
                    cur_atom_array_list.append(cur_res_atom_array)
                    has_restype_assigned = True
                    break
        else:
            cur_atom_array_list.append(cur_res_atom_array)
            has_restype_assigned = True

        # ... Give UNK as the residue name if the mapping fails (unrealistic sidechain)
        if not has_restype_assigned:
            cur_res_atom_array.res_name = np.array(["UNK"] * len(cur_res_atom_array))
            cur_atom_array_list.append(cur_res_atom_array)

    cur_atom_array = concatenate(cur_atom_array_list)

    return cur_atom_array


#######################################################################
# Unindexed output parsing
#######################################################################


def _reassign_unindexed_token_chains(atom_array):
    if np.any((mask := atom_array.is_motif_atom_unindexed)):
        # HACK: Since res_ids are the same, we should save them with a different chain index.
        atom_array.chain_id[mask] = "X"
        atom_array.res_id[mask] = atom_array.orig_res_id[mask]

        # Parse to separate chains
        starts = get_token_starts(atom_array)
        unindexed_starts = starts[mask[starts]]
        token_breaks = atom_array[
            unindexed_starts
        ].is_motif_atom_unindexed_motif_breakpoint
        token_group_id = np.cumsum(token_breaks, dtype=int)  # Group by motif breaks
        token_chain_id = np.array([f"X{i}" for i in token_group_id])

        chains = atom_array.chain_id[starts]
        chains[mask[starts]] = token_chain_id
        atom_array.chain_id = spread_token_wise(atom_array, chains)
    return atom_array


def process_unindexed_outputs(
    atom_array,
    match_atom_names=True,
    insert_guideposts=False,
    verbose=False,
):
    """
    Process design outputs containing unindexed tokens.
    Returns metadata such as the assigned positional indices from the input indices
    and the RMSD of the unindexed tokens.

    Returns:
        - Diffused atom array (without additional unindexed tokens)
        - Metadata:
            - diffused_indices: keys = original (contig) indices, values = diffused indices
            - insertion_rmsd: overall RMSD of insertion
            - insertion_rmsd_by_residue: RMSD of insertion for each token

        TODO: Add additional geometry metrics such as bond angle non-ideality, clashes etc.
        TODO: atom1d conditioning adherence - does the output contain HBonds in the right places, correct rasa values?
    """
    # ... Find assignments based on greedy search
    starts = get_token_starts(atom_array, add_exclusive_stop=True)

    # [N_diffused,]
    atom_array_diffused = atom_array[~atom_array.is_motif_atom_unindexed].copy()
    global_idx = np.arange(atom_array.array_length())[
        ~atom_array.is_motif_atom_unindexed
    ]

    metadata = {
        "diffused_index_map": {},
        "insertion_rmsd_by_token": {},
        "join_point_rmsd_by_token": {},
        "insertion_rmsd_by_restype": {},
    }
    token_maes = []
    token_rmcds = []
    n_conjoined_residues = 0

    # Initialize an empty array
    inserted_mask = np.full_like(atom_array_diffused.is_motif_atom_unindexed, False)
    processed_token_starts: set[int] = set()

    # First handle special unindexed motifs as contiguous segments mapped onto
    # contiguous scaffold windows. This preserves internal motif order for
    # larger floating motifs, rather than matching each residue independently.
    segment_specs = _collect_unindexed_floating_motif_segments(atom_array)
    for segment_spec in segment_specs:
        segment_tokens = []
        for start, end in segment_spec:
            token = atom_array[start:end]
            if not token.is_motif_atom_unindexed.all():
                continue
            segment_tokens.append(token)

        if not segment_tokens:
            continue

        window = _find_best_contiguous_window(
            segment_tokens,
            atom_array_diffused,
            inserted_mask,
        )
        if window is None:
            global_logger.warning(
                "Could not find contiguous scaffold window for unindexed floating motif "
                "segment starting at %s; falling back to per-token placement.",
                segment_tokens[0].src_component[0]
                if "src_component" in segment_tokens[0].get_annotation_categories()
                else "<unknown>",
            )
            continue

        for start, _end in segment_spec:
            processed_token_starts.add(start)

        _apply_contiguous_segment_cleanup(
            segment_tokens=segment_tokens,
            window=window,
            atom_array_diffused=atom_array_diffused,
            inserted_mask=inserted_mask,
            global_idx=global_idx,
            metadata=metadata,
            token_maes=token_maes,
            token_rmcds=token_rmcds,
        )

    for start, end in zip(starts[:-1], starts[1:]):
        if start in processed_token_starts:
            continue
        token = atom_array[start:end]
        if not token.is_motif_atom_unindexed.all():
            continue

        if "src_component" in token.get_annotation_categories():
            token_pdb_id = token.src_component[0]
        else:
            raise ValueError(
                "Missing annotation 'src_component' in token. Is this inference?"
            )

        if "src_sym_component" in token.get_annotation_categories():
            # if symmetry, token_pdb_id are updated to match the symmetrized component
            token_pdb_id = token.src_sym_component[0]

        res_name = token.res_name[0]

        # ... Calculate [N_unindex, N_diffused] distance matrix
        dists = np.linalg.norm(
            token.coord[:, None] - atom_array_diffused.coord[None, :], axis=-1
        )

        # ... Match atom indices based on atom names (mask out non-identical) and remove already inserted
        dists[:, inserted_mask.copy()] = np.inf
        if match_atom_names:
            matching_atom_name = (
                token.atom_name[:, None] == atom_array_diffused.atom_name[None, :]
            )
            dists[~matching_atom_name] = np.inf

        # ... Find the res_id's in the diffused regions belonging to the diffused indices
        row_ind, col_ind = linear_sum_assignment(dists)
        res_id, chain_id, is_conjoined = indices_to_components_(
            atom_array_diffused, col_ind
        )
        n_conjoined_residues += int(is_conjoined)

        # ... Recompute distance indices based on single residue pairings only
        token_match = (atom_array_diffused.res_id == res_id) & (
            atom_array_diffused.chain_id == chain_id
        )
        dists[:, ~token_match] = np.nan
        BIG = 1e12
        dists = np.nan_to_num(dists, nan=BIG, posinf=BIG, neginf=BIG)
        row_ind, col_ind = linear_sum_assignment(dists)
        res_id_, chain_id_, _ = indices_to_components_(atom_array_diffused, col_ind)

        assert (res_id_ == res_id) & (chain_id_ == chain_id)
        inserted_mask = np.logical_or(inserted_mask, token_match)

        # ... Compute metrics based on the new distances
        diff = token.coord[row_ind] - atom_array_diffused.coord[col_ind]
        token_rmsd = float(np.sqrt((diff**2).sum(-1).mean()))
        token_rmcd = float(np.cbrt((np.abs(diff) ** 3).sum(-1).mean()))
        token_mae = float((np.abs(diff)).sum(-1).mean())

        metadata["insertion_rmsd_by_token"][token_pdb_id] = token_rmsd
        token_maes.append(token_mae)
        token_rmcds.append(token_rmcd)

        if res_name not in metadata["insertion_rmsd_by_restype"]:
            metadata["insertion_rmsd_by_restype"][res_name] = []
        metadata["insertion_rmsd_by_restype"][res_name].append(token_rmsd)
        if not np.any(np.isin(token.atom_name, ["N", "CA", "C", "O"])):
            if np.sum(token.atomize) == 1:
                join_atom = np.where(token.atomize)[0][0]
            elif "CB" in token.atom_name:
                join_atom = np.where(token.atom_name == "CB")[0][0]
            else:
                join_atom = None

            if join_atom is None:
                global_logger.warning(
                    f"Token {token_pdb_id} does not contain backbone atoms or CB, skipping join point distance calculation {token}."
                )
            else:
                dist = float(dists[row_ind[join_atom], col_ind[join_atom]])
            metadata["join_point_rmsd_by_token"][token_pdb_id] = dist

        metadata["diffused_index_map"][token_pdb_id] = f"{chain_id}{res_id}"

        # ... Decide whether to cleanup guideposts or not
        if insert_guideposts:
            atom_array_diffused.coord[global_idx[col_ind]] = token.coord[row_ind]
            if token.is_motif_atom_with_fixed_seq[0]:
                atom_array_diffused.res_name[token_match] = token.res_name[0]
            # atom_array_diffused.is_motif_token[token_match] = True
            # atom_array_diffused.is_motif_atom[global_idx[col_ind]] = True
            atom_array_diffused.is_motif_atom_with_fixed_coord[global_idx[col_ind]] = (
                True
            )

    # ... Calculate global metrics
    def safe_mean(x):
        """Return nan-safe mean for empty or nan arrays."""
        x = np.asarray(x, float)
        if x.size == 0 or not np.isfinite(x).any():
            return float("nan")
        return float(np.nanmean(x))

    metadata["insertion.mae"] = safe_mean(token_maes)
    metadata["insertion.rmcd"] = safe_mean(token_rmcds)
    metadata["insertion_rmsd"] = safe_mean(
        list(metadata["insertion_rmsd_by_token"].values())
    )
    metadata["join_point_rmsd"] = safe_mean(
        list(metadata["join_point_rmsd_by_token"].values())
    )
    metadata["insertion_rmsd_by_restype"] = {
        a: safe_mean(v) for a, v in metadata["insertion_rmsd_by_restype"].items()
    }
    metadata["n_conjoined_residues"] = n_conjoined_residues

    if not verbose:
        metadata = {
            k: v for k, v in metadata.items() if not k.startswith("insertion_rmsd_by_")
        }

    return atom_array_diffused, metadata


def _collect_unindexed_floating_motif_segments(atom_array):
    if UNINDEXED_FLOATING_MOTIF_ANNOTATION not in atom_array.get_annotation_categories():
        return []

    starts = get_token_starts(atom_array, add_exclusive_stop=True)
    src = (
        np.asarray(atom_array.src_component).astype(str)
        if "src_component" in atom_array.get_annotation_categories()
        else None
    )
    is_special = atom_array.get_annotation(UNINDEXED_FLOATING_MOTIF_ANNOTATION).astype(
        bool
    )

    segments = []
    current = []
    prev_component = None

    for start, end in zip(starts[:-1], starts[1:]):
        token_special = bool(np.any(is_special[start:end]))
        if not token_special:
            if current:
                segments.append(current)
                current = []
                prev_component = None
            continue

        component = src[start] if src is not None else None
        if current and not _are_consecutive_components(prev_component, component):
            segments.append(current)
            current = [(start, end)]
        elif not current:
            current = [(start, end)]
        else:
            current.append((start, end))
        prev_component = component

    if current:
        segments.append(current)
    return segments


def _find_best_contiguous_window(segment_tokens, atom_array_diffused, inserted_mask):
    n = len(segment_tokens)
    residue_starts = get_token_starts(atom_array_diffused, add_exclusive_stop=True)
    if len(residue_starts) - 1 < n:
        return None
    best = None
    best_score = np.inf

    for i in range(len(residue_starts) - 1 - n + 1):
        window_pairs = [
            (residue_starts[j], residue_starts[j + 1]) for j in range(i, i + n)
        ]
        if not _window_is_contiguous(atom_array_diffused, window_pairs):
            continue
        if any(inserted_mask[start:end].any() for start, end in window_pairs):
            continue

        score = _score_contiguous_window(
            segment_tokens=segment_tokens,
            window_pairs=window_pairs,
            atom_array_diffused=atom_array_diffused,
        )
        if score < best_score:
            best_score = score
            best = window_pairs

    return best


def _apply_contiguous_segment_cleanup(
    *,
    segment_tokens,
    window,
    atom_array_diffused,
    inserted_mask,
    global_idx,
    metadata,
    token_maes,
    token_rmcds,
):
    for token, (start, end) in zip(segment_tokens, window):
        residue = atom_array_diffused[start:end]
        token_pdb_id = (
            token.src_sym_component[0]
            if "src_sym_component" in token.get_annotation_categories()
            else token.src_component[0]
        )
        res_name = token.res_name[0]

        dists = np.linalg.norm(
            token.coord[:, None] - residue.coord[None, :], axis=-1
        )
        matching_atom_name = token.atom_name[:, None] == residue.atom_name[None, :]
        dists[~matching_atom_name] = np.inf
        row_ind, col_ind = linear_sum_assignment(dists)

        diff = token.coord[row_ind] - residue.coord[col_ind]
        token_rmsd = float(np.sqrt((diff**2).sum(-1).mean()))
        token_rmcd = float(np.cbrt((np.abs(diff) ** 3).sum(-1).mean()))
        token_mae = float((np.abs(diff)).sum(-1).mean())

        metadata["insertion_rmsd_by_token"][token_pdb_id] = token_rmsd
        token_maes.append(token_mae)
        token_rmcds.append(token_rmcd)

        if res_name not in metadata["insertion_rmsd_by_restype"]:
            metadata["insertion_rmsd_by_restype"][res_name] = []
        metadata["insertion_rmsd_by_restype"][res_name].append(token_rmsd)
        metadata["diffused_index_map"][token_pdb_id] = (
            f"{atom_array_diffused.chain_id[start]}{atom_array_diffused.res_id[start]}"
        )

        residue_global = global_idx[start:end]
        atom_array_diffused.coord[residue_global[col_ind]] = token.coord[row_ind]
        if token.is_motif_atom_with_fixed_seq[0]:
            atom_array_diffused.res_name[start:end] = token.res_name[0]
        atom_array_diffused.is_motif_atom_with_fixed_coord[residue_global[col_ind]] = (
            True
        )
        inserted_mask[start:end] = True


def _window_is_contiguous(atom_array_diffused, window_pairs):
    prev_chain = None
    prev_resid = None
    for start, _end in window_pairs:
        chain = atom_array_diffused.chain_id[start]
        resid = int(atom_array_diffused.res_id[start])
        if prev_chain is not None and (chain != prev_chain or resid != prev_resid + 1):
            return False
        prev_chain = chain
        prev_resid = resid
    return True


def _score_contiguous_window(*, segment_tokens, window_pairs, atom_array_diffused):
    fit_terms = []
    for token, (start, end) in zip(segment_tokens, window_pairs):
        residue = atom_array_diffused[start:end]
        fit_terms.append(_token_residue_fit_score(token, residue))

    fit_score = float(np.mean(fit_terms)) if fit_terms else 0.0
    junction_score = _junction_penalty(segment_tokens, window_pairs, atom_array_diffused)
    clash_score = _clash_penalty(segment_tokens, window_pairs, atom_array_diffused)

    # Fit should dominate; junction quality and clashes break ties away from bad windows.
    return fit_score + 4.0 * junction_score + 10.0 * clash_score


def _token_residue_fit_score(token, residue):
    token_anchor, residue_anchor = _shared_anchor_coords(token, residue)
    if token_anchor is None or residue_anchor is None:
        token_rep = _get_token_rep_coord(token)
        residue_rep = _get_token_rep_coord(residue)
        return float(np.sum((token_rep - residue_rep) ** 2))
    diff = token_anchor - residue_anchor
    return float(np.mean(np.sum(diff**2, axis=-1)))


def _shared_anchor_coords(token, residue):
    anchor_priority = ("N", "CA", "C", "O", "CB")
    token_coords = []
    residue_coords = []
    for atom_name in anchor_priority:
        token_idx = np.where(token.atom_name == atom_name)[0]
        residue_idx = np.where(residue.atom_name == atom_name)[0]
        if token_idx.size == 0 or residue_idx.size == 0:
            continue
        token_coords.append(token.coord[token_idx[0]])
        residue_coords.append(residue.coord[residue_idx[0]])
    if not token_coords:
        return None, None
    return np.stack(token_coords, axis=0), np.stack(residue_coords, axis=0)


def _junction_penalty(segment_tokens, window_pairs, atom_array_diffused):
    penalty = 0.0
    first_token = segment_tokens[0]
    last_token = segment_tokens[-1]
    first_start, _first_end = window_pairs[0]
    _last_start, last_end = window_pairs[-1]

    prev_residue = _neighbor_residue(atom_array_diffused, first_start, direction=-1)
    next_residue = _neighbor_residue(atom_array_diffused, last_end - 1, direction=1)

    if prev_residue is not None:
        penalty += _pair_join_penalty(prev_residue, first_token)
    if next_residue is not None:
        penalty += _pair_join_penalty(last_token, next_residue)
    return penalty


def _neighbor_residue(atom_array, anchor_idx, direction):
    starts = get_token_starts(atom_array, add_exclusive_stop=True)
    token_idx = np.searchsorted(starts, anchor_idx, side="right") - 1
    neigh_idx = token_idx + direction
    if neigh_idx < 0 or neigh_idx >= len(starts) - 1:
        return None
    start, end = starts[neigh_idx], starts[neigh_idx + 1]
    return atom_array[start:end]


def _pair_join_penalty(left_residue, right_residue):
    left_ca = _atom_coord(left_residue, "CA")
    right_ca = _atom_coord(right_residue, "CA")
    if left_ca is None or right_ca is None:
        return 0.0
    ca_dist = float(np.linalg.norm(left_ca - right_ca))
    return (ca_dist - 3.8) ** 2


def _clash_penalty(segment_tokens, window_pairs, atom_array_diffused):
    window_mask = np.zeros(atom_array_diffused.array_length(), dtype=bool)
    starts = get_token_starts(atom_array_diffused, add_exclusive_stop=True)
    window_token_indices = []
    for start, end in window_pairs:
        window_mask[start:end] = True
        token_idx = np.searchsorted(starts, start, side="right") - 1
        window_token_indices.append(token_idx)

    excluded_token_indices = set(window_token_indices)
    if window_token_indices:
        excluded_token_indices.add(window_token_indices[0] - 1)
        excluded_token_indices.add(window_token_indices[-1] + 1)

    for idx in list(excluded_token_indices):
        if 0 <= idx < len(starts) - 1:
            start, end = starts[idx], starts[idx + 1]
            window_mask[start:end] = True

    scaffold_coords = atom_array_diffused.coord[~window_mask]
    if scaffold_coords.size == 0:
        return 0.0

    motif_coords = np.concatenate([tok.coord for tok in segment_tokens], axis=0)
    dists = np.linalg.norm(motif_coords[:, None] - scaffold_coords[None, :], axis=-1)
    if dists.size == 0:
        return 0.0
    threshold = 2.5
    overlap = np.clip(threshold - dists, a_min=0.0, a_max=None)
    return float(np.mean(overlap**2) / (threshold**2))


def _atom_coord(residue, atom_name):
    idx = np.where(residue.atom_name == atom_name)[0]
    if idx.size == 0:
        return None
    return residue.coord[idx[0]]


def _get_token_rep_coord(token):
    if np.any(token.atom_name == "CA"):
        return token.coord[np.where(token.atom_name == "CA")[0][0]]
    if "atomize" in token.get_annotation_categories() and np.any(token.atomize):
        return token.coord[np.where(token.atomize)[0][0]]
    return np.mean(token.coord, axis=0)


def _are_consecutive_components(previous, current):
    prev_parsed = _parse_component(previous)
    cur_parsed = _parse_component(current)
    if prev_parsed is None or cur_parsed is None:
        return previous == current
    prev_chain, prev_resid = prev_parsed
    cur_chain, cur_resid = cur_parsed
    return prev_chain == cur_chain and cur_resid == prev_resid + 1


def _parse_component(component):
    if component is None:
        return None
    match = re.match(r"^([^0-9-]+)(-?\d+)", str(component))
    if match is None:
        return None
    return match.group(1), int(match.group(2))


def indices_to_components_(atom_array, col_ind):
    """
    Fetch chain and resids in atom array given a set of raw indices
    will return 'conjoined' if indices to not map to a unique residue
    """
    res_ids, chain_ids = (
        atom_array.res_id[col_ind],
        atom_array.chain_id[col_ind],
    )
    if len(set(res_ids.tolist())) > 1 or len(set(chain_ids.tolist())) > 1:
        global_logger.warning(
            f"Unindexed token mapped its atoms to multiple diffused residues: {res_ids.tolist()} and chains {chain_ids.tolist()}."
        )
        # Handle by majority
        pair_counts = Counter(zip(chain_ids.tolist(), res_ids.tolist()))
        (chain_id, res_id), _ = pair_counts.most_common(1)[0]
        conjoined = True
    else:
        res_id = res_ids[0]
        chain_id = chain_ids[0]
        conjoined = False

    return res_id, chain_id, conjoined

"""Differentiable scalar potentials for RFD3 inference guidance.

Each potential returns a scalar tensor that should be MAXIMIZED by gradient ascent.
The guidance system backpropagates through this scalar to get per-atom coordinate
gradients, which are then used to update the sampler trajectory.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch


class BasePotential:
    """Abstract base for all potentials."""

    def __init__(self, weight: float = 1.0):
        self.weight = weight

    def compute(
        self,
        xyz: torch.Tensor,  # [D, L, 3]
        masks: dict[str, torch.Tensor],
        metadata: dict,
    ) -> torch.Tensor:
        raise NotImplementedError


class BinderROG(BasePotential):
    """Radius of gyration of binder atoms.

    Returns -weight * ROG.  Gradient ascent (maximising this) compacts the binder.
    Binder atoms are identified by masks["binder_atom_mask"] which is built from
    the generated (non-fixed) real atoms in the RFD3 feature dict.
    """

    def compute(self, xyz, masks, metadata):
        binder_mask = masks["binder_atom_mask"]  # [L]
        binder_xyz = xyz[:, binder_mask, :]  # [D, N_b, 3]
        if binder_xyz.shape[1] == 0:
            return xyz.new_zeros(())
        com = binder_xyz.mean(dim=1, keepdim=True)  # [D, 1, 3]
        sq_dist = ((binder_xyz - com) ** 2).sum(dim=-1)  # [D, N_b]
        rog = sq_dist.mean(dim=-1).sqrt()  # [D]
        return -self.weight * rog.mean()


class MonomerROG(BasePotential):
    """Radius of gyration over all atoms selected by potential_atom_mask.

    Returns -weight * ROG.
    """

    def compute(self, xyz, masks, metadata):
        pot_mask = masks["potential_atom_mask"]  # [L]
        pot_xyz = xyz[:, pot_mask, :]  # [D, N, 3]
        if pot_xyz.shape[1] == 0:
            return xyz.new_zeros(())
        com = pot_xyz.mean(dim=1, keepdim=True)
        sq_dist = ((pot_xyz - com) ** 2).sum(dim=-1)
        rog = sq_dist.mean(dim=-1).sqrt()
        return -self.weight * rog.mean()


class InterfaceNContacts(BasePotential):
    """Differentiable soft contact count between binder and target atoms.

    Returns +weight * sum(soft_contacts).

    Contact function (RFdiffusion1-style):
        x = (distance - d_0) / r_0
        contact = (1 - x^6) / (1 - x^12 + eps)

    Note: target atoms contribute to the potential but do not receive gradient
    updates (excluded by guide_atom_mask in the guidance computation).
    """

    def __init__(self, weight: float = 1.0, r_0: float = 8.0, d_0: float = 2.0):
        super().__init__(weight)
        self.r_0 = r_0
        self.d_0 = d_0

    def compute(self, xyz, masks, metadata):
        binder_mask = masks["binder_atom_mask"]
        target_mask = masks["target_atom_mask"]
        binder_xyz = xyz[:, binder_mask, :]  # [D, N_b, 3]
        target_xyz = xyz[:, target_mask, :]  # [D, N_t, 3]
        if binder_xyz.shape[1] == 0 or target_xyz.shape[1] == 0:
            return xyz.new_zeros(())
        dists = torch.cdist(binder_xyz, target_xyz)  # [D, N_b, N_t]
        contacts = _soft_contacts(dists, self.r_0, self.d_0)
        return self.weight * contacts.sum(dim=(-2, -1)).mean()


class MonomerContacts(BasePotential):
    """Internal differentiable contacts within the selected potential atoms.

    Returns +weight * sum(soft_contacts).
    Uses upper triangle to avoid double-counting.
    """

    def __init__(self, weight: float = 1.0, r_0: float = 8.0, d_0: float = 2.0):
        super().__init__(weight)
        self.r_0 = r_0
        self.d_0 = d_0

    def compute(self, xyz, masks, metadata):
        pot_mask = masks["potential_atom_mask"]
        pot_xyz = xyz[:, pot_mask, :]  # [D, N, 3]
        N = pot_xyz.shape[1]
        if N < 2:
            return xyz.new_zeros(())
        dists = torch.cdist(pot_xyz, pot_xyz)  # [D, N, N]
        contacts = _soft_contacts(dists, self.r_0, self.d_0)
        # Upper triangle only — diagonal is self-contact (x=−d_0/r_0, not physically meaningful)
        upper = torch.triu(
            torch.ones(N, N, dtype=torch.bool, device=xyz.device), diagonal=1
        )
        contacts = contacts[:, upper]  # [D, N*(N-1)/2]
        return self.weight * contacts.sum(dim=-1).mean()


class AtomPairDistance(BasePotential):
    """Harmonic restraint pulling two specific atoms toward target_distance.

    Returns -weight * (dist - target_distance)^2.
    atom_i and atom_j are indices into the full atom array (before masking).
    """

    def __init__(
        self,
        weight: float = 1.0,
        atom_i: int = 0,
        atom_j: int = 1,
        target_distance: float = 8.0,
    ):
        super().__init__(weight)
        self.atom_i = int(atom_i)
        self.atom_j = int(atom_j)
        self.target_distance = target_distance

    def compute(self, xyz, masks, metadata):
        pos_i = xyz[:, self.atom_i, :]  # [D, 3]
        pos_j = xyz[:, self.atom_j, :]  # [D, 3]
        dist = (pos_i - pos_j).norm(dim=-1)  # [D]
        return -self.weight * ((dist - self.target_distance) ** 2).mean()


class MotifDistance(BasePotential):
    """Harmonic restraint on the center distance between two motif blocks.

    The motif blocks are inferred from contiguous motif-token runs in contig
    order.  For a contig like ``A10-20,50,A80-90``, motif_i=0 selects A10-20
    and motif_j=1 selects A80-90.

    Motif identity is independent of coordinate fixation: atoms can be diffused
    and still selected here when they carry motif conditioning such as fixed
    sequence or unindexed motif flags.

    Returns -weight * (distance - target_distance)^2.

    This potential computes the center of mass of every motif block using all
    real motif atoms. During guidance, each atom in a selected motif receives
    the same block-level translation so the potential can move motifs without
    distorting their internal noisy geometry; a subsequent Kabsch projection can
    then restore the original rigid all-atom motif geometry.
    """

    def __init__(
        self,
        weight: float = 1.0,
        motif_i: int = 0,
        motif_j: int = 1,
        target_distance: float = 10.0,
    ):
        super().__init__(weight)
        self.motif_i = int(motif_i)
        self.motif_j = int(motif_j)
        self.target_distance = float(target_distance)

    def compute(self, xyz, masks, metadata):
        motif_blocks = _motif_distance_blocks(masks, metadata, xyz.device)
        max_idx = max(self.motif_i, self.motif_j)
        if self.motif_i < 0 or self.motif_j < 0 or max_idx >= len(motif_blocks):
            return xyz.new_zeros(())

        motif_a = xyz[:, motif_blocks[self.motif_i], :]
        motif_b = xyz[:, motif_blocks[self.motif_j], :]
        if motif_a.shape[1] == 0 or motif_b.shape[1] == 0:
            return xyz.new_zeros(())

        com_a = motif_a.mean(dim=1)
        com_b = motif_b.mean(dim=1)
        dist = (com_a - com_b).norm(dim=-1)
        return -self.weight * ((dist - self.target_distance) ** 2).mean()

    def guide_atom_mask(self, masks, metadata, device):
        motif_blocks = _motif_distance_blocks(masks, metadata, device)
        if not motif_blocks:
            any_mask = next(iter(masks.values()))
            return torch.zeros_like(any_mask, dtype=torch.bool, device=device)
        guide_mask = torch.zeros_like(motif_blocks[0], dtype=torch.bool, device=device)
        max_idx = max(self.motif_i, self.motif_j)
        if self.motif_i < 0 or self.motif_j < 0 or max_idx >= len(motif_blocks):
            return guide_mask
        guide_mask |= motif_blocks[self.motif_i]
        guide_mask |= motif_blocks[self.motif_j]
        return guide_mask

    def transform_atom_gradient(self, atom_grad, masks, metadata, xyz):
        motif_blocks = _motif_distance_blocks(masks, metadata, xyz.device)
        if not motif_blocks:
            return atom_grad

        transformed = torch.zeros_like(atom_grad)
        max_idx = max(self.motif_i, self.motif_j)
        if self.motif_i < 0 or self.motif_j < 0 or max_idx >= len(motif_blocks):
            return transformed

        for motif_idx in (self.motif_i, self.motif_j):
            block_mask = motif_blocks[motif_idx].to(device=xyz.device, dtype=torch.bool)
            if block_mask.sum().item() == 0:
                continue
            block_translation = atom_grad[:, block_mask, :].mean(dim=1, keepdim=True)
            transformed[:, block_mask, :] = block_translation
        return transformed


class MotifInternalRotation(BasePotential):
    """Bias one motif block toward a target rigid rotation from its input-PDB pose.

    Zero angles reproduce the input-PDB motif orientation. The target is defined
    around the motif COM, so this potential affects internal rigid rotation
    without prescribing motif translation. The guidance is rigidized to pure
    rotation, so all atoms in the motif move as one body.
    """

    def __init__(
        self,
        weight: float = 1.0,
        motif_i: int = 0,
        angle_x: float = 0.0,
        angle_y: float = 0.0,
        angle_z: float = 0.0,
    ):
        super().__init__(weight)
        self.motif_i = int(motif_i)
        self.angle_x = float(angle_x)
        self.angle_y = float(angle_y)
        self.angle_z = float(angle_z)

    def compute(self, xyz, masks, metadata):
        block_mask = _single_motif_block_mask(
            self.motif_i,
            masks=masks,
            metadata=metadata,
            device=xyz.device,
        )
        if block_mask is None:
            return xyz.new_zeros(())

        current_xyz, ref_xyz = _motif_block_current_and_reference_xyz(
            xyz, masks, metadata, block_mask
        )
        if current_xyz is None or ref_xyz is None or ref_xyz.shape[0] < 3:
            return xyz.new_zeros(())

        ref_centered = ref_xyz - ref_xyz.mean(dim=0, keepdim=True)
        current_centered = current_xyz - current_xyz.mean(dim=1, keepdim=True)
        target_rotation = _euler_rotation_matrix_deg(
            self.angle_x,
            self.angle_y,
            self.angle_z,
            device=xyz.device,
            dtype=xyz.dtype,
        )
        target_xyz = _apply_row_rotation(ref_centered, target_rotation)
        diff = current_centered - target_xyz.unsqueeze(0)
        return -self.weight * diff.pow(2).sum(dim=-1).mean()

    def guide_atom_mask(self, masks, metadata, device):
        block_mask = _single_motif_block_mask(
            self.motif_i,
            masks=masks,
            metadata=metadata,
            device=device,
        )
        if block_mask is None:
            any_mask = next(iter(masks.values()))
            return torch.zeros_like(any_mask, dtype=torch.bool, device=device)
        return block_mask

    def transform_atom_gradient(self, atom_grad, masks, metadata, xyz):
        block_mask = _single_motif_block_mask(
            self.motif_i,
            masks=masks,
            metadata=metadata,
            device=xyz.device,
        )
        if block_mask is None or block_mask.sum().item() < 3:
            return torch.zeros_like(atom_grad)

        transformed = torch.zeros_like(atom_grad)
        transformed[:, block_mask, :] = _project_gradient_to_rigid_body(
            atom_grad[:, block_mask, :],
            xyz[:, block_mask, :],
            allow_translation=False,
            allow_rotation=True,
        )
        return transformed


class MotifRelativePose(BasePotential):
    """Bias a motif-pair plane relative to the input-PDB origin.

    RFD3 recenters inference coordinates before initialization, so the origin is
    the input protein/motif COM in that centered coordinate system. This
    potential uses the origin plus the two motif COMs as a three-point angular
    pose. Zero angles reproduce the input-PDB motif COM rays from the origin;
    nonzero Euler angles rotate that reference triangle/plane around the origin.

    The loss compares normalized directions of the non-degenerate triangle edges
    (origin→motif_i, origin→motif_j, motif_i→motif_j), so it controls angular
    pose without directly caring about motif distance from the origin or
    inter-motif distance. If one motif COM sits exactly at the origin, the
    remaining nonzero edge directions still provide guidance. Guidance is
    rigidized to pure block translations.
    """

    def __init__(
        self,
        weight: float = 1.0,
        motif_i: int = 0,
        motif_j: int = 1,
        angle_x: float = 0.0,
        angle_y: float = 0.0,
        angle_z: float = 0.0,
    ):
        super().__init__(weight)
        self.motif_i = int(motif_i)
        self.motif_j = int(motif_j)
        self.angle_x = float(angle_x)
        self.angle_y = float(angle_y)
        self.angle_z = float(angle_z)
        self.skip_reason: str | None = None
        self.skip_detail: dict | None = None

    def compute(self, xyz, masks, metadata):
        self.skip_reason = None
        self.skip_detail = None
        motif_blocks = _motif_distance_blocks(masks, metadata, xyz.device)
        self.skip_detail = {
            "motif_i": self.motif_i,
            "motif_j": self.motif_j,
            "n_motif_blocks": len(motif_blocks),
            "has_floating_motif_reference_pos": "floating_motif_reference_pos"
            in metadata,
            "has_input_pos": "input_pos" in metadata,
            "has_motif_pos": "motif_pos" in metadata,
            "has_ref_pos": "ref_pos" in metadata,
        }
        block_i = (
            motif_blocks[self.motif_i]
            if 0 <= self.motif_i < len(motif_blocks)
            else None
        )
        block_j = (
            motif_blocks[self.motif_j]
            if 0 <= self.motif_j < len(motif_blocks)
            else None
        )
        if block_i is None or block_j is None:
            self.skip_reason = "motif_block_missing"
            return xyz.new_zeros(())

        cur_com_i = _motif_block_current_com(xyz, block_i)
        cur_com_j = _motif_block_current_com(xyz, block_j)
        ref_com_i = _motif_block_reference_com(xyz, masks, metadata, block_i)
        ref_com_j = _motif_block_reference_com(xyz, masks, metadata, block_j)
        if (
            cur_com_i is None
            or cur_com_j is None
            or ref_com_i is None
            or ref_com_j is None
        ):
            self.skip_reason = "motif_com_or_reference_missing"
            self.skip_detail.update(
                {
                    "has_current_com_i": cur_com_i is not None,
                    "has_current_com_j": cur_com_j is not None,
                    "has_reference_com_i": ref_com_i is not None,
                    "has_reference_com_j": ref_com_j is not None,
                }
            )
            return xyz.new_zeros(())

        target_rotation = _euler_rotation_matrix_deg(
            self.angle_x,
            self.angle_y,
            self.angle_z,
            device=xyz.device,
            dtype=xyz.dtype,
        )
        ref_edges = torch.stack(
            [
                ref_com_i,
                ref_com_j,
                ref_com_j - ref_com_i,
            ],
            dim=0,
        )
        target_edges = _apply_row_rotation(ref_edges, target_rotation)
        target_norm = target_edges.norm(dim=-1)

        cur_edges = torch.stack(
            [
                cur_com_i,
                cur_com_j,
                cur_com_j - cur_com_i,
            ],
            dim=1,
        )
        cur_norm = cur_edges.norm(dim=-1)
        valid = (target_norm[None, :] > 1e-6) & (cur_norm > 1e-6)
        if not bool(valid.any()):
            self.skip_reason = "all_pose_triangle_edges_degenerate"
            self.skip_detail.update(
                {
                    "target_edge_norms": _rounded_float_list(target_norm),
                    "current_edge_norm_min": _rounded_float_list(
                        cur_norm.amin(dim=0)
                    ),
                    "current_edge_norm_max": _rounded_float_list(
                        cur_norm.amax(dim=0)
                    ),
                }
            )
            return xyz.new_zeros(())

        target_dirs = target_edges / target_norm.clamp_min(1e-6)[:, None]
        cur_dirs = cur_edges / cur_norm.clamp_min(1e-6)[..., None]
        diff = cur_dirs - target_dirs.unsqueeze(0)
        loss = diff.pow(2).sum(dim=-1)
        valid_weight = valid.to(dtype=xyz.dtype)
        return -self.weight * (loss * valid_weight).sum() / valid_weight.sum()

    def guide_atom_mask(self, masks, metadata, device):
        block_i = _single_motif_block_mask(
            self.motif_i,
            masks=masks,
            metadata=metadata,
            device=device,
        )
        block_j = _single_motif_block_mask(
            self.motif_j,
            masks=masks,
            metadata=metadata,
            device=device,
        )
        if block_i is None or block_j is None:
            any_mask = next(iter(masks.values()))
            return torch.zeros_like(any_mask, dtype=torch.bool, device=device)
        return block_i | block_j

    def transform_atom_gradient(self, atom_grad, masks, metadata, xyz):
        block_i = _single_motif_block_mask(
            self.motif_i,
            masks=masks,
            metadata=metadata,
            device=xyz.device,
        )
        block_j = _single_motif_block_mask(
            self.motif_j,
            masks=masks,
            metadata=metadata,
            device=xyz.device,
        )
        if block_i is None or block_j is None:
            return torch.zeros_like(atom_grad)

        transformed = torch.zeros_like(atom_grad)
        for block_mask in (block_i, block_j):
            if block_mask.sum().item() == 0:
                continue
            transformed[:, block_mask, :] = _project_gradient_to_rigid_body(
                atom_grad[:, block_mask, :],
                xyz[:, block_mask, :],
                allow_translation=True,
                allow_rotation=False,
            )
        return transformed


class MotifRigidBodyPose(BasePotential):
    """Frame-invariant rigid-body pose control for two motif blocks.

    This potential treats motif_i and motif_j as rigid bodies.  It fits the
    current motif_i atom cloud to its input reference with Kabsch, uses that fit
    as a local body frame, and compares three vectors in that local frame:

    - motif_i -> motif_j, controlling relative motif placement
    - protein/selected COM -> motif_i, controlling motif_i placement to origin
    - protein/selected COM -> motif_j, controlling motif_j placement to origin

    Because all comparisons are made in motif_i's local reference frame, the
    potential is invariant to RFD3's per-step global recentering and random
    rotation.  Gradients are projected back onto rigid-body translation and
    rotation fields for both selected motif blocks.
    """

    def __init__(
        self,
        weight: float = 1.0,
        motif_i: int = 0,
        motif_j: int = 1,
        angle_x: float = 0.0,
        angle_y: float = 0.0,
        angle_z: float = 0.0,
        pair_weight: float = 1.0,
        origin_weight: float = 1.0,
        distance_weight: float = 0.0,
        origin_atom_filter: str = "real",
        eps: float = 1e-6,
    ):
        super().__init__(weight)
        self.motif_i = int(motif_i)
        self.motif_j = int(motif_j)
        self.angle_x = float(angle_x)
        self.angle_y = float(angle_y)
        self.angle_z = float(angle_z)
        self.pair_weight = float(pair_weight)
        self.origin_weight = float(origin_weight)
        self.distance_weight = float(distance_weight)
        self.origin_atom_filter = origin_atom_filter
        self.eps = float(eps)
        self.skip_reason: str | None = None
        self.skip_detail: dict | None = None

    def compute(self, xyz, masks, metadata):
        self.skip_reason = None
        self.skip_detail = None
        motif_blocks = _motif_distance_blocks(masks, metadata, xyz.device)
        self.skip_detail = {
            "motif_i": self.motif_i,
            "motif_j": self.motif_j,
            "n_motif_blocks": len(motif_blocks),
            "has_floating_motif_reference_pos": "floating_motif_reference_pos"
            in metadata,
            "has_input_pos": "input_pos" in metadata,
            "has_motif_pos": "motif_pos" in metadata,
            "has_ref_pos": "ref_pos" in metadata,
            "origin_atom_filter": self.origin_atom_filter,
        }

        max_idx = max(self.motif_i, self.motif_j)
        if self.motif_i < 0 or self.motif_j < 0 or max_idx >= len(motif_blocks):
            self.skip_reason = "motif_block_missing"
            return xyz.new_zeros(())

        block_i = motif_blocks[self.motif_i]
        block_j = motif_blocks[self.motif_j]
        current_i, ref_i = _motif_block_current_and_reference_xyz(
            xyz, masks, metadata, block_i
        )
        current_j, ref_j = _motif_block_current_and_reference_xyz(
            xyz, masks, metadata, block_j
        )
        if (
            current_i is None
            or ref_i is None
            or current_j is None
            or ref_j is None
            or ref_i.shape[0] < 3
            or ref_j.shape[0] < 1
        ):
            self.skip_reason = "motif_reference_missing_or_too_small"
            self.skip_detail.update(
                {
                    "n_reference_atoms_i": 0 if ref_i is None else int(ref_i.shape[0]),
                    "n_reference_atoms_j": 0 if ref_j is None else int(ref_j.shape[0]),
                }
            )
            return xyz.new_zeros(())

        current_com_i = current_i.mean(dim=1)
        current_com_j = current_j.mean(dim=1)
        ref_com_i = ref_i.mean(dim=0)
        ref_com_j = ref_j.mean(dim=0)

        origin_mask = _pose_origin_atom_mask(
            self.origin_atom_filter, masks, xyz.device
        )
        current_origin = _masked_current_com(xyz, origin_mask)
        ref_origin = _masked_reference_com(xyz, masks, metadata, origin_mask)
        if current_origin is None or ref_origin is None:
            self.skip_reason = "origin_reference_missing"
            self.skip_detail.update(
                {
                    "has_current_origin": current_origin is not None,
                    "has_reference_origin": ref_origin is not None,
                }
            )
            return xyz.new_zeros(())

        ref_to_current = _kabsch_ref_to_current_rotation(ref_i, current_i, self.eps)
        if ref_to_current is None:
            self.skip_reason = "motif_i_frame_degenerate"
            self.skip_detail.update(
                {
                    "n_reference_atoms_i": int(ref_i.shape[0]),
                    "reference_i_centered_norm": round(
                        float(
                            (ref_i - ref_i.mean(dim=0, keepdim=True))
                            .pow(2)
                            .sum()
                            .detach()
                            .cpu()
                        ),
                        6,
                    ),
                    "current_i_centered_norm_min": _rounded_float_list(
                        (current_i - current_i.mean(dim=1, keepdim=True))
                        .pow(2)
                        .sum(dim=(-2, -1))
                        .amin(dim=0)
                        .unsqueeze(0)
                    ),
                }
            )
            return xyz.new_zeros(())

        target_rotation = _euler_rotation_matrix_deg(
            self.angle_x,
            self.angle_y,
            self.angle_z,
            device=xyz.device,
            dtype=xyz.dtype,
        )

        current_vectors = torch.stack(
            [
                current_com_j - current_com_i,
                current_com_i - current_origin,
                current_com_j - current_origin,
            ],
            dim=1,
        )
        current_local = _apply_batch_row_rotation(
            current_vectors,
            ref_to_current,
        )
        target_vectors = torch.stack(
            [
                ref_com_j - ref_com_i,
                ref_com_i - ref_origin,
                ref_com_j - ref_origin,
            ],
            dim=0,
        )
        target_vectors = _apply_row_rotation(target_vectors, target_rotation)

        direction_loss, valid = _pose_direction_loss(
            current_local,
            target_vectors,
            xyz.dtype,
            self.eps,
        )
        if not bool(valid.any()):
            self.skip_reason = "all_pose_vectors_degenerate"
            self.skip_detail.update(
                {
                    "target_vector_norms": _rounded_float_list(
                        target_vectors.norm(dim=-1)
                    ),
                    "current_vector_norm_min": _rounded_float_list(
                        current_local.norm(dim=-1).amin(dim=0)
                    ),
                    "current_vector_norm_max": _rounded_float_list(
                        current_local.norm(dim=-1).amax(dim=0)
                    ),
                }
            )
            return xyz.new_zeros(())

        weights = torch.tensor(
            [self.pair_weight, self.origin_weight, self.origin_weight],
            device=xyz.device,
            dtype=xyz.dtype,
        )
        valid_weight = valid.to(dtype=xyz.dtype) * weights[None, :]
        loss = (direction_loss * valid_weight).sum() / valid_weight.sum().clamp_min(
            self.eps
        )

        if self.distance_weight > 0.0:
            distance_loss = _pose_distance_loss(
                current_local,
                target_vectors,
                valid,
                self.eps,
            )
            loss = loss + self.distance_weight * distance_loss

        return -self.weight * loss

    def guide_atom_mask(self, masks, metadata, device):
        motif_blocks = _motif_distance_blocks(masks, metadata, device)
        if not motif_blocks:
            any_mask = next(iter(masks.values()))
            return torch.zeros_like(any_mask, dtype=torch.bool, device=device)
        guide_mask = torch.zeros_like(motif_blocks[0], dtype=torch.bool, device=device)
        max_idx = max(self.motif_i, self.motif_j)
        if self.motif_i < 0 or self.motif_j < 0 or max_idx >= len(motif_blocks):
            return guide_mask
        guide_mask |= motif_blocks[self.motif_i]
        guide_mask |= motif_blocks[self.motif_j]
        return guide_mask

    def transform_atom_gradient(self, atom_grad, masks, metadata, xyz):
        motif_blocks = _motif_distance_blocks(masks, metadata, xyz.device)
        if not motif_blocks:
            return torch.zeros_like(atom_grad)

        transformed = torch.zeros_like(atom_grad)
        max_idx = max(self.motif_i, self.motif_j)
        if self.motif_i < 0 or self.motif_j < 0 or max_idx >= len(motif_blocks):
            return transformed

        for motif_idx in (self.motif_i, self.motif_j):
            block_mask = motif_blocks[motif_idx].to(device=xyz.device, dtype=torch.bool)
            if block_mask.sum().item() < 2:
                continue
            transformed[:, block_mask, :] = _project_gradient_to_rigid_body(
                atom_grad[:, block_mask, :],
                xyz[:, block_mask, :],
                allow_translation=True,
                allow_rotation=True,
            )
        return transformed


class MotifBridge(BasePotential):
    """Spread generated non-motif atoms between two motif blocks.

    ``motif_distance`` only acts on the motif atoms that define the two motif
    centers.  This potential acts on the movable non-motif atoms instead.  It
    projects those atoms onto the motif-center axis and encourages the projected
    positions to be evenly distributed between motif_i and motif_j.  A soft tube
    penalty can also keep those atoms near the inter-motif region without
    collapsing them onto a line.

    Returns a negative penalty, so gradient ascent reduces scaffold spread error.
    """

    def __init__(
        self,
        weight: float = 1.0,
        motif_i: int = 0,
        motif_j: int = 1,
        spread_weight: float = 1.0,
        outside_weight: float = 1.0,
        tube_weight: float = 0.2,
        max_radius: float = 12.0,
        atom_filter: str = "guide",
        include_motif_atoms: bool = False,
        eps: float = 1e-6,
    ):
        super().__init__(weight)
        if max_radius < 0.0:
            raise ValueError("motif_bridge max_radius must be non-negative")
        valid_atom_filters = (
            "guide",
            "potential",
            "binder",
            "generated",
            "real",
            "backbone",
            "CA",
            "all",
        )
        if atom_filter not in valid_atom_filters:
            raise ValueError(
                "motif_bridge atom_filter must be one of "
                f"{valid_atom_filters}"
            )
        self.motif_i = int(motif_i)
        self.motif_j = int(motif_j)
        self.spread_weight = float(spread_weight)
        self.outside_weight = float(outside_weight)
        self.tube_weight = float(tube_weight)
        self.max_radius = float(max_radius)
        self.atom_filter = atom_filter
        self.include_motif_atoms = bool(include_motif_atoms)
        self.eps = float(eps)

    def compute(self, xyz, masks, metadata):
        motif_blocks = _motif_blocks(masks, metadata, xyz.device)
        max_idx = max(self.motif_i, self.motif_j)
        if self.motif_i < 0 or self.motif_j < 0 or max_idx >= len(motif_blocks):
            return xyz.new_zeros(())

        motif_a = xyz[:, motif_blocks[self.motif_i], :]
        motif_b = xyz[:, motif_blocks[self.motif_j], :]
        if motif_a.shape[1] == 0 or motif_b.shape[1] == 0:
            return xyz.new_zeros(())

        bridge_mask = _bridge_atom_mask(
            atom_filter=self.atom_filter,
            masks=masks,
            device=xyz.device,
        )
        if not self.include_motif_atoms:
            bridge_mask = bridge_mask & ~masks["motif_atom_mask"].to(
                device=xyz.device, dtype=torch.bool
            )
        if bridge_mask.sum().item() < 1:
            return xyz.new_zeros(())

        bridge_xyz = xyz[:, bridge_mask, :]
        com_a = motif_a.mean(dim=1)
        com_b = motif_b.mean(dim=1)
        axis = com_b - com_a
        axis_len = axis.norm(dim=-1).clamp(min=self.eps)
        axis_unit = axis / axis_len[:, None]

        rel = bridge_xyz - com_a[:, None, :]
        alpha = (rel * axis_unit[:, None, :]).sum(dim=-1) / axis_len[:, None]

        sorted_alpha = torch.sort(alpha, dim=-1).values
        if sorted_alpha.shape[-1] == 1:
            target_alpha = sorted_alpha.new_full(sorted_alpha.shape, 0.5)
        else:
            target_alpha = torch.linspace(
                0.0,
                1.0,
                sorted_alpha.shape[-1],
                device=xyz.device,
                dtype=xyz.dtype,
            )
            target_alpha = target_alpha[None, :].expand_as(sorted_alpha)

        spread_loss = (sorted_alpha - target_alpha).pow(2).mean()
        outside_loss = (
            torch.relu(-alpha).pow(2) + torch.relu(alpha - 1.0).pow(2)
        ).mean()

        projected = com_a[:, None, :] + alpha[:, :, None] * axis[:, None, :]
        radial_dist = (bridge_xyz - projected).norm(dim=-1)
        tube_loss = torch.relu(radial_dist - self.max_radius).pow(2).mean()

        total_loss = (
            self.spread_weight * spread_loss
            + self.outside_weight * outside_loss
            + self.tube_weight * tube_loss
        )
        return -self.weight * total_loss


class MotifBridge2(BasePotential):
    """Spread non-motif atoms inside a rounded cylinder around two motif blocks.

    This is a two-motif alternative to ``motif_bridge``.  It still uses the
    motif-center axis for longitudinal spread, but replaces the tube with a
    capped cylinder.  The cylinder has constant radius between the two motif
    COMs, then starts rounding at each motif COM and extends outward by
    ``end_padding``.  This lets guided atoms wrap around the motif ends without
    making the middle of the bridge artificially wider than the ends.

    ``end_bias`` controls the target longitudinal distribution.  At 0.0, atoms
    are spread evenly from motif_i to motif_j.  Larger values bias the target
    positions toward both motif ends, leaving fewer atoms in the middle.
    """

    def __init__(
        self,
        weight: float = 1.0,
        motif_i: int = 0,
        motif_j: int = 1,
        spread_weight: float = 1.0,
        outside_weight: float = 1.0,
        cylinder_weight: float | None = None,
        ellipsoid_weight: float = 1.0,
        radius: float = 12.0,
        end_padding: float = 2.0,
        end_bias: float = 0.0,
        atom_filter: str = "guide",
        include_motif_atoms: bool = False,
        eps: float = 1e-6,
    ):
        super().__init__(weight)
        if radius <= 0.0:
            raise ValueError("motif_bridge2 radius must be positive")
        if end_padding < 0.0:
            raise ValueError("motif_bridge2 end_padding must be non-negative")
        if end_bias < 0.0:
            raise ValueError("motif_bridge2 end_bias must be non-negative")
        valid_atom_filters = (
            "guide",
            "potential",
            "binder",
            "generated",
            "real",
            "backbone",
            "CA",
            "all",
        )
        if atom_filter not in valid_atom_filters:
            raise ValueError(
                "motif_bridge2 atom_filter must be one of "
                f"{valid_atom_filters}"
            )
        self.motif_i = int(motif_i)
        self.motif_j = int(motif_j)
        self.spread_weight = float(spread_weight)
        self.outside_weight = float(outside_weight)
        self.cylinder_weight = (
            float(ellipsoid_weight)
            if cylinder_weight is None
            else float(cylinder_weight)
        )
        self.ellipsoid_weight = self.cylinder_weight
        self.radius = float(radius)
        self.end_padding = float(end_padding)
        self.end_bias = float(end_bias)
        self.atom_filter = atom_filter
        self.include_motif_atoms = bool(include_motif_atoms)
        self.eps = float(eps)

    def compute(self, xyz, masks, metadata):
        motif_blocks = _motif_blocks(masks, metadata, xyz.device)
        max_idx = max(self.motif_i, self.motif_j)
        if self.motif_i < 0 or self.motif_j < 0 or max_idx >= len(motif_blocks):
            return xyz.new_zeros(())

        motif_a = xyz[:, motif_blocks[self.motif_i], :]
        motif_b = xyz[:, motif_blocks[self.motif_j], :]
        if motif_a.shape[1] == 0 or motif_b.shape[1] == 0:
            return xyz.new_zeros(())

        bridge_mask = _bridge_atom_mask(
            atom_filter=self.atom_filter,
            masks=masks,
            device=xyz.device,
        )
        if not self.include_motif_atoms:
            bridge_mask = bridge_mask & ~masks["motif_atom_mask"].to(
                device=xyz.device, dtype=torch.bool
            )
        if bridge_mask.sum().item() < 1:
            return xyz.new_zeros(())

        bridge_xyz = xyz[:, bridge_mask, :]
        com_a = motif_a.mean(dim=1)
        com_b = motif_b.mean(dim=1)
        axis = com_b - com_a
        axis_len = axis.norm(dim=-1).clamp(min=self.eps)
        axis_unit = axis / axis_len[:, None]

        rel_from_a = bridge_xyz - com_a[:, None, :]
        alpha = (rel_from_a * axis_unit[:, None, :]).sum(dim=-1) / axis_len[:, None]
        pad_alpha = self.end_padding / axis_len[:, None]
        alpha_min = -pad_alpha
        alpha_max = 1.0 + pad_alpha
        alpha_span = (alpha_max - alpha_min).clamp(min=self.eps)

        sorted_alpha = torch.sort(alpha, dim=-1).values
        n_atoms = sorted_alpha.shape[-1]
        if n_atoms == 1:
            target_unit = sorted_alpha.new_full(sorted_alpha.shape, 0.5)
        else:
            target_unit = torch.linspace(
                0.0,
                1.0,
                n_atoms,
                device=xyz.device,
                dtype=xyz.dtype,
            )
            target_unit = target_unit[None, :].expand_as(sorted_alpha)
            end_clustered = 0.5 * (1.0 - torch.cos(torch.pi * target_unit))
            bias_mix = self.end_bias / (1.0 + self.end_bias)
            target_unit = (1.0 - bias_mix) * target_unit + bias_mix * end_clustered

        target_alpha = alpha_min + target_unit * alpha_span
        spread_loss = (sorted_alpha - target_alpha).pow(2).mean()
        outside_loss = (
            torch.relu(alpha_min - alpha).pow(2)
            + torch.relu(alpha - alpha_max).pow(2)
        ).mean()

        center = 0.5 * (com_a + com_b)
        rel_center = bridge_xyz - center[:, None, :]
        axial = (rel_center * axis_unit[:, None, :]).sum(dim=-1)
        projected = center[:, None, :] + axial[:, :, None] * axis_unit[:, None, :]
        radial_dist = (bridge_xyz - projected).norm(dim=-1)

        radius = bridge_xyz.new_tensor(self.radius).clamp(min=self.eps)
        if self.end_padding > 0.0:
            before_a = alpha < 0.0
            after_b = alpha > 1.0
            cap_pos = torch.zeros_like(alpha)
            cap_pos = torch.where(before_a, alpha / pad_alpha.clamp(min=self.eps), cap_pos)
            cap_pos = torch.where(
                after_b,
                (alpha - 1.0) / pad_alpha.clamp(min=self.eps),
                cap_pos,
            )
            cap_scale = torch.sqrt(torch.relu(1.0 - cap_pos.pow(2)))
            allowed_radius = torch.where(before_a | after_b, radius * cap_scale, radius)
        else:
            allowed_radius = torch.ones_like(radial_dist) * radius
        cylinder_loss = torch.relu(radial_dist - allowed_radius).pow(2).mean()

        total_loss = (
            self.spread_weight * spread_loss
            + self.outside_weight * outside_loss
            + self.cylinder_weight * cylinder_loss
        )
        return -self.weight * total_loss


class SymmetryAwareMotifBridge(BasePotential):
    """Run motif_bridge independently per symmetry subunit.

    This is the two-motif bridge: selected non-motif atoms in each subunit are
    spread between local motif_i and motif_j for that same subunit.
    """

    def __init__(
        self,
        weight: float = 1.0,
        motif_i: int = 0,
        motif_j: int = 1,
        spread_weight: float = 1.0,
        outside_weight: float = 1.0,
        tube_weight: float = 0.2,
        max_radius: float = 12.0,
        atom_filter: str = "guide",
        include_motif_atoms: bool = False,
        eps: float = 1e-6,
        reduction: str = "sum",
    ):
        super().__init__(weight)
        if max_radius < 0.0:
            raise ValueError("symmetry_motif_bridge max_radius must be non-negative")
        self.motif_i = int(motif_i)
        self.motif_j = int(motif_j)
        self.spread_weight = float(spread_weight)
        self.outside_weight = float(outside_weight)
        self.tube_weight = float(tube_weight)
        self.max_radius = float(max_radius)
        self.atom_filter = atom_filter
        self.include_motif_atoms = bool(include_motif_atoms)
        self.eps = float(eps)
        self.reduction = _validate_symmetry_reduction(reduction)

    def compute(self, xyz, masks, metadata):
        subunits = _symmetry_subunit_motif_blocks(masks, metadata, xyz.device)
        bridge_masks = _symmetry_subunit_bridge_masks(
            masks,
            metadata,
            xyz.device,
            self.atom_filter,
            self.include_motif_atoms,
        )
        total_loss = xyz.new_zeros(())
        n_active = 0
        for subunit_idx, local_blocks in enumerate(subunits):
            if self.motif_i >= len(local_blocks) or self.motif_j >= len(local_blocks):
                continue
            if subunit_idx >= len(bridge_masks):
                continue
            loss = _two_motif_bridge_loss(
                xyz,
                local_blocks[self.motif_i],
                local_blocks[self.motif_j],
                bridge_masks[subunit_idx],
                self,
            )
            if loss is None:
                continue
            total_loss = total_loss + loss
            n_active += 1
        if n_active == 0:
            return xyz.new_zeros(())
        return _weighted_symmetry_loss(self.weight, total_loss, n_active, self.reduction)

    def guide_atom_mask(self, masks, metadata, device):
        return _symmetry_aware_bridge_guide_mask(
            masks,
            metadata,
            device,
            self.atom_filter,
            self.include_motif_atoms,
        )

    def instance_guide_masks(self, masks, metadata, device):
        return _symmetry_subunit_bridge_masks(
            masks,
            metadata,
            device,
            self.atom_filter,
            self.include_motif_atoms,
        )


class SymmetryAwareSingleMotifBridge(BasePotential):
    """Distribute selected atoms toward one local motif per symmetry subunit.

    The selected non-motif atoms in each subunit are sorted by distance to the
    local motif COM and encouraged to occupy an even radial distribution from
    the motif center out to ``max_radius``.  This gives a one-motif bridge-like
    scaffold packing term when there is no second motif endpoint.
    """

    def __init__(
        self,
        weight: float = 1.0,
        motif_i: int = 0,
        spread_weight: float = 1.0,
        outside_weight: float = 1.0,
        max_radius: float = 12.0,
        atom_filter: str = "guide",
        include_motif_atoms: bool = False,
        eps: float = 1e-6,
        reduction: str = "sum",
    ):
        super().__init__(weight)
        if max_radius <= 0.0:
            raise ValueError("symmetry_single_motif_bridge max_radius must be positive")
        self.motif_i = int(motif_i)
        self.spread_weight = float(spread_weight)
        self.outside_weight = float(outside_weight)
        self.max_radius = float(max_radius)
        self.atom_filter = atom_filter
        self.include_motif_atoms = bool(include_motif_atoms)
        self.eps = float(eps)
        self.reduction = _validate_symmetry_reduction(reduction)

    def compute(self, xyz, masks, metadata):
        subunits = _symmetry_subunit_motif_blocks(masks, metadata, xyz.device)
        bridge_masks = _symmetry_subunit_bridge_masks(
            masks,
            metadata,
            xyz.device,
            self.atom_filter,
            self.include_motif_atoms,
        )
        total_loss = xyz.new_zeros(())
        n_active = 0
        for subunit_idx, local_blocks in enumerate(subunits):
            if self.motif_i >= len(local_blocks) or subunit_idx >= len(bridge_masks):
                continue
            loss = _single_motif_bridge_loss(
                xyz,
                local_blocks[self.motif_i],
                bridge_masks[subunit_idx],
                self,
            )
            if loss is None:
                continue
            total_loss = total_loss + loss
            n_active += 1
        if n_active == 0:
            return xyz.new_zeros(())
        return _weighted_symmetry_loss(self.weight, total_loss, n_active, self.reduction)

    def guide_atom_mask(self, masks, metadata, device):
        return _symmetry_aware_bridge_guide_mask(
            masks,
            metadata,
            device,
            self.atom_filter,
            self.include_motif_atoms,
        )

    def instance_guide_masks(self, masks, metadata, device):
        return _symmetry_subunit_bridge_masks(
            masks,
            metadata,
            device,
            self.atom_filter,
            self.include_motif_atoms,
        )


class SymmetryEllipsoidBridge(BasePotential):
    """Distribute scaffold atoms inside a subunit-shaped ellipsoid.

    For each symmetric subunit the ellipsoid is defined by three axes anchored
    at the subunit COM:

      axis 1 – COM → motif COM            (long axis, semi-length = dist(COM, motif))
      axis 2 – COM → midpoint(COM, left-neighbor COM)  (semi-length = dist/2)
      axis 3 – COM → midpoint(COM, right-neighbor COM) (semi-length = dist/2, GS-orth)

    The three directions are Gram–Schmidt orthonormalised so the ellipsoidal
    radius is well-defined.  Scaffold atoms are spread evenly inside the
    ellipsoid using bin-midpoint targets (same approach as the fixed
    SymmetryAwareSingleMotifBridge), with a two-sided boundary penalty.

    Because the ellipsoid is centred at the subunit COM, atoms on the
    non-motif-facing half of the subunit are naturally included.
    """

    def __init__(
        self,
        weight: float = 1.0,
        motif_i: int = 0,
        spread_weight: float = 1.0,
        outside_weight: float = 1.0,
        atom_filter: str = "guide",
        include_motif_atoms: bool = False,
        eps: float = 1e-6,
        reduction: str = "sum",
    ):
        super().__init__(weight)
        self.motif_i = int(motif_i)
        self.spread_weight = float(spread_weight)
        self.outside_weight = float(outside_weight)
        self.atom_filter = atom_filter
        self.include_motif_atoms = bool(include_motif_atoms)
        self.eps = float(eps)
        self.reduction = _validate_symmetry_reduction(reduction)

    def compute(self, xyz, masks, metadata):
        subunits = _symmetry_subunit_motif_blocks(masks, metadata, xyz.device)
        bridge_masks = _symmetry_subunit_bridge_masks(
            masks, metadata, xyz.device, self.atom_filter, self.include_motif_atoms
        )
        real_mask = masks.get(
            "real_atom_mask",
            torch.ones(xyz.shape[1], dtype=torch.bool, device=xyz.device),
        )
        subunit_all_masks = _symmetry_subunit_atom_masks(metadata, real_mask, xyz.device)

        n_subunits = len(subunit_all_masks)
        if n_subunits < 2:
            return xyz.new_zeros(())

        # Precompute per-subunit COMs  [n_subunits] of (D, 3) or None
        subunit_coms = [
            _motif_block_current_com(xyz, m) for m in subunit_all_masks
        ]

        total_loss = xyz.new_zeros(())
        n_active = 0
        for subunit_idx in range(n_subunits):
            if subunit_idx >= len(subunits) or subunit_idx >= len(bridge_masks):
                continue
            local_blocks = subunits[subunit_idx]
            if self.motif_i >= len(local_blocks):
                continue

            motif_com = _motif_block_current_com(xyz, local_blocks[self.motif_i])
            subunit_com = subunit_coms[subunit_idx]
            com_left = subunit_coms[(subunit_idx - 1) % n_subunits]
            com_right = subunit_coms[(subunit_idx + 1) % n_subunits]

            if any(c is None for c in (motif_com, subunit_com, com_left, com_right)):
                continue

            loss = _ellipsoid_bridge_loss(
                xyz,
                motif_com,
                subunit_com,
                com_left,
                com_right,
                bridge_masks[subunit_idx],
                self,
            )
            if loss is None:
                continue
            total_loss = total_loss + loss
            n_active += 1

        if n_active == 0:
            return xyz.new_zeros(())
        return _weighted_symmetry_loss(self.weight, total_loss, n_active, self.reduction)

    def guide_atom_mask(self, masks, metadata, device):
        return _symmetry_aware_bridge_guide_mask(
            masks, metadata, device, self.atom_filter, self.include_motif_atoms
        )

    def instance_guide_masks(self, masks, metadata, device):
        return _symmetry_subunit_bridge_masks(
            masks, metadata, device, self.atom_filter, self.include_motif_atoms
        )


class MotifRigid(BasePotential):
    """Preserve motif geometry against RFD3 reference coordinates.

    The restraint compares current motif atom-pair distances to reference
    distances.  Using distances makes the potential invariant to global
    translation and rotation while preserving local secondary geometry and,
    by default, tertiary distances between separate motif blocks.

    Reference coordinates are read from ``metadata["ref_pos"]`` for fixed-seq
    motifs and from ``metadata["motif_pos"]`` for fixed-coordinate motifs.
    """

    def __init__(
        self,
        weight: float = 1.0,
        k: float = 1.0,
        loss: str = "pseudo_huber",
        group_mode: str = "all",
        atom_filter: str = "potential",
        motif_i: int | None = None,
        min_separation: int = 0,
    ):
        super().__init__(weight)
        if k <= 0.0:
            raise ValueError("motif_rigid k must be positive")
        if loss not in ("pseudo_huber", "mse", "l1"):
            raise ValueError(
                "motif_rigid loss must be one of 'pseudo_huber', 'mse', or 'l1'"
            )
        if group_mode not in ("all", "blocks"):
            raise ValueError("motif_rigid group_mode must be 'all' or 'blocks'")
        valid_atom_filters = ("potential", "all", "backbone", "CA", "real")
        if atom_filter not in valid_atom_filters:
            raise ValueError(
                "motif_rigid atom_filter must be one of "
                f"{valid_atom_filters}"
            )
        self.k = float(k)
        self.loss = loss
        self.group_mode = group_mode
        self.atom_filter = atom_filter
        self.motif_i = None if motif_i is None else int(motif_i)
        self.min_separation = int(min_separation)

    def compute(self, xyz, masks, metadata):
        motif_mask = _motif_reference_mask(masks, metadata, xyz.device)
        motif_mask = motif_mask & _atom_filter_mask(self.atom_filter, masks, xyz.device)
        if motif_mask.sum().item() < 2:
            return xyz.new_zeros(())

        ref_xyz = _motif_reference_xyz(xyz, masks, metadata)
        ref_xyz = ref_xyz.to(device=xyz.device, dtype=xyz.dtype)

        if self.group_mode == "all":
            groups = [motif_mask]
        elif self.group_mode == "blocks":
            groups = _motif_blocks(
                {"motif_atom_mask": motif_mask, "potential_atom_mask": motif_mask},
                metadata,
                xyz.device,
            )
        else:
            raise ValueError("motif_rigid group_mode must be 'all' or 'blocks'")

        if self.motif_i is not None:
            if self.group_mode != "blocks":
                raise ValueError(
                    "motif_rigid motif_i can only be used with group_mode='blocks'"
                )
            if self.motif_i < 0 or self.motif_i >= len(groups):
                return xyz.new_zeros(())
            groups = [groups[self.motif_i]]

        loss = xyz.new_zeros(())
        n_groups = 0
        for group_mask in groups:
            idx = torch.where(group_mask)[0]
            if idx.numel() < 2:
                continue
            cur = xyz[:, idx, :]
            ref = ref_xyz[idx]
            valid = torch.isfinite(ref).all(dim=-1)
            if valid.sum().item() < 2:
                continue
            cur = cur[:, valid, :]
            ref = ref[valid]

            dcur = torch.cdist(cur, cur)
            dref = torch.cdist(ref[None, :, :], ref[None, :, :]).to(dtype=xyz.dtype)
            pair_mask = torch.triu(
                torch.ones(dcur.shape[-2:], dtype=torch.bool, device=xyz.device),
                diagonal=1,
            )
            if self.min_separation > 0:
                residues = torch.arange(dcur.shape[-1], device=xyz.device)
                pair_mask &= (
                    (residues[:, None] - residues[None, :]).abs()
                    >= self.min_separation
                )
            if pair_mask.sum().item() == 0:
                continue
            diff = dcur[:, pair_mask] - dref[:, pair_mask]
            loss = loss + _distance_loss(diff, self.k, self.loss)
            n_groups += 1

        if n_groups == 0:
            return xyz.new_zeros(())
        return -self.weight * loss / n_groups


class MotifCOMDistance(BasePotential):
    """Harmonic restraint on the distance of each motif center from the protein COM.

    For each contig-defined motif block that has an entry in ``target_distances``,
    the distance from the block's center of mass to the protein-wide center of mass
    is penalised toward the specified target:

        loss_i = (dist(motif_i_center, protein_COM) - target_distances[i])²
        return  = -weight * mean_over_active_motifs(loss_i)

    The protein COM is computed from atoms selected by ``origin_atom_filter`` and is
    detached from the gradient, so the potential only moves motif atoms — it cannot
    drag the rest of the protein.

    Gradient is rigidised to pure rigid-body translation per block (all atoms in
    the block receive the same displacement).

    ``target_distances`` is a list of floats (Å), one per motif block in contig
    order.  Blocks without an entry are not constrained.  An empty list produces
    a zero potential.
    """

    def __init__(
        self,
        weight: float = 1.0,
        target_distances: list | None = None,
        origin_atom_filter: str = "real",
        eps: float = 1e-6,
    ):
        super().__init__(weight)
        self.target_distances: list = (
            list(target_distances) if target_distances is not None else []
        )
        valid_origin_filters = ("real", "potential", "guide", "motif", "all")
        if origin_atom_filter not in valid_origin_filters:
            raise ValueError(
                f"motif_com_distance origin_atom_filter must be one of {valid_origin_filters}"
            )
        self.origin_atom_filter = origin_atom_filter
        self.eps = float(eps)

    def compute(self, xyz, masks, metadata):
        motif_blocks = _motif_distance_blocks(masks, metadata, xyz.device)
        if not motif_blocks or not self.target_distances:
            return xyz.new_zeros(())

        origin_mask = _pose_origin_atom_mask(self.origin_atom_filter, masks, xyz.device)
        current_com = _masked_current_com(xyz, origin_mask)
        if current_com is None:
            return xyz.new_zeros(())
        current_com = current_com.detach()  # [D, 3] — no gradient through protein COM

        total_loss = xyz.new_zeros(())
        n_active = 0

        for motif_i, block_mask in enumerate(motif_blocks):
            if motif_i >= len(self.target_distances):
                break
            target_dist = float(self.target_distances[motif_i])
            motif_xyz = xyz[:, block_mask, :]  # [D, N, 3]
            if motif_xyz.shape[1] == 0:
                continue
            motif_com = motif_xyz.mean(dim=1)  # [D, 3]
            dist = (motif_com - current_com).norm(dim=-1)  # [D]
            total_loss = total_loss + ((dist - target_dist) ** 2).mean()
            n_active += 1

        if n_active == 0:
            return xyz.new_zeros(())
        return -self.weight * total_loss / n_active

    def guide_atom_mask(self, masks, metadata, device):
        motif_blocks = _motif_distance_blocks(masks, metadata, device)
        if not motif_blocks:
            any_mask = next(iter(masks.values()))
            return torch.zeros_like(any_mask, dtype=torch.bool, device=device)
        guide_mask = torch.zeros_like(motif_blocks[0], dtype=torch.bool, device=device)
        for motif_i, block_mask in enumerate(motif_blocks):
            if motif_i >= len(self.target_distances):
                break
            guide_mask |= block_mask
        return guide_mask

    def transform_atom_gradient(self, atom_grad, masks, metadata, xyz):
        motif_blocks = _motif_distance_blocks(masks, metadata, xyz.device)
        if not motif_blocks:
            return atom_grad
        transformed = torch.zeros_like(atom_grad)
        for motif_i, block_mask in enumerate(motif_blocks):
            if motif_i >= len(self.target_distances):
                break
            if block_mask.sum().item() == 0:
                continue
            block_translation = atom_grad[:, block_mask, :].mean(dim=1, keepdim=True)
            transformed[:, block_mask, :] = block_translation
        return transformed


class MotifSphericalPosition(BasePotential):
    """Constrain each motif's angular position on the sphere around the protein COM.

    Penalises the deviation of the unit vector from the protein COM to each motif
    center from a per-motif target direction derived from the input PDB.  The
    radial distance (how far the motif is from the COM) is completely ignored —
    the loss is a pure normalised-direction comparison, so it only acts on the
    angular (spherical) position of each motif.

    This potential is intentionally complementary to MotifRadialOrientationPotential:
    - MotifSphericalPosition controls *where* each motif sits on the sphere
      (latitude / longitude around the COM).
    - MotifRadialOrientationPotential controls *how* each motif is oriented
      relative to its radial direction (which way it faces outward).

    The reference radial direction for motif i is:
        r_ref_i = normalise(ref_com_i − global_ref_centre)
    where global_ref_centre is the mean of all motif-block reference COMs (the
    same convention as MotifRadialOrientationPotential).  A motif whose reference
    centre coincides with this global centre is skipped.

    ``motif_offsets`` is a list of [angle_x, angle_y, angle_z] Euler angles in
    degrees, one entry per motif block (contig order).  The offset rotates the
    reference radial *direction* in 3-D, allowing you to specify a target
    angular position different from the input PDB.  [0, 0, 0] = preserve PDB
    angular position (default).

    Gradient is projected to pure rigid-body translation per motif block (no
    rotation).  The protein COM is detached from the gradient graph so this
    potential only moves motif atoms, never the rest of the protein.
    """

    def __init__(
        self,
        weight: float = 1.0,
        motif_offsets: list | None = None,
        origin_atom_filter: str = "real",
        eps: float = 1e-6,
    ):
        super().__init__(weight)
        self.motif_offsets: list = list(motif_offsets) if motif_offsets is not None else []
        valid_origin_filters = ("real", "potential", "guide", "motif", "all")
        if origin_atom_filter not in valid_origin_filters:
            raise ValueError(
                "motif_spherical_position origin_atom_filter must be one of "
                f"{valid_origin_filters}"
            )
        self.origin_atom_filter = origin_atom_filter
        self.eps = float(eps)

    def _offset_matrix(
        self,
        motif_i: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if motif_i < len(self.motif_offsets):
            angles = self.motif_offsets[motif_i]
            if hasattr(angles, "__len__") and len(angles) >= 3:
                return _euler_rotation_matrix_deg(
                    float(angles[0]),
                    float(angles[1]),
                    float(angles[2]),
                    device=device,
                    dtype=dtype,
                )
        return torch.eye(3, device=device, dtype=dtype)

    def compute(self, xyz, masks, metadata):
        motif_blocks = _motif_distance_blocks(masks, metadata, xyz.device)
        if not motif_blocks:
            return xyz.new_zeros(())

        origin_mask = _pose_origin_atom_mask(self.origin_atom_filter, masks, xyz.device)
        current_com = _masked_current_com(xyz, origin_mask)
        if current_com is None:
            return xyz.new_zeros(())
        current_com = current_com.detach()  # [D, 3] — no gradient through protein COM

        # Reference global centre: mean of all motif-block reference COMs (same
        # convention as MotifRadialOrientationPotential).
        ref_centers: list[torch.Tensor | None] = []
        for block_mask in motif_blocks:
            ref_centers.append(_motif_block_reference_com(xyz, masks, metadata, block_mask))
        valid_ref_centers = [c for c in ref_centers if c is not None]
        if not valid_ref_centers:
            return xyz.new_zeros(())
        c_ref_global = torch.stack(valid_ref_centers, dim=0).mean(dim=0)  # [3]

        total_loss = xyz.new_zeros(())
        n_active = 0

        for motif_i, block_mask in enumerate(motif_blocks):
            ref_com_i = ref_centers[motif_i]
            if ref_com_i is None:
                continue

            # Reference radial direction (unit vector from global centre to motif).
            d_ref = ref_com_i - c_ref_global  # [3]
            d_ref_norm = d_ref.norm()
            if d_ref_norm < self.eps:
                continue  # motif sits at global centre — degenerate, skip
            r_ref_i = d_ref / d_ref_norm  # [3]

            # Apply per-motif Euler offset to the reference direction.
            R_offset = self._offset_matrix(motif_i, xyz.device, xyz.dtype)
            r_target_i = _apply_row_rotation(r_ref_i.unsqueeze(0), R_offset).squeeze(0)
            r_target_i = (r_target_i / r_target_i.norm().clamp_min(self.eps)).detach()  # [3]

            # Current radial direction — normalised, so radius has no effect.
            m_cur_i = _motif_block_current_com(xyz, block_mask)  # [D, 3]
            if m_cur_i is None:
                continue
            d_cur = m_cur_i - current_com  # [D, 3]
            d_cur_norm = d_cur.norm(dim=-1, keepdim=True).clamp_min(self.eps)
            r_cur_i = d_cur / d_cur_norm  # [D, 3]

            # Normalised-direction loss: 0 when aligned, up to 4 when antiparallel.
            diff = r_cur_i - r_target_i.unsqueeze(0)  # [D, 3]
            total_loss = total_loss + diff.pow(2).sum(dim=-1).mean()
            n_active += 1

        if n_active == 0:
            return xyz.new_zeros(())
        return -self.weight * total_loss / n_active

    def guide_atom_mask(self, masks, metadata, device):
        motif_blocks = _motif_distance_blocks(masks, metadata, device)
        if not motif_blocks:
            any_mask = next(iter(masks.values()))
            return torch.zeros_like(any_mask, dtype=torch.bool, device=device)
        guide_mask = torch.zeros_like(motif_blocks[0], dtype=torch.bool, device=device)
        for block_mask in motif_blocks:
            guide_mask = guide_mask | block_mask
        return guide_mask

    def transform_atom_gradient(self, atom_grad, masks, metadata, xyz):
        motif_blocks = _motif_distance_blocks(masks, metadata, xyz.device)
        if not motif_blocks:
            return torch.zeros_like(atom_grad)
        transformed = torch.zeros_like(atom_grad)
        for block_mask in motif_blocks:
            if block_mask.sum().item() == 0:
                continue
            block_translation = atom_grad[:, block_mask, :].mean(dim=1, keepdim=True)
            transformed[:, block_mask, :] = block_translation
        return transformed


class MotifRadialOrientationPotential(BasePotential):
    """Bias each motif's rotation to preserve its input-PDB radial orientation.

    Each contig-defined motif block is treated as a rigid body. The potential
    biases the rotational pose of each motif relative to the direction from
    the current protein center of mass (COM) to the motif's own center — the
    motif's *radial orientation*.

    Intuition: if a face of a motif points outward from the midpoint of all
    motifs in the input PDB, it will continue to point outward during
    diffusion, regardless of where on a conceptual sphere around the protein
    COM the motif is placed or how far it is from the COM.

    Enforced invariances:
    - Radius: COM-to-motif distance has no effect; the radial direction is
      normalized and the COM is detached from the gradient graph.
    - Angular position on sphere: each motif may orbit freely around the COM.
      The radial direction that builds the target frame is also detached, so
      no angular-position gradient is introduced.
    - Inter-motif distances: each motif is scored independently.

    The reference global centre is the mean of all motif-block reference
    centres. A motif whose reference centre coincides with this global centre
    (degenerate radial vector) is skipped. Therefore the potential requires at
    least two motif blocks with distinct reference centre positions.

    ``motif_offsets`` is a list of [angle_x, angle_y, angle_z] in **degrees**
    for each motif block, indexed in contig order. Missing entries default to
    [0, 0, 0].  The offset rotates the target orientation inside the motif's
    local radial frame:
    - axis 0 (x) = along the radial direction (inward/outward spin)
    - axes 1, 2 (y, z) = tangential rotations

    The input-PDB pose corresponds to offset (0, 0, 0) for every motif.

    Requires at least 3 atoms per motif block with valid reference coordinates
    for a stable local frame.
    """

    def __init__(
        self,
        weight: float = 1.0,
        motif_offsets: list | None = None,
        motif_axis_weights: list | None = None,
        origin_atom_filter: str = "real",
        parallel_transport_frame: bool = False,
        eps: float = 1e-6,
    ):
        super().__init__(weight)
        # Opt-in (default False). Parallel-transport the reference radial frame
        # along r_ref -> r_cur instead of re-seeding tangent axes from world axes.
        self.parallel_transport_frame = bool(parallel_transport_frame)
        self.motif_offsets: list = list(motif_offsets) if motif_offsets is not None else []
        self.motif_axis_weights: list = (
            list(motif_axis_weights) if motif_axis_weights is not None else []
        )
        valid_origin_filters = ("real", "potential", "guide", "motif", "all")
        if origin_atom_filter not in valid_origin_filters:
            raise ValueError(
                "motif_radial_orientation origin_atom_filter must be one of "
                f"{valid_origin_filters}"
            )
        self.origin_atom_filter = origin_atom_filter
        self.eps = float(eps)
        self.skip_reason: str | None = None
        self.skip_detail: dict | None = None

    def _offset_matrix(
        self,
        motif_i: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """3×3 Euler rotation (ZYX) for the per-motif offset, in degrees."""
        if motif_i < len(self.motif_offsets):
            angles = self.motif_offsets[motif_i]
            if hasattr(angles, "__len__") and len(angles) >= 3:
                return _euler_rotation_matrix_deg(
                    float(angles[0]),
                    float(angles[1]),
                    float(angles[2]),
                    device=device,
                    dtype=dtype,
                )
        return torch.eye(3, device=device, dtype=dtype)

    def _axis_weights_tensor(
        self,
        motif_i: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """[3] per-axis loss weights for motif_i. Default [1, 1, 1] (full constraint).

        Axis 0 (r) is the radial direction; axes 1-2 (t1, t2) are the two
        perpendicular tangents.  Setting a weight to 0 removes the loss
        contribution along that local-frame axis.  Because rotation around
        frame axis k leaves the k-th coordinate component unchanged while
        moving the other two, zeroing an axis weight frees the rotational
        DOFs whose atom displacement lives in the remaining two axes:

          axis_weights=[1,0,0] → only r-components penalised → free spin around r
          axis_weights=[0,1,1] → no r-component penalty    → free tumble in t1/t2 plane
          axis_weights=[1,1,1] → full constraint (default, identical to old behaviour)
        """
        if motif_i < len(self.motif_axis_weights):
            aw = self.motif_axis_weights[motif_i]
            if hasattr(aw, "__len__") and len(aw) >= 3:
                return torch.tensor(
                    [float(aw[0]), float(aw[1]), float(aw[2])],
                    device=device,
                    dtype=dtype,
                )
        return torch.ones(3, device=device, dtype=dtype)

    def compute(self, xyz, masks, metadata):
        self.skip_reason = None
        self.skip_detail = None

        motif_blocks = _motif_distance_blocks(masks, metadata, xyz.device)
        self.skip_detail = {
            "n_motif_blocks": len(motif_blocks),
            "has_floating_motif_reference_pos": "floating_motif_reference_pos"
            in metadata,
            "has_input_pos": "input_pos" in metadata,
            "has_motif_pos": "motif_pos" in metadata,
            "has_ref_pos": "ref_pos" in metadata,
            "origin_atom_filter": self.origin_atom_filter,
        }

        if len(motif_blocks) == 0:
            self.skip_reason = "no_motif_blocks"
            return xyz.new_zeros(())

        # Protein COM for this step — detached so the potential cannot push
        # atoms toward or away from the COM (no radial-distance gradient).
        origin_mask = _pose_origin_atom_mask(
            self.origin_atom_filter, masks, xyz.device
        )
        current_com = _masked_current_com(xyz, origin_mask)
        if current_com is None:
            self.skip_reason = "current_com_missing"
            return xyz.new_zeros(())
        current_com = current_com.detach()  # [D, 3]

        # Reference global centre = mean of all motif-block reference centres.
        # Used to define the reference radial direction per motif.
        ref_centers: list[torch.Tensor] = []
        for block_mask in motif_blocks:
            rc = _motif_block_reference_com(xyz, masks, metadata, block_mask)
            ref_centers.append(rc)
        valid_ref_centers = [c for c in ref_centers if c is not None]
        if not valid_ref_centers:
            self.skip_reason = "no_motif_reference_centers"
            return xyz.new_zeros(())
        c_ref_global = torch.stack(valid_ref_centers, dim=0).mean(dim=0)  # [3]

        total_loss = xyz.new_zeros(())
        n_active = 0

        for motif_i, block_mask in enumerate(motif_blocks):
            current_xyz_i, ref_xyz_i = _motif_block_current_and_reference_xyz(
                xyz, masks, metadata, block_mask
            )
            if current_xyz_i is None or ref_xyz_i is None or ref_xyz_i.shape[0] < 3:
                continue

            ref_com_i = ref_xyz_i.mean(dim=0)  # [3]

            # Reference radial direction: from the global reference centre to
            # this motif's reference centre.  Skip degenerate cases (motif at
            # the global centre, or only one motif block).
            d_ref = ref_com_i - c_ref_global  # [3]
            d_ref_norm = d_ref.norm()
            if d_ref_norm < self.eps:
                continue
            r_ref_i = d_ref / d_ref_norm  # [3] unit vector

            # Current motif centre (all block atoms for a stable estimate).
            m_cur_i = _motif_block_current_com(xyz, block_mask)  # [D, 3]
            if m_cur_i is None:
                continue

            # Current radial direction — detached so the gradient does NOT
            # push the motif to a specific angular position on the sphere.
            # Only the motif's internal rotation receives gradient updates.
            d_cur = m_cur_i - current_com  # [D, 3]
            d_cur_norm = d_cur.norm(dim=-1, keepdim=True).clamp_min(self.eps)
            r_cur_i = (d_cur / d_cur_norm).detach()  # [D, 3]

            # Orthonormal frames aligned to the two radial directions.
            # Columns of each frame are [r, t1, t2]: r is the radial axis,
            # t1 and t2 are two perpendicular tangents chosen deterministically.
            A_ref_i = _build_radial_frame(r_ref_i, self.eps)          # [3, 3]
            if self.parallel_transport_frame:
                R_transport = _min_rotation_matrix_batched(r_ref_i, r_cur_i, self.eps)
                A_cur_i = torch.matmul(R_transport, A_ref_i.unsqueeze(0)).detach()  # [D, 3, 3]
            else:
                A_cur_i = _build_radial_frame_batched(r_cur_i, self.eps)  # [D, 3, 3]

            # Express reference coordinates in the local radial frame of
            # A_ref_i, then apply any per-motif rotation offset.
            ref_centered = ref_xyz_i - ref_com_i  # [N, 3]
            R_offset_i = self._offset_matrix(motif_i, xyz.device, xyz.dtype)
            ref_local = ref_centered @ A_ref_i          # [N, 3]
            ref_local_offset = ref_local @ R_offset_i   # [N, 3]

            # Current motif coordinates, centred (removes COM translation).
            current_centered = (
                current_xyz_i - current_xyz_i.mean(dim=1, keepdim=True)
            )  # [D, N, 3]

            # Project current coords into the same local radial frame so the
            # diff can be weighted per axis.  Mathematically equivalent to
            # the global-frame diff when axis_weights=[1,1,1] (A_cur_i is
            # orthogonal → isometry), so the default behaviour is unchanged.
            current_local = current_centered @ A_cur_i                   # [D, N, 3]
            diff_local = current_local - ref_local_offset.unsqueeze(0)   # [D, N, 3]

            # Per-axis weights: 0 = free movement along that frame axis.
            axis_w = self._axis_weights_tensor(motif_i, xyz.device, xyz.dtype)
            total_loss = total_loss + (diff_local.pow(2) * axis_w).sum(dim=-1).mean()
            n_active += 1

        if n_active == 0:
            self.skip_reason = "no_active_motifs"
            return xyz.new_zeros(())

        return -self.weight * total_loss / n_active

    def guide_atom_mask(self, masks, metadata, device):
        motif_blocks = _motif_distance_blocks(masks, metadata, device)
        if not motif_blocks:
            any_mask = next(iter(masks.values()))
            return torch.zeros_like(any_mask, dtype=torch.bool, device=device)
        guide_mask = torch.zeros_like(motif_blocks[0], dtype=torch.bool, device=device)
        for block_mask in motif_blocks:
            guide_mask = guide_mask | block_mask
        return guide_mask

    def transform_atom_gradient(self, atom_grad, masks, metadata, xyz):
        motif_blocks = _motif_distance_blocks(masks, metadata, xyz.device)
        if not motif_blocks:
            return torch.zeros_like(atom_grad)
        transformed = torch.zeros_like(atom_grad)
        for block_mask in motif_blocks:
            if block_mask.sum().item() < 3:
                continue
            # Project to pure rigid rotation: zero translation, zero deformation.
            transformed[:, block_mask, :] = _project_gradient_to_rigid_body(
                atom_grad[:, block_mask, :],
                xyz[:, block_mask, :],
                allow_translation=False,
                allow_rotation=True,
            )
        return transformed


class MotifForbiddenRadialOrientation(BasePotential):
    """Repel each motif's rotation away from its 180-degree-flipped input pose.

    This is the repulsive counterpart to ``MotifRadialOrientationPotential``.
    Given a reference PDB where a motif-receptor pair is docked, rotating that
    pair 180 degrees about the motif-motif axis (the line from the shared
    reference centre through the motif's own reference COM) produces a known
    "forbidden" configuration — verified by inspecting where the receptor ends
    up in that reference PDB. The receptor itself is never part of the RFD3
    contig or the sampled structure, so only the motif's own coordinates are
    needed here; this potential requires no receptor-handling code at all.

    Because ``MotifRadialOrientationPotential`` already expresses each motif's
    reference coordinates in a local frame ``[r, t1, t2]`` with the radial
    (motif-motif axis) direction as axis 0, a 180-degree rotation about that
    axis is exactly the sign flip ``(r, t1, t2) -> (r, -t1, -t2)`` — no new
    rotation-matrix machinery is needed, just negating the two tangential
    components of the projected reference coordinates.

    Unlike the attractive potentials, this restraint is a bounded repulsive
    bump rather than a raw unbounded quadratic:

        bump = 1 - exp(-mean_loss / (2 * sigma**2))
        value = weight * bump

    ``bump`` is 0 exactly at the forbidden pose and saturates to 1 (full
    ``weight``, ~zero gradient) far away from it. This shape is required: an
    unbounded quadratic with a flipped sign would not add new information
    here, since (on the single rotational degree of freedom being probed)
    "maximize deviation from the 180-degree target" is mathematically
    equivalent to "minimize deviation from the 0-degree target" — i.e. it
    would just reproduce the attractive potential. The bounded bump instead
    creates a genuinely localized exclusion zone. ``sigma`` (same Å^2-scale
    units as the underlying local-frame squared-deviation loss) is the width
    hyperparameter: larger ``sigma`` undersamples a wider neighbourhood of
    poses similar to the forbidden one.

    Note: there is deliberately no "forbidden spherical position" counterpart.
    A 180-degree rotation about the motif-motif axis passes through the
    motif's own reference COM, so it never moves that COM — the forbidden
    target for angular position would always coincide exactly with the
    ordinary attractive ``MotifSphericalPosition`` target, so it could only
    ever fight that potential (or penalise the correct pose) without ever
    discriminating the flip. The flip is entirely an orientation-channel
    phenomenon, fully captured by this potential alone.

    IMPORTANT — pair this with ``motif_spherical_position``: the local frame
    ``[r, t1, t2]`` this potential reuses from ``MotifRadialOrientationPotential``
    is built via a Gram-Schmidt construction seeded from whichever fixed global
    axis is least aligned with the motif's *current* radial direction ``r_cur``.
    That seed choice is not equivariant under rotation, so if ``r_cur`` drifts
    far enough from the motif's reference direction ``r_ref`` to flip which
    global axis gets picked, the local-frame coordinates pick up a large,
    spurious jump unrelated to the motif's actual internal rotation (this is a
    pre-existing property of the shared frame-construction helper, inherited
    here unmodified, not something specific to this class). Empirically this
    artifact is negligible while the current and reference radial directions
    stay within roughly 5-10 degrees of each other, and jumps to an O(1)
    spurious contribution beyond that. Keeping ``motif_spherical_position``
    active with a meaningful nonzero weight alongside this potential keeps
    ``r_cur`` close to ``r_ref`` and is effectively a prerequisite for a
    trustworthy signal here, not an optional nicety.
    """

    def __init__(
        self,
        weight: float = 1.0,
        sigma: float = 3.0,
        motif_indices: list | None = None,
        motif_axis_weights: list | None = None,
        origin_atom_filter: str = "real",
        normalize_by_rho2: bool = False,
        parallel_transport_frame: bool = False,
        eps: float = 1e-6,
    ):
        super().__init__(weight)
        if sigma <= 0.0:
            raise ValueError("motif_forbidden_radial_orientation sigma must be positive")
        self.sigma = float(sigma)
        # Opt-in (default False = original behaviour). When True, each motif's
        # squared-deviation loss is divided by <rho^2> — the mean squared
        # tangential (perpendicular-to-radial) distance of its reference atoms.
        # That is the motif-size scale factor S^2 which otherwise makes `sigma`
        # a meaningless Angstrom^2 value; after normalisation mean_loss is
        # dimensionless and ~[0,4] for the parallel<->antiparallel rotation, so
        # `sigma` becomes an angular width (~1) and the potential exerts a real
        # gradient once the motif nears its correct angular position instead of
        # saturating to zero. See MotifForbiddenRadialOrientationCompact notes.
        self.normalize_by_rho2 = bool(normalize_by_rho2)
        # Opt-in (default False = original behaviour). When True, the current
        # radial frame is the reference frame parallel-transported along
        # r_ref -> r_cur, rather than rebuilt from world axes. This removes the
        # angular-position contamination of the orientation loss (see
        # _min_rotation_matrix_batched), so the loss/force reflect only the
        # genuine orientation and the potential can act before the angular
        # position has settled — without a spherical-position pin.
        self.parallel_transport_frame = bool(parallel_transport_frame)
        self.motif_indices: list | None = (
            None if motif_indices is None else [int(i) for i in motif_indices]
        )
        self.motif_axis_weights: list = (
            list(motif_axis_weights) if motif_axis_weights is not None else []
        )
        valid_origin_filters = ("real", "potential", "guide", "motif", "all")
        if origin_atom_filter not in valid_origin_filters:
            raise ValueError(
                "motif_forbidden_radial_orientation origin_atom_filter must be one of "
                f"{valid_origin_filters}"
            )
        self.origin_atom_filter = origin_atom_filter
        self.eps = float(eps)
        self.skip_reason: str | None = None
        self.skip_detail: dict | None = None

    def _selected_motif_indices(self, n_blocks: int) -> range | list:
        if self.motif_indices is None:
            return range(n_blocks)
        return [i for i in self.motif_indices if 0 <= i < n_blocks]

    def _axis_weights_tensor(
        self,
        motif_i: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """[3] per-axis loss weights for motif_i. Default [1, 1, 1] (full constraint).

        Same semantics as MotifRadialOrientationPotential._axis_weights_tensor.
        """
        if motif_i < len(self.motif_axis_weights):
            aw = self.motif_axis_weights[motif_i]
            if hasattr(aw, "__len__") and len(aw) >= 3:
                return torch.tensor(
                    [float(aw[0]), float(aw[1]), float(aw[2])],
                    device=device,
                    dtype=dtype,
                )
        return torch.ones(3, device=device, dtype=dtype)

    def compute(self, xyz, masks, metadata):
        self.skip_reason = None
        self.skip_detail = None

        motif_blocks = _motif_distance_blocks(masks, metadata, xyz.device)
        self.skip_detail = {
            "n_motif_blocks": len(motif_blocks),
            "motif_indices": self.motif_indices,
            "origin_atom_filter": self.origin_atom_filter,
        }

        if len(motif_blocks) == 0:
            self.skip_reason = "no_motif_blocks"
            return xyz.new_zeros(())

        selected_indices = self._selected_motif_indices(len(motif_blocks))
        if not selected_indices:
            self.skip_reason = "no_selected_motif_indices"
            return xyz.new_zeros(())

        # Protein COM for this step — detached so the potential cannot push
        # atoms toward or away from the COM (no radial-distance gradient).
        origin_mask = _pose_origin_atom_mask(
            self.origin_atom_filter, masks, xyz.device
        )
        current_com = _masked_current_com(xyz, origin_mask)
        if current_com is None:
            self.skip_reason = "current_com_missing"
            return xyz.new_zeros(())
        current_com = current_com.detach()  # [D, 3]

        # Reference global centre = mean of all motif-block reference centres,
        # exactly as MotifRadialOrientationPotential defines it.
        ref_centers: list[torch.Tensor] = []
        for block_mask in motif_blocks:
            rc = _motif_block_reference_com(xyz, masks, metadata, block_mask)
            ref_centers.append(rc)
        valid_ref_centers = [c for c in ref_centers if c is not None]
        if not valid_ref_centers:
            self.skip_reason = "no_motif_reference_centers"
            return xyz.new_zeros(())
        c_ref_global = torch.stack(valid_ref_centers, dim=0).mean(dim=0)  # [3]

        flip = xyz.new_tensor([1.0, -1.0, -1.0])  # 180-deg rotation about local axis 0 (r)

        total_loss = xyz.new_zeros(())
        n_active = 0

        for motif_i in selected_indices:
            block_mask = motif_blocks[motif_i]
            current_xyz_i, ref_xyz_i = _motif_block_current_and_reference_xyz(
                xyz, masks, metadata, block_mask
            )
            if current_xyz_i is None or ref_xyz_i is None or ref_xyz_i.shape[0] < 3:
                continue

            ref_com_i = ref_xyz_i.mean(dim=0)  # [3]

            d_ref = ref_com_i - c_ref_global  # [3]
            d_ref_norm = d_ref.norm()
            if d_ref_norm < self.eps:
                continue
            r_ref_i = d_ref / d_ref_norm  # [3] unit vector

            m_cur_i = _motif_block_current_com(xyz, block_mask)  # [D, 3]
            if m_cur_i is None:
                continue

            # Current radial direction — detached so the gradient does NOT
            # push the motif to a specific angular position on the sphere.
            d_cur = m_cur_i - current_com  # [D, 3]
            d_cur_norm = d_cur.norm(dim=-1, keepdim=True).clamp_min(self.eps)
            r_cur_i = (d_cur / d_cur_norm).detach()  # [D, 3]

            A_ref_i = _build_radial_frame(r_ref_i, self.eps)          # [3, 3]
            if self.parallel_transport_frame:
                # Carry the reference tangent axes along the geodesic r_ref->r_cur
                # so A_cur differs from A_ref only by the genuine r-rotation
                # (no world-axis re-seed) -> loss measures pure orientation.
                R_transport = _min_rotation_matrix_batched(r_ref_i, r_cur_i, self.eps)
                A_cur_i = torch.matmul(R_transport, A_ref_i.unsqueeze(0)).detach()  # [D, 3, 3]
            else:
                A_cur_i = _build_radial_frame_batched(r_cur_i, self.eps)  # [D, 3, 3]

            # Express reference coordinates in the local radial frame, then
            # flip 180 degrees about the radial axis to get the forbidden
            # target: negate the two tangential (t1, t2) components.
            ref_centered = ref_xyz_i - ref_com_i  # [N, 3]
            ref_local = ref_centered @ A_ref_i  # [N, 3]
            forbidden_local = ref_local * flip  # [N, 3]

            current_centered = (
                current_xyz_i - current_xyz_i.mean(dim=1, keepdim=True)
            )  # [D, N, 3]
            current_local = current_centered @ A_cur_i  # [D, N, 3]
            diff_local = current_local - forbidden_local.unsqueeze(0)  # [D, N, 3]

            axis_w = self._axis_weights_tensor(motif_i, xyz.device, xyz.dtype)
            per_motif_loss = (diff_local.pow(2) * axis_w).sum(dim=-1).mean()
            if self.normalize_by_rho2:
                # <rho^2>: mean squared tangential (t1, t2) distance of the
                # reference atoms from the radial axis — the S^2 scale factor.
                rho2 = (ref_local[:, 1] ** 2 + ref_local[:, 2] ** 2).mean()
                per_motif_loss = per_motif_loss / rho2.clamp_min(self.eps)
            total_loss = total_loss + per_motif_loss
            n_active += 1

        if n_active == 0:
            self.skip_reason = "no_active_motifs"
            return xyz.new_zeros(())

        mean_loss = total_loss / n_active
        bump = 1.0 - torch.exp(-mean_loss / (2.0 * self.sigma**2))
        return self.weight * bump

    def guide_atom_mask(self, masks, metadata, device):
        motif_blocks = _motif_distance_blocks(masks, metadata, device)
        if not motif_blocks:
            any_mask = next(iter(masks.values()))
            return torch.zeros_like(any_mask, dtype=torch.bool, device=device)
        guide_mask = torch.zeros_like(motif_blocks[0], dtype=torch.bool, device=device)
        for motif_i in self._selected_motif_indices(len(motif_blocks)):
            guide_mask = guide_mask | motif_blocks[motif_i]
        return guide_mask

    def transform_atom_gradient(self, atom_grad, masks, metadata, xyz):
        motif_blocks = _motif_distance_blocks(masks, metadata, xyz.device)
        if not motif_blocks:
            return torch.zeros_like(atom_grad)
        transformed = torch.zeros_like(atom_grad)
        for motif_i in self._selected_motif_indices(len(motif_blocks)):
            block_mask = motif_blocks[motif_i]
            if block_mask.sum().item() < 3:
                continue
            # Project to pure rigid rotation: zero translation, zero deformation.
            transformed[:, block_mask, :] = _project_gradient_to_rigid_body(
                atom_grad[:, block_mask, :],
                xyz[:, block_mask, :],
                allow_translation=False,
                allow_rotation=True,
            )
        return transformed


class MotifForbiddenRadialOrientationCompact(BasePotential):
    """Repel each motif's rotation away from its 180-degree-flipped input pose,
    using a compactly-supported (exactly-flat-beyond-cutoff) bump instead of
    ``MotifForbiddenRadialOrientation``'s Gaussian-tailed one.

    Same forbidden-pose concept as ``MotifForbiddenRadialOrientation`` (see
    that class's docstring for the shared derivation: local frame ``[r, t1,
    t2]``, forbidden target = reference coordinates with tangential axes
    sign-flipped, no receptor-handling code needed, must be paired with
    ``motif_spherical_position`` for the same frame-instability reason
    documented there — inherited unmodified from ``_build_radial_frame``/
    ``_build_radial_frame_batched``, negligible while current/reference radial
    directions stay within ~5-10 degrees, spurious beyond that).

    The difference is entirely in the final shaping step. The Gaussian-tailed
    sibling's ``bump = 1 - exp(-mean_loss / (2*sigma**2))`` only *asymptotically*
    approaches 1 (no gradient) far from the forbidden pose. Once ``sigma`` is
    large enough to matter against the very large ``mean_loss`` values seen
    early in a diffusion trajectory (structure not yet converged), that
    asymptotic shape becomes locally linear in ``mean_loss`` across the entire
    physically-relevant range — and since ``mean_loss(theta) = 2*S^2*(1+cos
    theta)`` (the closed form on the tracked spin angle) has a single global
    maximum at ``theta=0`` (the reference/"correct" pose, not a neighbourhood
    of anything), maximizing it via gradient ascent collapses this potential
    into a disguised version of the attractive ``MotifRadialOrientationPotential``
    — pulling toward one specific orientation, not repelling from a
    neighbourhood while staying neutral elsewhere. No choice of ``sigma``
    avoids this: the same parameter controls both how early the potential can
    activate and how flat it stays, and pushing one erodes the other.

    This class instead uses a quintic "smootherstep" polynomial:

        t    = clamp(mean_loss / cutoff, 0, 1)
        bump = 6*t**5 - 15*t**4 + 10*t**3
        value = weight * bump

    ``bump`` is *exactly* 0 at the forbidden pose and *exactly* 1 for every
    ``mean_loss >= cutoff`` — not asymptotically, exactly, with exactly zero
    gradient throughout that region (the polynomial's derivative,
    ``30*t**2*(t-1)**2``, is algebraically zero at both ``t=0`` and ``t=1``).
    This guarantees no preference is ever expressed between different "far
    from forbidden" orientations, however far apart they are, which is the
    entire point of this class: repel from a neighbourhood of the forbidden
    pose, remain genuinely indifferent everywhere else, never collapse into a
    single-orientation attractor. No ``torch.where``/``eps``-guarding is
    needed anywhere in this formula (unlike a C-infinity mollifier
    alternative built from ``exp(-1/t)``, which would need such guarding to
    avoid NaN gradients at its own domain boundary) — there is no division or
    exponential-of-reciprocal in this construction at all, so there is no
    asymptote to guard against.

    ``cutoff`` is in the same Å^2 units as ``mean_loss`` and has an exact,
    checkable meaning: no motif with ``mean_loss`` at or above this value ever
    receives any gradient from this potential. Since ``mean_loss(theta) =
    2*S^2*(1+cos theta)`` for a motif with tangential spread ``S^2`` (its own
    reference atoms' mean squared distance from the motif-motif axis), you can
    solve this relationship directly for whatever angular exclusion width you
    want around the forbidden pose, given your motifs' own ``S^2``.

    With multiple active motifs (``motif_indices=None``, the default), each
    motif's loss is shaped independently and the *minimum* of the resulting
    per-motif bumps is reported, not their average. Averaging the raw losses
    before shaping (as the Gaussian-tailed sibling does) would create a dead
    zone under a hard cutoff: one motif could sit exactly at its own forbidden
    pose while another is far from its own, and the *average* loss could
    still clear the cutoff, reporting "fully safe" despite one motif genuinely
    being in the excluded zone. Taking the min across per-motif bumps instead
    means the potential reports "unsafe" if *any* single motif is near its own
    forbidden neighbourhood. This does not affect SE(3) invariance: each
    per-motif bump is already invariant on its own (built only from relative
    geometry via the shared frame-construction helpers), and any combination
    of already-invariant scalars — min, mean, whatever — is itself invariant.
    """

    def __init__(
        self,
        weight: float = 1.0,
        cutoff: float = 100.0,
        motif_indices: list | None = None,
        motif_axis_weights: list | None = None,
        origin_atom_filter: str = "real",
        parallel_transport_frame: bool = False,
        eps: float = 1e-6,
    ):
        super().__init__(weight)
        if cutoff <= 0.0:
            raise ValueError(
                "motif_forbidden_radial_orientation_compact cutoff must be positive"
            )
        self.cutoff = float(cutoff)
        self.parallel_transport_frame = bool(parallel_transport_frame)
        self.motif_indices: list | None = (
            None if motif_indices is None else [int(i) for i in motif_indices]
        )
        self.motif_axis_weights: list = (
            list(motif_axis_weights) if motif_axis_weights is not None else []
        )
        valid_origin_filters = ("real", "potential", "guide", "motif", "all")
        if origin_atom_filter not in valid_origin_filters:
            raise ValueError(
                "motif_forbidden_radial_orientation_compact origin_atom_filter must be one of "
                f"{valid_origin_filters}"
            )
        self.origin_atom_filter = origin_atom_filter
        self.eps = float(eps)
        self.skip_reason: str | None = None
        self.skip_detail: dict | None = None

    def _selected_motif_indices(self, n_blocks: int) -> range | list:
        if self.motif_indices is None:
            return range(n_blocks)
        return [i for i in self.motif_indices if 0 <= i < n_blocks]

    def _axis_weights_tensor(
        self,
        motif_i: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """[3] per-axis loss weights for motif_i. Default [1, 1, 1] (full constraint).

        Same semantics as MotifRadialOrientationPotential._axis_weights_tensor.
        """
        if motif_i < len(self.motif_axis_weights):
            aw = self.motif_axis_weights[motif_i]
            if hasattr(aw, "__len__") and len(aw) >= 3:
                return torch.tensor(
                    [float(aw[0]), float(aw[1]), float(aw[2])],
                    device=device,
                    dtype=dtype,
                )
        return torch.ones(3, device=device, dtype=dtype)

    def compute(self, xyz, masks, metadata):
        self.skip_reason = None
        self.skip_detail = None

        motif_blocks = _motif_distance_blocks(masks, metadata, xyz.device)
        self.skip_detail = {
            "n_motif_blocks": len(motif_blocks),
            "motif_indices": self.motif_indices,
            "origin_atom_filter": self.origin_atom_filter,
        }

        if len(motif_blocks) == 0:
            self.skip_reason = "no_motif_blocks"
            return xyz.new_zeros(())

        selected_indices = self._selected_motif_indices(len(motif_blocks))
        if not selected_indices:
            self.skip_reason = "no_selected_motif_indices"
            return xyz.new_zeros(())

        # Protein COM for this step — detached so the potential cannot push
        # atoms toward or away from the COM (no radial-distance gradient).
        origin_mask = _pose_origin_atom_mask(
            self.origin_atom_filter, masks, xyz.device
        )
        current_com = _masked_current_com(xyz, origin_mask)
        if current_com is None:
            self.skip_reason = "current_com_missing"
            return xyz.new_zeros(())
        current_com = current_com.detach()  # [D, 3]

        # Reference global centre = mean of all motif-block reference centres,
        # exactly as MotifRadialOrientationPotential defines it.
        ref_centers: list[torch.Tensor] = []
        for block_mask in motif_blocks:
            rc = _motif_block_reference_com(xyz, masks, metadata, block_mask)
            ref_centers.append(rc)
        valid_ref_centers = [c for c in ref_centers if c is not None]
        if not valid_ref_centers:
            self.skip_reason = "no_motif_reference_centers"
            return xyz.new_zeros(())
        c_ref_global = torch.stack(valid_ref_centers, dim=0).mean(dim=0)  # [3]

        flip = xyz.new_tensor([1.0, -1.0, -1.0])  # 180-deg rotation about local axis 0 (r)

        per_motif_bumps: list[torch.Tensor] = []

        for motif_i in selected_indices:
            block_mask = motif_blocks[motif_i]
            current_xyz_i, ref_xyz_i = _motif_block_current_and_reference_xyz(
                xyz, masks, metadata, block_mask
            )
            if current_xyz_i is None or ref_xyz_i is None or ref_xyz_i.shape[0] < 3:
                continue

            ref_com_i = ref_xyz_i.mean(dim=0)  # [3]

            d_ref = ref_com_i - c_ref_global  # [3]
            d_ref_norm = d_ref.norm()
            if d_ref_norm < self.eps:
                continue
            r_ref_i = d_ref / d_ref_norm  # [3] unit vector

            m_cur_i = _motif_block_current_com(xyz, block_mask)  # [D, 3]
            if m_cur_i is None:
                continue

            # Current radial direction — detached so the gradient does NOT
            # push the motif to a specific angular position on the sphere.
            d_cur = m_cur_i - current_com  # [D, 3]
            d_cur_norm = d_cur.norm(dim=-1, keepdim=True).clamp_min(self.eps)
            r_cur_i = (d_cur / d_cur_norm).detach()  # [D, 3]

            A_ref_i = _build_radial_frame(r_ref_i, self.eps)          # [3, 3]
            if self.parallel_transport_frame:
                R_transport = _min_rotation_matrix_batched(r_ref_i, r_cur_i, self.eps)
                A_cur_i = torch.matmul(R_transport, A_ref_i.unsqueeze(0)).detach()  # [D, 3, 3]
            else:
                A_cur_i = _build_radial_frame_batched(r_cur_i, self.eps)  # [D, 3, 3]

            # Express reference coordinates in the local radial frame, then
            # flip 180 degrees about the radial axis to get the forbidden
            # target: negate the two tangential (t1, t2) components.
            ref_centered = ref_xyz_i - ref_com_i  # [N, 3]
            ref_local = ref_centered @ A_ref_i  # [N, 3]
            forbidden_local = ref_local * flip  # [N, 3]

            current_centered = (
                current_xyz_i - current_xyz_i.mean(dim=1, keepdim=True)
            )  # [D, N, 3]
            current_local = current_centered @ A_cur_i  # [D, N, 3]
            diff_local = current_local - forbidden_local.unsqueeze(0)  # [D, N, 3]

            axis_w = self._axis_weights_tensor(motif_i, xyz.device, xyz.dtype)
            loss_i = (diff_local.pow(2) * axis_w).sum(dim=-1).mean()

            t_i = (loss_i / self.cutoff).clamp(min=0.0, max=1.0)
            bump_i = t_i * t_i * t_i * (10.0 + t_i * (-15.0 + 6.0 * t_i))  # quintic smootherstep
            per_motif_bumps.append(bump_i)

        if not per_motif_bumps:
            self.skip_reason = "no_active_motifs"
            return xyz.new_zeros(())

        bump = torch.stack(per_motif_bumps).min()
        return self.weight * bump

    def guide_atom_mask(self, masks, metadata, device):
        motif_blocks = _motif_distance_blocks(masks, metadata, device)
        if not motif_blocks:
            any_mask = next(iter(masks.values()))
            return torch.zeros_like(any_mask, dtype=torch.bool, device=device)
        guide_mask = torch.zeros_like(motif_blocks[0], dtype=torch.bool, device=device)
        for motif_i in self._selected_motif_indices(len(motif_blocks)):
            guide_mask = guide_mask | motif_blocks[motif_i]
        return guide_mask

    def transform_atom_gradient(self, atom_grad, masks, metadata, xyz):
        motif_blocks = _motif_distance_blocks(masks, metadata, xyz.device)
        if not motif_blocks:
            return torch.zeros_like(atom_grad)
        transformed = torch.zeros_like(atom_grad)
        for motif_i in self._selected_motif_indices(len(motif_blocks)):
            block_mask = motif_blocks[motif_i]
            if block_mask.sum().item() < 3:
                continue
            # Project to pure rigid rotation: zero translation, zero deformation.
            transformed[:, block_mask, :] = _project_gradient_to_rigid_body(
                atom_grad[:, block_mask, :],
                xyz[:, block_mask, :],
                allow_translation=False,
                allow_rotation=True,
            )
        return transformed


class SymmetryAwareMotifDistance(BasePotential):
    """Run motif-distance restraints independently inside each symmetry copy.

    ``motif_pairs`` is a list of local motif block index pairs, e.g.
    ``[[0, 1], [1, 2]]``.  Each pair is evaluated separately for every
    symmetric subunit using motif blocks that belong to that subunit.
    """

    def __init__(
        self,
        weight: float = 1.0,
        motif_pairs: list | None = None,
        motif_i: int = 0,
        motif_j: int = 1,
        target_distance: float = 10.0,
        target_distances: list | None = None,
        reduction: str = "sum",
    ):
        super().__init__(weight)
        self.motif_pairs = (
            [tuple(int(x) for x in pair[:2]) for pair in motif_pairs]
            if motif_pairs is not None
            else [(int(motif_i), int(motif_j))]
        )
        self.target_distance = float(target_distance)
        self.target_distances = (
            [float(x) for x in target_distances]
            if target_distances is not None
            else []
        )
        self.reduction = _validate_symmetry_reduction(reduction)

    def _target_distance(self, pair_idx: int) -> float:
        if pair_idx < len(self.target_distances):
            return self.target_distances[pair_idx]
        return self.target_distance

    def compute(self, xyz, masks, metadata):
        subunits = _symmetry_subunit_motif_blocks(masks, metadata, xyz.device)
        total_loss = xyz.new_zeros(())
        n_active = 0
        for local_blocks in subunits:
            for pair_idx, (motif_i, motif_j) in enumerate(self.motif_pairs):
                if motif_i >= len(local_blocks) or motif_j >= len(local_blocks):
                    continue
                motif_a = xyz[:, local_blocks[motif_i], :]
                motif_b = xyz[:, local_blocks[motif_j], :]
                if motif_a.shape[1] == 0 or motif_b.shape[1] == 0:
                    continue
                dist = (motif_a.mean(dim=1) - motif_b.mean(dim=1)).norm(dim=-1)
                target = self._target_distance(pair_idx)
                total_loss = total_loss + ((dist - target) ** 2).mean()
                n_active += 1
        if n_active == 0:
            return xyz.new_zeros(())
        return _weighted_symmetry_loss(self.weight, total_loss, n_active, self.reduction)

    def guide_atom_mask(self, masks, metadata, device):
        return _symmetry_aware_selected_motif_mask(
            masks, metadata, device, self.motif_pairs
        )

    def instance_guide_masks(self, masks, metadata, device):
        return _symmetry_aware_pair_instance_masks(
            masks, metadata, device, self.motif_pairs
        )

    def transform_atom_gradient(self, atom_grad, masks, metadata, xyz):
        selected_blocks = _symmetry_aware_selected_blocks(
            masks, metadata, xyz.device, self.motif_pairs
        )
        return _rigidize_blocks_translation(atom_grad, selected_blocks)


class SymmetryAwareMotifCenterDistance(BasePotential):
    """Distance from each subunit motif to the symmetry center/axis."""

    def __init__(
        self,
        weight: float = 1.0,
        target_distances: list | None = None,
        target_distance: float = 10.0,
        center: list | None = None,
        center_type: str = "axis",
        axis: list | None = None,
        reduction: str = "sum",
    ):
        super().__init__(weight)
        self.target_distances = (
            [float(x) for x in target_distances]
            if target_distances is not None
            else []
        )
        self.target_distance = float(target_distance)
        self.center = [float(x) for x in center] if center is not None else [0.0, 0.0, 0.0]
        self.center_type = center_type
        self.axis = [float(x) for x in axis] if axis is not None else [0.0, 0.0, 1.0]
        self.reduction = _validate_symmetry_reduction(reduction)

    def _target_distance(self, motif_i: int) -> float:
        if motif_i < len(self.target_distances):
            return self.target_distances[motif_i]
        return self.target_distance

    def compute(self, xyz, masks, metadata):
        return _symmetry_aware_center_distance_compute(
            self,
            xyz,
            masks,
            metadata,
            center_kind="symmetry",
        )

    def guide_atom_mask(self, masks, metadata, device):
        return _symmetry_aware_all_motif_mask(masks, metadata, device)

    def instance_guide_masks(self, masks, metadata, device):
        return _symmetry_aware_all_blocks(masks, metadata, device)

    def transform_atom_gradient(self, atom_grad, masks, metadata, xyz):
        return _rigidize_blocks_translation(
            atom_grad, _symmetry_aware_all_blocks(masks, metadata, xyz.device)
        )


class SymmetryAwareMotifCOMDistance(SymmetryAwareMotifCenterDistance):
    """Distance from each subunit motif to the current protein COM."""

    def __init__(
        self,
        weight: float = 1.0,
        target_distances: list | None = None,
        target_distance: float = 10.0,
        origin_atom_filter: str = "real",
        reduction: str = "sum",
    ):
        super().__init__(
            weight=weight,
            target_distances=target_distances,
            target_distance=target_distance,
            center_type="point",
            reduction=reduction,
        )
        self.origin_atom_filter = origin_atom_filter

    def compute(self, xyz, masks, metadata):
        return _symmetry_aware_center_distance_compute(
            self,
            xyz,
            masks,
            metadata,
            center_kind="com",
        )


class SymmetryAwareMotifRadialPosition(BasePotential):
    """Keep each motif's radial position relative to the symmetry center/axis."""

    def __init__(
        self,
        weight: float = 1.0,
        motif_offsets: list | None = None,
        center: list | None = None,
        center_type: str = "axis",
        axis: list | None = None,
        eps: float = 1e-6,
        reduction: str = "sum",
    ):
        super().__init__(weight)
        self.motif_offsets = list(motif_offsets) if motif_offsets is not None else []
        self.center = [float(x) for x in center] if center is not None else [0.0, 0.0, 0.0]
        self.center_type = center_type
        self.axis = [float(x) for x in axis] if axis is not None else [0.0, 0.0, 1.0]
        self.eps = float(eps)
        self.reduction = _validate_symmetry_reduction(reduction)

    def _offset_matrix(self, motif_i: int, device: torch.device, dtype: torch.dtype):
        if motif_i < len(self.motif_offsets):
            angles = self.motif_offsets[motif_i]
            if hasattr(angles, "__len__") and len(angles) >= 3:
                return _euler_rotation_matrix_deg(
                    float(angles[0]),
                    float(angles[1]),
                    float(angles[2]),
                    device=device,
                    dtype=dtype,
                )
        return torch.eye(3, device=device, dtype=dtype)

    def compute(self, xyz, masks, metadata):
        return _symmetry_aware_radial_position_compute(
            self, xyz, masks, metadata, center_kind="symmetry"
        )

    def guide_atom_mask(self, masks, metadata, device):
        return _symmetry_aware_all_motif_mask(masks, metadata, device)

    def instance_guide_masks(self, masks, metadata, device):
        return _symmetry_aware_all_blocks(masks, metadata, device)

    def transform_atom_gradient(self, atom_grad, masks, metadata, xyz):
        return _rigidize_blocks_translation(
            atom_grad, _symmetry_aware_all_blocks(masks, metadata, xyz.device)
        )


class SymmetryAwareMotifCOMRadialPosition(SymmetryAwareMotifRadialPosition):
    """Keep each motif's radial position relative to the protein COM."""

    def __init__(
        self,
        weight: float = 1.0,
        motif_offsets: list | None = None,
        origin_atom_filter: str = "real",
        eps: float = 1e-6,
        reduction: str = "sum",
    ):
        super().__init__(
            weight=weight,
            motif_offsets=motif_offsets,
            center_type="point",
            eps=eps,
            reduction=reduction,
        )
        self.origin_atom_filter = origin_atom_filter

    def compute(self, xyz, masks, metadata):
        return _symmetry_aware_radial_position_compute(
            self, xyz, masks, metadata, center_kind="com"
        )


class SymmetryAwareMotifRadialOrientation(MotifRadialOrientationPotential):
    """Run radial-orientation restraints per symmetry copy around symmetry center."""

    def __init__(
        self,
        weight: float = 1.0,
        motif_offsets: list | None = None,
        motif_axis_weights: list | None = None,
        center: list | None = None,
        center_type: str = "axis",
        axis: list | None = None,
        eps: float = 1e-6,
        reduction: str = "sum",
    ):
        super().__init__(
            weight=weight,
            motif_offsets=motif_offsets,
            motif_axis_weights=motif_axis_weights,
            origin_atom_filter="real",
            eps=eps,
        )
        self.center = [float(x) for x in center] if center is not None else [0.0, 0.0, 0.0]
        self.center_type = center_type
        self.axis = [float(x) for x in axis] if axis is not None else [0.0, 0.0, 1.0]
        self.reduction = _validate_symmetry_reduction(reduction)

    def compute(self, xyz, masks, metadata):
        return _symmetry_aware_radial_orientation_compute(
            self, xyz, masks, metadata, center_kind="symmetry"
        )

    def guide_atom_mask(self, masks, metadata, device):
        return _symmetry_aware_all_motif_mask(masks, metadata, device)

    def instance_guide_masks(self, masks, metadata, device):
        return _symmetry_aware_all_blocks(masks, metadata, device)

    def transform_atom_gradient(self, atom_grad, masks, metadata, xyz):
        all_blocks = _symmetry_aware_all_blocks(masks, metadata, xyz.device)
        transformed = torch.zeros_like(atom_grad)
        for block_mask in all_blocks:
            if block_mask.sum().item() < 2:
                continue
            transformed[:, block_mask, :] = _project_gradient_to_rigid_body(
                atom_grad[:, block_mask, :],
                xyz[:, block_mask, :],
                allow_translation=False,
                allow_rotation=True,
            )
        return transformed


class SymmetryAwareMotifCOMRadialOrientation(SymmetryAwareMotifRadialOrientation):
    """Run radial-orientation restraints per symmetry copy around protein COM."""

    def __init__(
        self,
        weight: float = 1.0,
        motif_offsets: list | None = None,
        motif_axis_weights: list | None = None,
        origin_atom_filter: str = "real",
        eps: float = 1e-6,
        reduction: str = "sum",
    ):
        super().__init__(
            weight=weight,
            motif_offsets=motif_offsets,
            motif_axis_weights=motif_axis_weights,
            center_type="point",
            eps=eps,
            reduction=reduction,
        )
        self.origin_atom_filter = origin_atom_filter

    def compute(self, xyz, masks, metadata):
        return _symmetry_aware_radial_orientation_compute(
            self, xyz, masks, metadata, center_kind="com"
        )


class SymmetryAwareMotifAxisPosition(BasePotential):
    """Spherical angular position (theta, phi) of each subunit's motif COM measured in
    the subunit's own local frame relative to the symmetry center.

    Coordinates (both in degrees):
      theta — signed elevation from the equatorial plane
              (0 = same level as symmetry center, + toward +axis, - toward -axis).
      phi   — azimuthal angle in the plane perpendicular to the axis, measured
              relative to the centerline of that symmetry instance.  A single
              phi target therefore means the same thing for every Cn copy.
              Positive phi moves toward the next subunit; negative phi moves
              toward the previous subunit.  Values are compared modulo 360°.

    The COM is expressed in the local frame of each subunit (i.e. the atom
    coordinates are transformed by the inverse of the subunit symmetry frame),
    then phi is shifted so phi=0 sits in the middle of the subunit's symmetry
    wedge instead of on the frame boundary.  The same ``target_theta`` /
    ``target_phi`` applies identically to all Cn copies.  For hetero symmetry,
    supply one ``[theta]`` or ``[theta, phi]`` entry per subunit in
    ``target_positions`` to override independently.

    Radial distance is intentionally not constrained here — use
    ``symmetry_motif_center_distance`` for that.
    """

    def __init__(
        self,
        weight: float = 1.0,
        target_theta: float | None = None,
        target_phi: float | None = None,
        target_positions: list | None = None,
        weight_theta: float = 1.0,
        weight_phi: float = 1.0,
        center: list | None = None,
        axis: list | None = None,
        motif_i: int = 0,
        eps: float = 1e-6,
        reduction: str = "sum",
    ):
        super().__init__(weight)
        self.target_theta = float(target_theta) if target_theta is not None else None
        self.target_phi = float(target_phi) if target_phi is not None else None
        self.target_positions = list(target_positions) if target_positions is not None else []
        self.weight_theta = float(weight_theta)
        self.weight_phi = float(weight_phi)
        self.center = [float(x) for x in center] if center is not None else [0.0, 0.0, 0.0]
        self.axis = [float(x) for x in axis] if axis is not None else [0.0, 0.0, 1.0]
        self.motif_i = int(motif_i)
        self.eps = float(eps)
        self.reduction = _validate_symmetry_reduction(reduction)

    def _get_target_theta(self, subunit_idx: int) -> float | None:
        if subunit_idx < len(self.target_positions):
            pos = self.target_positions[subunit_idx]
            if hasattr(pos, "__len__") and len(pos) >= 1:
                return float(pos[0])
        return self.target_theta

    def _get_target_phi(self, subunit_idx: int) -> float | None:
        if subunit_idx < len(self.target_positions):
            pos = self.target_positions[subunit_idx]
            if hasattr(pos, "__len__") and len(pos) >= 2:
                return float(pos[1])
        return self.target_phi

    def _subunit_rotation(self, subunit_idx, transform_ids, metadata, device, dtype):
        sym_transform = metadata.get("sym_transform", {})
        if not sym_transform or subunit_idx >= len(transform_ids):
            return None
        tid = int(transform_ids[subunit_idx].item() if hasattr(transform_ids[subunit_idx], "item") else transform_ids[subunit_idx])
        if tid not in sym_transform and str(tid) not in sym_transform:
            return None
        key = tid if tid in sym_transform else str(tid)
        return sym_transform[key][0].to(device=device, dtype=dtype)

    def _phi_midline_offset(self, n_instances: int, device, dtype):
        if n_instances <= 1:
            return torch.tensor(0.0, device=device, dtype=dtype)
        return torch.tensor(180.0 / float(n_instances), device=device, dtype=dtype)

    def compute(self, xyz, masks, metadata):
        subunits = _symmetry_subunit_motif_blocks(masks, metadata, xyz.device)
        if not subunits:
            return xyz.new_zeros(())

        # Ordered transform IDs — mirrors the ordering in _symmetry_subunit_motif_blocks
        transform_ids = []
        if "sym_transform_id" in metadata:
            sym_tid = metadata["sym_transform_id"].to(device=xyz.device, dtype=torch.long)
            sym_eid = metadata.get("sym_entity_id")
            if sym_eid is not None:
                valid = sym_eid.to(device=xyz.device, dtype=torch.long) != -1
            else:
                valid = torch.ones_like(sym_tid, dtype=torch.bool)
            valid = valid & (sym_tid != -1)
            transform_ids = torch.unique(sym_tid[valid]).sort().values

        center = torch.tensor(self.center, device=xyz.device, dtype=xyz.dtype)
        axis = torch.tensor(self.axis, device=xyz.device, dtype=xyz.dtype)
        axis = axis / axis.norm().clamp_min(self.eps)

        total_loss = xyz.new_zeros(())
        n_active = 0

        for subunit_idx, local_blocks in enumerate(subunits):
            if self.motif_i >= len(local_blocks):
                continue
            com = _motif_block_current_com(xyz, local_blocks[self.motif_i])
            if com is None:
                continue

            # Vector from symmetry center to motif COM [B, 3], then transform to local frame.
            vec = com - center.unsqueeze(0)
            R_i = self._subunit_rotation(subunit_idx, transform_ids, metadata, xyz.device, xyz.dtype)
            # Coordinates are row vectors. R_i maps local->global as x @ R_i.T,
            # so the inverse/global->local operation is x @ R_i.
            vec_local = vec @ R_i if R_i is not None else vec

            # Elevation axis in local frame (invariant under Cn/Dn Z-rotations, correct otherwise)
            axis_local = (axis @ R_i) if R_i is not None else axis

            r = vec_local.norm(dim=-1).clamp_min(self.eps)  # [B]

            tgt_theta = self._get_target_theta(subunit_idx)
            if tgt_theta is not None:
                sin_elevation = (vec_local * axis_local).sum(dim=-1) / r
                theta_deg = torch.rad2deg(
                    torch.asin(sin_elevation.clamp(-1.0 + self.eps, 1.0 - self.eps))
                )
                total_loss = total_loss + (self.weight_theta * (theta_deg - tgt_theta) ** 2).mean()
                n_active += 1

            tgt_phi = self._get_target_phi(subunit_idx)
            if tgt_phi is not None:
                phi_deg = torch.rad2deg(torch.atan2(vec_local[:, 1], vec_local[:, 0]))
                phi_deg = phi_deg - self._phi_midline_offset(
                    len(transform_ids) if len(transform_ids) > 0 else len(subunits),
                    xyz.device,
                    xyz.dtype,
                )
                dphi = torch.remainder(phi_deg - tgt_phi + 180.0, 360.0) - 180.0
                total_loss = total_loss + (self.weight_phi * dphi ** 2).mean()
                n_active += 1

        if n_active == 0:
            return xyz.new_zeros(())
        return _weighted_symmetry_loss(self.weight, total_loss, n_active, self.reduction)

    def guide_atom_mask(self, masks, metadata, device):
        return _symmetry_aware_all_motif_mask(masks, metadata, device)

    def instance_guide_masks(self, masks, metadata, device):
        return _symmetry_aware_all_blocks(masks, metadata, device)

    def transform_atom_gradient(self, atom_grad, masks, metadata, xyz):
        return _rigidize_blocks_translation(
            atom_grad, _symmetry_aware_all_blocks(masks, metadata, xyz.device)
        )


class SymmetryAwareInterInstanceMotifDistance(BasePotential):
    """Penalise pairwise distances between motif COMs across different symmetry subunits.

    By default, homomer/normal symmetry only scores neighbouring subunit pairs:
    (0,1), (1,2), ..., (N-1,0).  This avoids over-constraining all pairwise
    distances in one cyclic ring.

    For hetero or hand-specified layouts, use ``target_pairs`` as a list of
    dictionaries.  Each dict defines the two subunit/motif instances and the
    distance target, e.g. ``{subunit_i: 0, motif_i: 0, subunit_j: 2,
    motif_j: 1, target_distance: 30.0}``.
    """

    def __init__(
        self,
        weight: float = 1.0,
        target_distance: float = 20.0,
        target_distances: list | None = None,
        target_pairs: list | None = None,
        motif_i: int = 0,
        motif_j: int | None = None,
        neighbor_only: bool = True,
        reduction: str = "sum",
    ):
        super().__init__(weight)
        self.target_distance = float(target_distance)
        self.target_distances = (
            [float(x) for x in target_distances] if target_distances is not None else []
        )
        self.target_pairs = list(target_pairs) if target_pairs is not None else []
        self.motif_i = int(motif_i)
        self.motif_j = int(motif_j) if motif_j is not None else self.motif_i
        self.neighbor_only = bool(neighbor_only)
        self.reduction = _validate_symmetry_reduction(reduction)

    def _get_pair_target(self, pair_idx: int) -> float:
        if pair_idx < len(self.target_distances):
            return self.target_distances[pair_idx]
        return self.target_distance

    def _default_subunit_pairs(self, n: int) -> list[tuple[int, int, int, int, float]]:
        if n < 2:
            return []
        subunit_pairs: list[tuple[int, int]]
        if self.neighbor_only:
            subunit_pairs = [(i, (i + 1) % n) for i in range(n)]
            if n == 2:
                subunit_pairs = [(0, 1)]
        else:
            subunit_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        return [
            (si, sj, self.motif_i, self.motif_j, self._get_pair_target(pair_idx))
            for pair_idx, (si, sj) in enumerate(subunit_pairs)
        ]

    def _explicit_subunit_pairs(self) -> list[tuple[int, int, int, int, float]]:
        pairs = []
        for pair in self.target_pairs:
            if not isinstance(pair, dict):
                raise ValueError(
                    "symmetry_motif_inter_instance_distance target_pairs entries "
                    "must be dictionaries"
                )
            si = int(pair.get("subunit_i", pair.get("instance_i", 0)))
            sj = int(pair.get("subunit_j", pair.get("instance_j", 1)))
            mi = int(pair.get("motif_i", self.motif_i))
            mj = int(pair.get("motif_j", self.motif_j))
            if "target_distance" not in pair and "distance" not in pair:
                raise ValueError(
                    "symmetry_motif_inter_instance_distance target_pairs entries "
                    "must include target_distance"
                )
            target = float(pair.get("target_distance", pair.get("distance")))
            pairs.append((si, sj, mi, mj, target))
        return pairs

    def compute(self, xyz, masks, metadata):
        subunits = _symmetry_subunit_motif_blocks(masks, metadata, xyz.device)
        if len(subunits) < 2:
            return xyz.new_zeros(())
        total_loss = xyz.new_zeros(())
        n_active = 0
        pair_specs = (
            self._explicit_subunit_pairs()
            if self.target_pairs
            else self._default_subunit_pairs(len(subunits))
        )
        for si, sj, motif_i, motif_j, target in pair_specs:
            if si < 0 or sj < 0 or si >= len(subunits) or sj >= len(subunits):
                continue
            local_i = subunits[si]
            local_j = subunits[sj]
            if motif_i < 0 or motif_j < 0 or motif_i >= len(local_i) or motif_j >= len(local_j):
                continue
            com_i = _motif_block_current_com(xyz, local_i[motif_i])
            com_j = _motif_block_current_com(xyz, local_j[motif_j])
            if com_i is None or com_j is None:
                continue
            dist = (com_i - com_j).norm(dim=-1)
            total_loss = total_loss + ((dist - target) ** 2).mean()
            n_active += 1
        if n_active == 0:
            return xyz.new_zeros(())
        return _weighted_symmetry_loss(self.weight, total_loss, n_active, self.reduction)

    def guide_atom_mask(self, masks, metadata, device):
        return _symmetry_aware_all_motif_mask(masks, metadata, device)

    def instance_guide_masks(self, masks, metadata, device):
        return _symmetry_aware_all_blocks(masks, metadata, device)

    def transform_atom_gradient(self, atom_grad, masks, metadata, xyz):
        return _rigidize_blocks_translation(
            atom_grad, _symmetry_aware_all_blocks(masks, metadata, xyz.device)
        )


def _symmetry_subunit_motif_blocks(
    masks: dict[str, torch.Tensor],
    metadata: dict,
    device: torch.device,
) -> list[list[torch.Tensor]]:
    motif_blocks = _motif_distance_blocks(masks, metadata, device)
    if not motif_blocks:
        return []
    if "sym_transform_id" not in metadata:
        return [motif_blocks]

    sym_transform_id = metadata["sym_transform_id"].to(device=device, dtype=torch.long)
    sym_entity_id = metadata.get("sym_entity_id")
    if sym_entity_id is not None:
        sym_entity_id = sym_entity_id.to(device=device, dtype=torch.long)
        valid_sym = sym_entity_id != -1
    else:
        valid_sym = torch.ones_like(sym_transform_id, dtype=torch.bool)
    valid_sym = valid_sym & (sym_transform_id != -1)

    transform_ids = torch.unique(sym_transform_id[valid_sym]).sort().values
    subunits: list[list[torch.Tensor]] = [[] for _ in transform_ids.tolist()]
    assigned_blocks = [False] * len(motif_blocks)
    for subunit_idx, transform_id in enumerate(transform_ids):
        subunit_atom_mask = valid_sym & (sym_transform_id == transform_id)
        for block_idx, block_mask in enumerate(motif_blocks):
            local_block = block_mask & subunit_atom_mask
            if local_block.any():
                subunits[subunit_idx].append(local_block)
                assigned_blocks[block_idx] = True

    # Hetero pseudo-symmetry can append independent unsymmetrized motifs with
    # FIXED_TRANSFORM_ID.  They are still intended to act as one motif instance
    # per subunit, so assign transform-less motif blocks by contig order instead
    # of dropping them from symmetry-aware potentials.
    if subunits:
        unassigned = [
            block_mask
            for block_idx, block_mask in enumerate(motif_blocks)
            if not assigned_blocks[block_idx]
        ]
        if unassigned:
            for block_idx, block_mask in enumerate(unassigned):
                subunits[block_idx % len(subunits)].append(block_mask)
    subunits = [subunit for subunit in subunits if subunit]
    return subunits if subunits else [motif_blocks]


def _validate_symmetry_reduction(reduction: str) -> str:
    valid = ("sum", "mean")
    if reduction not in valid:
        raise ValueError(f"symmetry-aware potential reduction must be one of {valid}")
    return reduction


def _weighted_symmetry_loss(
    weight: float,
    total_loss: torch.Tensor,
    n_active: int,
    reduction: str,
) -> torch.Tensor:
    if reduction == "mean":
        total_loss = total_loss / max(int(n_active), 1)
    return -weight * total_loss


def _symmetry_aware_selected_blocks(
    masks: dict[str, torch.Tensor],
    metadata: dict,
    device: torch.device,
    motif_pairs: list[tuple[int, int]],
) -> list[torch.Tensor]:
    selected = []
    for local_blocks in _symmetry_subunit_motif_blocks(masks, metadata, device):
        for motif_i, motif_j in motif_pairs:
            for motif_idx in (motif_i, motif_j):
                if 0 <= motif_idx < len(local_blocks):
                    selected.append(local_blocks[motif_idx])
    return selected


def _symmetry_aware_selected_motif_mask(
    masks: dict[str, torch.Tensor],
    metadata: dict,
    device: torch.device,
    motif_pairs: list[tuple[int, int]],
) -> torch.Tensor:
    any_mask = next(iter(masks.values()))
    guide_mask = torch.zeros_like(any_mask, dtype=torch.bool, device=device)
    for block_mask in _symmetry_aware_selected_blocks(
        masks, metadata, device, motif_pairs
    ):
        guide_mask |= block_mask
    return guide_mask


def _symmetry_aware_pair_instance_masks(
    masks: dict[str, torch.Tensor],
    metadata: dict,
    device: torch.device,
    motif_pairs: list[tuple[int, int]],
) -> list[torch.Tensor]:
    instance_masks = []
    for local_blocks in _symmetry_subunit_motif_blocks(masks, metadata, device):
        for motif_i, motif_j in motif_pairs:
            any_mask = next(iter(masks.values()))
            instance_mask = torch.zeros_like(any_mask, dtype=torch.bool, device=device)
            if 0 <= motif_i < len(local_blocks):
                instance_mask |= local_blocks[motif_i]
            if 0 <= motif_j < len(local_blocks):
                instance_mask |= local_blocks[motif_j]
            if instance_mask.any():
                instance_masks.append(instance_mask)
    return instance_masks


def _symmetry_aware_all_blocks(
    masks: dict[str, torch.Tensor],
    metadata: dict,
    device: torch.device,
) -> list[torch.Tensor]:
    blocks = []
    for local_blocks in _symmetry_subunit_motif_blocks(masks, metadata, device):
        blocks.extend(local_blocks)
    return blocks


def _symmetry_aware_all_motif_mask(
    masks: dict[str, torch.Tensor],
    metadata: dict,
    device: torch.device,
) -> torch.Tensor:
    any_mask = next(iter(masks.values()))
    guide_mask = torch.zeros_like(any_mask, dtype=torch.bool, device=device)
    for block_mask in _symmetry_aware_all_blocks(masks, metadata, device):
        guide_mask |= block_mask
    return guide_mask


def _symmetry_subunit_atom_masks(
    metadata: dict,
    base_mask: torch.Tensor,
    device: torch.device,
) -> list[torch.Tensor]:
    base_mask = base_mask.to(device=device, dtype=torch.bool)
    if "sym_transform_id" not in metadata:
        return [base_mask]
    sym_transform_id = metadata["sym_transform_id"].to(device=device, dtype=torch.long)
    sym_entity_id = metadata.get("sym_entity_id")
    if sym_entity_id is not None:
        sym_entity_id = sym_entity_id.to(device=device, dtype=torch.long)
        valid_sym = sym_entity_id != -1
    else:
        valid_sym = torch.ones_like(sym_transform_id, dtype=torch.bool)
    valid_sym = valid_sym & (sym_transform_id != -1)
    subunit_masks = []
    for transform_id in torch.unique(sym_transform_id[valid_sym]).sort().values:
        subunit_mask = base_mask & valid_sym & (sym_transform_id == transform_id)
        if subunit_mask.any():
            subunit_masks.append(subunit_mask)
    return subunit_masks if subunit_masks else [base_mask]


def _symmetry_subunit_bridge_masks(
    masks: dict[str, torch.Tensor],
    metadata: dict,
    device: torch.device,
    atom_filter: str,
    include_motif_atoms: bool,
) -> list[torch.Tensor]:
    bridge_mask = _bridge_atom_mask(atom_filter=atom_filter, masks=masks, device=device)
    if not include_motif_atoms:
        bridge_mask = bridge_mask & ~masks["motif_atom_mask"].to(
            device=device, dtype=torch.bool
        )
    return _symmetry_subunit_atom_masks(metadata, bridge_mask, device)


def _symmetry_aware_bridge_guide_mask(
    masks: dict[str, torch.Tensor],
    metadata: dict,
    device: torch.device,
    atom_filter: str,
    include_motif_atoms: bool,
) -> torch.Tensor:
    any_mask = next(iter(masks.values()))
    guide_mask = torch.zeros_like(any_mask, dtype=torch.bool, device=device)
    for bridge_mask in _symmetry_subunit_bridge_masks(
        masks, metadata, device, atom_filter, include_motif_atoms
    ):
        guide_mask |= bridge_mask
    return guide_mask


def _two_motif_bridge_loss(
    xyz: torch.Tensor,
    motif_i_mask: torch.Tensor,
    motif_j_mask: torch.Tensor,
    bridge_mask: torch.Tensor,
    obj,
) -> torch.Tensor | None:
    motif_a = xyz[:, motif_i_mask, :]
    motif_b = xyz[:, motif_j_mask, :]
    if motif_a.shape[1] == 0 or motif_b.shape[1] == 0 or bridge_mask.sum().item() < 1:
        return None

    bridge_xyz = xyz[:, bridge_mask, :]
    com_a = motif_a.mean(dim=1)
    com_b = motif_b.mean(dim=1)
    axis = com_b - com_a
    axis_len = axis.norm(dim=-1).clamp(min=obj.eps)
    axis_unit = axis / axis_len[:, None]

    rel = bridge_xyz - com_a[:, None, :]
    alpha = (rel * axis_unit[:, None, :]).sum(dim=-1) / axis_len[:, None]

    sorted_alpha = torch.sort(alpha, dim=-1).values
    if sorted_alpha.shape[-1] == 1:
        target_alpha = sorted_alpha.new_full(sorted_alpha.shape, 0.5)
    else:
        target_alpha = torch.linspace(
            0.0,
            1.0,
            sorted_alpha.shape[-1],
            device=xyz.device,
            dtype=xyz.dtype,
        )
        target_alpha = target_alpha[None, :].expand_as(sorted_alpha)

    spread_loss = (sorted_alpha - target_alpha).pow(2).mean()
    outside_loss = (
        torch.relu(-alpha).pow(2) + torch.relu(alpha - 1.0).pow(2)
    ).mean()
    projected = com_a[:, None, :] + alpha[:, :, None] * axis[:, None, :]
    radial_dist = (bridge_xyz - projected).norm(dim=-1)
    tube_loss = torch.relu(radial_dist - obj.max_radius).pow(2).mean()
    return (
        obj.spread_weight * spread_loss
        + obj.outside_weight * outside_loss
        + obj.tube_weight * tube_loss
    )


def _ellipsoid_bridge_loss(
    xyz: torch.Tensor,          # [D, L, 3]
    motif_com: torch.Tensor,    # [D, 3]
    subunit_com: torch.Tensor,  # [D, 3]
    com_left: torch.Tensor,     # [D, 3]  left-neighbor subunit COM
    com_right: torch.Tensor,    # [D, 3]  right-neighbor subunit COM
    bridge_mask: torch.Tensor,  # [L]     bool
    obj,
) -> torch.Tensor | None:
    """Loss that distributes bridge atoms evenly inside a subunit ellipsoid.

    Ellipsoid axes (all anchored at subunit_com):
      e1  motif direction        semi-length a = ||motif_com - subunit_com||
      e2  left-border direction  semi-length b = ||com_left  - subunit_com|| / 2
      e3  right-border direction semi-length c = ||com_right - subunit_com|| / 2

    e2 and e3 are Gram–Schmidt orthogonalised against the preceding axes;
    e3 falls back to e1 × e2 when the right-border vector is coplanar with (e1, e2).
    """
    if bridge_mask.sum().item() < 1:
        return None

    eps = obj.eps

    # --- ellipsoid axes -------------------------------------------------------
    v1 = motif_com - subunit_com                        # [D, 3]
    v2 = (com_left  - subunit_com) * 0.5               # border = midpoint → COM
    v3 = (com_right - subunit_com) * 0.5

    a = v1.norm(dim=-1).clamp(min=eps)                  # [D]
    b = v2.norm(dim=-1).clamp(min=eps)
    c = v3.norm(dim=-1).clamp(min=eps)

    e1 = v1 / a[:, None]                                # [D, 3]

    v2_orth = v2 - (v2 * e1).sum(dim=-1, keepdim=True) * e1
    e2 = v2_orth / v2_orth.norm(dim=-1, keepdim=True).clamp(min=eps)

    # e3: orthogonalise v3 against e1 and e2; fall back to e1 × e2
    v3_orth = v3 - (v3 * e1).sum(dim=-1, keepdim=True) * e1
    v3_orth = v3_orth - (v3_orth * e2).sum(dim=-1, keepdim=True) * e2
    v3_norm = v3_orth.norm(dim=-1, keepdim=True).clamp(min=eps)
    e3_cross = torch.linalg.cross(e1, e2, dim=-1)
    e3_cross = e3_cross / e3_cross.norm(dim=-1, keepdim=True).clamp(min=eps)
    degenerate = v3_norm < eps
    e3 = torch.where(degenerate.expand_as(v3_orth), e3_cross, v3_orth / v3_norm)

    # --- project bridge atoms into ellipsoidal coordinates -------------------
    rel = xyz[:, bridge_mask, :] - subunit_com[:, None, :]   # [D, N, 3]

    p1 = (rel * e1[:, None, :]).sum(dim=-1) / a[:, None]     # [D, N]
    p2 = (rel * e2[:, None, :]).sum(dim=-1) / b[:, None]
    p3 = (rel * e3[:, None, :]).sum(dim=-1) / c[:, None]

    d_ellip = (p1.pow(2) + p2.pow(2) + p3.pow(2)).sqrt()    # [D, N]

    # --- spread + boundary losses (bin-midpoint targets) ---------------------
    sorted_d = torch.sort(d_ellip, dim=-1).values
    n = sorted_d.shape[-1]
    target = (torch.arange(n, device=xyz.device, dtype=xyz.dtype) + 0.5) / n
    target = target[None, :].expand_as(sorted_d)

    spread_loss = (sorted_d - target).pow(2).mean()
    # Two-sided: penalise atoms outside the ellipsoid AND atoms collapsed
    # onto the center (d_ellip < 0 not possible, but keeps symmetry with bridge).
    outside_loss = (
        torch.relu(d_ellip - 1.0).pow(2) + torch.relu(-d_ellip).pow(2)
    ).mean()
    return obj.spread_weight * spread_loss + obj.outside_weight * outside_loss


def _single_motif_bridge_loss(
    xyz: torch.Tensor,
    motif_mask: torch.Tensor,
    bridge_mask: torch.Tensor,
    obj,
) -> torch.Tensor | None:
    motif_xyz = xyz[:, motif_mask, :]
    if motif_xyz.shape[1] == 0 or bridge_mask.sum().item() < 1:
        return None
    bridge_xyz = xyz[:, bridge_mask, :]
    motif_com = motif_xyz.mean(dim=1)
    dist = (bridge_xyz - motif_com[:, None, :]).norm(dim=-1)
    radius_fraction = dist / max(obj.max_radius, obj.eps)
    sorted_fraction = torch.sort(radius_fraction, dim=-1).values
    n = sorted_fraction.shape[-1]
    # Bin midpoints: (0.5/n, 1.5/n, ..., (n-0.5)/n).  No atom is targeted to
    # sit on the motif COM (fraction 0) or exactly at max_radius (fraction 1),
    # so the closest atom always has an outward target and the spread is symmetric.
    target_fraction = (
        torch.arange(n, device=xyz.device, dtype=xyz.dtype) + 0.5
    ) / n
    target_fraction = target_fraction[None, :].expand_as(sorted_fraction)
    spread_loss = (sorted_fraction - target_fraction).pow(2).mean()
    # Two-sided boundary: penalise atoms outside max_radius AND atoms sitting
    # directly on the motif COM (radius ≈ 0), mirroring the two-motif bridge.
    outside_loss = (
        torch.relu(radius_fraction - 1.0).pow(2) + torch.relu(-radius_fraction).pow(2)
    ).mean()
    return obj.spread_weight * spread_loss + obj.outside_weight * outside_loss


def _rigidize_blocks_translation(atom_grad, blocks):
    transformed = torch.zeros_like(atom_grad)
    for block_mask in blocks:
        if block_mask.sum().item() == 0:
            continue
        block_translation = atom_grad[:, block_mask, :].mean(dim=1, keepdim=True)
        transformed[:, block_mask, :] = block_translation
    return transformed


def _rigidize_blocks_rotation(atom_grad, blocks, xyz):
    transformed = torch.zeros_like(atom_grad)
    for block_mask in blocks:
        if block_mask.sum().item() < 3:
            continue
        transformed[:, block_mask, :] = _project_gradient_to_rigid_body(
            atom_grad[:, block_mask, :],
            xyz[:, block_mask, :],
            allow_translation=False,
            allow_rotation=True,
        )
    return transformed


def _center_tensor(obj, xyz, masks, metadata, center_kind: str):
    if center_kind == "com":
        origin_mask = _pose_origin_atom_mask(obj.origin_atom_filter, masks, xyz.device)
        center = _masked_current_com(xyz, origin_mask)
        return None if center is None else center.detach()
    center = torch.tensor(obj.center, device=xyz.device, dtype=xyz.dtype)
    return center.unsqueeze(0).expand(xyz.shape[0], -1)


def _reference_center_tensor(obj, xyz, masks, metadata, center_kind: str):
    if center_kind == "com":
        origin_mask = _pose_origin_atom_mask(obj.origin_atom_filter, masks, xyz.device)
        ref_center = _masked_reference_com(xyz, masks, metadata, origin_mask)
        if ref_center is not None:
            return ref_center
    return torch.tensor(obj.center, device=xyz.device, dtype=xyz.dtype)


def _axis_tensor(obj, xyz):
    axis = torch.tensor(obj.axis, device=xyz.device, dtype=xyz.dtype)
    return axis / axis.norm().clamp_min(getattr(obj, "eps", 1e-6))


def _center_vector(points, center, obj, xyz):
    vector = points - center
    if getattr(obj, "center_type", "point") == "axis":
        axis = _axis_tensor(obj, xyz)
        vector = vector - (vector * axis).sum(dim=-1, keepdim=True) * axis
    return vector


def _symmetry_aware_center_distance_compute(obj, xyz, masks, metadata, center_kind):
    subunits = _symmetry_subunit_motif_blocks(masks, metadata, xyz.device)
    if not subunits:
        return xyz.new_zeros(())
    center = _center_tensor(obj, xyz, masks, metadata, center_kind)
    if center is None:
        return xyz.new_zeros(())
    total_loss = xyz.new_zeros(())
    n_active = 0
    for local_blocks in subunits:
        for motif_i, block_mask in enumerate(local_blocks):
            motif_com = _motif_block_current_com(xyz, block_mask)
            if motif_com is None:
                continue
            vector = _center_vector(motif_com, center, obj, xyz)
            dist = vector.norm(dim=-1)
            total_loss = total_loss + ((dist - obj._target_distance(motif_i)) ** 2).mean()
            n_active += 1
    if n_active == 0:
        return xyz.new_zeros(())
    return _weighted_symmetry_loss(obj.weight, total_loss, n_active, obj.reduction)


def _symmetry_aware_radial_position_compute(obj, xyz, masks, metadata, center_kind):
    subunits = _symmetry_subunit_motif_blocks(masks, metadata, xyz.device)
    if not subunits:
        return xyz.new_zeros(())
    center = _center_tensor(obj, xyz, masks, metadata, center_kind)
    ref_center = _reference_center_tensor(obj, xyz, masks, metadata, center_kind)
    if center is None or ref_center is None:
        return xyz.new_zeros(())

    total_loss = xyz.new_zeros(())
    n_active = 0
    for local_blocks in subunits:
        for motif_i, block_mask in enumerate(local_blocks):
            ref_com = _motif_block_reference_com(xyz, masks, metadata, block_mask)
            cur_com = _motif_block_current_com(xyz, block_mask)
            if ref_com is None or cur_com is None:
                continue
            ref_vec = _center_vector(ref_com.unsqueeze(0), ref_center.unsqueeze(0), obj, xyz).squeeze(0)
            if ref_vec.norm() < obj.eps:
                continue
            ref_dir = ref_vec / ref_vec.norm().clamp_min(obj.eps)
            R_offset = obj._offset_matrix(motif_i, xyz.device, xyz.dtype)
            target_dir = _apply_row_rotation(ref_dir.unsqueeze(0), R_offset).squeeze(0)
            target_dir = (target_dir / target_dir.norm().clamp_min(obj.eps)).detach()

            cur_vec = _center_vector(cur_com, center, obj, xyz)
            cur_dir = cur_vec / cur_vec.norm(dim=-1, keepdim=True).clamp_min(obj.eps)
            total_loss = total_loss + (cur_dir - target_dir.unsqueeze(0)).pow(2).sum(dim=-1).mean()
            n_active += 1
    if n_active == 0:
        return xyz.new_zeros(())
    return _weighted_symmetry_loss(obj.weight, total_loss, n_active, obj.reduction)


def _symmetry_aware_radial_orientation_compute(obj, xyz, masks, metadata, center_kind):
    subunits = _symmetry_subunit_motif_blocks(masks, metadata, xyz.device)
    if not subunits:
        return xyz.new_zeros(())
    center = _center_tensor(obj, xyz, masks, metadata, center_kind)
    ref_center = _reference_center_tensor(obj, xyz, masks, metadata, center_kind)
    if center is None or ref_center is None:
        return xyz.new_zeros(())

    total_loss = xyz.new_zeros(())
    n_active = 0
    for local_blocks in subunits:
        for motif_i, block_mask in enumerate(local_blocks):
            current_xyz_i, ref_xyz_i = _motif_block_current_and_reference_xyz(
                xyz, masks, metadata, block_mask
            )
            if current_xyz_i is None or ref_xyz_i is None or ref_xyz_i.shape[0] < 3:
                continue
            ref_com = ref_xyz_i.mean(dim=0)
            cur_com = current_xyz_i.mean(dim=1)

            ref_vec = _center_vector(ref_com.unsqueeze(0), ref_center.unsqueeze(0), obj, xyz).squeeze(0)
            if ref_vec.norm() < obj.eps:
                continue
            ref_dir = ref_vec / ref_vec.norm().clamp_min(obj.eps)

            cur_vec = _center_vector(cur_com, center, obj, xyz)
            cur_dir = (cur_vec / cur_vec.norm(dim=-1, keepdim=True).clamp_min(obj.eps)).detach()

            A_ref = _build_radial_frame(ref_dir, obj.eps)
            A_cur = _build_radial_frame_batched(cur_dir, obj.eps)
            ref_centered = ref_xyz_i - ref_com
            R_offset = obj._offset_matrix(motif_i, xyz.device, xyz.dtype)
            ref_local = ref_centered @ A_ref
            ref_local_offset = ref_local @ R_offset
            current_centered = current_xyz_i - current_xyz_i.mean(dim=1, keepdim=True)
            current_local = current_centered @ A_cur
            axis_w = obj._axis_weights_tensor(motif_i, xyz.device, xyz.dtype)
            total_loss = total_loss + (
                (current_local - ref_local_offset.unsqueeze(0)).pow(2) * axis_w
            ).sum(dim=-1).mean()
            n_active += 1
    if n_active == 0:
        return xyz.new_zeros(())
    return _weighted_symmetry_loss(obj.weight, total_loss, n_active, obj.reduction)


def _motif_blocks(
    masks: dict[str, torch.Tensor],
    metadata: dict,
    device: torch.device,
) -> list[torch.Tensor]:
    """Return atom masks for contiguous motif-token blocks."""
    if "atom_to_token_map" not in metadata:
        return []
    if "motif_atom_mask" not in masks or "potential_atom_mask" not in masks:
        return []

    atom_to_token_map = metadata["atom_to_token_map"].to(
        device=device, dtype=torch.long
    )
    motif_mask = masks["motif_atom_mask"].to(device=device, dtype=torch.bool)
    potential_mask = masks["potential_atom_mask"].to(device=device, dtype=torch.bool)
    motif_atom_mask = motif_mask & potential_mask
    if motif_atom_mask.sum().item() == 0:
        return []

    motif_tokens = torch.unique(atom_to_token_map[motif_atom_mask]).sort().values
    if motif_tokens.numel() == 0:
        return []

    token_runs: list[list[int]] = []
    current_run = [int(motif_tokens[0].item())]
    previous_token = current_run[0]
    for token_tensor in motif_tokens[1:]:
        token = int(token_tensor.item())
        if token == previous_token + 1:
            current_run.append(token)
        else:
            token_runs.append(current_run)
            current_run = [token]
        previous_token = token
    token_runs.append(current_run)

    atom_blocks: list[torch.Tensor] = []
    for token_run in token_runs:
        block_mask = torch.zeros_like(motif_atom_mask)
        for token in token_run:
            block_mask |= atom_to_token_map == token
        atom_blocks.append(block_mask & motif_atom_mask)
    return atom_blocks


def _motif_distance_blocks(
    masks: dict[str, torch.Tensor],
    metadata: dict,
    device: torch.device,
) -> list[torch.Tensor]:
    """Return contiguous motif blocks using all real motif atoms.

    Unlike the generic potential mask path, motif_distance intentionally uses
    all non-virtual motif atoms so its guidance can translate each whole motif
    body together before the optional Kabsch projection restores rigid geometry.
    """
    if "atom_to_token_map" not in metadata or "motif_atom_mask" not in masks:
        return []

    atom_to_token_map = metadata["atom_to_token_map"].to(
        device=device, dtype=torch.long
    )
    motif_mask = masks["motif_atom_mask"].to(device=device, dtype=torch.bool)
    real_mask = masks.get("real_atom_mask")
    if real_mask is not None:
        motif_mask = motif_mask & real_mask.to(device=device, dtype=torch.bool)
    elif "virtual_atom_mask" in masks:
        motif_mask = motif_mask & ~masks["virtual_atom_mask"].to(
            device=device, dtype=torch.bool
        )

    if motif_mask.sum().item() == 0:
        return []

    motif_tokens = torch.unique(atom_to_token_map[motif_mask]).sort().values
    if motif_tokens.numel() == 0:
        return []

    token_runs: list[list[int]] = []
    current_run = [int(motif_tokens[0].item())]
    previous_token = current_run[0]
    for token_tensor in motif_tokens[1:]:
        token = int(token_tensor.item())
        if token == previous_token + 1:
            current_run.append(token)
        else:
            token_runs.append(current_run)
            current_run = [token]
        previous_token = token
    token_runs.append(current_run)

    atom_blocks: list[torch.Tensor] = []
    for token_run in token_runs:
        block_mask = torch.zeros_like(motif_mask)
        for token in token_run:
            block_mask |= atom_to_token_map == token
        atom_blocks.append(block_mask & motif_mask)
    return atom_blocks


def _motif_reference_mask(
    masks: dict[str, torch.Tensor],
    metadata: dict,
    device: torch.device,
) -> torch.Tensor:
    motif_mask = masks.get("motif_atom_mask")
    if motif_mask is None:
        raise ValueError("motif_rigid requires motif_atom_mask in masks")
    motif_mask = motif_mask.to(device=device, dtype=torch.bool)

    ref_pos = metadata.get("ref_pos")
    motif_pos = metadata.get("motif_pos")
    has_ref = _finite_coord_mask(ref_pos, device) if ref_pos is not None else None
    has_motif = _finite_coord_mask(motif_pos, device) if motif_pos is not None else None
    if has_ref is None and has_motif is None:
        raise ValueError("motif_rigid requires ref_pos and/or motif_pos in metadata")

    has_reference = torch.zeros_like(motif_mask)
    if has_ref is not None:
        fixed_seq_mask = _fixed_seq_mask(metadata, masks, device)
        has_reference |= has_ref & fixed_seq_mask
    if has_motif is not None:
        has_reference |= has_motif
    return motif_mask & has_reference


def _motif_reference_xyz(
    xyz: torch.Tensor,
    masks: dict[str, torch.Tensor],
    metadata: dict,
) -> torch.Tensor:
    device = xyz.device
    dtype = xyz.dtype
    ref_xyz = torch.full_like(xyz[0], float("nan"))

    ref_pos = metadata.get("ref_pos")
    if ref_pos is not None:
        ref_pos = ref_pos.to(device=device, dtype=dtype)
        ref_valid = torch.isfinite(ref_pos).all(dim=-1)
        ref_valid = ref_valid & _fixed_seq_mask(metadata, masks, device)
        ref_xyz[ref_valid] = ref_pos[ref_valid]

    motif_pos = metadata.get("motif_pos")
    if motif_pos is not None:
        motif_pos = motif_pos.to(device=device, dtype=dtype)
        motif_valid = torch.isfinite(motif_pos).all(dim=-1)
        fixed_mask = masks.get("fixed_atom_mask")
        if fixed_mask is not None:
            motif_valid = motif_valid & fixed_mask.to(device=device, dtype=torch.bool)
        ref_xyz[motif_valid] = motif_pos[motif_valid]

    return ref_xyz


def _motif_input_reference_xyz(
    xyz: torch.Tensor,
    masks: dict[str, torch.Tensor],
    metadata: dict,
) -> torch.Tensor:
    """Reference coordinates from the input PDB for inference-time motif controls.

    ``floating_motif_reference_pos`` is saved by input parsing before standard
    inference zeroes non-fixed-coordinate atoms.  ``input_pos`` and ``motif_pos``
    are useful fallbacks, but can already be zero-filled for floating motifs.
    ``ref_pos`` is only a final fallback for fixed-sequence atoms.
    """
    device = xyz.device
    dtype = xyz.dtype
    ref_xyz = torch.full_like(xyz[0], float("nan"))

    motif_mask = masks.get("motif_atom_mask")

    floating_reference_pos = metadata.get("floating_motif_reference_pos")
    if floating_reference_pos is not None:
        floating_reference_pos = (
            floating_reference_pos[0]
            if floating_reference_pos.ndim == 3
            else floating_reference_pos
        )
        floating_reference_pos = floating_reference_pos.to(device=device, dtype=dtype)
        floating_valid = torch.isfinite(floating_reference_pos).all(dim=-1)
        if motif_mask is not None:
            floating_valid = floating_valid & motif_mask.to(
                device=device, dtype=torch.bool
            )
        ref_xyz[floating_valid] = floating_reference_pos[floating_valid]

    input_pos = metadata.get("input_pos")
    if input_pos is not None:
        input_pos = input_pos[0] if input_pos.ndim == 3 else input_pos
        input_pos = input_pos.to(device=device, dtype=dtype)
        input_valid = torch.isfinite(input_pos).all(dim=-1)
        if motif_mask is not None:
            input_valid = input_valid & motif_mask.to(device=device, dtype=torch.bool)
        missing_reference = ~torch.isfinite(ref_xyz).all(dim=-1)
        input_valid = input_valid & missing_reference
        ref_xyz[input_valid] = input_pos[input_valid]

    motif_pos = metadata.get("motif_pos")
    if motif_pos is not None:
        motif_pos = motif_pos.to(device=device, dtype=dtype)
        motif_valid = torch.isfinite(motif_pos).all(dim=-1)
        if motif_mask is not None:
            motif_valid = motif_valid & motif_mask.to(device=device, dtype=torch.bool)
        missing_reference = ~torch.isfinite(ref_xyz).all(dim=-1)
        motif_valid = motif_valid & missing_reference
        ref_xyz[motif_valid] = motif_pos[motif_valid]

    ref_pos = metadata.get("ref_pos")
    if ref_pos is not None:
        ref_pos = ref_pos.to(device=device, dtype=dtype)
        ref_valid = torch.isfinite(ref_pos).all(dim=-1)
        if motif_pos is not None:
            ref_valid = ref_valid & _fixed_seq_mask(metadata, masks, device)
        missing_reference = ~torch.isfinite(ref_xyz).all(dim=-1)
        ref_valid = ref_valid & missing_reference
        ref_xyz[ref_valid] = ref_pos[ref_valid]

    return ref_xyz


def _all_input_reference_xyz(
    xyz: torch.Tensor,
    masks: dict[str, torch.Tensor],
    metadata: dict,
) -> torch.Tensor:
    device = xyz.device
    dtype = xyz.dtype
    ref_xyz = torch.full_like(xyz[0], float("nan"))

    floating_reference_pos = metadata.get("floating_motif_reference_pos")
    if floating_reference_pos is not None:
        floating_reference_pos = (
            floating_reference_pos[0]
            if floating_reference_pos.ndim == 3
            else floating_reference_pos
        )
        floating_reference_pos = floating_reference_pos.to(device=device, dtype=dtype)
        floating_valid = torch.isfinite(floating_reference_pos).all(dim=-1)
        ref_xyz[floating_valid] = floating_reference_pos[floating_valid]

    input_pos = metadata.get("input_pos")
    if input_pos is not None:
        input_pos = input_pos[0] if input_pos.ndim == 3 else input_pos
        input_pos = input_pos.to(device=device, dtype=dtype)
        input_valid = torch.isfinite(input_pos).all(dim=-1)
        missing_reference = ~torch.isfinite(ref_xyz).all(dim=-1)
        input_valid = input_valid & missing_reference
        ref_xyz[input_valid] = input_pos[input_valid]

    motif_pos = metadata.get("motif_pos")
    if motif_pos is not None:
        motif_pos = motif_pos.to(device=device, dtype=dtype)
        motif_valid = torch.isfinite(motif_pos).all(dim=-1)
        missing_reference = ~torch.isfinite(ref_xyz).all(dim=-1)
        motif_valid = motif_valid & missing_reference
        ref_xyz[motif_valid] = motif_pos[motif_valid]

    ref_pos = metadata.get("ref_pos")
    if ref_pos is not None:
        ref_pos = ref_pos.to(device=device, dtype=dtype)
        ref_valid = torch.isfinite(ref_pos).all(dim=-1)
        missing_reference = ~torch.isfinite(ref_xyz).all(dim=-1)
        ref_valid = ref_valid & missing_reference
        ref_xyz[ref_valid] = ref_pos[ref_valid]

    return ref_xyz


def _fixed_seq_mask(
    metadata: dict,
    masks: dict[str, torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    fixed_seq = metadata.get("is_motif_atom_with_fixed_seq")
    if fixed_seq is None:
        fixed_seq = masks.get("fixed_seq_atom_mask")
    if fixed_seq is None:
        any_mask = next(iter(masks.values()))
        return torch.ones_like(any_mask, dtype=torch.bool, device=device)
    return fixed_seq.to(device=device, dtype=torch.bool)


def _finite_coord_mask(coords, device: torch.device) -> torch.Tensor:
    coords = (
        coords.to(device=device)
        if isinstance(coords, torch.Tensor)
        else torch.as_tensor(coords, device=device)
    )
    return torch.isfinite(coords).all(dim=-1)


def _atom_filter_mask(
    atom_filter: str,
    masks: dict[str, torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    if atom_filter == "potential":
        key = "potential_atom_mask"
    elif atom_filter == "all":
        any_mask = next(iter(masks.values()))
        return torch.ones_like(any_mask, dtype=torch.bool, device=device)
    elif atom_filter == "backbone":
        key = "backbone_atom_mask"
    elif atom_filter == "CA":
        key = "ca_atom_mask"
    elif atom_filter == "real":
        key = "real_atom_mask"
    else:
        raise ValueError(
            "motif_rigid atom_filter must be one of "
            "'potential', 'all', 'backbone', 'CA', or 'real'"
        )
    if key not in masks:
        raise ValueError(
            f"motif_rigid requires {key} in masks for atom_filter={atom_filter!r}"
        )
    return masks[key].to(device=device, dtype=torch.bool)


def _bridge_atom_mask(
    atom_filter: str,
    masks: dict[str, torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    if atom_filter == "guide":
        key = "guide_atom_mask"
    elif atom_filter == "potential":
        key = "potential_atom_mask"
    elif atom_filter == "binder":
        key = "binder_atom_mask"
    elif atom_filter == "generated":
        key = "generated_atom_mask"
    elif atom_filter == "real":
        key = "real_atom_mask"
    elif atom_filter == "backbone":
        key = "backbone_atom_mask"
    elif atom_filter == "CA":
        key = "ca_atom_mask"
    elif atom_filter == "all":
        any_mask = next(iter(masks.values()))
        return torch.ones_like(any_mask, dtype=torch.bool, device=device)
    else:
        raise ValueError(
            "motif_bridge atom_filter must be one of "
            "'guide', 'potential', 'binder', 'generated', 'real', "
            "'backbone', 'CA', or 'all'"
        )
    if key not in masks:
        raise ValueError(
            f"motif_bridge requires {key} in masks for atom_filter={atom_filter!r}"
        )
    return masks[key].to(device=device, dtype=torch.bool)


def _single_motif_block_mask(
    motif_i: int,
    masks: dict[str, torch.Tensor],
    metadata: dict,
    device: torch.device,
) -> torch.Tensor | None:
    motif_blocks = _motif_distance_blocks(masks, metadata, device)
    if motif_i < 0 or motif_i >= len(motif_blocks):
        return None
    return motif_blocks[motif_i]


def _motif_block_current_and_reference_xyz(
    xyz: torch.Tensor,
    masks: dict[str, torch.Tensor],
    metadata: dict,
    block_mask: torch.Tensor,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    idx = torch.where(block_mask)[0]
    if idx.numel() == 0:
        return None, None

    current_xyz = xyz[:, idx, :]
    ref_xyz = _motif_input_reference_xyz(xyz, masks, metadata)[idx]
    valid = torch.isfinite(ref_xyz).all(dim=-1)
    if valid.sum().item() == 0:
        return None, None
    return current_xyz[:, valid, :], ref_xyz[valid]


def _motif_block_current_com(
    xyz: torch.Tensor,
    block_mask: torch.Tensor,
) -> torch.Tensor | None:
    block_xyz = xyz[:, block_mask, :]
    if block_xyz.shape[1] == 0:
        return None
    finite = torch.isfinite(block_xyz).all(dim=-1)
    if not bool(finite.any()):
        return None
    weights = finite.to(dtype=xyz.dtype)
    denom = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
    return (block_xyz * weights[..., None]).sum(dim=1) / denom


def _motif_block_reference_com(
    xyz: torch.Tensor,
    masks: dict[str, torch.Tensor],
    metadata: dict,
    block_mask: torch.Tensor,
) -> torch.Tensor | None:
    ref_xyz = _motif_input_reference_xyz(xyz, masks, metadata)[block_mask]
    if ref_xyz.shape[0] == 0:
        return None
    finite = torch.isfinite(ref_xyz).all(dim=-1)
    if finite.sum().item() == 0:
        return None
    return ref_xyz[finite].mean(dim=0)


def _pose_origin_atom_mask(
    atom_filter: str,
    masks: dict[str, torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    if atom_filter == "real":
        key = "real_atom_mask"
    elif atom_filter == "potential":
        key = "potential_atom_mask"
    elif atom_filter == "guide":
        key = "guide_atom_mask"
    elif atom_filter == "motif":
        key = "motif_atom_mask"
    elif atom_filter == "all":
        any_mask = next(iter(masks.values()))
        return torch.ones_like(any_mask, dtype=torch.bool, device=device)
    else:
        raise ValueError(
            "motif_rigid_body_pose origin_atom_filter must be one of "
            "'real', 'potential', 'guide', 'motif', or 'all'"
        )
    if key not in masks:
        raise ValueError(
            "motif_rigid_body_pose requires "
            f"{key} in masks for origin_atom_filter={atom_filter!r}"
        )
    return masks[key].to(device=device, dtype=torch.bool)


def _masked_current_com(
    xyz: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor | None:
    selected = xyz[:, mask, :]
    if selected.shape[1] == 0:
        return None
    finite = torch.isfinite(selected).all(dim=-1)
    if not bool(finite.any()):
        return None
    weights = finite.to(dtype=xyz.dtype)
    denom = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
    return (selected * weights[..., None]).sum(dim=1) / denom


def _masked_reference_com(
    xyz: torch.Tensor,
    masks: dict[str, torch.Tensor],
    metadata: dict,
    mask: torch.Tensor,
) -> torch.Tensor | None:
    ref_xyz = _all_input_reference_xyz(xyz, masks, metadata)[mask]
    if ref_xyz.shape[0] == 0:
        return None
    finite = torch.isfinite(ref_xyz).all(dim=-1)
    if finite.sum().item() == 0:
        return None
    return ref_xyz[finite].mean(dim=0)


def _kabsch_ref_to_current_rotation(
    ref_xyz: torch.Tensor,
    current_xyz: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor | None:
    if ref_xyz.shape[0] < 3 or current_xyz.shape[1] < 3:
        return None

    device_type = current_xyz.device.type
    with torch.autocast(device_type=device_type, enabled=False):
        ref = ref_xyz.unsqueeze(0).expand(current_xyz.shape[0], -1, -1).float()
        cur = current_xyz.float()
        ref_centered = ref - ref.mean(dim=1, keepdim=True)
        cur_centered = cur - cur.mean(dim=1, keepdim=True)
        if not bool((ref_centered.pow(2).sum(dim=(-2, -1)) > eps).all()):
            return None

        covariance = ref_centered.transpose(-1, -2) @ cur_centered
        eye = torch.eye(3, device=current_xyz.device, dtype=torch.float32)
        covariance = covariance + eps * eye.unsqueeze(0)
        U, _, Vh = torch.linalg.svd(covariance)
        rotation = U @ Vh
        det = torch.linalg.det(rotation)
        if bool((det < 0).any()):
            U_fixed = U.clone()
            U_fixed[det < 0, :, -1] *= -1
            rotation = U_fixed @ Vh
    return rotation.to(dtype=current_xyz.dtype)


def _euler_rotation_matrix_deg(
    angle_x: float,
    angle_y: float,
    angle_z: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    ax = torch.deg2rad(torch.tensor(angle_x, device=device, dtype=dtype))
    ay = torch.deg2rad(torch.tensor(angle_y, device=device, dtype=dtype))
    az = torch.deg2rad(torch.tensor(angle_z, device=device, dtype=dtype))
    one = torch.tensor(1.0, device=device, dtype=dtype)
    zero = torch.tensor(0.0, device=device, dtype=dtype)

    cx, sx = torch.cos(ax), torch.sin(ax)
    cy, sy = torch.cos(ay), torch.sin(ay)
    cz, sz = torch.cos(az), torch.sin(az)

    rx = torch.stack(
        [
            torch.stack([one, zero, zero]),
            torch.stack([zero, cx, -sx]),
            torch.stack([zero, sx, cx]),
        ]
    )
    ry = torch.stack(
        [
            torch.stack([cy, zero, sy]),
            torch.stack([zero, one, zero]),
            torch.stack([-sy, zero, cy]),
        ]
    )
    rz = torch.stack(
        [
            torch.stack([cz, -sz, zero]),
            torch.stack([sz, cz, zero]),
            torch.stack([zero, zero, one]),
        ]
    )
    return rz @ ry @ rx


def _apply_row_rotation(points: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    return points @ rotation.transpose(-1, -2)


def _apply_batch_row_rotation(
    points: torch.Tensor,
    rotation: torch.Tensor,
) -> torch.Tensor:
    return torch.matmul(points, rotation.transpose(-1, -2))


def _pose_direction_loss(
    current_vectors: torch.Tensor,
    target_vectors: torch.Tensor,
    dtype: torch.dtype,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    current_norm = current_vectors.norm(dim=-1)
    target_norm = target_vectors.norm(dim=-1)
    valid = (current_norm > eps) & (target_norm[None, :] > eps)
    current_dirs = current_vectors / current_norm.clamp_min(eps)[..., None]
    target_dirs = target_vectors / target_norm.clamp_min(eps)[:, None]
    loss = (current_dirs - target_dirs.unsqueeze(0)).pow(2).sum(dim=-1)
    return loss.to(dtype=dtype), valid


def _pose_distance_loss(
    current_vectors: torch.Tensor,
    target_vectors: torch.Tensor,
    valid: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    current_norm = current_vectors.norm(dim=-1)
    target_norm = target_vectors.norm(dim=-1)
    scale = target_norm.clamp_min(eps)[None, :]
    loss = ((current_norm - target_norm[None, :]) / scale).pow(2)
    valid_weight = valid.to(dtype=current_vectors.dtype)
    return (loss * valid_weight).sum() / valid_weight.sum().clamp_min(eps)


def _rounded_float_list(values: torch.Tensor) -> list[float]:
    return [round(float(value), 6) for value in values.detach().cpu().flatten()]


def _normalize_vectors(vectors: torch.Tensor, eps: float = 1e-6) -> torch.Tensor | None:
    norms = vectors.norm(dim=-1, keepdim=True)
    if not bool((norms > eps).all()):
        return None
    return vectors / norms


def _build_radial_frame(r: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Build a deterministic orthonormal frame with r as the first column.

    Picks the global axis least aligned with r as the second Gram-Schmidt seed
    to avoid degeneracy (ensures the frame is well-conditioned for any r).

    Args:
        r:   [3] unit vector.
        eps: denominator clamp for normalisation.

    Returns:
        [3, 3] matrix whose columns are [r, t1, t2] (all orthonormal).
    """
    # Axis whose component in r is smallest → least aligned with r
    abs_r = r.detach().abs()
    min_idx = int(abs_r.argmin().item())
    v = r.new_zeros(3)
    v[min_idx] = 1.0

    # Gram-Schmidt: remove r-projection from v
    t1 = v - (v * r).sum() * r
    t1 = t1 / t1.norm().clamp_min(eps)
    t2 = torch.cross(r.unsqueeze(0), t1.unsqueeze(0), dim=-1).squeeze(0)
    return torch.stack([r, t1, t2], dim=-1)  # [3, 3], columns are basis vectors


def _build_radial_frame_batched(r: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Batched version of _build_radial_frame.

    Args:
        r:   [D, 3] unit vectors (should be detached before calling to keep
             the returned frame out of the autograd graph).
        eps: denominator clamp for normalisation.

    Returns:
        [D, 3, 3] matrices whose columns are [r, t1, t2].
    """
    abs_r = r.abs()  # [D, 3]
    min_idx = abs_r.argmin(dim=-1)  # [D] — index of least-aligned axis per sample

    # One-hot: v[d, min_idx[d]] = 1.0
    v = torch.zeros_like(r)
    v.scatter_(-1, min_idx.unsqueeze(-1), 1.0)

    # Gram-Schmidt
    t1 = v - (v * r).sum(dim=-1, keepdim=True) * r  # [D, 3]
    t1 = t1 / t1.norm(dim=-1, keepdim=True).clamp_min(eps)
    t2 = torch.cross(r, t1, dim=-1)  # [D, 3]
    return torch.stack([r, t1, t2], dim=-1)  # [D, 3, 3], columns are basis vectors


def _min_rotation_matrix_batched(
    a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """Minimal (geodesic) rotation matrices R with R @ a = b, per batch element.

    a: [3] (broadcast) or [D, 3];  b: [D, 3];  returns [D, 3, 3].
    Used to parallel-transport a reference radial frame from r_ref to r_cur so
    the tangent axes are carried along the geodesic instead of being re-seeded
    from world axes (which leaks angular position into the orientation loss).
    Handles a~b (R=I) and a~-b (180 deg about an arbitrary perpendicular).
    Inputs are detached; the returned frame stays out of the autograd graph.
    """
    b = (b / b.norm(dim=-1, keepdim=True).clamp_min(eps)).detach()
    if a.dim() == 1:
        a = a.unsqueeze(0).expand_as(b)
    a = (a / a.norm(dim=-1, keepdim=True).clamp_min(eps)).detach()
    D = b.shape[0]
    dtype, device = b.dtype, b.device
    v = torch.cross(a, b, dim=-1)          # axis * sin(theta)   [D, 3]
    c = (a * b).sum(dim=-1)                # cos(theta)          [D]
    s2 = (v * v).sum(dim=-1)               # sin^2(theta)        [D]
    eye = torch.eye(3, dtype=dtype, device=device).expand(D, 3, 3)
    zero = torch.zeros(D, dtype=dtype, device=device)
    vx = torch.stack([
        torch.stack([zero, -v[:, 2], v[:, 1]], dim=-1),
        torch.stack([v[:, 2], zero, -v[:, 0]], dim=-1),
        torch.stack([-v[:, 1], v[:, 0], zero], dim=-1),
    ], dim=-2)                             # [D, 3, 3]
    coef = ((1.0 - c) / s2.clamp_min(eps)).view(D, 1, 1)
    R = eye + vx + torch.matmul(vx, vx) * coef   # Rodrigues (valid for sin != 0)
    # Degenerate: sin(theta) ~ 0.  a~b -> identity; a~-b -> 180 deg flip.
    near_deg = s2 < eps
    if near_deg.any():
        Rdeg = eye.clone()
        anti = near_deg & (c < 0)
        if anti.any():
            aa = a[anti]
            widx = aa.abs().argmin(dim=-1)
            wv = torch.zeros_like(aa)
            wv.scatter_(-1, widx.unsqueeze(-1), 1.0)
            perp = wv - (wv * aa).sum(dim=-1, keepdim=True) * aa
            perp = perp / perp.norm(dim=-1, keepdim=True).clamp_min(eps)
            Rdeg[anti] = (2.0 * torch.matmul(perp.unsqueeze(-1), perp.unsqueeze(-2))
                          - torch.eye(3, dtype=dtype, device=device))
        R = torch.where(near_deg.view(D, 1, 1), Rdeg, R)
    return R.detach()


def _project_gradient_to_rigid_body(
    atom_grad: torch.Tensor,
    xyz_block: torch.Tensor,
    allow_translation: bool,
    allow_rotation: bool,
    eps: float = 1e-6,
) -> torch.Tensor:
    transformed = torch.zeros_like(atom_grad)
    if atom_grad.shape[1] == 0:
        return transformed

    centered = xyz_block - xyz_block.mean(dim=1, keepdim=True)
    residual = atom_grad
    if allow_translation:
        translation = atom_grad.mean(dim=1, keepdim=True)
        transformed = transformed + translation
        residual = atom_grad - translation

    if allow_rotation and atom_grad.shape[1] >= 2:
        device_type = xyz_block.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            eye = torch.eye(3, device=xyz_block.device, dtype=torch.float32).unsqueeze(0)
            centered32 = centered.float()
            residual32 = residual.float()
            sq_norm = centered32.pow(2).sum(dim=-1)[..., None, None]
            outer = centered32[:, :, :, None] * centered32[:, :, None, :]
            system = (sq_norm * eye[:, None, :, :] - outer).sum(dim=1) + eps * eye
            rhs = torch.cross(centered32, residual32, dim=-1).sum(dim=1)
            omega = torch.linalg.solve(system, rhs.unsqueeze(-1)).squeeze(-1)
            rotation_field = torch.cross(
                omega.unsqueeze(1).expand_as(centered32), centered32, dim=-1
            ).to(dtype=atom_grad.dtype)
        transformed = transformed + rotation_field

    return transformed


def _distance_loss(diff: torch.Tensor, k: float, loss: str) -> torch.Tensor:
    if loss == "mse":
        return diff.pow(2).mean()
    if loss == "l1":
        return diff.abs().mean()
    if loss == "pseudo_huber":
        k_tensor = diff.new_tensor(k)
        return (
            k_tensor.pow(2)
            * (torch.sqrt((diff / k_tensor).pow(2) + 1.0) - 1.0)
        ).mean()
    raise ValueError("motif_rigid loss must be one of 'pseudo_huber', 'mse', or 'l1'")


def _soft_contacts(dists: torch.Tensor, r_0: float, d_0: float) -> torch.Tensor:
    """RFdiffusion-style soft contacts, written in a numerically stable form."""
    x = (dists - d_0) / r_0
    return 1.0 / (1.0 + x.pow(6))


class MotifPairOrientation(BasePotential):
    """Control each motif's orientation via two frame-free rotation scalars.

    Replacement for MotifRadialOrientationPotential / MotifForbiddenRadialOrientation
    that measures the motif's *rotation* directly instead of differencing atom
    coordinates in an ad-hoc local frame.

    Per motif block:
      Q       = Kabsch rotation taking the centred reference onto the centred
                current motif.  Because it is a rigid fit, Q is exactly invariant
                to the motif's internal deformation -- the diffuse early-trajectory
                motif yields a *bounded* rotation rather than an O(t_hat^2)
                Angstrom^2 loss.  (Verified: Kabsch(ref, blob) equals
                Kabsch(ref, rigidify(blob)) to float32 precision, so this loss is
                unchanged by whether the floating-motif projection has run yet.)
      r_ref   = unit vector from the mean of all motif reference centres to this
                motif's reference centre.  A CONSTANT from the input PDB -- no
                noisy per-step axis measurement, so no world-axis frame re-seeding
                and no angular-position contamination.  This assumes the motif
                stays near its input angular position, which is exactly what
                motif_spherical_position enforces (measured: final inter-motif
                axis lands 4 +/- 5 deg from the input axis at weight 5000, versus
                107 +/- 36 deg at weight 0).

    Two scalars, both basis-independent traces of Q and both invariant to the
    row-vs-column rotation convention:

        cos_spin = 0.5 * (tr(Q) - r_ref . Q r_ref)   # rotation ABOUT r_ref
        cos_tilt = r_ref . Q r_ref                   # rotation AWAY FROM r_ref

    Reference pose  -> cos_spin = +1, cos_tilt = +1.
    180-deg flip about r_ref (the "forbidden" pose) -> cos_spin = -1, cos_tilt = +1.

    Reward (this potential is MAXIMISED):

        spin_weight * sqrt(1 + cos_spin)  +  tilt_weight * cos_tilt

    The sqrt is deliberate: sqrt(1 + cos_spin) = sqrt(2)*|cos(spin/2)|, whose
    torque goes as sin(spin/2) -- MAXIMAL at the forbidden 180-deg pose and
    decaying to zero at the reference pose.  That is the opposite of
    MotifForbiddenRadialOrientation's Gaussian bump, which saturates to *zero*
    force exactly where repulsion is needed.  A plain (1 + cos_spin) reward would
    instead peak at 90 deg and vanish at 180 deg.

    Both terms are dimensionless and bounded (spin in [0, sqrt(2)], tilt in
    [-1, 1]), so the weights are directly comparable and there is no
    Angstrom^2-scale `sigma` to mis-set.

    Gradient is projected to pure rigid rotation per motif (no translation), so
    this potential never fights motif_distance / motif_spherical_position.
    """

    def __init__(
        self,
        weight: float = 1.0,
        spin_weight: float = 1.0,
        tilt_weight: float = 0.0,
        motif_indices: list | None = None,
        origin_atom_filter: str = "real",
        sqrt_eps: float = 1e-3,
        eps: float = 1e-6,
        debug_log: bool = False,
    ):
        super().__init__(weight)
        self.spin_weight = float(spin_weight)
        self.tilt_weight = float(tilt_weight)
        self.motif_indices: list | None = (
            None if motif_indices is None else [int(i) for i in motif_indices]
        )
        valid_origin_filters = ("real", "potential", "guide", "motif", "all")
        if origin_atom_filter not in valid_origin_filters:
            raise ValueError(
                "motif_pair_orientation origin_atom_filter must be one of "
                f"{valid_origin_filters}"
            )
        self.origin_atom_filter = origin_atom_filter
        # Floor inside the sqrt.  d/dcos sqrt(1+cos) diverges at the forbidden
        # pose; the vanishing d cos/d(coords) there keeps the product finite, but
        # the floor keeps the intermediate bounded in float32.
        self.sqrt_eps = float(sqrt_eps)
        self.eps = float(eps)
        self.debug_log = bool(debug_log)
        self.skip_reason: str | None = None
        self.skip_detail: dict | None = None

    def _selected_motif_indices(self, n_blocks: int) -> range | list:
        if self.motif_indices is None:
            return range(n_blocks)
        return [i for i in self.motif_indices if 0 <= i < n_blocks]

    def compute(self, xyz, masks, metadata):
        self.skip_reason = None
        self.skip_detail = None

        motif_blocks = _motif_distance_blocks(masks, metadata, xyz.device)
        self.skip_detail = {
            "n_motif_blocks": len(motif_blocks),
            "motif_indices": self.motif_indices,
        }
        if len(motif_blocks) == 0:
            self.skip_reason = "no_motif_blocks"
            return xyz.new_zeros(())

        selected_indices = self._selected_motif_indices(len(motif_blocks))
        if not selected_indices:
            self.skip_reason = "no_selected_motif_indices"
            return xyz.new_zeros(())

        # Reference global centre = mean of all motif-block reference centres,
        # matching MotifRadialOrientationPotential's convention.
        ref_centers: list[torch.Tensor | None] = [
            _motif_block_reference_com(xyz, masks, metadata, bm) for bm in motif_blocks
        ]
        valid_ref_centers = [c for c in ref_centers if c is not None]
        if not valid_ref_centers:
            self.skip_reason = "no_motif_reference_centers"
            return xyz.new_zeros(())
        c_ref_global = torch.stack(valid_ref_centers, dim=0).mean(dim=0)  # [3]

        total = xyz.new_zeros(())
        n_active = 0
        debug_rows: list[dict] = []

        for motif_i in selected_indices:
            block_mask = motif_blocks[motif_i]
            current_xyz_i, ref_xyz_i = _motif_block_current_and_reference_xyz(
                xyz, masks, metadata, block_mask
            )
            if current_xyz_i is None or ref_xyz_i is None or ref_xyz_i.shape[0] < 3:
                continue

            ref_com_i = ref_centers[motif_i]
            if ref_com_i is None:
                continue
            d_ref = ref_com_i - c_ref_global
            d_ref_norm = d_ref.norm()
            if d_ref_norm < self.eps:
                continue  # motif sits at the global centre -- no radial axis
            r_ref_i = (d_ref / d_ref_norm).detach()  # [3] constant unit vector

            Q = _kabsch_ref_to_current_rotation(ref_xyz_i, current_xyz_i, self.eps)
            if Q is None:
                continue  # degenerate fit

            tr = Q.diagonal(dim1=-2, dim2=-1).sum(dim=-1)          # [D]
            rQr = torch.einsum("i,dij,j->d", r_ref_i, Q, r_ref_i)  # [D]
            cos_spin = (0.5 * (tr - rQr)).clamp(-1.0, 1.0)         # [D]
            cos_tilt = rQr.clamp(-1.0, 1.0)                        # [D]

            spin_term = (1.0 + cos_spin).clamp_min(self.sqrt_eps).sqrt()
            reward = self.spin_weight * spin_term + self.tilt_weight * cos_tilt
            total = total + reward.mean()
            n_active += 1

            if self.debug_log:
                # Internal deformation is what the floating-motif projection
                # removes.  Q is provably invariant to it, so log it to judge
                # whether reordering the potential after the projection could
                # still matter via the gradient (the loss value cannot change).
                with torch.no_grad():
                    rc = (ref_xyz_i - ref_xyz_i.mean(dim=0, keepdim=True)).float()
                    cc = (
                        current_xyz_i - current_xyz_i.mean(dim=1, keepdim=True)
                    ).float()
                    fitted = torch.matmul(rc.unsqueeze(0), Q.float())
                    fit_rmsd = (cc - fitted).pow(2).sum(-1).mean(-1).sqrt()
                    rg_ratio = cc.pow(2).sum(-1).mean(-1).sqrt() / rc.pow(2).sum(
                        -1
                    ).mean().sqrt().clamp_min(self.eps)
                    debug_rows.append(
                        {
                            "motif": int(motif_i),
                            "spin_deg": [
                                round(float(v), 2)
                                for v in torch.rad2deg(torch.acos(cos_spin))
                            ],
                            "tilt_deg": [
                                round(float(v), 2)
                                for v in torch.rad2deg(torch.acos(cos_tilt))
                            ],
                            "fit_rmsd": [round(float(v), 3) for v in fit_rmsd],
                            "rg_ratio": [round(float(v), 3) for v in rg_ratio],
                        }
                    )

        if n_active == 0:
            self.skip_reason = "no_active_motifs"
            return xyz.new_zeros(())

        if self.debug_log and debug_rows:
            # This module's loggers do not reach the SLURM log; print like
            # potentials/integration.py does.
            print(f"[pair_ori] {debug_rows}", file=sys.stderr, flush=True)

        return self.weight * total / n_active

    def guide_atom_mask(self, masks, metadata, device):
        motif_blocks = _motif_distance_blocks(masks, metadata, device)
        if not motif_blocks:
            any_mask = next(iter(masks.values()))
            return torch.zeros_like(any_mask, dtype=torch.bool, device=device)
        guide_mask = torch.zeros_like(motif_blocks[0], dtype=torch.bool, device=device)
        for motif_i in self._selected_motif_indices(len(motif_blocks)):
            guide_mask = guide_mask | motif_blocks[motif_i]
        return guide_mask

    def transform_atom_gradient(self, atom_grad, masks, metadata, xyz):
        motif_blocks = _motif_distance_blocks(masks, metadata, xyz.device)
        if not motif_blocks:
            return torch.zeros_like(atom_grad)
        transformed = torch.zeros_like(atom_grad)
        for motif_i in self._selected_motif_indices(len(motif_blocks)):
            block_mask = motif_blocks[motif_i]
            if block_mask.sum().item() < 3:
                continue
            # Pure rigid rotation: no translation, no deformation.
            transformed[:, block_mask, :] = _project_gradient_to_rigid_body(
                atom_grad[:, block_mask, :],
                xyz[:, block_mask, :],
                allow_translation=False,
                allow_rotation=True,
            )
        return transformed


class MotifPairAxisDot(BasePotential):
    """Drive the receptor-receptor axis dot product to its input-PDB value.

    This targets the QC metric directly.  The screen's pass/fail criterion is

        dot = v_receptor_i . v_receptor_j

    where each receptor axis is CA(distal) - CA(proximal) of the *full receptor*
    (HER2 571->496, FGFR 263->277 for FGFR-HER2), taken after the receptor is
    superimposed onto its designed motif.  The receptors are not part of the
    diffused structure, but the motif is rigid, so each receptor axis is a
    CONSTANT VECTOR IN THE MOTIF'S BODY FRAME.  Given the Kabsch rotation R_i
    from reference motif to current motif,

        v_i = a_i @ R_i        (row convention: cur_centred ~ ref_centred @ R)

    recovers the receptor axis without ever needing receptor atoms.

    Why this and not per-motif spin/tilt:
      * It is SE(3)-invariant UNCONDITIONALLY.  Under a global rotation G both
        R_i pick up G, and (G v_i).(G v_j) = v_i.v_j.  Unlike a potential
        referenced to a fixed r_ref, it does not rely on the motif positions
        being pinned.
      * It constrains exactly the 1 DOF the metric measures.  Forcing both
        motifs back to their input orientation would be a 6-DOF constraint for a
        1-DOF criterion, needlessly fighting linker designability.
      * It correctly captures the "forbidden" antiparallel state.  A negative dot
        does NOT require either motif to be flipped 180 degrees -- two ~90 degree
        rotations can produce it.  Measured on 400 designs, per-motif rotation
        angle correlates with dot at |r| <= 0.21, so repelling each motif from
        its own flipped pose does not prevent a negative dot.

    Reward (MAXIMISED):  -weight * (dot - target_dot)^2

    Harmonic in dot: the force is linear in the error, MAXIMAL at the
    antiparallel extreme and zero at the target.  Bounded and dimensionless
    ((dot-target)^2 <= ~4), so `weight` is a plain gain with no Angstrom^2 scale.

    ``motif_axes`` gives the two body-frame axis unit vectors directly, in
    motif-block (contig) order -- computed offline by hand from a reference
    structure. As an alternative, ``motif_chains`` computes the same body-frame
    vectors internally: each entry is ``{motif_index, chain_pdb, chain_id,
    receptor_residues: [proximal_resid, distal_resid], align, align_chain_id,
    align_atom_selection}`` -- the same shape as MinimalOverlapPotential's own
    ``motif_chains`` (chain_pdb is never part of the diffused structure or
    necessarily even the design's own input PDB, see that class's docstring).
    ``receptor_residues``' two residues' CA atoms (``atom_name`` overrides "CA")
    give a raw-frame vector ``distal - proximal``; with ``align=True`` that
    vector is rotated (translation does not apply to a direction) by the same
    offline Kabsch fit ``align_chain_id`` -> the design's own motif reference
    already uses elsewhere in this file, landing it in the motif's body frame
    with no further correction needed. Exactly one of ``motif_axes`` /
    ``motif_chains`` must be given. ``target_dot`` defaults to a_i . a_j, which
    is the input geometry's own dot product -- so R = I reproduces the control
    exactly and no separate target has to be supplied.

    ``motif_i`` / ``motif_j`` select which two motif blocks (in contig order)
    are being compared. With ``motif_chains``, this is redundant with each
    entry's own ``motif_index`` and does not need to be given -- it is derived
    automatically from ``motif_chains``' two ``motif_index`` values (in list
    order). Only needed explicitly when using ``motif_axes`` (default 0, 1),
    or to override the derived values (which must then still match one of
    ``motif_chains``' own ``motif_index`` entries).

    Gradient is projected to pure rigid rotation per motif (no translation), so
    this never fights motif_distance / motif_spherical_position.
    """

    def __init__(
        self,
        weight: float = 1.0,
        motif_axes: list | None = None,
        motif_chains: list | None = None,
        motif_i: int | None = None,
        motif_j: int | None = None,
        target_dot: float | None = None,
        eps: float = 1e-6,
        debug_log: bool = False,
    ):
        super().__init__(weight)
        if (motif_axes is None) == (motif_chains is None):
            raise ValueError(
                "motif_pair_axis_dot requires exactly one of motif_axes=[[x,y,z],"
                "[x,y,z]] (the receptor axis unit vectors in each motif's body "
                "frame, given directly) or motif_chains=[{motif_index, chain_pdb, "
                "chain_id, receptor_residues: [proximal_resid, distal_resid], "
                "...}, {...}] (computed internally from two residues per motif)"
            )
        if motif_axes is not None:
            if len(motif_axes) < 2:
                raise ValueError("motif_pair_axis_dot: motif_axes needs 2 entries")
            self.motif_axes = [[float(c) for c in a[:3]] for a in motif_axes[:2]]
            self._motif_chain_by_index = None
            # motif_i/motif_j have no self-describing source here (motif_axes
            # is just 2 raw vectors, positionally paired) -- default to the
            # historical 0/1 convention, same as before this param became
            # optional.
            motif_i = 0 if motif_i is None else motif_i
            motif_j = 1 if motif_j is None else motif_j
        else:
            if len(motif_chains) != 2:
                raise ValueError(
                    "motif_pair_axis_dot: motif_chains must have exactly 2 entries "
                    f"(one per motif), got {len(motif_chains)}"
                )
            resolved: dict[int, dict] = {}
            for i, spec in enumerate(motif_chains):
                missing = [
                    k for k in ("motif_index", "chain_pdb", "chain_id", "receptor_residues")
                    if k not in spec
                ]
                if missing:
                    raise ValueError(f"motif_pair_axis_dot: motif_chains[{i}] is missing {missing}")
                residues = spec["receptor_residues"]
                if len(residues) != 2:
                    raise ValueError(
                        f"motif_pair_axis_dot: motif_chains[{i}] receptor_residues "
                        "must be [proximal_resid, distal_resid] (exactly 2)"
                    )
                entry = {
                    "chain_pdb": str(spec["chain_pdb"]),
                    "chain_id": str(spec["chain_id"]),
                    "proximal_residue": int(residues[0]),
                    "distal_residue": int(residues[1]),
                    "atom_name": str(spec.get("atom_name", "CA")),
                    "align": bool(spec.get("align", False)),
                    "align_chain_id": spec.get("align_chain_id"),
                    "align_atom_selection": str(spec.get("align_atom_selection", "CA")),
                }
                if entry["align"] and not entry["align_chain_id"]:
                    raise ValueError(
                        f"motif_pair_axis_dot: motif_chains[{i}] sets align=True "
                        "but is missing 'align_chain_id'"
                    )
                resolved[int(spec["motif_index"])] = entry
            self.motif_axes = None
            self._motif_chain_by_index = resolved
            # motif_chains already self-describes which two motif blocks are
            # being compared via each entry's own motif_index -- redundant to
            # also require motif_i/motif_j. Derive them (in the given list's
            # order) when not explicitly overridden; if they ARE given, they
            # must actually match one of the motif_chains entries or the
            # later dict lookup in _resolve_axis would silently make no sense.
            derived_i, derived_j = (int(spec["motif_index"]) for spec in motif_chains)
            motif_i = derived_i if motif_i is None else motif_i
            motif_j = derived_j if motif_j is None else motif_j
            if motif_i not in resolved or motif_j not in resolved:
                raise ValueError(
                    f"motif_pair_axis_dot: motif_i={motif_i}, motif_j={motif_j} must "
                    f"each match one of motif_chains' own motif_index values "
                    f"({sorted(resolved)})"
                )
        self._axis_cache: dict[int, torch.Tensor] = {}
        self.motif_i = int(motif_i)
        self.motif_j = int(motif_j)
        self.target_dot = None if target_dot is None else float(target_dot)
        self.eps = float(eps)
        self.debug_log = bool(debug_log)
        self.skip_reason: str | None = None
        self.skip_detail: dict | None = None

    def _resolve_axis(self, slot, motif_i, ref_xyz_i, device, dtype):
        if self.motif_axes is not None:
            a = torch.tensor(self.motif_axes[slot], device=device, dtype=dtype)
            return a / a.norm().clamp_min(self.eps)

        if motif_i in self._axis_cache:
            return self._axis_cache[motif_i]
        spec = self._motif_chain_by_index[motif_i]
        proximal = _load_chain_residue_atom(
            spec["chain_pdb"], spec["chain_id"], spec["proximal_residue"], spec["atom_name"]
        ).to(device=device, dtype=dtype)
        distal = _load_chain_residue_atom(
            spec["chain_pdb"], spec["chain_id"], spec["distal_residue"], spec["atom_name"]
        ).to(device=device, dtype=dtype)
        raw_vector = distal - proximal

        if spec["align"]:
            align_xyz = _load_chain_atoms(
                spec["chain_pdb"], spec["align_chain_id"], spec["align_atom_selection"], None
            ).to(device=device, dtype=dtype)
            if align_xyz.shape[0] != ref_xyz_i.shape[0]:
                raise ValueError(
                    f"motif_pair_axis_dot: motif_chains[motif_index={motif_i}] "
                    f"align_chain_id={spec['align_chain_id']!r} has "
                    f"{align_xyz.shape[0]} atoms but the design's motif reference "
                    f"has {ref_xyz_i.shape[0]} -- align_atom_selection must "
                    "produce a 1:1 atom correspondence (same count and order) "
                    "with the contig motif block"
                )
            R_align = _kabsch_ref_to_current_rotation(align_xyz, ref_xyz_i.unsqueeze(0), self.eps)
            if R_align is None:
                raise ValueError(
                    f"motif_pair_axis_dot: motif_chains[motif_index={motif_i}] "
                    "offline alignment is degenerate (too few or co-linear "
                    "align atoms)"
                )
            # A vector (unlike a point) transforms by rotation alone -- the
            # translation/COM terms MinimalOverlapPotential's own point-based
            # align needs cancel out exactly when differencing two points.
            raw_vector = raw_vector @ R_align[0]

        axis = raw_vector / raw_vector.norm().clamp_min(self.eps)
        self._axis_cache[motif_i] = axis
        return axis

    def compute(self, xyz, masks, metadata):
        self.skip_reason = None
        self.skip_detail = None

        motif_blocks = _motif_distance_blocks(masks, metadata, xyz.device)
        self.skip_detail = {
            "n_motif_blocks": len(motif_blocks),
            "motif_i": self.motif_i,
            "motif_j": self.motif_j,
        }
        if max(self.motif_i, self.motif_j) >= len(motif_blocks):
            self.skip_reason = "motif_index_out_of_range"
            return xyz.new_zeros(())

        axes = []
        vecs = []
        for slot, motif_i in enumerate((self.motif_i, self.motif_j)):
            block_mask = motif_blocks[motif_i]
            current_xyz_i, ref_xyz_i = _motif_block_current_and_reference_xyz(
                xyz, masks, metadata, block_mask
            )
            if current_xyz_i is None or ref_xyz_i is None or ref_xyz_i.shape[0] < 3:
                self.skip_reason = f"motif_{motif_i}_coords_missing"
                return xyz.new_zeros(())
            R = _kabsch_ref_to_current_rotation(ref_xyz_i, current_xyz_i, self.eps)
            if R is None:
                self.skip_reason = f"motif_{motif_i}_frame_degenerate"
                return xyz.new_zeros(())
            axis = self._resolve_axis(slot, motif_i, ref_xyz_i, xyz.device, xyz.dtype)
            axes.append(axis)
            # Row convention: a vector in the reference frame maps to a @ R.
            v = torch.einsum("j,djk->dk", axis, R)  # [D, 3]
            vecs.append(v / v.norm(dim=-1, keepdim=True).clamp_min(self.eps))

        target = (
            float((axes[0] * axes[1]).sum())
            if self.target_dot is None
            else self.target_dot
        )

        dot = (vecs[0] * vecs[1]).sum(dim=-1).clamp(-1.0, 1.0)  # [D]
        loss = (dot - target).pow(2)

        if self.debug_log:
            print(
                "[axis_dot] "
                + repr(
                    {
                        "target": round(target, 4),
                        "dot": [round(float(v), 4) for v in dot],
                        "angle_deg": [
                            round(float(v), 2) for v in torch.rad2deg(torch.acos(dot))
                        ],
                    }
                ),
                file=sys.stderr,
                flush=True,
            )

        return -self.weight * loss.mean()

    def guide_atom_mask(self, masks, metadata, device):
        motif_blocks = _motif_distance_blocks(masks, metadata, device)
        if not motif_blocks:
            any_mask = next(iter(masks.values()))
            return torch.zeros_like(any_mask, dtype=torch.bool, device=device)
        guide_mask = torch.zeros_like(motif_blocks[0], dtype=torch.bool, device=device)
        for motif_i in (self.motif_i, self.motif_j):
            if motif_i < len(motif_blocks):
                guide_mask = guide_mask | motif_blocks[motif_i]
        return guide_mask

    def transform_atom_gradient(self, atom_grad, masks, metadata, xyz):
        motif_blocks = _motif_distance_blocks(masks, metadata, xyz.device)
        if not motif_blocks:
            return torch.zeros_like(atom_grad)
        transformed = torch.zeros_like(atom_grad)
        for motif_i in (self.motif_i, self.motif_j):
            if motif_i >= len(motif_blocks):
                continue
            block_mask = motif_blocks[motif_i]
            if block_mask.sum().item() < 3:
                continue
            transformed[:, block_mask, :] = _project_gradient_to_rigid_body(
                atom_grad[:, block_mask, :],
                xyz[:, block_mask, :],
                allow_translation=False,
                allow_rotation=True,
            )
        return transformed


def _load_chain_atoms(
    pdb_path: str,
    chain_id: str,
    atom_selection: str = "CA",
    max_atoms: int | None = None,
) -> torch.Tensor:
    """Load one chain's atom coordinates from a PDB/mmCIF file.

    Used by MinimalOverlapPotential to bring in chains that are never part of
    the diffused structure -- the same situation MotifPairAxisDot documents for
    its receptor axis ("The receptor itself is never part of the RFD3 contig or
    the sampled structure"), just generalised from a single axis vector to a
    full point cloud so steric overlap can be measured.

    ``atom_selection``: 'CA' (default -- cheap and sufficient as a steric
    proxy), 'backbone' (N, CA, C, O), or 'heavy' (all non-hydrogen atoms).
    ``max_atoms``: optional uniform-stride subsample cap so a very large chain
    can't blow up the pairwise-overlap cost; deterministic, not random.

    Biotite is imported lazily here (not at module scope) so importing
    potentials.py stays cheap for callers that never touch this potential.
    """
    import biotite.structure as struc
    import biotite.structure.io as strucio
    import numpy as np

    path = Path(pdb_path)
    if not path.is_file():
        raise FileNotFoundError(f"minimal_overlap chain_pdb not found: {pdb_path}")

    atom_array = strucio.load_structure(str(path), model=1)

    chain_mask = atom_array.chain_id == str(chain_id)
    if atom_selection == "CA":
        sel_mask = atom_array.atom_name == "CA"
    elif atom_selection == "backbone":
        sel_mask = np.isin(atom_array.atom_name, ["N", "CA", "C", "O"])
    elif atom_selection == "heavy":
        element = atom_array.element
        if element is None or getattr(element, "size", 0) == 0:
            atom_array = atom_array.copy()
            atom_array.element = struc.infer_elements(atom_array)
            element = atom_array.element
        # OXT (the C-terminal carboxylate oxygen) is present on essentially
        # every raw PDB's terminal residue but never modeled by RFD3's own
        # fixed per-residue atom template -- excluding it here (rather than
        # requiring hand-cleaned input files) keeps 'heavy' 1:1-comparable
        # with RFD3's internal reference for *any* chain/receptor, not just
        # ones someone happened to pre-strip.
        sel_mask = (np.asarray(element) != "H") & (
            np.asarray(atom_array.atom_name) != "OXT"
        )
    else:
        raise ValueError(
            "minimal_overlap atom_selection must be one of 'CA', 'backbone', "
            f"'heavy', got {atom_selection!r}"
        )

    coords = atom_array.coord[chain_mask & sel_mask]
    if coords.shape[0] == 0:
        raise ValueError(
            f"minimal_overlap: no atoms found for chain_id={chain_id!r} "
            f"atom_selection={atom_selection!r} in {pdb_path}"
        )
    if max_atoms is not None and coords.shape[0] > int(max_atoms):
        stride = max(1, coords.shape[0] // int(max_atoms))
        coords = coords[::stride][: int(max_atoms)]
    return torch.as_tensor(np.asarray(coords), dtype=torch.float32)


def _load_chain_residue_atom(
    pdb_path: str,
    chain_id: str,
    res_id: int,
    atom_name: str = "CA",
) -> torch.Tensor:
    """Load the coordinate of exactly one named atom of one residue from a
    PDB/mmCIF file -- used by TargetAnchorDistance to pick out a single
    "anchor" atom (e.g. a membrane-proximal residue) on a chain that is never
    part of the diffused structure, the same chain_pdb file MinimalOverlapPotential
    already loads for that motif's attached receptor.
    """
    import biotite.structure.io as strucio
    import numpy as np

    path = Path(pdb_path)
    if not path.is_file():
        raise FileNotFoundError(f"target_anchor_distance chain_pdb not found: {pdb_path}")

    atom_array = strucio.load_structure(str(path), model=1)
    sel_mask = (
        (atom_array.chain_id == str(chain_id))
        & (atom_array.res_id == int(res_id))
        & (atom_array.atom_name == str(atom_name))
    )
    coords = atom_array.coord[sel_mask]
    if coords.shape[0] != 1:
        raise ValueError(
            f"target_anchor_distance: expected exactly 1 atom for "
            f"chain_id={chain_id!r} res_id={res_id!r} atom_name={atom_name!r} "
            f"in {pdb_path}, found {coords.shape[0]}"
        )
    return torch.as_tensor(np.asarray(coords[0]), dtype=torch.float32)


def _pairwise_clash_overlap(
    xyz_a: torch.Tensor,  # [D, Na, 3]
    xyz_b: torch.Tensor,  # [D, Nb, 3]
    clash_distance: float,
) -> torch.Tensor:
    """Soft steric overlap: sum of squared sphere-overlap depths.

    Zero wherever the closest a-b atom pair is >= clash_distance apart; grows
    smoothly (gradient defined everywhere, including exactly at the cutoff) as
    atoms interpenetrate. Not normalised by atom count -- callers combine
    several such terms and only normalise (via `.mean()` over the batch dim)
    at the very end, same as the rest of this file's potentials.
    """
    if xyz_a.shape[1] == 0 or xyz_b.shape[1] == 0:
        return xyz_a.new_zeros(())
    dists = torch.cdist(xyz_a, xyz_b)  # [D, Na, Nb]
    overlap = (clash_distance - dists).clamp_min(0.0).pow(2)
    return overlap.sum(dim=(-2, -1)).mean()


class MinimalOverlapPotential(BasePotential):
    """Minimise steric overlap of external chains rigidly attached to motifs.

    Generalises MotifPairAxisDot's "Fake-Kabsch" trick from a single body-frame
    axis vector to a full external chain's atom cloud. For each entry in
    ``motif_chains`` (one motif -> one attached chain, e.g. a bound receptor or
    other context chain that is *never* part of the diffused structure):

      1. Load the chain's atoms once from ``chain_pdb`` / ``chain_id``
         (``_load_chain_atoms``).  By default (``align=False``) these are
         assumed already co-registered in the same coordinate frame as the
         design's own reference/input PDB -- the common case where the chain
         lives in the same combined input structure as the motifs, just
         outside the contig.  Opt in with ``align=True`` (+ ``align_chain_id``)
         when the chain instead comes from a *different* structure (the
         MotifPairAxisDot receptor case): on first use this automatically
         Kabsch-fits ``align_chain_id`` atoms in that same file onto the
         motif's own reference atoms (``_kabsch_ref_to_current_rotation``),
         automating the offline superposition that had to be done by hand to
         produce MotifPairAxisDot's ``motif_axes``.  Either way the result is a
         constant point cloud in the design's reference frame, cached after
         first use (the reference geometry is fixed for the whole run, same
         assumption ``motif_axes`` already makes).
      2. Every step, transport that constant cloud into current-step world
         coordinates with the *same* per-motif rigid transform that tracks the
         motif's own atoms -- rotate about the motif's own reference COM (not
         the chain's own centroid, which would collapse the physical offset
         between them), then translate to the motif's current COM:
         ``current = (chain_ref - motif_ref_com) @ R_i + current_com_i`` --
         this is "aligning the chain to the Kabsch alignment".
      3. Accumulate soft steric overlap (``_pairwise_clash_overlap``) (a)
         between every pair of configured chains and (b) between each chain and
         the rest of the currently-generated structure (``protein_atom_filter``,
         excluding that chain's own parent motif atoms).

    Returns ``-weight * total_overlap`` (maximised => overlap minimised), the
    same sign convention as every other potential in this file.

    Gradient is projected to pure rigid rotation per motif (no translation),
    so this only ever changes *orientation* -- never fights
    motif_distance / motif_spherical_position -- matching MotifPairAxisDot /
    MotifPairOrientation.

    Registered as both ``minimal_overlap`` and ``minimal_clashes`` (same
    class).
    """

    def __init__(
        self,
        weight: float = 1.0,
        motif_chains: list | None = None,
        clash_distance: float = 4.0,
        protein_atom_filter: str = "CA",
        chain_chain_weight: float = 1.0,
        chain_protein_weight: float = 1.0,
        eps: float = 1e-6,
        debug_log: bool = False,
    ):
        super().__init__(weight)
        if not motif_chains:
            raise ValueError(
                "minimal_overlap requires motif_chains=[{motif_index, chain_pdb, "
                "chain_id, ...}, ...] -- the external chain(s) rigidly attached "
                "to each floating motif whose steric overlap should be minimised"
            )
        resolved: list[dict] = []
        for i, spec in enumerate(motif_chains):
            missing = [k for k in ("motif_index", "chain_pdb", "chain_id") if k not in spec]
            if missing:
                raise ValueError(
                    f"minimal_overlap: motif_chains[{i}] is missing {missing}"
                )
            entry = {
                "motif_index": int(spec["motif_index"]),
                "chain_pdb": str(spec["chain_pdb"]),
                "chain_id": str(spec["chain_id"]),
                "atom_selection": str(spec.get("atom_selection", "CA")),
                "max_atoms": spec.get("max_atoms"),
                "align": bool(spec.get("align", False)),
                "align_chain_id": spec.get("align_chain_id"),
                "align_atom_selection": str(spec.get("align_atom_selection", "CA")),
            }
            if entry["align"] and not entry["align_chain_id"]:
                raise ValueError(
                    f"minimal_overlap: motif_chains[{i}] sets align=True but is "
                    "missing 'align_chain_id'"
                )
            resolved.append(entry)
        self.motif_chains = resolved

        valid_filters = ("all", "real", "potential", "backbone", "CA")
        if protein_atom_filter not in valid_filters:
            raise ValueError(
                f"minimal_overlap protein_atom_filter must be one of {valid_filters}"
            )
        self.protein_atom_filter = protein_atom_filter
        self.clash_distance = float(clash_distance)
        self.chain_chain_weight = float(chain_chain_weight)
        self.chain_protein_weight = float(chain_protein_weight)
        self.eps = float(eps)
        self.debug_log = bool(debug_log)

        # Populated lazily on first compute(): per-entry constant chain point
        # cloud, expressed in the *design's* reference frame (same frame as
        # ref_xyz_i -- NOT re-centred on the chain's own centroid, since the
        # whole point is to preserve the chain's physical offset from its
        # parent motif). Loading and the optional offline alignment only need
        # to happen once -- the reference geometry is constant across the
        # batch and across the whole trajectory.
        self._chain_ref_xyz: dict[int, torch.Tensor] = {}
        self.skip_reason: str | None = None
        self.skip_detail: dict | None = None

    def _resolve_chain_ref_xyz(self, idx, spec, ref_xyz_i, device, dtype):
        if idx in self._chain_ref_xyz:
            return self._chain_ref_xyz[idx]

        chain_xyz = _load_chain_atoms(
            spec["chain_pdb"], spec["chain_id"], spec["atom_selection"], spec["max_atoms"]
        ).to(device=device, dtype=dtype)

        if spec["align"]:
            align_xyz = _load_chain_atoms(
                spec["chain_pdb"], spec["align_chain_id"], spec["align_atom_selection"], None
            ).to(device=device, dtype=dtype)
            if align_xyz.shape[0] != ref_xyz_i.shape[0]:
                raise ValueError(
                    f"minimal_overlap: motif_chains[{idx}] align_chain_id="
                    f"{spec['align_chain_id']!r} has {align_xyz.shape[0]} atoms "
                    f"but the design's motif reference has {ref_xyz_i.shape[0]} "
                    "-- align_atom_selection must produce a 1:1 atom "
                    "correspondence (same count and order) with the contig "
                    "motif block"
                )
            R_align = _kabsch_ref_to_current_rotation(
                align_xyz, ref_xyz_i.unsqueeze(0), self.eps
            )
            if R_align is None:
                raise ValueError(
                    f"minimal_overlap: motif_chains[{idx}] offline alignment is "
                    "degenerate (too few or co-linear align atoms)"
                )
            align_com = align_xyz.mean(dim=0)
            chain_xyz = (chain_xyz - align_com) @ R_align[0] + ref_xyz_i.mean(dim=0)

        self._chain_ref_xyz[idx] = chain_xyz
        return chain_xyz

    def compute(self, xyz, masks, metadata):
        self.skip_reason = None
        self.skip_detail = None

        motif_blocks = _motif_distance_blocks(masks, metadata, xyz.device)
        self.skip_detail = {"n_motif_blocks": len(motif_blocks)}
        if not motif_blocks:
            self.skip_reason = "no_motif_blocks"
            return xyz.new_zeros(())

        device, dtype = xyz.device, xyz.dtype
        current_chains: list[torch.Tensor] = []  # transported to this step, [D, N, 3]
        own_motif_masks: list[torch.Tensor] = []

        for idx, spec in enumerate(self.motif_chains):
            motif_i = spec["motif_index"]
            if motif_i < 0 or motif_i >= len(motif_blocks):
                self.skip_reason = f"motif_chains_{idx}_index_out_of_range"
                continue
            block_mask = motif_blocks[motif_i]
            current_xyz_i, ref_xyz_i = _motif_block_current_and_reference_xyz(
                xyz, masks, metadata, block_mask
            )
            if current_xyz_i is None or ref_xyz_i is None or ref_xyz_i.shape[0] < 3:
                self.skip_reason = f"motif_chains_{idx}_coords_missing"
                continue
            R = _kabsch_ref_to_current_rotation(ref_xyz_i, current_xyz_i, self.eps)
            if R is None:
                self.skip_reason = f"motif_chains_{idx}_frame_degenerate"
                continue

            chain_ref_xyz = self._resolve_chain_ref_xyz(idx, spec, ref_xyz_i, device, dtype)
            motif_ref_com = ref_xyz_i.mean(dim=0)  # [3] -- the pivot, not the chain's own COM
            current_com_i = current_xyz_i.mean(dim=1)  # [D, 3]
            # "Align the additional chain to this Kabsch alignment": the same
            # per-motif rigid transform that tracks the motif's own atoms
            # (rotate about the motif's reference COM, translate to its
            # current COM) is applied to the chain's constant body-frame
            # point cloud, preserving its physical offset from the motif.
            transported = (
                (chain_ref_xyz - motif_ref_com).unsqueeze(0) @ R
            ) + current_com_i.unsqueeze(1)  # [D, N_chain, 3]
            current_chains.append(transported)
            own_motif_masks.append(block_mask)

        if len(current_chains) == 0:
            self.skip_reason = self.skip_reason or "no_active_chains"
            return xyz.new_zeros(())

        total = xyz.new_zeros(())
        n_terms = 0

        # (a) chain vs chain
        for i in range(len(current_chains)):
            for j in range(i + 1, len(current_chains)):
                total = total + self.chain_chain_weight * _pairwise_clash_overlap(
                    current_chains[i], current_chains[j], self.clash_distance
                )
                n_terms += 1

        # (b) chain vs the rest of the currently-generated structure (own
        # parent motif excluded -- a chain rigidly riding on its motif isn't
        # meaningfully "clashing" with the thing it's attached to)
        protein_mask = _atom_filter_mask(self.protein_atom_filter, masks, device)
        for i, chain_xyz in enumerate(current_chains):
            other_mask = protein_mask & ~own_motif_masks[i]
            other_xyz = xyz[:, other_mask, :]
            total = total + self.chain_protein_weight * _pairwise_clash_overlap(
                chain_xyz, other_xyz, self.clash_distance
            )
            n_terms += 1

        if self.debug_log:
            print(
                "[minimal_overlap] "
                + repr({"total_overlap": round(float(total), 4), "n_terms": n_terms}),
                file=sys.stderr,
                flush=True,
            )

        return -self.weight * total

    def guide_atom_mask(self, masks, metadata, device):
        motif_blocks = _motif_distance_blocks(masks, metadata, device)
        if not motif_blocks:
            any_mask = next(iter(masks.values()))
            return torch.zeros_like(any_mask, dtype=torch.bool, device=device)
        guide_mask = torch.zeros_like(motif_blocks[0], dtype=torch.bool, device=device)
        for spec in self.motif_chains:
            motif_i = spec["motif_index"]
            if 0 <= motif_i < len(motif_blocks):
                guide_mask = guide_mask | motif_blocks[motif_i]
        return guide_mask

    def transform_atom_gradient(self, atom_grad, masks, metadata, xyz):
        motif_blocks = _motif_distance_blocks(masks, metadata, xyz.device)
        if not motif_blocks:
            return torch.zeros_like(atom_grad)
        transformed = torch.zeros_like(atom_grad)
        seen: set[int] = set()
        for spec in self.motif_chains:
            motif_i = spec["motif_index"]
            if not (0 <= motif_i < len(motif_blocks)) or motif_i in seen:
                continue
            seen.add(motif_i)
            block_mask = motif_blocks[motif_i]
            if block_mask.sum().item() < 3:
                continue
            transformed[:, block_mask, :] = _project_gradient_to_rigid_body(
                atom_grad[:, block_mask, :],
                xyz[:, block_mask, :],
                allow_translation=False,
                allow_rotation=True,
            )
        return transformed


class TargetAnchorDistance(BasePotential):
    """Harmonic restraint on the distance between one specific atom on each
    of two RECEPTOR chains -- e.g. the membrane-proximal residue of each
    ectodomain -- rather than the motif's own center of mass (motif_distance)
    or any atom of the motif itself (as an earlier, now-removed version of
    this potential did).

    The receptor is never part of the diffused structure -- exactly the
    situation MinimalOverlapPotential already handles for its clash cloud.
    This potential reuses that same mechanism for a single point instead of a
    full cloud: for each ``motif_chains`` entry (one motif -> one attached
    receptor chain), the target residue's CA (``target_residue`` /
    ``target_atom``) is loaded once from ``chain_pdb`` (optionally corrected
    into the design's reference frame via ``align`` / ``align_chain_id`` --
    same offline-registration mechanism and same reasoning as
    MinimalOverlapPotential, see its docstring), then transported every step
    with the *same* per-motif Kabsch rotation that tracks the motif's own
    atoms: ``anchor_current = (anchor_ref - motif_ref_com) @ R + motif_current_com``.

    Because the transported anchor position is a differentiable function of
    the *whole* motif block's current coordinates (through the Kabsch fit and
    the current-COM translation), autograd's raw gradient is already
    naturally spread across the block -- unlike a potential that reads one
    atom directly out of ``xyz``, there is no single-atom-dilution concern
    here, so transform_atom_gradient can project the true per-block gradient
    directly with ``_project_gradient_to_rigid_body``, the same pattern
    MinimalOverlapPotential itself uses.

    ``allow_translation`` / ``allow_rotation`` (both default True) select
    which rigid-body degrees of freedom this potential is allowed to use to
    reduce the anchor distance. Reaching a single target point can always be
    done by pure translation, but when the receptor's target_residue sits far
    from the motif's own reference COM (a long lever arm), a small rotation
    can move that distant point a lot for very little RMS atom displacement
    -- often more efficient than translating the whole rigid body. Turn OFF
    whichever DOF another active potential on the same motif pair already
    owns, to avoid two potentials fighting over the same DOF with different
    objectives: set ``allow_translation=False`` if motif_distance /
    symmetry_motif_distance is also active on this pair, and
    ``allow_rotation=False`` if motif_pair_axis_dot (or another rotation-only
    potential) is also active on this pair.

    Returns -weight * (distance - target_distance)^2, same convention as
    MotifDistance / MinimalOverlapPotential.

    Registered as ``target_anchor_distance``.
    """

    def __init__(
        self,
        weight: float = 1.0,
        motif_chains: list | None = None,
        target_distance: float = 8.0,
        allow_translation: bool = True,
        allow_rotation: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__(weight)
        if not motif_chains or len(motif_chains) != 2:
            raise ValueError(
                "target_anchor_distance requires exactly 2 motif_chains=["
                "{motif_index, chain_pdb, chain_id, target_residue, ...}, "
                "{...}] -- the two receptor anchor points whose distance "
                "should be restrained"
            )
        resolved: list[dict] = []
        for i, spec in enumerate(motif_chains):
            missing = [
                k for k in ("motif_index", "chain_pdb", "chain_id", "target_residue")
                if k not in spec
            ]
            if missing:
                raise ValueError(
                    f"target_anchor_distance: motif_chains[{i}] is missing {missing}"
                )
            entry = {
                "motif_index": int(spec["motif_index"]),
                "chain_pdb": str(spec["chain_pdb"]),
                "chain_id": str(spec["chain_id"]),
                "target_residue": int(spec["target_residue"]),
                "target_atom": str(spec.get("target_atom", "CA")),
                "align": bool(spec.get("align", False)),
                "align_chain_id": spec.get("align_chain_id"),
                "align_atom_selection": str(spec.get("align_atom_selection", "CA")),
            }
            if entry["align"] and not entry["align_chain_id"]:
                raise ValueError(
                    f"target_anchor_distance: motif_chains[{i}] sets align=True "
                    "but is missing 'align_chain_id'"
                )
            resolved.append(entry)
        self.motif_chains = resolved
        self.target_distance = float(target_distance)
        self.allow_translation = bool(allow_translation)
        self.allow_rotation = bool(allow_rotation)
        self.eps = float(eps)

        # Populated lazily on first compute(): per-entry constant anchor point
        # in the design's reference frame (same caching assumption as
        # MinimalOverlapPotential._chain_ref_xyz -- the reference geometry is
        # fixed for the whole run).
        self._anchor_ref: dict[int, torch.Tensor] = {}
        self.skip_reason: str | None = None

    def _resolve_anchor_ref(self, idx, spec, ref_xyz_i, device, dtype):
        if idx in self._anchor_ref:
            return self._anchor_ref[idx]

        anchor_xyz = _load_chain_residue_atom(
            spec["chain_pdb"], spec["chain_id"], spec["target_residue"], spec["target_atom"]
        ).to(device=device, dtype=dtype)

        if spec["align"]:
            align_xyz = _load_chain_atoms(
                spec["chain_pdb"], spec["align_chain_id"], spec["align_atom_selection"], None
            ).to(device=device, dtype=dtype)
            if align_xyz.shape[0] != ref_xyz_i.shape[0]:
                raise ValueError(
                    f"target_anchor_distance: motif_chains[{idx}] align_chain_id="
                    f"{spec['align_chain_id']!r} has {align_xyz.shape[0]} atoms "
                    f"but the design's motif reference has {ref_xyz_i.shape[0]} "
                    "-- align_atom_selection must produce a 1:1 atom "
                    "correspondence (same count and order) with the contig "
                    "motif block"
                )
            R_align = _kabsch_ref_to_current_rotation(
                align_xyz, ref_xyz_i.unsqueeze(0), self.eps
            )
            if R_align is None:
                raise ValueError(
                    f"target_anchor_distance: motif_chains[{idx}] offline "
                    "alignment is degenerate (too few or co-linear align atoms)"
                )
            align_com = align_xyz.mean(dim=0)
            anchor_xyz = (anchor_xyz - align_com) @ R_align[0] + ref_xyz_i.mean(dim=0)

        self._anchor_ref[idx] = anchor_xyz
        return anchor_xyz

    def _transported_anchor(self, idx, spec, xyz, masks, metadata, device, dtype):
        block_mask = _single_motif_block_mask(spec["motif_index"], masks, metadata, device)
        if block_mask is None or block_mask.sum().item() < 3:
            return None, None
        current_xyz_i, ref_xyz_i = _motif_block_current_and_reference_xyz(
            xyz, masks, metadata, block_mask
        )
        if current_xyz_i is None or ref_xyz_i is None or ref_xyz_i.shape[0] < 3:
            return None, None
        R = _kabsch_ref_to_current_rotation(ref_xyz_i, current_xyz_i, self.eps)
        if R is None:
            return None, None

        anchor_ref = self._resolve_anchor_ref(idx, spec, ref_xyz_i, device, dtype)
        motif_ref_com = ref_xyz_i.mean(dim=0)
        current_com_i = current_xyz_i.mean(dim=1)  # [D, 3]
        # R is [D, 3, 3] -- genuinely different per design in the batch (each
        # design has its own current pose). The offset must be rotated by
        # EACH design's own R, not a single shared one -- broadcast [1, 1, 3]
        # against [D, 3, 3] to get a per-design [D, 1, 3] result, mirroring
        # MinimalOverlapPotential's own (chain_ref_xyz - motif_ref_com).unsqueeze(0) @ R
        # pattern. A prior version of this line used R[0] (design 0's
        # rotation only) for every design in the batch -- correct only by
        # coincidence when D=1, and silently wrong for D>1.
        anchor_offset = (anchor_ref - motif_ref_com).unsqueeze(0).unsqueeze(0)  # [1, 1, 3]
        anchor_current = (anchor_offset @ R).squeeze(1) + current_com_i  # [D, 3]
        return anchor_current, block_mask

    def compute(self, xyz, masks, metadata):
        self.skip_reason = None
        device, dtype = xyz.device, xyz.dtype

        anchors = []
        for idx, spec in enumerate(self.motif_chains):
            anchor_current, _ = self._transported_anchor(
                idx, spec, xyz, masks, metadata, device, dtype
            )
            if anchor_current is None:
                self.skip_reason = f"motif_chains_{idx}_coords_missing"
                return xyz.new_zeros(())
            anchors.append(anchor_current)

        dist = (anchors[0] - anchors[1]).norm(dim=-1)
        return -self.weight * ((dist - self.target_distance) ** 2).mean()

    def guide_atom_mask(self, masks, metadata, device):
        guide_mask = None
        for spec in self.motif_chains:
            block_mask = _single_motif_block_mask(spec["motif_index"], masks, metadata, device)
            if block_mask is None:
                continue
            guide_mask = block_mask if guide_mask is None else (guide_mask | block_mask)
        if guide_mask is None:
            any_mask = next(iter(masks.values()))
            return torch.zeros_like(any_mask, dtype=torch.bool, device=device)
        return guide_mask

    def transform_atom_gradient(self, atom_grad, masks, metadata, xyz):
        device = xyz.device
        transformed = torch.zeros_like(atom_grad)
        for spec in self.motif_chains:
            block_mask = _single_motif_block_mask(spec["motif_index"], masks, metadata, device)
            if block_mask is None or block_mask.sum().item() < 3:
                continue
            transformed[:, block_mask, :] = _project_gradient_to_rigid_body(
                atom_grad[:, block_mask, :],
                xyz[:, block_mask, :],
                allow_translation=self.allow_translation,
                allow_rotation=self.allow_rotation,
            )
        return transformed


POTENTIAL_REGISTRY: dict[str, type[BasePotential]] = {
    "binder_ROG": BinderROG,
    "monomer_ROG": MonomerROG,
    "interface_ncontacts": InterfaceNContacts,
    "monomer_contacts": MonomerContacts,
    "atom_pair_distance": AtomPairDistance,
    "motif_distance": MotifDistance,
    "motif_internal_rotation": MotifInternalRotation,
    "motif_relative_pose": MotifRelativePose,
    "rigid_pose": MotifRelativePose,
    "motif_rigid_body_pose": MotifRigidBodyPose,
    "rigid_body_pose": MotifRigidBodyPose,
    "motif_bridge": MotifBridge,
    "motif_bridge2": MotifBridge2,
    "motif_rigid": MotifRigid,
    "motif_com_distance": MotifCOMDistance,
    "motif_spherical_position": MotifSphericalPosition,
    "motif_radial_orientation": MotifRadialOrientationPotential,
    "motif_forbidden_radial_orientation": MotifForbiddenRadialOrientation,
    "motif_forbidden_radial_orientation_compact": MotifForbiddenRadialOrientationCompact,
    "motif_pair_orientation": MotifPairOrientation,
    "motif_pair_axis_dot": MotifPairAxisDot,
    "minimal_overlap": MinimalOverlapPotential,
    "target_anchor_distance": TargetAnchorDistance,
    "minimal_clashes": MinimalOverlapPotential,
    "symmetry_motif_distance": SymmetryAwareMotifDistance,
    "symmetry_motif_bridge": SymmetryAwareMotifBridge,
    "symmetry_single_motif_bridge": SymmetryAwareSingleMotifBridge,
    "symmetry_ellipsoid_bridge": SymmetryEllipsoidBridge,
    "symmetry_motif_center_distance": SymmetryAwareMotifCenterDistance,
    "symmetry_motif_radial_position": SymmetryAwareMotifRadialPosition,
    "symmetry_motif_radial_orientation": SymmetryAwareMotifRadialOrientation,
    "symmetry_motif_com_distance": SymmetryAwareMotifCOMDistance,
    "symmetry_motif_com_radial_position": SymmetryAwareMotifCOMRadialPosition,
    "symmetry_motif_com_radial_orientation": SymmetryAwareMotifCOMRadialOrientation,
    "symmetry_motif_axis_position": SymmetryAwareMotifAxisPosition,
    "symmetry_motif_inter_instance_distance": SymmetryAwareInterInstanceMotifDistance,
}

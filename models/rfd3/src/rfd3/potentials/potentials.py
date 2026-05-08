"""Differentiable scalar potentials for RFD3 inference guidance.

Each potential returns a scalar tensor that should be MAXIMIZED by gradient ascent.
The guidance system backpropagates through this scalar to get per-atom coordinate
gradients, which are then used to update the sampler trajectory.
"""

from __future__ import annotations

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
        eps: float = 1e-6,
    ):
        super().__init__(weight)
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
    "motif_rigid": MotifRigid,
    "motif_com_distance": MotifCOMDistance,
    "motif_spherical_position": MotifSphericalPosition,
    "motif_radial_orientation": MotifRadialOrientationPotential,
}

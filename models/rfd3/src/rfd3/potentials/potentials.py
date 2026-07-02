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
    "motif_bridge2": MotifBridge2,
    "motif_rigid": MotifRigid,
    "motif_com_distance": MotifCOMDistance,
    "motif_spherical_position": MotifSphericalPosition,
    "motif_radial_orientation": MotifRadialOrientationPotential,
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

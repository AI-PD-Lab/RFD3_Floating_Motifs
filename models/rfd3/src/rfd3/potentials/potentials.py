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
        motif_blocks = _motif_blocks(masks, metadata, xyz.device)
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

    atom_to_token_map = metadata["atom_to_token_map"].to(device=device, dtype=torch.long)
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
    "motif_bridge": MotifBridge,
    "motif_rigid": MotifRigid,
}

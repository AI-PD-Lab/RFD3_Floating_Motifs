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
    """Harmonic restraint on the center distance between two fixed motif blocks.

    The motif blocks are inferred from contiguous fixed-token runs in contig
    order.  For a contig like ``A10-20,50,A80-90``, motif_i=0 selects A10-20
    and motif_j=1 selects A80-90.

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
        motif_blocks = _fixed_motif_blocks(masks, metadata, xyz.device)
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


def _fixed_motif_blocks(
    masks: dict[str, torch.Tensor],
    metadata: dict,
    device: torch.device,
) -> list[torch.Tensor]:
    """Return atom masks for contiguous fixed-token motif blocks."""
    if "atom_to_token_map" not in metadata:
        return []
    if "fixed_atom_mask" not in masks or "potential_atom_mask" not in masks:
        return []

    atom_to_token_map = metadata["atom_to_token_map"].to(device=device, dtype=torch.long)
    fixed_mask = masks["fixed_atom_mask"].to(device=device, dtype=torch.bool)
    potential_mask = masks["potential_atom_mask"].to(device=device, dtype=torch.bool)
    motif_atom_mask = fixed_mask & potential_mask
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
}

from typing import Optional, Tuple, Dict

import numpy as np
from matscipy.neighbours import neighbour_list
import torch
from e3nn.util.jit import compile_mode


@compile_mode("script")
class NeighborModule(torch.nn.Module):
    def __init__(self, r_max: float):
        super().__init__()
        self.r_max = r_max

    def forward(
        self,
        positions: torch.Tensor,
        box: torch.Tensor,
        periodic: bool = True,
        full_list: bool = False,  # include symmetric pairs
        sort: bool = True,  # sort neighbors by distance
        true_self_interaction=False,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute neighbor list indices, distances, and displacement info using PyTorch only.

        Returns:
            edge_index: [2, num_pairs]
            distances: [num_pairs]
            vectors: [num_pairs, 3]
            unit_shifts: [num_pairs, 3]
            shifts: [num_pairs, 3] = unit_shifts @ box
            cell: [3,3] (same as input box)
        """
        N = positions.size(0)

        # Compute displacement vectors
        disp = positions.view(N, 1, 3) - positions.view(1, N, 3)
        if periodic:
            box_inv = torch.inverse(box)
            frac_disp = torch.matmul(disp, box_inv)
            frac_disp = frac_disp - frac_disp.round()
            disp = torch.matmul(frac_disp, box)

        # Pairwise distances
        distances = torch.norm(disp, dim=2)

        # Mask out self-pairs
        mask = torch.ones_like(distances, dtype=torch.bool)
        if not true_self_interaction:
            mask.fill_diagonal_(False)
        mask &= distances <= self.r_max

        indices = mask.nonzero(as_tuple=False)  # [num_pairs, 2]
        neighbor_indices = indices
        neighbor_vectors = disp[indices[:, 0], indices[:, 1]]
        neighbor_distances = distances[indices[:, 0], indices[:, 1]]

        # For half list (no symmetric duplicates)
        if not full_list:
            keep_mask = neighbor_indices[:, 0] < neighbor_indices[:, 1]
            neighbor_indices = neighbor_indices[keep_mask]
            neighbor_vectors = neighbor_vectors[keep_mask]
            neighbor_distances = neighbor_distances[keep_mask]

        # Sort neighbors by distance
        if sort and neighbor_distances.numel() > 0:
            sorted_distances, sort_idx = torch.sort(neighbor_distances)
            assert torch.allclose(neighbor_distances[sort_idx], sorted_distances)
            neighbor_distances = sorted_distances
            neighbor_indices = neighbor_indices[sort_idx]
            neighbor_vectors = neighbor_vectors[sort_idx]
            assert torch.allclose(
                torch.norm(neighbor_vectors, dim=1), neighbor_distances
            )

        if periodic:
            box_inv = torch.inverse(box)
            unit_shifts = torch.round(torch.matmul(neighbor_vectors, box_inv)).to(
                torch.int
            )
            shifts = torch.matmul(unit_shifts.to(box.dtype), box)
        else:
            unit_shifts = torch.zeros_like(neighbor_vectors, dtype=torch.int)
            shifts = neighbor_vectors

        return {
            "edge_index": neighbor_indices.T,  # [2, num_pairs]
            "distances": neighbor_distances,
            "vectors": neighbor_vectors,
            "unit_shifts": unit_shifts,
            "shifts": shifts,
        }


def get_neighborhood(
    positions: np.ndarray,  # [num_positions, 3]
    cutoff: float,
    pbc: Optional[Tuple[bool, bool, bool]] = None,
    cell: Optional[np.ndarray] = None,  # [3, 3]
    true_self_interaction=False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    assert true_self_interaction == False
    nl = NeighborModule(cutoff)
    info = nl(positions, cell, pbc, True, True)
    return info["edge_index"], info["shifts"], info["unit_shifts"], cell

    if pbc is None:
        pbc = (False, False, False)

    if cell is None or cell.any() == np.zeros((3, 3)).any():
        cell = np.identity(3, dtype=float)

    assert len(pbc) == 3 and all(isinstance(i, (bool, np.bool_)) for i in pbc)
    assert cell.shape == (3, 3)

    pbc_x = pbc[0]
    pbc_y = pbc[1]
    pbc_z = pbc[2]
    identity = np.identity(3, dtype=float)
    max_positions = np.max(np.absolute(positions)) + 1
    # Extend cell in non-periodic directions
    # For models with more than 5 layers, the multiplicative constant needs to be increased.
    # temp_cell = np.copy(cell)
    if not pbc_x:
        cell[0, :] = max_positions * 5 * cutoff * identity[0, :]
    if not pbc_y:
        cell[1, :] = max_positions * 5 * cutoff * identity[1, :]
    if not pbc_z:
        cell[2, :] = max_positions * 5 * cutoff * identity[2, :]

    sender, receiver, unit_shifts = neighbour_list(
        quantities="ijS",
        pbc=pbc,
        cell=cell,
        positions=positions,
        cutoff=cutoff,
        # self_interaction=True,  # we want edges from atom to itself in different periodic images
        # use_scaled_positions=False,  # positions are not scaled positions
    )

    if not true_self_interaction:
        # Eliminate self-edges that don't cross periodic boundaries
        true_self_edge = sender == receiver
        true_self_edge &= np.all(unit_shifts == 0, axis=1)
        keep_edge = ~true_self_edge

        # Note: after eliminating self-edges, it can be that no edges remain in this system
        sender = sender[keep_edge]
        receiver = receiver[keep_edge]
        unit_shifts = unit_shifts[keep_edge]

    # Build output
    edge_index = np.stack((sender, receiver))  # [2, n_edges]

    # From the docs: With the shift vector S, the distances D between atoms can be computed from
    # D = positions[j]-positions[i]+S.dot(cell)
    shifts = np.dot(unit_shifts, cell)  # [n_edges, 3]

    return edge_index, shifts, unit_shifts, cell

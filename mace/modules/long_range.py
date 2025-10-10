import torch
from typing import Dict, Optional, Tuple
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute neighbor list indices and distances using PyTorch only.

        Args:
            positions: [N,3] tensor of atomic positions
            box: [3,3] tensor of simulation cell vectors
            periodic: whether to apply periodic boundary conditions
            full_list: if True, include both (i,j) and (j,i) pairs
            sort: if True, sort neighbors by distance (ascending)

        Returns:
            neighbor_indices: LongTensor [num_pairs,2]
            neighbor_distances: Float64Tensor [num_pairs]
        """
        N = positions.size(0)

        # Compute displacement vectors
        disp = positions.view(N, 1, 3) - positions.view(1, N, 3)
        if periodic:
            box_inv = torch.inverse(box)
            frac_disp = torch.matmul(disp, box_inv)
            frac_disp = frac_disp - frac_disp.round()
            disp = torch.matmul(frac_disp, box)

        # Compute pairwise distances
        distances = torch.norm(disp, dim=2)

        # Mask out self-pairs
        mask = torch.ones_like(distances, dtype=torch.bool)
        mask.fill_diagonal_(0)

        # Mask out pairs beyond cutoff if not using full list
        # if not full_list:
        mask = mask & (distances <= self.r_max)

        # TorchScript-compatible nonzero
        indices = mask.nonzero()  # [num_pairs,2]
        neighbor_indices = indices
        neighbor_distances = distances[indices[:, 0], indices[:, 1]]

        # For half list, remove symmetric duplicates: keep only i < j
        if not full_list:
            keep_mask = neighbor_indices[:, 0] < neighbor_indices[:, 1]
            neighbor_indices = neighbor_indices[keep_mask]
            neighbor_distances = neighbor_distances[keep_mask]

        # Sort neighbors by distance if requested
        if sort and neighbor_distances.numel() > 0:
            sorted_distances, sort_idx = torch.sort(neighbor_distances)
            neighbor_indices = neighbor_indices[sort_idx]
            neighbor_distances = sorted_distances

        return neighbor_indices, neighbor_distances


class LongRange(torch.nn.Module):
    def __init__(self, pme_arguments: Dict):
        super().__init__()
        # sanity checks
        if not isinstance(pme_arguments, dict):
            raise TypeError("'pme_arguments' must be a dictionary.")

        if "lr_wavelength" not in pme_arguments:
            raise ValueError("Must specify 'lr_wavelength'.")


class ChargeCharge(LongRange):
    def __init__(self, pme_arguments: Dict):
        super().__init__(pme_arguments)
        try:
            from torchpme import EwaldCalculator
        except ImportError as exc:
            raise ImportError(
                "Cannot import 'EwaldCalculator'. Please install the 'torch-pme' library from https://github.com/lab-cosmo/torch-pme.git."
            ) from exc

        try:
            from torchpme import CoulombPotential
        except ImportError as exc:
            raise ImportError(
                "Cannot import 'CoulombPotential'. Please install the 'torch-pme' library from https://github.com/lab-cosmo/torch-pme.git."
            ) from exc

        # set up PME calculator
        potential = CoulombPotential(
            smearing=pme_arguments.get("smearing", None),
            exclusion_radius=pme_arguments.get("exclusion_radius", None),
            exclusion_degree=pme_arguments.get("exclusion_radius", 1),
        )

        self.pme = EwaldCalculator(
            potential=potential,
            lr_wavelength=pme_arguments.get("lr_wavelength", None),
            full_neighbor_list=pme_arguments.get("full_neighbor_list", False),
            prefactor=pme_arguments.get("prefactor", 1.0),
        )

    def forward(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:

        # electrostatic potential
        pot = self.pme(
            data["charges"][:, None],  # [Natoms,Nchannels = 1]
            data["cell"],  # [3,3]
            data["positions"],  # [Natoms,3]
            data["neighbor_indices"],  # [num_neighbor_i,2]
            data["neighbor_distances"],  # [num_neighbor_i,]
        )

        # [Natoms,Nchannels] --> [Natoms] since we have only once channel
        pot = pot[:, 0]

        # electrostatic (atomic) energy
        return pot * data["charges"]


class ChargeDipole(LongRange):
    def __init__(self, pme_arguments: Optional[Dict] = None, **kwargs):
        super().__init__(pme_arguments)
        raise ValueError("'ChargeDipole' not implemented yet.")


class DipoleDipole(LongRange):
    def __init__(self, pme_arguments: Optional[Dict] = None, **kwargs):
        super().__init__(pme_arguments)
        raise ValueError("'DipoleDipole' not implemented yet.")

import torch
from typing import Dict, Optional, Tuple
import vesin.torch

@torch.jit.ignore
def get_neighborlist(
    r_max: float, positions: torch.Tensor, cell: torch.Tensor, periodic: torch.Tensor
)->Tuple[torch.Tensor,torch.Tensor]:
    nl = vesin.torch.NeighborList(cutoff=r_max, full_list=False)
    return nl.compute(points=positions, box=cell, periodic=bool(periodic), quantities="Pd")


class LongRange(torch.nn.Module):
    def __init__(self, pme_arguments: Optional[Dict], **kwargs):
        super().__init__(**kwargs)
        # sanity checks
        if not isinstance(pme_arguments, dict):
            raise TypeError("'pme_arguments' must be a dictionary.")

        if "lr_wavelength" not in pme_arguments:
            raise ValueError("Must specify 'lr_wavelength'.")


class ChargeCharge(LongRange):
    def __init__(self, pme_arguments: Optional[Dict] = None, **kwargs):
        super().__init__(pme_arguments, **kwargs)
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
            exclusion_radius=kwargs.get("exclusion_radius", None),
            exclusion_degree=kwargs.get("exclusion_radius", 1),
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

import torch
from typing import Dict, Optional
from e3nn.util.jit import compile_mode
from mace.modules.blocks import LinearReadoutBlock
from mace.modules.utils import get_edge_vectors_and_lengths

# ToDo:
# - mask out atoms with zero charge/dipole to speed up the calculations
# - reimplement DipoleDpole usign InversePowerLaw
# - implement ChargeDipole
# - autodiff ChargeDipole and check that it is the same as DipoleDipole


@compile_mode("script")
class PME(torch.nn.Module):
    """
    Particle Mesh Ewald.

    Class that adds a long-ranged Coulomb interaction based on Ewald summation to a short-ranged MACE model.
    You need to install 'torch-pme' (https://github.com/lab-cosmo/torch-pme.git).
    In the input file, you need to specify 'pme_arguments' as a JSON formatted string or the file path to a JSON file.
    'pme_arguments' must contain at least 'lr_wavelength'.
    You can also run 'tune_pme.py' by providing your dataset as input to get optimized parameters.
    Then provide in 'pme_arguments' the parameters computed by that script.
    Please, use the same cutoff radius in 'tune_pme.py' and in the input file for training!
    """

    def __init__(self, pme_arguments: Optional[Dict] = None, kwargs=Optional[Dict]):
        super().__init__()

        assert (
            pme_arguments is not None
        ), "Must provide 'pme_arguments' as dictionary, JSON formatted string or JSON file."

        # extract pme arguments
        if isinstance(pme_arguments, str):
            import os, json

            if os.path.isfile(pme_arguments):
                with open(pme_arguments, "r", encoding="utf-8") as f:
                    pme_arguments = json.load(f)
            else:
                try:
                    pme_arguments = json.loads(pme_arguments)
                except Exception: 
                    import ast

                    pme_arguments = ast.literal_eval(pme_arguments)
        elif isinstance(pme_arguments, dict):
            pass
        else:
            raise TypeError(
                "'pme_arguments' must be a dictionary, JSON formatted string or a JSON file."
            )

        # if "r_max" not in pme_arguments:
        pme_arguments["exclusion_radius"] = kwargs["r_max"]

        interactions = pme_arguments["interactions"]

        self.use_charges = False
        self.use_dipoles = False

        from e3nn import o3

        def get_readout():
            return LinearReadoutBlock(
                o3.Irreps("0e"),
                o3.Irreps("0e"),
                kwargs.get("cueq_config", None),
                kwargs.get("oeq_config", None),
            )

        self.interactions = torch.nn.ModuleList()
        self.extra_readouts = torch.nn.ModuleList()

        for interaction in interactions:
            interaction = str(interaction)

            if interaction == "q-q":
                from .long_range import ChargeCharge

                self.interactions.append(ChargeCharge(pme_arguments["q-q"]))
                self.extra_readouts.append(get_readout())
                self.use_charges = True

            elif interaction == "q-mu":
                from .long_range import ChargeDipole

                self.interactions.append(ChargeDipole(pme_arguments["q-mu"]))
                self.extra_readouts.append(get_readout())
                self.use_charges = True
                self.use_dipoles = True

            elif interaction == "mu-mu":
                from .long_range import DipoleDipole

                self.interactions.append(DipoleDipole(pme_arguments["mu-mu"]))
                self.extra_readouts.append(get_readout())
                self.use_dipoles = True

            else:
                raise ValueError(f"Interaction '{interaction}' is not implemented")

    def forward(
        self,
        data: Dict[str, torch.Tensor],  # input data
        results: Dict[str, torch.Tensor],  # short-range results
    ) -> Dict[str, torch.Tensor]:

        # ---------------------------- #
        # long-ranged Coulomb potential

        # sanity check
        unique_batches = torch.unique(data["batch"])  # Get unique batch indices

        assert data["batch"].ndim == 1
        assert data["positions"].shape[1] == 3
        assert data["batch"].shape[0] == data["positions"].shape[0]

        out_results = {
            "energy": torch.zeros(len(unique_batches)),
            "node_energy": torch.zeros(len(data["positions"])),
        }

        # loop over structures (batching is not supported yet in torch-pme)
        prev_edge = 0
        prev_Natoms = 0
        for i in unique_batches:

            # Create a mask for the i-th configuration
            mask = data["batch"] == i

            # number of atoms
            Natoms = int(sum(mask))
            n_edges = data["n_edges"][i]

            # structure properties
            cell = data["cell"][i * 3 : (i + 1) * 3, :]
            positions = data["positions"][mask, :]
            pbc = data["pbc"][i]
            edge_index = data["edge_index"][:, prev_edge:n_edges]
            shifts = data["shifts"][prev_edge:n_edges, :]

            # graph
            edge_mask = edge_index[0, :] < edge_index[1, :]
            edge_index = edge_index[:, edge_mask] - prev_Natoms
            shifts = shifts[edge_mask, :]

            # counters
            prev_edge += data["n_edges"][i]
            prev_Natoms += Natoms

            # input data for PME
            pme_data = {"cell": cell, "positions": positions, "pbc": pbc}

            charges = torch.empty(0)
            atomic_dipoles = torch.empty(0)

            if self.use_charges:
                charges = results["charges"][mask]
                pme_data["charges"] = charges
            if self.use_dipoles:
                atomic_dipoles = results["atomic_dipoles"][mask]
                pme_data["atomic_dipoles"] = atomic_dipoles

            # ToDo: filter these tensors to remove atoms with zero charge

            # sanity check
            assert cell.shape == (
                3,
                3,
            ), f"Error: 'cell' should have shape (3,3) but it has {cell.shape}"
            assert pbc.shape == (
                3,
            ), f"Error: 'pbc' should have shape (3,) but it has {pbc.shape}"
            assert positions.shape == (
                Natoms,
                3,
            ), f"Error: 'positions' should have shape ({Natoms},3) but it has {positions.shape}"
            # if charges are needed
            if self.use_charges:
                assert charges.shape == (
                    Natoms,
                ), f"Error: 'charges' should have shape ({Natoms},) but they have {charges.shape}"
            # if dipoles are needed
            if self.use_dipoles:
                assert atomic_dipoles.shape == (
                    Natoms,
                    3,
                ), f"Error: 'atomic_dipoles' should have shape ({Natoms},3) but it has {atomic_dipoles.shape}"

            # check autograd
            assert (
                positions.requires_grad
            ), "'positions should have attribute 'requires_grad' equal to True"

            vectors, distances = get_edge_vectors_and_lengths(
                positions=positions,
                edge_index=edge_index,
                shifts=shifts,
            )

            assert (
                distances.requires_grad
            ), "'neighbor_distances' should have attribute 'requires_grad' equal to True"
            assert (
                distances.dtype == positions.dtype
            ), f"'neighbor_distances' should have the same dtype as 'positions' but they have {distances} and {positions.dtype} respectively"

            pme_data["neighbor_indices"] = edge_index.T
            pme_data["neighbor_distances"] = distances[:, 0]
            pme_data["neighbor_vectors"] = vectors

            # electrostatic potential(s)
            for interaction, readout in zip(self.interactions, self.extra_readouts):

                atomic_energies = interaction(pme_data)

                # readout to atomic_energies
                atomic_energies = readout(atomic_energies[:, None])[:, 0]

                # update node_energy
                out_results["node_energy"][mask] = atomic_energies

                # compute total energy
                total_electrostatic_energy = torch.sum(atomic_energies)
                out_results["energy"][i] = total_electrostatic_energy

        return out_results


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
            exclusion_degree=pme_arguments.get("exclusion_degree", 1),
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
        try:
            from torchpme import CalculatorDipole
        except ImportError as exc:
            raise ImportError(
                "Cannot import 'CalculatorDipole'. Please install the 'torch-pme' library from https://github.com/lab-cosmo/torch-pme.git."
            ) from exc

        try:
            from torchpme import PotentialDipole
        except ImportError as exc:
            raise ImportError(
                "Cannot import 'PotentialDipole'. Please install the 'torch-pme' library from https://github.com/lab-cosmo/torch-pme.git."
            ) from exc

        # set up PME calculator
        potential = PotentialDipole(
            smearing=pme_arguments.get("smearing", None),
            exclusion_radius=pme_arguments.get("exclusion_radius", None),
            exclusion_degree=pme_arguments.get("exclusion_degree", 1),
        )

        self.pme = CalculatorDipole(
            potential=potential,
            lr_wavelength=pme_arguments.get("lr_wavelength", None),
            full_neighbor_list=False,
            prefactor=pme_arguments.get("prefactor", 1.0),
        )

    def forward(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:

        # electrostatic potential
        pot = self.pme(
            data["atomic_dipoles"],  # [Natoms,Nchannels = 1]
            data["cell"],  # [3,3]
            data["positions"],  # [Natoms,3]
            data["neighbor_indices"],  # [num_neighbor_i,2]
            data["neighbor_vectors"],  # [num_neighbor_i,]
        )

        # electrostatic (atomic) energy
        return torch.einsum("ij,ij->i", pot, data["atomic_dipoles"])

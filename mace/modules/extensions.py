from typing import Dict, List, Optional, Any

import torch
from e3nn.util.jit import compile_mode

from mace.modules.blocks import LinearReadoutBlock, NonLinearReadoutBlock
from mace.modules.models import ScaleShiftMACE, EnergyDipoleMACE
from mace.modules.utils import (
    get_atomic_virials_stresses,
    get_outputs,
    prepare_graph,
    neighbor_ranges,
)
from mace.modules.wrapper_ops import CuEquivarianceConfig
from mace.tools.scatter import scatter_sum

try:
    from .long_range import get_neighborlist
except:
    pass


def _copy_mace_readout(
    mace_readout: torch.nn.Module, cueq_config: Optional[CuEquivarianceConfig] = None
) -> torch.nn.Module:
    """
    Helper function to copy a MACE readout block.
    """
    if isinstance(mace_readout, LinearReadoutBlock):
        return LinearReadoutBlock(
            irreps_in=mace_readout.linear.irreps_in,  # type:ignore
            irrep_out=mace_readout.linear.irreps_out,  # type:ignore
            cueq_config=cueq_config,
        )
    if isinstance(mace_readout, NonLinearReadoutBlock):  # type:ignore
        return NonLinearReadoutBlock(
            irreps_in=mace_readout.linear_1.irreps_in,  # type:ignore
            MLP_irreps=mace_readout.hidden_irreps,
            gate=mace_readout.non_linearity._modules["acts"][  # pylint: disable=W0212
                0
            ].f,
            irrep_out=mace_readout.linear_2.irreps_out,  # type:ignore
            num_heads=mace_readout.num_heads,
            cueq_config=cueq_config,
        )
    raise TypeError("Unsupported readout type.")


def _get_readout_input_dim(block: torch.nn.Module) -> int:
    if isinstance(block, LinearReadoutBlock):
        return block.linear.irreps_in.dim  # type:ignore
    if isinstance(block, NonLinearReadoutBlock):  # type:ignore
        return block.linear_1.irreps_in.dim  # type:ignore
    raise TypeError("Unsupported readout type for input dimension retrieval.")


@compile_mode("script")
class MACELES(ScaleShiftMACE):
    def __init__(self, les_arguments: Optional[Dict] = None, **kwargs):
        super().__init__(**kwargs)
        try:
            from les import Les
        except ImportError as exc:
            raise ImportError(
                "Cannot import 'les'. Please install the 'les' library from https://github.com/ChengUCB/les."
            ) from exc
        if les_arguments is None:
            les_arguments = {"use_atomwise": False}
        self.compute_bec = les_arguments.get("compute_bec", False)
        self.bec_output_index = les_arguments.get("bec_output_index", None)
        self.les = Les(les_arguments=les_arguments)
        self.les_readouts = torch.nn.ModuleList()
        self.readout_input_dims = [
            _get_readout_input_dim(readout) for readout in self.readouts  # type:ignore
        ]
        cueq_config = kwargs.get("cueq_config", None)
        for readout in self.readouts:  # type:ignore
            self.les_readouts.append(
                _copy_mace_readout(readout, cueq_config=cueq_config)
            )

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        training: bool = False,
        compute_force: bool = True,
        compute_virials: bool = False,
        compute_stress: bool = False,
        compute_displacement: bool = False,
        compute_hessian: bool = False,
        compute_edge_forces: bool = False,
        compute_atomic_stresses: bool = False,
        lammps_mliap: bool = False,
        compute_bec: bool = False,
    ) -> Dict[str, Optional[torch.Tensor]]:
        ctx = prepare_graph(
            data,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
            compute_displacement=compute_displacement,
            lammps_mliap=lammps_mliap,
        )

        is_lammps = ctx.is_lammps
        num_atoms_arange = ctx.num_atoms_arange
        num_graphs = ctx.num_graphs
        displacement = ctx.displacement
        positions = ctx.positions
        vectors = ctx.vectors
        lengths = ctx.lengths
        cell = ctx.cell
        node_heads = ctx.node_heads
        interaction_kwargs = ctx.interaction_kwargs
        lammps_natoms = interaction_kwargs.lammps_natoms
        lammps_class = interaction_kwargs.lammps_class

        # Setting LES cell input to zero when boundary conditions are not periodic
        cell_les = cell.clone()
        pbc_tensor = data["pbc"].to(device=data["cell"].device)
        no_pbc_mask_cfg = ~pbc_tensor.any(dim=-1)
        no_pbc_mask_rows = no_pbc_mask_cfg.repeat_interleave(3)
        cell_les[no_pbc_mask_rows] = torch.zeros(
            (no_pbc_mask_rows.sum(), 3), dtype=cell_les.dtype, device=cell_les.device
        )

        # Atomic energies
        node_e0 = self.atomic_energies_fn(data["node_attrs"])[
            num_atoms_arange, node_heads
        ]
        e0 = scatter_sum(
            src=node_e0, index=data["batch"], dim=0, dim_size=num_graphs
        ).to(
            vectors.dtype
        )  # [n_graphs, num_heads]

        # Embeddings
        node_feats = self.node_embedding(data["node_attrs"])
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats, cutoff = self.radial_embedding(
            lengths, data["node_attrs"], data["edge_index"], self.atomic_numbers
        )

        if hasattr(self, "pair_repulsion"):
            pair_node_energy = self.pair_repulsion_fn(
                lengths, data["node_attrs"], data["edge_index"], self.atomic_numbers
            )
            if is_lammps:
                pair_node_energy = pair_node_energy[: lammps_natoms[0]]
        else:
            pair_node_energy = torch.zeros_like(node_e0)

        # Embeddings of additional features
        if hasattr(self, "joint_embedding"):
            embedding_features: Dict[str, torch.Tensor] = {}
            for name, _ in self.embedding_specs.items():
                embedding_features[name] = data[name]
            node_feats += self.joint_embedding(
                data["batch"],
                embedding_features,
            )
            if hasattr(self, "embedding_readout"):
                embedding_node_energy = self.embedding_readout(
                    node_feats, node_heads
                ).squeeze(-1)
                embedding_energy = scatter_sum(
                    src=embedding_node_energy,
                    index=data["batch"],
                    dim=0,
                    dim_size=num_graphs,
                )
                e0 += embedding_energy

        # Interactions
        node_es_list = [pair_node_energy]
        node_feats_list: List[torch.Tensor] = []
        node_qs_list: List[torch.Tensor] = []

        for i, (interaction, product) in enumerate(
            zip(self.interactions, self.products)
        ):
            node_attrs_slice = data["node_attrs"]
            if is_lammps and i > 0:
                node_attrs_slice = node_attrs_slice[: lammps_natoms[0]]
            node_feats, sc = interaction(
                node_attrs=node_attrs_slice,
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data["edge_index"],
                cutoff=cutoff,
                first_layer=(i == 0),
                lammps_class=lammps_class,
                lammps_natoms=lammps_natoms,
            )
            if is_lammps and i == 0:
                node_attrs_slice = node_attrs_slice[: lammps_natoms[0]]
            node_feats = product(
                node_feats=node_feats, sc=sc, node_attrs=node_attrs_slice
            )
            node_feats_list.append(node_feats)

        for i, (readout, les_readout) in enumerate(
            zip(self.readouts, self.les_readouts)
        ):
            feat_idx = -1 if len(self.readouts) == 1 else i
            node_es = readout(node_feats_list[feat_idx], node_heads)[
                num_atoms_arange, node_heads
            ]
            node_qs = les_readout(node_feats_list[feat_idx], node_heads)[
                num_atoms_arange, node_heads
            ]  # type:ignore
            node_qs_list.append(node_qs)
            node_es_list.append(node_es)

        node_feats_out = torch.cat(node_feats_list, dim=-1)
        node_inter_es = torch.sum(torch.stack(node_es_list, dim=0), dim=0)
        node_inter_es = self.scale_shift(node_inter_es, node_heads)
        inter_e = scatter_sum(node_inter_es, data["batch"], dim=-1, dim_size=num_graphs)

        total_energy = e0 + inter_e
        node_energy = node_e0.clone().double() + node_inter_es.clone().double()

        les_q = torch.sum(torch.stack(node_qs_list, dim=1), dim=1)
        les_result = self.les(
            latent_charges=les_q,
            positions=positions,
            cell=cell_les.view(-1, 3, 3),
            batch=data["batch"],
            compute_energy=True,
            compute_bec=(compute_bec or self.compute_bec),
            bec_output_index=self.bec_output_index,
        )
        les_energy_opt = les_result["E_lr"]
        if les_energy_opt is None:
            les_energy = torch.zeros_like(total_energy)
        else:
            les_energy = les_energy_opt
        total_energy += les_energy

        forces, virials, stress, hessian, edge_forces = get_outputs(
            energy=inter_e + les_energy,
            positions=positions,
            displacement=displacement,
            vectors=vectors,
            cell=cell,
            training=training,
            compute_force=compute_force,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
            compute_hessian=compute_hessian,
            compute_edge_forces=compute_edge_forces,
        )

        atomic_virials: Optional[torch.Tensor] = None
        atomic_stresses: Optional[torch.Tensor] = None
        if compute_atomic_stresses and edge_forces is not None:
            atomic_virials, atomic_stresses = get_atomic_virials_stresses(
                edge_forces=edge_forces,
                edge_index=data["edge_index"],
                vectors=vectors,
                num_atoms=positions.shape[0],
                batch=data["batch"],
                cell=cell,
            )
        return {
            "energy": total_energy,
            "node_energy": node_energy,
            "forces": forces,
            "edge_forces": edge_forces,
            "virials": virials,
            "stress": stress,
            "atomic_virials": atomic_virials,
            "atomic_stresses": atomic_stresses,
            "displacement": displacement,
            "hessian": hessian,
            "node_feats": node_feats_out,
            "les_energy": les_energy,
            "latent_charges": les_q,
            "BEC": les_result["BEC"],
        }


# @compile_mode("script")
class PME(torch.nn.Module):
    """
    Particle Mesh Ewald.

    Class that adds a long-ranged Coulomb interaction based on Ewald summation to a short-ranged MACE model.
    You need to install 'torch-pme' (https://github.com/lab-cosmo/torch-pme.git) and 'vesin' (pip install .[examples] in 'torch-pme').
    In the input file, you need to specify 'pme_arguments' as a JSON formatted string or the file path to a JSON file.
    'pme_arguments' must contain at least 'lr_wavelength'.
    You can also run 'tune_pme.py' by providing your dataset as input to get optimized parameters.
    Then provide in 'pme_arguments' the parameters computed by that script.
    Please, use the same cutoff radius in 'tune_pme.py' and in the input file for training!
    """

    def __init__(self, pme_arguments: Optional[Dict] = None, **kwargs):
        super().__init__()

        try:
            import vesin.torch  # used in forward method
        except ImportError as exc:
            raise ImportError(
                "Cannot import 'vesin.torch'. Please install it through 'pip install .[examples]' in 'torch-pme' library from https://github.com/lab-cosmo/torch-pme.git."
            ) from exc

        assert (
            pme_arguments is not None
        ), "Must provide 'pme_arguments' as dictionary, JSON formatted string or JSON file."

        # extract pme arguments
        if isinstance(pme_arguments, str):
            import os, json

            if os.path.isfile(pme_arguments):
                with open(pme_arguments, "r") as f:
                    pme_arguments = json.load(f)
            else:
                try:
                    pme_arguments = json.loads(pme_arguments)
                except:
                    import ast

                    pme_arguments = ast.literal_eval(pme_arguments)
        elif isinstance(pme_arguments, dict):
            pass
        else:
            raise TypeError(
                "'pme_arguments' must be a dictionary, JSON formatted string or a JSON file."
            )

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
            
        self.register_buffer("r_max", torch.tensor(kwargs["r_max"], dtype=torch.float64))

    def forward(
        self,
        data: Dict[str, torch.Tensor],  # input data
        results: Dict[str, torch.Tensor],  # short-range results
    ) -> Dict[str, torch.Tensor]:

        # ---------------------------- #
        # long-ranged Coulomb potential

        # sanity check
        unique_batches = torch.unique(data["batch"])  # Get unique batch indices
        assert len(unique_batches) == len(
            results["energy"]
        ), "Batch size mismatch between data and computed results"

        assert data["batch"].ndim == 1
        assert data["positions"].shape[1] == 3
        assert data["batch"].shape[0] == data["positions"].shape[0]

        # loop over structures (batching is not supported yet in torch-pme)
        for i in unique_batches:

            # Create a mask for the i-th configuration
            mask = data["batch"] == i

            # number of atoms
            Natoms = int(sum(mask))

            # structure properties
            cell = data["cell"][i * 3 : (i + 1) * 3, :]
            positions = data["positions"][mask, :]
            pbc = data["pbc"][i]

            pme_data = {"cell": cell, "positions": positions, "pbc": pbc}
            
            charges = torch.empty(0)
            atomic_dipoles = torch.empty(0)

            if self.use_charges:
                charges = data["oxn"][mask]
                pme_data["charges"] = charges
            if self.use_dipoles:
                atomic_dipoles = data["atomic_dipoles"][mask]
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

            # compute interatomic distances
            neighbor_indices, neighbor_distances = get_neighborlist(
                self.r_max, positions, cell, any(pbc)
            )

            assert (
                neighbor_distances.requires_grad
            ), "'neighbor_distances' should have attribute 'requires_grad' equal to True"
            assert (
                neighbor_distances.dtype == positions.dtype
            ), f"'neighbor_distances' should have the same dtype as 'positions' but they have {neighbor_distances.dtype} and {positions.dtype} respectively"

            pme_data["neighbor_distances"] = neighbor_distances
            pme_data["neighbor_indices"] = neighbor_indices

            # electrostatic potential(s)
            for interaction,readout in zip(self.interactions,self.extra_readouts):

                atomic_energies = interaction(pme_data)

                # readout to atomic_energies
                atomic_energies = readout(atomic_energies[:, None])[:, 0]

                # update node_energy
                results["node_energy"][mask] += atomic_energies

                # compute total energy
                total_electrostatic_energy = torch.sum(atomic_energies)
                results["energy"][i] += total_electrostatic_energy

        return results


@compile_mode("script")
class MACEPME(PME, ScaleShiftMACE):
    """
    ScaleShiftMACE + Particle Mesh Ewald.
    """

@compile_mode("script")
class EnergyDipoleMACEPME(EnergyDipoleMACE):
    """
    EnergyDipoleMACE + Particle Mesh Ewald.
    """
    
    def __init__(self,pme_arguments, **kwargs):
        super().__init__(**kwargs)
        self.pme_model = PME(pme_arguments,**kwargs)
        
    def forward(
        self,
        data: Dict[str, torch.Tensor],
        options: Optional[Dict[str, bool]] = {},
    ) -> Dict[str, torch.Tensor]:
        results = self.mace_model(data,options)
        results = self.pme_model(data,results)
        return results

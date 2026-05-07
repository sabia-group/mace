from typing import Any, Callable, Dict, List, Optional, Type

import numpy as np
import torch
from e3nn import o3
from e3nn.util.jit import compile_mode

from mace.tools.scatter import scatter_sum
from mace.tools.torch_tools import get_change_of_basis, spherical_to_cartesian

from .blocks import (
    AtomicEnergiesBlock,
    EquivariantProductBasisBlock,
    InteractionBlock,
    LinearDipolePolarReadoutBlock,
    LinearNodeEmbeddingBlock,
    NonLinearDipolePolarReadoutBlock,
    RadialEmbeddingBlock,
)
from .utils import (
    compute_dielectric_gradients_loop,
    get_edge_vectors_and_lengths,
    get_outputs,
    get_symmetric_displacement,
)


@compile_mode("script")
class EnergyDipoleMACE(torch.nn.Module):
    def __init__(
        self,
        r_max: float,
        num_bessel: int,
        num_polynomial_cutoff: int,
        max_ell: int,
        interaction_cls: Type[InteractionBlock],
        interaction_cls_first: Type[InteractionBlock],
        num_interactions: int,
        num_elements: int,
        hidden_irreps: o3.Irreps,
        MLP_irreps: o3.Irreps,
        avg_num_neighbors: float,
        atomic_numbers: List[int],
        correlation: int,
        gate: Optional[Callable],
        atomic_energies: Optional[np.ndarray],
        apply_cutoff: bool = True,  # pylint: disable=unused-argument
        use_reduced_cg: bool = True,  # pylint: disable=unused-argument
        use_so3: bool = False,  # pylint: disable=unused-argument
        distance_transform: str = "None",  # pylint: disable=unused-argument
        radial_MLP: Optional[List[int]] = None,
        cueq_config: Optional[Dict[str, Any]] = None,  # pylint: disable=unused-argument
        oeq_config: Optional[Dict[str, Any]] = None,  # pylint: disable=unused-argument
        edge_irreps: Optional[o3.Irreps] = None,  # pylint: disable=unused-argument
    ):
        super().__init__()
        self.register_buffer(
            "atomic_numbers", torch.tensor(atomic_numbers, dtype=torch.int64)
        )
        self.register_buffer("r_max", torch.tensor(r_max, dtype=torch.float64))
        self.register_buffer(
            "num_interactions", torch.tensor(num_interactions, dtype=torch.int64)
        )
        self.register_buffer("change_of_basis", get_change_of_basis())
        # Embedding
        node_attr_irreps = o3.Irreps([(num_elements, (0, 1))])
        node_feats_irreps = o3.Irreps([(hidden_irreps.count(o3.Irrep(0, 1)), (0, 1))])
        self.node_embedding = LinearNodeEmbeddingBlock(
            irreps_in=node_attr_irreps,
            irreps_out=node_feats_irreps,
            cueq_config=cueq_config,
        )
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
        )
        edge_feats_irreps = o3.Irreps(f"{self.radial_embedding.out_dim}x0e")

        assert max_ell >= 2, f"'max_ell' must be >= 2 but you provided {max_ell}."
        sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
        num_features = hidden_irreps.count(o3.Irrep(0, 1))
        interaction_irreps = (sh_irreps * num_features).sort()[0].simplify()
        self.spherical_harmonics = o3.SphericalHarmonics(
            sh_irreps, normalize=True, normalization="component"
        )
        if radial_MLP is None:
            radial_MLP = [64, 64, 64]
        # Interactions and readouts
        self.atomic_energies_fn = AtomicEnergiesBlock(atomic_energies)

        inter = interaction_cls_first(
            node_attrs_irreps=node_attr_irreps,
            node_feats_irreps=node_feats_irreps,
            edge_attrs_irreps=sh_irreps,
            edge_feats_irreps=edge_feats_irreps,
            target_irreps=interaction_irreps,
            hidden_irreps=hidden_irreps,
            avg_num_neighbors=avg_num_neighbors,
            radial_MLP=radial_MLP,
            cueq_config=cueq_config,
        )
        self.interactions = torch.nn.ModuleList([inter])

        # Use the appropriate self connection at the first layer
        use_sc_first = False
        if "Residual" in str(interaction_cls_first):
            use_sc_first = True

        node_feats_irreps_out = inter.target_irreps
        prod = EquivariantProductBasisBlock(
            node_feats_irreps=node_feats_irreps_out,
            target_irreps=hidden_irreps,
            correlation=correlation,
            num_elements=num_elements,
            use_sc=use_sc_first,
            cueq_config=cueq_config,
        )
        self.products = torch.nn.ModuleList([prod])

        self.readouts = torch.nn.ModuleList()
        self.readouts.append(
            LinearDipolePolarReadoutBlock(
                hidden_irreps, use_polarizability=True, cueq_config=cueq_config
            )
        )

        for i in range(num_interactions - 1):
            if i == num_interactions - 2:
                assert (
                    len(hidden_irreps) > 1
                ), "To predict dipoles use at least l=1 hidden_irreps"
                hidden_irreps_out = str(
                    hidden_irreps[:2]
                )  # Select scalars and l=1 vectors for last layer
            else:
                hidden_irreps_out = hidden_irreps
            inter = interaction_cls(
                node_attrs_irreps=node_attr_irreps,
                node_feats_irreps=hidden_irreps,
                edge_attrs_irreps=sh_irreps,
                edge_feats_irreps=edge_feats_irreps,
                target_irreps=interaction_irreps,
                hidden_irreps=hidden_irreps_out,
                avg_num_neighbors=avg_num_neighbors,
                radial_MLP=radial_MLP,
                cueq_config=cueq_config,
            )
            self.interactions.append(inter)
            prod = EquivariantProductBasisBlock(
                node_feats_irreps=interaction_irreps,
                target_irreps=hidden_irreps_out,
                correlation=correlation,
                num_elements=num_elements,
                use_sc=True,
                cueq_config=cueq_config,
            )
            self.products.append(prod)
            if i == num_interactions - 2:
                self.readouts.append(
                    NonLinearDipolePolarReadoutBlock(
                        hidden_irreps_out,
                        MLP_irreps,
                        gate,
                        use_polarizability=True,
                        cueq_config=cueq_config,
                    )
                )
            else:
                self.readouts.append(
                    LinearDipolePolarReadoutBlock(
                        hidden_irreps, use_polarizability=True, cueq_config=cueq_config
                    )
                )

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        training: bool = False,
        compute_force: bool = True,
        compute_virials: bool = False,
        compute_stress: bool = False,
        compute_displacement: bool = False,
        compute_bec: bool = False,
        compute_edge_forces: bool = False,  # pylint: disable=W0613
        compute_atomic_stresses: bool = False,  # pylint: disable=W0613
    ) -> Dict[str, Optional[torch.Tensor]]:
        # Setup
        data["node_attrs"].requires_grad_(True)
        data["positions"].requires_grad_(True)
        num_graphs = data["ptr"].numel() - 1
        num_nodes = data["positions"].shape[0]
        num_atoms_arange = torch.arange(num_nodes)
        displacement = torch.zeros(
            (num_graphs, 3, 3),
            dtype=data["positions"].dtype,
            device=data["positions"].device,
        )
        if compute_virials or compute_stress or compute_displacement:
            (
                data["positions"],
                data["shifts"],
                displacement,
            ) = get_symmetric_displacement(
                positions=data["positions"],
                unit_shifts=data["unit_shifts"],
                cell=data["cell"],
                edge_index=data["edge_index"],
                num_graphs=num_graphs,
                batch=data["batch"],
            )

        # Atomic energies
        node_e0 = self.atomic_energies_fn(data["node_attrs"])[
            num_atoms_arange, data["head"][data["batch"]]
        ]
        # e0 = scatter_sum(
        #     src=node_e0, index=data["batch"], dim=-1, dim_size=num_graphs
        # )  # [n_graphs,]

        # Embeddings
        node_feats = self.node_embedding(data["node_attrs"])
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=data["positions"],
            edge_index=data["edge_index"],
            shifts=data["shifts"],
        )
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats, cutoff = self.radial_embedding(
            lengths, data["node_attrs"], data["edge_index"], self.atomic_numbers
        )

        # Interactions
        n_components = 1 + 3 + 6
        num_interactions = len(self.interactions)
        attributes = torch.zeros(
            (num_nodes, num_interactions + 1, n_components),
            device=data["positions"].device,
        )  # [n_nodes,n_contributions,n_components]
        attributes[:, 0, 0] = node_e0
        for n, (interaction, product, readout) in enumerate(
            zip(self.interactions, self.products, self.readouts)
        ):
            node_feats, sc = interaction(
                node_attrs=data["node_attrs"],
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data["edge_index"],
                cutoff=cutoff,
            )
            node_feats = product(
                node_feats=node_feats,
                sc=sc,
                node_attrs=data["node_attrs"],
            )
            attributes[:, n + 1, :] = readout(node_feats).squeeze(-1)  # [n_nodes, ]

        # sum over all the contributions (interations)
        # [n_nodes,n_contributions,n_components] --> [n_nodes,n_components]
        node_features = torch.sum(attributes, dim=1)
        assert node_features.shape == (
            num_nodes,
            n_components,
        ), f"'node_features' has the wrong shape, expected {(num_nodes,n_components)} but got {node_features.shape}"
        node_dipole_baseline = data["positions"] * data["oxn"].unsqueeze(-1)
        node_features = torch.hstack((node_features, node_dipole_baseline))
        node_energy = node_features[:, 0]
        node_dipoles = node_features[:, 2:5] + node_features[:, n_components:]
        node_polarizability = torch.hstack(
            (node_features[:, 1][:, None], node_features[:, 5:n_components])
        )
        node_polarizability = spherical_to_cartesian(
            node_polarizability, self.change_of_basis
        )

        # gather over all nodes (atoms) belonging to the same graph (structure)
        # [n_nodes,n_components] --> [n_graphs,n_components]
        graph_features = scatter_sum(
            src=node_features, index=data["batch"], dim=0, dim_size=num_graphs
        )
        assert graph_features.shape == (
            num_graphs,
            n_components + 3,
        ), f"'graph_features' has the wrong shape, expected {(num_graphs,n_components+3)} but got {graph_features.shape}"
        total_energy = graph_features[:, 0]
        total_dipole = graph_features[:, 2:5] + graph_features[:, n_components:]
        total_polarizability = torch.hstack(
            (graph_features[:, 1][:, None], graph_features[:, 5:n_components])
        )
        total_polarizability = spherical_to_cartesian(
            total_polarizability, self.change_of_basis
        )

        # Attention:
        # if we want to compute the Born Charges we need to call 'torch.autograd.grad' on the dipoles w.r.t. the positions.
        # However, since the forces are always computed, MACE always calls 'torch.autograd.grad' on the energy w.r.t. the positions.
        # This happens in 'compute_forces' in 'mace/modules/utils.py', which is called inside 'get_outputs'.
        # If 'training' == False, in that function the computational graph will be destroy and the Born Charges can not be computed afterwards.
        # For this reason, we set 'training' == True if we need the Born Charges so that the computational graph is preserved and we can call 'torch.autograd.grad' in 'compute_dielectric_gradients'.
        # If you don't believe me, please have a look at the keyword 'retain_graph' in 'mace/modules/utils.py' in the function 'compute_forces'.

        # energy, forces and stress
        forces, virials, stress, _, _ = get_outputs(
            energy=total_energy,
            positions=data["positions"],
            displacement=displacement,
            cell=data["cell"],
            training=training or compute_bec,
            compute_force=compute_force,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
        )

        out = {
            "energy": total_energy,  # [n_graphs,]
            "node_energy": node_energy,  # [n_nodes,]
            "forces": forces,  # [n_nodes,3]
            "virials": virials,
            "stress": stress,
            "displacement": displacement,
            "dipole": total_dipole,  # [n_graphs,3]
            "atomic_dipoles": node_dipoles,  # [n_nodes,3]
            "atomic-oxn-dipole": node_dipole_baseline,  # [n_nodes,3]
            "atomic_polarizability": node_polarizability,  # [n_nodes,9]
            "polarizability": total_polarizability,  # [n_graphs,9]
        }

        if compute_bec:

            # bec.shape should be [3, n_nodes, 3]
            # where the first dimension corresponds to the Cartesian components of the dipole
            # the second dimension corresponds to the atoms,
            # the third dimension corresponds to the Cartesian components of the positions.

            # if you passed total_dipole[:,:2] into 'compute_dielectric_gradients_loop'
            # you would get bec.shape == [2, n_nodes, 3]

            bec = compute_dielectric_gradients_loop(
                dielectric=total_dipole,  # [:,:2] try for debugging
                inputs=[data["positions"]],
                clean=not training,
            )[0]
            assert bec.shape == (
                3,
                num_nodes,
                3,
            ), f"'bec' has the wrong shape, expected {(3, num_nodes, 3)} but got {bec.shape}."

            # We reshape 'bec' so that it will have a shape that is ASE-readable
            # ASE expects the Born Effective Charges to be in a shape (n_atoms, 3, 3):
            # - the first dimension corresponds to the atoms
            # - the second dimension corresponds to the Cartesian components of the positions,
            # - the third dimension corresponds to the Cartesian components of the dipole.
            # In this way every atom has 3x3 matrix, with dipole components as rows and position components as columns.
            out["bec"] = bec.moveaxis(0, 2)  # [3', n_nodes, 3] --> [n_nodes, 3, 3']

        return out

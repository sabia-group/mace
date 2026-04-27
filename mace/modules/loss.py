###########################################################################################
# Implementation of different loss functions
# Authors: Ilyes Batatia, Gregor Simm
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import os
from typing import Callable, Optional

import torch
import torch.distributed as dist

from mace.tools import TensorDict
from mace.tools.torch_geometric import Batch
from mace.tools.utils import pbc_dipole


# ------------------------------------------------------------------------------
# Helper function for loss reduction that handles DDP correction
# ------------------------------------------------------------------------------
def is_ddp_enabled():
    return dist.is_initialized() and dist.get_world_size() > 1


def reduce_loss(raw_loss: torch.Tensor, ddp: Optional[bool] = None) -> torch.Tensor:
    """
    Reduces an element-wise loss tensor.

    If ddp is True and distributed is initialized, the function computes:

        loss = (local_sum * world_size) / global_num_elements

    Otherwise, it returns the regular mean.
    """
    ddp = is_ddp_enabled() if ddp is None else ddp
    if ddp and dist.is_initialized():
        world_size = dist.get_world_size()
        n_local = raw_loss.numel()
        loss_sum = raw_loss.sum()
        total_samples = torch.tensor(
            n_local, device=raw_loss.device, dtype=raw_loss.dtype
        )
        dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)
        return loss_sum * world_size / total_samples
    return raw_loss.mean()


# ------------------------------------------------------------------------------
# double where trick
# - https://github.com/pytorch/pytorch/issues/156212
# - https://docs.jax.dev/en/latest/faq.html#gradients-contain-nan-where-using-where
# - https://github.com/tensorflow/probability/blob/main/discussion/where-nan.pdf
# ------------------------------------------------------------------------------
def general_loss_with_nan(
    ref_weight: torch.Tensor,
    quantity_weight: torch.Tensor,
    ref: torch.Tensor,
    pred: torch.Tensor,
    func: Callable[[torch.Tensor], torch.Tensor],
    num_atoms: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # Create mask: True where all elements along non-batch dims are valid
    ii = ~torch.isnan(ref)
    if ii.ndim > 1:
        ii = ii.all(dim=tuple(range(1, ii.ndim)))  # mask along all dims except first

    if not ii.any():
        # Return zero loss if no valid entries
        return torch.tensor(0.0, device=ref.device, dtype=ref.dtype)

    # Ensure num_atoms is broadcastable or masked properly
    if num_atoms is None:
        num_atoms = torch.ones_like(ref)

    # Masked computation
    safe_ref = ref[ii]
    safe_pred = pred[ii]
    safe_num_atoms = num_atoms[ii] if num_atoms.ndim == ii.ndim else num_atoms[ii]

    raw_loss = (
        ref_weight[ii]
        * quantity_weight[ii]
        * func(safe_ref - safe_pred, safe_num_atoms, ii)
    )

    return raw_loss


# ------------------------------------------------------------------------------
# Energy Loss Functions
# ------------------------------------------------------------------------------


def weighted_mean_squared_error_energy(
    ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
) -> torch.Tensor:
    # Calculate per-graph number of atoms.
    num_atoms = ref.ptr[1:] - ref.ptr[:-1]  # shape: [n_graphs]
    raw_loss = general_loss_with_nan(
        ref.weight,
        ref.energy_weight,
        ref["energy"],
        pred["energy"],
        lambda x, a, i: torch.square(x / a),
        num_atoms,
    )
    # ii = ~torch.isnan(ref["energy"])
    # raw_loss = (
    #     ref.weight[ii]
    #     * ref.energy_weight[ii]
    #     * torch.square((ref["energy"][ii] - pred["energy"][ii]) / num_atoms[ii])
    # )
    return reduce_loss(raw_loss, ddp)


def weighted_mean_absolute_error_energy(
    ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
) -> torch.Tensor:
    num_atoms = ref.ptr[1:] - ref.ptr[:-1]
    raw_loss = general_loss_with_nan(
        ref.weight,
        ref.energy_weight,
        ref["energy"],
        pred["energy"],
        lambda x, a, i: torch.abs(x / a),
        num_atoms,
    )
    # raw_loss = (
    #     ref.weight
    #     * ref.energy_weight
    #     * torch.abs((ref["energy"] - pred["energy"]) / num_atoms)
    # )
    return reduce_loss(raw_loss, ddp)


# ------------------------------------------------------------------------------
# Stress and Virials Loss Functions
# ------------------------------------------------------------------------------


def weighted_mean_squared_stress(
    ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
) -> torch.Tensor:
    configs_weight = ref.weight.view(-1, 1, 1)
    configs_stress_weight = ref.stress_weight.view(-1, 1, 1)
    raw_loss = general_loss_with_nan(
        configs_weight,
        configs_stress_weight,
        ref["stress"],
        pred["stress"],
        lambda x, a, i: torch.square(x),
    )
    # ii = ~torch.isnan(ref["stress"])
    # raw_loss = (
    #     configs_weight
    #     * configs_stress_weight
    #     * torch.square(ref["stress"] - pred["stress"])
    # )[ii]
    return reduce_loss(raw_loss, ddp)


def weighted_mean_squared_virials(
    ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
) -> torch.Tensor:
    configs_weight = ref.weight.view(-1, 1, 1)
    configs_virials_weight = ref.virials_weight.view(-1, 1, 1)
    num_atoms = (ref.ptr[1:] - ref.ptr[:-1]).view(-1, 1, 1)
    raw_loss = general_loss_with_nan(
        configs_weight,
        configs_virials_weight,
        ref["virials"],
        pred["virials"],
        lambda x, a, i: torch.square(x / a),
        num_atoms,
    )
    # ii = ~torch.isnan(ref["virials"])
    # raw_loss = (
    #     configs_weight
    #     * configs_virials_weight
    #     * torch.square((ref["virials"] - pred["virials"]) / num_atoms)
    # )[ii]
    return reduce_loss(raw_loss, ddp)


# ------------------------------------------------------------------------------
# BEC Loss Functions
# ------------------------------------------------------------------------------
def mean_squared_error_bec(
    ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
) -> torch.Tensor:
    if "bec" not in pred:
        raise ValueError(
            "Predictions do not contain 'bec' key required for BEC loss."
            + "Please add --compute_bec True to your config.yml file."
        )
    # Repeat per-graph weights to per-atom level.
    configs_weight = torch.repeat_interleave(
        ref.weight, ref.ptr[1:] - ref.ptr[:-1]
    ).unsqueeze(-1)
    configs_bec_weight = torch.repeat_interleave(
        ref.bec_weight, ref.ptr[1:] - ref.ptr[:-1]
    ).unsqueeze(-1)
    n_nodes = ref["bec"].shape[0]
    assert pred["bec"].shape == (
        n_nodes,
        3,
        3,
    ), f"Expected 'bec' to have shape [{n_nodes}, 3, 3], but got {pred['bec'].shape}"
    raw_loss = sum(
        general_loss_with_nan(
            configs_weight,
            configs_bec_weight,
            ref["bec"][:, :, n],
            pred["bec"][:, :, n],
            lambda x, a, i: torch.square(x),
        )
        for n in range(3)
    )
    return reduce_loss(raw_loss, ddp)


# ------------------------------------------------------------------------------
# Forces Loss Functions
# ------------------------------------------------------------------------------


def mean_squared_error_forces(
    ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
) -> torch.Tensor:
    # Repeat per-graph weights to per-atom level.
    configs_weight = torch.repeat_interleave(
        ref.weight, ref.ptr[1:] - ref.ptr[:-1]
    ).unsqueeze(-1)
    configs_forces_weight = torch.repeat_interleave(
        ref.forces_weight, ref.ptr[1:] - ref.ptr[:-1]
    ).unsqueeze(-1)
    raw_loss = general_loss_with_nan(
        configs_weight,
        configs_forces_weight,
        ref["forces"],
        pred["forces"],
        lambda x, a, i: torch.square(x),
    )
    # ii = ~torch.isnan(ref["forces"]).any(dim=-1)
    # raw_loss = (
    #     configs_weight[ii]
    #     * configs_forces_weight[ii]
    #     * torch.square(ref["forces"][ii] - pred["forces"][ii])
    # )
    return reduce_loss(raw_loss, ddp)


def mean_normed_error_forces(
    ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
) -> torch.Tensor:
    raw_loss = torch.linalg.vector_norm(ref["forces"] - pred["forces"], ord=2, dim=-1)
    return reduce_loss(raw_loss, ddp)


# ------------------------------------------------------------------------------
# Dipole Loss Function
# ------------------------------------------------------------------------------


def weighted_mean_squared_error_dipole(
    ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
) -> torch.Tensor:
    num_atoms = (ref.ptr[1:] - ref.ptr[:-1]).unsqueeze(-1)

    def dipole_loss_func(x, a, i):
        """Compute squared dipole loss for PBC or non-PBC environments."""
        if os.environ.get("USE_PBC_DIPOLE", "false").lower() in (
            "1",
            "true",
            "yes",
            "on",
        ):
            return torch.square(pbc_dipole(ref.cell, ref.pbc, x, i) / a)
        return torch.square(x / a)

    raw_loss = general_loss_with_nan(
        ref.weight.unsqueeze(-1),
        ref.dipole_weight.unsqueeze(-1),
        ref["dipole"],
        pred["dipole"],
        dipole_loss_func,
        num_atoms,
    )

    return reduce_loss(raw_loss, ddp)


# ------------------------------------------------------------------------------
# Polarizability Loss Function
# ------------------------------------------------------------------------------


def weighted_mean_squared_error_polarizability(
    ref: Batch,
    pred: TensorDict,
    ddp: Optional[
        bool
    ] = None,  # ,mean: Optional[torch.Tensor] = None , std: Optional[torch.Tensor] = None
) -> torch.Tensor:
    # polarizability: [n_graphs, ]
    # ref_polar = ref["polarizability"].view(-1, 3, 3) * std.view(1, 3, 3) + mean.view(1, 3, 3) if mean is not None and std is not None else ref["polarizability"]
    num_atoms = (ref.ptr[1:] - ref.ptr[:-1]).view(-1, 1, 1)  # [n_graphs,1]
    raw_loss = torch.square(
        (ref["polarizability"].view(-1, 3, 3) - pred["polarizability"]) / num_atoms
    )
    return reduce_loss(raw_loss, ddp)


# ------------------------------------------------------------------------------
# Conditional Losses for Forces
# ------------------------------------------------------------------------------


def conditional_mse_forces(
    ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
) -> torch.Tensor:
    configs_weight = torch.repeat_interleave(
        ref.weight, ref.ptr[1:] - ref.ptr[:-1]
    ).unsqueeze(-1)
    configs_forces_weight = torch.repeat_interleave(
        ref.forces_weight, ref.ptr[1:] - ref.ptr[:-1]
    ).unsqueeze(-1)
    # Define multiplication factors for different regimes.
    factors = torch.tensor(
        [1.0, 0.7, 0.4, 0.1], device=ref["forces"].device, dtype=ref["forces"].dtype
    )
    err = ref["forces"] - pred["forces"]
    se = torch.zeros_like(err)
    norm_forces = torch.norm(ref["forces"], dim=-1)
    c1 = norm_forces < 100
    c2 = (norm_forces >= 100) & (norm_forces < 200)
    c3 = (norm_forces >= 200) & (norm_forces < 300)
    se[c1] = torch.square(err[c1]) * factors[0]
    se[c2] = torch.square(err[c2]) * factors[1]
    se[c3] = torch.square(err[c3]) * factors[2]
    se[~(c1 | c2 | c3)] = torch.square(err[~(c1 | c2 | c3)]) * factors[3]
    raw_loss = configs_weight * configs_forces_weight * se
    return reduce_loss(raw_loss, ddp)


def conditional_huber_forces(
    ref_forces: torch.Tensor,
    pred_forces: torch.Tensor,
    huber_delta: float,
    ddp: Optional[bool] = None,
) -> torch.Tensor:
    factors = huber_delta * torch.tensor(
        [1.0, 0.7, 0.4, 0.1], device=ref_forces.device, dtype=ref_forces.dtype
    )
    norm_forces = torch.norm(ref_forces, dim=-1)
    c1 = norm_forces < 100
    c2 = (norm_forces >= 100) & (norm_forces < 200)
    c3 = (norm_forces >= 200) & (norm_forces < 300)
    c4 = ~(c1 | c2 | c3)
    se = torch.zeros_like(pred_forces)
    se[c1] = torch.nn.functional.huber_loss(
        ref_forces[c1], pred_forces[c1], reduction="none", delta=factors[0]
    )
    se[c2] = torch.nn.functional.huber_loss(
        ref_forces[c2], pred_forces[c2], reduction="none", delta=factors[1]
    )
    se[c3] = torch.nn.functional.huber_loss(
        ref_forces[c3], pred_forces[c3], reduction="none", delta=factors[2]
    )
    se[c4] = torch.nn.functional.huber_loss(
        ref_forces[c4], pred_forces[c4], reduction="none", delta=factors[3]
    )
    return reduce_loss(se, ddp)


# ------------------------------------------------------------------------------
# Loss Modules Combining Multiple Quantities
# ------------------------------------------------------------------------------


class PESDielectricLoss(torch.nn.Module):
    def __init__(
        self,
        energy_weight=0.0,
        forces_weight=0.0,
        stress_weight=0.0,
        virials_weight=0.0,
        dipole_weight=0.0,
        bec_weight=0.0,
        polarizability_weight=0.0,
    ) -> None:
        super().__init__()
        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "stress_weight",
            torch.tensor(stress_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "virials_weight",
            torch.tensor(virials_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "dipole_weight",
            torch.tensor(dipole_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "bec_weight",
            torch.tensor(bec_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "polarizability_weight",
            torch.tensor(polarizability_weight, dtype=torch.get_default_dtype()),
        )

    def forward(
        self, ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
    ) -> torch.Tensor:
        loss = torch.tensor(0.0, dtype=torch.get_default_dtype())
        if self.energy_weight > 0.0:
            loss = loss + self.energy_weight * weighted_mean_squared_error_energy(
                ref, pred, ddp
            )
        if self.forces_weight > 0.0:
            loss = loss + self.forces_weight * mean_squared_error_forces(ref, pred, ddp)
        if self.stress_weight > 0.0:
            loss = loss + self.stress_weight * weighted_mean_squared_stress(
                ref, pred, ddp
            )
        if self.virials_weight > 0.0:
            loss = loss + self.virials_weight * weighted_mean_squared_virials(
                ref, pred, ddp
            )
        if self.dipole_weight > 0.0:
            loss = loss + self.dipole_weight * weighted_mean_squared_error_dipole(
                ref, pred, ddp
            )
        if self.bec_weight > 0.0:
            loss = loss + self.bec_weight * mean_squared_error_bec(ref, pred, ddp)
        if self.polarizability_weight > 0.0:
            loss = (
                loss
                + self.polarizability_weight
                * weighted_mean_squared_error_polarizability(ref, pred, ddp)
            )
        return loss

    def __repr__(self):
        return (
            f"{self.__class__.__name__}"
            + f"(energy_weight={self.energy_weight:.2e}, "
            + f"forces_weight={self.forces_weight:.2e}, "
            + f"stress_weight={self.stress_weight:.2e}, "
            + f"virials_weight={self.virials_weight:.2e},"
            + f"dipole_weight={self.dipole_weight:.2e},"
            + f"polarizability_weight={self.polarizability_weight:.2e})"
        )


class WeightedHuberEnergyForcesStressLoss(torch.nn.Module):
    def __init__(
        self, energy_weight=1.0, forces_weight=1.0, stress_weight=1.0, huber_delta=0.01
    ) -> None:
        super().__init__()
        # We store the huber_delta rather than a loss with fixed reduction.
        self.huber_delta = huber_delta
        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "stress_weight",
            torch.tensor(stress_weight, dtype=torch.get_default_dtype()),
        )

    def forward(
        self, ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
    ) -> torch.Tensor:
        num_atoms = ref.ptr[1:] - ref.ptr[:-1]
        if ddp:
            loss_energy = torch.nn.functional.huber_loss(
                ref["energy"] / num_atoms,
                pred["energy"] / num_atoms,
                reduction="none",
                delta=self.huber_delta,
            )
            loss_energy = reduce_loss(loss_energy, ddp)
            loss_forces = torch.nn.functional.huber_loss(
                ref["forces"], pred["forces"], reduction="none", delta=self.huber_delta
            )
            loss_forces = reduce_loss(loss_forces, ddp)
            loss_stress = torch.nn.functional.huber_loss(
                ref["stress"], pred["stress"], reduction="none", delta=self.huber_delta
            )
            loss_stress = reduce_loss(loss_stress, ddp)
        else:
            loss_energy = torch.nn.functional.huber_loss(
                ref["energy"] / num_atoms,
                pred["energy"] / num_atoms,
                reduction="mean",
                delta=self.huber_delta,
            )
            loss_forces = torch.nn.functional.huber_loss(
                ref["forces"], pred["forces"], reduction="mean", delta=self.huber_delta
            )
            loss_stress = torch.nn.functional.huber_loss(
                ref["stress"], pred["stress"], reduction="mean", delta=self.huber_delta
            )
        return (
            self.energy_weight * loss_energy
            + self.forces_weight * loss_forces
            + self.stress_weight * loss_stress
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f}, stress_weight={self.stress_weight:.3f})"
        )


class UniversalLoss(torch.nn.Module):
    def __init__(
        self, energy_weight=1.0, forces_weight=1.0, stress_weight=1.0, huber_delta=0.01
    ) -> None:
        super().__init__()
        self.huber_delta = huber_delta
        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "stress_weight",
            torch.tensor(stress_weight, dtype=torch.get_default_dtype()),
        )

    def forward(
        self, ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
    ) -> torch.Tensor:
        num_atoms = ref.ptr[1:] - ref.ptr[:-1]
        configs_stress_weight = ref.stress_weight.view(-1, 1, 1)
        configs_energy_weight = ref.energy_weight
        configs_forces_weight = torch.repeat_interleave(
            ref.forces_weight, ref.ptr[1:] - ref.ptr[:-1]
        ).unsqueeze(-1)
        if ddp:
            loss_energy = torch.nn.functional.huber_loss(
                configs_energy_weight * ref["energy"] / num_atoms,
                configs_energy_weight * pred["energy"] / num_atoms,
                reduction="none",
                delta=self.huber_delta,
            )
            loss_energy = reduce_loss(loss_energy, ddp)
            loss_forces = conditional_huber_forces(
                configs_forces_weight * ref["forces"],
                configs_forces_weight * pred["forces"],
                huber_delta=self.huber_delta,
                ddp=ddp,
            )
            loss_stress = torch.nn.functional.huber_loss(
                configs_stress_weight * ref["stress"],
                configs_stress_weight * pred["stress"],
                reduction="none",
                delta=self.huber_delta,
            )
            loss_stress = reduce_loss(loss_stress, ddp)
        else:
            loss_energy = torch.nn.functional.huber_loss(
                configs_energy_weight * ref["energy"] / num_atoms,
                configs_energy_weight * pred["energy"] / num_atoms,
                reduction="mean",
                delta=self.huber_delta,
            )
            loss_forces = conditional_huber_forces(
                configs_forces_weight * ref["forces"],
                configs_forces_weight * pred["forces"],
                huber_delta=self.huber_delta,
                ddp=ddp,
            )
            loss_stress = torch.nn.functional.huber_loss(
                configs_stress_weight * ref["stress"],
                configs_stress_weight * pred["stress"],
                reduction="mean",
                delta=self.huber_delta,
            )
        return (
            self.energy_weight * loss_energy
            + self.forces_weight * loss_forces
            + self.stress_weight * loss_stress
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f}, stress_weight={self.stress_weight:.3f})"
        )


class WeightedEnergyForcesL1L2Loss(torch.nn.Module):
    def __init__(self, energy_weight=1.0, forces_weight=1.0) -> None:
        super().__init__()
        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )

    def forward(
        self, ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
    ) -> torch.Tensor:
        loss_energy = weighted_mean_absolute_error_energy(ref, pred, ddp)
        loss_forces = mean_normed_error_forces(ref, pred, ddp)
        return self.energy_weight * loss_energy + self.forces_weight * loss_forces

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f})"
        )

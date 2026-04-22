###########################################################################################
# Statistics utilities
# Authors: Ilyes Batatia, Gregor Simm, David Kovacs
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Union

import numpy as np
import torch

from mace.tools.torch_geometric import Batch

from .torch_tools import to_numpy


def compute_mae(delta: np.ndarray) -> float:
    return np.mean(np.abs(delta)).item()


def compute_rel_mae(delta: np.ndarray, target_val: np.ndarray) -> float:
    target_norm = np.mean(np.abs(target_val))
    return np.mean(np.abs(delta)).item() / (target_norm + 1e-9) * 100


def compute_rmse(delta: np.ndarray) -> float:
    return np.sqrt(np.mean(np.square(delta))).item()


def compute_rel_rmse(delta: np.ndarray, target_val: np.ndarray) -> float:
    target_norm = np.sqrt(np.mean(np.square(target_val))).item()
    return np.sqrt(np.mean(np.square(delta))).item() / (target_norm + 1e-9) * 100


def compute_q95(delta: np.ndarray) -> float:
    return np.percentile(np.abs(delta), q=95)


def compute_c(delta: np.ndarray, eta: float) -> float:
    return np.mean(np.abs(delta) < eta).item()


def get_tag(name: str, seed: int) -> str:
    return f"{name}_run-{seed}"


def setup_logger(
    level: Union[int, str] = logging.INFO,
    tag: Optional[str] = None,
    directory: Optional[str] = None,
    rank: Optional[int] = 0,
):
    # Create a logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Set to DEBUG to capture all levels

    # Create formatters
    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Add filter for rank
    logger.addFilter(lambda _: rank == 0)

    # Create console handler
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if directory is not None and tag is not None:
        os.makedirs(name=directory, exist_ok=True)

        # Create file handler for non-debug logs
        main_log_path = os.path.join(directory, f"{tag}.log")
        fh_main = logging.FileHandler(main_log_path)
        fh_main.setLevel(level)
        fh_main.setFormatter(formatter)
        logger.addHandler(fh_main)

        # Create file handler for debug logs
        debug_log_path = os.path.join(directory, f"{tag}_debug.log")
        fh_debug = logging.FileHandler(debug_log_path)
        fh_debug.setLevel(logging.DEBUG)
        fh_debug.setFormatter(formatter)
        fh_debug.addFilter(lambda record: record.levelno >= logging.DEBUG)
        logger.addHandler(fh_debug)


class AtomicNumberTable:
    def __init__(self, zs: Sequence[int]):
        self.zs = zs

    def __len__(self) -> int:
        return len(self.zs)

    def __str__(self):
        return f"AtomicNumberTable: {tuple(s for s in self.zs)}"

    def index_to_z(self, index: int) -> int:
        return self.zs[index]

    def z_to_index(self, atomic_number: str) -> int:
        return self.zs.index(atomic_number)


def get_atomic_number_table_from_zs(zs: Iterable[int]) -> AtomicNumberTable:
    z_set = set()
    for z in zs:
        z_set.add(z)
    return AtomicNumberTable(sorted(list(z_set)))


def atomic_numbers_to_indices(
    atomic_numbers: np.ndarray, z_table: AtomicNumberTable
) -> np.ndarray:
    to_index_fn = np.vectorize(z_table.z_to_index)
    return to_index_fn(atomic_numbers)


class UniversalEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, torch.Tensor):
            return to_numpy(o)
        return json.JSONEncoder.default(self, o)


class MetricsLogger:
    def __init__(self, directory: str, tag: str) -> None:
        self.directory = directory
        self.filename = tag + ".txt"
        self.path = os.path.join(self.directory, self.filename)

    def log(self, d: Dict[str, Any]) -> None:
        os.makedirs(name=self.directory, exist_ok=True)
        with open(self.path, mode="a", encoding="utf-8") as f:
            f.write(json.dumps(d, cls=UniversalEncoder))
            f.write("\n")


# pylint: disable=abstract-method, arguments-differ
class LAMMPS_MP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args):
        feats, data = args  # unpack
        ctx.vec_len = feats.shape[-1]
        ctx.data = data
        out = torch.empty_like(feats)
        data.forward_exchange(feats, out, ctx.vec_len)
        return out

    @staticmethod
    def backward(ctx, *grad_outputs):
        (grad,) = grad_outputs  # unpack
        gout = torch.empty_like(grad)
        ctx.data.reverse_exchange(grad, gout, ctx.vec_len)
        return gout, None


def get_cache_dir() -> Path:
    # get cache dir from XDG_CACHE_HOME if set, otherwise appropriate default
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "mace"


def filter_nonzero_weight(
    batch,
    quantity_l,
    weight,
    quantity_weight,
    spread_atoms=False,
    spread_quantity_vector=True,
) -> float:
    quantity = quantity_l[-1]
    # repeat with interleaving for per-atom quantities
    if spread_atoms:
        weight = torch.repeat_interleave(
            weight, batch.ptr[1:] - batch.ptr[:-1]
        ).unsqueeze(-1)
        quantity_weight = torch.repeat_interleave(
            quantity_weight, batch.ptr[1:] - batch.ptr[:-1]
        ).unsqueeze(-1)

    # repeat for additional dimensions
    if len(quantity.shape) > 1:
        repeats = [1] + list(quantity.shape[1:])
        view = [-1] + [1] * (len(quantity.shape) - 1)
        weight = weight.view(*view).repeat(*repeats)
        if spread_quantity_vector:
            quantity_weight = quantity_weight.view(*view).repeat(*repeats)
    filtered_q = quantity[weight * quantity_weight > 0]

    if len(filtered_q) == 0:
        quantity_l.pop()
        return 0.0

    quantity_l[-1] = filtered_q
    return 1.0


def wrap05(x: torch.Tensor) -> torch.Tensor:
    """Wrap number to [-0.5, 0.5)"""
    return x - torch.round(x)


def wrap01(x: torch.Tensor) -> torch.Tensor:
    """Wrap number to [0, 1)"""
    return wrap05(x) + torch.tensor(0.5)


def shift_ref_dipole(
    cell: torch.Tensor, pbc: torch.Tensor, ref: torch.Tensor, pred: torch.Tensor
) -> torch.Tensor:
    """
    Adjusts the reference dipole positions under periodic boundary conditions (PBC)
    so that the reference and predicted dipoles are PBC-consistent.

    Args:
        cell: Tensor of shape (N, 3, 3) representing the simulation cell vectors.
        pbc: Tensor of shape (N, 3) indicating periodicity along each axis.
        ref: Reference dipole tensor of shape (N, 3).
        pred: Predicted dipole tensor of shape (N, 3).

    Returns:
        Updated reference dipole tensor with PBC-consistent shifts applied.
    """
    i = ~torch.any(torch.isnan(ref), dim=1)
    delta = ref - pred
    final_delta = pbc_dipole(cell, pbc, delta[i], i)

    icell = torch.linalg.inv(cell.reshape((-1, 3, 3)))
    shift = final_delta - delta[i]
    ref[i] = ref[i] + shift

    # debug
    frac = torch.einsum("ijk,ik->ij", icell[i], shift)
    frac = wrap05(frac)
    assert torch.allclose(frac, torch.tensor(0.0)), "coding error"

    return ref


def pbc_dipole(
    cell: torch.Tensor,
    pbc: torch.Tensor,
    delta: torch.Tensor,
    i: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Applies periodic boundary conditions (PBC) to displacement vectors to ensure
    dipole calculations remain invariant under quantum polarization jumps in
    periodic systems.

    The function maps each displacement (`delta`) to its nearest periodic image
    using the provided cell vectors and periodicity flags.

    Note:
        Only fully 3D periodic systems are currently supported.

    Args:
        cell: Tensor of shape (N, 3, 3) containing the lattice vectors for each structure.
        pbc: Tensor of shape (N, 3) specifying periodicity along x, y, and z (must all be True).
        delta: Tensor of shape (N, 3) representing displacement vectors.
        i: Optional boolean mask tensor selecting valid structures.

    Returns:
        Tensor of shape (N, 3) containing PBC-corrected displacement vectors.
    """
    # Select relevant structures
    cell = torch.reshape(cell, (-1, 3, 3))[i]
    pbc = torch.reshape(pbc, (-1, 3))[i]
    delta = torch.reshape(delta, (-1, 3))  # already masked in 'general_loss_with_nan'

    if not torch.all(pbc.bool()):
        raise ValueError("This function only supports fully 3D periodic systems.")

    N = cell.shape[0]
    assert cell.shape == (N, 3, 3), "error in cell shape"
    assert delta.shape == (N, 3), "error in delta shape"
    assert delta.shape == pbc.shape, "delta.shape should be the same as pbc.shape"
    assert not torch.any(torch.isnan(delta)), "Found NaN values"

    # Compute fractional displacements
    icell = torch.linalg.inv(cell)
    frac = torch.einsum("ijk,ik->ij", icell, delta)

    # Wrap to [-0.5, 0.5)
    frac = wrap05(frac)
    assert torch.all(frac >= -0.5)
    assert torch.all(frac <= 1.5)

    # Back to Cartesian
    final = torch.einsum("ijk,ik->ij", cell, frac)

    cart_shift = final - delta
    frac_shift = torch.einsum("ijk,ik->ij", icell, cart_shift)
    frac_shift = wrap05(frac_shift)
    assert torch.allclose(frac_shift, torch.tensor(0.0)), "coding error"

    return final


def get_model_output(
    model: torch.nn.Module,
    batch: Batch,
    output_args: Dict[str, bool],
) -> Dict[str, torch.Tensor]:
    assert isinstance(batch, Batch), "'batch' should be of type Batch"
    kwargs = {
        "training": True,
        "compute_force": output_args.get("forces", True),
        "compute_virials": output_args.get("virials", True),
        "compute_stress": output_args.get("stress", True),
    }
    # not supported by all models
    if "bec" in output_args and output_args["bec"] is not None:
        kwargs["compute_bec"] = output_args["bec"]

    return model(batch.to_dict(), **kwargs)

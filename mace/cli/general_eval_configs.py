import argparse
import warnings
from typing import Dict, List

import ase.io
import numpy as np
import torch

from mace import data
from mace.cli.convert_e3nn_cueq import run as run_e3nn_to_cueq
from mace.tools import DefaultKeys, torch_geometric, torch_tools, utils

ase_like_properties = {
    "energy": (),
    "node_energy": ("natoms",),
    # 'contributions': ('n_contributions',),
    "forces": ("natoms", 3),
    "displacement": (3, 3),
    "stress": (3, 3),
    "virials": (3, 3),
    "dipole": (3,),
    "atomic_dipoles": ("natoms", 3),
    "atomic-oxn-dipole": ("natoms", 3),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--configs", help="path to XYZ configurations", required=True)
    parser.add_argument("--model", help="path to model", required=True)
    parser.add_argument("--output", help="output path", required=True)
    parser.add_argument(
        "--device",
        help="select device",
        type=str,
        choices=["cpu", "cuda"],
        default="cpu",
    )
    parser.add_argument(
        "--enable_cueq",
        help="enable cuequivariance acceleration",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--default_dtype",
        help="set default dtype",
        type=str,
        choices=["float32", "float64"],
        default="float64",
    )
    parser.add_argument("--batch_size", help="batch size", type=int, default=64)
    parser.add_argument(
        "--compute_stress",
        help="compute stress",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--compute_bec",
        help="compute BEC",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--prefix",
        help="prefix for energy, forces and stress keys",
        type=str,
        default="MACE_",
    )
    parser.add_argument(
        "--head",
        help="Model head used for evaluation",
        type=str,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--oxn_key",
        help="Key of oxidation numbers in training xyz",
        type=str,
        default=DefaultKeys.OXN.value,
    )
    return parser.parse_args()


def get_model_output(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    compute_stress: bool,
    compute_bec: bool,
) -> Dict[str, torch.Tensor]:
    forward_args = {
        "compute_stress": compute_stress,
    }
    if compute_bec:
        # Only add `compute_bec` if it is requested
        # We check if the model is MACELES at the start of the run function
        forward_args["compute_bec"] = compute_bec
    return model(batch, **forward_args)


def main() -> None:
    args = parse_args()
    run(args)


def run(args: argparse.Namespace) -> None:
    torch_tools.set_default_dtype(args.default_dtype)
    device = torch_tools.init_device(args.device)

    # Load model
    model = torch.load(f=args.model, map_location=args.device)
    if model.__class__.__name__ != "MACELES" and args.compute_bec:
        raise ValueError("BEC can only be computed with MACELES model. ")
    if args.enable_cueq:
        print("Converting models to CuEq for acceleration")
        model = run_e3nn_to_cueq(model, device=device)
    model = model.to(
        args.device
    )  # shouldn't be necessary but seems to help with CUDA problems

    for param in model.parameters():
        param.requires_grad = False

    # Load data and prepare input
    key_specification = data.KeySpecification()
    data.update_keyspec_from_kwargs(key_specification, vars(args))
    atoms_list = ase.io.read(args.configs, index=":")
    if args.head is not None:
        for atoms in atoms_list:
            atoms.info["head"] = args.head
    configs = [
        data.config_from_atoms(
            atoms=atoms, key_specification=key_specification, head_name=args.head
        )
        for atoms in atoms_list
    ]
    n_atoms = np.asarray([a.get_global_number_of_atoms() for a in atoms_list])

    z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])

    try:
        heads = model.heads
    except AttributeError:
        heads = None

    data_loader = torch_geometric.dataloader.DataLoader(
        dataset=[
            data.AtomicData.from_config(
                config, z_table=z_table, cutoff=float(model.r_max), heads=heads
            )
            for config in configs
        ],
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    # Collect data
    results: Dict[str, List[np.ndarray]] = {}
    k = 0
    for batch in data_loader:
        ki = k
        batch = batch.to(device)
        output = get_model_output(
            model, batch.to_dict(), args.compute_stress, args.compute_bec
        )

        # remove empty fields
        to_delete = []
        for key in output.keys():
            value = output[key]
            if value is None:
                to_delete.append(key)
                continue
            output[key] = value.detach().cpu().numpy()

        for key in to_delete:
            del output[key]

        # reshape results
        for key in output.keys():

            if key not in ase_like_properties:
                warnings.warn(
                    f"Please add '{key}' to 'ase_like_properties' in {__file__}"
                )
                continue

            value = output[key]
            shape = ase_like_properties[key]
            if key not in results:
                results[key] = [None] * len(n_atoms)

            ki = k
            if "natoms" in shape:
                # [n_nodes,...] --> [ [n_atoms_0,...] , [n_atoms_1,...], ... ]
                value = np.split(value, batch.ptr[1:], axis=0)[:-1]
                for v in value:
                    assert v.shape[0] == n_atoms[ki], "coding error"
                    results[key][ki] = v
                    ki += 1
            else:
                for v in value:
                    results[key][ki] = v
                    ki += 1

        k = ki

    # save results in ase.Atoms
    for n, atoms in enumerate(atoms_list):
        for key, _ in results.items():
            shape = ase_like_properties[key]
            if "natoms" in shape:
                atoms.arrays[f"{args.prefix}{key}"] = results[key][n]
            else:
                atoms.info[f"{args.prefix}{key}"] = results[key][n]

    # Write atoms to output path
    ase.io.write(args.output, images=atoms_list, format="extxyz")


if __name__ == "__main__":
    main()

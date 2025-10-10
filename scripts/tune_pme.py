#!/usr/bin/env python3
import argparse
import numpy as np
import torch
import json
from ase.io import read
from mace.modules.long_range import NeighborModule


dtype = torch.float64


def main():
    argv = {
        "metavar": "\b",
    }
    parser = argparse.ArgumentParser(
        description="Tune parameters for a system from an extxyz file."
    )
    parser.add_argument("-i", "--input", **argv, type=str, help="input extxyz dataset")
    parser.add_argument("-o", "--output", **argv, type=str, help="output JSON file")
    parser.add_argument(
        "-m",
        "--method",
        **argv,
        type=str,
        help="method [%(choices)s] (default: %(default)s)",
        default="ewald",
        choices=["ewald", "pme", "p3m"],
    )
    parser.add_argument(
        "-c",
        "--charges",
        **argv,
        type=str,
        help="keyword or the charges (default: %(default)s)s",
        default="Qs",
    )
    parser.add_argument(
        "-r",
        "--cutoff",
        **argv,
        type=float,
        default=5.0,
        help="Neighbor cutoff radius (Å) (default: %(default)s)",
    )
    args = parser.parse_args()

    print(f"Loading structures from {args.input}...")
    # Load system from extxyz
    structures = read(args.input, index=":", format="extxyz")
    print(f"Loaded {len(structures)} structures from the file.")

    # Select the structure with the most atoms
    ii = np.argmax([a.get_global_number_of_atoms() for a in structures])
    atoms = structures[ii]
    print(f"Selected structure {ii} with {atoms.get_global_number_of_atoms()} atoms.")

    # Extract positions, cell, and charges
    positions = torch.tensor(atoms.get_positions(), dtype=dtype)
    cell = torch.tensor(atoms.cell.array, dtype=dtype)
    charges = torch.tensor(atoms.arrays[args.charges].reshape(-1, 1), dtype=dtype)
    print(
        f"Positions shape: {positions.shape}, Cell shape: {cell.shape}, Charges shape: {charges.shape}"
    )

    # Build neighbor list
    print(f"Building neighbor list with cutoff {args.cutoff} Å...")
    nl = NeighborModule(args.cutoff)
    neighbor_indices, neighbor_distances = nl(
        positions=positions,
        box=cell,
        periodic=True,
    )
    print(f"Neighbor list computed. Found neighbors for each atom.")

    if args.method == "ewald":
        from torchpme.tuning import tune_ewald

        tune = tune_ewald
        print("Using 'tune_ewald' for tuning")
    elif args.method == "pme":
        from torchpme.tuning import tune_pme

        tune = tune_pme
        print("Using 'tune_pme' for tuning")
    elif args.method == "p3m":
        from torchpme.tuning import tune_p3m

        tune = tune_p3m
        print("Using 'tune_p3m' for tuning")
    else:
        raise ValueError(
            f"'--method' can be only 'ewald', 'pme', or 'p3m' but you provided {args.method}"
        )

    # Tune
    print("Tuning parameters...")
    smearing, params, _ = tune(
        charges=charges,
        cell=cell,
        positions=positions,
        cutoff=args.cutoff,
        neighbor_indices=neighbor_indices,
        neighbor_distances=neighbor_distances,
    )
    print("Tuning complete.")
    print(f"Smearing parameter: {smearing}")
    print(f"Parameters: {params}")

    # Save results to JSON
    results = {
        "smearing": smearing.item() if torch.is_tensor(smearing) else smearing,
        **{k: (v.item() if torch.is_tensor(v) else v) for k, v in params.items()},
    }

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved tuning results to {args.output}")


if __name__ == "__main__":
    main()

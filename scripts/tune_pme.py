#!/usr/bin/env python3
import argparse
import json

import numpy as np
import torch
from ase.io import read

from mace.data.neighborhood import get_neighborhood
from mace.modules.utils import get_edge_vectors_and_lengths

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
    neighbor_indices, shifts, _, _ = get_neighborhood(
        positions, args.cutoff, True, cell
    )
    neighbor_distances, _ = get_edge_vectors_and_lengths(
        positions, neighbor_indices, shifts
    )

    print("Neighbor list computed. Found neighbors for each atom.")

    if args.method == "ewald":
        try:
            from torchpme.tuning import tune_ewald
        except ImportError as e:
            raise ImportError(
                "torchpme is not installed. Please install it to use the 'ewald' method."
            ) from e

        tune = tune_ewald
        print("Using 'tune_ewald' for tuning")
    elif args.method == "pme":
        try:
            from torchpme.tuning import tune_pme
        except ImportError as e:
            raise ImportError(
                "torchpme is not installed. Please install it to use the 'pme' method."
            ) from e

        tune = tune_pme
        print("Using 'tune_pme' for tuning")
    elif args.method == "p3m":
        try:
            from torchpme.tuning import tune_p3m
        except ImportError as e:
            raise ImportError(
                "torchpme is not installed. Please install it to use the 'p3m' method."
            ) from e

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
        neighbor_indices=neighbor_indices.T,
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

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved tuning results to {args.output}")


if __name__ == "__main__":
    main()

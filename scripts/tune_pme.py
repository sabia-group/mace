#!/usr/bin/env python3
import argparse
import numpy as np
import torch
import json
from ase.io import read
import vesin.torch
from torchpme.tuning import tune_pme

dtype = torch.float64


def main():
    parser = argparse.ArgumentParser(
        description="Tune PME parameters for a system from an extxyz file."
    )
    parser.add_argument("-i", "--input", help="input extxyz dataset")
    parser.add_argument("-o", "--output", help="output JSON file")
    parser.add_argument(
        "-c",
        "--charges",
        help="keyword or the charges (default: %(default)s)s",
        default="Qs",
    )
    parser.add_argument(
        "-r",
        "--cutoff",
        type=float,
        default=6.0,
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
    nl = vesin.torch.NeighborList(cutoff=args.cutoff, full_list=False)
    neighbor_indices, neighbor_distances = nl.compute(
        points=positions,
        box=cell,
        periodic=True,
        quantities="Pd",
    )
    print(f"Neighbor list computed. Found neighbors for each atom.")

    # Tune PME
    print("Tuning PME parameters...")
    smearing, pme_params, _ = tune_pme(
        charges=charges,
        cell=cell,
        positions=positions,
        cutoff=args.cutoff,
        neighbor_indices=neighbor_indices,
        neighbor_distances=neighbor_distances,
    )
    print("PME tuning complete.")
    print(f"Smearing parameter: {smearing}")
    print(f"PME parameters: {pme_params}")

    # Save results to JSON
    results = {
        "smearing": smearing.item() if torch.is_tensor(smearing) else smearing,
        **{k: (v.item() if torch.is_tensor(v) else v) for k, v in pme_params.items()},
    }

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved PME tuning results to {args.output}")


if __name__ == "__main__":
    main()

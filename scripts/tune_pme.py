#!/usr/bin/env python3
import argparse
import json
import torch
from ase.io import read
import vesin.torch
from torchpme.tuning import tune_pme

dtype = torch.float64


def main():
    parser = argparse.ArgumentParser(
        description="Tune PME parameters for a system from an extxyz file."
    )
    parser.add_argument("input", help="Path to input extxyz dataset")
    parser.add_argument("output", help="Path to output JSON file")
    parser.add_argument(
        "--cutoff", type=float, default=6.0, help="Neighbor cutoff radius (Å)"
    )
    args = parser.parse_args()

    # Load system from extxyz
    atoms = read(args.input)

    positions = torch.tensor(atoms.get_positions(), dtype=dtype)
    cell = torch.tensor(atoms.cell.array, dtype=dtype)
    charges = torch.tensor(atoms.get_initial_charges().reshape(-1, 1), dtype=dtype)

    # Build neighbor list
    nl = vesin.torch.NeighborList(cutoff=args.cutoff, full_list=False)
    neighbor_indices, neighbor_distances = nl.compute(
        points=positions,
        box=cell,
        periodic=True,
        quantities="Pd",
    )

    # Tune PME
    smearing, pme_params, _ = tune_pme(
        charges=charges,
        cell=cell,
        positions=positions,
        cutoff=args.cutoff,
        neighbor_indices=neighbor_indices,
        neighbor_distances=neighbor_distances,
    )

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

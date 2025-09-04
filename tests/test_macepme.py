import importlib.util
import os
from pathlib import Path

import ase.io
import numpy as np
import pytest
import torch
from ase.atoms import Atoms
from e3nn import o3

from mace.calculators import MACECalculator
from mace.cli.run_train import run as mace_run
from mace.modules import interaction_classes
from mace.modules.extensions import MACEPME
from mace.tools.arg_parser import build_default_arg_parser
from mace.tools.torch_tools import default_dtype

PME_AVAILABLE = bool((spec := importlib.util.find_spec("torchpme")) is not None)
CUET_AVAILABLE = bool((spec := importlib.util.find_spec("cuequivariance")) is not None)
CUDA_AVAILABLE = torch.cuda.is_available()


@pytest.fixture(name="fitting_configs")
def fixture_fitting_configs():
    water = Atoms(
        numbers=[8, 1, 1],
        positions=[[0, -2.0, 0], [1, 0, 0], [0, 1, 0]],
        cell=[4] * 3,
        pbc=[True] * 3,
    )
    water.arrays["Qs"] = np.array([-0.834, 0.417, 0.417])
    fit_configs = [
        Atoms(numbers=[8], positions=[[0, 0, 0]], cell=[6] * 3),
        Atoms(numbers=[1], positions=[[0, 0, 0]], cell=[6] * 3),
    ]
    fit_configs[0].info["REF_energy"] = 0.0
    fit_configs[0].info["config_type"] = "IsolatedAtom"
    fit_configs[1].info["REF_energy"] = 0.0
    fit_configs[1].info["config_type"] = "IsolatedAtom"
    fit_configs[0].arrays["Qs"] = np.array([0.0])
    fit_configs[1].arrays["Qs"] = np.array([0.0])

    np.random.seed(5)
    for _ in range(20):
        c = water.copy()
        c.positions += np.random.normal(0.1, size=c.positions.shape)
        c.info["REF_energy"] = np.random.normal(0.1)
        print(c.info["REF_energy"])
        c.new_array("REF_forces", np.random.normal(0.1, size=c.positions.shape))
        c.info["REF_stress"] = np.random.normal(0.1, size=6)
        fit_configs.append(c)

    return fit_configs


_mace_params = {
    "name": "MACEPME",
    "valid_fraction": 0.05,
    "energy_weight": 1.0,
    "forces_weight": 10.0,
    "stress_weight": 1.0,
    "model": "MACEPME",
    "hidden_irreps": "128x0e",
    "r_max": 3.5,
    "batch_size": 5,
    "max_num_epochs": 10,
    "swa": None,
    "start_swa": 5,
    "ema": None,
    "ema_decay": 0.99,
    "amsgrad": None,
    "restart_latest": None,
    "device": "cpu",
    "seed": 5,
    "loss": "stress",
    "energy_key": "REF_energy",
    "forces_key": "REF_forces",
    "stress_key": "REF_stress",
    "charges_key": "Qs",
    "eval_interval": 2,
    "use_reduced_cg": False,
    "pme_arguments": '{"lr_wavelength": 0.5, "smearing":1.0 }',
}


@pytest.mark.skipif(not PME_AVAILABLE, reason="torch-pme library is not available")
def test_run_train(tmp_path, fitting_configs):
    ase.io.write(tmp_path / "fit.xyz", fitting_configs)

    mace_params = _mace_params.copy()
    mace_params["checkpoints_dir"] = str(tmp_path)
    mace_params["model_dir"] = str(tmp_path)
    mace_params["train_file"] = tmp_path / "fit.xyz"
    args = build_default_arg_parser().parse_args(
        [f"--{k}={v}" if v is not None else f"--{k}" for k, v in mace_params.items()]
    )

    mace_run(args)

    calc = MACECalculator(model_paths=tmp_path / "MACEPME.model", device="cpu")

    Es = []
    for at in fitting_configs:
        at.calc = calc
        Es.append(at.get_potential_energy())


@pytest.mark.skipif(not PME_AVAILABLE, reason="torch-pme library is not available")
def test_run_train_with_mp(tmp_path, fitting_configs):
    ase.io.write(tmp_path / "fit.xyz", fitting_configs)

    mace_params = _mace_params.copy()
    mace_params["checkpoints_dir"] = str(tmp_path)
    mace_params["foundation_model"] = "small"
    mace_params["hidden_irreps"] = "128x0e"
    mace_params["r_max"] = 6.0
    mace_params["default_dtype"] = "float64"
    mace_params["num_radial_basis"] = 10
    mace_params["interaction_first"] = "RealAgnosticResidualInteractionBlock"
    mace_params["multiheads_finetuning"] = False
    mace_params["model_dir"] = str(tmp_path)
    mace_params["train_file"] = tmp_path / "fit.xyz"
    args = build_default_arg_parser().parse_args(
        [f"--{k}={v}" if v is not None else f"--{k}" for k, v in mace_params.items()]
    )

    mace_run(args)

    calc = MACECalculator(model_paths=tmp_path / "MACEPME.model", device="cpu")

    Es = []
    for at in fitting_configs:
        at.calc = calc
        Es.append(at.get_potential_energy())


@pytest.mark.skipif(
    not (PME_AVAILABLE and CUET_AVAILABLE and CUDA_AVAILABLE),
    reason="Testing MACEPME cueq training requires torch-pme, cuequivariance, and CUDA to be available",
)
def test_run_train_macepes_cueq(tmp_path, fitting_configs):
    ase.io.write(tmp_path / "fit.xyz", fitting_configs)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    mace_params = _mace_params.copy()
    mace_params["checkpoints_dir"] = str(tmp_path)
    mace_params["model_dir"] = str(tmp_path)
    mace_params["train_file"] = tmp_path / "fit.xyz"
    mace_params["device"] = "cuda"
    mace_params["enable_cueq"] = True
    args = build_default_arg_parser().parse_args(
        [f"--{k}={v}" if v is not None else f"--{k}" for k, v in mace_params.items()]
    )
    # Seed torch, and enable deterministic algorithms for reproducibility
    torch.manual_seed(5)
    torch.use_deterministic_algorithms(True)

    # Run the training
    mace_run(args)

    calc = MACECalculator(model_paths=tmp_path / "MACEPME.model", device="cpu")

    Es = []
    for at in fitting_configs:
        at.calc = calc
        Es.append(at.get_potential_energy())


MODEL_CONFIG = dict(
    r_max=5,
    num_bessel=8,
    num_polynomial_cutoff=6,
    max_ell=2,
    interaction_cls=interaction_classes["RealAgnosticResidualInteractionBlock"],
    interaction_cls_first=interaction_classes["RealAgnosticResidualInteractionBlock"],
    num_interactions=5,
    num_elements=2,
    hidden_irreps=o3.Irreps("32x0e + 32x1o"),
    MLP_irreps=o3.Irreps("16x0e"),
    gate=torch.nn.functional.silu,
    atomic_energies=np.zeros(2),
    avg_num_neighbors=8,
    atomic_numbers=[1, 8],
    correlation=3,
    radial_type="bessel",
    atomic_inter_shift=0.0,
    atomic_inter_scale=1.0,
    pme_arguments={
        "lr_wavelength": 0.5,
    },
)


@pytest.mark.skipif(not PME_AVAILABLE, reason="torch-pme library is not available")
@pytest.fixture(name="macepme_model_path")
def macepme_model_path_fixture(tmp_path: Path) -> Path:
    """Create and save a MACEPME model."""
    with default_dtype(torch.float32):
        model = MACEPME(**MODEL_CONFIG)
        path = tmp_path / "MACEPME.model"
        torch.save(model, path)
    return path


if __name__ == "__main__":
    import pytest
    import sys

    # Run pytest with the Python debugger on failure
    sys.exit(pytest.main([__file__, "--pdb"]))

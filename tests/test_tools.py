import tempfile

import numpy as np
import torch
import torch.nn.functional
from torch import nn, optim
from torch.optim.swa_utils import AveragedModel

from mace.tools import (
    AtomicNumberTable,
    CheckpointHandler,
    CheckpointState,
    atomic_numbers_to_indices,
)
from mace.tools.scripts_utils import unwrap_swa_model


def test_atomic_number_table():
    table = AtomicNumberTable(zs=[1, 8])
    array = np.array([8, 8, 1])
    indices = atomic_numbers_to_indices(array, z_table=table)
    expected = np.array([1, 1, 0], dtype=int)
    assert np.allclose(expected, indices)


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 4)

    def forward(self, x):
        return torch.nn.functional.relu(self.linear(x))


def test_save_load():
    model = MyModel()
    initial_lr = 0.001
    optimizer = optim.SGD(model.parameters(), lr=initial_lr, momentum=0.9)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.99)

    with tempfile.TemporaryDirectory() as directory:
        handler = CheckpointHandler(directory=directory, tag="test", keep=True)
        handler.save(state=CheckpointState(model, optimizer, scheduler), epochs=50)

        optimizer.step()
        scheduler.step()
        assert not np.isclose(optimizer.param_groups[0]["lr"], initial_lr)

        handler.load_latest(state=CheckpointState(model, optimizer, scheduler))
        assert np.isclose(optimizer.param_groups[0]["lr"], initial_lr)


def test_save_load_swa_model():
    model = MyModel()
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.99)
    swa_model = AveragedModel(model)

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(1.0)
    swa_model.update_parameters(model)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(3.0)
    swa_model.update_parameters(model)

    expected_state = {
        name: tensor.clone() for name, tensor in swa_model.state_dict().items()
    }
    with tempfile.TemporaryDirectory() as directory:
        handler = CheckpointHandler(
            directory=directory, tag="test", keep=True, swa_start=1
        )
        handler.save(
            state=CheckpointState(model, optimizer, scheduler, swa_model), epochs=1
        )

        with torch.no_grad():
            for parameter in swa_model.parameters():
                parameter.zero_()
        handler.load_latest(
            state=CheckpointState(model, optimizer, scheduler, swa_model), swa=True
        )

    for name, tensor in swa_model.state_dict().items():
        assert torch.equal(tensor, expected_state[name])


def test_unwrap_swa_model():
    model = MyModel()
    swa_model = AveragedModel(model)

    assert unwrap_swa_model(model) is model
    assert unwrap_swa_model(swa_model) is swa_model.module

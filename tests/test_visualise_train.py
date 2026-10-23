import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import mace.tools  # noqa: F401
from mace.cli.visualise_train import (
    colors,
    plot_epoch_dependence,
    plot_inference_from_results,
)


def test_plot_epoch_dependence_skips_missing_metric():
    data = pd.DataFrame(
        [
            {"mode": "opt", "epoch": 1, "head": "head", "loss": 1.0},
            {
                "mode": "eval",
                "epoch": 1,
                "head": "head",
                "loss": 1.0,
                "rmse_e_per_atom": 0.1,
                "rmse_stress": None,
            },
        ]
    )
    fig, axes = matplotlib.pyplot.subplots(1, 2)

    plot_epoch_dependence(
        axes,
        data,
        "head",
        "1",
        [
            ("rmse_e_per_atom", "RMSE E/atom [meV]"),
            ("rmse_stress", "RMSE Stress [meV / A^3]"),
        ],
    )

    assert len(axes[1].lines) == 2  # energy error and loaded-model marker
    assert not np.isnan(axes[1].lines[0].get_ydata()).any()
    matplotlib.pyplot.close(fig)


def test_test_bec_scatter_uses_grey_circles():
    fig, ax = matplotlib.pyplot.subplots()
    plot_inference_from_results(
        np.array([ax]),
        train_valid_dict={
            "valid_head": {
                "bec": {"reference": np.array([1.0]), "predicted": np.array([1.1])}
            }
        },
        test_dict={
            "test_head": {
                "bec": {"reference": np.array([2.0]), "predicted": np.array([2.1])}
            }
        },
        head="head",
        quantities=[("bec", "Born Charges [e]")],
    )

    test_scatter = ax.collections[-1]
    assert test_scatter.get_label() == "Test"
    np.testing.assert_allclose(
        test_scatter.get_facecolor()[0], matplotlib.colors.to_rgba(colors[2])
    )
    assert test_scatter.get_paths()[0].vertices.shape == (26, 2)
    matplotlib.pyplot.close(fig)

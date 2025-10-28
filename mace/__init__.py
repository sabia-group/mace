import os

from .__version__ import __version__
import warnings

warnings.filterwarnings(
    "ignore", category=UserWarning, message=".*TorchScript type system.*"
)

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

import os
import warnings

from .__version__ import __version__

warnings.filterwarnings(
    "ignore", category=UserWarning, message=".*TorchScript type system.*"
)

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

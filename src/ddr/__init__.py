from . import validation
from ._version import __version__
from .geometry import GeometryPredictor
from .io import functions as ddr_functions
from .io.readers import StreamflowReader as streamflow
from .nn import kan
from .routing.torch_mc import dmc

try:
    from . import bmi
except ImportError:
    bmi = None  # type: ignore[assignment]

__all__ = [
    "__version__",
    "dmc",
    "streamflow",
    "ddr_functions",
    "kan",
    "validation",
    "bmi",
    "GeometryPredictor",
]

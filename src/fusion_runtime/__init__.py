"""Public package surface for Fusion Runtime."""

from .config import FusionSpec, load_spec
from .runtime import FusionRuntime

__all__ = ["FusionRuntime", "FusionSpec", "load_spec"]
__version__ = "0.1.0"

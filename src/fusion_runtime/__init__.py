"""Public package surface for Fusion Runtime."""

from .config import FusionSpec, load_spec
from .errors import CapabilityError
from .runtime import FusionRuntime
from .types import ThinkingConfig

__all__ = [
    "CapabilityError",
    "FusionRuntime",
    "FusionSpec",
    "ThinkingConfig",
    "load_spec",
]
__version__ = "0.1.0"

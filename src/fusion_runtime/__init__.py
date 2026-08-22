"""Public package surface for Fusion Runtime."""

from .config import FusionSpec, load_spec
from .errors import CapabilityError
from .runtime import FusionRuntime
from .types import RecoveryOutcome, RecoveryRecord, ThinkingConfig

__all__ = [
    "CapabilityError",
    "FusionRuntime",
    "FusionSpec",
    "RecoveryOutcome",
    "RecoveryRecord",
    "ThinkingConfig",
    "load_spec",
]
__version__ = "0.1.0"

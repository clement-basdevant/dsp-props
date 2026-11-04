"""dsp-props: property-based testing for digital signal processing code."""

from . import core, generators, relations
from .core import calibrate
from .api import (
    PropertyViolation,
    Report,
    LTISpec,
    MultirateSpec,
    resampler,
    StreamingSpec,
    lti_filter,
    TransformSpec,
    stateful,
    streaming,
    transform,
    verify,
)

__version__ = "0.1.3"

__all__ = [
    "core",
    "generators",
    "relations",
    "transform",
    "calibrate",
    "streaming",
    "lti_filter",
    "LTISpec",
    "resampler",
    "MultirateSpec",
    "stateful",
    "StreamingSpec",
    "verify",
    "Report",
    "TransformSpec",
    "PropertyViolation",
]

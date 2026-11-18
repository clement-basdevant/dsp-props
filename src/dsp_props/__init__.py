"""dsp-props: property-based testing for digital signal processing code."""

from . import core, generators, relations
from .api import (
    PROFILES,
    LTISpec,
    MultirateSpec,
    Profile,
    PropertyViolation,
    Report,
    StreamingSpec,
    TransformSpec,
    get_profile,
    lti_filter,
    resampler,
    set_profile,
    stateful,
    streaming,
    transform,
    verify,
)
from .core import calibrate

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
    "Profile",
    "PROFILES",
    "get_profile",
    "set_profile",
    "Report",
    "TransformSpec",
    "PropertyViolation",
]

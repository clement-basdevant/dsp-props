"""Cross-precision relation: the float32 path of an operator agrees with its float64 path."""

from __future__ import annotations

import numpy as np
from hypothesis import strategies as st

from ..core.tolerance import ToleranceError, assert_close, check_precision, norm
from ..generators import lengths, signals
from .base import Relation


@st.composite
def _float32_signal(draw, max_n):
    n = draw(lengths(max_n))
    return draw(signals(n, np.float32))


def _run(spec, x):
    """The operator's output on ``x``, as the relations of its kind see it."""
    from ..api import StreamingSpec, TransformSpec

    if isinstance(spec, StreamingSpec):
        return np.asarray(spec.make().process(x))
    if isinstance(spec, TransformSpec) and spec.is_fourier:
        return np.asarray(spec.full(x))
    return np.asarray(spec.func(x))


class Float32Consistency(Relation):
    """f(x) in float32 agrees with f(x) in float64, to float32 accuracy, for x exactly representable in both.

    Applies to operators declared with ``float32=True`` (and the default
    float64 precision). The float32 input is drawn within float32 range
    (amplitudes 1, 1e-3, 1e3), cast exactly to float64, and both outputs are
    compared with the declared tolerance model at float32 epsilon. The float32
    output may be float64 (upcasting is not a fault) but not less precise
    than float32 (float16 is). A float32 path that is a different algorithm,
    a dtype-dependent branch, or arithmetic in half precision shows here.
    """

    name = "float32 consistency"

    def applies(self, spec):
        return getattr(spec, "float32", False) and np.dtype(spec.precision) == np.float64

    def inputs(self, spec):
        return _float32_signal(spec.max_n)

    def check(self, spec, x):
        from ..api import LTISpec

        x32 = np.asarray(x, dtype=np.float32)
        x64 = x32.astype(np.float64)  # exact
        y64 = _run(spec, x64)
        y32 = _run(spec, x32)
        what = f"{self.name} (float32 output against float64 output)"
        check_precision(y32, np.float32, what)
        if y32.shape != y64.shape:
            raise ToleranceError(f"{what}: shape {y32.shape} in float32, {y64.shape} in float64")
        # filters bound their error relative to the input, like their other relations
        ref = norm(x64) if isinstance(spec, LTISpec) else norm(y64)
        assert_close(y32.astype(np.result_type(y32.dtype, y64.dtype)), y64, spec.tolerance, ref_norm=ref,
                     what=what, precision=np.float32)

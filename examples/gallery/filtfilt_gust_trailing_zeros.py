"""filtfilt(method="gust") depends on how the filter is written (scipy/scipy#26610).

``b, a`` and the same coefficients followed by zeros describe the same
filter, so ``filtfilt`` must return the same output for both. With
``method="gust"`` it does not: the trailing zeros raise the order, add an
impulse at the edges of the output and make Gustafsson's least-squares
system singular. Reproduced with SciPy 1.15 to 1.18.1.

There is no catalogue relation for "two spellings of one filter", so this is
a plain Hypothesis test built from dsp-props parts: the filter strategy
``iir_coefficients``, the signal strategy ``signals`` and the tolerance
check ``assert_close``.
"""

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import signal

from dsp_props.core import FFTTolerance, assert_close
from dsp_props.core.tolerance import norm
from dsp_props.generators import iir_coefficients, signals

TOLERANCE = FFTTolerance(c=1e4)  # filtfilt chains two passes and a least-squares solve


@settings(max_examples=200, deadline=None, database=None)
@given(filt=iir_coefficients(max_order=3),
       x=st.integers(16, 128).flatmap(signals),
       zeros=st.integers(1, 2))
def test_trailing_zeros_do_not_change_filtfilt(filt, x, zeros):
    b, a = filt
    y = signal.filtfilt(b, a, x, method="gust")
    padded = signal.filtfilt(np.r_[b, np.zeros(zeros)], np.r_[a, np.zeros(zeros)], x, method="gust")
    assert_close(padded, y, TOLERANCE, ref_norm=max(norm(y), norm(x)), what="filtfilt(gust) with trailing zeros")


if __name__ == "__main__":
    test_trailing_zeros_do_not_change_filtfilt()

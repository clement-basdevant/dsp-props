"""Hypothesis strategies for operator parameters: filter coefficients and rate pairs.

A declaration fixes its parameters, so a fault that shows only for other
parameter values (an order-0 filter, a numerator shorter than the
denominator, a rate of 1) is invisible to the relations. These strategies,
used with ``dp.family``, draw the parameters too. They shrink towards the
degenerate cases that break DSP code most often: order 0, a single
coefficient, ``a[0] = 1``, ``L = M = 1``.
"""

from __future__ import annotations

from math import gcd

import numpy as np
from hypothesis import strategies as st

# Readable coefficients, simplest first (Hypothesis shrinks a sampled value towards the first).
COEFFICIENTS = [1.0, 0.5, -1.0, 0.0, 2.0, -0.25, 0.75, -1.5, 0.125]
# Pole radii, shrinking towards 0.5; 0 gives a pure delay in the recursion.
POLE_RADII = [0.5, 0.0, 0.3, 0.9, 0.95, 0.99]
# Leading denominator coefficients: implementations must normalise by a[0].
LEADING = [1.0, 2.0, 0.5, -1.0, 4.0]


def fir_taps(max_taps: int = 16) -> st.SearchStrategy[np.ndarray]:
    """FIR coefficients ``b``: 1 to ``max_taps`` taps, at least one of them non-zero."""
    return (st.lists(st.sampled_from(COEFFICIENTS), min_size=1, max_size=max_taps)
            .filter(lambda b: any(b)).map(lambda b: np.array(b, dtype=float)))


@st.composite
def _poles(draw, count: int, radii):
    poles: list[complex] = []
    while len(poles) < count:
        r = draw(st.sampled_from(radii))
        if count - len(poles) >= 2 and draw(st.booleans()):
            theta = draw(st.floats(0.05, np.pi - 0.05))
            p = r * np.exp(1j * theta)
            poles += [p, np.conj(p)]
        else:
            poles.append(r * draw(st.sampled_from([1.0, -1.0])))
    return poles


@st.composite
def iir_coefficients(draw, max_order: int = 6, max_numerator: int | None = None,
                     radii=tuple(POLE_RADII)) -> tuple[np.ndarray, np.ndarray]:
    """Stable ``(b, a)``: denominators of order 0 to ``max_order``, numerators of any length.

    The poles are real or complex-conjugate pairs of radius in ``radii``
    (up to 0.99 by default: close to the unit circle, but not so close that
    the impulse response outgrows the checks); the numerator has 1 to
    ``max_numerator`` coefficients (default ``max_order + 2``), so it is
    shorter than, as long as, or longer than the denominator; ``a[0]`` is not
    always 1. Order 0 (``a = [a0]``) is a gain, or an FIR filter.
    """
    max_numerator = max_order + 2 if max_numerator is None else max_numerator
    order = draw(st.integers(0, max_order))
    poles = draw(_poles(order, list(radii)))
    a = np.real(np.poly(poles)) if poles else np.ones(1)
    a = draw(st.sampled_from(LEADING)) * a
    b = draw(fir_taps(max_numerator))
    return b, a


def rate_pairs(max_rate: int = 8) -> st.SearchStrategy[tuple[int, int]]:
    """``(up, down)`` coprime, each in 1..``max_rate``, ``(1, 1)`` included (it shrinks there)."""

    def reduce(pair):
        g = gcd(*pair)
        return pair[0] // g, pair[1] // g

    return st.tuples(st.integers(1, max_rate), st.integers(1, max_rate)).map(reduce)

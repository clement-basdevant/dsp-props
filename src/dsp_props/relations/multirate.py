"""Relations for rational resamplers (up by L, down by M), applied with zero initial state.

A resampler is not time-invariant but L/M periodically time-varying:
delaying the input by M samples delays the output by L samples. Tolerances
are relative to the norm of the input, as for LTI filters.
"""

from __future__ import annotations

import numpy as np
from hypothesis import strategies as st

from ..core.tolerance import assert_close, check_precision, norm
from ..generators import lengths, signals
from .base import Relation
from .lti import LTILinearity


@st.composite
def _signal(draw, max_n, dtype=np.float64):
    n = draw(lengths(max_n))
    return draw(signals(n, dtype))


def _long_length(spec):
    """Lengths long enough for the middle half of the output to be free of edge transients."""
    n0 = 64 * max(spec.up, spec.down)
    return st.integers(n0, max(n0, 4 * n0, spec.max_n))


def _interior(y):
    m = len(y)
    return y[m // 4 : m - m // 4]


class OutputLength(Relation):
    """len(f(x)) follows the declared length rule (default ceil(N L / M))."""

    name = "output length"

    def applies(self, spec):
        return spec.length is not None

    def inputs(self, spec):
        return lengths(spec.max_n)

    def check(self, spec, n):
        got = len(np.asarray(spec.func(np.zeros(n, spec.precision))))
        want = spec.expected_length(n)
        if got != want:
            raise AssertionError(f"{self.name}: N = {n} gave {got} output samples, expected {want}")


class PeriodicShift(Relation):
    """Delaying the input by M s samples delays the output by L s samples."""

    name = "periodic time invariance"

    def inputs(self, spec):
        return st.tuples(_signal(spec.max_n, spec.precision), st.integers(1, 8))

    def check(self, spec, inputs):
        x, s = inputs
        L, M = spec.up, spec.down
        y = np.asarray(spec.func(x))
        yd = np.asarray(spec.func(np.concatenate([np.zeros(M * s, x.dtype), x])))
        what = f"{self.name} (input delay {M * s}, output delay {L * s})"
        if len(yd) != len(y) + L * s:
            raise AssertionError(f"{what}: delayed output has {len(yd)} samples, expected {len(y) + L * s}")
        expected = np.concatenate([np.zeros(L * s), y]) if spec.causal else y
        actual = yd if spec.causal else yd[L * s :]
        assert_close(actual, expected, spec.tolerance, ref_norm=norm(x), what=what, precision=spec.precision)


class DCGain(Relation):
    """A constant input gives a constant output of the declared gain, away from the edges."""

    name = "DC gain"

    def applies(self, spec):
        return spec.dc_gain is not None

    def inputs(self, spec):
        return st.tuples(_long_length(spec), st.sampled_from([1.0, -3.0, 1e-3, 1e3]))

    def check(self, spec, inputs):
        n, c = inputs
        y = np.asarray(spec.func(np.full(n, c, spec.precision)))
        check_precision(y, spec.precision, self.name)
        y = _interior(y)
        err = float(np.max(np.abs(y - spec.dc_gain * c))) if len(y) else 0.0
        if not err <= spec.ripple * abs(spec.dc_gain * c):
            raise AssertionError(
                f"{self.name}: constant {c:g} over N = {n} gave interior output up to {err:.3e} "
                f"away from {spec.dc_gain * c:g} (ripple allowed {spec.ripple:g})"
            )


class AliasRejection(Relation):
    """A tone above the output Nyquist frequency is attenuated, not folded back."""

    name = "alias rejection"

    def applies(self, spec):
        # The tone must fit between 1.25 x the output Nyquist and 0.95 x the input Nyquist.
        return spec.stopband is not None and 1.25 * spec.up / spec.down < 0.9

    def inputs(self, spec):
        f_lo = 1.25 * spec.up / spec.down
        return st.tuples(_long_length(spec), st.floats(f_lo, 0.95), st.floats(0, 2 * np.pi))

    def check(self, spec, inputs):
        n, f, phase = inputs
        x = np.cos(np.pi * f * np.arange(n) + phase).astype(spec.precision)  # f in units of the input Nyquist
        y = np.asarray(spec.func(x))
        check_precision(y, spec.precision, self.name)
        y = _interior(y)
        rms = float(np.sqrt(np.mean(y ** 2))) if len(y) else 0.0
        limit = spec.stopband / np.sqrt(2)
        if not rms <= limit:
            raise AssertionError(
                f"{self.name}: tone at {f:.4f} x input Nyquist (output Nyquist is "
                f"{spec.up / spec.down:.4f}) leaves rms {rms:.3e} > {limit:.3e} after resampling"
            )


MULTIRATE_RELATIONS = [LTILinearity(), OutputLength(), PeriodicShift(), DCGain(), AliasRejection()]

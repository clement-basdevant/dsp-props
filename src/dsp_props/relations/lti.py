"""Relations for linear time-invariant filters (zero initial state).

Tolerances here are relative to the norm of the *input* signals, which is
what filter error models such as ``IIRTolerance`` bound.
"""

from __future__ import annotations

import numpy as np
from hypothesis import strategies as st

from ..core.tolerance import assert_close, check_precision, norm
from ..generators import lengths, signal_pairs, signals
from .base import Relation

REAL_COEFFS = [1.0, -1.0, 2.0, 0.5, -0.75, 3.0]


@st.composite
def _signal(draw, max_n, dtype=np.float64):
    n = draw(lengths(max_n))
    return draw(signals(n, dtype))


class LTILinearity(Relation):
    """f(a x + b y) = a f(x) + b f(y)."""

    name = "linearity"

    def inputs(self, spec):
        c = st.sampled_from(REAL_COEFFS)
        return st.tuples(signal_pairs(spec.max_n, dtype=spec.precision), c, c)

    def check(self, spec, inputs):
        (x, y), a, b = inputs
        f = spec.func
        expected = a * np.asarray(f(x)) + b * np.asarray(f(y))
        assert_close(f(a * x + b * y), expected, spec.tolerance,
                     ref_norm=abs(a) * norm(x) + abs(b) * norm(y), what=self.name, precision=spec.precision)


class TimeInvariance(Relation):
    """Delaying the input by s samples delays the output by s samples (zero state)."""

    name = "time invariance"

    def inputs(self, spec):
        return st.tuples(_signal(spec.max_n, spec.precision), st.integers(1, 32))

    def check(self, spec, inputs):
        x, s = inputs
        y = np.asarray(spec.func(x))
        yd = np.asarray(spec.func(np.concatenate([np.zeros(s, x.dtype), x])))
        if not spec.causal:
            # a non-causal filter may respond before the delayed input starts
            yd = yd[s:]
        expected = np.concatenate([np.zeros(s), y]) if spec.causal else y
        assert_close(yd, expected, spec.tolerance, ref_norm=norm(x), what=f"{self.name} (delay {s})",
                     precision=spec.precision)


class Causality(Relation):
    """Changing x[k:] leaves y[:k] unchanged."""

    name = "causality"

    def applies(self, spec):
        return spec.causal

    @staticmethod
    @st.composite
    def _case(draw, max_n, dtype=np.float64):
        n = draw(lengths(max_n).filter(lambda m: m >= 2))
        x = draw(signals(n, dtype))
        k = draw(st.integers(1, n - 1))
        bump = draw(st.sampled_from([1.0, -1e3, 1e-3]))
        return x, k, bump

    def inputs(self, spec):
        return self._case(spec.max_n, spec.precision)

    def check(self, spec, inputs):
        x, k, bump = inputs
        x2 = x.copy()
        x2[k:] += bump * (1 + np.arange(len(x) - k))
        y1 = np.asarray(spec.func(x))[:k]
        y2 = np.asarray(spec.func(x2))[:k]
        assert_close(y2, y1, spec.tolerance, ref_norm=norm(x) + norm(x2), what=f"{self.name} (k={k})",
                     precision=spec.precision)


class TransferConsistency(Relation):
    """The measured frequency response matches the declared transfer function B(z)/A(z).

    The impulse response is measured over L samples and its DFT is compared
    with H(e^{jw}) on the same grid. L is capped at 2^16, so with a pole close
    to the unit circle the response has not decayed by then: the comparison
    then carries the truncated tail, bounded by ``tail_l1_bound``.
    """

    name = "transfer function"

    def applies(self, spec):
        return spec.transfer is not None

    def inputs(self, spec):
        return st.just(None)

    def check(self, spec, _):
        from scipy.signal import freqz

        from ..core.iir import tail_l1_bound

        b, a = (np.asarray(c, dtype=float) for c in spec.transfer)
        poles = np.roots(a / a[0]) if len(a) > 1 else np.array([])
        r = float(np.max(np.abs(poles))) if poles.size else 0.0
        L = int(min(1 << 16, (np.log(1e-17) / np.log(r) if r > 0 else 1) * len(a) + len(b) + 16))
        L = max(L, len(a) + len(b))  # tail_l1_bound needs the recursion alone beyond L
        impulse = np.zeros(L, spec.precision)
        impulse[0] = 1.0
        h = np.asarray(spec.func(impulse))
        check_precision(h, spec.precision, self.name)
        measured = np.fft.fft(h)
        _, declared = freqz(b, a, worN=L, whole=True)
        # On the L-point grid, H(e^{jw_k}) is the DFT of the aliased response
        # sum_q h[m + qL], i.e. of h[:L] plus t[m] = sum_{q>=1} h[m + qL]. So
        # ||H_meas - H||_2 <= sqrt(L) (||h_meas - h[:L]||_2 + ||t||_2) (Parseval,
        # the impulse has norm 1), with ||t||_2 <= ||t||_1 <= sum_{n>=L} |h[n]|.
        # That tail is not small when L is capped (r^L = 1.4e-3 for r = 0.9999).
        tail = tail_l1_bound(b, a, L)
        tol = np.sqrt(L) * (spec.tolerance.bound(L, spec.precision, 1.0) + tail)
        err = norm(measured - declared)
        if not err <= tol:
            worst = int(np.argmax(np.abs(measured - declared)))
            raise AssertionError(
                f"{self.name}: ||H_measured - H_declared|| = {err:.3e} > {tol:.3e}; worst bin "
                f"w = {2 * np.pi * worst / L:.4f} rad/sample: measured {measured[worst]:.6g}, "
                f"declared {declared[worst]:.6g}"
            )


class BoundedGain(Relation):
    """No input bounded by 1 is amplified beyond the declared gain, for N <= max_n.

    The worst input of length N is sign(h[N-1-k]) (h measured from the
    impulse response): it drives y[N-1] to sum |h[:N]|, the largest possible
    output for |x| <= 1. Only lengths up to ``max_n`` are tried, so this is a
    bounded gain on N <= max_n, not BIBO stability: a response that grows
    beyond max_n samples is not seen.
    """

    name = "bounded gain"

    def applies(self, spec):
        return spec.max_gain is not None

    def inputs(self, spec):
        return lengths(spec.max_n).filter(lambda m: m >= 1)

    def check(self, spec, n):
        impulse = np.zeros(n, spec.precision)
        impulse[0] = 1.0
        h = np.asarray(spec.func(impulse))[:n]
        x = np.sign(h[::-1]).astype(spec.precision)
        y = np.asarray(spec.func(x))
        check_precision(y, spec.precision, self.name)
        if not np.all(np.isfinite(y)):
            raise AssertionError(f"{self.name}: non-finite output for a bounded input of length {n}")
        peak = float(np.max(np.abs(y))) if len(y) else 0.0
        limit = spec.max_gain + spec.tolerance.bound(n, spec.precision, np.sqrt(n))
        if not peak <= limit:
            raise AssertionError(
                f"{self.name}: input bounded by 1 (worst case for N = {n}) gave |y| = {peak:.6g} "
                f"> declared gain {spec.max_gain:g}"
            )


LTI_RELATIONS = [LTILinearity(), TimeInvariance(), Causality(), TransferConsistency(), BoundedGain()]

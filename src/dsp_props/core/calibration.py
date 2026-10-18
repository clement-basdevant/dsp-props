"""Empirical error models: calibrate a tolerance against extended precision.

A fixed formula such as c * eps * log2(N) cannot describe every
implementation: an FFT library may switch algorithm with the factorisation
of N (mixed radix, naive small DFTs, Bluestein), and its error then varies by
two orders of magnitude between neighbouring lengths. ``calibrate`` measures
the actual relative error of an operator against an extended-precision
reference for each length and turns it into a tolerance model.

Calibration doubles as a sanity check: an error far above what rounding can
explain (``max_plausible``) means the operator is wrong, not imprecise, and
calibration refuses to absorb it.

The default reference, ``extended_dft``, is the long double FFT where it is
really extended (Linux x86-64 with NumPy >= 2), and a double-double DFT
elsewhere: long double is float64 under MSVC and on macOS arm64, and NumPy
1.x computes the FFT of a clongdouble array in complex128.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from .doubledouble import DoubleDouble, doubledouble_dft
from .tolerance import FFTTolerance, ToleranceError, ToleranceModel, norm


def longdouble_dft(x):
    """DFT in long double. Extended only where ``longdouble_fft_is_extended()``."""
    return np.fft.fft(np.asarray(x).astype(np.clongdouble))


@lru_cache(maxsize=None)
def longdouble_fft_is_extended():
    """True when ``longdouble_dft`` is more accurate than float64 on this platform.

    Checks the type (long double is float64 under MSVC and on macOS arm64) and
    then the FFT itself against ``doubledouble_dft`` (NumPy 1.x returns
    complex128 for a clongdouble input).
    """
    if not np.finfo(np.longdouble).eps < np.finfo(np.float64).eps:
        return False
    n = 97  # prime: exercises the slowest, least accurate code path
    x = np.cos(0.7 * np.arange(n)) + np.sin(1.3 * np.arange(n) ** 2)
    got = np.asarray(longdouble_dft(x))
    if got.dtype != np.clongdouble:
        return False
    ref = doubledouble_dft(x)
    err = norm(((got - ref.hi) - ref.lo).astype(np.complex128)) / norm(ref.hi)
    return err < 1e-2 * np.finfo(np.float64).eps


def extended_dft(x):
    """Default calibration reference: long double where extended, double-double otherwise."""
    return longdouble_dft(x) if longdouble_fft_is_extended() else doubledouble_dft(x)


def _eps(dtype):
    return float(np.finfo(np.result_type(dtype, np.float32)).eps)


def relative_error(op, reference, x):
    """||op(x) - reference(x)|| / (eps * ||reference(x)||), eps of op's output dtype.

    ``reference`` returns an array, or a ``DoubleDouble`` (hi, lo) pair.
    """
    y = np.asarray(op(x))
    ref = reference(x)
    hi = ref.hi if isinstance(ref, DoubleDouble) else np.asarray(ref)
    if y.shape != hi.shape:
        raise ToleranceError(f"shape {y.shape} != reference shape {hi.shape}")
    ref_norm = float(norm(hi.astype(np.complex128)))
    if ref_norm == 0.0:
        return 0.0 if norm(y) == 0 else np.inf
    if isinstance(ref, DoubleDouble):
        diff = ((y - ref.hi) - ref.lo).astype(np.complex128)  # y - hi is exact when close (Sterbenz)
    else:
        diff = (y.astype(np.clongdouble) - ref).astype(np.complex128)
    return float(norm(diff)) / (_eps(y.dtype) * ref_norm)


@dataclass
class CalibratedTolerance(ToleranceModel):
    """Analytic bound, raised where measurements show the implementation is worse.

    bound(N) = max(base.bound(N), margin * chained * eps * ratio(N) * ||y||)

    ``ratios[N]`` is the worst measured error for length N, in units of eps.
    Unmeasured lengths fall back to the worst ratio over all measured lengths,
    which keeps the model conservative. The analytic ``base`` model covers
    the inputs calibration did not try.
    """

    ratios: dict = field(default_factory=dict)
    margin: float = 8.0
    chained: int = 2  # relations chain a few operations (e.g. inverse(f(x)))
    base: ToleranceModel = field(default_factory=FFTTolerance)

    @property
    def worst(self):
        return max(self.ratios.values(), default=1.0)

    def ratio(self, n):
        return max(self.ratios.get(n, self.worst), 1.0)

    def bound(self, n, dtype, ref_norm):
        measured = self.margin * self.chained * _eps(dtype) * self.ratio(n) * ref_norm
        return max(self.base.bound(n, dtype, ref_norm), measured)

    def summary(self, top=5):
        worst = sorted(self.ratios.items(), key=lambda kv: -kv[1])[:top]
        lines = [f"calibrated on {len(self.ratios)} lengths, worst ratio {self.worst:.1f} eps"]
        lines += [f"  N={n:<6} {r:8.1f} eps" for n, r in worst]
        return "\n".join(lines)


class CalibrationError(AssertionError):
    """Raised by ``calibrate`` when the measured error is too large to be rounding."""


def calibrate(op, reference=extended_dft, lengths=range(1, 257), trials=6, *,
              margin=8.0, max_plausible=1e6, seed=0) -> CalibratedTolerance:
    """Measure ``op`` against ``reference`` on random signals of each length.

    The default reference is ``extended_dft``; pass ``reference=doubledouble_dft``
    to force the platform-independent one.

    Raises ``CalibrationError`` when the error is too large to be rounding
    (``max_plausible`` eps): calibrating a wrong operator would hide its bug.
    """
    rng = np.random.default_rng(seed)
    ratios = {}
    for n in lengths:
        worst = 0.0
        for x in _probe_signals(n, trials, rng):
            worst = max(worst, relative_error(op, reference, x))
        if not worst <= max_plausible:
            raise CalibrationError(
                f"error of {worst:.3g} eps at N={n}: the operator disagrees with the reference "
                "beyond any plausible rounding error"
            )
        ratios[n] = worst
    return CalibratedTolerance(ratios=ratios, margin=margin)


def _probe_signals(n, trials, rng):
    """Gaussian noise plus the structured signals that stress FFT rounding."""
    t = np.arange(n)
    for _ in range(trials):
        yield rng.standard_normal(n)
    x = np.zeros(n)
    x[rng.integers(n)] = 1.0
    yield x
    yield np.cos(2 * np.pi * rng.uniform(0, 0.5) * t + rng.uniform(-np.pi, np.pi))
    yield np.cos(np.pi * t) + 1e-6 * rng.standard_normal(n)  # near Nyquist, small detail

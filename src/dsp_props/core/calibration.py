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
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .tolerance import FFTTolerance, ToleranceError, ToleranceModel, norm


def longdouble_dft(x):
    """Extended-precision DFT reference (NumPy computes FFTs in long double)."""
    return np.fft.fft(np.asarray(x).astype(np.clongdouble))


def _eps(dtype):
    return float(np.finfo(np.result_type(dtype, np.float32)).eps)


def relative_error(op, reference, x):
    """||op(x) - reference(x)|| / (eps * ||reference(x)||), eps of the input's dtype.

    The input sets the precision expected of ``op``: an operator that returns
    complex64 for float64 input is measured in float64 epsilons (and refused
    as implausible), not in its own.
    """
    y = np.asarray(op(x))
    ref = np.asarray(reference(x))
    if y.shape != ref.shape:
        raise ToleranceError(f"shape {y.shape} != reference shape {ref.shape}")
    ref_norm = float(norm(ref.astype(np.complex128)))
    if ref_norm == 0.0:
        return 0.0 if norm(y) == 0 else np.inf
    diff = (y.astype(np.clongdouble) - ref).astype(np.complex128)
    return float(norm(diff)) / (_eps(np.asarray(x).dtype) * ref_norm)


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


def calibrate(op, reference=longdouble_dft, lengths=range(1, 257), trials=6, *,
              margin=8.0, max_plausible=1e6, seed=0) -> CalibratedTolerance:
    """Measure ``op`` against ``reference`` on random signals of each length.

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

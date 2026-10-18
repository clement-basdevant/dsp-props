"""Operator-aware tolerances.

Phase 1 ships a single analytic model for FFT-like transforms, based on the
classical O(eps * log2 N) relative error bound of the FFT (Schatzman 1996).
Phase 2 will add per-operator models calibrated against extended precision.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class ToleranceError(AssertionError):
    """Raised when two signals differ by more than the tolerance model allows."""


class ToleranceModel:
    """Maps (signal length, dtype, reference norm) to an admissible L2 error."""

    def bound(self, n: int, dtype, ref_norm: float) -> float:
        raise NotImplementedError


@dataclass(frozen=True)
class FFTTolerance(ToleranceModel):
    """||y_hat - y||_2 <= c * eps * max(log2 N, 1) * ||y||_2 + floor.

    ``c`` absorbs the constants of the analysis and the number of chained
    transforms in a relation; it will be calibrated empirically in phase 2.
    """

    c: float = 20.0

    def bound(self, n: int, dtype, ref_norm: float) -> float:
        eps = np.finfo(np.result_type(dtype, np.float32)).eps
        logn = max(np.log2(max(n, 2)), 1.0)
        # The floor keeps exact-zero signals strict while tolerating
        # underflow on tiny ones.
        floor = np.finfo(np.result_type(dtype, np.float32)).tiny * n
        return self.c * eps * logn * ref_norm + floor


def norm(v) -> float:
    """L2 norm that does not overflow or underflow for extreme amplitudes."""
    v = np.asarray(v)
    if np.iscomplexobj(v):
        # Complex division by a subnormal scale computes 1/scale first and
        # overflows; scale the real and imaginary parts separately instead.
        v = np.concatenate([v.real.ravel(), v.imag.ravel()])
    scale = float(np.max(np.abs(v))) if v.size else 0.0
    if scale == 0.0 or not np.isfinite(scale):
        return scale
    return scale * float(np.linalg.norm(v / scale))


def precision_of(dtype):
    """The real floating dtype whose epsilon governs ``dtype`` (float32 for complex64, at least float32)."""
    return np.finfo(np.result_type(dtype, np.float32)).dtype


def check_precision(actual, expected, what=""):
    """Raise ``ToleranceError`` when the output ``actual`` is less precise than the dtype ``expected``.

    An operator that silently computes in lower precision (complex64 out of
    float64 in) must not be judged with the epsilon of its own output.
    """
    got = np.asarray(actual).dtype
    if np.finfo(precision_of(got)).eps > np.finfo(precision_of(expected)).eps:
        raise ToleranceError(
            f"{what}: output dtype {got} is less precise than the expected {np.dtype(expected)} "
            "(the precision of the input, or the one declared)"
        )


def assert_close(actual, expected, model: ToleranceModel, *, ref_norm=None, what="", precision=None):
    """Compare two arrays with an operator-aware tolerance (L2 norm).

    ``precision`` is the dtype the operator is expected to keep (its input's,
    or the declared one): a less precise ``actual`` is an error, and the
    tolerance uses its epsilon. Without it, the tolerance uses the epsilon of
    ``actual`` itself.
    """
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if precision is not None:
        check_precision(actual, precision, what)
    if actual.shape != expected.shape:
        raise ToleranceError(f"{what}: shape {actual.shape} != {expected.shape}")
    if not np.all(np.isfinite(actual)):
        raise ToleranceError(f"{what}: non-finite values in output")
    if ref_norm is None:
        ref_norm = norm(expected)
    n = max(actual.size, 1)
    err = norm(actual - expected)
    tol = model.bound(n, actual.dtype if precision is None else precision, ref_norm)
    if not err <= tol:
        raise ToleranceError(
            f"{what}: L2 error {err:.3e} exceeds tolerance {tol:.3e} (N={n}, ref norm {ref_norm:.3e})"
        )

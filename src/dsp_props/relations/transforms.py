"""Relations for discrete transforms: the DFT family, and linear transforms in general."""

from __future__ import annotations

import numpy as np
from hypothesis import strategies as st

from ..core.tolerance import ToleranceError, assert_close, check_precision, norm
from ..generators import complex_signals, lengths, signals, signal_pairs
from .base import Relation

COEFFS = [1.0, -1.0, 2.0, 0.5, 1j, -0.75 + 0.25j, 3.0 - 2.0j]
REAL_COEFFS = [1.0, -1.0, 2.0, 0.5, -0.75, 3.0]


@st.composite
def _signal(draw, max_n, complex=False, dtype=np.float64):
    n = draw(lengths(max_n))
    return draw(complex_signals(n, dtype) if complex else signals(n, dtype))


def _accepts_complex(spec):
    """Unless declared real-input only, a transform must handle complex input."""
    return not spec.real_input_only


class Linearity(Relation):
    """f(a x + b y) = a f(x) + b f(y)."""

    name = "linearity"

    def applies(self, spec):
        return spec.linear

    def inputs(self, spec):
        coeffs = REAL_COEFFS if spec.real_input_only else COEFFS
        c = st.sampled_from(coeffs)
        return st.tuples(signal_pairs(spec.max_n, complex=_accepts_complex(spec), dtype=spec.precision), c, c)

    def check(self, spec, inputs):
        (x, y), a, b = inputs
        f = spec.func
        fx, fy = np.asarray(f(x)), np.asarray(f(y))
        expected = a * fx + b * fy
        ref = abs(a) * norm(fx) + abs(b) * norm(fy)
        assert_close(f(a * x + b * y), expected, spec.tolerance, ref_norm=ref, what=self.name,
                     precision=spec.precision)


class CircularShift(Relation):
    """DFT shift theorem: f(roll(x, s))[k] = f(x)[k] * exp(-2i pi k s / N)."""

    name = "circular shift"

    def applies(self, spec):
        return spec.is_fourier

    def inputs(self, spec):
        return st.tuples(_signal(spec.max_n, _accepts_complex(spec), spec.precision), st.integers(-1024, 1024))

    def check(self, spec, inputs):
        x, s = inputs
        n = len(x)
        fx = spec.full(x)
        k = np.arange(n)
        # Reduce k*s modulo N in integers so the twiddle argument stays in
        # [0, 2*pi) and the reference itself is accurate.
        expected = fx * np.exp(-2j * np.pi * ((k * s) % n) / n)
        assert_close(spec.full(np.roll(x, s)), expected, spec.tolerance,
                     ref_norm=norm(fx), what=f"{self.name} (s={s})", precision=spec.precision)


class Parseval(Relation):
    """||f(x)||^2 = scale(N) * ||x||^2."""

    name = "parseval"

    def applies(self, spec):
        return spec.parseval

    def inputs(self, spec):
        return _signal(spec.max_n, _accepts_complex(spec), spec.precision)

    def check(self, spec, x):
        n = len(x)
        scale = spec.parseval_scale(n)
        spectrum = spec.full(x) if spec.is_fourier else np.asarray(spec.func(x))
        check_precision(spectrum, spec.precision, self.name)
        lhs = norm(spectrum) / np.sqrt(scale)
        rhs = norm(x)
        tol = spec.tolerance.bound(n, spec.precision, rhs)
        if not abs(lhs - rhs) <= tol:
            raise ToleranceError(
                f"{self.name}: ||f(x)||/sqrt({scale:g}) = {lhs:.6e} but ||x|| = {rhs:.6e} (N={n})"
            )


class HermitianSymmetry(Relation):
    """Real input: X[k] = conj(X[-k mod N])."""

    name = "hermitian symmetry"

    def applies(self, spec):
        return spec.kind == "dft" and spec.real_input_hermitian

    def inputs(self, spec):
        return _signal(spec.max_n, dtype=spec.precision)

    def check(self, spec, x):
        fx = np.asarray(spec.func(x))
        mirrored = np.conj(np.roll(fx[::-1], 1))
        assert_close(fx, mirrored, spec.tolerance, what=self.name, precision=spec.precision)


class Inversibility(Relation):
    """inverse(f(x)) = x."""

    name = "inversibility"

    def applies(self, spec):
        return spec.inverse is not None

    def inputs(self, spec):
        return _signal(spec.max_n, _accepts_complex(spec), spec.precision)

    def check(self, spec, x):
        y = spec.invert(spec.func(x), len(x))
        assert_close(y, x, spec.tolerance, what=self.name, precision=spec.precision)


def _pow2_normalise(v):
    peak = float(np.max(np.abs(v))) if len(v) else 0.0
    if peak == 0.0 or not np.isfinite(peak):
        return v
    return np.ldexp(v.real, -np.frexp(peak)[1]) + (1j * np.ldexp(v.imag, -np.frexp(peak)[1]) if np.iscomplexobj(v) else 0)


class ConvolutionTheorem(Relation):
    """inverse(f(x) * f(y)) = circular convolution of x and y (computed directly)."""

    name = "convolution theorem"

    def applies(self, spec):
        return spec.is_fourier and spec.inverse is not None

    def inputs(self, spec):
        return signal_pairs(min(spec.max_n, 256), complex=_accepts_complex(spec), dtype=spec.precision)

    def check(self, spec, inputs):
        x, y = inputs
        n = len(x)
        # The relation is bilinear: scaling each input by a power of two is exact and keeps
        # the products x[i] y[j] and X[k] Y[k] away from underflow (1e-100 x 1e-100 inputs).
        x, y = _pow2_normalise(x), _pow2_normalise(y)
        idx = (np.arange(n)[:, None] - np.arange(n)[None, :]) % n
        direct = y[idx] @ x
        fast = spec.invert(np.asarray(spec.func(x)) * np.asarray(spec.func(y)), n)
        # By Cauchy-Schwarz, |c[k]| <= ||x|| ||y||, hence ||c|| <= sqrt(N) ||x|| ||y||.
        ref = np.sqrt(n) * norm(x) * norm(y)
        assert_close(fast, direct, spec.tolerance, ref_norm=ref, what=self.name, precision=spec.precision)


class RealBins(Relation):
    """Real-input DFT: the DC bin (and the Nyquist bin for even N) are real."""

    name = "real DC/Nyquist bins"

    def applies(self, spec):
        return spec.kind == "rdft"

    def inputs(self, spec):
        return _signal(spec.max_n, dtype=spec.precision)

    def check(self, spec, x):
        n = len(x)
        X = spec.full(x)
        check_precision(X, spec.precision, self.name)
        idx = [0] + ([n // 2] if n % 2 == 0 else [])
        imag = np.imag(X[idx])
        tol = spec.tolerance.bound(n, spec.precision, norm(X))
        if not norm(imag) <= tol:
            raise ToleranceError(f"{self.name}: imaginary part {imag} at bins {idx} (N={n})")


TRANSFORM_RELATIONS = [
    Linearity(),
    CircularShift(),
    Parseval(),
    HermitianSymmetry(),
    Inversibility(),
    ConvolutionTheorem(),
    RealBins(),
]

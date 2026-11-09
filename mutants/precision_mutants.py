"""Precision mutants: operators that compute the right thing, only less accurately.

The structural mutants of the other modules break a property; these keep every
property and lose accuracy, from a few eps to about 1e6 eps. They measure what
the tolerance engine can tell apart from rounding (action M-1 of the review
plan): realistic faults, and graded ones whose loss is set by the number of
significand bits they drop.

Graded mutants keep ``53 - bits`` significand bits, that is a relative
rounding of 2**bits * eps / 2 on each value rounded, in one of three places:

- coefficients: the FFT's twiddle table, the IIR coefficients, the resampler's
  filter taps (the operator is then another, slightly different, correct
  operator of the same class);
- arithmetic: the input and output of the IIR filter and of the resampler,
  as if they computed in a narrower format;
- state: the history a streaming processor carries from one block to the next.

Rounding each output sample of a streaming processor would not do: a pointwise
function of a correct output does not depend on how the signal is split.
"""

import numpy as np
import scipy.signal as ss

import dsp_props as dp

from . import iir_mutants, multirate_mutants, streaming_mutants


def round_bits(x, bits):
    """Round ``x`` to 53 - bits significand bits (bits = 0 leaves float64 unchanged)."""
    x = np.asarray(x)
    if bits <= 0:
        return x
    if np.iscomplexobj(x):
        return round_bits(x.real, bits) + 1j * round_bits(x.imag, bits)
    m, e = np.frexp(x.astype(np.float64))
    scale = 2.0 ** (53 - bits)
    return np.ldexp(np.round(m * scale) / scale, e)


# --- FFT -------------------------------------------------------------------


def _twiddles(n, *, bits=0, single=False):
    """exp(-2i pi k/n) for k < n, accurately computed, then degraded."""
    w = np.exp(-2j * np.pi * np.arange(n) / n)
    if single:
        w = w.astype(np.complex64).astype(np.complex128)
    return round_bits(w, bits)


def dft_with_rounded_twiddles(bits):
    """Direct DFT (k*n reduced mod N) whose twiddle table keeps 53 - bits bits."""

    def dft(x):
        x = np.asarray(x)
        n = len(x)
        if n == 0:
            return x.astype(complex)
        w = _twiddles(n, bits=bits)
        k = np.arange(n)
        return w[np.outer(k, k) % n] @ x

    dft.__name__ = f"dft_twiddles_minus_{bits}_bits"
    return dft


def _radix2(x, twiddles):
    """Decimation in time while N is even, NumPy's FFT on the odd leaves."""
    n = len(x)
    if n % 2 or n < 2:
        return np.fft.fft(x)
    even, odd = _radix2(x[0::2], twiddles), _radix2(x[1::2], twiddles)
    t = twiddles(n)[: n // 2] * odd
    return np.concatenate([even + t, even - t])


def _recursive_twiddles(n):
    """w^k by the recurrence w^k = w^(k-1) * w: the error grows with k."""
    w1 = np.exp(-2j * np.pi / n)
    w = np.empty(n, complex)
    acc = 1.0 + 0j
    for k in range(n):
        w[k] = acc
        acc = acc * w1
    return w


def radix2_recursive_twiddles(x):
    """Radix-2 FFT whose twiddle factors come from a recurrence."""
    return _radix2(np.asarray(x, complex), _recursive_twiddles)


def radix2_single_twiddles(x):
    """Radix-2 FFT whose twiddle table is stored in single precision (data in double)."""
    return _radix2(np.asarray(x, complex), lambda n: _twiddles(n, single=True))


def radix2_reference(x):
    """Radix-2 FFT with accurate twiddles: the correct version of the two above."""
    return _radix2(np.asarray(x, complex), _twiddles)


def naive_dft_unreduced(x):
    """Direct DFT with exp(-2i pi k n / N): the argument grows to 2 pi N, so does its rounding."""
    x = np.asarray(x)
    n = len(x)
    if n == 0:
        return x.astype(complex)
    k = np.arange(n)
    return np.exp(-2j * np.pi * np.outer(k, k) / n) @ x


def goertzel(x):
    """One Goertzel recursion per bin, a classic replacement for a full FFT."""
    x = np.asarray(x)
    n = len(x)
    if n == 0:
        return x.astype(complex)
    k = np.arange(n)
    omega = 2 * np.pi * k / n
    coeff = 2 * np.cos(omega)
    s1 = np.zeros(n, complex)
    s2 = np.zeros(n, complex)
    for v in x:
        s1, s2 = v + coeff * s1 - s2, s1
    # X[k] = s[N-1] - exp(-i omega) s[N-2], times exp(-i omega (N-1)) to align the phase
    return (s1 - np.exp(-1j * omega) * s2) * np.exp(-1j * omega * (n - 1))


def declare_fft(func, tolerance=None):
    return dp.transform(inverse=np.fft.ifft, parseval=True, real_input_hermitian=True, tolerance=tolerance)(func)


FFT_REALISTIC = {
    "fft_radix2_recursive_twiddles": radix2_recursive_twiddles,
    "fft_radix2_single_twiddles": radix2_single_twiddles,
    "fft_naive_dft_unreduced": naive_dft_unreduced,
    "fft_goertzel": goertzel,
}
FFT_REFERENCES = {"numpy_fft": np.fft.fft, "radix2_accurate_twiddles": radix2_reference,
                  "dft_accurate_twiddles": dft_with_rounded_twiddles(0)}


# --- IIR -------------------------------------------------------------------

B, A = iir_mutants.B, iir_mutants.A


def lfilter_with_rounded_coefficients(bits):
    """``lfilter`` on coefficients that keep 53 - bits bits (the declared ones are exact)."""
    b, a = round_bits(B, bits), round_bits(A, bits)

    def filt(x):
        return ss.lfilter(b, a, x)

    filt.__name__ = f"lfilter_coefficients_minus_{bits}_bits"
    return filt


def lfilter_in_narrow_format(bits):
    """``lfilter`` on an input rounded to 53 - bits bits, output rounded likewise."""

    def filt(x):
        return round_bits(ss.lfilter(B, A, round_bits(x, bits)), bits)

    filt.__name__ = f"lfilter_arithmetic_minus_{bits}_bits"
    return filt


def lfilter_single_coefficients(x):
    """Coefficients stored in single precision, recursion in double."""
    return ss.lfilter(B.astype(np.float32).astype(float), A.astype(np.float32).astype(float), x)


def declare_iir(func, tolerance=None):
    return dp.lti_filter(causal=True, transfer=(B, A), tolerance=tolerance)(func)


# --- resampler -------------------------------------------------------------

UP, DOWN = multirate_mutants.UP, multirate_mutants.DOWN
_TAPS = multirate_mutants._design(UP)  # the window resample_poly designs, gain included


def resample_with_rounded_taps(bits):
    taps = round_bits(_TAPS, bits)

    def resample(x):
        return ss.resample_poly(x, UP, DOWN, window=taps)

    resample.__name__ = f"resample_taps_minus_{bits}_bits"
    return resample


def resample_in_narrow_format(bits):
    """``resample_poly`` on an input rounded to 53 - bits bits, output rounded likewise."""

    def resample(x):
        return round_bits(ss.resample_poly(round_bits(x, bits), UP, DOWN), bits)

    resample.__name__ = f"resample_arithmetic_minus_{bits}_bits"
    return resample


def declare_resampler(func, tolerance=None):
    return dp.resampler(UP, DOWN, tolerance=tolerance)(func)


# --- streaming -------------------------------------------------------------


def moving_sum_with_rounded_state(bits):
    """The moving sum, its two-sample history stored with 53 - bits bits between blocks."""

    class RoundedStateMovingSum(streaming_mutants.MovingSum):
        def process(self, block):
            out = super().process(block)
            self.hist = round_bits(self.hist, bits)
            return out

    RoundedStateMovingSum.__name__ = f"MovingSumStateMinus{bits}Bits"
    return RoundedStateMovingSum


def declare_streaming(cls, tolerance=None):
    return dp.streaming(tolerance=tolerance)(cls)


# --- graded families, by operator class --------------------------------------

# Each family: bits -> the undeclared mutant, and the declaration of its class
# (which takes an optional tolerance model). At 0 bits the mutant is the
# trusted implementation (numpy.fft.fft, lfilter, resample_poly, the moving sum).
FAMILIES = {
    "fft_twiddles": (dft_with_rounded_twiddles, declare_fft),
    "iir_coefficients": (lfilter_with_rounded_coefficients, declare_iir),
    "iir_arithmetic": (lfilter_in_narrow_format, declare_iir),
    "resampler_taps": (resample_with_rounded_taps, declare_resampler),
    "resampler_arithmetic": (resample_in_narrow_format, declare_resampler),
    "streaming_state": (moving_sum_with_rounded_state, declare_streaming),
}

GRADED = {name: (lambda make, declare: lambda bits, tolerance=None: declare(make(bits), tolerance))(make, declare)
          for name, (make, declare) in FAMILIES.items()}

REALISTIC = {
    **{name: declare_fft(f) for name, f in FFT_REALISTIC.items()},
    "iir_single_coefficients": declare_iir(lfilter_single_coefficients),
}


# --- extended-precision references (long double), for calibration -------------


def longdouble_resample(x):
    """``resample_poly(x, UP, DOWN)`` computed in long double (same filter, same alignment)."""
    x = np.asarray(x)
    n_out = -(-len(x) * UP // DOWN)
    if len(x) == 0:
        return np.zeros(0)
    h = (_TAPS * UP).astype(np.longdouble)
    xu = np.zeros(len(x) * UP, np.longdouble)
    xu[::UP] = x
    full = np.concatenate([np.convolve(xu, h), np.zeros(len(h) + DOWN, np.longdouble)])
    half_len = (len(_TAPS) - 1) // 2
    return full[half_len::DOWN][:n_out]


def longdouble_moving_sum(x):
    """The moving sum of streaming_mutants.MovingSum, in long double."""
    buf = np.concatenate([np.zeros(2, np.longdouble), np.asarray(x, np.longdouble)])
    return buf[2:] + buf[1:-1] + buf[:-2]

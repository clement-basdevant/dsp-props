"""Double-double DFT: a ~32-digit reference that needs nothing but float64.

``np.longdouble`` is not an extended type everywhere: it is float64 under
Windows (MSVC) and on macOS arm64, and NumPy before 2.0 computes the FFT of a
``clongdouble`` array in complex128 anyway. This module gives calibration a
reference that does not depend on the platform: a direct O(N^2) DFT in
double-double arithmetic (each value is an unevaluated sum hi + lo of two
float64, Dekker 1971; Hida, Li and Bailey 2001), vectorised with NumPy.

The twiddle factors are computed in double-double too: exact integer
reduction of the angle 2*pi*k/N to [0, pi/4], then Taylor series. The
result carries about 1e-30 relative error, far below the float64 rounding
being measured.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

_SPLIT = 134217729.0  # 2**27 + 1
_PI_2 = (1.5707963267948966, 6.123233995736766e-17)  # pi/2 as hi + lo
_TAYLOR_TERMS = 15  # (pi/4)**31 / 31! < 1e-34
_BLOCK = 256  # columns of the N x N product matrix summed at once


class DoubleDouble(NamedTuple):
    """An array value hi + lo, with |lo| <= ulp(hi) / 2."""

    hi: np.ndarray
    lo: np.ndarray


def _two_sum(a, b):
    s = a + b
    bb = s - a
    return s, (a - (s - bb)) + (b - bb)


def _quick_two_sum(a, b):
    s = a + b
    return s, b - (s - a)


def _split(a):
    c = _SPLIT * a
    hi = c - (c - a)
    return hi, a - hi


def _two_prod(a, b):
    p = a * b
    ah, al = _split(a)
    bh, bl = _split(b)
    return p, ((ah * bh - p) + ah * bl + al * bh) + al * bl


def _add(ah, al, bh, bl):
    s, e = _two_sum(ah, bh)
    t, f = _two_sum(al, bl)
    s, e = _quick_two_sum(s, e + t)
    return _quick_two_sum(s, e + f)


def _mul(ah, al, bh, bl):
    p, e = _two_prod(ah, bh)
    return _quick_two_sum(p, e + (ah * bl + al * bh))


def _mul_d(ah, al, b):
    p, e = _two_prod(ah, b)
    return _quick_two_sum(p, e + al * b)


def _div_d(ah, al, d):
    """(ah + al) / d for a float64 d (here a small integer)."""
    q1 = ah / d
    p, e = _two_prod(q1, d)
    r = ((ah - p) - e) + al
    return _quick_two_sum(q1, r / d)


def _one_minus(h, l):
    return _add(np.ones_like(h), np.zeros_like(h), -h, -l)


def _sin_cos(ph, pl):
    """sin and cos of a double-double angle in [0, pi/4], by Horner on the Taylor series."""
    x2h, x2l = _mul(ph, pl, ph, pl)
    sh, sl = np.ones_like(ph), np.zeros_like(ph)
    ch, cl = np.ones_like(ph), np.zeros_like(ph)
    for k in range(_TAYLOR_TERMS, 0, -1):
        # sin: 1 - x^2 t / ((2k)(2k+1)); cos: 1 - x^2 t / ((2k-1)(2k))
        th, tl = _div_d(*_mul(x2h, x2l, sh, sl), float(2 * k * (2 * k + 1)))
        sh, sl = _one_minus(th, tl)
        th, tl = _div_d(*_mul(x2h, x2l, ch, cl), float((2 * k - 1) * 2 * k))
        ch, cl = _one_minus(th, tl)
    sh, sl = _mul(ph, pl, sh, sl)
    return (sh, sl), (ch, cl)


def twiddles(n):
    """exp(-2j*pi*r/n) for r = 0..n-1, as double-double (re_hi, re_lo, im_hi, im_lo)."""
    r = np.arange(n, dtype=np.int64)
    quadrant, rem = np.divmod(4 * r, n)  # angle = (pi/2) * (quadrant + rem/n)
    upper = 2 * rem > n  # reduce to [0, pi/4]: use the complementary angle, swap sin and cos
    rem = np.where(upper, n - rem, rem).astype(float)
    fh, fl = _div_d(rem, np.zeros(n), float(n))
    (sh, sl), (ch, cl) = _sin_cos(*_mul(np.full(n, _PI_2[0]), np.full(n, _PI_2[1]), fh, fl))
    ch, cl, sh, sl = (np.where(upper, sh, ch), np.where(upper, sl, cl),
                      np.where(upper, ch, sh), np.where(upper, cl, sl))
    # cos and sin of the full angle, by quadrant
    q = quadrant
    cos_h = np.select([q == 0, q == 1, q == 2], [ch, -sh, -ch], sh)
    cos_l = np.select([q == 0, q == 1, q == 2], [cl, -sl, -cl], sl)
    sin_h = np.select([q == 0, q == 1, q == 2], [sh, ch, -sh], -ch)
    sin_l = np.select([q == 0, q == 1, q == 2], [sl, cl, -sl], -cl)
    return cos_h, cos_l, -sin_h, -sin_l


def _sum_rows(h, l):
    """Pairwise double-double sum along axis 1."""
    while h.shape[1] > 1:
        if h.shape[1] % 2:
            h = np.concatenate([h, np.zeros_like(h[:, :1])], axis=1)
            l = np.concatenate([l, np.zeros_like(l[:, :1])], axis=1)
        h, l = _add(h[:, ::2], l[:, ::2], h[:, 1::2], l[:, 1::2])
    return h[:, 0], l[:, 0]


def doubledouble_dft(x):
    """DFT of ``x`` in double-double arithmetic, returned as ``DoubleDouble`` complex arrays.

    The input is converted to complex128 (exact for float16/32/64 and complex64/128;
    a long double input is rounded to float64). O(N^2) time, O(N * 256) memory.
    """
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError("doubledouble_dft expects a 1-D signal")
    n = x.size
    if n == 0:
        z = np.zeros(0, dtype=np.complex128)
        return DoubleDouble(z, z.copy())
    x = x.astype(np.complex128)
    xr, xi = x.real.copy(), x.imag.copy()
    wrh, wrl, wih, wil = twiddles(n)
    k = np.arange(n, dtype=np.int64)
    acc = [np.zeros(n) for _ in range(4)]  # re_hi, re_lo, im_hi, im_lo
    for start in range(0, n, _BLOCK):
        t = np.arange(start, min(start + _BLOCK, n), dtype=np.int64)
        idx = np.outer(k, t) % n
        a, b = xr[t], xi[t]
        # (a + ib)(wr + i wi) = (a wr - b wi) + i (a wi + b wr)
        rh, rl = _add(*_mul_d(wrh[idx], wrl[idx], a), *_mul_d(-wih[idx], -wil[idx], b))
        ih, il = _add(*_mul_d(wih[idx], wil[idx], a), *_mul_d(wrh[idx], wrl[idx], b))
        acc[0], acc[1] = _add(acc[0], acc[1], *_sum_rows(rh, rl))
        acc[2], acc[3] = _add(acc[2], acc[3], *_sum_rows(ih, il))
    return DoubleDouble(acc[0] + 1j * acc[2], acc[1] + 1j * acc[3])

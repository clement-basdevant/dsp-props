"""The precision mutants compute the right thing, less accurately (benchmarks/precision_mutants.py)."""

import pathlib
import sys

import numpy as np
import pytest
import scipy.signal as ss

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from mutants import precision_mutants as pm  # noqa: E402
from mutants.streaming_mutants import MovingSum  # noqa: E402

EPS = np.finfo(float).eps


def _rel(a, b):
    return np.linalg.norm(np.asarray(a) - b) / np.linalg.norm(b)


def test_round_bits_keeps_the_requested_significand():
    x = np.random.default_rng(0).standard_normal(1000) * 10.0 ** np.linspace(-100, 100, 1000)
    assert np.array_equal(pm.round_bits(x, 0), x)
    for bits in (1, 10, 29):
        err = np.abs(pm.round_bits(x, bits) - x) / np.abs(x)
        assert err.max() <= 2.0 ** bits * EPS / 2 * (1 + 1e-12)
        assert err.max() > 2.0 ** bits * EPS / 8  # the rounding is really that coarse
    assert pm.round_bits(np.array([1.0 + 1e-12j]), 29).imag[0] != 0


@pytest.mark.parametrize("n", [1, 2, 3, 7, 16, 100, 128])
def test_fft_mutants_compute_the_dft(n):
    x = np.random.default_rng(n).standard_normal(n) + 1j * np.random.default_rng(n + 1).standard_normal(n)
    ref = np.fft.fft(x)
    for f in [*pm.FFT_REFERENCES.values(), *pm.FFT_REALISTIC.values(), pm.dft_with_rounded_twiddles(10)]:
        if f is pm.radix2_single_twiddles:
            assert _rel(f(x), ref) < 1e-6
        else:
            assert _rel(f(x), ref) < 1e-9, f.__name__


def test_graded_mutants_are_exact_at_zero_bits():
    x = np.random.default_rng(0).standard_normal(64)
    assert np.allclose(pm.lfilter_with_rounded_coefficients(0)(x), ss.lfilter(pm.B, pm.A, x), rtol=0, atol=0)
    assert np.array_equal(pm.lfilter_in_narrow_format(0)(x), ss.lfilter(pm.B, pm.A, x))
    assert np.array_equal(pm.resample_with_rounded_taps(0)(x), ss.resample_poly(x, pm.UP, pm.DOWN))
    assert np.array_equal(pm.resample_in_narrow_format(0)(x), ss.resample_poly(x, pm.UP, pm.DOWN))
    proc, ref = pm.moving_sum_with_rounded_state(0)(), MovingSum()
    for block in np.array_split(x, 7):
        assert np.array_equal(proc.process(block), ref.process(block))


def test_rounded_state_only_matters_across_blocks():
    x = np.random.default_rng(1).standard_normal(64)
    whole = pm.moving_sum_with_rounded_state(20)().process(x)
    assert np.array_equal(whole, MovingSum().process(x))
    proc = pm.moving_sum_with_rounded_state(20)()
    split = np.concatenate([proc.process(b) for b in np.array_split(x, 16)])
    assert 0 < _rel(split, whole) < 1e-9


@pytest.mark.parametrize("n", [0, 1, 2, 3, 7, 50, 511])
def test_long_double_references_match_the_double_implementations(n):
    x = np.random.default_rng(n).standard_normal(n)
    resampled = pm.longdouble_resample(x)
    assert resampled.shape == (-(-n * pm.UP // pm.DOWN),)
    if n:
        assert np.max(np.abs(resampled - ss.resample_poly(x, pm.UP, pm.DOWN))) < 1e-14
    assert np.max(np.abs(pm.longdouble_moving_sum(x) - MovingSum().process(x)), initial=0) < 1e-14

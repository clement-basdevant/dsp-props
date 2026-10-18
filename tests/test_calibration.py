import numpy as np
import pytest

import dsp_props as dp
from dsp_props.core import CalibratedTolerance, CalibrationError, longdouble_dft, norm, relative_error


def test_numpy_fft_is_within_a_few_eps():
    tol = dp.calibrate(np.fft.fft, lengths=range(1, 65), trials=2)
    assert tol.worst < 50


def test_calibration_refuses_a_wrong_operator():
    with pytest.raises(CalibrationError):
        dp.calibrate(lambda x: np.roll(np.fft.fft(x), 1), lengths=range(2, 10), trials=1)


def test_calibrated_model_never_below_analytic_base():
    tol = CalibratedTolerance(ratios={8: 0.1})
    assert tol.bound(8, np.float64, 1.0) >= tol.base.bound(8, np.float64, 1.0)


def test_unmeasured_length_uses_worst_ratio():
    tol = CalibratedTolerance(ratios={8: 3.0, 9: 500.0})
    assert tol.ratio(1000) == 500.0


def test_relative_error_of_exact_operator_is_zero():
    x = np.zeros(16)
    assert relative_error(np.fft.fft, longdouble_dft, x) == 0.0


def test_calibrated_tolerance_still_catches_mutants():
    from mutants.fft_mutants import MUTANTS

    tol = dp.calibrate(np.fft.fft, lengths=range(1, 129), trials=2)
    m = MUTANTS["fft_float32_internal"]
    spec = dp.transform(inverse=np.fft.ifft, parseval=True, real_input_hermitian=True, tolerance=tol)(
        m.__wrapped__ if hasattr(m, "__wrapped__") else m.__dsp_props__.func
    )
    assert not dp.verify(spec, max_examples=100, raise_on_failure=False).ok


# --------------------------------------------------------------- reference without extended long double

from dsp_props.core import calibration as calmod
from dsp_props.core import DoubleDouble, doubledouble_dft, extended_dft, longdouble_fft_is_extended

needs_extended_longdouble = pytest.mark.skipif(
    not longdouble_fft_is_extended(), reason="long double FFT is not extended on this platform"
)


@pytest.fixture
def no_extended_longdouble(monkeypatch):
    """Simulate Windows / macOS arm64 / NumPy 1.x: the long double FFT is plain complex128."""
    monkeypatch.setattr(calmod, "longdouble_dft",
                        lambda x: np.fft.fft(np.asarray(x, dtype=np.complex128)).astype(np.clongdouble))
    calmod.longdouble_fft_is_extended.cache_clear()
    yield
    calmod.longdouble_fft_is_extended.cache_clear()


def test_doubledouble_dft_matches_mpmath():
    mpmath = pytest.importorskip("mpmath")
    mpmath.mp.dps = 50
    rng = np.random.default_rng(3)
    for n in [1, 2, 3, 8, 17, 97, 128]:
        x = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        got = doubledouble_dft(x)
        xs = [mpmath.mpc(complex(v)) for v in x]
        ref = [mpmath.fsum(xs[t] * mpmath.expjpi(-2 * mpmath.mpf(k * t % n) / n) for t in range(n))
               for k in range(n)]
        err = mpmath.sqrt(mpmath.fsum(
            abs(mpmath.mpf(h.real) + mpmath.mpf(l.real) + 1j * (mpmath.mpf(h.imag) + mpmath.mpf(l.imag)) - r) ** 2
            for h, l, r in zip(got.hi, got.lo, ref)))
        assert err / mpmath.sqrt(mpmath.fsum(abs(r) ** 2 for r in ref)) < 1e-30


def test_doubledouble_dft_of_an_impulse_is_exact():
    x = np.zeros(13)
    x[0] = 2.0
    got = doubledouble_dft(x)
    assert np.array_equal(got.hi, np.full(13, 2.0 + 0j)) and not np.any(got.lo)


@needs_extended_longdouble
def test_doubledouble_agrees_with_longdouble():
    x = np.random.default_rng(4).standard_normal(997)
    dd, ld = doubledouble_dft(x), longdouble_dft(x)
    err = norm(((ld - dd.hi) - dd.lo).astype(np.complex128)) / norm(dd.hi)
    assert err < 1e-2 * np.finfo(float).eps


def test_complex128_longdouble_fft_is_detected(no_extended_longdouble):
    assert not calmod.longdouble_fft_is_extended()
    assert isinstance(extended_dft(np.ones(8)), DoubleDouble)


def test_fallback_still_measures_numpy_and_refuses_a_wrong_operator(no_extended_longdouble):
    tol = dp.calibrate(np.fft.fft, lengths=range(1, 65), trials=2)
    assert 0.1 < tol.worst < 50  # a complex128 reference would measure 0 at most lengths
    with pytest.raises(CalibrationError):
        dp.calibrate(lambda x: np.roll(np.fft.fft(x), 1), lengths=range(2, 10), trials=1)


@needs_extended_longdouble
def test_fallback_measures_the_same_error_as_longdouble():
    lengths = [2, 8, 17, 64, 97, 128, 255]
    ld = dp.calibrate(np.fft.fft, reference=longdouble_dft, lengths=lengths, trials=3)
    dd = dp.calibrate(np.fft.fft, reference=doubledouble_dft, lengths=lengths, trials=3)
    for n in lengths:
        assert 0.5 <= dd.ratios[n] / ld.ratios[n] <= 2, (n, dd.ratios[n], ld.ratios[n])

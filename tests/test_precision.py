"""The expected precision comes from the input (or the declaration), not from the output."""

import importlib

import numpy as np
import pytest
import scipy.signal as ss

import dsp_props as dp
from dsp_props.api import hermitian_extend
from dsp_props.core import FFTTolerance, ToleranceError, assert_close

signals_module = importlib.import_module("dsp_props.generators.signals")  # the package re-exports a function of that name

BOUNDED = [1.0, 1e-3, 1e3]  # amplitudes complex64 can hold: the overflow cannot give the bug away


@pytest.fixture
def bounded_amplitudes(monkeypatch):
    monkeypatch.setattr(signals_module, "AMPLITUDES", BOUNDED)


def test_fft_that_silently_returns_complex64_is_caught(bounded_amplitudes):
    # No Parseval relation: the other relations used to compare complex64
    # outputs with a single-precision tolerance and passed.
    @dp.transform(inverse=np.fft.ifft, real_input_hermitian=True)
    def degraded(x):
        return np.fft.fft(x).astype(np.complex64)

    report = dp.verify(degraded, max_examples=50, raise_on_failure=False)
    assert len(report.failures) == len(report.checked)
    for failure in report.failures:
        assert "complex64" in str(failure.error) and "float64" in str(failure.error)


def test_filter_that_silently_returns_float32_is_caught(bounded_amplitudes):
    b, a = ss.butter(2, 0.2)

    @dp.lti_filter(transfer=(b, a))
    def degraded(x):
        return ss.lfilter(b, a, x).astype(np.float32)

    report = dp.verify(degraded, max_examples=50, raise_on_failure=False)
    assert len(report.failures) == len(report.checked)
    for failure in report.failures:
        assert "float32" in str(failure.error) and "float64" in str(failure.error)


class _FIRToFloat32:
    """Exact block FIR in float64 whose output is silently cast to float32."""

    h = np.array([0.5, 0.3, 0.2])

    def __init__(self):
        self.tail = np.zeros(len(self.h) - 1)

    def process(self, x):
        buf = np.concatenate([self.tail, x])
        self.tail = buf[len(buf) - len(self.tail):]
        return np.convolve(buf, self.h)[len(self.tail): len(self.tail) + len(x)].astype(np.float32)


def test_streaming_processor_that_returns_float32_is_caught(bounded_amplitudes):
    report = dp.verify(dp.streaming()(_FIRToFloat32), max_examples=50, raise_on_failure=False)
    assert not report.ok
    assert "float32" in str(report.failures[0].error) and "float64" in str(report.failures[0].error)


def test_assert_close_names_both_dtypes():
    x = np.ones(8, dtype=np.complex64)
    with pytest.raises(ToleranceError, match="complex64.*float64"):
        assert_close(x, x.astype(np.complex128), FFTTolerance(), precision=np.float64)
    # without a declared precision the output's own epsilon is used, as before
    assert_close(x, x.astype(np.complex128), FFTTolerance())


def test_more_precise_output_is_accepted():
    x = np.ones(8, dtype=np.float32)
    assert_close(x.astype(np.float64), x, FFTTolerance(), precision=np.float32)


def test_float32_transform_declared_as_such_passes():
    @dp.transform(inverse=np.fft.ifft, parseval=True, real_input_hermitian=True, precision=np.float32)
    def fft32(x):
        assert x.dtype in (np.float32, np.complex64)
        return np.fft.fft(x).astype(np.complex64)

    assert dp.verify(fft32, max_examples=100).ok


def test_float32_rfft_declared_as_such_passes():
    @dp.transform(inverse=lambda X, n: np.fft.irfft(X, n).astype(np.float32), kind="rdft",
                  parseval=True, precision=np.float32)
    def rfft32(x):
        return np.fft.rfft(x).astype(np.complex64)

    assert dp.verify(rfft32, max_examples=100).ok


def test_float32_filter_declared_as_such_passes():
    b, a = ss.butter(2, 0.2)
    b32, a32 = b.astype(np.float32), a.astype(np.float32)

    @dp.lti_filter(transfer=(b, a), max_gain=float(np.sum(np.abs(ss.lfilter(b, a, np.eye(1, 512)[0])))),
                   precision=np.float32)
    def lfilter32(x):
        assert x.dtype == np.float32
        return ss.lfilter(b32, a32, x)

    assert dp.verify(lfilter32, max_examples=100).ok


def test_float32_resampler_declared_as_such_passes():
    @dp.resampler(3, 2, precision=np.float32)
    def resample32(x):
        assert x.dtype == np.float32
        return ss.resample_poly(x, 3, 2).astype(np.float32)

    assert dp.verify(resample32, max_examples=50).ok


class _FIR32(_FIRToFloat32):
    """The same FIR computed in single precision throughout."""

    h = _FIRToFloat32.h.astype(np.float32)

    def __init__(self):
        self.tail = np.zeros(len(self.h) - 1, np.float32)

    def process(self, x):
        assert x.dtype == np.float32
        buf = np.concatenate([self.tail, x])
        self.tail = buf[len(buf) - len(self.tail):]
        return np.convolve(buf, self.h)[len(self.tail): len(self.tail) + len(x)]


def test_float32_streaming_declared_as_such_passes():
    # (lfilter with zi is not used here: it fails on empty blocks, scipy/scipy#22571)
    assert dp.verify(dp.streaming(precision=np.float32)(_FIR32), max_examples=100).ok


def test_precision_must_be_a_floating_dtype():
    with pytest.raises(ValueError):
        dp.transform(precision=np.int32)


def test_hermitian_extend_keeps_the_precision_of_its_input():
    assert hermitian_extend(np.fft.rfft(np.ones(8)), 8).dtype == np.complex128
    assert hermitian_extend(np.fft.rfft(np.ones(8)).astype(np.complex64), 8).dtype == np.complex64
    assert hermitian_extend(np.ones(5), 8).dtype == np.complex128


def test_float64_rfft_is_not_flagged_by_the_hermitian_extension():
    f = dp.transform(inverse=np.fft.irfft, kind="rdft", parseval=True)(np.fft.rfft)
    assert dp.verify(f, max_examples=100).ok

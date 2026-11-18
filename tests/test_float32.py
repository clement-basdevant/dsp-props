"""The float32 consistency relation (plan, action 2.3)."""

import numpy as np
import pytest
import scipy.fft as sfft
from scipy import signal

import dsp_props as dp

B, A = signal.butter(4, 0.2)


def _correct():
    return {
        "numpy fft": dp.transform(inverse=np.fft.ifft, parseval=True, float32=True)(lambda x: np.fft.fft(x)),
        "scipy rfft": dp.transform(kind="rdft", inverse=sfft.irfft, float32=True)(lambda x: sfft.rfft(x)),
        "dct": dp.transform(kind="generic", float32=True)(lambda x: sfft.dct(x, norm="ortho")),
        "lfilter": dp.lti_filter(transfer=(B, A), float32=True)(lambda x: signal.lfilter(B, A, x)),
        "sosfilt": dp.lti_filter(transfer=(B, A), float32=True)(lambda x: signal.sosfilt(signal.tf2sos(B, A), x)),
        "resample_poly": dp.resampler(2, 3, float32=True)(lambda x: signal.resample_poly(x, 2, 3)),
        "streamed lfilter": dp.stateful(lambda block, z: signal.lfilter(B, A, block, zi=z),
                                        lambda: np.zeros(4), float32=True),
    }


@pytest.mark.parametrize("name", list(_correct()))
def test_correct_operators_pass(name):
    op = _correct()[name]
    for seed in range(3):
        report = dp.verify(op, relations="float32 consistency", seed=seed, raise_on_failure=False)
        assert report.ok, report


def _wrong():
    def coefficients_in_half_precision(x):
        if x.dtype == np.float32:  # a dtype-dependent path that stores its coefficients in float16
            return signal.lfilter(B.astype(np.float16), A.astype(np.float16), x)
        return signal.lfilter(B, A, x)

    return {
        "float32 path, coefficients in float16": dp.lti_filter(transfer=(B, A), float32=True)(
            coefficients_in_half_precision),
        "float32 input rounded to float16": dp.transform(inverse=np.fft.ifft, float32=True)(
            lambda x: np.fft.fft(x.astype(np.float16) if x.dtype == np.float32 else x)),
        "float16 output": dp.transform(kind="generic", float32=True)(
            lambda x: x.astype(np.float16) if x.dtype == np.float32 else x.copy()),
        "different scale in float32": dp.transform(inverse=np.fft.ifft, float32=True)(
            lambda x: np.fft.fft(x) * (1.0001 if x.dtype == np.float32 else 1.0)),
    }


@pytest.mark.parametrize("name", list(_wrong()))
def test_float32_faults_are_caught_and_only_by_this_relation(name):
    op = _wrong()[name]
    report = dp.verify(op, seed=0, raise_on_failure=False)
    assert [f.relation for f in report.failures] == ["float32 consistency"]


def test_only_declared_operators_get_the_relation():
    plain = dp.transform(inverse=np.fft.ifft)(lambda x: np.fft.fft(x))
    assert "float32 consistency" not in [r.name for r in plain.__dsp_props__.relations]
    single = dp.transform(inverse=np.fft.ifft, precision=np.float32, float32=True)(lambda x: np.fft.fft(x))
    assert "float32 consistency" not in [r.name for r in single.__dsp_props__.relations]
    both = _correct()["lfilter"]
    assert [r.name for r in both.__dsp_props__.relations][-1] == "float32 consistency"

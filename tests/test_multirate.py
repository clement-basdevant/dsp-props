import numpy as np
import pytest
from scipy import signal

import dsp_props as dp


@pytest.mark.parametrize("up,down", [(1, 2), (3, 1), (2, 3), (5, 4), (1, 4)])
def test_resample_poly_passes(up, down):
    f = dp.resampler(up, down)(lambda x: signal.resample_poly(x, up, down))
    dp.verify(f, max_examples=60)


@pytest.mark.full_examples
def test_plain_decimation_aliases():
    f = dp.resampler(1, 2)(lambda x: np.asarray(x)[::2])
    report = dp.verify(f, max_examples=60, raise_on_failure=False)
    assert [fl.relation for fl in report.failures] == ["alias rejection"]


def test_alias_rejection_only_when_decimating():
    f = dp.resampler(3, 1)(lambda x: signal.resample_poly(x, 3, 1))
    assert "alias rejection" not in [r.name for r in f.__dsp_props__.relations]


def test_length_rule_can_be_disabled():
    f = dp.resampler(2, 3, length=None)(lambda x: signal.resample_poly(x, 2, 3)[:-1] if len(x) else np.zeros(0))
    assert "output length" not in [r.name for r in f.__dsp_props__.relations]


@pytest.mark.full_examples
def test_bounded_gain_accepts_stable_filter_and_rejects_resonance():
    b, a = [0.2, 0.2], [1.0, -0.5]
    l1 = np.sum(np.abs(signal.lfilter(b, a, np.r_[1.0, np.zeros(200)])))
    stable = dp.lti_filter(max_gain=l1)(lambda x: signal.lfilter(b, a, x))
    dp.verify(stable, max_examples=60)
    louder = dp.lti_filter(max_gain=0.9 * l1)(lambda x: signal.lfilter(b, a, x))
    report = dp.verify(louder, max_examples=60, raise_on_failure=False)
    assert "bounded gain" in [fl.relation for fl in report.failures]


def test_time_invariance_for_non_causal_filter():
    # centred 3-tap smoother, output aligned with the input (np.convolve "same" grows short inputs)
    f = dp.lti_filter(causal=False)(lambda x: np.convolve(x, [0.25, 0.5, 0.25])[1 : 1 + len(x)])
    dp.verify(f, max_examples=60)

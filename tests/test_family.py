"""Operator families: parameters drawn with the signals (plan, action 2.4)."""

import numpy as np
import pytest
from hypothesis import given, settings
from scipy import signal

import dsp_props as dp
from dsp_props.cli import discover
from dsp_props.core.iir import pole_radius
from dsp_props.generators import fir_taps, iir_coefficients, rate_pairs
from dsp_props.regression import replay, to_source

pytest_plugins = ["pytester"]


@settings(max_examples=200)
@given(iir_coefficients(max_order=6))
def test_drawn_filters_are_stable_and_varied_in_shape(ba):
    b, a = ba
    assert a[0] != 0 and len(b) >= 1 and np.any(b)
    assert len(a) == 1 or pole_radius(a) < 1


@given(rate_pairs(8))
def test_rate_pairs_are_coprime(pair):
    up, down = pair
    assert 1 <= up <= 8 and 1 <= down <= 8 and np.gcd(up, down) == 1


def _lfilter_family(func):
    @dp.family(iir_coefficients(max_order=4))
    def lfilter(b, a):
        return dp.lti_filter(transfer=(b, a))(lambda x: func(b, a, x))

    return lfilter


def test_correct_lfilter_and_resample_poly_pass():
    assert dp.verify(_lfilter_family(signal.lfilter), seed=0, profile="fast").ok

    @dp.family(rate_pairs(6))
    def resample(up, down):
        return dp.resampler(up, down)(lambda x: signal.resample_poly(x, up, down))

    names = [r.name for r in resample.__dsp_props__.relations]
    assert "alias rejection" in names and "DC gain" in names  # found on some member, not on (1, 1)
    assert dp.verify(resample, seed=0, profile="fast").ok


def _forgets_a0_without_poles(b, a, x):
    if len(a) == 1:  # an order-0 path that does not divide by a[0]
        return signal.lfilter(b, [1.0], x)
    return signal.lfilter(b, a, x)


def _short_numerator_padding_bug(b, a, x):
    if len(b) < len(a):  # pads b at the front instead of the back
        b = np.concatenate([np.zeros(len(a) - len(b)), b])
    return signal.lfilter(b, a, x)


@pytest.mark.parametrize("func", [_forgets_a0_without_poles, _short_numerator_padding_bug])
def test_faults_on_untested_parameters_are_found_with_small_parameters(func):
    fixed = dp.lti_filter(transfer=signal.butter(4, 0.2))(lambda x: func(*signal.butter(4, 0.2), x))
    assert dp.verify(fixed, seed=0, profile="fast").ok  # a fixed order-4 declaration misses it
    report = dp.verify(_lfilter_family(func), seed=0, profile="fast", raise_on_failure=False)
    assert not report.ok
    (b, a), _ = report.failures[0].example
    assert len(a) + len(b) <= 4  # shrunk to a tiny filter


def test_counterexample_replays_and_writes_as_a_regression_test(tmp_path):
    fam = _lfilter_family(_forgets_a0_without_poles)
    report = dp.verify(fam, seed=0, profile="fast", raise_on_failure=False)
    f = report.failures[0]
    with pytest.raises(AssertionError):
        replay(fam, f.relation, f.example)
    assert "np.array" in to_source(f.example)


def test_family_misuse_is_reported():
    @dp.family(rate_pairs())
    def not_declared(up, down):
        return lambda x: x

    with pytest.raises(TypeError, match="declared operator"):
        _ = not_declared.__dsp_props__.relations


def test_families_are_collected_by_pytest_and_the_cli(pytester):
    pytester.makepyfile(test_fam='''
from scipy import signal
import dsp_props as dp

@dp.family(dp.generators.iir_coefficients(max_order=3))
def lfilter(b, a):
    return dp.lti_filter(transfer=(b, a))(lambda x: signal.lfilter(b, a, x))
''')
    result = pytester.runpytest("--dsp-props-profile", "fast", "--dsp-props-database", "none",
                                "-p", "no:cacheprovider", "-v")
    result.assert_outcomes(passed=4)
    assert "test_fam.py::lfilter::transfer-function PASSED" in result.stdout.str()
    assert [n.split(":")[1] for n, _ in discover(str(pytester.path / "test_fam.py"))] == ["lfilter"]


def test_fir_taps_are_never_all_zero():
    @given(fir_taps(5))
    def check(b):
        assert np.any(b) and 1 <= len(b) <= 5

    check()

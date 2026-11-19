"""Declared input-length range (min_n) and slow operators in families (rediscovery campaign, plan 3.1)."""

import time

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import signal

import dsp_props as dp
from dsp_props.generators import lengths

B, A = signal.butter(2, 0.3)


@given(lengths(64, 10))
@settings(max_examples=50)
def test_lengths_respect_min_n(n):
    assert 10 <= n <= 64


def test_lengths_without_pathological_value_in_range():
    assert lengths(40, 34).example() in range(34, 41)


def test_filtfilt_needs_min_n():
    """filtfilt is documented to need more samples than its padding (3 max(len(a), len(b)))."""
    padlen = 3 * max(len(A), len(B))

    def zero_phase(x):
        return signal.filtfilt(B, A, x)

    short = dp.transform(kind="generic")(zero_phase)
    assert not dp.verify(short, max_examples=30, seed=0, raise_on_failure=False).ok  # ValueError on N <= padlen
    declared = dp.transform(kind="generic", min_n=padlen + 1)(zero_phase)
    assert dp.verify(declared, max_examples=30, seed=0, raise_on_failure=False).ok


@pytest.mark.parametrize("decorator", [
    lambda m: dp.lti_filter(min_n=m), lambda m: dp.resampler(1, 2, min_n=m),
    lambda m: dp.streaming(min_n=m), lambda m: dp.transform(kind="generic", min_n=m)])
def test_min_n_is_checked(decorator):
    with pytest.raises(ValueError, match="min_n"):
        decorator(0)
    with pytest.raises(ValueError, match="min_n"):
        decorator(600)  # above the default max_n = 512


def test_every_relation_draws_at_least_min_n():
    seen = []

    @dp.lti_filter(transfer=(B, A), min_n=20)
    def lowpass(x):
        seen.append(len(x))
        return signal.lfilter(B, A, x)

    assert dp.verify(lowpass, max_examples=20, seed=0, raise_on_failure=False).ok
    assert min(seen) >= 20


def test_slow_member_is_not_a_flaky_family():
    """Searching which relations apply to a family must not apply Hypothesis' 200 ms deadline.

    Building a member can be slow the first time (imports, a tolerance model, a JIT): with the
    deadline, the first build failed and the second did not, and verify raised FlakyFailure.
    """
    built = []

    @dp.family(st.sampled_from([0.0, 0.5, 0.9]))
    def slow(c):
        if not built:
            time.sleep(0.3)
        built.append(c)
        return dp.lti_filter(transfer=([1.0, -c], [1.0]))(lambda x: signal.lfilter([1.0, -c], [1.0], x))

    assert dp.verify(slow, max_examples=10, seed=0, raise_on_failure=False).ok

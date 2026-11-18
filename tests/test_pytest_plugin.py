"""The pytest plugin, cost profiles and the example database."""

import numpy as np
import pytest

import dsp_props as dp

pytest_plugins = ["pytester"]

OPERATORS = '''
import numpy as np
import dsp_props as dp

@dp.transform(inverse=np.fft.ifft, parseval=True)
def good_fft(x):
    return np.fft.fft(x)

@dp.transform(inverse=lambda X: 2 * np.fft.ifft(X))
def bad_inverse(x):
    return np.fft.fft(x)

FILTERS = {
    "gain": dp.lti_filter()(lambda x: 0.5 * x),
    "delay": dp.lti_filter()(lambda x: np.concatenate([[0.0], x[:-1]])),
}

def test_plain():
    assert True
'''


def test_declared_operators_become_one_test_per_relation(pytester):
    pytester.makepyfile(test_ops=OPERATORS)
    result = pytester.runpytest("--collect-only", "-q", "-p", "no:cacheprovider")
    out = result.stdout.str()
    for node in ["test_ops.py::good_fft::linearity", "test_ops.py::good_fft::parseval",
                 "test_ops.py::bad_inverse::inversibility", "test_ops.py::FILTERS[gain]::causality",
                 "test_ops.py::FILTERS[delay]::time-invariance", "test_ops.py::test_plain"]:
        assert node in out
    assert "16 tests collected" in out


def test_failing_relation_reports_its_minimal_input(pytester):
    pytester.makepyfile(test_ops=OPERATORS)
    result = pytester.runpytest("--dsp-props-profile", "fast", "--dsp-props-database", "none",
                                "-p", "no:cacheprovider")
    result.assert_outcomes(passed=14, failed=2)
    out = result.stdout.str()
    assert "bad_inverse violates inversibility" in out
    assert "minimal input:" in out
    assert "FAILED test_ops.py::bad_inverse::inversibility" in out


def test_imported_operators_are_not_collected_unless_listed(pytester):
    pytester.makepyfile(ops=OPERATORS.replace("def test_plain", "def plain"))
    pytester.makepyfile(test_imported="from ops import good_fft, bad_inverse, FILTERS\n")
    pytester.makepyfile(test_listed="from ops import good_fft\ndsp_props_operators = [good_fft]\n")
    result = pytester.runpytest("--collect-only", "-q", "-p", "no:cacheprovider")
    out = result.stdout.str()
    assert "test_imported.py" not in out
    assert "test_listed.py::good_fft::linearity" in out


def test_plugin_can_be_disabled(pytester):
    pytester.makepyfile(test_ops=OPERATORS)
    result = pytester.runpytest("--collect-only", "-q", "-p", "no:dsp_props", "-p", "no:cacheprovider")
    assert "1 test collected" in result.stdout.str()


@pytest.mark.full_examples
def test_profiles_set_the_number_of_inputs():
    calls = []

    @dp.transform(kind="generic")
    def identity(x):
        calls.append(len(x))
        return np.array(x)

    dp.verify(identity, profile="fast", derandomize=True)
    fast = len(calls)
    calls.clear()
    dp.verify(identity, profile="default", derandomize=True)
    assert 0 < fast < len(calls)
    calls.clear()
    dp.verify(identity, profile="nightly", max_examples=5, derandomize=True)
    assert len(calls) <= 3 * 5  # an explicit max_examples wins over the profile


def test_profile_from_environment(monkeypatch):
    monkeypatch.setenv(dp.api.PROFILE_ENV, "fast")
    assert dp.get_profile().max_examples == 30
    dp.set_profile("nightly")
    try:
        assert dp.get_profile().max_examples == 2000
    finally:
        dp.set_profile(None)
    with pytest.raises(ValueError, match="unknown dsp-props profile"):
        dp.get_profile("weekly")


def test_relations_can_be_selected():
    f = dp.transform(inverse=np.fft.ifft, parseval=True)(np.fft.fft)
    assert dp.verify(f, relations=["parseval"], max_examples=5).checked == ["parseval"]
    with pytest.raises(ValueError, match="not among the relations"):
        dp.verify(f, relations=["hermitian symmetry"], max_examples=5)


@pytest.mark.full_examples
def test_example_database_replays_the_counterexample_first(tmp_path):
    # no seed: Hypothesis neither saves nor replays examples under @seed or derandomize
    @dp.transform(kind="generic")
    def nonlinear_for_some_lengths(x):  # not linear for 100 <= N < 110 only, so rarely hit by chance
        x = np.asarray(x)
        return x + (0.1 * x * np.abs(x) if 100 <= len(x) < 110 else 0.0)

    first = dp.verify(nonlinear_for_some_lengths, relations=["linearity"], max_examples=2000,
                      database=tmp_path, raise_on_failure=False)
    assert not first.ok
    again = dp.verify(nonlinear_for_some_lengths, relations=["linearity"], max_examples=2000,
                      database=tmp_path, raise_on_failure=False)
    assert not again.ok
    assert again.failures[0].examples_tried == 1


def test_database_keys_differ_between_relations_and_operators():
    from dsp_props.api import _hypothesis_test, _operator_key

    f = dp.transform(inverse=np.fft.ifft, parseval=True)(np.fft.fft)
    spec = f.__dsp_props__
    digests = set()
    for key in (_operator_key(f), b"other:operator"):
        for rel in spec.relations:
            run, _ = _hypothesis_test(rel, spec, 1, False, key=key)
            digests.add(run.hypothesis.inner_test._hypothesis_internal_add_digest)
    assert len(digests) == 2 * len(spec.relations)

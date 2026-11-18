"""Regression tests generated from minimal counterexamples (plan, action 2.1)."""

import pathlib

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

import dsp_props as dp
from dsp_props.cli import main
from dsp_props.regression import _evaluate, _same, load_operator, replay, to_source

pytest_plugins = ["pytester"]

EXAMPLE = str(pathlib.Path(__file__).resolve().parents[1] / "examples" / "filters.py")


# --- exact source ---------------------------------------------------------------------------------

DTYPES = [np.float64, np.float32, np.complex128, np.complex64, np.int64, np.bool_]


@given(hnp.arrays(st.sampled_from(DTYPES), hnp.array_shapes(min_dims=1, max_dims=2, min_side=0, max_side=6),
                  elements=None))
def test_arrays_round_trip_bit_for_bit(a):
    # elements=None draws any value of the dtype: NaN, infinities, -0.0, subnormals, extremes
    back = _evaluate(to_source(a))
    assert _same(back, a)


@given(st.floats(allow_nan=True, allow_infinity=True), st.complex_numbers(allow_nan=True),
       st.integers(-10**30, 10**30))
def test_scalars_and_containers_round_trip(x, z, k):
    with np.errstate(over="ignore"):  # float32 and complex64 of a large double are infinite
        value = ((np.float32(x), np.float64(x), x), [z, np.complex64(z)], k, None, True, (x,))
    assert _same(_evaluate(to_source(value)), value)


def test_constant_arrays_are_written_compactly():
    assert to_source(np.zeros(300)) == "np.zeros(300, dtype=np.float64)"
    assert to_source(np.full(300, 1e3, np.float32)) == "np.full(300, 1000.0, dtype=np.float32)"
    assert to_source(np.array([0.0, -0.0])) == "np.array([0.0, -0.0], dtype=np.float64)"
    assert to_source(np.float32(0.1)) == "np.float32(0.1)"


def test_unsupported_inputs_are_refused():
    with pytest.raises(TypeError):
        to_source(object())
    with pytest.raises(TypeError):
        to_source(np.array(["a"]))


# --- replay ---------------------------------------------------------------------------------------


@dp.transform(inverse=lambda X: 2 * np.fft.ifft(X))
def _bad_inverse(x):
    return np.fft.fft(x)


class _Counter:
    def __init__(self):
        self.calls = 0

    def process(self, block):
        self.calls += 1
        return block * (2.0 if self.calls == 3 else 1.0)


_third_block_doubled = dp.streaming()(_Counter)


def test_replay_runs_one_relation_without_hypothesis():
    with pytest.raises(AssertionError):
        replay(_bad_inverse, "inversibility", np.array([1.0]))
    replay(_third_block_doubled, "block-invariance", (np.ones(3), [3, 3]))  # two blocks: no bug
    with pytest.raises(AssertionError):
        replay(_third_block_doubled, "block invariance", (np.ones(3), [1, 2]))
    with pytest.raises(ValueError, match="not a relation"):
        replay(_bad_inverse, "causality", np.array([1.0]))


# --- writing files --------------------------------------------------------------------------------

OPERATORS = '''
import numpy as np
import dsp_props as dp

FIXED = False

@dp.transform(inverse=lambda X: (1 if FIXED else 2) * np.fft.ifft(X))
def bad_inverse(x):
    return np.fft.fft(x)

BANK = {"leaky": dp.lti_filter()(lambda x: x + (0 if FIXED else 1e-3) * x**2)}
'''


def _module(pytester, name="ops"):
    pytester.makepyfile(**{name: OPERATORS})
    pytester.syspathinsert()
    import importlib

    return importlib.import_module(name)


def test_written_tests_fail_until_the_bug_is_fixed(pytester):
    ops = _module(pytester, "ops_fix")
    report = dp.verify(ops.bad_inverse, raise_on_failure=False, seed=0, max_examples=30)
    written = dp.write_regression_tests(ops.bad_inverse, report, pytester.path)
    assert written.path.name == "test_regression_bad_inverse.py"
    assert len(written.added) == len(report.failures) >= 1 and not written.skipped
    text = written.path.read_text()
    assert "from ops_fix import bad_inverse as OPERATOR" in text and "import hypothesis" not in text
    pytester.runpytest("-p", "no:cacheprovider", "-p", "no:dsp_props").assert_outcomes(failed=len(written.added))
    # the same counterexamples are not appended twice
    assert dp.write_regression_tests(ops.bad_inverse, report, pytester.path).added == []
    pytester.makepyfile(conftest="import ops_fix\nops_fix.FIXED = True\n")
    pytester.runpytest("-p", "no:cacheprovider", "-p", "no:dsp_props").assert_outcomes(passed=len(written.added))


def test_operators_in_dictionaries_and_other_directories(pytester):
    ops = _module(pytester, "ops_bank")
    op = ops.BANK["leaky"]
    report = dp.verify(op, raise_on_failure=False, seed=0, max_examples=30)
    written = dp.write_regression_tests(op, report, pytester.path / "regressions")
    assert written.path.name == "test_regression_bank_leaky.py"
    text = written.path.read_text()
    # a top-level module of another directory is loaded by path, not imported by name
    assert 'load_operator(Path(__file__).parent / "../ops_bank.py", "BANK", "leaky")' in text
    result = pytester.runpytest("regressions", "-p", "no:cacheprovider", "-p", "no:dsp_props")
    result.assert_outcomes(failed=len(written.added))


def test_input_is_recorded_before_the_operator_writes_into_it():
    @dp.lti_filter()
    def overwrites(x):
        y = x.copy()
        x[:] = 0.0  # a bug: the caller's signal is destroyed
        return y

    report = dp.verify(overwrites, raise_on_failure=False, seed=0, max_examples=50)
    assert report.failures
    failure = next(f for f in report.failures if f.relation == "linearity")
    assert np.any(failure.example[0][0])  # the input as drawn, not the zeroed one


def test_non_reproducible_failures_give_no_test(tmp_path):
    state = {"broken": True}

    @dp.lti_filter()
    def sometimes(x):
        return 2 * x * x if state["broken"] else x

    report = dp.verify(sometimes, raise_on_failure=False, seed=0, max_examples=30)
    assert report.failures
    state["broken"] = False  # what a hidden state or a random operator does between two runs
    written = dp.write_regression_tests(sometimes, report, tmp_path)
    assert written.path is None and written.added == []
    assert written.skipped and all("passes" in why for _, why in written.skipped)


def test_load_operator_reads_a_file(tmp_path):
    (tmp_path / "ops_file.py").write_text(OPERATORS)
    op = load_operator(tmp_path / "ops_file.py", "BANK", "leaky")
    assert hasattr(op, "__dsp_props__")
    assert load_operator(tmp_path / "ops_file.py", "BANK", "leaky") is op  # loaded once


# --- pytest plugin and command line ---------------------------------------------------------------

PLUGIN_OPERATORS = '''
import numpy as np
import dsp_props as dp

@dp.transform(inverse=lambda X: 2 * np.fft.ifft(X))
def bad_inverse(x):
    return np.fft.fft(x)

@dp.transform(inverse=np.fft.ifft, parseval=True)
def good_fft(x):
    return np.fft.fft(x)
'''


def test_plugin_writes_regression_tests_next_to_the_test_module(pytester):
    pytester.makepyfile(test_ops=PLUGIN_OPERATORS)
    args = ["--dsp-props-profile", "fast", "--dsp-props-database", "none", "-p", "no:cacheprovider"]
    result = pytester.runpytest(*args, "--dsp-props-regressions")
    result.assert_outcomes(passed=7, failed=2)
    assert "regression test written:" in result.stdout.str()
    path = pytester.path / "test_regression_bad_inverse.py"
    assert "from test_ops import bad_inverse as OPERATOR" in path.read_text()
    # next run: the generated tests are collected too, and fail first, without Hypothesis
    result = pytester.runpytest(*args, "test_regression_bad_inverse.py")
    result.assert_outcomes(failed=2)
    # off by default
    path.unlink()
    pytester.runpytest(*args, "test_ops.py")
    assert not path.exists()


def test_plugin_ini_option_and_directory(pytester):
    pytester.makepyfile(test_ops=PLUGIN_OPERATORS)
    pytester.makeini("[pytest]\ndsp_props_regressions = regressions\n")
    result = pytester.runpytest("--dsp-props-profile", "fast", "--dsp-props-database", "none",
                                "-p", "no:cacheprovider")
    result.assert_outcomes(passed=7, failed=2)
    assert (pytester.path / "regressions" / "test_regression_bad_inverse.py").exists()


def test_cli_writes_regression_tests(tmp_path, capsys):
    assert main(["check", f"{EXAMPLE}:buggy_fft", "--examples", "20", "--regressions", str(tmp_path)]) == 1
    assert "regression test written:" in capsys.readouterr().out
    (path,) = tmp_path.glob("test_regression_*.py")
    assert "load_operator(" in path.read_text()  # a file given to the CLI is loaded by path
    assert main(["check", f"{EXAMPLE}:fft", "--examples", "20", "--regressions", str(tmp_path)]) == 0
    assert len(list(tmp_path.glob("test_regression_*.py"))) == 1

"""Versioned calibration profiles (tolerances.json) and precision regressions (plan, action 2.2)."""

import json
import pathlib

import numpy as np
import pytest
from scipy import signal

import dsp_props as dp
from dsp_props.cli import main
from dsp_props.core import CalibrationError, FFTTolerance
from dsp_props.tolerances import NoReferenceError, operator_name
from mutants import precision_mutants as pm

pytest_plugins = ["pytester"]

EXAMPLE = str(pathlib.Path(__file__).resolve().parents[1] / "examples" / "filters.py")
LENGTHS = range(1, 65)

B, A = signal.butter(4, 0.2)


def _fft():
    @dp.transform(inverse=np.fft.ifft, parseval=True)
    def fft(x):
        return np.fft.fft(x)

    return fft


def test_calibration_is_stored_and_read_back(tmp_path):
    fft = _fft()
    path = tmp_path / "tolerances.json"
    tf = dp.ToleranceFile(path)
    tol = tf.calibrate(fft, lengths=LENGTHS)
    tf.save()
    data = json.loads(path.read_text())
    (key,) = data["operators"]
    entry = data["operators"][key]
    assert key == operator_name(fft) and key.endswith(":_fft.<locals>.fft")
    assert entry["lengths"] == "1-64" and entry["reference"] == "extended_dft" and entry["base_c"] == 0.0
    assert entry["calibrated_with"]["numpy"] == np.__version__
    back = dp.ToleranceFile(path).tolerance_for(fft)
    assert back.ratios == tol.ratios and back.margin == 8.0 and back.chained == 2
    # stored ratios are rounded up, never down: the tolerance cannot shrink by storage
    exact = dp.calibrate(fft.__dsp_props__.func, lengths=LENGTHS, base=None)
    assert all(back.ratios[n] >= exact.ratios[n] for n in LENGTHS)
    assert back.bound(64, np.float64, 1.0) < FFTTolerance().bound(64, np.float64, 1.0)


def test_verify_uses_the_file_unless_the_declaration_sets_a_tolerance(tmp_path, monkeypatch):
    fft = _fft()
    tf = dp.ToleranceFile(tmp_path / "t.json")
    tf.calibrate(fft, lengths=LENGTHS)
    tf.save()
    assert dp.verify(fft, max_examples=20, seed=0).tolerance == "default"
    report = dp.verify(fft, max_examples=20, seed=0, tolerances=tmp_path / "t.json")
    assert report.ok and report.tolerance == "calibrated (t.json)"
    assert report.to_dict()["tolerance"] == "calibrated (t.json)"
    monkeypatch.setenv("DSP_PROPS_TOLERANCES", str(tmp_path / "t.json"))
    assert dp.verify(fft, max_examples=20, seed=0).tolerance == "calibrated (t.json)"
    assert dp.verify(fft, max_examples=20, seed=0, tolerances="none").tolerance == "default"

    @dp.transform(inverse=np.fft.ifft, tolerance=FFTTolerance(c=50))
    def explicit(x):
        return np.fft.fft(x)

    tf.calibrate(explicit, lengths=LENGTHS)
    tf.save()
    assert dp.verify(explicit, max_examples=20, seed=0).tolerance == "declared"


def test_a_less_accurate_version_fails_against_the_old_calibration(tmp_path):
    # the same operator name, a new implementation that lost about 6 bits in its twiddles:
    # it passes the default tolerance, not the calibration of the accurate version
    degraded = pm.declare_fft(pm.dft_with_rounded_twiddles(6))
    tf = dp.ToleranceFile(tmp_path / "t.json")
    tf.calibrate(pm.declare_fft(pm.radix2_reference), lengths=range(1, 129), name=operator_name(degraded))
    assert dp.verify(degraded, max_examples=100, seed=0, raise_on_failure=False).ok
    assert not dp.verify(degraded, max_examples=100, seed=0, raise_on_failure=False, tolerances=tf).ok


def test_compare_reports_precision_regressions(tmp_path):
    trusted = pm.declare_fft(pm.radix2_reference)
    degraded = pm.declare_fft(pm.radix2_recursive_twiddles)  # 19 eps: too small for the relations
    name = operator_name(degraded)
    tf = dp.ToleranceFile(tmp_path / "t.json")
    tf.calibrate(trusted, lengths=range(1, 129), name=name)
    same = tf.compare(trusted, name=name)
    assert same.ok and not same.regressions and same.worst_now == pytest.approx(same.worst_before, rel=1e-3)
    # the relations miss it, even with the calibration of the accurate version
    assert dp.verify(degraded, max_examples=100, seed=0, raise_on_failure=False, tolerances=tf).ok
    worse = tf.compare(degraded)
    assert not worse.ok and len(worse.regressions) > 10
    n, before, now = max(worse.regressions, key=lambda r: r[2])
    assert now > 2 * max(before, 1.0)
    tf.entries[name]["calibrated_with"]["numpy"] = "1.26.4"
    assert "numpy 1.26.4 -> " in str(tf.compare(degraded))
    assert tf.compare(degraded, factor=1e6).ok
    with pytest.raises(KeyError):
        tf.compare(_fft())


def test_filters_and_real_transforms_have_a_reference(tmp_path):
    @dp.lti_filter(transfer=(B, A))
    def lowpass(x):
        return signal.lfilter(B, A, x)

    @dp.transform(kind="rdft", inverse=np.fft.irfft)
    def rfft(x):
        return np.fft.rfft(x)

    tf = dp.ToleranceFile(tmp_path / "t.json")
    if np.finfo(np.longdouble).eps < np.finfo(np.float64).eps:
        assert tf.calibrate(lowpass, lengths=range(1, 33)).worst < 1e3
    assert tf.calibrate(rfft, lengths=LENGTHS).worst < 10
    assert dp.verify(rfft, max_examples=30, seed=0, tolerances=tf).ok


def test_calibration_refuses_what_it_cannot_measure(tmp_path):
    tf = dp.ToleranceFile(tmp_path / "t.json")

    @dp.resampler(2, 3)
    def resample(x):
        return signal.resample_poly(x, 2, 3)

    with pytest.raises(NoReferenceError):
        tf.calibrate(resample)

    @dp.transform(inverse=np.fft.ifft, precision=np.float32)
    def fft32(x):
        return np.fft.fft(x).astype(np.complex64)

    with pytest.raises(CalibrationError, match="float64"):
        tf.calibrate(fft32)

    @dp.transform(inverse=np.fft.ifft)
    def shifted(x):
        return np.roll(np.fft.fft(x), 1)

    with pytest.raises(CalibrationError, match="plausible"):
        tf.calibrate(shifted, lengths=LENGTHS)
    assert tf.entries == {}

    @dp.streaming()
    class Gain:
        def process(self, block):
            return 2 * block

    tol = tf.calibrate(Gain, reference=lambda x: 2 * np.asarray(x, np.longdouble), lengths=LENGTHS)
    assert tol.worst == 0.0


def test_unknown_format_is_refused(tmp_path):
    path = tmp_path / "t.json"
    path.write_text('{"format": 99, "operators": {}}')
    with pytest.raises(ValueError, match="format"):
        dp.ToleranceFile(path)
    with pytest.raises(FileNotFoundError):
        dp.verify(_fft(), max_examples=5, tolerances=tmp_path / "missing.json")


def test_cli_calibrate_check_and_compare(tmp_path, capsys):
    path = str(tmp_path / "tolerances.json")
    # buggy_fft is refused (exit 1), resample_2_3 has no reference (skipped)
    assert main(["calibrate", EXAMPLE, "--tolerances", path, "--lengths", "1-32"]) == 1
    out = capsys.readouterr().out
    assert "FAIL" in out and "buggy_fft" in out and "skip" in out and "wrote" in out
    operators = json.loads(pathlib.Path(path).read_text())["operators"]
    assert "filters:fft" in operators and "filters:buggy_fft" not in operators
    assert main(["check", f"{EXAMPLE}:fft", "--examples", "20", "--tolerances", path]) == 0
    assert "calibrated tolerance" in capsys.readouterr().out
    report = tmp_path / "cmp.json"
    assert main(["calibrate", f"{EXAMPLE}:fft", "--tolerances", path, "--compare", "--json", str(report)]) == 0
    assert "no precision regression" in capsys.readouterr().out
    assert json.loads(report.read_text())["operators"][0]["ok"] is True


def test_pytest_option(pytester):
    pytester.makepyfile(test_ops='''
import numpy as np
import dsp_props as dp

@dp.transform(inverse=np.fft.ifft, parseval=True)
def my_fft(x):
    return np.fft.fft(x)
''')
    pytester.syspathinsert()
    import test_ops

    tf = dp.ToleranceFile(pytester.path / "tolerances.json")
    tf.calibrate(test_ops.my_fft, lengths=LENGTHS)
    tf.save()
    result = pytester.runpytest("--dsp-props-profile", "fast", "--dsp-props-database", "none",
                                "--dsp-props-tolerances", "tolerances.json", "-p", "no:cacheprovider")
    result.assert_outcomes(passed=5)

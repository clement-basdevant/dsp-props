"""The gallery scripts (docs/gallery.md, examples/gallery/) run and show what the page says."""

import importlib.util
import pathlib

import pytest
import scipy

import dsp_props as dp

GALLERY = pathlib.Path(__file__).resolve().parents[1] / "examples" / "gallery"
# the SciPy releases the gallery was written against (1.18.1 and earlier) still have the bugs
OLD_SCIPY = tuple(int(p) for p in scipy.__version__.split(".")[:2]) < (1, 19)


def _load(name):
    spec = importlib.util.spec_from_file_location(f"gallery_{name}", GALLERY / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_script_has_a_test():
    assert {p.stem for p in GALLERY.glob("*.py")} == {
        "empty_block_state", "filtfilt_gust_trailing_zeros", "filtfilt_gust_near_zero_poles", "torch_fft_precision"}


@pytest.mark.full_examples
def test_empty_block_state():
    g = _load("empty_block_state")
    fixed = dp.verify(g.fixed_stream, max_examples=100, seed=0, raise_on_failure=False)
    assert fixed.ok
    for op in (g.iir_stream, g.sos_stream):
        report = dp.verify(op, max_examples=100, seed=0, raise_on_failure=False)
        if OLD_SCIPY:
            assert [f.relation for f in report.failures] == ["block invariance"]
            x, cuts = report.failures[0].example
            assert len(x) <= 2  # a minimal input: one or two samples around an empty block


def test_filtfilt_gust_trailing_zeros():
    g = _load("filtfilt_gust_trailing_zeros")
    if OLD_SCIPY:
        with pytest.raises(AssertionError, match="trailing zeros"):
            g.test_trailing_zeros_do_not_change_filtfilt()
    else:
        try:
            g.test_trailing_zeros_do_not_change_filtfilt()
        except AssertionError:
            pass  # still present in this SciPy: the page stays true


def test_filtfilt_gust_near_zero_poles():
    g = _load("filtfilt_gust_near_zero_poles")
    assert g.reversal_gap(0.1) < 10
    if OLD_SCIPY:
        gaps = [g.reversal_gap(r) for r in g.RADII]
        assert gaps == sorted(gaps) and gaps[-2] > 1e5  # grows as the poles shrink, 1e6 eps at 1e-4


def test_torch_fft_precision():
    pytest.importorskip("torch")
    g = _load("torch_fft_precision")
    numpy_eps, torch_eps = g.worst_eps(g.np.fft.fft, [596]), g.worst_eps(g.torch_fft, [596])
    assert numpy_eps[596] < 5 and torch_eps[596] > 50

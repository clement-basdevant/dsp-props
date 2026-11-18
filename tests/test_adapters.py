import pathlib
import shutil

import numpy as np
import pytest
from scipy import signal

import dsp_props as dp
from dsp_props.adapters import build_shared, c_block_processor, c_function

SOURCE = pathlib.Path(__file__).resolve().parents[1] / "examples" / "c" / "biquad.c"
B, A = signal.butter(2, 0.2)

pytestmark = pytest.mark.skipif(
    not (shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")), reason="no C compiler"
)


@pytest.fixture(scope="module")
def lib():
    return build_shared(SOURCE)


def test_c_function_matches_lfilter(lib):
    f = c_function(lib, "biquad", state_size=2)
    x = np.random.default_rng(0).standard_normal(100)
    np.testing.assert_allclose(f(x), signal.lfilter(B, A, x), rtol=0, atol=1e-13)


def test_correct_c_biquad_passes_lti_and_streaming(lib):
    dp.verify(dp.lti_filter(transfer=(B, A))(c_function(lib, "biquad", state_size=2)), max_examples=60)
    dp.verify(dp.streaming()(c_block_processor(lib, "biquad", state_size=2)), max_examples=60)


@pytest.mark.full_examples
def test_single_precision_accumulator_is_caught(lib):
    f = dp.lti_filter(transfer=(B, A))(c_function(lib, "biquad_float_acc", state_size=2))
    assert not dp.verify(f, max_examples=60, raise_on_failure=False).ok


@pytest.mark.full_examples
def test_state_dropped_between_blocks_is_caught_only_by_streaming(lib):
    # one call over the whole signal is a correct filter...
    dp.verify(dp.lti_filter(transfer=(B, A))(c_function(lib, "biquad_drops_state", state_size=2)),
              max_examples=60)
    # ...but cutting it into blocks breaks it
    report = dp.verify(dp.streaming()(c_block_processor(lib, "biquad_drops_state", state_size=2)),
                       max_examples=60, raise_on_failure=False)
    assert [f.relation for f in report.failures] == ["block invariance"]

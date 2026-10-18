import numpy as np
from hypothesis import given, settings

from dsp_props.generators import block_splits, complex_signals, lengths, signal_pairs, signals, split_blocks
from hypothesis import strategies as st


@settings(max_examples=100, deadline=None)
@given(st.data())
def test_signals_have_requested_length_and_are_finite(data):
    n = data.draw(lengths(256))
    x = data.draw(signals(n))
    assert x.shape == (n,)
    assert np.all(np.isfinite(x))


@settings(max_examples=50, deadline=None)
@given(signal_pairs(64))
def test_pairs_share_length(pair):
    x, y = pair
    assert x.shape == y.shape


@settings(max_examples=100, deadline=None)
@given(st.data())
def test_complex_signals_have_requested_length_and_are_finite(data):
    n = data.draw(lengths(256))
    x = data.draw(complex_signals(n))
    assert x.shape == (n,)
    assert np.all(np.isfinite(x))


def test_complex_signals_include_complex_input():
    seen = []

    @settings(max_examples=100, deadline=None, database=None)
    @given(complex_signals(8))
    def record(x):
        seen.append(np.iscomplexobj(x) and np.any(np.imag(x) != 0))

    record()
    assert any(seen)


def test_block_splits_cover_signal_and_go_beyond_eight_blocks():
    counts, sizes = [], []

    @settings(max_examples=300, deadline=None, database=None)
    @given(st.integers(0, 64).flatmap(lambda n: st.tuples(st.just(n), block_splits(n))))
    def record(item):
        n, cuts = item
        blocks = split_blocks(np.arange(n), cuts)
        assert np.array_equal(np.concatenate(blocks), np.arange(n))
        counts.append(len(blocks))
        sizes.extend(len(b) for b in blocks)

    record()
    assert max(counts) > 8
    assert 0 in sizes and 1 in sizes

"""Hypothesis strategies that produce signals a DSP engineer would recognise.

Every strategy takes the length ``n`` explicitly so that relations can draw
several signals of the same length. Generators are ordered so that Hypothesis
shrinks towards simple cases: short lengths, impulses, unit amplitude.
"""

from __future__ import annotations

import numpy as np
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

# Lengths that historically break DSP code: tiny, odd, prime, 2^k +/- 1.
PATHOLOGICAL_LENGTHS = [1, 2, 3, 5, 7, 8, 15, 16, 17, 31, 32, 33, 97, 127, 128, 129, 255, 256, 257]

AMPLITUDES = [1.0, 1e-3, 1e3, 1e-100, 1e100]


def lengths(max_n: int = 512) -> st.SearchStrategy[int]:
    """Signal lengths, biased towards pathological values."""
    patho = [n for n in PATHOLOGICAL_LENGTHS if n <= max_n]
    return st.one_of(st.sampled_from(patho), st.integers(1, max_n))


def amplitudes(dtype=np.float64):
    """The test amplitudes that ``dtype`` holds with room to spare (all of them in float64).

    A float32 operator cannot be given 1e-100 or 1e100: the cast alone would
    underflow or overflow, and the failure would say nothing about the operator.
    """
    info = np.finfo(dtype)
    return [a for a in AMPLITUDES if float(info.tiny) * 1e6 < a < float(info.max) / 1e6]


def _amplitude(dtype=np.float64):
    return st.sampled_from(amplitudes(dtype))


def _cast(dtype, strategy):
    """``strategy`` with its signals cast to ``dtype`` (complex ones to the matching complex dtype).

    The families build their signals in float64, the default, which is then left untouched.
    """
    if np.dtype(dtype) == np.float64:
        return strategy
    return strategy.map(lambda x: x.astype(np.result_type(dtype, np.complex64) if np.iscomplexobj(x) else dtype))


def impulse(n: int, dtype=np.float64):
    """A single non-zero sample at an arbitrary position, of length ``n``."""
    return _cast(dtype, st.builds(
        lambda k, a: _impulse(n, k, a), st.integers(0, n - 1), _amplitude(dtype)
    ))


def _impulse(n, k, a):
    x = np.zeros(n)
    x[k] = a
    return x


def dc(n: int, dtype=np.float64):
    """A constant signal of length ``n``."""
    return _cast(dtype, _amplitude(dtype).map(lambda a: np.full(n, a)))


def _sine(n, f, phi, a):
    t = np.arange(n)
    return a * np.cos(2 * np.pi * f * t + phi)


def offgrid_sine(n: int, dtype=np.float64):
    """Sinusoid whose normalised frequency is not a multiple of 1/N."""
    freq = st.floats(0.0, 0.5, allow_nan=False, exclude_max=False)
    phase = st.floats(-np.pi, np.pi, allow_nan=False)
    return _cast(dtype, st.builds(lambda f, p, a: _sine(n, f, p, a), freq, phase, _amplitude(dtype)))


def near_nyquist_sine(n: int, dtype=np.float64):
    """Sinusoid with normalised frequency in [0.45, 0.5], close to Nyquist."""
    freq = st.floats(0.45, 0.5, allow_nan=False)
    phase = st.floats(-np.pi, np.pi, allow_nan=False)
    return _cast(dtype, st.builds(lambda f, p, a: _sine(n, f, p, a), freq, phase, _amplitude(dtype)))


def close_tones(n: int, dtype=np.float64):
    """Two sinusoids separated by less than one frequency bin."""
    f0 = st.floats(0.01, 0.45, allow_nan=False)
    df = st.floats(0.0, 1.0, allow_nan=False)
    return _cast(dtype, st.builds(
        lambda f, d, a: _sine(n, f, 0.0, a) + _sine(n, f + d / max(n, 1), 0.3, a),
        f0,
        df,
        _amplitude(dtype),
    ))


def white_noise(n: int, dtype=np.float64):
    """Noise with independent samples in [-1, 1], scaled by one of the test amplitudes."""
    elems = st.floats(-1.0, 1.0, allow_nan=False, allow_infinity=False, width=64)
    return _cast(dtype, st.builds(
        lambda x, a: a * x, hnp.arrays(np.float64, n, elements=elems), _amplitude(dtype)
    ))


def sparse(n: int, dtype=np.float64):
    """A few impulses at arbitrary positions."""
    return _cast(dtype, st.builds(
        lambda idx, a: _sparse(n, idx, a),
        st.lists(st.integers(0, n - 1), min_size=1, max_size=min(n, 4)),
        _amplitude(dtype),
    ))


def _sparse(n, idx, a):
    x = np.zeros(n)
    for i, k in enumerate(idx):
        x[k] += a * (-1) ** i
    return x


SIGNAL_KINDS = [impulse, dc, offgrid_sine, near_nyquist_sine, close_tones, sparse, white_noise]


def signals(n: int, dtype=np.float64):
    """Any of the signal families above, of length ``n``, in the real floating dtype ``dtype``."""
    return st.one_of([kind(n, dtype) for kind in SIGNAL_KINDS])


def complex_exponential(n: int, dtype=np.float64):
    """exp(i (2 pi f t + phi)) with f in [-0.5, 0.5]: negative frequencies have no real counterpart."""
    freq = st.floats(-0.5, 0.5, allow_nan=False)
    phase = st.floats(-np.pi, np.pi, allow_nan=False)
    return _cast(dtype, st.builds(
        lambda f, p, a: a * np.exp(1j * (2 * np.pi * f * np.arange(n) + p)), freq, phase, _amplitude(dtype)
    ))


def complex_noise(n: int, dtype=np.float64):
    """Noise with independent real and imaginary parts in [-1, 1], scaled by a test amplitude."""
    elems = st.floats(-1.0, 1.0, allow_nan=False, allow_infinity=False, width=64)
    part = hnp.arrays(np.float64, n, elements=elems)
    return _cast(dtype, st.builds(lambda re, im, a: a * (re + 1j * im), part, part, _amplitude(dtype)))


def complex_signals(n: int, dtype=np.float64):
    """Real signals first (so failures shrink to real input when they can), then complex ones.

    Besides the two families above, ``x + i y`` with ``x`` and ``y`` any real
    signals catches code that mixes up or drops the imaginary part of its input.
    """
    return st.one_of(
        signals(n, dtype),
        complex_exponential(n, dtype),
        st.builds(lambda x, y: x + 1j * y, signals(n, dtype), signals(n, dtype)),
        complex_noise(n, dtype),
    )


@st.composite
def signal_pairs(draw, max_n: int = 512, complex: bool = False, dtype=np.float64):
    """Two signals of the same length (drawn from ``lengths(max_n)``), complex ones too if ``complex``."""
    n = draw(lengths(max_n))
    kind = complex_signals if complex else signals
    return draw(kind(n, dtype)), draw(kind(n, dtype))


@st.composite
def block_splits(draw, n: int, max_blocks: int | None = None):
    """Cut points splitting a length-``n`` signal into blocks.

    Three kinds of split are mixed:

    - arbitrary cut points, up to ``max_blocks`` blocks (default
      ``max(n, 16)``, so that even short signals can be cut into more than ten
      blocks by adding empty ones);
    - every block of a single sample (``n`` blocks);
    - blocks of a fixed size ``B``, the last one possibly shorter.

    Arbitrary cuts come first so that counterexamples shrink towards few
    blocks. They include empty blocks and blocks of a single sample, the
    cases that expose state-handling bugs in streaming code; the two regular
    splits reach large numbers of calls, which a few arbitrary cuts rarely
    do (a processor that loses its state at its tenth call, say).
    """
    if max_blocks is None:
        max_blocks = max(n, 16)
    kind = draw(st.sampled_from(["arbitrary", "ones", "fixed"]))
    if kind == "ones":
        return list(range(1, n))
    if kind == "fixed":
        size = draw(st.integers(1, max(n, 1)))
        return list(range(size, n, size))
    return sorted(draw(st.lists(st.integers(0, n), max_size=max_blocks - 1)))


def split_blocks(x, cuts):
    """Split ``x`` at the sorted cut points ``cuts`` (from ``block_splits``) into a list of blocks."""
    edges = [0, *cuts, len(x)]
    return [x[i:j] for i, j in zip(edges[:-1], edges[1:])]

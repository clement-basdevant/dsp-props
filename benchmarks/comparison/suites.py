"""Test suites applied to each subject (and to each of its mutants).

For every subject module this file provides:

* ``decls(mod)``: the dsp-props declarations a user would write (a few
  decorators per subject);
* ``sop(mod)``: the state-of-practice suite, ported from the test suite of
  the library whose interface the subject implements (NumPy's and SciPy's
  FFT tests, SciPy's lfilter / lfilter_zi, upfirdn and resample_poly tests).
  Each test keeps the inputs, sizes and tolerances of the original; the
  origin is given in its docstring. The overlap-add subject has no library
  counterpart: its suite is written in the same style;
* ``oracle(mod)``: the outputs of the subject on a fixed battery of inputs,
  compared with an independent reference to classify mutants (equivalent,
  accuracy fault, functional fault). Only the classification uses it.
"""

from __future__ import annotations

import math

import numpy as np
import scipy.signal as ss
from numpy.testing import assert_allclose, assert_array_almost_equal

import dsp_props as dp

# Default relative tolerance of scipy's xp_assert_close for float64: 4 * sqrt(eps).
XP_RTOL = 4 * np.finfo(float).eps ** 0.5


def xp_assert_close(actual, desired, rtol=XP_RTOL, atol=0.0):
    actual, desired = np.asarray(actual), np.asarray(desired)
    assert actual.shape == desired.shape, (actual.shape, desired.shape)
    assert_allclose(actual, desired, rtol=rtol, atol=atol)


# --------------------------------------------------------------------------
# Subject 1: FFT
# --------------------------------------------------------------------------

def fft1(x):
    """Naive DFT used as oracle by NumPy's and SciPy's tests."""
    L = len(x)
    phase = -2j * np.pi * (np.arange(L) / L)
    phase = np.arange(L).reshape(-1, 1) * phase
    return np.sum(x * np.exp(phase), axis=1)


def fft_decls(mod):
    fwd = dp.transform(inverse=mod.ifft, parseval=True, real_input_hermitian=True)(lambda x: mod.fft(x))
    inv = dp.transform(inverse=mod.fft, kind="generic", parseval=True,
                       parseval_scale=lambda n: 1.0 / n)(lambda X: mod.ifft(X))
    return {"fft": fwd, "ifft": inv}


def fft_sop(mod):
    rng = np.random.RandomState(1234)  # the original tests use the unseeded global generator
    random = rng.random_sample

    def numpy_test_identity():
        """numpy/fft/tests/test_pocketfft.py::TestFFT1D::test_identity (complex part)."""
        maxlen = 512
        x = random(maxlen) + 1j * random(maxlen)
        for i in range(1, maxlen):
            assert_allclose(mod.ifft(mod.fft(x[0:i])), x[0:i], atol=1e-12)

    def numpy_test_fft():
        """numpy/fft/tests/test_pocketfft.py::TestFFT1D::test_fft (default norm)."""
        x = random(30) + 1j * random(30)
        assert_allclose(fft1(x), mod.fft(x), atol=1e-6)

    def numpy_test_ifft():
        """numpy/fft/tests/test_pocketfft.py::TestFFT1D::test_ifft (default norm)."""
        x = random(30) + 1j * random(30)
        assert_allclose(x, mod.ifft(mod.fft(x)), atol=1e-6)

    def scipy_test_identity():
        """scipy/fft/tests/test_basic.py::TestFFT::test_identity (complex part)."""
        maxlen = 512
        x = random(maxlen) + 1j * random(maxlen)
        for i in [1, 2, 16, 128, 512, 53, 149, 281, 397]:
            xp_assert_close(mod.ifft(mod.fft(x[0:i])), x[0:i])

    def scipy_test_fft():
        """scipy/fft/tests/test_basic.py::TestFFT::test_fft (default norm)."""
        x = random(30) + 1j * random(30)
        xp_assert_close(mod.fft(x), fft1(x))

    def scipy_test_norm_preserving():
        """scipy/fft/tests/test_basic.py::TestFFT::test_all_1d_norm_preserving (fft/ifft, n = size)."""
        x = random(30).astype(complex)
        xp_assert_close(np.linalg.norm(mod.ifft(mod.fft(x))), np.linalg.norm(x))

    return [numpy_test_identity, numpy_test_fft, numpy_test_ifft, scipy_test_identity, scipy_test_fft,
            scipy_test_norm_preserving]


def fft_oracle(mod):
    """Outputs on all lengths 1..130 and a few large ones, real and complex inputs."""
    rng = np.random.default_rng(7)
    outs = []
    for n in list(range(1, 131)) + [255, 256, 257, 384, 511, 512]:
        for x in (rng.standard_normal(n), rng.standard_normal(n) + 1j * rng.standard_normal(n),
                  np.eye(n)[rng.integers(n)]):
            outs.append(("fft", n, mod.fft(x), np.fft.fft(x.astype(np.clongdouble))))
            outs.append(("ifft", n, mod.ifft(x), np.fft.ifft(x.astype(np.clongdouble))))
    return outs


# --------------------------------------------------------------------------
# Subject 2: lfilter
# --------------------------------------------------------------------------

POLES4 = 0.95 * np.exp(1j * np.array([0.3, -0.3, 1.2, -1.2]))
A4 = np.real(np.poly(POLES4))
B4 = np.array([0.2, 0.0, -0.4, 0.0, 0.2])
B2, A2 = np.array([1.0, 0.5, 0.25]), np.array([2.0, -0.6])


def lfilter_decls(mod):
    f4 = dp.lti_filter(transfer=(B4, A4))(lambda x: mod.lfilter(B4, A4, x))
    f2 = dp.lti_filter(transfer=(B2, A2))(lambda x: mod.lfilter(B2, A2, x))
    stream = dp.stateful(lambda block, z: mod.lfilter(B4, A4, block, zi=z), lambda: np.zeros(4))
    return {"lfilter_resonant": f4, "lfilter_a0_not_1": f2, "lfilter_streaming": stream}


def lfilter_family_decls(mod):
    """Approach dsp-props/params: the filter coefficients are drawn too (``dp.family``)."""

    @dp.family(dp.generators.iir_coefficients(max_order=4, radii=(0.5, 0.0, 0.3, 0.9)))
    def lfilter_params(b, a):
        return dp.lti_filter(transfer=(b, a))(lambda x: mod.lfilter(b, a, x))

    return {**lfilter_decls(mod), "lfilter_params": lfilter_params}


def lfilter_sop(mod):
    x6 = np.linspace(0, 5, 6)

    def test_rank_1_IIR():
        """scipy/signal/tests/test_signaltools.py::_TestLinearFilter::test_rank_1_IIR."""
        assert_array_almost_equal(mod.lfilter([1, -1], [0.5, -0.5], x6), [0, 2, 4, 6, 8, 10.])

    def test_rank_1_FIR():
        """::test_rank_1_FIR."""
        assert_array_almost_equal(mod.lfilter([1, 1], [1], x6), [0, 1, 3, 5, 7, 9.])

    def test_rank_1_IIR_init_cond():
        """::test_rank_1_IIR_init_cond."""
        y, zf = mod.lfilter([1, 0, -1], [0.5, -0.5], x6, zi=[1, 2])
        assert_array_almost_equal(y, [1, 5, 9, 13, 17, 21])
        assert_array_almost_equal(zf, [13, -10])

    def test_rank_1_FIR_init_cond():
        """::test_rank_1_FIR_init_cond."""
        y, zf = mod.lfilter([1, 1, 1], [1], x6, zi=[1, 1])
        assert_array_almost_equal(y, [1, 2, 3, 6, 9, 12.])
        assert_array_almost_equal(zf, [9, 5])

    def test_short_x_FIR():
        """::test_short_x_FIR (regression test for gh-5116)."""
        y, zf = mod.lfilter([1, 0, -1], [1], [72.0], zi=[2, 7])
        assert_array_almost_equal(y, [74])
        assert_array_almost_equal(zf, [7, -72])

    def test_short_x_IIR():
        """::test_short_x_IIR (regression test for gh-5116)."""
        y, zf = mod.lfilter([1, 0, -1], [1, 1], [72.0], zi=[2, 7])
        assert_array_almost_equal(y, [74])
        assert_array_almost_equal(zf, [-67, -72])

    def test_empty_zi():
        """::test_empty_zi (regression test for gh-880)."""
        x = np.linspace(0, 4, 5)
        y, zf = mod.lfilter([1], [1], x, zi=np.array([]))
        assert_array_almost_equal(y, x)
        assert np.size(zf) == 0

    def test_lfilter_empty_input():
        """::test_lfilter_empty_input (SciPy development version, gh-22571)."""
        y, zf = mod.lfilter([1.0, 0.5], [1.0, -0.5], np.array([]), zi=np.array([0.25]))
        assert np.shape(y) == (0,)
        assert_allclose(zf, [0.25], rtol=0, atol=0)
        assert np.shape(mod.lfilter([1.0, 0.5], [1.0, -0.5], np.array([]))) == (0,)

    def test_lfilter_zi_basic():
        """::TestLFilterZI::test_basic (steady state from lfilter_zi gives a constant output)."""
        for b, a in [([1., 0., 2.], [1., -1., 0.5]), ([1., 2., 3., 4.], [1.]), ([1., 9.], [1., 2., 3., 4.])]:
            zi = ss.lfilter_zi(b, a)
            y = mod.lfilter(b, a, np.ones(9), zi=zi)[0]
            assert_array_almost_equal(y, np.full(9, sum(b) / sum(a)))

    return [test_rank_1_IIR, test_rank_1_FIR, test_rank_1_IIR_init_cond, test_rank_1_FIR_init_cond,
            test_short_x_FIR, test_short_x_IIR, test_empty_zi, test_lfilter_empty_input, test_lfilter_zi_basic]


def lfilter_oracle(mod):
    rng = np.random.default_rng(11)
    filters = [(B4, A4), (B2, A2), (np.array([1.0, 1.0]), np.array([1.0])),
               (np.array([1.0, 0.0, -1.0]), np.array([0.5, -0.5])),
               (rng.standard_normal(3), np.real(np.poly([0.5, -0.3, 0.8j, -0.8j]))),
               # order 0 (pure gain, empty state), which the SciPy suite tests
               (np.array([2.0]), np.array([1.0])), (np.array([1.0]), np.array([4.0]))]
    outs = []
    for b, a in filters:
        nz = max(len(a), len(b)) - 1
        for n in list(range(0, 21)) + [64, 257]:
            x = rng.standard_normal(n)
            zi = rng.standard_normal(nz)
            outs.append(("y", n, mod.lfilter(b, a, x), _ref_lfilter(b, a, x, np.zeros(nz))[0]))
            y, zf = mod.lfilter(b, a, x, zi=zi)
            ry, rz = _ref_lfilter(b, a, x, zi)
            outs.append(("y_zi", n, y, ry))
            outs.append(("zf", n, zf, rz))
    return outs


def _ref_lfilter(b, a, x, zi):
    if len(x) == 0:
        return np.zeros(0), np.asarray(zi, dtype=float)  # scipy <= 1.18 is wrong here (gh-22571)
    return ss.lfilter(b, a, x, zi=zi)


# --------------------------------------------------------------------------
# Subject 3: upfirdn (and resample_poly built on it)
# --------------------------------------------------------------------------

def _output_len(len_h, len_x, up, down):
    return ((len_x - 1) * up + len_h - 1) // down + 1


def resample_poly(mod, x, up, down, window=("kaiser", 5.0)):
    """scipy.signal.resample_poly (1-D, padtype='constant'), calling the subject's upfirdn."""
    x = np.asarray(x, dtype=float)
    g = math.gcd(up, down)
    up, down = up // g, down // g
    if up == down == 1:
        return x.copy()
    n_in = len(x)
    n_out = n_in * up // down + bool(n_in * up % down)
    if isinstance(window, (list, np.ndarray)):
        h = np.array(window, dtype=float)
        half_len = (len(h) - 1) // 2
    else:
        max_rate = max(up, down)
        half_len = 10 * max_rate
        h = ss.firwin(2 * half_len + 1, 1.0 / max_rate, window=window)
    h = h * up
    n_pre_pad = down - half_len % down
    n_post_pad = 0
    n_pre_remove = (half_len + n_pre_pad) // down
    while _output_len(len(h) + n_pre_pad + n_post_pad, n_in, up, down) < n_out + n_pre_remove:
        n_post_pad += 1
    h = np.concatenate([np.zeros(n_pre_pad), h, np.zeros(n_post_pad)])
    return mod.upfirdn(h, x, up, down)[n_pre_remove:n_pre_remove + n_out]


H15 = ss.firwin(15, 0.3)


def upfirdn_decls(mod):
    out = {}
    for up, down in [(2, 3), (3, 1), (1, 2)]:
        out[f"resample_{up}_{down}"] = dp.resampler(up, down)(
            lambda x, up=up, down=down: resample_poly(mod, x, up, down))
    out["upfirdn_as_fir"] = dp.lti_filter(transfer=(H15, [1.0]))(lambda x: mod.upfirdn(H15, x)[:len(x)])
    return out


def upfirdn_family_decls(mod):
    """Approach dsp-props/params: taps, rates and the use of default arguments are drawn too."""
    from hypothesis import strategies as st

    params = st.tuples(dp.generators.fir_taps(16), dp.generators.rate_pairs(4), st.booleans())

    @dp.family(params)
    def upfirdn_params(h, rates, defaults):
        up, down = rates

        def run(x):
            if defaults and up == 1 and down == 1:
                return mod.upfirdn(h, x)  # default arguments
            if defaults and down == 1:
                return mod.upfirdn(h, x, up)
            return mod.upfirdn(h, x, up, down)

        return dp.resampler(up, down, causal=True, length=lambda n: _output_len(len(h), n, up, down),
                            dc_gain=None, stopband=None)(run)

    return {**upfirdn_decls(mod), "upfirdn_params": upfirdn_params}


def upfirdn_naive(x, h, up=1, down=1):
    """scipy/signal/tests/test_upfirdn.py::upfirdn_naive."""
    x = np.asarray(x)
    h = np.asarray(h)
    out = np.zeros(len(x) * up, x.dtype)
    out[::up] = x
    return np.convolve(h, out)[::down][:_output_len(len(h), len(x), up, down)]


def upfirdn_sop(mod):
    def scrub(x, h, up, down):
        yr = upfirdn_naive(x, h, up, down)
        want_len = _output_len(len(h), len(x), up, down)
        y = mod.upfirdn(h, x, up, down)
        assert len(yr) == want_len and len(y) == want_len
        xp_assert_close(y, yr)

    def case(up, down, h):
        """scipy/signal/tests/test_upfirdn.py::UpFIRDnCase (1-D, float64)."""
        h = np.atleast_1d(np.asarray(h, dtype=float))
        rng = np.random.RandomState(17)
        scrub(np.ones(1), h, up, down)
        scrub(np.ones(10), h, up, down)
        scrub(rng.randn(10), h, up, down)
        scrub(np.arange(10).astype(float), h, up, down)

    def test_singleton():
        """::test_singleton."""
        for len_h in [1, 2, 3, 4, 5]:
            for len_x in [1, 2, 3, 4, 5]:
                h = np.zeros(len_h)
                h[len_h // 2] = 1.0
                x = np.ones(len_x)
                want = np.pad(x, (len_h // 2, (len_h - 1) // 2), "constant")
                xp_assert_close(mod.upfirdn(h, x, 1, 1), want)

    def test_shift_x():
        """::test_shift_x."""
        xp_assert_close(mod.upfirdn([1, 1], [1.0], 1, 1), [1.0, 1.0])
        xp_assert_close(mod.upfirdn([1, 1], [0.0, 1.0], 1, 1), [0.0, 1.0, 1.0])

    def test_length_factors():
        """::test_length_factors (expected values from MATLAB)."""
        for len_h, len_x, up, down, expected in [(2, 2, 5, 2, [1, 0, 0, 0]), (2, 3, 6, 3, [1, 0, 1, 0, 1]),
                                                 (2, 4, 4, 3, [1, 0, 0, 0, 1]), (3, 2, 6, 2, [1, 0, 0, 1, 0]),
                                                 (4, 11, 3, 5, [1, 0, 0, 1, 0, 0, 1])]:
            h = np.zeros(len_h)
            h[0] = 1.0
            xp_assert_close(mod.upfirdn(h, np.ones(len_x), up, down), np.asarray(expected, dtype=float))

    def test_vs_convolve():
        """::test_vs_convolve (float64; lengths from MATLAB)."""
        for down, want_len in [(2, 5015), (11, 912), (79, 127)]:
            rng = np.random.RandomState(17)
            x = rng.randn(10000)
            h = ss.firwin(31, 1.0 / down, window="hamming")
            yl = upfirdn_naive(x, h, 1, down)
            y = mod.upfirdn(h, x, 1, down)
            assert y.shape == (want_len,) and yl.shape[0] == y.shape[0]
            xp_assert_close(yl, y, atol=1e-7, rtol=1e-7)

    def test_vs_naive_delta():
        """::test_vs_naive_delta (float64, h = 1)."""
        for up, down in [(1, 1), (2, 2), (3, 2), (2, 3)]:
            case(up, down, 1.0)

    def test_vs_naive():
        """::test_vs_naive (float64; _random_factors draws a single-tap h, as in the original)."""
        for p_max, q_max in [(10, 10), (10, 100), (100, 10), (100, 100)]:
            rng = np.random.RandomState(17)
            for _ in range(3):
                p_add = q_max if p_max > q_max else 1
                q_add = p_max if q_max > p_max else 1
                p = rng.randint(p_max) + p_add
                q = rng.randint(q_max) + q_add
                len_h = rng.randint(25) + 1
                h = np.atleast_1d(rng.randint(len_h)).astype(float)
                case(p, q, h)

    def test_resample_methods():
        """scipy/signal/tests/test_signaltools.py::TestResample::test_resample_methods (polyphase, constant)."""
        rate = 100
        t = np.arange(rate) / float(rate)
        freqs = np.array([1.0, 10.0, 40.0])
        for rate_to in [49, 50, 51, 99, 100, 101, 199, 200, 201]:
            t_to = np.arange(rate_to) / float(rate_to)
            for f in freqs:
                x = np.sin(2 * np.pi * f * t) * ss.windows.hann(rate)
                y_to = np.sin(2 * np.pi * f * t_to) * ss.windows.hann(rate_to)
                y = resample_poly(mod, x, rate_to, rate)
                if f >= 0.5 * rate_to:
                    xp_assert_close(y, np.zeros_like(y_to), atol=1e-3)
                else:
                    assert y_to.shape == y.shape
                    assert np.corrcoef(y_to, y)[0, 1] > 0.99
        rng = np.random.RandomState(0)
        x = ss.windows.hann(rate) * np.cumsum(rng.randn(rate))
        for rate_to in [49, 50, 51, 99, 100, 101, 199, 200, 201]:
            t_to = np.arange(rate_to) / float(rate_to)
            y_to = np.interp(t_to, t, x)
            y = resample_poly(mod, x, rate_to, rate)
            assert y_to.shape == y.shape
            assert np.corrcoef(y_to, y)[0, 1] > 0.99

    def test_poly_vs_filtfilt():
        """::TestResample::test_poly_vs_filtfilt (float64)."""
        for down_factor in [2, 11, 79]:
            rng = np.random.RandomState(17)
            x = rng.randn(10000)
            x[0] = 0
            x[-1] = 0
            h = ss.firwin(31, 1.0 / down_factor, window="hamming")
            yf = ss.filtfilt(h, 1.0, x, padtype="constant")[::down_factor]
            hc = np.convolve(h, np.flip(h))
            y = resample_poly(mod, x, 1, down_factor, window=hc)
            xp_assert_close(yf, y, atol=3e-7, rtol=6e-7)

    def test_correlate1d():
        """::TestResample::test_correlate1d."""
        from scipy.ndimage import correlate1d

        rng = np.random.RandomState(3)
        for down in [2, 4]:
            for nx in range(1, 40, down):
                for nweights in (32, 33):
                    x = rng.random_sample(nx)
                    weights = rng.random_sample(nweights)
                    y_g = correlate1d(x, np.flip(weights), mode="constant")
                    y_s = resample_poly(mod, x, 1, down, window=weights)
                    xp_assert_close(y_g[::down], y_s)

    return [test_singleton, test_shift_x, test_length_factors, test_vs_convolve, test_vs_naive_delta,
            test_vs_naive, test_resample_methods, test_poly_vs_filtfilt, test_correlate1d]


def upfirdn_oracle(mod):
    rng = np.random.default_rng(13)
    outs = []
    for up, down in [(1, 1), (2, 3), (3, 2), (1, 2), (3, 1), (5, 4), (1, 7), (4, 4)]:
        for len_h in [1, 2, 3, 5, 25, 61]:
            h = rng.standard_normal(len_h)
            for n in [1, 2, 3, 4, 7, 10, 41, 100]:
                x = rng.standard_normal(n)
                outs.append(("upfirdn", n, mod.upfirdn(h, x, up, down), ss.upfirdn(h, x, up, down)))
                # default arguments, which callers (and the SciPy suite) rely on
                if down == 1:
                    outs.append(("upfirdn_default_down", n, mod.upfirdn(h, x, up), ss.upfirdn(h, x, up)))
                if up == down == 1:
                    outs.append(("upfirdn_defaults", n, mod.upfirdn(h, x), ss.upfirdn(h, x)))
    return outs


# --------------------------------------------------------------------------
# Subject 4: overlap-add block FIR
# --------------------------------------------------------------------------

H16 = ss.firwin(16, 0.25)


def ola_decls(mod):
    stream = dp.streaming()(lambda: mod.OverlapAdd(H16))
    oneshot = dp.lti_filter(transfer=(H16, [1.0]))(lambda x: mod.OverlapAdd(H16).process(x))
    return {"ola_streaming": stream, "ola_as_fir": oneshot}


def ola_sop(mod):
    rng = np.random.RandomState(5)

    def test_impulse_response():
        """The response to an impulse is the filter."""
        x = np.zeros(32)
        x[0] = 1.0
        assert_allclose(mod.OverlapAdd(H16).process(x)[:16], H16)

    def test_vs_convolve():
        """One call equals np.convolve truncated to the input length."""
        x = rng.randn(1000)
        assert_allclose(mod.OverlapAdd(H16).process(x), np.convolve(x, H16)[:1000])

    def test_blocks_of_64():
        """Blocks of 64 samples give the same output as one call."""
        x = rng.randn(1024)
        proc = mod.OverlapAdd(H16)
        y = np.concatenate([proc.process(x[i:i + 64]) for i in range(0, 1024, 64)])
        assert_allclose(y, mod.OverlapAdd(H16).process(x))

    def test_two_unequal_blocks():
        """Two blocks of unequal length give the same output as one call."""
        x = rng.randn(300)
        proc = mod.OverlapAdd(H16)
        y = np.concatenate([proc.process(x[:117]), proc.process(x[117:])])
        assert_allclose(y, np.convolve(x, H16)[:300])

    return [test_impulse_response, test_vs_convolve, test_blocks_of_64, test_two_unequal_blocks]


def ola_oracle(mod):
    rng = np.random.default_rng(17)
    outs = []
    for taps in [1, 2, 5, 16]:
        h = rng.standard_normal(taps)
        for n in list(range(0, 12)) + [40, 257]:
            x = rng.standard_normal(n)
            ref = np.convolve(x, h)[:n] if n else np.zeros(0)
            for trial in range(4):
                cuts = sorted(rng.integers(0, n + 1, size=trial * 2))
                edges = [0, *cuts, n]
                proc = mod.OverlapAdd(h)
                parts = [proc.process(x[i:j]) for i, j in zip(edges[:-1], edges[1:])]
                outs.append(("blocks", n, np.concatenate(parts) if parts else np.zeros(0), ref))
    return outs


SUBJECTS = {
    "fft": (fft_decls, fft_sop, fft_oracle),
    "lfilter": (lfilter_decls, lfilter_sop, lfilter_oracle),
    "upfirdn": (upfirdn_decls, upfirdn_sop, upfirdn_oracle),
    "ola": (ola_decls, ola_sop, ola_oracle),
}

# Declarations of the approach dsp-props/params (operator families, plan action 2.4).
FAMILY_DECLS = {"lfilter": lfilter_family_decls, "upfirdn": upfirdn_family_decls}

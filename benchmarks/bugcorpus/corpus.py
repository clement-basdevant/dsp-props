"""The rediscovery corpus: real bugs of released DSP libraries, and the declarations that test them.

Each ``Bug`` names the last release that has it and the first release (or nightly build) that fixes
it, both installed from wheels into isolated environments (``remote.Environment``). The bugs are the
in-scope rows of ``triage.tsv`` (see ``docs/rediscovery.md`` for the protocol).

Declarations are written per *function*, not per bug, from the function's documentation: the
parameters a user would choose are drawn by a family (``dp.family``) over their documented range,
and only what the documentation states is declared (no stopband claim when the documentation gives
no attenuation, for instance). Every bug of a function is tested with the same declaration, on the
faulty and on the fixed release, so that a failure that is still there after the fix (a declaration
the fixed release does not meet either) does not count as a rediscovery.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.signal as scipy_signal
from hypothesis import strategies as st

import dsp_props as dp
from dsp_props.core.iir import IIRTolerance
from dsp_props.generators import fir_taps, iir_coefficients, rate_pairs

from .remote import Environment

PYTORCH_CPU = "https://download.pytorch.org/whl/cpu"
SCIPY_NIGHTLY = "https://pypi.anaconda.org/scientific-python-nightly-wheels/simple"


def scipy_env(python, scipy, numpy):
    return Environment(python, (f"numpy=={numpy}", f"scipy=={scipy}"))


def scipy_nightly(python="3.13", numpy="2.5.3"):
    return Environment(python, (f"numpy=={numpy}", "scipy"),
                       extra=("--index-url", SCIPY_NIGHTLY, "--extra-index-url", "https://pypi.org/simple",
                              "--prerelease", "allow", "--index-strategy", "unsafe-best-match"))


LIBROSA_DEPS = {
    "0.7.2": ("numpy==1.18.5", "scipy==1.4.1", "numba==0.48.0", "llvmlite==0.31.0", "resampy==0.2.2",
              "scikit-learn==0.23.2", "joblib==0.17.0", "decorator==4.4.2", "audioread==2.1.9",
              "soundfile==0.10.3.post1", "librosa==0.7.2"),
    "0.8.0": ("numpy==1.18.5", "scipy==1.4.1", "numba==0.48.0", "llvmlite==0.31.0", "resampy==0.2.2",
              "scikit-learn==0.23.2", "joblib==0.17.0", "decorator==4.4.2", "audioread==2.1.9",
              "soundfile==0.10.3.post1", "pooch==1.2.0", "librosa==0.8.0"),
    "0.8.1": ("numpy==1.20.3", "scipy==1.6.3", "numba==0.53.1", "llvmlite==0.36.0", "resampy==0.2.2",
              "scikit-learn==0.24.2", "joblib==1.0.1", "decorator==5.0.9", "audioread==2.1.9",
              "soundfile==0.10.3.post1", "pooch==1.4.0", "librosa==0.8.1"),
    "0.9.0": ("numpy==1.20.3", "scipy==1.6.3", "numba==0.53.1", "llvmlite==0.36.0", "resampy==0.2.2",
              "scikit-learn==0.24.2", "joblib==1.0.1", "decorator==5.0.9", "audioread==2.1.9",
              "soundfile==0.10.3.post1", "pooch==1.6.0", "librosa==0.9.0"),
}


def librosa_env(version):
    return Environment("3.8", LIBROSA_DEPS[version])


# --------------------------------------------------------------------------------------------
# Declarations, one per function. Each takes a ``remote.Remote`` and returns declared operators
# (a dict name -> operator). The remote functions are defined once; the families close over them.
# --------------------------------------------------------------------------------------------

def _real_biquads(max_sections=3):
    """Stable second-order sections (b, a padded to 3 coefficients), as an (n, 6) array."""
    def to_section(ba):
        b, a = (np.asarray(c, float) / ba[1][0] for c in ba)  # sosfilt wants a0 = 1
        return np.concatenate([np.pad(b, (0, 3 - len(b))), np.pad(a, (0, 3 - len(a)))])

    section = iir_coefficients(max_order=2, max_numerator=3).map(to_section)
    return st.lists(section, min_size=1, max_size=max_sections).map(np.array)


def declare_upfirdn(r):
    """scipy.signal.upfirdn(h, x, up, down): upsample, FIR filter, downsample.

    Linear and L/M periodically time-invariant; the documented output is the full convolution of
    the upsampled input with h, downsampled: ceil(((N - 1) up + len(h)) / down) samples.
    """
    call = r.fn("lambda h, x, up, down: scipy.signal.upfirdn(h, x, up, down)", "upfirdn")

    @dp.family(st.tuples(fir_taps(16), rate_pairs(8)))
    def upfirdn(h, rates):
        up, down = rates
        length = lambda n: -(-((n - 1) * up + len(h)) // down)  # noqa: E731
        return dp.resampler(up, down, causal=True, length=length, dc_gain=None, stopband=None)(
            lambda x: call(h, x, up, down))

    return {"upfirdn": upfirdn}


DECIMATE_FILTERS = ["iir", "fir", "dlti-fir", "dlti-iir"]


def declare_decimate(r):
    """scipy.signal.decimate(x, q, ftype=..., zero_phase=False).

    A low-pass filter then keeping every q-th sample: linear, q-periodically time-invariant, causal
    when ``zero_phase=False``, unit DC gain, ceil(N / q) samples. ``ftype`` is drawn among the
    documented choices: 'iir', 'fir', or a ``dlti`` instance (a FIR from ``firwin`` or an IIR from
    ``cheby1`` in zeros-poles-gain form, as a user would build one; the transfer-function form of an
    order-8 filter is ill-conditioned, which ``scipy.signal`` warns about). ``zero_phase=True``, the default, is left out: its
    ``filtfilt`` pads the edges with an odd extension of the signal, so a delayed input does not
    give a delayed output and the resampler declaration does not hold (see the documentation).
    The documentation gives no stopband attenuation, so none is declared.
    """
    r.setup("import scipy.signal as ss")
    call = r.fn(
        "lambda x, q, kind: ss.decimate(x, q, ftype=("
        "    kind if kind in ('iir', 'fir') else"
        "    ss.dlti(ss.firwin(10 * q + 1, 0.8 / q), [1.0]) if kind == 'dlti-fir' else"
        "    ss.dlti(*ss.cheby1(8, 0.05, 0.8 / q, output='zpk'))), zero_phase=False)",
        "decimate")

    @dp.family(st.tuples(st.integers(2, 8), st.sampled_from(DECIMATE_FILTERS)))
    def decimate(q, kind):
        # The IIR choices use the documented filter (Chebyshev I, order 8, 0.05 dB, cutoff 0.8 / q),
        # whose poles sit close to the unit circle: the tolerance is the library's error model for
        # that recursion, as for any declared IIR filter, not the FIR default.
        tolerance = IIRTolerance(*scipy_signal.cheby1(8, 0.05, 0.8 / q)) if kind in ("iir", "dlti-iir") else None
        return dp.resampler(1, q, causal=True, length="ceil", dc_gain=1.0, ripple=0.02, stopband=None,
                            tolerance=tolerance)(lambda x: call(x, q, kind))

    return {"decimate": decimate}


def declare_dst(r):
    """scipy.fft.dst(x, type, norm): linear, inverted by idst with the same type and norm; with
    norm='ortho' the transform is orthonormal (Parseval with scale 1)."""
    fwd = r.fn("lambda x, t, norm: scipy.fft.dst(x, type=t, norm=norm)", "dst")
    inv = r.fn("lambda X, t, norm: scipy.fft.idst(X, type=t, norm=norm)", "idst")

    @dp.family(st.tuples(st.sampled_from([1, 2, 3, 4]), st.sampled_from([None, "ortho"])))
    def dst(t, norm):
        return dp.transform(inverse=lambda X: inv(X, t, norm), kind="generic", parseval=norm == "ortho",
                            parseval_scale=lambda n: 1.0, min_n=2 if t == 1 else 1)(lambda x: fwd(x, t, norm))

    return {"dst": dst}


def declare_fht(r):
    """scipy.fft.fht(a, dln, mu, offset, bias): linear in a real input, inverted by ifht with the
    same parameters ("This is the inverse operation to fht")."""
    fwd = r.fn("lambda a, dln, mu: scipy.fft.fht(a, dln, mu)", "fht")
    inv = r.fn("lambda A, dln, mu: scipy.fft.ifht(A, dln, mu)", "ifht")

    @dp.family(st.tuples(st.sampled_from([0.01, 0.05, 0.1]), st.sampled_from([0.0, 0.5, 1.0, 2.0])))
    def fht(dln, mu):
        return dp.transform(inverse=lambda A: inv(A, dln, mu), kind="generic", real_input_only=True)(
            lambda a: fwd(a, dln, mu))

    return {"fht": fht}


def declare_resample(r):
    """scipy.signal.resample(x, num) for num = L N: Fourier interpolation, linear, real or complex
    input; truncating the spectrum back (resample to N samples) recovers x exactly."""
    call = r.fn("lambda x, num: scipy.signal.resample(x, num)", "resample")

    @dp.family(st.sampled_from([2, 3, 4]))
    def resample(L):
        return dp.transform(inverse=lambda Y: call(Y, len(Y) // L), kind="generic")(
            lambda x: call(x, L * len(x)))

    return {"resample": resample}


def declare_lfilter(r):
    """scipy.signal.lfilter(b, a, x, zi=z) in blocks, the documented streaming use (return zf)."""
    call = r.fn("lambda b, a, x, zi: scipy.signal.lfilter(b, a, x, zi=zi)", "lfilter")

    @dp.family(iir_coefficients(max_order=4))
    def lfilter(b, a):
        order = max(len(a), len(b)) - 1  # zi has max(len(a), len(b)) - 1 entries (none for a gain)
        return dp.stateful(lambda blk, z: call(b, a, blk, z), init_state=np.zeros(order))

    return {"lfilter (streaming)": lfilter}


def declare_sosfilt(r):
    """scipy.signal.sosfilt(sos, x, zi=z) in blocks (return zf), stable second-order sections."""
    call = r.fn("lambda sos, x, zi: scipy.signal.sosfilt(sos, x, zi=zi)", "sosfilt")

    @dp.family(_real_biquads())
    def sosfilt(sos):
        return dp.stateful(lambda blk, z: call(sos, blk, z), init_state=np.zeros((len(sos), 2)))

    return {"sosfilt (streaming)": sosfilt}


def declare_short_time_fft(r):
    """scipy.signal.ShortTimeFFT(win, hop, fs=1).stft(x), inverted by .istft(S, k1=N) when the
    window and hop satisfy the NOLA condition (a periodic Hann window, hop <= m / 2). The default
    ``fft_mode='onesided'`` takes real signals only; the input needs at least ceil(m / 2) samples."""
    r.setup("import scipy.signal as ss")
    fwd = r.fn("lambda x, m, hop: ss.ShortTimeFFT(ss.windows.hann(m, sym=False), hop, fs=1.0).stft(x)", "stft")
    inv = r.fn("lambda S, m, hop, n: ss.ShortTimeFFT(ss.windows.hann(m, sym=False), hop, fs=1.0).istft(S, k1=n)",
               "istft")

    @st.composite
    def window_and_hop(draw):
        m = draw(st.integers(4, 32))
        return m, draw(st.integers(1, m // 2))

    @dp.family(window_and_hop())
    def short_time_fft(m, hop):
        return dp.transform(inverse=lambda S, n: inv(S, m, hop, n), kind="generic", inverse_takes_length=True,
                            real_input_only=True, min_n=-(-m // 2))(lambda x: fwd(x, m, hop))

    return {"ShortTimeFFT": short_time_fft}


def declare_librosa_stft(r):
    """librosa.stft(y, n_fft, hop_length) (center=True, Hann window), inverted by
    librosa.istft(D, hop_length, length=N) (``length`` trims the padding); hop <= n_fft / 4,
    the overlap the documentation recommends. Real input only; N >= n_fft. The output types are
    asked for explicitly (``dtype``): the defaults of librosa 0.7 are single precision."""
    fwd = r.fn("lambda y, n_fft, hop: __import__('librosa').stft(y, n_fft=n_fft, hop_length=hop,"
               " dtype=np.complex128)", "stft")
    inv = r.fn("lambda D, hop, n: __import__('librosa').istft(D, hop_length=hop, length=n, dtype=np.float64)",
               "istft")

    @st.composite
    def frame_and_hop(draw):
        n_fft = 2 * draw(st.integers(2, 32))
        return n_fft, draw(st.integers(1, n_fft // 4))

    @dp.family(frame_and_hop())
    def stft(n_fft, hop):
        return dp.transform(inverse=lambda D, n: inv(D, hop, n), kind="generic", real_input_only=True,
                            inverse_takes_length=True, min_n=n_fft)(lambda y: fwd(y, n_fft, hop))

    return {"stft": stft}


def declare_preemphasis(r):
    """librosa.effects.preemphasis(y, coef): y[n] = x[n] - coef x[n-1] (the documented filter),
    called as most users do (default initial state), and in blocks with zi / return_zf."""
    one = r.fn("lambda y, c: __import__('librosa').effects.preemphasis(y, coef=c)", "preemphasis")
    block = r.fn("lambda y, c, zi: __import__('librosa').effects.preemphasis(y, coef=c, zi=zi, return_zf=True)",
                 "preemphasis_block")
    coefs = st.sampled_from([0.0, 0.5, 0.9, 0.97, 1.0])

    @dp.family(coefs)
    def preemphasis(c):
        return dp.lti_filter(transfer=([1.0, -c], [1.0]), min_n=2)(lambda y: one(y, c))

    @dp.family(coefs)
    def preemphasis_streaming(c):
        return dp.stateful(lambda blk, z: block(blk, c, z), init_state=np.zeros(1))

    return {"preemphasis": preemphasis, "preemphasis (streaming)": preemphasis_streaming}


def declare_deemphasis(r):
    """librosa.effects.deemphasis(y, coef): the inverse of preemphasis with the same coefficient
    (the documented example: deemphasis(preemphasis(y)) is y), linear."""
    de = r.fn("lambda y, c: __import__('librosa').effects.deemphasis(y, coef=c)", "deemphasis")
    pre = r.fn("lambda y, c: __import__('librosa').effects.preemphasis(y, coef=c)", "preemphasis")

    @dp.family(st.sampled_from([0.0, 0.5, 0.9, 0.97]))
    def deemphasis(c):
        return dp.transform(inverse=lambda y: pre(y, c), kind="generic", real_input_only=True, min_n=2)(
            lambda y: de(y, c))

    return {"deemphasis": deemphasis}


DECLARATIONS = {
    "upfirdn": declare_upfirdn,
    "decimate": declare_decimate,
    "dst": declare_dst,
    "fht": declare_fht,
    "resample": declare_resample,
    "lfilter": declare_lfilter,
    "sosfilt": declare_sosfilt,
    "ShortTimeFFT": declare_short_time_fft,
    "librosa.stft": declare_librosa_stft,
    "preemphasis": declare_preemphasis,
    "deemphasis": declare_deemphasis,
}


@dataclass(frozen=True)
class Bug:
    key: str
    library: str
    function: str  # key of DECLARATIONS
    title: str
    refs: tuple[str, ...]  # upstream issues and pull requests
    fix_commit: str
    faulty: Environment
    fixed: Environment
    fixed_label: str  # "1.5.0", or "nightly" when no release has the fix yet
    trigger: str  # what the fix says triggers the bug, from the commit and the issue
    notes: str = ""
    ours: bool = False  # found by dsp-props itself before this campaign (circular, reported apart)
    tags: tuple[str, ...] = field(default_factory=tuple)


BUGS = [
    Bug("scipy-upfirdn-length", "scipy", "upfirdn", "upfirdn output length computed from the padded filter",
        ("scipy/scipy#12159",), "6bc1c1c6df",
        scipy_env("3.8", "1.4.1", "1.18.5"), scipy_env("3.8", "1.5.0", "1.18.5"), "1.5.0",
        "up > 1 and len(h) not a multiple of up: the length used the filter padded to a multiple of up"),
    Bug("scipy-decimate-iir-stability", "scipy", "decimate", "decimate's IIR filter in (b, a) form is unstable",
        ("scipy/scipy#16096",), "010196647b",
        scipy_env("3.8", "1.9.3", "1.23.5"), scipy_env("3.8", "1.10.0", "1.23.5"), "1.10.0",
        "default ftype='iir' (Chebyshev order 8) applied as a transfer function: inaccurate for large q"),
    Bug("scipy-decimate-dlti", "scipy", "decimate", "decimate mishandles a user-supplied dlti filter",
        ("scipy/scipy#17845", "scipy/scipy#17881"), "cf1999530f",
        scipy_env("3.9", "1.10.1", "1.24.4"), scipy_env("3.9", "1.11.0", "1.24.4"), "1.11.0",
        "ftype given as a dlti instance: a FIR went through zpk2sos, complex coefficients through sosfilt"),
    Bug("scipy-dst-ortho", "scipy", "dst", "norm='ortho' wrong for DST types 2 and 3",
        ("scipy/scipy#18452", "scipy/scipy#18613"), "299df543da",
        scipy_env("3.10", "1.11.4", "1.26.4"), scipy_env("3.10", "1.12.0", "1.26.4"), "1.12.0",
        "dst/idst type 2 or 3 with norm='ortho': wrong scaling of one coefficient"),
    Bug("scipy-fht-odd", "scipy", "fht", "fht zeroes an imaginary part that exists for odd n",
        ("scipy/scipy#21668",), "170f9e69a5",
        scipy_env("3.12", "1.14.1", "1.26.4"), scipy_env("3.12", "1.15.0", "1.26.4"), "1.15.0",
        "fht with an odd number of points"),
    Bug("scipy-resample-complex", "scipy", "resample", "resample: unpaired Nyquist bin and complex signals",
        ("scipy/scipy#14569", "scipy/scipy#22814"), "34cf1d9e16",
        scipy_env("3.12", "1.15.3", "2.2.6"), scipy_env("3.12", "1.16.0", "2.2.6"), "1.16.0",
        "resample of an even-length signal (unpaired Nyquist bin), complex input, num=2"),
    Bug("scipy-lfilter-empty", "scipy", "lfilter", "lfilter raises on an empty block with an initial state",
        ("scipy/scipy#22571",), "c1badd5cc6",
        scipy_env("3.13", "1.18.1", "2.5.3"), scipy_nightly(), "nightly",
        "lfilter(b, a, x, zi=zi) with len(x) == 0 (FIR and IIR paths)"),
    Bug("scipy-sosfilt-empty", "scipy", "sosfilt", "sosfilt raises on an empty block with an initial state",
        ("scipy/scipy#26406", "scipy/scipy#26415"), "5ccd743e5e",
        scipy_env("3.13", "1.18.1", "2.5.3"), scipy_nightly(), "nightly",
        "sosfilt(sos, x, zi=zi) with len(x) == 0", ours=True,
        notes="reported by us after a dsp-props campaign (October 2026): circular, counted apart"),
    Bug("scipy-decimate-complex-zeros", "scipy", "decimate", "decimate checks the poles twice, not the zeros",
        ("scipy/scipy#17881",), "8e2ddfdfe3",
        scipy_env("3.13", "1.18.1", "2.5.3"), scipy_nightly(), "nightly",
        "a dlti filter with real poles and complex zeros"),
    Bug("scipy-shorttimefft-padding", "scipy", "ShortTimeFFT", "ShortTimeFFT pads short signal slices wrongly",
        ("scipy/scipy#26418",), "a11c360740",
        scipy_env("3.13", "1.18.1", "2.5.3"), scipy_nightly(), "nightly",
        "a slice whose padded part is longer than the signal part (short signals or slices)"),
    Bug("librosa-preemphasis-glitch", "librosa", "preemphasis", "preemphasis warm-up glitch",
        ("librosa/librosa#1032", "librosa/librosa#1159"), "478be6f",
        librosa_env("0.7.2"), librosa_env("0.8.0"), "0.8.0",
        "default zi = lfilter_zi(b, a), the steady state of a unit step: a constant offset on y[0]"),
    Bug("librosa-stft-large-window", "librosa", "librosa.stft", "stft fails for large window sizes",
        ("librosa/librosa#1094",), "a391b30",
        librosa_env("0.7.2"), librosa_env("0.8.0"), "0.8.0",
        "n_fft so large that one frame exceeds librosa's memory block (n_fft above about 2^15)"),
    Bug("librosa-deemphasis-inplace", "librosa", "deemphasis", "deemphasis modifies its input in place",
        ("librosa/librosa#1362",), "c31f2f2",
        librosa_env("0.8.1"), librosa_env("0.9.0"), "0.9.0",
        "any call: y[0] -= zi writes into the caller's array"),
]

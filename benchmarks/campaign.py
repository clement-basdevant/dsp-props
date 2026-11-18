"""Bug-hunting campaign: run dsp-props relations against real libraries.

Usage: python benchmarks/campaign.py [--examples N] [--only SUBSTRING]

Each target is a (name, check, strategy) triple. Failures are reported with
the minimal input found by Hypothesis.
"""

import argparse
import pathlib
import sys
import time
import traceback

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import numpy as np  # noqa: E402
from hypothesis import HealthCheck, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

import dsp_props as dp  # noqa: E402
from dsp_props.core import FFTTolerance, assert_close, norm  # noqa: E402
from dsp_props.generators import block_splits, lengths, signals, split_blocks  # noqa: E402

TOL = FFTTolerance()
TARGETS = []


def target(name, strategy):
    def register(check):
        TARGETS.append((name, check, strategy))
        return check

    return register


@st.composite
def sig(draw, max_n=256, min_n=1):
    n = draw(lengths(max_n).filter(lambda k: k >= min_n))
    return draw(signals(n))


@st.composite
def sig_pair(draw, max_n=256, min_n=1, same_length=False):
    x = draw(sig(max_n, min_n))
    y = draw(signals(len(x))) if same_length else draw(sig(max_n, min_n))
    return x, y


@st.composite
def stable_iir(draw, max_order=6):
    """Random stable IIR filter (poles inside radius 0.97)."""
    order = draw(st.integers(1, max_order))
    radii = draw(st.lists(st.floats(0.0, 0.97), min_size=order, max_size=order))
    angles = draw(st.lists(st.floats(0.0, np.pi), min_size=order, max_size=order))
    poles = []
    for r, t in zip(radii, angles):
        poles += [r * np.exp(1j * t), r * np.exp(-1j * t)]
    a = np.real(np.poly(poles))
    b = np.array(draw(st.lists(st.floats(-1, 1), min_size=1, max_size=2 * order + 1)))
    if not np.any(b):
        b[0] = 1.0
    return b, a


# --------------------------------------------------------------------------
# Transforms: reuse the library relations through @dp.transform
# --------------------------------------------------------------------------

def _transform_targets():
    import scipy.fft as sfft
    import torch

    def t_fft(x):
        return torch.fft.fft(torch.from_numpy(np.asarray(x))).numpy()

    def t_ifft(X):
        return torch.fft.ifft(torch.from_numpy(np.asarray(X))).numpy()

    out = {
        "scipy.fft.fft": dp.transform(inverse=sfft.ifft, parseval=True, real_input_hermitian=True)(
            lambda x: sfft.fft(x)),
        "torch.fft.fft (float64)": dp.transform(inverse=t_ifft, parseval=True, real_input_hermitian=True)(t_fft),
        "scipy.fft.dct-II ortho": dp.transform(
            kind="generic", inverse=lambda X: sfft.idct(X, norm="ortho"), parseval=True,
            parseval_scale=lambda n: 1.0, real_input_only=True)(lambda x: sfft.dct(x, norm="ortho")),
        "scipy.fft.dst-II ortho": dp.transform(
            kind="generic", inverse=lambda X: sfft.idst(X, norm="ortho"), parseval=True,
            parseval_scale=lambda n: 1.0, real_input_only=True)(lambda x: sfft.dst(x, norm="ortho")),
        "scipy.fft.dct-IV ortho": dp.transform(
            kind="generic", inverse=lambda X: sfft.idct(X, type=4, norm="ortho"), parseval=True,
            parseval_scale=lambda n: 1.0, real_input_only=True)(lambda x: sfft.dct(x, type=4, norm="ortho")),
    }
    return out


# --------------------------------------------------------------------------
# Real FFTs: inversibility with explicit length, agreement with full FFT
# --------------------------------------------------------------------------

@target("scipy.fft.irfft(rfft(x), n) == x", sig())
def _(x):
    import scipy.fft as sfft
    assert_close(sfft.irfft(sfft.rfft(x), len(x)), x, TOL, what="rfft roundtrip")


@target("torch irfft(rfft(x), n) == x", sig())
def _(x):
    import torch
    t = torch.from_numpy(x)
    y = torch.fft.irfft(torch.fft.rfft(t), n=len(x)).numpy()
    assert_close(y, x, TOL, what="torch rfft roundtrip")


@target("scipy.fft.rfft == fft[:n//2+1]", sig())
def _(x):
    import scipy.fft as sfft
    full = sfft.fft(x)
    assert_close(sfft.rfft(x), full[: len(x) // 2 + 1], TOL, ref_norm=norm(full), what="rfft vs fft")


@target("float32 vs float64 fft (scipy)", sig())
def _(x):
    import scipy.fft as sfft
    x32 = x.astype(np.float32)
    if not np.all(np.isfinite(x32)) or norm(x32) < np.finfo(np.float32).tiny * 1e6:
        return  # out of float32 range: not a meaningful comparison
    y32 = sfft.fft(x32)
    assert y32.dtype == np.complex64, f"dtype {y32.dtype}"
    ref = sfft.fft(x32.astype(np.float64))
    assert_close(y32.astype(np.complex128), ref, FFTTolerance(c=20 * np.finfo(np.float32).eps / np.finfo(np.float64).eps), what="fft float32")


@target("float32 vs float64 fft (torch)", sig())
def _(x):
    import torch
    x32 = x.astype(np.float32)
    if not np.all(np.isfinite(x32)) or norm(x32) < np.finfo(np.float32).tiny * 1e6:
        return
    y32 = torch.fft.fft(torch.from_numpy(x32)).numpy()
    ref = np.fft.fft(x32.astype(np.float64))
    assert_close(y32.astype(np.complex128), ref, FFTTolerance(c=20 * np.finfo(np.float32).eps / np.finfo(np.float64).eps), what="torch fft float32")


@target("scipy.signal.czt default == fft", sig())
def _(x):
    import scipy.signal as ss
    full = np.fft.fft(x)
    assert_close(ss.czt(x), full, FFTTolerance(c=200), ref_norm=norm(full), what="czt vs fft")


@target("scipy.signal.zoom_fft full band == fft", sig(min_n=2))
def _(x):
    import scipy.signal as ss
    full = np.fft.fft(x)
    assert_close(ss.zoom_fft(x, 2.0, endpoint=False), full, FFTTolerance(c=200), ref_norm=norm(full), what="zoom_fft vs fft")


# --------------------------------------------------------------------------
# Convolution: fast methods vs direct sum
# --------------------------------------------------------------------------

def _conv_ref(x, y, out):
    return np.sqrt(len(out)) * norm(x) * norm(y)


for _mode in ["full", "same", "valid"]:
    for _fn in ["fftconvolve", "oaconvolve"]:

        @target(f"scipy.signal.{_fn}(mode={_mode}) == direct", sig_pair())
        def _(pair, fn=_fn, mode=_mode):
            import scipy.signal as ss
            x, y = pair
            if mode == "valid" and len(x) < len(y):
                x, y = y, x
            direct = ss.convolve(x, y, mode=mode, method="direct")
            fast = getattr(ss, fn)(x, y, mode=mode)
            assert_close(fast, direct, TOL, ref_norm=_conv_ref(x, y, direct), what=f"{fn} {mode}")


@target("scipy.signal.correlate == convolve(x, reversed y)", sig_pair())
def _(pair):
    import scipy.signal as ss
    x, y = pair
    a = ss.correlate(x, y, mode="full", method="fft")
    b = ss.convolve(x, y[::-1], mode="full", method="direct")
    assert_close(a, b, TOL, ref_norm=_conv_ref(x, y, b), what="correlate")


# --------------------------------------------------------------------------
# Streaming: block processing with carried state == one-shot processing
# --------------------------------------------------------------------------

@st.composite
def stream_case(draw):
    x = draw(sig(256))
    return x, draw(stable_iir()), draw(block_splits(len(x)))


@target("scipy.signal.lfilter by blocks == one shot", stream_case())
def _(case):
    import scipy.signal as ss
    x, (b, a), cuts = case
    whole = ss.lfilter(b, a, x)
    zi = np.zeros(max(len(a), len(b)) - 1)
    parts = []
    for blk in split_blocks(x, cuts):
        y, zi = ss.lfilter(b, a, blk, zi=zi)
        parts.append(y)
    assert_close(np.concatenate(parts), whole, FFTTolerance(c=1000), what="lfilter blocks")


@target("scipy.signal.sosfilt by blocks == one shot", stream_case())
def _(case):
    import scipy.signal as ss
    x, (b, a), cuts = case
    sos = ss.tf2sos(b, a)
    whole = ss.sosfilt(sos, x)
    zi = np.zeros((sos.shape[0], 2))
    parts = []
    for blk in split_blocks(x, cuts):
        y, zi = ss.sosfilt(sos, blk, zi=zi)
        parts.append(y)
    assert_close(np.concatenate(parts), whole, FFTTolerance(c=1000), what="sosfilt blocks")


@target("scipy.signal.upfirdn by blocks (no resampling) == one shot", stream_case())
def _(case):
    import scipy.signal as ss
    x, (b, _a), cuts = case
    whole = ss.lfilter(b, [1.0], x)
    via = ss.upfirdn(b, x)[: len(x)]
    assert_close(via, whole, FFTTolerance(c=1000), what="upfirdn vs lfilter")


# --------------------------------------------------------------------------
# Time-frequency: STFT inversibility
# --------------------------------------------------------------------------

@target("scipy ShortTimeFFT: istft(stft(x)) == x", st.tuples(sig(256, min_n=16), st.sampled_from([8, 16, 32]), st.sampled_from([2, 4])))
def _(case):
    import scipy.signal as ss
    x, nperseg, frac = case
    win = ss.windows.hann(nperseg, sym=False)
    hop = nperseg // frac
    stft = ss.ShortTimeFFT(win, hop=hop, fs=1.0)
    S = stft.stft(x)
    y = stft.istft(S, k1=len(x))
    assert_close(y, x, FFTTolerance(c=200), what=f"ShortTimeFFT nperseg={nperseg} hop={hop}")


@target("torch istft(stft(x)) == x", st.tuples(sig(256, min_n=32), st.sampled_from([8, 16, 32]), st.sampled_from([2, 4])))
def _(case):
    import torch
    x, n_fft, frac = case
    t = torch.from_numpy(x)
    win = torch.hann_window(n_fft, dtype=torch.float64)
    hop = n_fft // frac
    S = torch.stft(t, n_fft, hop_length=hop, window=win, return_complex=True, center=True)
    y = torch.istft(S, n_fft, hop_length=hop, window=win, center=True, length=len(x)).numpy()
    assert_close(y, x, FFTTolerance(c=200), what=f"torch stft n_fft={n_fft} hop={hop}")


@target("scipy.signal.resample_poly(x, 1, 1) == x", sig())
def _(x):
    import scipy.signal as ss
    assert_close(ss.resample_poly(x, 1, 1), x, TOL, what="resample_poly identity")


@target("scipy.signal.decimate is linear", sig_pair(256, min_n=40, same_length=True))
def _(pair):
    import scipy.signal as ss
    x, y = pair
    lhs = ss.decimate(2 * x - 3 * y, 2)
    rhs = 2 * ss.decimate(x, 2) - 3 * ss.decimate(y, 2)
    ref = 2 * norm(ss.decimate(x, 2)) + 3 * norm(ss.decimate(y, 2))
    assert_close(lhs, rhs, FFTTolerance(c=1000), ref_norm=ref, what="decimate linearity")


# --------------------------------------------------------------------------
# Campaign 2: more scipy.signal, librosa, torchaudio
# --------------------------------------------------------------------------

F32 = FFTTolerance(c=200 * np.finfo(np.float32).eps / np.finfo(np.float64).eps)


def _f32_ok(x):
    x32 = np.asarray(x, dtype=np.float32)
    return bool(np.all(np.isfinite(x32))) and (norm(x32) == 0 or norm(x32) > 1e-30) and norm(x32) < 1e30


@target("scipy.signal.filtfilt(method='gust') commutes with time reversal", st.tuples(sig(256, min_n=40), stable_iir(3)))
def _(case):
    # The default padding method is not reversal-symmetric by design;
    # Gustafsson's initial conditions are.
    import scipy.signal as ss
    x, (b, a) = case
    y = ss.filtfilt(b, a, x, method="gust")
    yr = ss.filtfilt(b, a, x[::-1], method="gust")[::-1]
    assert_close(yr, y, FFTTolerance(c=1e4), ref_norm=max(norm(y), norm(x)), what="filtfilt reversal")


@st.composite
def iir_with_delay(draw):
    b, a = draw(stable_iir(2))
    delay = draw(st.integers(0, 3))
    return np.concatenate([np.zeros(delay), b]), a


@target("scipy.signal.sosfilt(tf2sos(b, a)) == lfilter(b, a)", st.tuples(sig(256), iir_with_delay()))
def _(case):
    import scipy.signal as ss
    x, (b, a) = case
    m = max(len(a), len(b))
    b, a = np.pad(b, (0, m - len(b))), np.pad(a, (0, m - len(a)))
    y1 = ss.lfilter(b, a, x)
    y2 = ss.sosfilt(ss.tf2sos(b, a), x)
    assert_close(y2, y1, FFTTolerance(c=1e5), ref_norm=max(norm(y1), norm(x)), what="sosfilt(tf2sos) vs lfilter")


@target("scipy.signal.resample(x, n) == x", sig())
def _(x):
    import scipy.signal as ss
    assert_close(ss.resample(x, len(x)), x, TOL, what="resample identity")


@target("scipy.signal.resample up x2 then down == x", sig())
def _(x):
    import scipy.signal as ss
    n = len(x)
    y = ss.resample(ss.resample(x, 2 * n), n)
    assert_close(y, x, TOL, what=f"resample 2n->n (n={n})")


@target("scipy.signal.hilbert: real part == x", sig())
def _(x):
    import scipy.signal as ss
    assert_close(np.real(ss.hilbert(x)), x, TOL, what="hilbert real part")


@target("scipy.signal.periodogram (boxcar, two-sided) sums to mean power", sig())
def _(x):
    import scipy.signal as ss
    f, p = ss.periodogram(x, window="boxcar", detrend=False, scaling="spectrum", return_onesided=False)
    lhs, rhs = float(np.sum(p)), float(np.mean(x ** 2))
    assert abs(lhs - rhs) <= TOL.bound(len(x), np.float64, rhs), f"sum(P)={lhs!r} mean(x^2)={rhs!r}"


@target("scipy.signal.periodogram one-sided == two-sided folded", sig(min_n=2))
def _(x):
    import scipy.signal as ss
    kw = dict(window="boxcar", detrend=False, scaling="spectrum")
    _, p1 = ss.periodogram(x, return_onesided=True, **kw)
    _, p2 = ss.periodogram(x, return_onesided=False, **kw)
    lhs, rhs = float(np.sum(p1)), float(np.sum(p2))
    assert abs(lhs - rhs) <= TOL.bound(len(x), np.float64, rhs), f"one-sided {lhs!r} vs two-sided {rhs!r} (n={len(x)})"


@target("scipy.signal.istft(stft(x)) == x (legacy API)", st.tuples(sig(256, min_n=16), st.sampled_from([8, 16, 32])))
def _(case):
    import scipy.signal as ss
    x, nperseg = case
    if len(x) < nperseg:
        return  # stft shrinks nperseg to len(x) with a warning; istft cannot know
    _, _, Z = ss.stft(x, nperseg=nperseg)
    _, y = ss.istft(Z, nperseg=nperseg)
    assert_close(y[: len(x)], x, FFTTolerance(c=200), what=f"stft/istft nperseg={nperseg}")


@target("scipy.signal.lfilter with zi: empty-block streaming via dp.stateful", st.tuples(stable_iir(3)))
def _(case):
    import scipy.signal as ss
    ((b, a),) = case
    nz = max(len(a), len(b)) - 1
    op = dp.stateful(lambda blk, z: ss.lfilter(b, a, blk, zi=z), lambda: np.zeros(nz), max_n=64)
    dp.verify(op, max_examples=20)


# ----- librosa -----

@target("librosa.istft(stft(x)) == x", st.tuples(sig(512, min_n=64), st.sampled_from([32, 64, 128]), st.sampled_from([2, 4]), st.booleans()))
def _(case):
    import librosa
    x, n_fft, frac, center = case
    if not _f32_ok(x):
        return
    x = x.astype(np.float32)
    hop = n_fft // frac
    if not center and len(x) < n_fft:
        return
    S = librosa.stft(x, n_fft=n_fft, hop_length=hop, center=center)
    y = librosa.istft(S, hop_length=hop, n_fft=n_fft, center=center, length=len(x))
    if center:
        assert_close(y, x, F32, what=f"librosa stft n_fft={n_fft} hop={hop}")
    else:
        # Without centering the first and last n_fft samples are not covered by full overlap.
        m = slice(n_fft, len(x) - n_fft)
        if m.start < m.stop:
            assert_close(y[m], x[m], F32, ref_norm=norm(x), what=f"librosa stft center=False n_fft={n_fft} hop={hop}")


@target("librosa.resample same rate == x", sig())
def _(x):
    import librosa
    y = librosa.resample(x, orig_sr=16000, target_sr=16000)
    assert_close(y, x, TOL, what="librosa resample identity")


@target("librosa.resample is linear", sig_pair(256, min_n=8, same_length=True))
def _(pair):
    import librosa
    x, y = pair
    r = lambda s: librosa.resample(s, orig_sr=22050, target_sr=16000, res_type="soxr_hq")
    lhs = r(2 * x - 3 * y)
    rhs = 2 * r(x) - 3 * r(y)
    # soxr "HQ" is a 20-bit-precision resampler by design: float32-level tolerance.
    if not (_f32_ok(x) and _f32_ok(y)):
        return
    assert_close(lhs, rhs, F32, ref_norm=2 * norm(r(x)) + 3 * norm(r(y)), what="librosa resample linearity")


@target("librosa preemphasis by blocks (zi/return_zf) == one shot", st.tuples(sig(256), st.floats(0.0, 0.99)))
def _(case):
    import librosa
    x, coef = case

    class Pre:
        def __init__(self):
            self.zf = None

        def process(self, blk):
            y, self.zf = librosa.effects.preemphasis(blk, coef=coef, zi=self.zf, return_zf=True)
            return y

    # librosa's default zi uses the first sample: give an explicit zero state instead
    class Pre0(Pre):
        def __init__(self):
            self.zf = np.zeros(1)

    dp.verify(dp.streaming(max_n=128)(Pre0), max_examples=30)


@target("librosa deemphasis(preemphasis(x)) == x", st.tuples(sig(), st.floats(0.0, 0.95)))
def _(case):
    import librosa
    x, coef = case
    y = librosa.effects.deemphasis(librosa.effects.preemphasis(x, coef=coef, zi=np.zeros(1)), coef=coef, zi=np.zeros(1))
    assert_close(y, x, FFTTolerance(c=1e4), what=f"librosa pre/deemphasis coef={coef}")


@target("librosa mu_expand(mu_compress(x)) == x", sig())
def _(x):
    import librosa
    m = np.max(np.abs(x))
    if m == 0:
        return
    x = x / m
    y = librosa.mu_expand(librosa.mu_compress(x, quantize=False), quantize=False)
    assert_close(y, x, FFTTolerance(c=1e4), what="mu law")


# ----- torchaudio -----

@target("torchaudio deemphasis(preemphasis(x)) == x", st.tuples(sig(), st.floats(0.0, 0.95)))
def _(case):
    import torch
    import torchaudio.functional as F
    x, coef = case
    t = torch.from_numpy(x)
    y = F.deemphasis(F.preemphasis(t, coeff=coef), coeff=coef).numpy()
    assert_close(y, x, FFTTolerance(c=1e4), what=f"torchaudio pre/deemphasis coef={coef}")


@target("torchaudio.functional.lfilter == scipy lfilter", st.tuples(sig(256), stable_iir(3)))
def _(case):
    import scipy.signal as ss
    import torch
    import torchaudio.functional as F
    x, (b, a) = case
    m = max(len(a), len(b))
    a = np.pad(a, (0, m - len(a)))
    b = np.pad(b, (0, m - len(b)))
    y = F.lfilter(torch.from_numpy(x), torch.from_numpy(a), torch.from_numpy(b), clamp=False).numpy()
    ref = ss.lfilter(b, a, x)
    assert_close(y, ref, FFTTolerance(c=1e4), ref_norm=max(norm(ref), norm(x)), what="torchaudio lfilter")


@target("torchaudio.functional.resample same rate == x", sig())
def _(x):
    import torch
    import torchaudio.functional as F
    y = F.resample(torch.from_numpy(x), 16000, 16000).numpy()
    assert_close(y, x, TOL, what="torchaudio resample identity")


@target("torchaudio fftconvolve == direct", sig_pair())
def _(pair):
    import torch
    import torchaudio.functional as F
    x, y = pair
    direct = np.convolve(x, y)
    fast = F.fftconvolve(torch.from_numpy(x), torch.from_numpy(y)).numpy()
    assert_close(fast, direct, TOL, ref_norm=_conv_ref(x, y, direct), what="torchaudio fftconvolve")


@target("torchaudio InverseSpectrogram(Spectrogram(x)) == x", st.tuples(sig(512, min_n=64), st.sampled_from([32, 64, 128]), st.sampled_from([2, 4])))
def _(case):
    import torch
    import torchaudio.transforms as T
    x, n_fft, frac = case
    hop = n_fft // frac
    t = torch.from_numpy(x)
    S = T.Spectrogram(n_fft=n_fft, hop_length=hop, power=None, window_fn=torch.hann_window, wkwargs={"dtype": torch.float64})(t)
    y = T.InverseSpectrogram(n_fft=n_fft, hop_length=hop, window_fn=torch.hann_window, wkwargs={"dtype": torch.float64})(S, length=len(x)).numpy()
    assert_close(y, x, FFTTolerance(c=200), what=f"torchaudio spectrogram n_fft={n_fft} hop={hop}")


def run(examples, only):
    results = []
    for name, spec in _transform_targets().items():
        if only and only not in name:
            continue
        t0 = time.perf_counter()
        rep = dp.verify(spec, max_examples=examples, raise_on_failure=False)
        results.append((name, rep.ok, str(rep) if not rep.ok else "", time.perf_counter() - t0))
    for name, check, strategy in TARGETS:
        if only and only not in name:
            continue
        seen = []

        @settings(max_examples=examples, deadline=None, database=None, print_blob=False,
                  suppress_health_check=list(HealthCheck))
        @given(strategy)
        def test(inputs):
            seen.append(inputs)
            check(inputs)

        t0 = time.perf_counter()
        try:
            test()
            results.append((name, True, "", time.perf_counter() - t0))
        except Exception as exc:  # noqa: BLE001
            msg = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            minimal = seen[-1] if seen else None
            results.append((name, False, f"{msg}\n    minimal input: {_short(minimal)}", time.perf_counter() - t0))
    return results


def _short(obj):
    with np.printoptions(precision=4, threshold=12, edgeitems=3):
        return repr(obj)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--examples", type=int, default=300)
    p.add_argument("--only", default="")
    args = p.parse_args()
    results = run(args.examples, args.only)
    for name, ok, detail, dt in results:
        print(f"{'ok  ' if ok else 'FAIL'} {dt:6.1f}s  {name}")
        if detail:
            print("    " + detail.replace("\n", "\n    "))
    bad = sum(not ok for _, ok, _, _ in results)
    print(f"\n{len(results) - bad}/{len(results)} targets pass")


if __name__ == "__main__":
    main()

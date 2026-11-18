"""Classify every candidate of ``candidates.tsv`` (step 2 of the protocol) and write ``triage.tsv``.

Usage: python benchmarks/bugcorpus/triage.py

Each candidate gets one class, decided by reading the commit (message and diff) or the changelog
entry, in this order of precedence:

* ``not-behaviour``: no change of computed output (documentation, typing, build, tests, error
  message wording, warnings, deprecations, memory or caching);
* ``out-of-category``: the function is not a transform, an LTI filter, a rational resampler or a
  streaming processor (filter design, windows, spectral estimation, peak finding, LTI system
  objects, splines, median filters, features, pitch, display, I/O...);
* ``out-of-domain``: an operator of those categories, but the bug needs inputs or settings that a
  declaration does not express: N-D arrays or an ``axis``, integer, boolean, object, byte-swapped
  or 0-d inputs, lists, other array backends or devices, threads, gradients, a platform (32-bit,
  compiler), or an input-shaping argument (``n`` that truncates, ``out=``, ``overwrite_x``);
* ``api-change``: the fix adds a parameter, so the faulty release cannot express the correct call;
* ``unreleased``: the bug never reached a release (introduced and fixed between two releases);
* ``in-scope``: the corpus (``corpus.py``), replayed by ``benchmarks/rediscovery.py``.

The script fails if a candidate is unclassified or a classification names no candidate, so that a
new run of ``mine_candidates.py`` cannot silently change the funnel.
"""

from __future__ import annotations

import collections
import pathlib

HERE = pathlib.Path(__file__).resolve().parent

NB, OC, OD, API, UNREL, IN = "not-behaviour", "out-of-category", "out-of-domain", "api-change", "unreleased", "in-scope"

CLASSES = {
    # ---------------------------------------------------------------- numpy.fft
    "ec62be21cf": (OD, "thread safety of SIGINT handling"),
    "f0322c83c8": (UNREL, "irfft with n != size in the NumPy 2.0 development rewrite, fixed before 2.0.0"),
    "aaf2c12dbb": (OD, "integer and boolean inputs"),
    "075e3df1af": (NB, "comment typos"),
    "760a1892ce": (OD, "platform (aligned_alloc)"),
    "63a4f61f16": (OD, "rfft(x, n) with n < len(x) (truncating argument): Nyquist bin not real, NumPy 2.0.0"),
    "0c36d5592d": (OD, "memory order of the output"),
    "22c791356d": (OD, "N-D (fftn axes)"),
    "c910b6157a": (NB, "build (header order)"),
    "19f2a720d8": (OD, "platform (AIX strides)"),
    "845f93ce51": (OD, "out= argument"),
    # ---------------------------------------------------------------- scipy.signal, scipy.fft
    "5681835ec5": (OC, "FIR filter design"),
    "7893dba235": (OD, "platform (32-bit integer overflow in upfirdn allocation)"),
    "b890b8f502": (OD, "non-native byte order"),
    "81697624f3": (OD, "dct/dst of complex input with overwrite_x=True"),
    "47d6700e99": (OD, "N-D (oaconvolve with no axes convolved)"),
    "a167a5475e": (OC, "StateSpace object"),
    "fd966a4007": (OC, "peak finding, wavelets"),
    "42f85a977c": (IN, "upfirdn output length: same pull request as 6bc1c1c6df (scipy-upfirdn-length)"),
    "6bc1c1c6df": (IN, "scipy-upfirdn-length"),
    "eab4cb57e1": (OC, "median filter"),
    "fd3e8c0b87": (OD, "2-D (convolve2d indexing overflow)"),
    "0b5921ed6b": (OC, "elliptic filter design"),
    "a620bfcdab": (OC, "2-D spline filter"),
    "aa3a12dc0b": (OC, "next_fast_len helper (keyword arguments)"),
    "55a13f5d6f": (OD, "threads (GIL handling in sosfilt)"),
    "8a149f29e2": (OC, "ss2tf conversion"),
    "702dba36b6": (OC, "median filter"),
    "983be289dc": (OC, "median filter"),
    "9cc5ac8172": (OC, "median filter"),
    "996062192e": (OD, "fftpack shape given as an ndarray"),
    "b5495d813d": (OC, "get_window"),
    "e10a76dac4": (OC, "get_window"),
    "54fdcaafb2": (OC, "csd (spectral estimation)"),
    "da33c0c973": (OC, "median filter"),
    "5e80b2886d": (OC, "B-spline utilities"),
    "bd98068207": (OD, "dtype preservation of resample_poly (float32 input gave float64)"),
    "5fc8ab679f": (OC, "IIR filter design"),
    "2461f2b9b6": (OC, "group delay"),
    "d5497bdc3c": (OD, "N-D (deconvolve)"),
    "ad09488269": (OC, "csd (spectral estimation)"),
    "88270c078e": (NB, "exceptions for private attributes"),
    "672619a7d7": (API, "stft/istft gain a scaling parameter"),
    "e7322848cf": (OC, "find_peaks_cwt"),
    "8ae563f541": (NB, "warning for NaN or inf in convolve"),
    "44ec457ae1": (OD, "2-D (convolve2d image indices)"),
    "0336363269": (NB, "extension module initialisation"),
    "cbec046260": (OC, "IIR filter design"),
    "ba77fbd0f4": (OC, "B-spline utilities"),
    "e2f796365d": (OC, "find_peaks"),
    "c1ce773461": (OC, "Savitzky-Golay filter (N-D window length check)"),
    "d513ce7c46": (OC, "periodogram"),
    "95f173c7d9": (OD, "platform (Intel compiler, complex sosfilt)"),
    "58e90e37ea": (OD, "platform (32-bit, Cython on Windows) in upfirdn length"),
    "010196647b": (IN, "scipy-decimate-iir-stability"),
    "4463eb0d3f": (NB, "build (check_malloc declaration)"),
    "cf1999530f": (IN, "scipy-decimate-dlti"),
    "97de33f753": (OD, "detrend with an array-like breakpoint list"),
    "bdfa4683f9": (OC, "filter order selection"),
    "299df543da": (IN, "scipy-dst-ortho"),
    "07f9217488": (NB, "__all__"),
    "a0801b8ff4": (OC, "freqz"),
    "fe967c70d7": (NB, "error message for complex input to ShortTimeFFT"),
    "54b53022da": (OD, "N-D (ShortTimeFFT fftshift over every axis)"),
    "74f607d97b": (OC, "symiirorder (mirror-symmetric IIR smoothing, not time-invariant at the edges)"),
    "b0cf648b3c": (OC, "spline namespace"),
    "cd60e810c1": (OC, "group delay"),
    "1a48b79ba5": (OD, "0-d (scalar) input to lfilter"),
    "67f338ac55": (OC, "spline filters"),
    "86c3a5bf8d": (OC, "iirpeak/iirnotch/iircomb design"),
    "e41cf1e227": (OC, "spline filters"),
    "ed85ac13c3": (OC, "spline filters"),
    "cb1fb36dd3": (OC, "spline filters"),
    "efff26b056": (OC, "IIR filter design"),
    "b67fdec29c": (OC, "median filter"),
    "77060bc96b": (OD, "array-like (list) input"),
    "4cca348df2": (OD, "other array backends"),
    "b2d624cac3": (NB, "tests under threads"),
    "b384546727": (OC, "cont2discrete"),
    "297f6cd82d": (OD, "multichannel ShortTimeFFT round trip"),
    "fec5d20126": (OD, "integer input to resample_poly"),
    "170f9e69a5": (IN, "scipy-fht-odd"),
    "7e5dd6f18b": (OC, "ShortTimeFFT.extent (plot axes)"),
    "35ad5912fa": (NB, "error message"),
    "60be494b5d": (OD, "object arrays"),
    "bb0cf02135": (OD, "list inputs"),
    "50c1dfe3ce": (OD, "other array backends (delegation)"),
    "c471c8d337": (OD, "other array backends (delegation)"),
    "0a67523f98": (OC, "dpss window"),
    "83e59a9067": (OC, "bilinear transform"),
    "87241476bd": (OD, "mutable ShortTimeFFT attributes"),
    "e8a26f037c": (OC, "lfiltic (initial-state helper)"),
    "5791cea612": (OD, "dtype preservation of resample_poly (complex input)"),
    "4b2460eae7": (OC, "window"),
    "b434b47b8a": (NB, "documentation"),
    "34cf1d9e16": (IN, "scipy-resample-complex"),
    "c4cee6140a": (OC, "normalize"),
    "378ca7a6b6": (OD, "JAX backend"),
    "42471a4c94": (OC, "polynomial utilities"),
    "07d9d03b0c": (OD, "CuPy/JAX delegation"),
    "a3c73a5b9f": (OD, "JAX backend"),
    "a723f335b4": (OC, "polynomial utilities"),
    "2efb7a66ca": (OD, "array backends with a float32 default"),
    "bb68857d7e": (OD, "non-NumPy inputs to firwin"),
    "3905d029d3": (OC, "csd (spectral estimation)"),
    "f311a50667": (OC, "remez"),
    "fa4f00d896": (NB, "memory (cache)"),
    "51c848eb0f": (NB, "memory (cache)"),
    "e368cd42ce": (OC, "lombscargle"),
    "923eb9f15a": (NB, "error messages"),
    "75412843e1": (NB, "error messages"),
    "d92f54e6f6": (OC, "LTI system conversions"),
    "9f823b9ebe": (OD, "2-D (convolve2d out-of-bounds read)"),
    "1875094f55": (OC, "zpk2tf"),
    "0b20cb99a5": (OC, "peak finding"),
    "7399ad96c2": (OC, "zpk2tf"),
    "dd4723ba8b": (OC, "minimum_phase (FIR design)"),
    "7b67444a6a": (OC, "peak widths"),
    "3c6a853b7f": (OC, "Savitzky-Golay least squares"),
    "15d7f976e5": (OD, "array API scalar assignment in lp2bp/lp2bs"),
    "c1badd5cc6": (IN, "scipy-lfilter-empty"),
    "75fa652835": (OD, "2-D (hilbert2)"),
    "00ea7de703": (OC, "remez"),
    "8e2ddfdfe3": (IN, "scipy-decimate-complex-zeros"),
    "d9b45feb89": (OC, "lfiltic (initial-state helper)"),
    "bf7fe3324b": (OC, "firwin_2d"),
    "b2adf6626c": (OC, "place_poles"),
    "c61a7be607": (OC, "2-D spline filter"),
    "5ccd743e5e": (IN, "scipy-sosfilt-empty"),
    "a11c360740": (IN, "scipy-shorttimefft-padding"),
    # ---------------------------------------------------------------- torchaudio
    "fb3ef9ba42": (NB, "docstring"), "2a02d7f539": (NB, "documentation"), "16de2b5b79": (NB, "docstring"),
    "4100ad893b": (NB, "typos"), "4863030259": (NB, "deprecation warnings"), "faf5523508": (NB, "typo"),
    "cb40dd72e0": (NB, "documentation"), "05d2580a68": (NB, "docstring"), "c15eee2396": (NB, "style"),
    "85cb37e262": (NB, "documentation"), "a8dc4de5fa": (NB, "documentation"), "04f6754668": (NB, "documentation"),
    "4771677223": (NB, "style"), "0f682c7722": (NB, "documentation build"), "18b20f77ba": (NB, "documentation"),
    "135cb7ba5e": (NB, "typo"), "457148ea93": (NB, "documentation of apply_codec"), "fa5142928e": (NB, "lint"),
    "3fc07b5427": (NB, "TorchScript test failures"), "20378a7f24": (NB, "TorchScript (jit) error"),
    "fb4eb981a4": (NB, "argument validation messages"),
    "a7797d5cae": (OD, "gradients (NaN gradient of abs)"),
    "6ab1325a55": (OD, "gradients (backward through lfilter)"),
    "7f6209b44a": (OD, "gradients (non-contiguous dy)"),
    "4e99c12dc0": (OC, "amplitude_to_DB"),
    "2f4eb4ac2f": (OC, "MVDR beamforming"),
    "c18a103b78": (OD, "device of the resampling kernel"),
    "1634ed01bc": (OC, "Griffin-Lim"),
    "5428e28399": (OD, "build without the C++ extension (fallback)"),
    "1dfac46980": (OC, "apply_codec"),
    "a3b6bfb699": (OD, "ONNX export"),
    "df2a6bcda7": (OC, "voice activity detection"),
    "265bc5ca97": (OC, "voice activity detection (trimming)"),
    "b6d4675c7a": (OC, "voice activity detection"),
    # ---------------------------------------------------------------- librosa (commits)
    "2750213ec0": (NB, "docstring"), "b64d696513": (NB, "deprecation cycle"), "187131414f": (NB, "documentation"),
    "38c5a2edc8": (NB, "documentation"), "5fb622c0ad": (NB, "typo"), "a1015d2fc8": (NB, "documentation"),
    "a9598f9d5a": (NB, "deprecated NumPy type names"), "a42d3ef74c": (NB, "documentation example"),
    "072ccf8c4f": (NB, "documentation"), "9bff37233f": (NB, "documentation"), "a19da50756": (NB, "documentation"),
    "23f4f16726": (NB, "docstring"), "876d852aaf": (NB, "documentation"), "6e0d0f5f44": (NB, "type annotations"),
    "e403272fc9": (NB, "typo"), "2d3f282e45": (NB, "documentation"),
    "bb76f86de3": (OC, "tone synthesis length"), "cd85952e39": (OC, "tuning conversions"),
    "3887e982fb": (OC, "iirt filterbank spectrogram"), "f997b6a333": (OC, "constant-Q transform"),
    "4ccee9d972": (OC, "iirt filterbank spectrogram"), "80c50e6298": (OD, "NumPy integer arguments"),
    "bc09736dc7": (OC, "iirt filterbank spectrogram"), "68232f7e5b": (OC, "yin/pyin"),
    "7ac022a849": (OC, "effects.split"), "f101492a5b": (OC, "window bandwidth"),
    "ed422f8ab7": (OC, "autocorrelate (not linear)"), "d0a12c87cd": (OC, "constant-Q transform (sparse matrices)"),
    # ---------------------------------------------------------------- librosa (changelog "Bug fixes")
    "changelog:2122": (OC, "nnls"), "changelog:2119": (OC, "effects.remix"),
    "changelog:2109+2107": (OC, "time_stretch/pitch_shift arguments"), "changelog:2106+2099": (OC, "spectral features"),
    "changelog:2097+2096": (OC, "hpss arguments"), "changelog:2087": (OC, "feature.rms"),
    "changelog:2085": (OC, "nnls"), "changelog:2063": (OC, "rqa"), "changelog:2039": (OD, "scalar inputs"),
    "changelog:2026": (OC, "display"), "changelog:1992": (OC, "display"), "changelog:1947": (OC, "note_to_hz"),
    "changelog:1916": (NB, "type annotations of decorators"), "changelog:1900": (NB, "example data registry"),
    "changelog:1897": (NB, "packaging"), "changelog:1878": (OC, "yin/pyin"), "changelog:1851": (OC, "constant-Q"),
    "changelog:1844": (OC, "mfcc"), "changelog:1814": (OC, "phase vocoder"), "changelog:1784": (OC, "ivqt"),
    "changelog:1733": (OC, "beat tracking"), "changelog:1755": (NB, "matplotlib compatibility"),
    "changelog:1677": (OC, "variable-Q transform"), "changelog:1693": (OC, "yin/pyin"),
    "changelog:1726": (OC, "time conversions"), "changelog:1560": (OC, "window bandwidth"),
    "changelog:1571": (OC, "multirate filterbank design"), "changelog:1594": (NB, "deprecated NumPy types"),
    "changelog:1596": (OC, "harmonic interpolation"), "changelog:1457": (NB, "show_versions"),
    "changelog:1493": (OC, "effects.split"),
    "changelog:1387": (API, "odd frame lengths: istft and others gain an n_fft parameter"),
    "changelog:1273": (OC, "perceptual weighting"), "changelog:1350": (NB, "deprecated NumPy types"),
    "changelog:1361": (OC, "onset strength"), "changelog:1362": (IN, "librosa-deemphasis-inplace"),
    "changelog:1375": (OD, "memory layout and number of axes of util.frame"), "changelog:1425": (OC, "yin/pyin"),
    "changelog:1430": (NB, "__all__"), "changelog:1407": (OC, "inverse constant-Q transform"),
    "changelog:1290": (OC, "display"), "changelog:1240": (OC, "display"), "changelog:1311": (OC, "chroma filterbank"),
    "changelog:1322": (OC, "mel_to_audio"), "changelog:1341": (OC, "onset_detect"),
    "changelog:1078": (OC, "stack_memory"), "changelog:1091": (NB, "joblib requirement"),
    "changelog:1094": (IN, "librosa-stft-large-window"), "changelog:1157+1196": (OC, "iirt filterbank spectrogram"),
    "changelog:1159": (IN, "librosa-preemphasis-glitch"), "changelog:1179": (NB, "unit tests"),
    "changelog:1031": (OC, "dynamic time warping"), "changelog:1041": (OC, "rms"), "changelog:1054": (OC, "tone"),
}


def main():
    lines = (HERE / "candidates.tsv").read_text().splitlines()
    header, columns, rows = lines[0], lines[1], [line.split("\t") for line in lines[2:] if line]
    ids = [r[1] for r in rows]
    missing = [i for i in ids if i not in CLASSES]
    unknown = [i for i in CLASSES if i not in ids]
    if missing or unknown:
        raise SystemExit(f"unclassified candidates: {missing}; classifications without a candidate: {unknown}")
    out = [header, columns + "\tclass\treason"]
    out += ["\t".join([*r, *CLASSES[r[1]]]) for r in rows]
    (HERE / "triage.tsv").write_text("\n".join(out) + "\n")
    counts = collections.Counter((r[0], CLASSES[r[1]][0]) for r in rows)
    libraries = sorted({r[0] for r in rows})
    classes = [NB, OC, OD, API, UNREL, IN]
    print("library      " + "  ".join(f"{c:>15}" for c in classes) + "  total")
    for lib in libraries:
        n = [counts[(lib, c)] for c in classes]
        print(f"{lib:<12} " + "  ".join(f"{k:>15}" for k in n) + f"  {sum(n):>5}")
    n = [sum(counts[(lib, c)] for lib in libraries) for c in classes]
    print(f"{'total':<12} " + "  ".join(f"{k:>15}" for k in n) + f"  {sum(n):>5}")


if __name__ == "__main__":
    main()

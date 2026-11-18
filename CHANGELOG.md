# Changelog

## Unreleased

- Frozen environment for reproducibility: `env/requirements-py312-linux.txt`, every development,
  documentation and benchmark dependency pinned with its hash (compiled by `env/lock.sh` from
  `env/requirements.in`), and a `Dockerfile` on a base image pinned by digest. The nightly CI job
  installs the lock and runs the tests in it.
- `dp.array_api(func, xp=..., device=...)`: an operator written for PyTorch, CuPy, JAX or any Array
  API namespace takes and returns NumPy arrays, so that it can be declared like any other. Arguments
  and outputs are converted recursively (a `(y, state)` pair stays a pair), and dtypes are kept.
  `benchmarks/gpu_fft_profile.py` measures the FFT error of each GPU backend found (CUDA, MPS, CuPy,
  JAX) against an extended-precision DFT. First profile (CuPy on an RTX 5070): at most 3.6 ε in
  complex128, the same as NumPy on the CPU. Docs: `docs/array-api.md`.
- Automatic mutation of SciPy code (`benchmarks/automutation/`, `docs/automutation.md`): 701
  cosmic-ray mutants of a verbatim copy of `resample`, `resample_poly`, `decimate` and `hilbert`
  (SciPy 1.18.1), checked with SciPy's own tests and with dsp-props declarations. SciPy's tests kill
  78 % of them, dsp-props 57 %, and both together 82 %. Most of the mutants only SciPy's tests
  catch are on code that the declarations never execute (options, `axis`, argument checks); most
  of those only dsp-props catches are in the odd-length branch of `hilbert`.

- `min_n=` on every declaration (`transform`, `lti_filter`, `resampler`, `streaming`, `stateful`):
  the shortest input the operator accepts, for functions with a documented minimum length
  (`filtfilt` and its `padlen`, a short-time transform and its window). Every relation draws
  lengths in `[min_n, max_n]`; pathological lengths outside the range are dropped. Default 1.
- Fix: `dp.family` built a member inside a Hypothesis search that kept the default deadline, so a
  factory slower than 200 ms (a filter design, a remote call) raised `FlakyFailure` instead of
  checking the member.
- Rediscovery campaign (`benchmarks/rediscovery.py`, `docs/rediscovery.md`): a corpus of
  historical bugs of SciPy and librosa, mined mechanically from the projects' histories
  (`benchmarks/bugcorpus/`), each checked on the last faulty release and the first fixed one,
  installed in their own environments. By-product: `benchmarks/decimate_dlti_route.py` measures a
  remaining accuracy defect of `scipy.signal.decimate` with a real IIR `dlti` filter.

- `IIRTolerance(b, a, structure="df2t")`: a first-order bound for the transposed direct form II
  of `scipy.signal.lfilter` (three roundings per state instead of the length of the whole
  recursion), 1 to 3 times tighter than the direct form I bound on usual designs
  (`benchmarks/iir_structure.py`). The default stays `structure="df1"`, valid for any summation
  order. Docs: the remaining gap to the measured error is the worst-case assumption, and `c = 1000`
  is a guard, not a model, for resamplers and streaming processors.
- Fix: `IIRTolerance` computed the gain of 1/A(z) before normalising by `a[0]`, so its bound was
  `|a[0]|` times too small for a filter declared with `a[0] != 1` (too large when `|a[0]| < 1`).
- Task-oriented documentation: "Test your filter in 5 minutes" (`docs/quickstart.md`, one recipe
  per task) and a gallery of defects found in real libraries (`docs/gallery.md`: empty blocks in
  SciPy's `lfilter` and `sosfilt`, `filtfilt(method="gust")` with trailing zeros and with poles
  near zero, the precision of PyTorch's FFT), each with a script in `examples/gallery/` run by
  `tests/test_gallery.py`.
- Operator families: `@dp.family(params)` on a factory that returns a declared operator draws the
  parameters with the signals, and the counterexample is `(parameters, input)`, both shrunk.
  Parameter strategies in `dp.generators`: `iir_coefficients` (stable filters of order 0 up,
  numerators shorter or longer than the denominator, `a[0] != 1`), `fir_taps`, `rate_pairs`
  (coprime, `(1, 1)` included). The comparison benchmark gains the approach `dsp-props/params`.
- `float32=True` on any declaration (float64 precision) adds the `float32 consistency` relation:
  the output on a float32 signal agrees with the output on the same values in float64, to float32
  accuracy, and is at least float32. It catches a float32 path that differs from the float64 one
  (dtype-dependent branch, half-precision coefficients or input, float16 output).
- Versioned calibrations: `dsp-props calibrate TARGETS --tolerances tolerances.json` measures the
  declared operators against an extended-precision reference (extended DFT for `dft` and `rdft`
  transforms, `longdouble_lfilter` for LTI filters with `transfer=`) and stores the errors per
  length with the library versions (`dp.ToleranceFile`). `dp.verify(tolerances=...)`,
  `dsp-props check --tolerances`, `--dsp-props-tolerances` (pytest) and `DSP_PROPS_TOLERANCES` then
  use the stored calibration instead of the default model; a tolerance given in the declaration
  still wins. `Report.tolerance` says which applied.
- Precision regressions: `dsp-props calibrate ... --compare` (`ToleranceFile.compare`) measures
  again on the same probes and reports the lengths whose error grew by more than a factor (2).
- Regression tests from counterexamples: on failure, `--dsp-props-regressions [DIR]` (pytest),
  `dsp-props check --regressions DIR` or `dp.write_regression_tests` write
  `test_regression_<operator>.py`, one plain test per minimal counterexample, replayed exactly and
  without Hypothesis (`dsp_props.regression.replay`). A case is written only if replaying it from
  its source fails.
- `dp.verify` records each counterexample as drawn, before the operator runs, so an operator that
  writes into its input no longer reports the overwritten input.

## 0.1.4 (2026-11-18)

- pytest plugin (entry point `pytest11`): an operator declared in a test module becomes one test
  per relation (`test_x.py::my_fft::parseval`), with the minimal counterexample in the report
  (`docs/pytest.md`). `-p no:dsp_props` disables it.
- `dp.verify(relations=..., database=...)`: failing inputs are saved in Hypothesis's example
  database and replayed first.
- Cost profiles `fast` (30 inputs), `default` (200) and `nightly` (2000), chosen with `profile=`,
  `dp.set_profile`, `DSP_PROPS_PROFILE`, `dsp-props check --profile` or `--dsp-props-profile`.
- `benchmarks/make_results.py` generates `benchmarks/results.md` and `results.tex` from the
  benchmark results; the numbers quoted in the README, `CLAUDE.md` and the docs are checked
  against them in CI.
- CI: lint job (ruff, mypy on `src/`), coverage, internal suite on the `fast` profile, nightly run.
- Precision mutants (`mutants/precision_mutants.py`, `benchmarks/precision_mutants.py`): operators
  that keep every property but lose accuracy, graded from 0 to 24 dropped significand bits, and the
  smallest loss the relations detect for each operator class (`docs/blind-spots.md`).
- `benchmarks/precision_oracles.py` compares the relations with default, fixed (1e-7) and calibrated
  tolerances, and a differential `assert_allclose`, on the same precision mutants.
- `dp.calibrate(..., base=None)` keeps no analytic floor, so calibrating a trusted implementation
  gives a tolerance tighter than the default model. The default is unchanged.
- `benchmarks/calibrated_tolerance.py` checks other correct implementations against a tolerance
  calibrated on a trusted one, and measures detection thresholds on five filters and four ratios.
- CI: the macOS calibration job runs on `macos-15` (`macos-14` runners are no longer assigned).
- `benchmarks/campaign.py` uses the library's `block_splits` (arbitrary cuts up to max(N, 16)
  blocks, one-sample blocks, fixed-size blocks) instead of its own copy limited to nine blocks.
- `benchmarks/validation.json`: false-positive runs redone on the current code.

## 0.1.3 (2026-11-04)

Fixes found by the October 2026 reviews of the library and the article.

- The transfer-function check bounds the truncated tail of the impulse response. Filters with poles
  very close to the unit circle (a DC blocker with a pole at 0.9999, `butter(2, 5e-5)`) no longer
  raise a false alarm against a correct `scipy.signal.lfilter`.
- The expected precision comes from the input, not from the output: an operator that silently
  returns `complex64` for `complex128` input is now reported.
- `dp.calibrate` checks that `long double` is wider than `double` and otherwise calibrates against
  a double-double DFT (Windows, macOS arm64).
- Block-invariance draws splits with more than eight blocks, including all one-sample blocks.
- The README, the documentation and the papers describe the tolerances, block splits and upstream
  findings as the code and the trackers show them.

## 0.1.2 (2026-10-17)

- Unstable transfer functions are rejected at declaration time.
- Complex signals are fed to complex transforms.
- Documentation site on Read the Docs.

## 0.1.1 (2026-10-16)

- README in English.

## 0.1.0 (2026-10-14)

- First release on PyPI.

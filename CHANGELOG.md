# Changelog

## Unreleased

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

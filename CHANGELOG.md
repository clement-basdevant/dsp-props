# Changelog

## Unreleased

- Precision mutants (`mutants/precision_mutants.py`, `benchmarks/precision_mutants.py`): operators
  that keep every property but lose accuracy, graded from 0 to 24 dropped significand bits, and the
  smallest loss the relations detect for each operator class (`docs/blind-spots.md`).

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

# dsp-props

[![PyPI](https://img.shields.io/pypi/v/dsp-props)](https://pypi.org/project/dsp-props/) [![CI](https://github.com/clement-basdevant/dsp-props/actions/workflows/ci.yml/badge.svg)](https://github.com/clement-basdevant/dsp-props/actions/workflows/ci.yml) [![Documentation](https://readthedocs.org/projects/dsp-props/badge/?version=latest)](https://dsp-props.readthedocs.io/) [![DOI](https://zenodo.org/badge/1418824686.svg)](https://doi.org/10.5281/zenodo.23437840)

**Property-based testing for digital signal processing code**

DSP code is usually tested by comparing its output with a reference (MATLAB, SciPy) on a few signals, with a hand-picked tolerance. `dsp-props` instead checks **relations that must always hold** (linearity, shift theorem, Parseval, Hermitian symmetry, invertibility, convolution theorem, causality, time invariance, consistency across block sizes), on **automatically generated signals** from [Hypothesis](https://hypothesis.works/), chosen to trip up DSP code: impulses, off-grid sinusoids, close tones, near-Nyquist tones, extreme amplitudes, lengths 1, 2, prime or 2^k ± 1 (about 15 % of the lengths drawn), empty and one-sample blocks. Tolerances scale with the operation instead of a fixed `atol`: c · ε · log N for an FFT, with an engineering constant (c = 20 for transforms, c = 1000 for filters without a declared transfer function, resamplers and streaming), a derived bound for direct-form I recursive filters, and `dp.calibrate` when you need a tight tolerance. Every failure is reported with a small counterexample.

> Status: alpha. The six phases of the [development plan](https://github.com/clement-basdevant/dsp-props/blob/main/docs/schema-directeur.md) (in French) are complete; version 0.1.2 on PyPI.

**Documentation:** [dsp-props.readthedocs.io](https://dsp-props.readthedocs.io/) (tutorial, tolerance models, known blind spots, API reference).

## Example

```python
import numpy as np
import dsp_props as dp

@dp.transform(inverse=np.fft.ifft, parseval=True, real_input_hermitian=True)
def my_fft(x):
    return np.fft.fft(x)

print(dp.verify(my_fft))
```

```
dsp-props report for my_fft
  ok   linearity
  ok   circular shift
  ok   parseval
  ok   hermitian symmetry
  ok   inversibility
  ok   convolution theorem
```

On failure, `verify` raises `PropertyViolation` with, for each violated relation, the counterexample shrunk by Hypothesis.

Other declarations:

```python
# Real FFT: the inverse receives the length N
dp.transform(inverse=np.fft.irfft, kind="rdft", parseval=True)(np.fft.rfft)

# Any linear transform (DCT, wavelets...)
dp.transform(kind="generic", inverse=idct_ortho, parseval=True, parseval_scale=lambda n: 1.0)(dct_ortho)

# LTI filter: linearity, time invariance, causality, frequency response = B(z)/A(z).
# The tolerance follows the conditioning of the recursion (IIRTolerance, derived for direct form I).
dp.lti_filter(causal=True, transfer=(b, a))(lambda x: scipy.signal.lfilter(b, a, x))

# Bounded gain on N <= max_n: no input bounded by 1, of length up to max_n, may exceed the declared gain
# (the worst-case input is constructed); this is not a BIBO stability proof
dp.lti_filter(max_gain=l1_norm_of_h)(my_filter)

# Rational L/M resampler: linearity, a delay of M input samples = a delay of L output samples,
# length ceil(N L / M), DC gain, and alias rejection when decimating
dp.resampler(2, 3)(lambda x: scipy.signal.resample_poly(x, 2, 3))

# Block processing: splitting the input into blocks must give the same output as one call. The splits drawn
# are arbitrary cuts (up to max(N, 16) blocks, empty and one-sample blocks included), all blocks of one
# sample, and blocks of a fixed size
@dp.streaming()
class MyFilter:
    def process(self, block): ...

# Function with explicit state, for example lfilter with zi
dp.stateful(lambda block, z: scipy.signal.lfilter(b, a, block, zi=z), init_state=np.zeros(2))
```

## Command line

`dsp-props check` finds every operator declared in a module or file and verifies it:

```
$ dsp-props check examples/filters.py --examples 100
FAIL examples/filters.py:buggy_fft  (4 relations, 3.4s)
      FAIL inversibility
           inversibility: L2 error 2.000e+00 exceeds tolerance 6.280e-15 (N=2, ref norm 1.414e+00)
           minimal input: [1, 1]
ok   examples/filters.py:fft  (6 relations, 6.5s)
ok   examples/filters.py:lowpass  (4 relations, 3.1s)
ok   examples/filters.py:resample_2_3  (5 relations, 2.8s)

3/4 operators pass
```

Targets: `package.module`, `file.py`, `module:operator`. Options: `--json`, `--markdown`, `--html`, `--examples N`, `--derandomize` (reproducible inputs in CI). The exit code is 1 if a relation fails, which is enough to block a CI run.

## C code

`dsp_props.adapters` loads a C function (shared library, through ctypes) and declares it like any other operator. Convention: `void f(const double *x, size_t n, double *y, double *state)`.

```python
from dsp_props.adapters import build_shared, c_function, c_block_processor

lib = build_shared("examples/c/biquad.c")
dp.verify(dp.lti_filter(transfer=(b, a))(c_function(lib, "biquad", state_size=2)))
dp.verify(dp.streaming()(c_block_processor(lib, "biquad", state_size=2)))
```

`examples/c/biquad.c` contains a correct biquad and two buggy variants. The one that forgets a state variable between blocks passes every test on a single call and is caught only by block invariance.

## Calibrated tolerances

A fixed ε · log N bound does not describe every library: over N = 2…1200, the PyTorch FFT (MKL) reaches 195 ε against 4 ε for NumPy, depending on the factorisation of N. The default constants are wide on purpose, so an FFT a few tens of ε less accurate than NumPy still passes them (see [the blind spots](https://github.com/clement-basdevant/dsp-props/blob/main/docs/blind-spots.md)). `dp.calibrate` is the recommended path for a tight tolerance: it measures the actual error against an extended-precision reference (long double where it is extended, a double-double DFT elsewhere) and derives the tolerance from it:

```python
tol = dp.calibrate(torch_fft, lengths=range(1, 513))   # refuses a wrong operator
print(tol.summary())
dp.transform(inverse=torch_ifft, parseval=True, tolerance=tol)(torch_fft)
```

`python benchmarks/error_profile.py` produces the full error profile (`benchmarks/error_profile.csv`), and `python benchmarks/plot_error_profile.py` the figure below.

![Relative FFT error as a function of N, NumPy against PyTorch](https://raw.githubusercontent.com/clement-basdevant/dsp-props/main/docs/figures/error_profile.png)

## Validating the tool: the mutant suite

`mutants/` contains 31 deliberately buggy implementations, in six families. They were written by hand, by the author of the relations and in their vocabulary (each mutant breaks a structural property a relation states), so the mutation score measures that the relations catch the faults they were designed for, not their power on independent faults:

| Family | Mutants |
| --- | --- |
| Complex FFT | missing 1/N, missing conjugation, off-by-one index, wrong normalisation, float32 computation, twiddle sign, last sample ignored, constant offset |
| Real FFT (`rfft`) | one bin short, inverse that ignores N, Nyquist bin halved, conjugated spectrum, inverse without Hermitian symmetry |
| Orthonormal DCT-II | DC coefficient weight forgotten, inverse of the wrong type, float32 computation |
| IIR filter (order 4, resonant) | float32 recursion, filtfilt declared causal, coefficients that change midway, non-zero initial state, gain wrong by 1 % |
| 2/3 resampler | interpolation gain forgotten, anti-aliasing too weak, length rounded down, phase that depends on the parity of N, float32 filtering, global state leaking from one call to the next |
| Block streaming | state reset, history shifted by one sample, empty block that clears the state, short blocks left unfiltered |

```
python mutants/run_mutants.py
...
Mutation score: 31/31 (100%)
```

Counterexamples are tiny. In a run with every `dp.verify` seeded with 0, for 26 of 30 mutants (excluding `iir_wrong_gain`, detected on the impulse response) the signal shrunk by Hypothesis has at most 3 samples (median: 2), and 28 of 31 mutants fail within the first 3 generated inputs. The exceptions are expected: 15 samples for `filtfilt` declared causal, which falls back to `lfilter` on short inputs, 34 for a filter that changes after 64 samples (the time-invariance relation delays the input), and 192 for the DC-gain and aliasing relations, which need long signals. `mutants/run_mutants.py` fixes no seed, so these counts vary between runs (the run recorded in `benchmarks/mutant_results.json` gave 27 and 27); `python mutants/run_mutants.py --json benchmarks/mutant_results.json` records a run.

Every report gives the smallest input found and can be exported:

```python
report = dp.verify(my_fft, raise_on_failure=False)
print(report)              # FAIL inversibility ... minimal input: [1, 1]
report.to_json()           # for CI or a results table
report.to_markdown()       # for a PR or an issue
```

Known blind spot: a filter that is wrong but still linear, time-invariant and causal (gain multiplied by 1.01) passes every relation unless its transfer function is declared. With `transfer=(b, a)`, it is detected.

## Bug hunting in real libraries

`benchmarks/campaign.py` applies the relations to SciPy, NumPy, PyTorch, librosa and torchaudio (46 targets). What it found, at its actual size:

- **`scipy.signal.lfilter` on an empty block with an initial state** returns an uninitialised final state for an IIR filter (and raises for a FIR one), so block processing goes silently wrong after an empty block. This is a **rediscovery**: it was already reported as [scipy/scipy#22571](https://github.com/scipy/scipy/issues/22571) and is fixed in the development version of SciPy. dsp-props found it independently, and the FIR case also affects librosa's streaming pre-emphasis.
- **`scipy.signal.sosfilt` on an empty block** is new, but it **raises an exception** (`ValueError`) on any input that is empty along the filtered axis; it does not diverge silently. Reported together with the `lfilter` case in [scipy/scipy#26406](https://github.com/scipy/scipy/issues/26406).
- **`scipy.signal.tf2sos` drops pure delays** ([scipy/scipy#26417](https://github.com/scipy/scipy/issues/26417)): a documented behaviour that is surprising, possibly a documentation issue rather than a bug.
- **torchaudio's `deemphasis` clamps its output to [-1, 1]** ([pytorch/audio#4238](https://github.com/pytorch/audio/issues/4238)): the clamp comes from `lfilter`'s documented default `clamp=True`, in a library now in maintenance mode.

None of these reports had been confirmed by a maintainer when this was written (October 2026). Detailed results are in `benchmarks/RESULTS-*.md` ([benchmarks/](https://github.com/clement-basdevant/dsp-props/tree/main/benchmarks)).

## Installation

From PyPI ([pypi.org/project/dsp-props](https://pypi.org/project/dsp-props/)):

```
python -m pip install "dsp-props[scipy]"   # [scipy] for IIR filters and transfer functions
```

For development:

```
git clone https://github.com/clement-basdevant/dsp-props
cd dsp-props
python -m pip install -e ".[dev]"
python -m pytest
```

## Layout

| Folder | Content |
| --- | --- |
| `src/dsp_props/core/` | Tolerance models, robust comparison |
| `src/dsp_props/generators/` | Hypothesis strategies for signals |
| `src/dsp_props/relations/` | One metamorphic relation = one class |
| `mutants/` | Buggy implementations and mutation score |
| `tests/` | Tests of the tool itself |
| `benchmarks/` | Bug-hunting campaign, error profiles, evaluation of the article |
| `proofs/` | Coq proofs of the relations and of the IIR error bound |
| `docs/` | Development plan and documentation |
| `paper/`, `papers/` | Article, bibliography and drafts |

## Roadmap

1. **Foundations**: generators, basic relations, tests on `numpy.fft` *(done)*
2. **Tolerance engine**: error models per operator class, calibration against extended precision *(done)*
3. **Advanced relations**: causality, bounded gain, block streaming, multirate *(done)*
4. **Mutant suite**: 31 hand-written mutants in six families, campaign on 46 real targets *(done; to be extended)*
5. **Shrinking and reporting**: small counterexamples, HTML report *(done)*
6. **Auto-discovery and C adapter** *(done: `dsp-props check`, `dsp_props.adapters`)*

## Citation

If you use `dsp-props`, please cite it with its DOI, [10.5281/zenodo.23437840](https://doi.org/10.5281/zenodo.23437840) (all versions; each release also has its own DOI on Zenodo), or with the metadata in [`CITATION.cff`](CITATION.cff) (GitHub's *Cite this repository* button).

## License

BSD-3-Clause, like NumPy and SciPy.

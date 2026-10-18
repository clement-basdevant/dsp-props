# dsp-props

[![PyPI](https://img.shields.io/pypi/v/dsp-props)](https://pypi.org/project/dsp-props/) [![CI](https://github.com/clement-basdevant/dsp-props/actions/workflows/ci.yml/badge.svg)](https://github.com/clement-basdevant/dsp-props/actions/workflows/ci.yml) [![Documentation](https://readthedocs.org/projects/dsp-props/badge/?version=latest)](https://dsp-props.readthedocs.io/) [![DOI](https://zenodo.org/badge/1418824686.svg)](https://doi.org/10.5281/zenodo.23437840)

**Property-based testing for digital signal processing code**

DSP code is usually tested by comparing its output with a reference (MATLAB, SciPy) on a few signals, with a hand-picked tolerance. `dsp-props` instead checks **relations that must always hold** (linearity, shift theorem, Parseval, Hermitian symmetry, invertibility, convolution theorem, causality, time invariance, consistency across block sizes), on **automatically generated signals** from [Hypothesis](https://hypothesis.works/), chosen to trip up DSP code: impulses, off-grid sinusoids, close tones, near-Nyquist tones, extreme amplitudes, lengths 1, 2, prime or 2^k ± 1, empty and one-sample blocks. Tolerances follow an error model of each operation (O(ε log N) error for an FFT) instead of a fixed `atol`, and every failure is reported with a small counterexample.

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
# The tolerance follows the conditioning of the recursion (IIRTolerance).
dp.lti_filter(causal=True, transfer=(b, a))(lambda x: scipy.signal.lfilter(b, a, x))

# BIBO stability: no input bounded by 1 may exceed the declared gain (worst-case input is constructed)
dp.lti_filter(max_gain=l1_norm_of_h)(my_filter)

# Rational L/M resampler: linearity, a delay of M input samples = a delay of L output samples,
# length ceil(N L / M), DC gain, and alias rejection when decimating
dp.resampler(2, 3)(lambda x: scipy.signal.resample_poly(x, 2, 3))

# Block processing: any split into blocks (empty and one-sample blocks included) must give the same output
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

A fixed ε · log N bound does not describe every library: over N = 2…1200, the PyTorch FFT (MKL) reaches 195 ε against 4 ε for NumPy, depending on the factorisation of N. `dp.calibrate` measures the actual error against an extended-precision reference and derives the tolerance from it:

```python
tol = dp.calibrate(torch_fft, lengths=range(1, 513))   # refuses a wrong operator
print(tol.summary())
dp.transform(inverse=torch_ifft, parseval=True, tolerance=tol)(torch_fft)
```

`python benchmarks/error_profile.py` produces the full error profile (`benchmarks/error_profile.csv`), and `python benchmarks/plot_error_profile.py` the figure below.

![Relative FFT error as a function of N, NumPy against PyTorch](https://raw.githubusercontent.com/clement-basdevant/dsp-props/main/docs/figures/error_profile.png)

## Validating the tool: the mutant suite

`mutants/` contains 31 deliberately buggy implementations, in six families:

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

Counterexamples are tiny: for 27 of 30 mutants (excluding `iir_wrong_gain`, detected on the impulse response), the signal shrunk by Hypothesis has at most 3 samples (median: 2), and 27 mutants fail within the first 3 generated inputs. The exceptions are expected: 97 samples for a filter that changes after 64, and 192 for the DC-gain and aliasing relations, which need long signals. `python mutants/run_mutants.py --json benchmarks/mutant_results.json` records these measurements.

Every report gives the smallest input found and can be exported:

```python
report = dp.verify(my_fft, raise_on_failure=False)
print(report)              # FAIL inversibility ... minimal input: [1, 1]
report.to_json()           # for CI or a results table
report.to_markdown()       # for a PR or an issue
```

Known blind spot: a filter that is wrong but still linear, time-invariant and causal (gain multiplied by 1.01) passes every relation unless its transfer function is declared. With `transfer=(b, a)`, it is detected.

## Bug hunting in real libraries

`benchmarks/campaign.py` applies the relations to SciPy, NumPy, PyTorch, librosa and torchaudio (46 targets).
It found that `scipy.signal.lfilter` and `sosfilt` fail on an empty block with an initial state, reported in [scipy/scipy#26406](https://github.com/scipy/scipy/issues/26406). The `lfilter` part was already known ([scipy/scipy#22571](https://github.com/scipy/scipy/issues/22571), fixed in the development version of SciPy): dsp-props rediscovered it independently. The `sosfilt` part, which fails on any input that is empty along the filtered axis, is new. It also found that `scipy.signal.tf2sos` drops pure delays ([scipy/scipy#26417](https://github.com/scipy/scipy/issues/26417)) and that torchaudio's `deemphasis` clamps its output to [-1, 1] ([pytorch/audio#4238](https://github.com/pytorch/audio/issues/4238)). Detailed results are in `benchmarks/RESULTS-*.md` ([benchmarks/](https://github.com/clement-basdevant/dsp-props/tree/main/benchmarks)).

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
3. **Advanced relations**: causality, stability, block streaming, multirate *(done)*
4. **Mutant suite**: about forty mutants, trial on `scipy.signal` *(31 mutants, campaign on 46 real targets; to be extended)*
5. **Shrinking and reporting**: small counterexamples, HTML report *(done)*
6. **Auto-discovery and C adapter** *(done: `dsp-props check`, `dsp_props.adapters`)*

## Citation

If you use `dsp-props`, please cite it with its DOI, [10.5281/zenodo.23437840](https://doi.org/10.5281/zenodo.23437840) (all versions; each release also has its own DOI on Zenodo), or with the metadata in [`CITATION.cff`](CITATION.cff) (GitHub's *Cite this repository* button).

## License

BSD-3-Clause, like NumPy and SciPy.

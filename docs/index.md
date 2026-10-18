# dsp-props

**Property-based testing for digital signal processing code.**

DSP code is usually tested by comparing its output with a reference (MATLAB,
SciPy) on a few signals, with a hand-picked tolerance. `dsp-props` instead
checks **relations that must always hold** for any correct implementation:
linearity, the shift and convolution theorems, Parseval's identity, Hermitian
symmetry, invertibility, causality, time invariance, consistency with a declared
transfer function, periodic time invariance of a resampler, and invariance of a
streaming processor to the way its input is cut into blocks.

The relations are checked on **automatically generated signals** from
[Hypothesis](https://hypothesis.works/), chosen to trip up DSP code: impulses,
constants, off-grid sinusoids, close tones, near-Nyquist tones, extreme
amplitudes (1e-100 to 1e100), lengths 1, 2, prime or 2^k ± 1, and empty and
one-sample blocks. Tolerances follow an **error model** of each operation
(O(ε log N) for an FFT, a conditioning-aware bound for recursive filters)
instead of a fixed `atol`, and every failure is reported with a **small
counterexample**.

!!! note "Status"
    Alpha. Version 0.1.2 is on [PyPI](https://pypi.org/project/dsp-props/). DOI: [10.5281/zenodo.23437840](https://doi.org/10.5281/zenodo.23437840).

## Install

```bash
python -m pip install "dsp-props[scipy]"
```

The `scipy` extra is needed for IIR filters and transfer-function checks
(`dp.lti_filter(transfer=...)`, `IIRTolerance`). Without it, `dsp-props` depends
only on NumPy and Hypothesis. Python 3.10 or later.

For development:

```bash
git clone https://github.com/clement-basdevant/dsp-props
cd dsp-props
python -m pip install -e ".[dev]"
python -m pytest
```

## 30-second example

Declare what an operator is with one decorator, then call `dp.verify`:

```python
import numpy as np
import dsp_props as dp

@dp.transform(inverse=np.fft.ifft, parseval=True, real_input_hermitian=True)
def my_fft(x):
    return np.fft.fft(x)

print(dp.verify(my_fft))
```

```text
dsp-props report for my_fft
  ok   linearity
  ok   circular shift
  ok   parseval
  ok   hermitian symmetry
  ok   inversibility
  ok   convolution theorem
```

On failure, `verify` raises `PropertyViolation`, which carries for each violated
relation the counterexample shrunk by Hypothesis. A transform that is off by one
bin fails on a two-sample input:

```python
@dp.transform(inverse=np.fft.ifft)
def buggy_fft(x):
    return np.roll(np.fft.fft(x), 1)

print(dp.verify(buggy_fft, raise_on_failure=False))
```

```text
dsp-props report for buggy_fft
  ok   linearity
  FAIL circular shift
       circular shift (s=1): L2 error 4.000e+00 exceeds tolerance 8.882e-15 (N=2, ref norm 2.000e+00)
       minimal input: ([1, 1], 1)
  FAIL inversibility
       inversibility: L2 error 2.000e+00 exceeds tolerance 6.280e-15 (N=2, ref norm 1.414e+00)
       minimal input: [1, 1]
  FAIL convolution theorem
       convolution theorem: L2 error 1.000e+00 exceeds tolerance 3.140e-15 (N=2, ref norm 7.071e-01)
       minimal input: ([1, 1], [1, 1])
```

Inputs are random, so the exact counterexample can differ from one run to the
next; pass `seed=...` or `derandomize=True` to `verify` for reproducible runs.

## What can be declared

| Declaration | For | Relations checked |
| --- | --- | --- |
| `dp.transform(kind="dft")` | complex DFT | linearity, circular shift; Parseval, Hermitian symmetry, inversion and convolution theorem when declared |
| `dp.transform(kind="rdft")` | real-input DFT (`N // 2 + 1` bins) | output shape, linearity, circular shift, real DC and Nyquist bins; Parseval, inversion and convolution theorem when declared |
| `dp.transform(kind="generic")` | any linear transform (DCT, wavelets...) | linearity; Parseval and inversion when declared |
| `dp.lti_filter()` | LTI filter from zero state | linearity, time invariance, causality; transfer function and gain bound when declared |
| `dp.resampler(L, M)` | rational resampler | linearity, output length, periodic time invariance, DC gain, alias rejection |
| `dp.streaming()`, `dp.stateful()` | block processors | block invariance (empty and one-sample blocks included) |

See the [relations catalogue](relations.md) for the exact statement of each
relation, and the [tutorial](tutorial.md) for an end-to-end example on a filter.
C code can be tested through `ctypes` with
[`dsp_props.adapters`](api/adapters.md).

## Where to go next

- [Tutorial: test your own filter](tutorial.md): declarations, reading a
  counterexample, the command line, streaming processors.
- [Choosing a tolerance](tolerances.md): the error models and `dp.calibrate`.
- [What the relations don't see](blind-spots.md): known blind spots, and how to
  close some of them.
- [API reference](api/index.md).

## License

BSD-3-Clause, like NumPy and SciPy.

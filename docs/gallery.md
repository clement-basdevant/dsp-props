# Gallery: bugs found in real libraries

Each case below is a defect that dsp-props found, or found again, in a widely
used library. Each one comes with a script in
[`examples/gallery/`](https://github.com/clement-basdevant/dsp-props/tree/main/examples/gallery),
and `tests/test_gallery.py` runs every script and checks what this page says
about it. The outputs come from SciPy 1.18.1, NumPy 2.5 and PyTorch 2.15 (CPU).

| Case | Library | Kind of defect | Found by | Status |
| --- | --- | --- | --- | --- |
| [An empty block in a streaming filter](#an-empty-block-in-a-streaming-filter) | SciPy `lfilter`, `sosfilt` | wrong state, or an exception | block invariance | fixed in SciPy 2.0.0 |
| [Trailing zeros change `filtfilt(method="gust")`](#trailing-zeros-change-filtfiltmethodgust) | SciPy `filtfilt` | output depends on how the filter is written | a custom relation | reported |
| [`filtfilt(method="gust")` with poles near zero](#filtfiltmethodgust-with-poles-near-zero) | SciPy `filtfilt` | digits lost on a well-conditioned problem | a custom relation (time reversal) | reported |
| [PyTorch's FFT at lengths with a large prime factor](#pytorchs-fft-at-lengths-with-a-large-prime-factor) | PyTorch `torch.fft.fft` | precision, not a bug | calibration | measured |

## An empty block in a streaming filter

Real-time code filters a signal block by block and carries the filter state
between calls; an empty block must leave the state alone. With SciPy before
2.0.0 it does not:

- `lfilter` with an IIR filter and `zi` returns a wrong final state for an
  empty block, so every later block is wrong, and nothing is raised. This is
  [scipy/scipy#22571](https://github.com/scipy/scipy/issues/22571), which
  dsp-props found again.
- `lfilter` with an FIR filter, and `sosfilt`, raise on an empty block with
  `zi` ([scipy/scipy#26406](https://github.com/scipy/scipy/issues/26406)).
  The FIR case reaches librosa's streaming pre-emphasis.

Both are fixed by [scipy/scipy#26415](https://github.com/scipy/scipy/pull/26415),
in SciPy 2.0.0. `dp.stateful` declares the block processors, and the block
invariance relation finds each bug on one or two samples:

```python
--8<-- "examples/gallery/empty_block_state.py:16:"
```

```text
lfilter: FAIL
  minimal input: (array([1., 0.]), [1, 1])
sosfilt: FAIL
  minimal input: (array([1.]), [0])
lfilter, empty blocks skipped: ok
```

The input `[1, 0]` cut at `[1, 1]` is the blocks `[1]`, `[]` and `[0]`. For
`sosfilt`, the cut `[0]` makes the first block empty.

## Trailing zeros change `filtfilt(method="gust")`

`(b, a)` and the same coefficients followed by zeros are the same filter, so
`filtfilt` must give the same output for both. With `method="gust"`
(Gustafsson's initial conditions) it does not. The zeros raise the order, add
an impulse at both edges of the output, and make the least-squares system
singular. Reproduced with SciPy 1.15 to 1.18.1
([scipy/scipy#26610](https://github.com/scipy/scipy/issues/26610); the fix
proposed in [scipy/scipy#26624](https://github.com/scipy/scipy/pull/26624) is
under review).

No catalogue relation covers "two ways of writing one filter". The test below
is plain Hypothesis, built from dsp-props parts: the filter strategy
`iir_coefficients`, the signal strategy `signals` and the tolerance check
`assert_close`.

```python
--8<-- "examples/gallery/filtfilt_gust_trailing_zeros.py:15:"
```

```text
dsp_props.core.tolerance.ToleranceError: filtfilt(gust) with trailing zeros: L2 error 1.732e+00 exceeds tolerance 2.465e-10 (N=32, ref norm 2.220e+01)
Failing test case: test_trailing_zeros_do_not_change_filtfilt(
    filt=(array([1., 1.]), array([1.])),
    x=np.full(n, 1.0),
    zeros=1,
)
```

The smallest case is an FIR filter `b = [1, 1]` on a constant signal: writing
it as `b = [1, 1, 0]`, `a = [1, 0]` changes the output by 1.7 in norm.

## `filtfilt(method="gust")` with poles near zero

Gustafsson's initial conditions make the output independent of the direction
of time, so filtering the reversed signal and reversing the result must give
the same output. Campaign 3 found this relation broken by a filter with
poles close to zero but not exactly zero, which the fix for the trailing-zeros
case does not cover ([scipy/scipy#26755](https://github.com/scipy/scipy/issues/26755)).

The problem is well conditioned: its relative condition number is close to 1,
measured in 80-digit arithmetic by `benchmarks/gust_near_zero_poles.py`. The
lost digits therefore come from the algorithm, which solves a least-squares
system that becomes nearly singular as the poles shrink.

```python
--8<-- "examples/gallery/filtfilt_gust_near_zero_poles.py:16:"
```

```text
double pole at 0.1: reversal gap      1.18 eps
double pole at 0.01: reversal gap      54.6 eps
double pole at 0.001: reversal gap  7.64e+03 eps
double pole at 0.0001: reversal gap  2.15e+06 eps
double pole at 1e-06: reversal gap  2.15e+09 eps
```

## PyTorch's FFT at lengths with a large prime factor

This one is not a bug. Both FFTs pass every relation with the default
tolerance of 20 eps log2 N. The difference only shows when each length is
measured against an extended-precision DFT with `dp.calibrate`. At
N = 596 = 4 × 149, PyTorch's CPU FFT in float64 is about 50 times less
accurate than NumPy's. Over N = 2 to 1200, its worst error is
<!-- r:profile.torch.worst -->195 ε<!-- /r --> at N = <!-- r:profile.torch.worst_n -->596<!-- /r -->,
against <!-- r:profile.numpy.worst -->4.1 ε<!-- /r --> for NumPy (`benchmarks/error_profile.py`).

```python
--8<-- "examples/gallery/torch_fft_precision.py:15:"
```

```text
N=  512  numpy    1.0 eps   torch    0.9 eps
N=  596  numpy    2.3 eps   torch  133.6 eps
N=  600  numpy    1.2 eps   torch    1.4 eps
N= 1021  numpy    2.2 eps   torch    2.1 eps
```

The script uses 4 random signals per length. The full profile uses more,
which is why it finds a larger worst case. `dsp-props calibrate --compare`
turns this measurement into a check between two versions of a library.

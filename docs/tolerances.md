# Choosing a tolerance

Floating-point results can only be compared within a tolerance. A fixed `rtol`
is either too loose to catch a single-precision intermediate or too tight for a
large transform or an ill-conditioned recursion. dsp-props derives the
tolerance from an **error model** of the operation instead, and lets you
**calibrate** it against an extended-precision reference when the analytic
model does not describe a library.

!!! warning "Project rule: never widen a tolerance to make a test pass"
    When a relation fails, either the operator has a bug, or the error model is
    wrong. In the second case, fix the model and justify the change with an
    error analysis or a measurement; do not raise a constant until the failure
    goes away. `dp.calibrate` enforces this for measured tolerances: it refuses
    to absorb errors that rounding cannot explain.

## How a comparison works

Every relation ends in a normwise comparison
(`dsp_props.core.assert_close`):

$$
\|\hat y - y\|_2 \le \text{bound}(N, \text{dtype}, \|\text{ref}\|_2)
$$

where `bound` is the `bound(n, dtype, ref_norm)` method of a
[`ToleranceModel`](api/core.md), ε is the machine epsilon of the *expected*
precision (float32 at the coarsest), and the reference norm is chosen by the
relation: the norm of the expected output for transforms, the norm of the input
for filters and resamplers. The output must also have the expected shape,
contain only finite values, and be at least as precise as expected.

The expected precision is that of the inputs, which the generators draw in the
dtype given by the declaration's optional `precision` argument (`np.float64`
by default). It never comes from the output: an operator that silently returns
complex64 for float64 input fails with
`output dtype complex64 is less precise than the expected float64`, instead of
being judged with the float32 ε of its own output. An operator meant to run in
single precision is declared as such, for instance
`@dp.transform(inverse=np.fft.ifft, precision=np.float32)`: it then receives
float32 (or complex64) inputs, with the amplitudes float32 can hold (1, 1e-3
and 1e3, not 1e-100 or 1e100), and is checked with the float32 ε. A more
precise output than expected is always accepted. Using norms rather than element-wise comparisons follows the form of
the error bounds, which are normwise, and avoids spurious failures on output
samples close to zero.

Every model adds an absolute floor of N × `tiny` (the smallest normal number of
the dtype). It keeps exact zeros strict while tolerating underflow on signals
of amplitude 1e-100 or on subnormal values, where a purely relative tolerance
would be zero.

## FFT and transforms: `FFTTolerance`

For transforms, the model follows the O(ε log N) forward error bound of the
FFT (Schatzman 1996; Higham 2002, Thm. 24.2):

$$
\|\hat y - y\|_2 \le c\,\varepsilon\,\max(\log_2 N, 1)\,\|y\|_2 + N\,\mathrm{tiny}
$$

The constant `c` absorbs the constants of the analysis and the few chained
operations in a relation (for example `inverse(f(x))`).

| Where it is used | Default |
| --- | --- |
| `dp.transform(...)` | `FFTTolerance()`, `c = 20` |
| `dp.lti_filter()` without `transfer` | `FFTTolerance(c=1000.0)` |
| `dp.resampler(...)` | `FFTTolerance(c=1000.0)` |
| `dp.streaming()`, `dp.stateful(...)` | `FFTTolerance(c=1000.0)` |

Operators without a dedicated model get the same formula with a larger
constant. When the coefficients of a recursive filter are known, the model
below follows its conditioning instead.

## Recursive filters: `IIRTolerance`

For a filter y = (B/A) x computed by a direct-form recursion, the rounding
errors injected at each step are themselves filtered by 1/A(z), so they are
amplified by the gain of the recursion. The repository derives (and checks in
Coq, `proofs/IIRBound.v`) the first-order bound

$$
\|\hat y - y\|_2 \le c\,\varepsilon\,k\,\|h_A\|_1
\bigl(\|b\|_1 + \|a_{1:}\|_1\,\|h_{B/A}\|_1\bigr)\,\|x\|_2 + N\,\mathrm{tiny}
$$

in the style of Higham (2002), where

- h_A and h_{B/A} are the impulse responses of 1/A(z) and B(z)/A(z), so their
  ℓ1 norms grow as poles approach the unit circle;
- a_{1:} are the feedback coefficients (b and a are normalised by a_0);
- k = len(a) + len(b) − 1 grows with the order of the filter;
- c = 4 by default.

The second term in the parenthesis matters: with clustered poles near the unit
circle, the state grows far larger than the input and dominates the rounding.
For a FIR filter (a = [1]) the bound reduces to the error of a dot product.

`dp.lti_filter(transfer=(b, a))` uses `IIRTolerance(b, a)` by default. You can
also pass it explicitly, for instance to test an implementation against
coefficients you do not want to check as a transfer function:

```python
import numpy as np
from scipy import signal
import dsp_props as dp
from dsp_props.core import IIRTolerance

b, a = signal.butter(4, 0.2)
dp.lti_filter(tolerance=IIRTolerance(b, a))(lambda x: signal.lfilter(b, a, x))
```

The model's ingredients are exposed as attributes, which shows how fast the
tolerance grows with conditioning (bound for ‖x‖ = 1 and N = 256, float64):

```python
for order, wn in [(2, 0.25), (4, 0.2), (8, 0.05)]:
    b, a = signal.butter(order, wn)
    t = IIRTolerance(b, a)
    print(f"order {order}, wn={wn}: ||h_A||_1={t.h_a:.3g}  ||h_BA||_1={t.h_ba:.3g}  "
          f"gain={t.gain:.3g}  k={t.order}  bound/||x||={t.bound(256, np.float64, 1.0):.3g}")
```

```text
order 2, wn=0.25: ||h_A||_1=2.9  ||h_BA||_1=1.12  gain=5.27  k=5  bound/||x||=2.34e-14
order 4, wn=0.2: ||h_A||_1=18.4  ||h_BA||_1=1.34  gain=147  k=9  bound/||x||=1.18e-12
order 8, wn=0.05: ||h_A||_1=6.57e+06  ||h_BA||_1=1.64  gain=1.83e+09  k=17  bound/||x||=2.76e-05
```

The eighth-order Butterworth filter with a cut-off at 0.05 × Nyquist, computed
as a single transfer function, is ill-conditioned: the model allows a relative
error of about 3e-5, where a fixed `rtol=1e-10` would raise false alarms and a
fixed `rtol=1e-4` would hide a float32 recursion on the well-conditioned
filters. (In practice, such a filter should be implemented as second-order
sections; declare each section, or the cascade with its own tolerance.)

`dsp_props.core.recursion_gain(a, b)` computes ‖h_{B/A}‖_1 on its own; it is
also the tightest gain bound to declare with `max_gain=` (see the
[tutorial](tutorial.md#2-declare-the-transfer-function-and-a-gain-bound)).

!!! note "Stable filters only"
    The model assumes a stable declared filter. If A(z) has a pole on or outside
    the unit circle, `recursion_gain` returns infinity, so `IIRTolerance(b, a)`
    and `dp.lti_filter(transfer=(b, a))` raise `ValueError` instead of checking
    against an infinite tolerance. Declare unstable recursions with an explicit
    `tolerance=` and without `transfer`.

`dsp_props.core.longdouble_lfilter(b, a, x)` is a slow reference recursion in
extended precision, used by the tests to check that the bound holds.

## Calibration: `dp.calibrate`

An analytic bound cannot describe every library. An FFT library may switch
algorithm with the factorisation of N (mixed radix, naive small DFTs,
Bluestein), and its error then varies by two orders of magnitude between
neighbouring lengths. Over N = 2…1200, PyTorch's CPU FFT reaches 195 ε for some
lengths with large prime factors (N = 596 = 4 × 149), against about 4 ε for
NumPy and SciPy:

![Relative FFT error as a function of N, NumPy against PyTorch](figures/error_profile.png)

(`python benchmarks/error_profile.py` and `python benchmarks/plot_error_profile.py`
in the repository reproduce this profile.)

`dp.calibrate` measures the actual error of an implementation against an
extended-precision reference, length by length, and turns it into a tolerance
model:

```text
dp.calibrate(op, reference=longdouble_dft, lengths=range(1, 257), trials=6, *,
             margin=8.0, max_plausible=1e6, seed=0) -> CalibratedTolerance
```

| Argument | Meaning |
| --- | --- |
| `op` | the operator to measure, `y = op(x)` |
| `reference` | extended-precision reference with the same output shape; default `longdouble_dft`, a DFT computed by NumPy in `long double` |
| `lengths` | the lengths N to measure |
| `trials` | number of Gaussian-noise signals per length; each length also gets an impulse at a random position, a cosine of random frequency and phase, and a near-Nyquist alternating signal with a 1e-6 noise detail |
| `margin` | safety factor applied to the measured error (8 by default) |
| `max_plausible` | the largest error, in units of ε, that rounding can explain (1e6 by default); above it, calibration raises `CalibrationError` |
| `seed` | seed of the probe signals, so calibration is reproducible |

For each length, the worst relative error over the probe signals,
‖op(x) − ref(x)‖ / (ε ‖ref(x)‖) with ε that of the input's dtype, is recorded
as a ratio in units of ε. The result is a `CalibratedTolerance`:

$$
\text{bound}(N) = \max\bigl(\text{base.bound}(N),\;
\text{margin}\times\text{chained}\times\varepsilon\times\text{ratio}(N)\times\|y\|_2\bigr)
$$

- `base` is the analytic model (`FFTTolerance()` by default), which still
  covers the inputs calibration did not try: calibration can only **raise** a
  tolerance above the analytic one, never lower it;
- `chained = 2` accounts for relations that chain two operations, such as
  `inverse(f(x))`;
- `ratio(N)` is the measured ratio for N (at least 1); a length that was not
  measured falls back to the worst ratio over all measured lengths, which keeps
  the model conservative.

```python
import numpy as np
import dsp_props as dp

tol = dp.calibrate(np.fft.fft, lengths=range(1, 129))
print(tol.summary())
```

```text
calibrated on 128 lengths, worst ratio 2.3 eps
  N=103         2.3 eps
  N=113         2.2 eps
  N=107         2.2 eps
  N=101         2.1 eps
  N=127         2.0 eps
```

Pass the result as the `tolerance=` of a declaration:

```python
@dp.transform(inverse=np.fft.ifft, parseval=True, tolerance=tol)
def calibrated_fft(x):
    return np.fft.fft(x)
```

`CalibratedTolerance(ratios=..., margin=..., chained=..., base=...)` can also be
built directly, for example from ratios stored by an earlier measurement.

### Why calibration refuses implausible errors

Calibrating a wrong operator would turn its bug into a tolerance: an FFT that
is off by one bin has an error of order ‖y‖, and a tolerance fitted to it
would let every relation pass. So `calibrate` doubles as a sanity check: when
the error at some length exceeds `max_plausible` (10^6 ε, far above what
rounding can produce), it raises instead of returning a model.

```python
from dsp_props.core import CalibrationError

def shifted_fft(x):
    return np.roll(np.fft.fft(x), 1)

try:
    dp.calibrate(shifted_fft, lengths=range(1, 65))
except CalibrationError as exc:
    print(exc)
```

```text
error of 9.01e+15 eps at N=2: the operator disagrees with the reference beyond any plausible rounding error
```

Do not raise `max_plausible` to get past this error: it means the operator does
not compute what the reference computes.

### Caveats

- **The reference must be more precise than the operator.** `longdouble_dft`
  and `longdouble_lfilter` use NumPy's `long double`, which is 80-bit extended
  precision on x86-64 Linux, but is the same as `double` on some platforms
  (for example Windows and Apple silicon). There, the reference is no more
  accurate than a float64 operator, and the measured ratios understate its
  error. Check `np.finfo(np.longdouble).eps` before trusting a calibration.
- **ε is that of the input's dtype.** The probe signals are float64, so an
  implementation that computes in float32 is measured against the float64 ε,
  whether it returns float64 or complex64: its error (of order 1e8 ε for an FFT)
  exceeds `max_plausible` and calibration refuses it, since float64 input
  promises float64 accuracy. Calibration does not yet take a `precision`
  argument: check a float32 operator with `dp.verify` and `precision=np.float32`.
- **The default reference is a DFT.** To calibrate another operator, pass its
  own extended-precision reference, for instance
  `reference=lambda x: longdouble_lfilter(b, a, x)` for a recursive filter.
  `CalibratedTolerance` bounds the error relative to the norm of the reference
  output, as transform relations expect; the filter relations pass the norm of
  the input instead, so for filters `IIRTolerance` is the model to use, and a
  calibration is best read as a measurement to compare with it.

## Writing your own model

A tolerance model is any object with a `bound(n, dtype, ref_norm)` method
returning the admissible ℓ2 error; subclass `dsp_props.core.ToleranceModel`.
Derive it from an error analysis of the operation, include the N × `tiny`
floor, and check it against an extended-precision reference over the inputs the
generators produce, as `tests/test_iir.py` does for `IIRTolerance`.

## References

- M. C. Schatzman, "Accuracy of the discrete Fourier transform and the fast
  Fourier transform", *SIAM J. Sci. Comput.* 17(5), 1996.
- N. J. Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed.,
  SIAM, 2002.

# Choosing a tolerance

Floating-point results can only be compared within a tolerance. A fixed `rtol`
is either too loose to catch a single-precision intermediate or too tight for a
large transform or an ill-conditioned recursion. dsp-props scales the
tolerance with the operation instead, and lets you **calibrate** it against an
extended-precision reference. Only one model is derived from an error
analysis, the one for recursive filters (`IIRTolerance`, for the direct form I
recursion). The others have the shape of a known bound (ε log N for an FFT)
with an engineering constant chosen to avoid false alarms, not derived from
the analysis: `c = 20` for transforms and `c = 1000` for the operators without
a dedicated model. **Calibration is the recommended path when you need a
tight tolerance.**

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

For transforms, the model has the shape of the worst-case O(ε log N) normwise
forward error bound of the FFT (Higham 2002, Thm. 24.2; Schatzman 1996 shows
that the root-mean-square error grows only as O(ε √log N)):

$$
\|\hat y - y\|_2 \le c\,\varepsilon\,\max(\log_2 N, 1)\,\|y\|_2 + N\,\mathrm{tiny}
$$

The constant `c` is not the constant of the theorem, which depends on how the
implementation computes its twiddle factors and is not known for a given
library. It is an engineering choice, large enough for correct implementations
not to raise false alarms and to cover the few chained operations in a
relation (for example `inverse(f(x))`). The consequence is a blind spot: an FFT
whose error is a few tens of ε above a good one still passes (see
[What the relations don't see](blind-spots.md#an-accuracy-gap-between-the-default-tolerance-and-calibration)).

| Where it is used | Default |
| --- | --- |
| `dp.transform(...)` | `FFTTolerance()`, `c = 20` |
| `dp.lti_filter()` without `transfer` | `FFTTolerance(c=1000.0)` |
| `dp.resampler(...)` | `FFTTolerance(c=1000.0)` |
| `dp.streaming()`, `dp.stateful(...)` | `FFTTolerance(c=1000.0)` |

Operators without a dedicated model get the same formula with a larger
constant, `c = 1000`, which is not derived from any analysis of a filter, a
resampler or a streaming processor either: it is wide enough for their
correct implementations (in `benchmarks/validation.json`, 2000 runs of correct
implementations raised no false alarm, a measurement made before the latest
changes to the relations) and still far below the error of a structural bug.
It is a guard against structural bugs, not a model of the rounding error:
dsp-props derives no model for resamplers or streaming processors, because
their declarations do not say how they compute. A resampler's taps and a
processor's internal recursion are unknown to the declaration. For their
accuracy, calibrate them against a reference
([precision regressions](#precision-regressions-between-two-versions) show this
for a resampler). When the coefficients of a recursive filter are known, the
model below follows its conditioning instead.

## Recursive filters: `IIRTolerance`

For a filter y = (B/A) x computed by the direct form I recursion
y[n] = Σ b_k x[n−k] − Σ a_k y[n−k], the rounding
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
- c = 4 by default, a margin on top of the first-order bound.

This bound holds for the direct form I whatever the order of the summations, so it is the
default when the implementation is unknown. `scipy.signal.lfilter` and
`sosfilt` compute other structures, and the tests check this bound against
them (`tests/test_iir.py`, against `longdouble_lfilter`) without a proof.

### The structure of `lfilter`: `structure="df2t"`

`scipy.signal.lfilter` computes a transposed direct form II. Its states are
z_k[n] = z_{k+1}[n−1] + b_k x[n] − a_k y[n], and its output is
y[n] = z_1[n−1] + b_0 x[n]. Each state equation rounds three terms, and the
output equation two. An error injected into z_k reaches y k samples later and
is then filtered by 1/A(z). With s_k the impulse response from x to the state
z_k, whose transfer function is (B_k A − A_k B)/A (B_k and A_k are the
coefficients from index k on), the first-order bound is

$$
\|\hat y - y\|_2 \le c\,\varepsilon\,\|h_A\|_1 \Bigl[2\bigl(|b_0| + \|s_1\|_1\bigr)
+ 3 \sum_{k=1}^{p} \bigl(|b_k| + |a_k|\,\|h_{B/A}\|_1 + \|s_{k+1}\|_1\bigr)
+ \nu\,\bigl(\|b\|_1 + \|a_{1:}\|_1\,\|h_{B/A}\|_1\bigr)\Bigr]\|x\|_2
$$

where s_{p+1} = 0, and ν = 1 when a_0 ≠ 1, because `lfilter` then rounds the
coefficients when it divides them by a_0. The derivation is on paper only; it
is not checked in Coq.

```python
IIRTolerance(b, a, structure="df2t")   # for lfilter only
```

The direct form I bound multiplies every term by the length k of the whole
recursion. Here, each state adds only three roundings. On the five filters of
`benchmarks/calibrated_tolerance.py`, the df2t bound is
<!-- r:iir_structure.named_ratio_min -->1.1<!-- /r --> to <!-- r:iir_structure.named_ratio_max -->2.8<!-- /r --> times
smaller. On 80 random stable filters, it ranges from <!-- r:iir_structure.random_ratio_min -->0.62<!-- /r --> to
<!-- r:iir_structure.random_ratio_max -->5.7<!-- /r --> times smaller, with a median of <!-- r:iir_structure.random_ratio_median -->2.0<!-- /r -->
(`benchmarks/iir_structure.py`). It can be larger than the direct form I bound
when a state carries far more energy than the output. Each bound applies to
its own structure only.

On the IIR precision mutants (`mutants/precision_mutants.py`, `lfilter`
inside, 3 seeds, one bit per step), the df2t bound catches rounded
coefficients from <!-- r:iir_structure.iir_coefficients.df2t -->1.9e5 ε<!-- /r --> of measured loss, against
<!-- r:iir_structure.iir_coefficients.df1 -->7.1e5 ε<!-- /r --> with the default. A narrower arithmetic is caught at the same level with
both bounds, <!-- r:iir_structure.iir_arithmetic.df2t -->1.5e6 ε<!-- /r --> with df2t and
<!-- r:iir_structure.iir_arithmetic.df1 -->1.5e6 ε<!-- /r --> with the default: that loss comes from rounding
the input and output, not from the recursion.

Both bounds remain far above the error that `lfilter` actually makes: the
median ratio is <!-- r:iir_structure.random_gap_median -->220<!-- /r --> on the random filters. Most of
this gap comes from the worst-case assumption that every rounding error has
the same sign and adds up, not from the structure. A rigorous bound cannot
remove it. To get a tight tolerance, calibrate a trusted implementation
(see [calibration](#calibration-dpcalibrate)), which is <!-- r:calibrated.gain_min -->7<!-- /r --> to
<!-- r:calibrated.gain_max -->3400<!-- /r --> times tighter than the default on these five filters and four
resampling ratios.

`IIRTolerance` normalises b and a by a_0 before computing the gains. Up to
version 0.1.4, the gain of 1/A(z) was computed before this normalisation, so
the bound was |a_0| times too small (or too large when |a_0| < 1). Only filters
declared with a_0 ≠ 1 are affected.

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

`dsp_props.core.longdouble_lfilter(b, a, x)` is a slow direct form I reference
recursion in `long double`, used by the tests to check that the bound holds.
It is extended precision only where `long double` is (Linux x86-64), and,
unlike the DFT reference below, has no double-double fallback.

## Calibration: `dp.calibrate`

An analytic bound cannot describe every library. An FFT library may switch
algorithm with the factorisation of N (mixed radix, naive small DFTs,
Bluestein), and its error then varies by two orders of magnitude between
neighbouring lengths. Over N = 2…1200, PyTorch's CPU FFT reaches 195 ε for some
lengths with large prime factors (N = 596 = 4 × 149), against about 4 ε for
NumPy and SciPy. The figure depends on the CPU, not only on the versions: on an
AMD Ryzen 7 9800X3D with torch 2.15.0+cpu, the same length gives 1.6 ε. A
calibration is valid for the machine it was measured on:

![Relative FFT error as a function of N, NumPy against PyTorch](figures/error_profile.png)

(`python benchmarks/error_profile.py` and `python benchmarks/plot_error_profile.py`
in the repository reproduce this profile.)

`dp.calibrate` measures the actual error of an implementation against an
extended-precision reference, length by length, and turns it into a tolerance
model:

```text
dp.calibrate(op, reference=extended_dft, lengths=range(1, 257), trials=6, *,
             margin=8.0, max_plausible=1e6, seed=0) -> CalibratedTolerance
```

| Argument | Meaning |
| --- | --- |
| `op` | the operator to measure, `y = op(x)` |
| `reference` | extended-precision reference with the same output shape; default `extended_dft`: NumPy's FFT in `long double` where that is really extended (`longdouble_fft_is_extended()`), otherwise `doubledouble_dft`, an O(N²) DFT in double-double arithmetic |
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

- **The reference must be more precise than the operator.** NumPy's
  `long double` is 80-bit extended precision on Linux x86-64, but the same as
  `double` under MSVC and on macOS arm64, and NumPy 1.x computes the FFT of a
  `clongdouble` array in complex128. The default DFT reference, `extended_dft`,
  checks this (`longdouble_fft_is_extended()`) and falls back to
  `doubledouble_dft` (relative error below 1e-30, checked against mpmath in
  `tests/test_calibration.py`) where `long double` is not extended; pass
  `reference=doubledouble_dft` to use it everywhere. The fallback is O(N²), so
  calibrating long lengths is slower there. `longdouble_lfilter`
  has no such fallback: where `long double` is float64 it is no more accurate
  than the filter it checks.
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

## A versioned calibration: `tolerances.json`

A tolerance calibrated in a notebook is lost, and passing one by hand to every
declaration is easy to forget. `dsp-props calibrate` measures the declared
operators of a module or file and stores, for each, the error at every length
with the library versions it was measured with, in a file to commit next to
the tests:

```text
$ dsp-props calibrate examples/filters.py --tolerances tolerances.json
FAIL examples/filters.py:buggy_fft: error of 9.01e+15 eps at N=2: the operator disagrees with the reference beyond any plausible rounding error
ok   examples/filters.py:fft: worst error 3.04 eps over 256 lengths (0.1s)
ok   examples/filters.py:lowpass: worst error 45.2 eps over 256 lengths (0.8s)
skip examples/filters.py:resample_2_3: no built-in extended-precision reference for this MultirateSpec declaration (only dft and rdft transforms, and LTI filters with transfer=); pass reference=

wrote tolerances.json
```

```json
"filters:fft": {
 "kind": "transform",
 "reference": "extended_dft",
 "lengths": "1-256", "trials": 6, "seed": 0,
 "margin": 8.0, "chained": 2, "base_c": 0.0,
 "worst": 3.042,
 "ratios": {"1": 0.0, "2": 0.3898, "3": 0.4102, ...},
 "calibrated_with": {"python": "3.13.18", "platform": "Linux-x86_64", "dsp_props": "0.1.4",
                     "numpy": "2.5.3", "scipy": "1.18.1", "date": "2026-11-18"}
}
```

The built-in references are the extended DFT for `kind="dft"` and
`kind="rdft"` transforms and `longdouble_lfilter` for an LTI filter declared
with `transfer=(b, a)` (where `long double` is extended). Other declarations
are skipped; calibrate them in Python with their own reference. As with
`dp.calibrate`, an error no rounding explains is refused (`buggy_fft` above),
and stored ratios are rounded up, never down. The default is the tight
tolerance (`base=None`: no analytic floor); `--floor 20` keeps the default
transform model as a minimum.

Then point the checks at the file, and the stored calibration becomes the
tolerance of every operator it names, in place of the default model:

| Where | How |
| --- | --- |
| `dp.verify` | `tolerances="tolerances.json"` |
| `dsp-props check` | `--tolerances tolerances.json` |
| pytest | `--dsp-props-tolerances tolerances.json`, or the ini key `dsp_props_tolerances` |
| anywhere | the `DSP_PROPS_TOLERANCES` environment variable |

A tolerance given in the declaration (`tolerance=...`) still wins, and an
operator the file does not name keeps its default model. The report says
which applied (`Report.tolerance`: `"default"`, `"declared"` or
`"calibrated (tolerances.json)"`). Operators are named `module:attribute`
(`module:REGISTRY[key]` for a dictionary); a file given to `dsp-props check`
is named by its stem, as when pytest imports it, and an entry whose
attribute alone is unique in the file also matches.

In Python, `dp.ToleranceFile` does the same:

```python
tf = dp.ToleranceFile("tolerances.json")
tf.calibrate(my_fft, lengths=range(1, 513))       # default reference for the declaration
tf.calibrate(MyProcessor, reference=my_extended)  # any declaration, with its own reference
tf.save()
```

## Precision regressions between two versions

Recalibrating hides a loss of accuracy: the new tolerance simply follows the
new error. `--compare` measures each operator again, on the probe signals of
its calibration (same lengths, trials and seed), and reports the lengths
where the error grew by more than a factor (2 by default; errors below 1 ε
count as 1 ε, so that a length computed exactly does not turn rounding noise
into a regression). It writes nothing and exits with 1 on a regression,
which makes it a CI step after upgrading NumPy, SciPy or the library under
test:

Here the accurate radix-2 FFT of the test suite (`mutants/precision_mutants.py`)
was calibrated as version 1.4.0 of a package, and version 1.5.0 computes its
twiddle factors by recurrence:

```text
$ dsp-props calibrate mypkg.ffts --tolerances tolerances.json --compare
LOSS mypkg.ffts:fft: worst error 2.98 eps recorded, 14.8 eps now; 81 length(s) less accurate by more than x2
    N=198          1.16 eps ->       14.2 eps
    N=244          1.12 eps ->       13.6 eps
    N=230          1.12 eps ->       13.2 eps
    ...
    since the calibration: mypkg 1.4.0 -> 1.5.0
```

The versions recorded are Python's, the platform's, those of the usual
numerical libraries that are imported (NumPy, SciPy, PyTorch...) and that of
the top-level package defining the operator.

(`ToleranceFile.compare` in Python.) This is finer than the relations: the
FFT with recursive twiddles above passes every relation, even with the
calibrated tolerance (`tests/test_tolerances.py`).
The comparison is only as stable as the platform: calibrate and compare on
the same kind of machine.

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

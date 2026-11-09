# What the relations don't see

A relation detects a fault only if the fault breaks it. dsp-props is strongest
on faults of **structure** (state handling, edges, indexing, normalisation,
precision of intermediates) and weakest on faults of **value**, where an
expected output remains the only oracle. This page lists the known blind spots,
with what you can do about each.

For scale: the repository's mutation suite (`python mutants/run_mutants.py`)
has 31 deliberately buggy operators in six families, and all 31 are detected
with the default 200 inputs per relation. These mutants were written by hand in
the vocabulary of the relations, so they measure that the relations catch the
faults they were designed for. In the companion article, on 166
generated faults in FFT, `lfilter`, `upfirdn` and overlap-add subjects, dsp-props
detects 90 %, ported NumPy and SciPy test suites 96 %, and the two together
100 %: the sets are complementary, not nested. The misses below explain why.

## A wrong operator of the right class

The structural relations say that an operator *is* an LTI filter, a resampler,
a DFT; they do not say that it is *the one you meant*.

**A filter with wrong coefficients that is still linear, time-invariant and
causal passes every structural relation.** This is the one documented blind
spot of the mutation suite: a resonant IIR filter whose gain is wrong by 1 %
(`iir_wrong_gain`) is invisible to linearity, time invariance and causality. It
is detected only because its transfer function is declared:

```python
from scipy import signal
import dsp_props as dp

b, a = signal.butter(2, 0.25)

@dp.lti_filter()                    # passes: still LTI and causal
def off_by_one_percent(x):
    return signal.lfilter(1.01 * b, a, x)

@dp.lti_filter(transfer=(b, a))     # fails the transfer-function relation
def off_by_one_percent(x):
    return signal.lfilter(1.01 * b, a, x)
```

The [tutorial](tutorial.md#a-bug-the-structural-relations-cannot-see) shows
the full output. **Declare `transfer=(b, a)` whenever you know the
coefficients.**

The same holds in other classes, and some of these gaps have no declaration
that closes them:

- a resampler that consistently ignores the last input sample satisfies
  linearity and periodic time invariance, since both sides of each relation
  ignore it identically;
- a transform declared with `kind="generic"` is only checked for linearity,
  Parseval (if declared) and inversion: a DCT of the wrong type that is
  orthonormal and correctly inverted passes;
- a transform with no declared `inverse` gets neither the inversion nor the
  convolution-theorem relation;
- `real_input_hermitian=False` (the default) skips the Hermitian-symmetry
  relation for a complex DFT.

## Parameters are not generated

A declaration fixes the operator's parameters (filter coefficients, rates, the
inverse) and the relations vary **only the input signal**. A fault that shows
only for other parameter values is invisible. In the article's comparison,
every fault dsp-props missed is of this kind: seven `lfilter` faults break only
filters of order 0 or 1, or filters whose numerator is shorter than the
denominator, and the declared filters had orders 4 and 2; four `upfirdn` faults
change only its default arguments, which the declarations never used.

**Declare each operator for several representative parameter sets, including
degenerate ones**: order 0 and 1 filters, a numerator shorter than the
denominator, rate 1, default arguments. A module-level dictionary of declared
operators is picked up by `dsp-props check`:

```python
from scipy import signal
import dsp_props as dp

FILTERS = {
    f"butter{order}": dp.lti_filter(transfer=(b, a))(
        lambda x, b=b, a=a: signal.lfilter(b, a, x))
    for order in (1, 2, 4)
    for b, a in [signal.butter(order, 0.2)]
}
```

## Inputs that are never generated

The generators are aimed at the inputs that break DSP code, but they do not
cover everything:

- **dtype and shape.** Generated signals are 1-D arrays of the declared
  `precision` (float64 or complex128 by default, float32 or complex64 with
  `precision=np.float32`). Complex signals go to every transform not declared
  `real_input_only` (`kind="rdft"` is real-input only by default); filters,
  resamplers and streaming processors get real signals only. Integer inputs,
  multi-channel arrays and the `axis=` argument are never exercised, and an
  operator is checked in one precision per declaration (see
  [below](#the-expected-precision-is-the-declared-one)).
- **Length.** Lengths go up to `max_n` (512 by default; the convolution theorem
  caps at 256). A bug that appears only for longer signals, such as a code path
  for large N, is missed unless you raise `max_n` in the declaration.
- **Random search.** Each relation sees `max_examples` inputs (200 by default).
  A fault that needs a very specific input can escape in some runs. Raise
  `max_examples` (or `--examples` on the command line) for a deeper search;
  `derandomize=True` makes runs reproducible, but also fixes which inputs are
  tried.
- **Block splits.** See [below](#block-splits-are-sampled-not-enumerated).
- **Resampler edges.** The DC-gain and alias-rejection relations look at the
  middle half of a long output, away from edge transients; the edges are only
  covered by linearity and periodic time invariance.

## Errors within the tolerance

A relation passes whenever the error is below the [tolerance
model](tolerances.md). Accuracy losses smaller than the model are invisible by
design, and they can be large in absolute terms:

- a calibrated tolerance accepts what calibration measured: PyTorch's CPU FFT,
  at up to 195 ε, passes with a tolerance calibrated on it (see
  [below](#an-accuracy-gap-between-the-default-tolerance-and-calibration) for
  the general case);
- `IIRTolerance` grows with the conditioning of the recursion, so an
  ill-conditioned filter (an eighth-order Butterworth at 0.05 × Nyquist as a
  single transfer function) is checked to about 3e-5 relative error only;
- comparisons are normwise: an error concentrated in a few output samples is
  compared with the norm of the whole output, so a small localised fault in a
  large-norm output can stay below the bound.

The tolerance must not be widened to make a relation pass (see the [project
rule](tolerances.md)); by the same token, a passing relation only certifies
accuracy up to the model.

## Limits of the checks themselves

Recent fixes closed false alarms and gaps in the relations, but each left a
limit behind.

### The impulse response is measured over a capped window

**Not detected.** The transfer-function relation measures the impulse response
over L samples, with L long enough for it to decay to about 1e-17 but capped at
2^16. A filter whose response is wrong only beyond L samples is not seen. With a
pole close to the unit circle the response has not decayed by L, and the
relation adds a bound on the truncated tail (`tail_l1_bound`) to the tolerance,
which makes the check coarse: for the DC blocker b = [1, −1], a = [1, −0.9999],
L = 2^16 and the tail bound is 1.4e-3, so the measured response is only checked
to about 1.4e-3 relative to its norm: a correct `lfilter` multiplied by 1.001
passes, by 1.002 fails.

**Why.** Measuring longer would cost O(L log L) per check and memory, and the
tail must be bounded rather than ignored, or a correct `lfilter` would fail.

**What to do.** For a filter with poles very close to the unit circle, add an
expected-value test of its response at the frequencies that matter (for a DC
blocker, the gain near Nyquist). A long response that hides a fault beyond L
samples needs such a test too.

### The expected precision is the declared one

**Not detected.** The tolerance uses the ε of the declared `precision`
(float64 by default), and the output must be at least that precise. An
operator declared `precision=np.float32` is checked only with float32 inputs
of amplitude 1, 1e-3 and 1e3, so its float64 path, and amplitudes float32
cannot hold, are never checked. An operator that returns a float64 array but
computes part of its work in float32 is caught only when its error exceeds the
float64 tolerance, not by its dtype. `dp.calibrate` takes no `precision`: it
measures in units of the input's ε, so a float32 operator calibrated on its
float64 probe signals is refused as implausible.

**Why.** One declaration fixes one precision, and the relations cannot see how
an output was computed, only its value and dtype.

**What to do.** Declare an operator meant to run in both precisions twice, once
with each `precision`. Check a float32 operator with `dp.verify`, not
`dp.calibrate`.

### The IIR reference depends on `long double`

**Not detected.** `longdouble_lfilter`, the reference the tests use to check
`IIRTolerance`, runs in NumPy's `long double`, which is float64 under MSVC and
on macOS arm64. There it is no more accurate than the filter it checks, so a
calibration against it (`reference=lambda x: longdouble_lfilter(b, a, x)`)
understates the error. The DFT reference of `dp.calibrate` falls back to a
double-double DFT on those platforms; the IIR reference has no such fallback.

**Why.** The double-double fallback was written for the DFT only.

**What to do.** Calibrate IIR filters on Linux x86-64, or check
`np.finfo(np.longdouble).eps < np.finfo(np.float64).eps` first.

### Block splits are sampled, not enumerated

**Not detected.** `block_splits` draws one of three kinds of split: arbitrary
cut points (up to max(N, 16) blocks, empty and one-sample blocks included),
blocks of one sample, and blocks of a fixed size B. In runs of 200 examples
with `max_n = 512`, 29 % of the splits drawn have more than 8 blocks and the
median is 3 blocks. A processor that fails only on a specific irregular
pattern of many blocks (say, an empty block right after its tenth call) can
escape, since such a pattern is neither regular nor likely among arbitrary
cuts. `benchmarks/campaign.py` keeps its own copy of the generator, still
capped at 9 blocks.

**Why.** The number of splits of N samples grows exponentially; arbitrary cuts
come first so that counterexamples shrink towards few blocks.

**What to do.** Raise `max_examples` for streaming processors, and add a
regression test with the block pattern your application actually produces.

### An accuracy gap between the default tolerance and calibration

**Not detected.** The default transform tolerance is 20 ε log2 N (relative to
the output norm), that is 20 ε at N = 2 and 180 ε at N = 512, while NumPy's FFT
measures at most 3.0 ε over N = 1…512. An FFT a few times to a few tens of
times less accurate than NumPy therefore passes every relation. Measured on a
radix-2 FFT whose twiddle factors come from the recurrence
w^k = w^(k−1) · w (odd parts by `np.fft.fft`): its worst calibrated error over
N = 1…512 is 38 ε (at N = 452), and it passes all six transform relations with
the default tolerance. Whether a degraded FFT passes depends on how its error
grows with N: the same harness shows an error of a constant 20 ε at every N
failing, because it exceeds the bound at small N. Calibration does not close
the gap on its own: `dp.calibrate` refuses only errors above `max_plausible`
(1e6 ε), so calibrating a degraded implementation turns its error into its
tolerance.

**Why.** `c = 20` is chosen so that correct implementations never raise a false
alarm, and `max_plausible` so that calibration refuses wrong operators, not
imprecise ones.

**What to do.** When accuracy matters, calibrate the implementation and
compare its worst ratio (`tol.worst`, `tol.summary()`) with a trusted library
calibrated on the same lengths, or lower `max_plausible` to the accuracy you
require, so that calibration refuses anything worse.

### How large an accuracy loss the relations detect

`python benchmarks/precision_mutants.py` measures it with precision mutants
(`mutants/precision_mutants.py`): operators that compute the right thing with
fewer significand bits in one place, from 0 to 24 bits dropped (1 to about
1e7 ε). The loss is measured against an accurate reference, as the worst
relative error over N = 1…512 in units of ε; detection is `dp.verify` with
the default tolerance, 200 inputs per relation, 3 seeds
(`benchmarks/precision_mutants.json`). A family is caught "from" a loss when
that level and every larger one fail on all 3 seeds.

| Class and fault | Not caught up to | Caught from | By |
| --- | --- | --- | --- |
| FFT, twiddle table rounded | 21 ε | 90 ε | circular shift, inversibility, Parseval, convolution |
| IIR (order 4, poles at 0.95), coefficients rounded | 1.9e5 ε | 7.1e5 ε | transfer function |
| IIR, input and output rounded | 6.9e5 ε | 3.0e6 ε | linearity |
| 2/3 resampler, filter taps rounded | 6.2e7 ε (24 bits) | never | — |
| 2/3 resampler, input and output rounded | 6.7e3 ε | 1.1e5 ε | linearity |
| Streaming moving sum, state rounded between blocks | 750 ε | 7.1e3 ε | block invariance |

Realistic precision faults in an FFT, same protocol:

| FFT mutant | Loss | Caught (seeds) |
| --- | --- | --- |
| radix-2, twiddles by the recurrence w^k = w^(k−1) · w | 19 ε | 0/3 |
| direct DFT with exp(−2iπkn/N), argument not reduced mod N | 805 ε | 3/3 |
| one Goertzel recursion per bin | 1.4e4 ε | 3/3 |
| radix-2, twiddle table in single precision | 2.5e8 ε | 3/3 |

NumPy's FFT and the accurate versions of these mutants measure 2 to 3.5 ε and
pass. Single-precision IIR coefficients (7e9 ε) are caught by the transfer
function relation.

What this says: for transforms, the relations tell an FFT a few tens of ε
worse than NumPy from a correct one only from about 100 ε, the scale of the
default tolerance. For the other classes the detectable loss is three to
five orders of magnitude larger: the IIR bound is derived but grows with the
order and the ℓ1 norms of the impulse responses (poles at 0.95 make it large),
and filters without a dedicated model, resamplers and streaming processors use
c = 1000. A fault confined to the coefficients of a filter whose response is
not declared is never caught: the mutant is a different, equally valid,
operator of the same class (the "wrong operator of the right class" blind
spot above), whatever the loss.

### The same faults against other oracles

`python benchmarks/precision_oracles.py` runs the precision mutants against
four oracles, same levels and seeds (`benchmarks/precision_oracles.json`):

- **default**: the relations with the default tolerances (the table above);
- **fixed 1e-7**: the same relations with a relative tolerance of 1e-7 on
  the L2 error, the default `rtol` of `numpy.testing.assert_allclose`;
- **calibrated**: the same relations with a tolerance calibrated on the
  trusted implementation of the class (`numpy.fft.fft`, `lfilter`,
  `resample_poly`, the moving sum) against a long double reference, without
  analytic floor: `CalibratedTolerance(ratios, base=FFTTolerance(c=0))`;
- **differential**: `numpy.testing.assert_allclose(mutant(x), trusted(x))`
  with its defaults (elementwise, `rtol = 1e-7`, `atol = 0`) on the
  calibration probe signals at 18 lengths. Unlike the three others, it needs
  a trusted implementation to compare with.

Smallest loss caught on all 3 seeds, with every larger level also caught:

| Family | Default | Fixed 1e-7 | Calibrated | Differential |
| --- | --- | --- | --- | --- |
| FFT, twiddles rounded | 90 ε | never | 21 ε | 240 ε |
| IIR, coefficients rounded | 7.1e5 ε | never | 1.7e3 ε | 7.1e5 ε |
| IIR, input and output rounded | 3.0e6 ε | never | 2.9e3 ε | 190 ε |
| Resampler, taps rounded | never | never | never | 1.5e7 ε |
| Resampler, input and output rounded | 1.1e5 ε | never | 430 ε | 100 ε |
| Streaming, state rounded | 7.1e3 ε | never | 28 ε | 750 ε |

No oracle raised a false alarm on the correct implementations (level 0: the
trusted implementations, and for the FFT a direct DFT with accurate twiddles,
3.5 ε against 2.2 ε for NumPy), over 3 seeds. The calibrated ratios of the
trusted implementations are 2.2 ε (FFT), 27 ε (IIR), 17 ε (resampler) and
2.0 ε (moving sum).

What this says:

- A fixed relative tolerance of 1e-7 (4.5e8 ε) catches none of these faults,
  up to 24 bits dropped.
- With a tolerance calibrated on a trusted implementation, the relations catch
  losses two to four orders of magnitude smaller than with the default models,
  without a false alarm here: 21 ε for the FFT, 28 ε for the streaming
  processor, 430 ε to 3e3 ε for the filters. The error model, not the
  relations, sets what is detectable.
- The differential `assert_allclose` is uneven: elementwise, it is strict on
  the samples close to zero, so it catches some faults earlier than the
  calibrated relations (IIR and resampler arithmetic) and others much later
  (IIR coefficients), with results that depend on the probe signals (between
  1 and 2 seeds out of 3 over a wide range of levels). It is the only oracle
  that sees rounded resampler taps, because it compares with the very filter
  the mutant was meant to implement.
- By default `dp.calibrate` keeps the default model as a floor
  (`base=FFTTolerance()`, 20 ε log2 N): it widens a tolerance where the
  implementation measures worse and never tightens it. The calibrated column
  uses `dp.calibrate(..., base=None)`, which keeps no analytic floor.
- What the calibrated column assumes: a trusted implementation of the same
  operator and an extended-precision reference to measure it, the same
  ingredients as a differential test. Its advantage over `assert_allclose`
  is a threshold that follows the size of the loss and does not depend on the
  probe signals, not the absence of a reference.
- The derived IIR model is pessimistic: on the five filters below it is 7 to
  3000 times looser than a calibrated tolerance (first-order worst case,
  growing with the ℓ1 norms of the impulse responses).

### Does a calibrated tolerance hold for other implementations?

`python benchmarks/calibrated_tolerance.py` calibrates, without analytic
floor, a trusted implementation against long double, then checks other
correct implementations of the same operator with that tolerance (200 inputs
per relation, 3 seeds; `benchmarks/calibrated_tolerance.json`). Worst errors
are measured against long double over N = 1…512 for the FFTs, over 100
lengths up to 512 otherwise.

| Trusted (calibrated on) | Other implementation | Its worst error | Seeds failing |
| --- | --- | --- | --- |
| `numpy.fft.fft` (2.2 ε) | `scipy.fft.fft` | 2.0 ε | 0/3 |
| | radix-2, accurate twiddles | 4.8 ε | 0/3 |
| | direct DFT, accurate twiddles | 4.5 ε | 0/3 |
| | `torch.fft.fft` (CPU, torch 2.15) | 156 ε | 3/3 |
| `lfilter`, five filters (27 to 275 ε) | `sosfilt(tf2sos(b, a))` | 47 to 631 ε | 0/3 each |
| `resample_poly`, 2/3, 3/2, 5/4, 1/4 (1.2 to 57 ε) | direct polyphase convolution | 0.9 to 94 ε | 0/3 each |
| moving sum (2.0 ε) | `np.convolve` with carried history | 1.5 ε | 0/3 |

The filters are the resonant fourth-order filter of the mutants,
`butter(4, 0.2)`, `cheby1(6, 1, 0.3)`, `ellip(4, 0.5, 40, 0.25)` and
`butter(2, 0.01)`. `sosfilt` measures up to 2.7 times worse than `lfilter` on
the same filter and still passes: the calibrated bound carries a margin of
16 (8, times 2 for chained operations).

The PyTorch FFT is the only one flagged. It is correct, but up to 156 ε where
NumPy stays under 2.7 ε (N = 412; 82 ε against 1.1 ε at N = 354): a tolerance
calibrated on NumPy is an accuracy requirement, and PyTorch does not meet it.
Calibrate PyTorch on itself to test it for correctness rather than accuracy.

A first version of the `np.convolve` moving sum used `mode="valid"`. It failed
block invariance on every seed: NumPy swaps the arguments when the signal is
shorter than the kernel, so an empty block returned two samples. That was a
real bug, and the table shows the fixed version.

Detection thresholds on the same filters and ratios (graded mutants, 0 to 24
bits in steps of 4, smallest loss caught on every seed from there on; no false
alarm at 0 bits):

| Setting | Fault | Default | Calibrated |
| --- | --- | --- | --- |
| resonant order 4 | coefficients / arithmetic | 6.6e5 / 9.8e6 ε | 2.5e3 / 2.9e3 ε |
| `butter(4, 0.2)` | coefficients / arithmetic | 2.3e4 / 1.9e6 ε | 3.3e3 / 1.3e5 ε |
| `cheby1(6, 1, 0.3)` | coefficients / arithmetic | 6.1e6 / 2.6e7 ε | 4.3e4 / 1.1e5 ε |
| `ellip(4, 0.5, 40, 0.25)` | coefficients / arithmetic | 1.2e6 / 7.5e5 ε | 2.9e3 / 3.0e3 ε |
| `butter(2, 0.01)` | coefficients / arithmetic | 1.4e7 / 3.3e7 ε | 2.5e5 / 1.7e6 ε |
| resampler 2/3 | arithmetic | 4.6e5 ε | 1.7e3 ε |
| resampler 3/2 | arithmetic | 2.6e3 ε | 130 ε |
| resampler 5/4 | arithmetic | 2.2e3 ε | 180 ε |
| resampler 1/4 | arithmetic | 1.4e6 ε | 5.5e3 ε |

Steps of 4 bits make these thresholds coarser (a factor 16) than the 2-bit
steps of the previous tables. The gain of calibration varies from 7 to 3400
with the filter: it is largest where the derived model is most pessimistic
(poles close to the unit circle, high order).

## Hidden state

Each relation call builds its inputs and calls the operator several times. An
operator whose output depends on previous calls (a global accumulator, a cache
that is not invalidated) is caught when the leak breaks a relation, as in the
`rate_state_leak` mutant, or when a failing input passes once Hypothesis
replays it, which the report states as a "non-deterministic operator". Neither
is guaranteed: state that changes the output only rarely may go unnoticed.
Streaming processors get a fresh instance for each side of the block-invariance
relation, so state carried *between instances* is not checked.

## Unstable declarations

`IIRTolerance` and the transfer-function relation assume a stable declared
filter: with a pole on or outside the unit circle the error bound would be
infinite and every relation would pass vacuously. `dp.lti_filter(transfer=...)`
and `IIRTolerance(b, a)` therefore raise `ValueError` for an unstable B(z)/A(z).
Declare unstable recursions with an explicit `tolerance=` and without `transfer`.

## Things outside the scope

There are no relations for nonlinear operators, estimators (bias, variance
against the Cramér-Rao bound), multirate identities beyond the L/M resampler
(noble identities), float32/float64 consistency, denormal handling, or
fixed-point code. Some of these are on the roadmap.

## Complementary, not competing

In practice the method and expected-value tests combine: a library keeps its
regression tests, which pin values and parameter combinations, and adds
relations, which need no values and explore inputs no one thought to write
down. The cost is one declaration per operator and a few seconds per run.

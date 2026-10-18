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

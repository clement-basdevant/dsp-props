# Test your filter in 5 minutes

This page is a list of tasks, each with the shortest code that does it, so
you can find what you need and go straight to it. The [tutorial](tutorial.md)
explains the same steps in more detail. Every output below comes from a real
run with dsp-props 0.1.4, NumPy 2.5 and SciPy 1.18.1.

```bash
python -m pip install "dsp-props[scipy]"
```

## Check a filter I wrote

Put one decorator on each operator. It says what the operator is meant to
be, here a linear time-invariant filter with a known transfer function:

```python
# my_filters.py
import numpy as np
from scipy import signal
import dsp_props as dp

b, a = signal.butter(4, 0.2)
h = np.ones(5) / 5


@dp.lti_filter(transfer=(b, a))
def lowpass(x):
    return signal.lfilter(b, a, x)


@dp.lti_filter(transfer=(h, [1.0]))
def smoother(x):
    return np.convolve(x, h, mode="same")
```

Then run every declared operator in the file:

```text
$ dsp-props check my_filters.py
ok   my_filters.py:lowpass  (4 relations, 3.1s)
FAIL my_filters.py:smoother  (4 relations, 3.1s)
      ok   linearity
      FAIL time invariance
           Hypothesis found 2 distinct failures. (2 sub-exceptions)
           minimal input: ([1], 1)
      FAIL causality
           causality (k=1): L2 error 2.000e-01 exceeds tolerance 1.532e-14 (N=1, ref norm 3.449e+00)
           minimal input: ([1, 0, 0], 1, 1)
      FAIL transfer function
           transfer function: ||H_measured - H_declared|| = 1.327e+00 > 2.083e-14; worst bin w = 0.0000 rad/sample: measured 0.6+0j, declared 1+0j

1/2 operators pass
```

The exit code is 1 if any operator fails. `mode="same"` centres the
convolution, so the smoother looks two samples ahead (causality fails) and
loses the edges (the DC gain at the start is 0.6, not 1). Each minimal input
is as small as Hypothesis could make it: with `[1, 0, 0]`, changing the
second sample changes the first output.

From Python, `dp.verify(lowpass)` returns the same report and raises
`PropertyViolation` on a failure. The declarations available:

| I have | Decorator |
| --- | --- |
| a filter `y = f(x)` from zero state | `dp.lti_filter(transfer=(b, a))`, or `dp.lti_filter()` without known coefficients |
| an FFT, a real FFT, a DCT | `dp.transform(inverse=..., kind="dft" / "rdft" / "generic")` |
| a rational resampler | `dp.resampler(up, down)` |
| a block processor with a `process(block)` method | `dp.streaming()` |
| a function `(block, state) -> (out, state)` | `dp.stateful(func, init_state)` |

What each one checks: [relations catalogue](relations.md).

## Run the checks in my test suite

Declared operators become pytest tests, one per relation. List operators
defined elsewhere in `dsp_props_operators`:

```python
# tests/test_filters.py
from my_filters import lowpass, smoother

dsp_props_operators = [lowpass, smoother]
```

```text
$ pytest tests/test_filters.py --dsp-props-profile fast
FAILED tests/test_filters.py::smoother::time-invariance - smoother violates time in...
FAILED tests/test_filters.py::smoother::causality - smoother violates causality
FAILED tests/test_filters.py::smoother::transfer-function - smoother violates trans...
3 failed, 5 passed in 2.86s
```

`fast` runs 30 inputs per relation, `default` 200 and `nightly` 2000. Details:
[running with pytest](pytest.md).

## Keep a counterexample as a regression test

Add `--dsp-props-regressions` (or `dsp-props check --regressions DIR`). Each
minimal counterexample is written as a plain pytest test, which replays that
exact input without Hypothesis:

```python
# test_regression_smoother.py (generated)
# causality (k=1): L2 error 2.000e-01 exceeds tolerance 2.431e-14 (N=1, ref norm 5.474e+00)
def test_causality_880c4db8():
    inputs = (np.full(3, 1.0, dtype=np.float64), 1, 1.0)
    replay(OPERATOR, "causality", inputs)
```

Commit the file: the test fails while the bug is there and passes once it is
fixed. Details: [regression tests](pytest.md#regression-tests-from-counterexamples).

## Get a tighter tolerance from a trusted implementation

The default tolerances are wide enough for any correct implementation. To
measure the error of yours against an extended-precision reference, and
check against that measurement instead:

```text
$ dsp-props calibrate my_filters.py:lowpass --tolerances tolerances.json
ok   my_filters.py:lowpass: worst error 14.4 eps over 64 lengths (0.1s)
wrote tolerances.json
$ dsp-props check my_filters.py:lowpass --tolerances tolerances.json
ok   my_filters.py:lowpass  (4 relations, 3.2s, calibrated tolerance)
```

(This run used `--lengths 1-64`; without it, lengths 1 to 256 are measured.)
Commit `tolerances.json`; it records the library versions it was measured
with. Details: [a versioned calibration](tolerances.md#a-versioned-calibration-tolerancesjson).

## Find out whether an upgrade made my code less accurate

After upgrading NumPy, SciPy or your own code, measure again on the same
probe signals and compare:

```text
$ dsp-props calibrate my_filters.py:lowpass --tolerances tolerances.json --compare
ok   my_filters:lowpass: worst error 14.4 eps recorded, 14.4 eps now; no precision regression  (0.1s)
```

A length whose error is now more than twice the recorded one is reported as
a regression. This catches precision losses that no relation sees, such as a
radix-2 FFT computing its twiddle factors by recurrence. Details:
[precision regressions](tolerances.md#precision-regressions-between-two-versions).

## Test every filter my code can build, not just one

A declaration fixes the coefficients. If the bug only shows for some
coefficients, draw them too with `dp.family`. This hand-written filter forgets
to divide by `a[0]`, which `butter` always sets to 1:

```python
import numpy as np
import dsp_props as dp

def by_hand(b, a):
    def run(x):
        y = np.zeros(len(x))
        for n in range(len(x)):
            acc = sum(b[k] * x[n - k] for k in range(len(b)) if n >= k)
            acc -= sum(a[k] * y[n - k] for k in range(1, len(a)) if n >= k)
            y[n] = acc  # bug: no division by a[0]
        return y
    return dp.lti_filter(transfer=(b, a))(run)

my_filters = dp.family(dp.generators.iir_coefficients(max_order=4))(by_hand)
print(dp.verify(my_filters, seed=0, raise_on_failure=False))
```

```text
dsp-props report for by_hand
  ...
  FAIL transfer function
       transfer function: ||H_measured - H_declared|| = 2.121e+00 > 9.421e-16; worst bin w = 0.0000 rad/sample: measured 1+0j, declared 0.5+0j
       minimal input: (([1], [2]), None)
```

Linearity fails too (output elided). The counterexample is the filter
`b = [1]`, `a = [2]`. Details:
[operator families](relations.md#operator-families-dpfamily).

## Check a block processor

Real-time code filters block by block. `dp.streaming()` checks that any split
into blocks, including empty and one-sample blocks, gives the output of a
single call. The [gallery](gallery.md#an-empty-block-in-a-streaming-filter)
shows it finding a SciPy bug; the [tutorial](tutorial.md#4-streaming-and-stateful-processors)
walks through it.

## Know what the checks cannot see

A filter that is wrong but still linear, time-invariant and causal passes
unless you declare its transfer function, and some precision losses pass
every relation. [What the relations don't see](blind-spots.md) lists these
cases and what closes each one.

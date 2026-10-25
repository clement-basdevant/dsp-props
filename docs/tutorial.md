# Tutorial: test your own filter

This tutorial takes a second-order low-pass filter from a bare declaration to a
check in continuous integration. Every snippet below was run with dsp-props
0.1.1, NumPy 2.5 and SciPy 1.18.1, and the outputs are copied from those runs.
Inputs are random, so your counterexamples may differ slightly; pass `seed=0`
or `derandomize=True` to `dp.verify` to make a run reproducible.

```bash
python -m pip install "dsp-props[scipy]"
```

## 1. Declare an LTI filter

`dp.lti_filter` declares a function `y = f(x)` that applies a linear
time-invariant filter **from zero initial state** to a 1-D signal and returns an
output of the same length. The decorator only attaches a declaration to the
function (as `f.__dsp_props__`); the function itself is unchanged.

```python
import numpy as np
from scipy import signal
import dsp_props as dp

b, a = signal.butter(2, 0.25)          # 2nd-order low-pass biquad

@dp.lti_filter()
def lowpass(x):
    return signal.lfilter(b, a, x)

print(dp.verify(lowpass))
```

```text
dsp-props report for lowpass
  ok   linearity
  ok   time invariance
  ok   causality
```

With no other argument, three relations are checked on generated signals:

- **linearity**: `f(a x + b y) = a f(x) + b f(y)` for real coefficients;
- **time invariance**: prepending `s` zeros to the input prepends `s` zeros to
  the output;
- **causality**: changing `x[k:]` leaves `y[:k]` unchanged.

Pass `causal=False` for a non-causal filter (`filtfilt`, a centred FIR): the
causality relation is then skipped, and time invariance compares the delayed
output after discarding its first `s` samples.

## 2. Declare the transfer function and a gain bound

The three relations above say that the function *is* an LTI filter, not that it
is *the right one*. If you know the coefficients the implementation is meant to
realise, declare them:

```python
import numpy as np
from scipy import signal
import dsp_props as dp
from dsp_props.core import recursion_gain

b, a = signal.butter(2, 0.25)
gain = recursion_gain(a, b)            # ||h||_1, the l1 norm of the impulse response
print(f"l1 gain: {gain:.4f}")

@dp.lti_filter(transfer=(b, a), max_gain=gain)
def lowpass(x):
    return signal.lfilter(b, a, x)

print(dp.verify(lowpass))
```

```text
l1 gain: 1.1181
dsp-props report for lowpass
  ok   linearity
  ok   time invariance
  ok   causality
  ok   transfer function
  ok   bounded gain
```

- `transfer=(b, a)` adds the **transfer function** relation: the DFT of the
  measured impulse response (long enough for it to decay) must match
  `B(e^{jω}) / A(e^{jω})` on the same frequency grid. It also switches the
  default tolerance to `IIRTolerance(b, a)`, which grows with the order of the
  filter and the conditioning of its recursion (see
  [Choosing a tolerance](tolerances.md)). This relation needs SciPy.
- `max_gain=G` adds the **bounded gain** relation (a bounded gain on
  N ≤ `max_n`, not a proof of BIBO stability): for each generated length `N`,
  dsp-props builds the worst-case input bounded by 1, `x[k] = sign(h[N-1-k])`, and checks that no
  output sample exceeds `G`. The tightest valid bound is the ℓ1 norm of the
  impulse response, which `dsp_props.core.recursion_gain(a, b)` computes.

Other arguments of `lti_filter`: `tolerance=` (any
[`ToleranceModel`](api/core.md)) and `max_n=` (the longest generated signal,
512 by default).

## 3. Read a counterexample

Here is a filter whose author wanted a two-tap smoother but used the *next*
sample instead of the previous one:

```python
import numpy as np
import dsp_props as dp

@dp.lti_filter()
def lookahead(x):
    # Bug: uses the next sample, so it is not causal.
    return 0.5 * x + 0.5 * np.append(x[1:], 0.0)

dp.verify(lookahead)
```

```text
dsp_props.report.PropertyViolation: dsp-props report for lookahead
  ok   linearity
  FAIL time invariance
       time invariance (delay 1): L2 error 5.000e-01 exceeds tolerance 2.220e-13 (N=2, ref norm 1.000e+00)
       minimal input: ([1], 1)
  FAIL causality
       causality (k=1): L2 error 5.000e-01 exceeds tolerance 5.361e-13 (N=1, ref norm 2.414e+00)
       minimal input: ([1, 0], 1, 1)
```

Each failing relation gives its error message and the **minimal input** found by
Hypothesis' shrinking. The input is whatever the relation draws, so its shape
depends on the relation:

| Relation | Minimal input | Meaning |
| --- | --- | --- |
| time invariance | `(x, s)` | the signal and the delay `s` |
| causality | `(x, k, bump)` | the signal, the index `k` from which it is changed, and the size of the change (`x[k:] += bump * (1, 2, 3, ...)`) |
| linearity | `((x, y), a, b)` | the two signals and the two coefficients |
| block invariance | `(x, cuts)` | the signal and the cut points between blocks |

Read the time-invariance case: with `x = [1]`, `f(x) = [0.5]`. Delayed by one
sample, `f([0, 1]) = [0.5, 0.5]`, but a time-invariant filter would give
`[0, 0.5]`: the output reacts *before* the input arrives. The causality case
says the same thing: changing `x[1]` (from `[1, 0]` to `[1, 1]`) changed `y[0]`.

To inspect a failure programmatically, ask for the report instead of the
exception:

```python
report = dp.verify(lookahead, raise_on_failure=False)
report.ok                        # False
f = report.failures[1]
f.relation                       # 'causality'
f.message                        # 'causality (k=1): L2 error 5.000e-01 exceeds ...'
f.example                        # (array([1., 0.]), 1, 1.0), the shrunk input itself
f.examples_tried                 # inputs generated before the first failure (here 1)
```

`PropertyViolation` carries the same object as `exc.report`. A report can be
exported with `report.to_json()`, `report.to_markdown()` (for a pull request or
an issue) or `report.to_html()`.

### A bug the structural relations cannot see

A filter whose numerator is off by 1 % is still linear, time-invariant and
causal. Declared without a transfer function, it passes:

```python
b, a = signal.butter(2, 0.25)

@dp.lti_filter()
def off_by_one_percent(x):
    return signal.lfilter(1.01 * b, a, x)

print(dp.verify(off_by_one_percent))
```

```text
dsp-props report for off_by_one_percent
  ok   linearity
  ok   time invariance
  ok   causality
```

Declared with the transfer function it is meant to realise, it fails:

```python
@dp.lti_filter(transfer=(b, a))
def off_by_one_percent(x):
    return signal.lfilter(1.01 * b, a, x)

dp.verify(off_by_one_percent)
```

```text
dsp_props.report.PropertyViolation: dsp-props report for off_by_one_percent
  ok   linearity
  ok   time invariance
  ok   causality
  FAIL transfer function
       transfer function: ||H_measured - H_declared|| = 7.831e-02 > 3.568e-13; worst bin w = 0.0000 rad/sample: measured 1.01+0j, declared 1+0j
```

The transfer-function relation has no generated input (it uses the impulse
response), so there is no "minimal input" line; the message names the worst
frequency instead. This is the main blind spot of the method; see
[What the relations don't see](blind-spots.md).

## 4. Streaming and stateful processors

Real-time code processes a signal block by block and carries state between
calls. `dp.streaming()` declares a class (or any zero-argument factory) whose
instances have a `process(block)` method. The **block invariance** relation then
checks that splitting the input into blocks gives the same output as one call
on the whole signal. The splits drawn are arbitrary cut points (up to
max(N, 16) blocks, empty blocks and single samples included), blocks of one
sample, and blocks of a fixed size.

The obvious implementation with `lfilter` and its `zi` state fails:

```python
import numpy as np
from scipy import signal
import dsp_props as dp

b, a = signal.butter(2, 0.25)

@dp.streaming()
class StreamingLowpass:
    def __init__(self):
        self.z = np.zeros(max(len(a), len(b)) - 1)

    def process(self, block):
        y, self.z = signal.lfilter(b, a, block, zi=self.z)
        return y

dp.verify(StreamingLowpass)
```

```text
dsp_props.report.PropertyViolation: dsp-props report for StreamingLowpass
  FAIL block invariance
       block invariance (cuts=[1, 1]): L2 error 2.873e-01 exceeds tolerance 8.818e-14 (N=2, ref norm 3.971e-01)
       minimal input: ([1, 1], (1, 1))
```

The cut points `[1, 1]` split `[1, 1]` into `[1]`, `[]` and `[1]`: an empty
block in the middle. This is a bug in SciPy itself, not in the class above:
with SciPy 1.18.1, `lfilter` returns an uninitialised final state when given an
empty block with an initial state (reported by dsp-props as
[scipy/scipy#26406](https://github.com/scipy/scipy/issues/26406); the `lfilter`
part is the known [scipy/scipy#22571](https://github.com/scipy/scipy/issues/22571),
fixed in SciPy's development version; the fix for both `lfilter` and `sosfilt`,
[scipy/scipy#26415](https://github.com/scipy/scipy/pull/26415), is scheduled for
SciPy 2.0.0). Until then, the workaround is to skip empty blocks:

```python
@dp.streaming()
class StreamingLowpass:
    def __init__(self):
        self.z = np.zeros(max(len(a), len(b)) - 1)

    def process(self, block):
        if len(block) == 0:            # work around scipy/scipy#26406
            return np.zeros(0)
        y, self.z = signal.lfilter(b, a, block, zi=self.z)
        return y

print(dp.verify(StreamingLowpass))
```

```text
dsp-props report for StreamingLowpass
  ok   block invariance
```

A processor that loses its state on short blocks is caught with a two-sample
input split after its first sample:

```python
@dp.streaming()
class ForgetfulLowpass:
    """Bug: the state is only kept for blocks of two samples or more."""
    def __init__(self):
        self.z = np.zeros(2)

    def process(self, block):
        if len(block) == 0:
            return np.zeros(0)
        y, z = signal.lfilter(b, a, block, zi=self.z)
        if len(block) >= 2:
            self.z = z
        return y

print(dp.verify(ForgetfulLowpass, raise_on_failure=False))
```

```text
dsp-props report for ForgetfulLowpass
  FAIL block invariance
       block invariance (cuts=[1]): L2 error 2.873e-01 exceeds tolerance 6.738e-14 (N=2, ref norm 3.034e-01)
       minimal input: ([1, 0], (1,))
```

### Functions with explicit state

When the state is passed explicitly, `func(block, state) -> (out, new_state)`,
`dp.stateful` wraps it into a streaming operator without writing a class.
`init_state` is either an array (copied for each fresh processor) or a callable
that returns one:

```python
def step(block, z):
    if len(block) == 0:                # work around scipy/scipy#26406
        return np.zeros(0), z
    return signal.lfilter(b, a, block, zi=z)   # returns (y, zf)

lowpass_stream = dp.stateful(step, init_state=np.zeros(2))
print(dp.verify(lowpass_stream))
```

```text
dsp-props report for <lambda>
  ok   block invariance
```

(`dp.stateful` returns an internal factory, hence the `<lambda>` name in the
report.) Both `streaming` and `stateful` accept `tolerance=` and `max_n=`.

Block invariance and the LTI relations are complementary: a filter that drops
part of its state between blocks passes every LTI relation on a single call,
and only block invariance sees it. The C biquad in `examples/c/biquad.c` shows
this with a variant, `biquad_drops_state`, that passes linearity, time
invariance, causality and the transfer function, and fails block invariance on
`[1, 0, 0]` split after the first sample.

## 5. Run it from the command line

`dsp-props check` imports a module or a file, finds every object that carries a
dsp-props declaration (functions, classes, and the values of module-level
dictionaries), and verifies each one. Put the operators in a file:

```python
# my_filters.py
import numpy as np
from scipy import signal

import dsp_props as dp

b, a = signal.butter(2, 0.25)


@dp.lti_filter(transfer=(b, a))
def lowpass(x):
    return signal.lfilter(b, a, x)


@dp.lti_filter()
def lookahead(x):
    return 0.5 * x + 0.5 * np.append(x[1:], 0.0)


@dp.streaming()
class StreamingLowpass:
    def __init__(self):
        self.z = np.zeros(2)

    def process(self, block):
        if len(block) == 0:
            return np.zeros(0)
        y, self.z = signal.lfilter(b, a, block, zi=self.z)
        return y
```

```text
$ dsp-props check my_filters.py --examples 100 --derandomize
ok   my_filters.py:StreamingLowpass  (1 relations, 0.4s)
FAIL my_filters.py:lookahead  (3 relations, 1.6s)
      ok   linearity
      FAIL time invariance
           time invariance (delay 1): L2 error 5.000e-01 exceeds tolerance 6.336e-12 (N=33, ref norm 5.657e+00)
           minimal input: (signal(N=32, max|x|=1, 32 non-zero), 1)
      FAIL causality
           causality (k=1): L2 error 5.000e-01 exceeds tolerance 5.361e-13 (N=1, ref norm 2.414e+00)
           minimal input: ([1, 0], 1, 1)
ok   my_filters.py:lowpass  (4 relations, 1.5s)

2/3 operators pass
$ echo $?
1
```

Signals longer than eight samples are summarised as `signal(N=..., max|x|=...,
... non-zero)`. Targets can be:

| Target | Example |
| --- | --- |
| a file | `dsp-props check my_filters.py` |
| a module | `dsp-props check mypkg.filters` |
| one operator | `dsp-props check my_filters.py:lowpass` or `mypkg.filters:lowpass` |

Options:

| Option | Effect |
| --- | --- |
| `--examples N` | inputs per relation (default 200) |
| `--derandomize` | reproducible inputs, for CI |
| `--json FILE` | write the full report as JSON |
| `--markdown FILE` | write the report as Markdown |
| `--html FILE` | write a standalone HTML page |
| `-v`, `--verbose` | list the relations of passing operators too |

The exit status is 0 if every operator passes, 1 if any relation fails, and 2
on a usage error (a target that cannot be imported, or no declared operator
found). A CI step is then a single line, as in the repository's own workflow:

```yaml
- run: dsp-props check examples/filters.py:fft examples/filters.py:lowpass --derandomize
```

## 6. In a pytest suite

`dp.verify` raises `PropertyViolation`, a subclass of `AssertionError`, so it
can be called directly from a test:

```python
def test_lowpass():
    dp.verify(lowpass)
```

pytest then prints the report with the minimal counterexample. Use
`max_examples=` to trade run time for search effort (200 inputs per relation by
default), and `derandomize=True` or `seed=` to make the inputs reproducible.

## 7. C code

[`dsp_props.adapters`](api/adapters.md) compiles C sources with the system
compiler and wraps a function that follows the convention
`void f(const double *x, size_t n, double *y, double *state)`:

```python
from dsp_props.adapters import build_shared, c_function, c_block_processor

B = [0.06745527388907189, 0.13491054777814378, 0.06745527388907189]
A = [1.0, -1.1429805025399011, 0.41280159809618877]
lib = build_shared("examples/c/biquad.c")

dp.verify(dp.lti_filter(transfer=(B, A))(c_function(lib, "biquad", state_size=2)))
dp.verify(dp.streaming()(c_block_processor(lib, "biquad", state_size=2)))
```

`c_function` calls the C function once with a zeroed state; `c_block_processor`
returns a class whose instances keep the state between `process` calls. A
stateless convention, `void f(const double *x, size_t n, double *y)`, is used
when `state_size` is omitted, and `out_length=` gives the output length of a
function that changes it.

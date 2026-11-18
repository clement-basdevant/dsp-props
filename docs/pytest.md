# Running the checks with pytest

dsp-props installs a pytest plugin. A declared operator in a test module
becomes one pytest test per relation, with no test function to write:

```python
# tests/test_my_filters.py
import numpy as np
from scipy import signal
import dsp_props as dp

b, a = signal.butter(4, 0.2)

@dp.lti_filter(transfer=(b, a))
def my_lowpass(x):
    return signal.lfilter(b, a, x)

@dp.transform(inverse=np.fft.ifft, parseval=True)
def my_fft(x):
    return np.fft.fft(x)
```

```
$ pytest tests/test_my_filters.py -v
tests/test_my_filters.py::my_lowpass::linearity PASSED
tests/test_my_filters.py::my_lowpass::time-invariance PASSED
tests/test_my_filters.py::my_lowpass::causality PASSED
tests/test_my_filters.py::my_lowpass::transfer-function PASSED
tests/test_my_filters.py::my_fft::linearity PASSED
...
```

A failing relation reports its minimal counterexample. Here, a block
processor built on `scipy.signal.lfilter` with `zi`, on SciPy 1.18, which
mishandles empty blocks ([scipy/scipy#22571](https://github.com/scipy/scipy/issues/22571)):

```
FAILED tests/test_my_filters.py::streamed::block-invariance
streamed violates block invariance
  block invariance (cuts=[1, 1]): L2 error 2.120e-01 exceeds tolerance 6.384e-14 (N=2, ref norm 2.875e-01)
  minimal input: ([1, 1], (1, 1))
  full input:    (array([1., 1.]), [1, 1])
  first failure after 16 generated input(s)
```

## What is collected

In test modules (the files pytest collects, `test_*.py` by default):

- operators declared **in the module itself**, and dictionaries of them
  (`FILTERS = {"butter2": ..., "butter4": ...}` gives `FILTERS[butter2]::linearity`...);
- everything listed in a module-level `dsp_props_operators` (a list or a
  dictionary), for operators defined elsewhere:

```python
from mypackage.filters import lowpass, highpass
dsp_props_operators = [lowpass, highpass]
```

Imported operators are not collected otherwise, so a test module can import a
deliberately wrong operator to check that `dp.verify` rejects it. Run one
relation with its node id (`pytest "tests/test_my_filters.py::my_fft::parseval"`)
or with `-k parseval`. `-p no:dsp_props` turns the plugin off.

## Cost profiles

How many inputs each relation sees is set by a profile:

| Profile | Inputs per relation | Use |
| --- | --- | --- |
| `fast` | 30 | every commit; a few seconds per operator |
| `default` | 200 | the default of `dp.verify` and `dsp-props check` |
| `nightly` | 2000 | scheduled bug hunts |

Choose it with `--dsp-props-profile` (pytest), `--profile` (`dsp-props check`),
`profile=` (`dp.verify`), `dp.set_profile`, or the `DSP_PROPS_PROFILE`
environment variable. An explicit `max_examples` (or `--examples`) wins over
the profile. Most of the hand-written mutants of the repository fail within
the first three inputs, so `fast` catches structural faults; faults that
need a particular length or a long signal (DC gain, aliasing) need `default`
or more.

## Replaying past counterexamples

Failing inputs are saved in Hypothesis's example database,
`.hypothesis/examples` under the pytest root directory, and replayed first
on the next run, so a failure found once by a nightly run fails again at the
first input in every later run until it is fixed. Keep that directory between
CI runs, for example with a cache:

```yaml
- uses: actions/cache@v4
  with:
    path: .hypothesis
    key: hypothesis-${{ github.run_id }}
    restore-keys: hypothesis-
```

`--dsp-props-database DIR` moves the database, `--dsp-props-database none`
disables it. Hypothesis neither saves nor replays examples when inputs are
derandomized (`--dsp-props-derandomize`, `derandomize=True`) or seeded
(`seed=`). `dp.verify(..., database="dir")` and `dsp-props check --database dir`
use the same mechanism outside pytest.

## Options

| Command line | `ini` key | Default |
| --- | --- | --- |
| `--dsp-props-profile` | `dsp_props_profile` | `DSP_PROPS_PROFILE`, else `default` |
| `--dsp-props-derandomize` | `dsp_props_derandomize` | off |
| `--dsp-props-database` | `dsp_props_database` | `.hypothesis/examples` |

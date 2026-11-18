"""Block filtering with lfilter and sosfilt breaks on an empty block (scipy/scipy#22571, #26406).

A streaming filter must give the same output however the input is split
into blocks, and an empty block, which real-time code receives, must leave
the state alone. With SciPy before 2.0.0:

* ``lfilter`` (IIR) with ``zi`` returns a wrong final state for an empty
  block, so the following blocks are wrong (silent);
* ``lfilter`` (FIR) and ``sosfilt`` with ``zi`` raise on an empty block.

The fix for both (scipy/scipy#26415) is scheduled for SciPy 2.0.0. The block
invariance relation finds each case with a two-sample input split around an
empty block.
"""

import numpy as np
from scipy import signal

import dsp_props as dp

b, a = signal.butter(2, 0.25)
sos = signal.butter(2, 0.25, output="sos")


def lfilter_step(block, z):
    return signal.lfilter(b, a, block, zi=z)


def sosfilt_step(block, z):
    return signal.sosfilt(sos, block, zi=z)


iir_stream = dp.stateful(lfilter_step, init_state=np.zeros(2))
sos_stream = dp.stateful(sosfilt_step, init_state=np.zeros((1, 2)))


def lfilter_step_fixed(block, z):
    if len(block) == 0:  # workaround until SciPy 2.0.0
        return np.zeros(0), z
    return signal.lfilter(b, a, block, zi=z)


fixed_stream = dp.stateful(lfilter_step_fixed, init_state=np.zeros(2))

if __name__ == "__main__":
    for name, op in [("lfilter", iir_stream), ("sosfilt", sos_stream), ("lfilter, empty blocks skipped", fixed_stream)]:
        report = dp.verify(op, max_examples=100, seed=0, raise_on_failure=False)
        print(f"{name}: {'ok' if report.ok else 'FAIL'}")
        for failure in report.failures:
            print(f"  minimal input: {failure.example!r}")

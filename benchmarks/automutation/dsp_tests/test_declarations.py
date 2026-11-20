"""dsp-props declarations of the copied scipy.signal functions (plan 3.2), one per function.

Written from the documentation, like the declarations of the rediscovery corpus
(benchmarks/bugcorpus/corpus.py): resample as an invertible upsampling, resample_poly and decimate
as rational resamplers over drawn rates, hilbert as an analytic signal whose real part is the input.
Collected by the dsp-props pytest plugin: each relation of each operator is one test.
"""

import numpy as np
import scipy.signal as ss
import scipy_subset as sub
from hypothesis import strategies as st

import dsp_props as dp
from dsp_props.core.iir import IIRTolerance
from dsp_props.generators import rate_pairs


@dp.family(st.sampled_from([2, 3, 4]))
def resample(L):
    return dp.transform(inverse=lambda Y: sub.resample(Y, len(Y) // L), kind="generic")(
        lambda x: sub.resample(x, L * len(x)))


@dp.family(rate_pairs(8))
def resample_poly(up, down):
    return dp.resampler(up, down, length="ceil", dc_gain=1.0, ripple=0.02, stopband=None)(
        lambda x: sub.resample_poly(x, up, down))


@dp.family(st.tuples(st.integers(2, 8), st.sampled_from(["iir", "fir"])))
def decimate(q, kind):
    tolerance = IIRTolerance(*ss.cheby1(8, 0.05, 0.8 / q)) if kind == "iir" else None
    return dp.resampler(1, q, causal=True, length="ceil", dc_gain=1.0, ripple=0.02, stopband=None,
                        tolerance=tolerance)(lambda x: sub.decimate(x, q, ftype=kind, zero_phase=False))


@dp.transform(inverse=np.real, kind="generic", real_input_only=True)
def hilbert(x):
    return sub.hilbert(x)

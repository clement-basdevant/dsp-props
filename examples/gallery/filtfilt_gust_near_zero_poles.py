"""filtfilt(method="gust") loses digits when poles approach zero (scipy/scipy#26755).

Gustafsson's method chooses initial conditions so that the output does not
depend on the direction of time: filtering the reversed signal and reversing
the result must give the same output. Campaign 3 found this relation broken
on a filter with poles close to zero, without being exactly zero (the exact
zeros are scipy/scipy#26610). The problem itself is well conditioned (a
relative condition number close to 1, ``benchmarks/gust_near_zero_poles.py``),
so the lost digits come from the algorithm: a least-squares system that
becomes nearly singular as the poles shrink.

This script measures the reversal gap, in units of eps times the size of the
output, for one pole at 0.5 and a double pole at r.
"""

import numpy as np
from scipy import signal

EPS = np.finfo(np.float64).eps


def reversal_gap(r, n=97):
    """|filtfilt(x) - reverse(filtfilt(reverse(x)))| / (eps |filtfilt(x)|) for poles 0.5, r, r."""
    a = np.poly([0.5, r, r])
    x = np.ones(n)
    y = signal.filtfilt([1.0], a, x, method="gust")
    yr = signal.filtfilt([1.0], a, x[::-1], method="gust")[::-1]
    return float(np.linalg.norm(y - yr) / (EPS * np.linalg.norm(y)))


RADII = [1e-1, 1e-2, 1e-3, 1e-4, 1e-6]

if __name__ == "__main__":
    for r in RADII:
        print(f"double pole at {r:g}: reversal gap {reversal_gap(r):9.3g} eps")

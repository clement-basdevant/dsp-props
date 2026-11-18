"""decimate with a real IIR ``dlti`` filter falls back to the transfer-function form (plan 3.1).

Usage: python benchmarks/decimate_dlti_route.py [--json benchmarks/decimate_dlti_route.json]

Seen in the rediscovery campaign (``benchmarks/rediscovery.py``): since scipy/scipy#17881 (SciPy
1.11), ``decimate`` sends a ``dlti`` filter to ``lfilter(b, a)`` instead of ``sosfilt`` when
``any(np.iscomplex(system.poles))``, a guard meant for complex *coefficients*. The poles of a real
IIR filter come in complex-conjugate pairs, so every real IIR low-pass takes the (b, a) route that
scipy/scipy#16096 replaced by second-order sections for the default filter, because it loses
accuracy. The script measures the gap between ``decimate(x, q, ftype=dlti(z, p, k))`` and the
same filter applied as second-order sections, for the documented default filter (Chebyshev I,
order 8, 0.05 dB, cutoff 0.8 / q). The reference applies the same second-order sections in
``long double`` (extended precision on Linux x86-64), so each route's error is measured.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import scipy
import scipy.signal as ss

from dsp_props.core.iir import longdouble_lfilter

HERE = pathlib.Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=pathlib.Path, default=HERE / "decimate_dlti_route.json")
    args = parser.parse_args()
    x = np.random.default_rng(0).standard_normal(1024)
    rows = []
    for q in [2, 4, 8, 12, 16, 20]:
        z, p, k = ss.cheby1(8, 0.05, 0.8 / q, output="zpk")
        sos = ss.zpk2sos(z, p, k)
        y = x.astype(np.longdouble)
        for section in sos:
            y = longdouble_lfilter(section[:3], section[3:], y)
        reference = y[::q]
        user = ss.decimate(x, q, ftype=ss.dlti(z, p, k), zero_phase=False)
        default = ss.decimate(x, q, zero_phase=False)  # the same filter, built by decimate itself

        def err(y):
            return float(np.linalg.norm((y - reference).astype(float)) / np.linalg.norm(x))

        rows.append({"q": q, "dlti_error": err(user), "default_error": err(default),
                     "poles_complex": bool(np.any(np.iscomplex(p))),
                     "coefficients_real": bool(np.allclose(np.imag(np.poly(p)), 0))})
        r = rows[-1]
        print(f"q={q:2d}  error, dlti filter: {r['dlti_error']:.1e}   default filter: {r['default_error']:.1e}")
    args.json.write_text(json.dumps({"scipy": scipy.__version__, "numpy": np.__version__, "n": len(x),
                                     "rows": rows}, indent=1) + "\n")


if __name__ == "__main__":
    main()

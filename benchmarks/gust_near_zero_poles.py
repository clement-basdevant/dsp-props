"""Triage of the campaign 3 failure: filtfilt(method="gust") and poles close to zero.

Campaign 3 found ``filtfilt(b, a, x, method="gust")`` breaking the
time-reversal relation by about 6 times the target's tolerance on an order-6
filter with four poles clustered at 0.5 and two poles at 0.001. This script
decides between an ill-conditioned problem and an inaccurate algorithm:

* ``exact``: SciPy's own algorithm (``scipy.signal._signaltools._filtfilt_gust``)
  replayed in 80-digit arithmetic with mpmath on the same float64 inputs, so
  the reference has no rounding error of its own;
* ``kappa_x``: relative condition number of the exact map x -> y (it is
  linear, so its matrix G is built column by column);
* ``kappa_a``: sensitivity of the exact output to random relative
  perturbations of the coefficients of ``a``;
* ``scipy error``: forward error of SciPy against the exact output.

A well-conditioned problem (kappa of order 1 to 100) with a SciPy error of
1e7 eps or more means the algorithm, not the problem, loses the digits.

Usage: python benchmarks/gust_near_zero_poles.py [--quick]   (needs mpmath)
Writes benchmarks/gust_near_zero_poles.json.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import mpmath as mpm
import numpy as np
import scipy
import scipy.signal as ss

mp = mpm.mp
EPS = float(np.finfo(np.float64).eps)
OUT = pathlib.Path(__file__).with_suffix(".json")

CAMPAIGN_POLES = [0.50008 + 7.8e-5j, 0.50008 - 7.8e-5j, 0.49992 + 7.8e-5j, 0.49992 - 7.8e-5j, 0.001, 0.001]
CASES = [
    ("campaign 3: 4 poles near 0.5, 2 at 0.001", CAMPAIGN_POLES),
    ("4 poles near 0.5 only", CAMPAIGN_POLES[:4]),
    ("4 poles near 0.5, 2 at 0.3", CAMPAIGN_POLES[:4] + [0.3, 0.3]),
    ("0.5, 2 at 1e-2", [0.5, 1e-2, 1e-2]),
    ("0.5, 2 at 1e-3", [0.5, 1e-3, 1e-3]),
    ("0.5, 2 at 1e-4", [0.5, 1e-4, 1e-4]),
    ("0.5, 2 at 1e-6", [0.5, 1e-6, 1e-6]),
    ("0.5, 2 at 0 (exactly)", [0.5, 0.0, 0.0]),
    ("0.9 +/- 0.3j, 0.8 (control)", [0.9 + 0.3j, 0.9 - 0.3j, 0.8]),
]


def _lfilter_mp(b, a, x, zi=None):
    """Direct form II transposed, as scipy.signal.lfilter, in mpmath."""
    order = max(len(a), len(b)) - 1
    b = list(b) + [0] * (order + 1 - len(b))
    a = list(a) + [0] * (order + 1 - len(a))
    z = list(zi) if zi is not None else [mp.mpf(0)] * order
    y = []
    for xn in x:
        yn = b[0] * xn + z[0]
        for k in range(order - 1):
            z[k] = z[k + 1] + b[k + 1] * xn - a[k + 1] * yn
        z[order - 1] = b[order] * xn - a[order] * yn
        y.append(yn)
    return y


def gust_exact(b, a, x, dps=80):
    """``_filtfilt_gust`` (m == n branch) in ``dps``-digit arithmetic; min-norm least squares by SVD."""
    with mp.workdps(dps):
        b = [mp.mpf(float(v)) for v in b]
        a = [mp.mpf(float(v)) for v in a]
        x = [mp.mpf(float(v)) for v in x]
        order, n = max(len(a), len(b)) - 1, len(x)
        h = _lfilter_mp(b, a, [0] * n, [1] + [0] * (order - 1))
        obs = mp.matrix(n, order)
        for k in range(order):
            for i in range(k, n):
                obs[i, k] = h[i - k]
        s = mp.matrix(n, order)
        for k in range(order):
            col = _lfilter_mp(b, a, [obs[n - 1 - i, k] for i in range(n)])
            for i in range(n):
                s[i, k] = col[i]
        m, w = mp.matrix(n, 2 * order), mp.matrix(n, 2 * order)
        for i in range(n):
            for k in range(order):
                m[i, k] = s[n - 1 - i, k] - obs[i, k]
                m[i, order + k] = obs[n - 1 - i, k] - s[i, k]
                w[i, k], w[i, order + k] = s[n - 1 - i, k], obs[n - 1 - i, k]
        yf = _lfilter_mp(b, a, x)
        yfb = _lfilter_mp(b, a, yf[::-1])[::-1]
        ybf = _lfilter_mp(b, a, _lfilter_mp(b, a, x[::-1])[::-1])
        d = [ybf[i] - yfb[i] for i in range(n)]
        u, sv, v = mp.svd_r(m)
        ic = [mp.mpf(0)] * (2 * order)
        for j in range(len(sv)):
            if sv[j] > max(sv) * mp.mpf(10) ** (10 - dps):
                c = mp.fsum(u[i, j] * d[i] for i in range(n)) / sv[j]
                for k in range(2 * order):
                    ic[k] += c * v[j, k]
        y = [yfb[i] + mp.fsum(w[i, k] * ic[k] for k in range(2 * order)) for i in range(n)]
        cond = float(max(sv) / min(sv)) if min(sv) > 0 else float("inf")
        return np.array([float(t) for t in y]), cond


def analyse(name, poles, x, rng, full_kappa=True):
    a = np.real(np.poly(poles))
    b = np.array([1.0])
    y_exact, cond_m = gust_exact(b, a, x)
    ny = np.linalg.norm(y_exact)
    y = ss.filtfilt(b, a, x, method="gust")
    y_rev = ss.filtfilt(b, a, x[::-1], method="gust")[::-1]
    row = {
        "case": name,
        "a_last": float(a[-1]),
        "cond_M": cond_m,
        "scipy_error_eps": float(np.linalg.norm(y - y_exact) / ny / EPS),
        "reversal_gap_eps": float(np.linalg.norm(y - y_rev) / ny / EPS),
    }
    if full_kappa:
        n = len(x)
        g = np.column_stack([gust_exact(b, a, np.eye(n)[j], dps=50)[0] for j in range(n)])
        row["kappa_x"] = float(np.linalg.norm(g, 2) * np.linalg.norm(x) / ny)
    if np.all(a != 0):
        k = []
        for _ in range(3):
            da = 1e-10 * rng.choice([-1.0, 1.0], size=len(a)) * a
            da[0] = 0.0
            k.append(np.linalg.norm(gust_exact(b, a + da, x)[0] - y_exact) / ny / 1e-10)
        row["kappa_a"] = float(max(k))
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="skip kappa_x (builds the 97 x 97 matrix of each case)")
    args = parser.parse_args()
    rng = np.random.default_rng(0)
    x = np.ones(97)  # the campaign counterexample: 97 samples at 1
    rows = [analyse(name, poles, x, rng, full_kappa=not args.quick) for name, poles in CASES]
    print(f"SciPy {scipy.__version__}, NumPy {np.__version__}; x = 97 samples at 1, b = [1]; errors relative, in eps")
    print(f"{'case':44s} {'a[-1]':>9s} {'cond(M)':>8s} {'kappa_x':>8s} {'kappa_a':>8s} {'scipy err':>10s} {'rev. gap':>9s}")
    for r in rows:
        print(
            f"{r['case']:44s} {r['a_last']:9.1e} {r['cond_M']:8.1e} {r.get('kappa_x', float('nan')):8.2g} "
            f"{r.get('kappa_a', float('nan')):8.2g} {r['scipy_error_eps']:10.1e} {r['reversal_gap_eps']:9.1e}"
        )
    meta = {"scipy": scipy.__version__, "numpy": np.__version__, "x": "ones(97)", "b": [1.0]}
    OUT.write_text(json.dumps({"meta": meta, "rows": rows}, indent=1) + "\n")
    print(f"written {OUT}")


if __name__ == "__main__":
    main()

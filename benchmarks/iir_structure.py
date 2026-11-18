"""How much does a structure-aware bound tighten the IIR tolerance? (plan, action 2.6)

``IIRTolerance(b, a)`` is the direct form I bound (``structure="df1"``);
``IIRTolerance(b, a, structure="df2t")`` is the bound of the transposed
direct form II that ``scipy.signal.lfilter`` computes. For the five filters
of ``benchmarks/calibrated_tolerance.py`` and 80 random stable filters, the
script reports:

* the two bounds (factor of ``eps * ||x||``, margin ``c`` excluded) and their ratio;
* the worst error of ``lfilter`` measured against ``longdouble_lfilter`` on the
  calibration probes, in the same units, so the remaining gap is visible;
* for the IIR precision mutants of ``mutants/precision_mutants.py`` (rounded
  coefficients, narrow arithmetic; ``lfilter`` inside, so df2t applies), the
  smallest loss caught on every seed with each bound.

The reference is extended only where ``long double`` is (Linux x86-64).

Usage: python benchmarks/iir_structure.py [--examples 200] [--seeds 3] [--quick]
Writes benchmarks/iir_structure.json.
"""

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import scipy.signal as ss

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(ROOT / "benchmarks")]

from calibrated_tolerance import FILTERS  # noqa: E402
from precision_mutants import LENGTHS, detection, measured_loss  # noqa: E402

from dsp_props.core import IIRTolerance  # noqa: E402
from dsp_props.core.calibration import _probe_signals  # noqa: E402
from dsp_props.core.iir import longdouble_lfilter  # noqa: E402
from mutants import precision_mutants as pm  # noqa: E402

OUT = pathlib.Path(__file__).with_suffix(".json")
EPS = float(np.finfo(np.float64).eps)


def measured(b, a, lengths=LENGTHS, trials=4, seed=0):
    """Worst ||lfilter - reference|| / (eps ||x||) over the calibration probes."""
    rng = np.random.default_rng(seed)
    worst = 0.0
    for n in lengths:
        for x in _probe_signals(n, trials, rng):
            x = np.real(x)
            if not np.any(x):
                continue
            err = np.linalg.norm(ss.lfilter(b, a, x) - longdouble_lfilter(b, a, x).astype(float))
            worst = max(worst, err / (EPS * np.linalg.norm(x)))
    return worst


def random_filters(count, seed=1):
    rng = np.random.default_rng(seed)
    for _ in range(count):
        order = int(rng.integers(2, 13))
        half = order // 2
        r, th = rng.uniform(0.5, 0.999, half), rng.uniform(0, np.pi, half)
        poles = np.r_[r * np.exp(1j * th), r * np.exp(-1j * th), rng.uniform(-0.999, 0.999, order - 2 * half)]
        yield rng.uniform(-1, 1, rng.integers(1, order + 3)), np.real(np.poly(poles))


def row(name, b, a, lengths):
    df1 = IIRTolerance(b, a).factor
    df2t = IIRTolerance(b, a, structure="df2t").factor
    return {"filter": name, "df1": df1, "df2t": df2t, "ratio": df1 / df2t, "measured": measured(b, a, lengths)}


def thresholds(examples, seeds, max_bits):
    """Smallest loss caught on every seed (and every larger one), per family and structure."""
    out = {}
    for family in ("iir_coefficients", "iir_arithmetic"):
        make = pm.FAMILIES[family][0]
        for structure in ("df1", "df2t"):
            tol = IIRTolerance(pm.B, pm.A, structure=structure)
            rows = []
            for bits in range(0, max_bits + 1):
                op = make(bits)
                declared = pm.declare_iir(op, tol)
                caught, _ = detection(declared, examples, seeds)
                rows.append({"bits": bits, "loss_eps": measured_loss(op, lambda x: longdouble_lfilter(pm.B, pm.A, x)),
                             "seeds_failing": caught})
            first = None
            for r in reversed(rows):
                if r["seeds_failing"] != seeds:
                    break
                first = r
            out[f"{family}.{structure}"] = {"caught_from_loss_eps": first["loss_eps"] if first else None,
                                            "caught_from_bits": first["bits"] if first else None, "rows": rows}
            print(f"  {family:<17} {structure:<5} caught from {first['loss_eps'] if first else float('nan'):9.3g} eps"
                  f" ({first['bits'] if first else '-'} bits)")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples", type=int, default=200)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--max-bits", type=int, default=24)
    parser.add_argument("--random", type=int, default=80)
    parser.add_argument("--quick", action="store_true", help="10 random filters, short lengths, no mutants")
    args = parser.parse_args()
    t0 = time.perf_counter()
    lengths = [1, 2, 5, 16, 64] if args.quick else LENGTHS

    named = [row(name, b, a, lengths) for name, (b, a) in FILTERS.items()]
    print("filter                          df1 bound  df2t bound  ratio  measured  df2t/measured")
    for r in named:
        print(f"  {r['filter']:<30} {r['df1']:9.3g} {r['df2t']:10.3g} {r['ratio']:6.2f} {r['measured']:9.3g}"
              f" {r['df2t'] / r['measured']:10.3g}")
    rand = [row(f"random {i}", b, a, lengths) for i, (b, a) in enumerate(random_filters(10 if args.quick else args.random))]
    ratios = sorted(r["ratio"] for r in rand)
    gaps = sorted(r["df2t"] / r["measured"] for r in rand if r["measured"] > 0)
    print(f"{len(rand)} random filters: df1/df2t from {ratios[0]:.2f} to {ratios[-1]:.2f} (median {np.median(ratios):.2f});"
          f" df2t/measured from {gaps[0]:.3g} to {gaps[-1]:.3g}")
    out = {"examples": args.examples, "seeds": args.seeds, "named": named, "random": rand,
           "random_ratio": {"min": ratios[0], "median": float(np.median(ratios)), "max": ratios[-1]},
           "random_gap": {"min": gaps[0], "median": float(np.median(gaps)), "max": gaps[-1]}}
    if not args.quick:
        print("IIR precision mutants (pm.B, pm.A):")
        out["thresholds"] = thresholds(args.examples, args.seeds, args.max_bits)
    out["seconds"] = time.perf_counter() - t0
    if not args.quick:
        OUT.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {OUT.name} ({out['seconds']:.0f} s)")


if __name__ == "__main__":
    main()

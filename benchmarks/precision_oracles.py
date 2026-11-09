"""Graded precision faults against four oracles (review plan, action A9).

The precision mutants of ``mutants/precision_mutants.py`` (0 to 24 significand
bits dropped, see ``benchmarks/precision_mutants.py``) are checked by:

* ``default``: the relations with dsp-props' default tolerance models;
* ``fixed``: the same relations with a fixed relative tolerance of 1e-7
  (the default rtol of ``numpy.testing.assert_allclose``) on the L2 error,
  as ``dsp-props/fixed-tol`` in ``benchmarks/comparison``;
* ``calibrated``: the same relations with a tolerance calibrated on the
  trusted implementation of the class (``numpy.fft.fft``, ``lfilter``,
  ``resample_poly``, the moving sum) against a long double reference, with no
  analytic floor: ``CalibratedTolerance(ratios, base=FFTTolerance(c=0))``;
* ``differential``: ``numpy.testing.assert_allclose(mutant(x), trusted(x))``
  with its defaults (rtol = 1e-7, elementwise), on the calibration probe
  signals (noise, impulse, off-grid tone, near-Nyquist tone) at 18 lengths.
  It needs a trusted implementation; the relations do not.

Level 0 is a correct implementation: a failure there is a false alarm. It is
the trusted implementation itself, except for the FFT, where it is a direct
DFT with accurate twiddles (3.5 eps, against 2 eps for numpy.fft.fft).

Usage: python benchmarks/precision_oracles.py [--examples 200] [--seeds 3]
                                               [--max-bits 24] [--step 2]
                                               [--json benchmarks/precision_oracles.json]
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

import dsp_props as dp  # noqa: E402
from dsp_props.core.calibration import CalibratedTolerance, _probe_signals, calibrate  # noqa: E402
from dsp_props.core.iir import longdouble_lfilter  # noqa: E402
from dsp_props.core.tolerance import FFTTolerance  # noqa: E402
from comparison.approaches import FixedTolerance  # noqa: E402
from mutants import precision_mutants as pm  # noqa: E402
from mutants.streaming_mutants import MovingSum  # noqa: E402
from precision_mutants import LENGTHS, REFERENCES, _callable, _run_stream, measured_loss  # noqa: E402

ORACLES = ["default", "fixed", "calibrated", "differential"]

# lengths calibrated; the others fall back to the worst measured ratio
CALIBRATION_LENGTHS = sorted(set(range(1, 65)) | set(LENGTHS) | set(range(64, 513, 16)))

TRUSTED = {
    "fft": (np.fft.fft, None),  # None: the default reference, extended_dft
    "iir": (lambda x: ss.lfilter(pm.B, pm.A, x), lambda x: longdouble_lfilter(pm.B, pm.A, x)),
    "resampler": (lambda x: ss.resample_poly(x, pm.UP, pm.DOWN) if len(x) else np.zeros(0), pm.longdouble_resample),
    "streaming": (_run_stream(MovingSum), pm.longdouble_moving_sum),
}


def calibrated_tolerance(cls):
    """Calibrate the trusted implementation of ``cls``; keep no analytic floor."""
    op, reference = TRUSTED[cls]
    kwargs = {} if reference is None else {"reference": reference}
    tol = calibrate(op, lengths=CALIBRATION_LENGTHS, **kwargs)
    return CalibratedTolerance(ratios=tol.ratios, margin=tol.margin, base=FFTTolerance(c=0.0))


def relations_fail(declared, examples, seeds):
    caught, relations = 0, set()
    for seed in range(seeds):
        report = dp.verify(declared, max_examples=examples, seed=seed, raise_on_failure=False)
        caught += not report.ok
        relations.update(f.relation for f in report.failures)
    return caught, sorted(relations)


def differential_fails(op, trusted, seeds, trials=2):
    """Seeds on which assert_allclose(op(x), trusted(x)) fails on some probe signal."""
    caught = 0
    for seed in range(seeds):
        rng = np.random.default_rng(1000 + seed)
        failed = False
        for n in LENGTHS:
            for x in _probe_signals(n, trials, rng):
                try:
                    np.testing.assert_allclose(op(x), trusted(x))
                except AssertionError:
                    failed = True
                    break
            if failed:
                break
        caught += failed
    return caught


def caught_from(rows, key, seeds):
    """Smallest level from which this level and every larger one fail on every seed."""
    first = None
    for r in reversed(rows):
        if r[key] != seeds:
            break
        first = r
    return first


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples", type=int, default=200)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--max-bits", type=int, default=24)
    parser.add_argument("--step", type=int, default=2)
    parser.add_argument("--json", help="write the results here")
    args = parser.parse_args()

    tolerances = {}
    print("calibrated ratios of the trusted implementations (worst, eps):")
    for cls in TRUSTED:
        tolerances[cls] = calibrated_tolerance(cls)
        print(f"  {cls:<10} {tolerances[cls].worst:.3g}")

    out = {"examples": args.examples, "seeds": args.seeds,
           "calibrated_worst_eps": {c: t.worst for c, t in tolerances.items()}, "graded": {}, "summary": {}}
    for family, make in pm.GRADED.items():
        cls = family.split("_")[0]
        trusted = TRUSTED[cls][0]
        rows = []
        print(f"\n{family}: bits, loss (eps), seeds failing / {args.seeds} for {', '.join(ORACLES)}")
        for bits in range(0, args.max_bits + 1, args.step):
            t0 = time.perf_counter()
            loss = measured_loss(_callable(family, make(bits)), REFERENCES[cls])
            row = {"bits": bits, "loss_eps": loss}
            row["default"], rels = relations_fail(make(bits), args.examples, args.seeds)
            row["fixed"], _ = relations_fail(make(bits, FixedTolerance()), args.examples, args.seeds)
            row["calibrated"], _ = relations_fail(make(bits, tolerances[cls]), args.examples, args.seeds)
            row["differential"] = differential_fails(_callable(family, make(bits)), trusted, args.seeds)
            row["default_relations"] = rels
            rows.append(row)
            print(f"  {bits:3d} {loss:11.3g}   " + "  ".join(f"{row[o]}" for o in ORACLES)
                  + f"   {time.perf_counter() - t0:5.1f}s")
        out["graded"][family] = rows
        out["summary"][family] = {}
        for o in ORACLES:
            first = caught_from(rows, o, args.seeds)
            out["summary"][family][o] = {
                "caught_from_loss_eps": first["loss_eps"] if first else None,
                "false_alarm_at_0_bits": rows[0][o],
            }

    print(f"\nsmallest loss (eps) caught on every seed from there on; [false alarms at 0 bits / {args.seeds}]")
    print(f"  {'family':<22}" + "".join(f"{o:>20}" for o in ORACLES))
    for family, s in out["summary"].items():
        cells = []
        for o in ORACLES:
            v = s[o]["caught_from_loss_eps"]
            cells.append(f"{('%.2g' % v) if v is not None else 'never':>15} [{s[o]['false_alarm_at_0_bits']}]")
        print(f"  {family:<22}" + "".join(f"{c:>20}" for c in cells))
    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(out, indent=2, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())

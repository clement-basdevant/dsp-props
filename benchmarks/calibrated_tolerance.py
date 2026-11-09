"""Does a tolerance calibrated on one implementation hold for others? (A9, follow-up)

``benchmarks/precision_oracles.py`` calibrates the tolerance on the very
implementation it then checks at level 0, and covers one filter and one
resampler. This script widens both:

1. False alarms. A tolerance is calibrated, without analytic floor
   (``dp.calibrate(..., base=None)``), on a trusted implementation; other
   correct implementations of the same operator are then checked with it
   (``dp.verify``, 200 inputs per relation, 3 seeds):
   FFT: NumPy -> SciPy, PyTorch (if installed), a radix-2 FFT and a direct DFT;
   IIR, five filters: ``lfilter`` -> ``sosfilt`` of ``tf2sos``;
   resampler, four ratios: ``resample_poly`` -> a direct polyphase convolution;
   streaming: the moving sum -> ``np.convolve`` with carried history.
   Each implementation's own worst error (in eps, against long double) is
   printed next to it: a failure of an implementation measurably less
   accurate than the trusted one is the tolerance doing its job, not a false
   alarm.

2. Detection thresholds on the same filters and ratios: the graded
   coefficient and arithmetic mutants of each filter, the arithmetic mutant of
   each resampler, with the default and the calibrated tolerance.

Usage: python benchmarks/calibrated_tolerance.py [--examples 200] [--seeds 3]
                                                  [--step 4] [--max-bits 24]
                                                  [--json benchmarks/calibrated_tolerance.json]
"""

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import scipy.fft
import scipy.signal as ss

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(ROOT / "benchmarks")]

import dsp_props as dp  # noqa: E402
from dsp_props.core.calibration import _probe_signals, extended_dft, relative_error  # noqa: E402
from dsp_props.core.iir import longdouble_lfilter  # noqa: E402
from mutants import precision_mutants as pm  # noqa: E402
from mutants.streaming_mutants import MovingSum  # noqa: E402
from precision_mutants import LENGTHS, _run_stream  # noqa: E402

CALIBRATION_LENGTHS = sorted(set(range(1, 65)) | set(LENGTHS) | set(range(64, 513, 16)))

FILTERS = {
    "resonant order 4 (poles 0.95)": (pm.B, pm.A),
    "butter(4, 0.2)": ss.butter(4, 0.2),
    "cheby1(6, 1, 0.3)": ss.cheby1(6, 1, 0.3),
    "ellip(4, 0.5, 40, 0.25)": ss.ellip(4, 0.5, 40, 0.25),
    "butter(2, 0.01)": ss.butter(2, 0.01),
}
RATIOS = [(2, 3), (3, 2), (5, 4), (1, 4)]


def worst_error(op, reference, lengths=LENGTHS, seed=0, trials=4):
    rng = np.random.default_rng(seed)
    return max(relative_error(op, reference, x) for n in lengths for x in _probe_signals(n, trials, rng))


def failing_seeds(declared, examples, seeds):
    fails, rels = 0, set()
    for seed in range(seeds):
        report = dp.verify(declared, max_examples=examples, seed=seed, raise_on_failure=False)
        fails += not report.ok
        rels.update(f.relation for f in report.failures)
    return fails, sorted(rels)


def _torch_fft():
    try:
        import torch
    except ImportError:
        return None

    def fft(x):
        return torch.fft.fft(torch.from_numpy(np.asarray(x))).numpy()

    return fft


def false_alarm_cases():
    """(class, setting, trusted op, reference, [(name, op)], declare(op, tol))."""
    cases = []
    others = [("scipy.fft.fft", scipy.fft.fft), ("radix-2, accurate twiddles", pm.radix2_reference),
              ("direct DFT, accurate twiddles", pm.dft_with_rounded_twiddles(0))]
    torch_fft = _torch_fft()
    if torch_fft is not None:
        others.append(("torch.fft.fft (CPU)", torch_fft))
    cases.append(("fft", "complex DFT", np.fft.fft, extended_dft, others, pm.declare_fft))
    for name, (b, a) in FILTERS.items():
        sos = ss.tf2sos(b, a)
        cases.append((
            "iir", name,
            (lambda b, a: lambda x: ss.lfilter(b, a, x))(b, a),
            (lambda b, a: lambda x: longdouble_lfilter(b, a, x))(b, a),
            [("sosfilt(tf2sos)", (lambda sos: lambda x: ss.sosfilt(sos, x))(sos))],
            (lambda b, a: lambda op, tol: pm.declare_iir(op, tol, b, a))(b, a),
        ))
    for up, down in RATIOS:
        cases.append((
            "resampler", f"{up}/{down}",
            (lambda u, d: lambda x: ss.resample_poly(x, u, d) if len(x) else np.zeros(0))(up, down),
            (lambda u, d: lambda x: pm.longdouble_resample(x, u, d))(up, down),
            [("direct polyphase convolution", (lambda u, d: lambda x: pm.polyphase_resample(x, u, d))(up, down))],
            (lambda u, d: lambda op, tol: pm.declare_resampler(op, tol, u, d))(up, down),
        ))
    cases.append(("streaming", "moving sum", MovingSum, pm.longdouble_moving_sum,
                  [("np.convolve with history", pm.ConvolveMovingSum)], pm.declare_streaming))
    return cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples", type=int, default=200)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--step", type=int, default=4)
    parser.add_argument("--max-bits", type=int, default=24)
    parser.add_argument("--only", choices=["false-alarms", "thresholds"])
    parser.add_argument("--json", help="write the results here (merged into an existing file)")
    args = parser.parse_args()
    out = {"examples": args.examples, "seeds": args.seeds, "false_alarms": [], "thresholds": []}
    if args.json and pathlib.Path(args.json).exists():
        out.update({k: v for k, v in json.loads(pathlib.Path(args.json).read_text()).items()
                    if k in ("false_alarms", "thresholds")})

    print(f"1. tolerance calibrated on the trusted implementation, others checked with it "
          f"(seeds failing / {args.seeds}; worst error in eps against long double)")
    tolerances = {}
    if args.only != "thresholds":
        out["false_alarms"] = []
    for cls, setting, trusted, reference, others, declare in false_alarm_cases():
        as_op = _run_stream if cls == "streaming" else (lambda f: f)  # streaming: classes, run whole
        tol = dp.calibrate(as_op(trusted), reference=reference, lengths=CALIBRATION_LENGTHS, base=None)
        tolerances[(cls, setting)] = tol
        print(f"\n  {cls}, {setting}: trusted worst {tol.worst:.3g} eps")
        if args.only == "thresholds":
            continue
        for name, op in [("trusted", trusted)] + others:
            # FFTs are cheap: measure every length the relations draw (N <= 512)
            lengths = range(1, 513) if cls == "fft" else CALIBRATION_LENGTHS
            err = worst_error(as_op(op), reference, lengths, trials=2)
            fails, rels = failing_seeds(declare(op, tol), args.examples, args.seeds)
            print(f"    {name:<32} {err:9.3g} eps   {fails}/{args.seeds}  {', '.join(rels) or '-'}")
            out["false_alarms"].append({"class": cls, "setting": setting, "implementation": name,
                                        "worst_eps": err, "trusted_worst_eps": tol.worst,
                                        "seeds_failing": fails, "relations": rels})

    print(f"\n2. smallest loss caught on every seed from there on (default -> calibrated), "
          f"bits 0..{args.max_bits} step {args.step}")
    families = []
    for name, (b, a) in FILTERS.items():
        ref = (lambda b, a: lambda x: longdouble_lfilter(b, a, x))(b, a)
        decl = (lambda b, a: lambda op, tol: pm.declare_iir(op, tol, b, a))(b, a)
        families.append(("iir", name, "coefficients",
                         (lambda b, a: lambda bits: pm.lfilter_with_rounded_coefficients(bits, b, a))(b, a), ref, decl))
        families.append(("iir", name, "arithmetic",
                         (lambda b, a: lambda bits: pm.lfilter_in_narrow_format(bits, b, a))(b, a), ref, decl))
    for up, down in RATIOS:
        families.append(("resampler", f"{up}/{down}", "arithmetic",
                         (lambda u, d: lambda bits: pm.resample_in_narrow_format(bits, u, d))(up, down),
                         (lambda u, d: lambda x: pm.longdouble_resample(x, u, d))(up, down),
                         (lambda u, d: lambda op, tol: pm.declare_resampler(op, tol, u, d))(up, down)))
    if args.only == "false-alarms":
        families = []
    else:
        out["thresholds"] = []
    for cls, setting, fault, make, reference, declare in families:
        tol = tolerances[(cls, setting)]
        rows = []
        t0 = time.perf_counter()
        for bits in range(0, args.max_bits + 1, args.step):
            op = make(bits)
            row = {"bits": bits, "loss_eps": worst_error(op, reference)}
            row["default"], _ = failing_seeds(declare(op, None), args.examples, args.seeds)
            row["calibrated"], _ = failing_seeds(declare(op, tol), args.examples, args.seeds)
            rows.append(row)
        summary = {}
        for o in ("default", "calibrated"):
            first = None
            for r in reversed(rows):
                if r[o] != args.seeds:
                    break
                first = r
            summary[o] = first["loss_eps"] if first else None
            summary[o + "_false_alarm_at_0"] = rows[0][o]
        fmt = lambda v: f"{v:.2g}" if v is not None else "never"  # noqa: E731
        print(f"  {cls:<9} {setting:<30} {fault:<12} {fmt(summary['default']):>8} -> {fmt(summary['calibrated']):>8}"
              f"   [false alarms at 0 bits: {summary['default_false_alarm_at_0']}, "
              f"{summary['calibrated_false_alarm_at_0']}]  {time.perf_counter() - t0:5.0f}s")
        out["thresholds"].append({"class": cls, "setting": setting, "fault": fault, "rows": rows, **summary})

    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(out, indent=2, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())

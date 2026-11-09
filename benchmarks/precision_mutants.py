"""Which accuracy losses do the relations detect? (review plan, action M-1)

For each operator class, a graded family of mutants keeps 53 - bits significand
bits somewhere in the computation (see ``mutants/precision_mutants.py``). For
each level, the script measures the loss against an accurate reference, in
units of eps, and runs ``dp.verify`` with the default tolerance over several
seeds. It then does the same for the realistic precision mutants (recursive
or single-precision twiddles, a direct DFT with unreduced arguments,
Goertzel, single-precision IIR coefficients).

Usage: python benchmarks/precision_mutants.py [--examples 200] [--seeds 3]
                                               [--max-bits 24] [--step 2]
                                               [--json benchmarks/precision_mutants.json]
"""

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import scipy.signal as ss

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import dsp_props as dp  # noqa: E402
from dsp_props.core.calibration import _probe_signals, extended_dft, relative_error  # noqa: E402
from dsp_props.core.iir import longdouble_lfilter  # noqa: E402
from mutants import precision_mutants as pm  # noqa: E402
from mutants.streaming_mutants import MovingSum  # noqa: E402

LENGTHS = [1, 2, 3, 5, 7, 8, 16, 17, 31, 64, 100, 127, 128, 255, 256, 257, 511, 512]


def _run_stream(cls, block=None):
    """Run a streaming processor on x, whole or in blocks of ``block`` samples."""
    def op(x):
        proc, x = cls(), np.asarray(x, float)
        if block is None:
            return proc.process(x)
        parts = [proc.process(x[i:i + block]) for i in range(0, len(x), block)]
        return np.concatenate(parts) if parts else proc.process(x)
    return op


REFERENCES = {
    "fft": extended_dft,
    "iir": lambda x: longdouble_lfilter(pm.B, pm.A, x),
    "resampler": lambda x: ss.resample_poly(x, pm.UP, pm.DOWN),
    "streaming": _run_stream(MovingSum),
}


def measured_loss(op, reference, seed=0, trials=4):
    """Worst relative error of ``op`` over LENGTHS, in eps (the calibration probes)."""
    rng = np.random.default_rng(seed)
    worst = 0.0
    for n in LENGTHS:
        for x in _probe_signals(n, trials, rng):
            worst = max(worst, relative_error(op, reference, x))
    return worst


def detection(declared, examples, seeds):
    """Number of seeds on which dp.verify fails, and the relations that failed."""
    caught, relations = 0, set()
    for seed in range(seeds):
        report = dp.verify(declared, max_examples=examples, seed=seed, raise_on_failure=False)
        caught += not report.ok
        relations.update(f.relation for f in report.failures)
    return caught, sorted(relations)


def _callable(cls_name, declared):
    if cls_name.startswith("streaming"):
        return _run_stream(declared.__dsp_props__.make, block=1)  # state rounded after every sample
    return declared


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples", type=int, default=200)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--max-bits", type=int, default=24)
    parser.add_argument("--step", type=int, default=2)
    parser.add_argument("--json", help="write the results here")
    args = parser.parse_args()

    out = {"examples": args.examples, "seeds": args.seeds, "graded": {}, "realistic": {}, "threshold": {}}
    for cls_name, make in pm.GRADED.items():
        rows = []
        print(f"\n{cls_name}: bits dropped, measured loss (eps), seeds failing / {args.seeds}, relations")
        for bits in range(0, args.max_bits + 1, args.step):
            declared = make(bits)
            t0 = time.perf_counter()
            loss = measured_loss(_callable(cls_name, declared), REFERENCES[cls_name.split("_")[0]])
            caught, rels = detection(declared, args.examples, args.seeds)
            dt = time.perf_counter() - t0
            print(f"  {bits:3d}  {loss:12.3g}  {caught}/{args.seeds}  {dt:5.1f}s  {', '.join(rels) or '-'}")
            rows.append({"bits": bits, "loss_eps": loss, "seeds_failing": caught, "relations": rels})
        out["graded"][cls_name] = rows
        always = [r for r in rows if r["seeds_failing"] == args.seeds]
        # smallest loss from which every larger level is caught on every seed
        first = None
        for r in reversed(rows):
            if r["seeds_failing"] != args.seeds:
                break
            first = r
        largest_missed = max((r["loss_eps"] for r in rows if r["seeds_failing"] == 0), default=None)
        out["threshold"][cls_name] = {
            "caught_from_loss_eps": first["loss_eps"] if first else None,
            "caught_from_bits": first["bits"] if first else None,
            "largest_loss_never_caught_eps": largest_missed,
            "any_caught": bool(always),
        }

    print(f"\nrealistic precision mutants: measured loss (eps), seeds failing / {args.seeds}, relations")
    realistic_refs = {"fft": extended_dft, "iir": REFERENCES["iir"]}
    for name, declared in pm.REALISTIC.items():
        cls_name = name.split("_")[0]
        loss = measured_loss(declared, realistic_refs[cls_name])
        caught, rels = detection(declared, args.examples, args.seeds)
        print(f"  {name:<34} {loss:12.3g}  {caught}/{args.seeds}  {', '.join(rels) or '-'}")
        out["realistic"][name] = {"loss_eps": loss, "seeds_failing": caught, "relations": rels}

    print("\nreferences (correct implementations): measured loss (eps), seeds failing")
    for name, f in pm.FFT_REFERENCES.items():
        declared = pm.declare_fft(f)
        loss = measured_loss(declared, extended_dft)
        caught, _ = detection(declared, args.examples, args.seeds)
        print(f"  {name:<34} {loss:12.3g}  {caught}/{args.seeds}")
        out.setdefault("references", {})[name] = {"loss_eps": loss, "seeds_failing": caught}

    print("\nsmallest loss caught on every seed, with every larger level also caught:")
    for cls_name, t in out["threshold"].items():
        print(f"  {cls_name:<10} from {t['caught_from_loss_eps']} eps ({t['caught_from_bits']} bits); "
              f"largest loss never caught: {t['largest_loss_never_caught_eps']} eps")
    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(out, indent=2, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Rediscovery campaign: do the relations find real bugs of past releases of DSP libraries?

Usage: python benchmarks/rediscovery.py [--examples 200] [--seeds 0 1 2] [--only KEY] [--json out.json]

For every bug of ``benchmarks/bugcorpus/corpus.py``, the declaration of the function is checked
on the last release that has the bug and on the first one that fixes it (each installed with
``uv`` in its own environment, called through ``benchmarks/bugcorpus/worker.py``), with the same
seeds. A bug counts as *rediscovered* for a seed when some relation fails on the faulty release and
passes on the fixed one; a relation that fails on both says something about the declaration (or a
behaviour the fix kept), not about the bug, and is reported apart. ``docs/rediscovery.md`` gives the
protocol and ``benchmarks/bugcorpus/triage.tsv`` the classification of every candidate.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "benchmarks")]

from bugcorpus.corpus import BUGS, DECLARATIONS  # noqa: E402
from bugcorpus.remote import Remote  # noqa: E402

import dsp_props as dp  # noqa: E402

SETUP = {
    "scipy": "import scipy, scipy.signal, scipy.fft",
    "librosa": "import scipy, scipy.signal\nimport librosa, librosa.effects",
}


def signal_length(example):
    """Length of the input signal of a shrunk counterexample (a family's is ``(parameters, input)``)."""
    if isinstance(example, tuple) and len(example) == 2 and not hasattr(example[0], "shape"):
        example = example[1]  # family: drop the parameters
    if hasattr(example, "shape") and getattr(example, "ndim", 0) >= 1:
        return int(example.shape[0])
    if isinstance(example, (list, tuple)):
        for item in example:
            n = signal_length(item)
            if n is not None:
                return n
    if isinstance(example, int) and not isinstance(example, bool):
        return example
    return None


def run_version(bug, env, examples, seeds):
    """Verify every declaration of the bug's function on one release; one entry per seed."""
    with Remote(env, SETUP[bug.library]) as remote:
        operators = DECLARATIONS[bug.function](remote)
        runs = []
        for seed in seeds:
            per_seed = {}
            for name, op in operators.items():
                t0 = time.perf_counter()
                report = dp.verify(op, max_examples=examples, seed=seed, raise_on_failure=False)
                per_seed[name] = {
                    "seconds": round(time.perf_counter() - t0, 1),
                    "checked": list(report.checked),
                    "failures": {f.relation: {"message": f.message[:300], "examples_tried": f.examples_tried,
                                              "counterexample_length": signal_length(f.example),
                                              "counterexample": dp.report.describe(f.example)}
                                 for f in report.failures},
                }
            runs.append(per_seed)
        return remote.versions, runs


def compare(faulty_runs, fixed_runs, seeds):
    """Per seed: the (operator, relation) pairs failing on the faulty release only, and on both."""
    out = []
    for seed, bad, good in zip(seeds, faulty_runs, fixed_runs):
        only, both = [], []
        for op, res in bad.items():
            for rel, failure in res["failures"].items():
                entry = {"operator": op, "relation": rel, **{k: failure[k] for k in
                         ("examples_tried", "counterexample_length", "message")}}
                (both if rel in good[op]["failures"] else only).append(entry)
        fixed_only = [{"operator": op, "relation": rel, "message": f["message"]}
                      for op, res in good.items() for rel, f in res["failures"].items()
                      if rel not in bad[op]["failures"]]
        out.append({"seed": seed, "rediscovered": bool(only), "faulty_only": only, "both": both,
                    "fixed_only": fixed_only})
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples", type=int, default=200)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--only", help="run the bugs whose key contains this text")
    parser.add_argument("--json", type=pathlib.Path, default=ROOT / "benchmarks" / "rediscovery.json")
    args = parser.parse_args()

    bugs = [b for b in BUGS if not args.only or args.only in b.key]
    previous = json.loads(args.json.read_text()) if args.only and args.json.exists() else {"bugs": []}
    rows = {r["key"]: r for r in previous["bugs"]}
    for bug in bugs:
        t0 = time.perf_counter()
        print(f"{bug.key}: {bug.title}", flush=True)
        try:
            v_bad, bad = run_version(bug, bug.faulty, args.examples, args.seeds)
            v_good, good = run_version(bug, bug.fixed, args.examples, args.seeds)
        except Exception as exc:  # an environment that cannot be built is reported, not hidden
            print(f"  ERROR {type(exc).__name__}: {exc}", flush=True)
            rows[bug.key] = {"key": bug.key, "error": f"{type(exc).__name__}: {exc}"}
            continue
        seeds = compare(bad, good, args.seeds)
        found = sum(s["rediscovered"] for s in seeds)
        firsts = [min(e["examples_tried"] for e in s["faulty_only"] if e["examples_tried"] is not None)
                  for s in seeds if s["rediscovered"] and any(e["examples_tried"] for e in s["faulty_only"])]
        sizes = [min(e["counterexample_length"] for e in s["faulty_only"] if e["counterexample_length"] is not None)
                 for s in seeds if s["rediscovered"]
                 and any(e["counterexample_length"] is not None for e in s["faulty_only"])]
        rows[bug.key] = {
            "key": bug.key, "library": bug.library, "function": bug.function, "title": bug.title,
            "refs": list(bug.refs), "fix_commit": bug.fix_commit, "ours": bug.ours, "trigger": bug.trigger,
            "faulty_versions": v_bad, "fixed_versions": v_good, "fixed_label": bug.fixed_label,
            "rediscovered_seeds": found, "seeds": seeds,
            "inputs_before_first_failure": firsts, "counterexample_lengths": sizes,
            "seconds": round(time.perf_counter() - t0, 1),
            "faulty_runs": bad, "fixed_runs": good,
        }
        status = f"found {found}/{len(args.seeds)}" if found else "not found"
        print(f"  {status}  first={firsts} N={sizes}  ({rows[bug.key]['seconds']} s)", flush=True)
        for s in seeds:
            for e in s["faulty_only"]:
                print(f"    seed {s['seed']} faulty only: {e['operator']} / {e['relation']}: {e['message'][:120]}")
            for e in s["both"]:
                print(f"    seed {s['seed']} both: {e['operator']} / {e['relation']}: {e['message'][:120]}")
            for e in s["fixed_only"]:
                print(f"    seed {s['seed']} fixed only: {e['operator']} / {e['relation']}: {e['message'][:120]}")
    ordered = [rows[b.key] for b in BUGS if b.key in rows]
    args.json.write_text(json.dumps({"examples": args.examples, "seeds": args.seeds, "bugs": ordered},
                                    indent=1, default=str) + "\n")
    done = [r for r in ordered if "error" not in r and not r["ours"]]
    print(f"\nRediscovered (at least one seed, our own reports excluded): "
          f"{sum(r['rediscovered_seeds'] > 0 for r in done)}/{len(done)}")


if __name__ == "__main__":
    main()

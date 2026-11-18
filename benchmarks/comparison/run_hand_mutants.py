"""The 31 hand-written mutants of mutants/ under dsp-props and its ablations.

The automatic mutants of run_comparison.py turn out to be either equivalent or
grossly wrong; the hand-written ones include accuracy faults (single-precision
intermediates, a gain wrong by 1 %), which is where the tolerance models matter.

    python benchmarks/comparison/run_hand_mutants.py [--seeds 5]

Writes benchmarks/comparison/hand_mutants.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT / "src"), str(ROOT)]

import approaches  # noqa: E402
import numpy as np  # noqa: E402

from mutants import ALL_MUTANTS, REFERENCES  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--out", default=str(HERE / "hand_mutants.json"))
    args = p.parse_args(argv)
    rows = []
    targets = [(f"reference:{k}", v) for k, v in REFERENCES.items()] + list(ALL_MUTANTS.items())
    for name, func in targets:
        for approach in approaches.RANDOMISED:
            for seed in range(args.seeds):
                t0 = time.perf_counter()
                with warnings.catch_warnings(), np.errstate(all="ignore"):
                    warnings.simplefilter("ignore")
                    out = approaches.run_approach(approach, {name: func}, [], seed)
                rows.append({"mutant": name, "approach": approach, "seed": seed, "killed": out["killed"],
                             "failures": out["failures"][:1], "seconds": round(time.perf_counter() - t0, 2)})
            rate = np.mean([r["killed"] for r in rows if r["mutant"] == name and r["approach"] == approach])
            print(f"{name:36s} {approach:20s} killed in {rate:.0%} of seeds", flush=True)
        Path(args.out).write_text(json.dumps({"seeds": args.seeds, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()

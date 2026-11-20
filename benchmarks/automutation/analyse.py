"""Summary of the automatic mutation run (plan 3.2): who kills what, and why the rest survives.

Usage: python benchmarks/automutation/analyse.py   (reads results.json and lines_reached.json)

For each suite, a surviving mutant is either *not reached* (its line never runs under that suite,
so no assertion or relation can see it) or *reached*. Writes ``summary.json``, read by
``benchmarks/make_results.py``.
"""

from __future__ import annotations

import collections
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent


def killed(outcome):
    return outcome in ("killed", "timeout")


def main():
    d = json.loads((HERE / "results.json").read_text())
    reached = {k: set(v) for k, v in json.loads((HERE / "lines_reached.json").read_text()).items()}
    rows = [r for r in d["mutants"] if r["function"]]  # one mutant outside the four functions
    out = {"mutants": len(rows), "by_function": {}, "overall": {}}
    for scope in ("all", *sorted({r["function"] for r in rows})):
        sel = [r for r in rows if scope == "all" or r["function"] == scope]
        c = collections.Counter()
        for r in sel:
            up, dsp = killed(r["upstream"]), killed(r["dsp_default"])
            c["mutants"] += 1
            c["upstream"] += up
            c["dsp_fast"] += killed(r["dsp_fast"])
            c["dsp_default"] += dsp
            c["union"] += up or dsp
            c["both" if up and dsp else "upstream_only" if up else "dsp_only" if dsp else "neither"] += 1
            for suite, key in (("upstream", "upstream"), ("dsp_default", "dsp")):
                if not killed(r[suite]):
                    c[f"{key}_survived_not_reached" if r["line"] not in reached[suite]
                      else f"{key}_survived_reached"] += 1
            if up and not dsp and r["line"] in reached["dsp_default"]:
                c["upstream_only_reached_by_dsp"] += 1
        (out["overall"] if scope == "all" else out["by_function"].setdefault(scope, {})).update(c)
    out["lines"] = {k: len(v) for k, v in reached.items()}
    (HERE / "summary.json").write_text(json.dumps(out, indent=1) + "\n")
    for k, v in [("all", out["overall"]), *out["by_function"].items()]:
        print(k, dict(v))


if __name__ == "__main__":
    main()

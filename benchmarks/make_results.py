"""Generate the numbers the documentation quotes from the benchmark result files.

    python benchmarks/make_results.py            # write results.md, results.tex, update the docs
    python benchmarks/make_results.py --check    # exit 1 if anything is out of date (CI)

Reads the JSON and CSV files the benchmark scripts write (``mutant_results.json``,
``precision_mutants.json``, ``precision_oracles.json``, ``calibrated_tolerance.json``,
``validation.json``, ``error_profile.csv``, ``rediscovery.json``, ``automutation/summary.json``) and derives one value per key, such as
``precision.fft_twiddles.default`` = ``90 ε``. It then

- writes ``benchmarks/results.md`` (one table: key, value, source file) and
  ``benchmarks/results.tex`` (one LaTeX macro per key, for the papers);
- rewrites every marked number in the documentation. A number quoted in
  ``README.md``, ``CLAUDE.md`` or ``docs/*.md`` is marked as
  ``<!-- r:KEY -->90 ε<!-- /r -->``: the comments do not render, and the text between
  them is replaced by the current value of KEY.

With ``--check`` nothing is written: the script fails if a generated file or a
marked number differs from what the result files give, or if a marker names an
unknown key. So a number copied by hand cannot drift from the script that
measured it.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks"
DOCS = [ROOT / "README.md", ROOT / "CLAUDE.md", *sorted((ROOT / "docs").glob("*.md"))]
RESULTS_MD = BENCH / "results.md"
RESULTS_TEX = BENCH / "results.tex"
MARKER = re.compile(r"<!-- r:([A-Za-z0-9_.]+) -->(.*?)<!-- /r -->")

FAMILIES = ["fft_twiddles", "iir_coefficients", "iir_arithmetic", "resampler_taps",
            "resampler_arithmetic", "streaming_state"]
ORACLES = ["default", "fixed", "calibrated", "differential"]


def sig(x: float, digits: int = 2, plain: bool = False) -> str:
    """``x`` to ``digits`` significant digits, in the style of the docs: 90, 2.2, 430, 7.1e3.

    From 1000 on the value is written m.me3, unless ``plain`` (3400).
    """
    if x == 0:
        return "0"
    e = math.floor(math.log10(abs(x)))
    m = round(x / 10**e, digits - 1)
    if m >= 10:  # rounding carried into the next decade
        m, e = m / 10, e + 1
    if e >= 3 and not plain:
        return f"{m:.{digits - 1}f}e{e}"
    return f"{round(x, digits - 1 - e):.{max(digits - 1 - e, 0)}f}"


def eps(x, digits: int = 2) -> str:
    return "never" if x is None else f"{sig(x, digits)} ε"


class Results:
    """Ordered ``key -> (value, source)``; values are the strings the docs print."""

    def __init__(self):
        self.values: dict[str, tuple[str, str]] = {}

    def add(self, key, value, source):
        if key in self.values:
            raise ValueError(f"duplicate key {key}")
        self.values[key] = (str(value), source)


def load(name):
    return json.loads((BENCH / name).read_text(encoding="utf-8"))


def mutants(r: Results):
    src = "mutant_results.json"
    d = load(src)
    ms = d["mutants"]
    lengths = [m["min_counterexample_length"] for m in ms
               if m["min_counterexample_length"] is not None and m["mutant"] != "iir_wrong_gain"]
    r.add("mutants.total", len(ms), src)
    r.add("mutants.killed", sum(m["killed"] for m in ms), src)
    r.add("mutants.examples", d["examples"], src)
    r.add("mutants.small_counterexamples", sum(n <= 3 for n in lengths), src)
    r.add("mutants.with_counterexample", len(lengths), src)
    r.add("mutants.counterexample_median", f"{statistics.median(lengths):g}", src)
    r.add("mutants.fail_within_3", sum((m["inputs_before_first_failure"] or 10**9) <= 3 for m in ms), src)


def precision(r: Results):
    src = "precision_mutants.json"
    d = load(src)
    r.add("precision.examples", d["examples"], src)
    r.add("precision.seeds", d["seeds"], src)
    for fam in FAMILIES:
        t = d["threshold"][fam]
        r.add(f"precision.{fam}.default", eps(t["caught_from_loss_eps"]), src)
        r.add(f"precision.{fam}.passes_up_to", eps(t["largest_loss_never_caught_eps"]), src)
    for name, row in d["realistic"].items():
        r.add(f"precision.realistic.{name}.loss", eps(row["loss_eps"], 3 if 100 <= row["loss_eps"] < 1e3 else 2), src)
        r.add(f"precision.realistic.{name}.caught", f"{row['seeds_failing']}/{d['seeds']}", src)
    for name, row in d["references"].items():
        r.add(f"precision.reference.{name}.loss", eps(row["loss_eps"]), src)


def oracles(r: Results):
    src = "precision_oracles.json"
    d = load(src)
    for fam in FAMILIES:
        for oracle in ORACLES:
            s = d["summary"][fam][oracle]
            r.add(f"oracles.{fam}.{oracle}", eps(s["caught_from_loss_eps"]), src)
    alarms = sum(d["summary"][f][o]["false_alarm_at_0_bits"] for f in FAMILIES for o in ORACLES)
    r.add("oracles.false_alarms", alarms, src)
    for cls, v in d["calibrated_worst_eps"].items():
        r.add(f"oracles.trusted.{cls}", eps(v), src)


def calibrated(r: Results):
    src = "calibrated_tolerance.json"
    d = load(src)
    rows = d["false_alarms"]
    others = [x for x in rows if x["implementation"] != "trusted"]
    correct = [x for x in others if not x["implementation"].startswith("torch")]
    r.add("calibrated.other_implementations", len(others), src)
    r.add("calibrated.correct_without_alarm", sum(x["seeds_failing"] == 0 for x in correct), src)
    r.add("calibrated.correct_alarms", sum(x["seeds_failing"] > 0 for x in correct), src)
    torch = next(x for x in others if x["implementation"].startswith("torch"))
    r.add("calibrated.torch_fft", eps(torch["worst_eps"], 3), src)
    r.add("calibrated.torch_fft_seeds_failing", f"{torch['seeds_failing']}/{d['seeds']}", src)
    numpy_fft = [x for x in rows if x["class"] == "fft" and x["implementation"] == "trusted"]
    r.add("calibrated.numpy_fft", eps(numpy_fft[0]["worst_eps"]), src)
    gains = [t["default"] / t["calibrated"] for t in d["thresholds"]
             if t["default"] is not None and t["calibrated"] is not None]
    r.add("calibrated.gain_min", sig(min(gains), 1), src)
    r.add("calibrated.gain_max", sig(max(gains), 2, plain=True), src)
    r.add("calibrated.filters", len({t["setting"] for t in d["thresholds"] if t["class"] == "iir"}), src)
    r.add("calibrated.ratios", len({t["setting"] for t in d["thresholds"] if t["class"] == "resampler"}), src)


def validation(r: Results):
    src = "validation.json"
    fp = load(src)["false-positives"]
    r.add("validation.runs", fp["runs"], src)
    r.add("validation.failed_runs", fp["failed_runs"], src)
    r.add("validation.comparisons", fp["comparisons"], src)


def error_profile(r: Results):
    src = "error_profile.csv"
    with (BENCH / src).open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    r.add("profile.n_max", rows[-1]["N"], src)
    for lib in ("numpy", "scipy", "torch"):
        worst = max(rows, key=lambda row: float(row[f"{lib}_eps"]))
        value = float(worst[f"{lib}_eps"])
        r.add(f"profile.{lib}.worst", eps(value, 3 if value >= 100 else 2), src)
        r.add(f"profile.{lib}.worst_n", worst["N"], src)


def comparison(r: Results):
    d = load("comparison/summary.json")
    src = "comparison/summary.json"
    for subject in ("lfilter", "upfirdn"):
        rates = d["kill_rates"][subject]["functional"]
        for approach, key in (("dsp-props", "dsp_props"), ("dsp-props/params", "params"), ("sop", "sop")):
            r.add(f"comparison.{subject}.{key}", f"{100 * rates[approach]['mean']:.0f} %", src)
        r.add(f"comparison.{subject}.mutants", rates["dsp-props"]["n_mutants"], src)


def iir_structure(r: Results):
    d = load("iir_structure.json")
    src = "iir_structure.json"
    ratios = [row["ratio"] for row in d["named"]]
    r.add("iir_structure.named_ratio_min", sig(min(ratios), 2), src)
    r.add("iir_structure.named_ratio_max", sig(max(ratios), 2), src)
    for k in ("min", "median", "max"):
        r.add(f"iir_structure.random_ratio_{k}", sig(d["random_ratio"][k], 2), src)
    r.add("iir_structure.random_gap_median", sig(d["random_gap"]["median"], 2, plain=True), src)
    for key, t in d["thresholds"].items():
        r.add(f"iir_structure.{key}", eps(t["caught_from_loss_eps"]), src)


def rediscovery(r: Results):
    src = "rediscovery.json"
    d = load(src)
    rows = [b for b in d["bugs"] if "error" not in b and not b["ours"]]
    found = [b for b in rows if b["rediscovered_seeds"] > 0]
    firsts = [n for b in found for n in b["inputs_before_first_failure"]]  # one per (bug, seed) found
    sizes = [n for b in found for n in b["counterexample_lengths"]]
    triage = [line.split("\t") for line in (BENCH / "bugcorpus" / "triage.tsv").read_text(encoding="utf-8")
              .splitlines()[2:] if line.strip()]
    r.add("rediscovery.candidates", len(triage), "bugcorpus/triage.tsv")
    r.add("rediscovery.in_scope", sum(t[5] == "in-scope" for t in triage), "bugcorpus/triage.tsv")
    r.add("rediscovery.bugs", len(rows), src)
    r.add("rediscovery.found", len(found), src)
    r.add("rediscovery.found_all_seeds", sum(b["rediscovered_seeds"] == len(d["seeds"]) for b in rows), src)
    r.add("rediscovery.seeds", len(d["seeds"]), src)
    r.add("rediscovery.examples", d["examples"], src)
    r.add("rediscovery.inputs_median", f"{statistics.median(firsts):g}", src)
    r.add("rediscovery.counterexample_median", f"{statistics.median(sizes):g}", src)
    r.add("rediscovery.found_seeds", len(sizes), src)
    r.add("rediscovery.small_counterexamples", sum(n <= 2 for n in sizes), src)


def automutation(r: Results):
    src = "automutation/summary.json"
    d = load(src)
    o = d["overall"]
    r.add("automutation.mutants", o["mutants"], src)
    for key in ("upstream", "dsp_default", "union", "both", "upstream_only", "dsp_only", "neither"):
        r.add(f"automutation.{key}", o[key], src)
    for key in ("upstream", "dsp_default", "union"):
        r.add(f"automutation.{key}_rate", f"{100 * o[key] / o['mutants']:.0f} %", src)
    r.add("automutation.upstream_only_not_reached",
          o["upstream_only"] - o["upstream_only_reached_by_dsp"], src)
    r.add("automutation.upstream_only_reached", o["upstream_only_reached_by_dsp"], src)
    r.add("automutation.dsp_survived_not_reached", o["dsp_survived_not_reached"], src)
    r.add("automutation.dsp_survived_reached", o["dsp_survived_reached"], src)
    h = d["by_function"]["hilbert"]
    r.add("automutation.hilbert_dsp_only", h["dsp_only"], src)
    r.add("automutation.hilbert_upstream_rate", f"{100 * h['upstream'] / h['mutants']:.0f} %", src)
    r.add("automutation.hilbert_dsp_rate", f"{100 * h['dsp_default'] / h['mutants']:.0f} %", src)
    r.add("automutation.dsp_fast", o["dsp_fast"], src)
    r.add("automutation.lines_upstream", d["lines"]["upstream"], src)
    r.add("automutation.lines_dsp", d["lines"]["dsp_default"], src)


def collect() -> Results:
    r = Results()
    for part in (mutants, precision, oracles, calibrated, validation, error_profile, comparison, iir_structure,
                 rediscovery, automutation):
        part(r)
    return r


def render_md(r: Results) -> str:
    lines = ["# Generated results", "",
             "Written by `python benchmarks/make_results.py` from the result files of the benchmark",
             "scripts; do not edit. A number quoted in the documentation is marked",
             "`<!-- r:KEY -->value<!-- /r -->` and checked against this table in CI.", "",
             "| Key | Value | Source |", "| --- | --- | --- |"]
    lines += [f"| `{k}` | {v} | `{s}` |" for k, (v, s) in r.values.items()]
    return "\n".join(lines) + "\n"


def tex_name(key: str) -> str:
    parts = re.split(r"[._]", key)
    words = "".join(p.capitalize() for p in parts)
    for d, w in zip("0123456789", ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven",
                                   "Eight", "Nine"]):
        words = words.replace(d, w)
    return "dp" + words


def tex_value(v: str) -> str:
    v = v.replace("ε", r"\varepsilon")
    v = re.sub(r"(\d+(?:\.\d+)?)e(-?\d+)", r"\1\\times10^{\2}", v)
    return rf"\ensuremath{{{v}}}" if "\\" in v else v


def render_tex(r: Results) -> str:
    lines = ["% Generated by benchmarks/make_results.py; do not edit."]
    lines += [rf"\newcommand{{\{tex_name(k)}}}{{{tex_value(v)}}}  % {k}" for k, (v, _) in r.values.items()]
    return "\n".join(lines) + "\n"


def sync(text: str, r: Results, path: Path, problems: list) -> str:
    def repl(m):
        key, old = m.group(1), m.group(2)
        if key not in r.values:
            problems.append(f"{path.relative_to(ROOT)}: unknown key {key}")
            return m.group(0)
        new = r.values[key][0]
        if old != new:
            problems.append(f"{path.relative_to(ROOT)}: {key} is {old!r}, results give {new!r}")
        return f"<!-- r:{key} -->{new}<!-- /r -->"

    return MARKER.sub(repl, text)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="write nothing, exit 1 if anything is stale")
    args = parser.parse_args(argv)
    r = collect()
    problems: list[str] = []
    outputs = {RESULTS_MD: render_md(r), RESULTS_TEX: render_tex(r)}
    for path, text in outputs.items():
        current = path.read_text(encoding="utf-8") if path.exists() else None
        if current != text:
            problems.append(f"{path.relative_to(ROOT)} is out of date")
            if not args.check:
                path.write_text(text, encoding="utf-8")
    markers = 0
    for path in DOCS:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        markers += len(MARKER.findall(text))
        found: list[str] = []
        new = sync(text, r, path, found)
        problems += found
        if new != text and not args.check:
            path.write_text(new, encoding="utf-8")
    print(f"{len(r.values)} values, {markers} marked numbers in {len(DOCS)} files")
    if problems:
        print("\n".join(("stale: " if args.check else "updated: ") + p for p in problems))
        unknown = [p for p in problems if "unknown key" in p]
        if args.check or unknown:
            print("run: python benchmarks/make_results.py" if args.check else "", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

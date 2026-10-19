"""Turn the result files of the repository into the numbers and tables of the article.

    python paper/article/make_numbers.py

Reads benchmarks/survey_results.json, benchmarks/comparison/results.jsonl
(through run_comparison.summarise), benchmarks/validation.json,
benchmarks/mutant_results.json and benchmarks/error_profile.csv, and writes
paper/article/numbers.tex and paper/article/tables/*.tex. Nothing in the
article is typed by hand.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BENCH = ROOT / "benchmarks"
sys.path[:0] = [str(BENCH / "comparison"), str(ROOT / "src")]

import run_comparison  # noqa: E402

APPROACH_LABELS = {"sop": "SoP", "dsp-props": "dsp-props", "dsp-props/uniform": "uniform",
                   "dsp-props/fixed-tol": "fixed-tol", "generic-pbt": "generic PBT"}
SUBJECT_LABELS = {"fft": "FFT", "lfilter": r"\code{lfilter}", "upfirdn": r"\code{upfirdn}",
                  "ola": "overlap-add", "all": "all"}

# Facts that come from the bug-hunting campaign reports (benchmarks/RESULTS-2026-10-14*.md),
# which are written by hand from the campaign output, and from the upstream issue trackers.
CAMPAIGN = {"campaignTargets": "46", "campaignTriaged": "three",
            "campaignTriagedShare": rf"{100 * 3 / 46:.1f}\,\%"}

# First pass of the comparison, before the oracle was extended (commit e62b0fb, message of aa19a04):
# it classified 158 mutants as functional and 25 as equivalent. The eight mutants that the extended
# oracle reclassified as functional are listed here; who detects them is read from the current results.
# Recompute with: git show e62b0fb:benchmarks/comparison/results.jsonl > old.jsonl and
# run_comparison.summarise(Path("old.jsonl")).
ORACLE_FIRST = {"functional": 158, "equivalent": 25}
RECLASSIFIED = ["lfilter-019", "lfilter-021", "lfilter-040", "lfilter-042",
                "upfirdn-012", "upfirdn-013", "upfirdn-014", "upfirdn-015"]


def pct(x):
    return rf"{100 * x:.0f}\,\%"


def tex_float(x, digits=2):
    if x == 0:
        return "0"
    e = math.floor(math.log10(abs(x)))
    m = x / 10 ** e
    if -2 <= e <= 2:
        return f"{x:.{max(digits - 1 - e, 0)}f}" if e < digits else f"{x:.0f}"
    return rf"{m:.{digits - 1}f}\times10^{{{e}}}"


def sci(x):
    e = round(math.log10(x))
    return rf"$10^{{{e}}}$" if abs(x / 10 ** e - 1) < 1e-9 else f"${tex_float(x)}$"


def survey(macros):
    s = json.loads((BENCH / "survey_results.json").read_text())
    sig = s["SciPy (scipy.signal)"]
    macros["surveyScipyDistinctTol"] = str(sig["distinct_tolerance_values"])
    macros["surveyScipyTolDecades"] = str(round(sig["tolerance_decades"]))
    macros["surveyScipyTolMin"] = sci(sig["tolerance_min"])
    macros["surveyScipyTolMax"] = f"${sig['tolerance_max']:g}$"
    macros["surveyScipyHardcoded"] = str(sig["tests_hardcoded_expected"])
    macros["surveyTotalTests"] = f"{sum(v['tests'] for v in s.values()):,}".replace(",", "\\,")
    macros["surveyZiEmpty"] = str(sum(v["tests_zi_and_empty"] for v in s.values()))
    macros["surveyScipyTests"] = f"{sig['tests']:,}".replace(",", "\\,")
    macros["surveyScipyComparisons"] = f"{sig['comparisons']:,}".replace(",", "\\,")
    macros["surveyScipyExplicit"] = str(sig["explicit_tolerance"])
    macros["surveyScipyEmpty"] = str(sig["tests_empty_input"])
    macros["surveyScipyZi"] = str(sig["tests_zi"])
    macros["surveyHardcodedAll"] = str(sum(v["tests_hardcoded_expected"] for v in s.values()))
    macros["surveyLibrosaTests"] = str(s["librosa"]["tests"])
    macros["surveyLibrosaCompared"] = str(s["librosa"]["tests_with_comparison"])
    rel, ab = sig["tolerance_by_kind"]["relative"], sig["tolerance_by_kind"]["absolute"]
    macros["surveyScipyRelCount"] = str(rel["count"])
    macros["surveyScipyRelDistinct"] = str(rel["distinct"])
    macros["surveyScipyRelBelowEps"] = str(rel["below_eps"])
    macros["surveyScipyRelMin"] = sci(rel["min_from_eps"])
    macros["surveyScipyRelMax"] = f"${rel['max']:g}$"
    macros["surveyScipyRelDecades"] = str(round(math.log10(rel["max"] / rel["min_from_eps"])))
    macros["surveyScipyAbsCount"] = str(ab["count"])
    macros["surveyScipyAbsDistinct"] = str(ab["distinct"])
    hist = sorted(((float(k), n) for k, n in rel["histogram"].items()), reverse=True)
    top = sorted(hist, key=lambda kv: -kv[1])[:3]
    macros["surveyScipyTopTols"] = ", ".join(f"$10^{{{round(math.log10(k))}}}$ ({n})" for k, n in top)
    lines = ["decade,count"] + [f"{round(math.log10(k))},{n}" for k, n in sorted(hist)]
    (HERE / "tables" / "tolhist.csv").write_text("\n".join(lines) + "\n")
    rows = []
    for name, v in s.items():
        r = v["tolerance_by_kind"]["relative"]
        lo, hi = r["min"], r["max"]
        span = sci(lo) if lo == hi else f"{sci(lo)}--{sci(hi) if hi < 0.01 else ('$%g$' % hi)}"
        rows.append(f"{name.replace('_', r'_')} & {v['tests']} & {v['tests_hypothesis']} & {v['comparisons']} & "
                    f"{v['explicit_tolerance']} & {r['count']} & {r['distinct']} & {span} & {v['tests_empty_input']} & "
                    f"{v['tests_zi_and_empty']} & {v['tests_hardcoded_expected']} \\\\")
    table = r"""\begin{table*}[t]
\caption{Test suites of five DSP libraries (syntactic counts). Tol.: comparisons with a hand-written tolerance; rel.: hand-written relative tolerances (\code{rtol}, \code{nulp} and similar), their number of distinct values and their range (absolute tolerances, in the units of the data, are not comparable across tests and are left out); empty: tests with an empty input; zi+empty: tests that also pass an initial state; hard-coded: tests that compare with an array of at least four typed-in floating-point values}
\label{tab:survey}
\centering\footnotesize
\begin{tabular}{@{}lrrrrrrlrrr@{}}
\toprule
Suite & Tests & Hypothesis & Comparisons & Tol. & Rel. & Distinct & Rel.\ range & Empty & zi+empty & Hard-coded \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    (HERE / "tables" / "survey.tex").write_text(table)


def comparison(macros):
    results = BENCH / "comparison" / "results.jsonl"
    summ = run_comparison.summarise(results)
    (BENCH / "comparison" / "summary.json").write_text(json.dumps(summ, indent=1))
    cls = summ["classification"]
    macros["nMutAuto"] = str(cls["all"]["total"])
    for subj, key in [("fft", "nMutFft"), ("lfilter", "nMutLfilter"), ("upfirdn", "nMutUpfirdn"), ("ola", "nMutOla")]:
        macros[key] = str(cls[subj]["total"])
    for cat, key in [("functional", "nFunctional"), ("accuracy", "nAccuracy"), ("equivalent", "nEquivalent")]:
        macros[key] = str(cls["all"].get(cat, 0))
    kr = summ["kill_rates"]["all"]
    short = {"sop": "Sop", "dsp-props": "Dsp", "dsp-props/uniform": "Uni", "dsp-props/fixed-tol": "Fix",
             "generic-pbt": "Gen"}
    for cat, ckey in [("functional", "Func"), ("accuracy", "Acc")]:
        for a, akey in short.items():
            v = kr.get(cat, {}).get(a)
            macros[f"kill{ckey}{akey}"] = pct(v["mean"]) if v else "--"
    seeds = max(v["n_seeds"] for v in kr["functional"].values())
    macros["nSeeds"] = str(seeds)
    rows = []
    approaches = [a for a in run_comparison.approaches.APPROACHES if a in kr["functional"]]
    for subj in ["fft", "lfilter", "upfirdn", "ola", "all"]:
        for cat in ["functional", "accuracy", "equivalent"]:
            n = summ["classification"][subj].get(cat, 0)
            if n == 0:
                continue
            cells = []
            for a in approaches:
                v = summ["kill_rates"].get(subj, {}).get(cat, {}).get(a)
                if v is None:
                    cells.append("--")
                elif "min" in v and v["max"] - v["min"] > 1e-9:
                    cells.append(rf"{100 * v['mean']:.0f} ({100 * v['min']:.0f}--{100 * v['max']:.0f})")
                else:
                    cells.append(f"{100 * v['mean']:.0f}")
            label = SUBJECT_LABELS[subj] if cat == "functional" or subj == "all" else ""
            rule = r"\midrule" + "\n" if subj == "all" and cat == "functional" else ""
            rows.append(f"{rule}{SUBJECT_LABELS[subj]} & {cat} ({n}) & " + " & ".join(cells) + r" \\")
    head = " & ".join(APPROACH_LABELS[a] for a in approaches)
    table = r"""\begin{table*}[t]
\caption{Mutants detected (\%, mean over seeds, with min--max when seeds disagree). For equivalent mutants the figure is a false-alarm rate and should be 0. SoP: ported test suites of NumPy and SciPy; uniform: dsp-props without its generators; fixed-tol: without its tolerance models; generic PBT: without both}
\label{tab:comparison}
\centering\footnotesize
\begin{tabular}{@{}llrrrrr@{}}
\toprule
Subject & Class (mutants) & """ + head + r""" \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    (HERE / "tables" / "comparison.tex").write_text(table)
    rec = [json.loads(line) for line in results.read_text().splitlines()]
    limits = {r["time_limit"] for r in rec if r["approach"] != "oracle" and r.get("timeout") and "time_limit" in r}
    macros["timeoutSeconds"] = str(max(limits)) if limits else "45"
    # complementarity of SoP and dsp-props on functional mutants (killed = by a majority of seeds)
    func = {m: v for m, v in summ["per_mutant"].items() if v["class"] == "functional"}
    sop = {m for m, v in func.items() if v["sop"] >= 0.5}
    dsp = {m for m, v in func.items() if v["dsp-props"] >= 0.5}
    macros["killSopOnly"] = str(len(sop - dsp))
    macros["killDspOnly"] = str(len(dsp - sop))
    macros["killNeither"] = str(len(set(func) - sop - dsp))
    macros["killUnion"] = pct(len(sop | dsp) / len(func))
    randomised = [a for a in short if a != "sop"]
    macros["ablationDisagree"] = str(sum(len({v[a] >= 0.5 for a in randomised}) > 1 for v in func.values()))
    # How dsp-props detects (seed 0): a relation violated, or an exception or time-out of the mutant
    dsp0 = [r for r in rec if r["approach"] == "dsp-props" and r["seed"] == 0 and r["killed"]
            and not r["mutant"].endswith("-orig")]
    kinds = [r["failures"][0]["error"] for r in dsp0]
    violated = {"ToleranceError", "AssertionError"}
    macros["dspKillsSeedZero"] = str(len(dsp0))
    macros["dspKillTolerance"] = str(kinds.count("ToleranceError"))
    macros["dspKillAssert"] = str(kinds.count("AssertionError"))
    macros["dspKillRelation"] = str(sum(k in violated for k in kinds))
    macros["dspKillOther"] = str(sum(k not in violated for k in kinds))
    macros["dspKillTimeout"] = str(kinds.count("Timeout"))
    macros["dspKillGroup"] = str(kinds.count("ExceptionGroup"))
    per_seed = {s: sum(r["failures"][0]["error"] in violated for r in rec if r["approach"] == "dsp-props"
                       and r["seed"] == s and r["killed"] and not r["mutant"].endswith("-orig"))
                for s in range(seeds)}
    macros["dspKillRelationStable"] = "in every seed" if len(set(per_seed.values())) == 1 else \
        f"between {min(per_seed.values())} and {max(per_seed.values())} across seeds"
    # Functional mutants that run on the oracle's inputs without raising or hanging
    oracle = {r["mutant"]: r for r in rec if r["approach"] == "oracle"}
    quiet = [m for m in func if not (oracle[m].get("crashed") or oracle[m].get("timeout"))]
    macros["nFuncQuiet"] = str(len(quiet))
    macros["nFuncLoud"] = str(len(func) - len(quiet))
    for a, akey in short.items():
        rates = [np.mean([r["killed"] for r in rec if r["approach"] == a and r["seed"] == s and r["mutant"] in quiet])
                 for s in range(seeds if a != "sop" else 1)]
        lo, hi = min(rates), max(rates)
        macros[f"killQuiet{akey}"] = pct(np.mean(rates)) if hi - lo < 1e-9 else \
            rf"{100 * np.mean(rates):.0f}\,\% ({100 * lo:.0f}--{100 * hi:.0f})"
    # The oracle's first pass and the mutants it reclassified
    macros["oracleFirstFunctional"] = str(ORACLE_FIRST["functional"])
    macros["oracleFirstEquivalent"] = str(ORACLE_FIRST["equivalent"])
    macros["nReclassified"] = str(len(RECLASSIFIED))
    assert all(summ["per_mutant"][m]["class"] == "functional" for m in RECLASSIFIED)
    recl_sop = [m for m in RECLASSIFIED if summ["per_mutant"][m]["sop"] >= 0.5]
    recl_dsp = [m for m in RECLASSIFIED if summ["per_mutant"][m]["dsp-props"] >= 0.5]
    macros["reclSopOnly"] = str(len(set(recl_sop) - set(recl_dsp)))
    macros["reclDspOnly"] = str(len(set(recl_dsp) - set(recl_sop)))
    macros["reclSopOnlySubject"] = ", ".join(sorted({m.split("-")[0] for m in set(recl_sop) - set(recl_dsp)}))
    macros["reclDspOnlySubject"] = ", ".join(sorted({m.split("-")[0] for m in set(recl_dsp) - set(recl_sop)}))
    macros["reclDspAllVariants"] = "every" if all(
        summ["per_mutant"][m][a] == 1.0 for m in recl_dsp for a in short if a != "sop") else "not every"
    ff, cl, sec = summ["first_failure"], summ["counterexample_length"], summ["seconds"]
    for a, akey in short.items():
        if a in ff:
            macros[f"firstMed{akey}"] = f"{ff[a]['median']:g}"
            macros[f"firstPnn{akey}"] = f"{ff[a]['p90']:.0f}"
            macros[f"cexMed{akey}"] = f"{cl[a]['median']:g}"
            macros[f"cexPnn{akey}"] = f"{cl[a]['p90']:.0f}"
        macros[f"secMed{akey}"] = f"{sec[a]['median']:.3g}"
    return summ


def hand_ablation(macros):
    path = BENCH / "comparison" / "hand_mutants.json"
    if not path.exists():
        (HERE / "tables" / "hand.tex").write_text("% hand_mutants.json not produced yet\n")
        return
    data = json.loads(path.read_text())
    rows = data["rows"]
    names = list(dict.fromkeys(r["mutant"] for r in rows))
    approaches = [a for a in run_comparison.approaches.RANDOMISED if any(r["approach"] == a for r in rows)]
    lines = []
    families = {"fft": ("FFT", "fft"), "rfft": ("real FFT", "rfft"), "dct": ("DCT", "dct"),
                "iir": ("IIR filter", "iir"), "rate": ("resampler", "multirate"),
                "stream": ("streaming", "streaming")}
    for fam, (title, ref) in families.items():
        refs = [n for n in names if n == f"reference:{ref}"]
        muts = [n for n in names if not n.startswith("reference:") and n.split("_")[0] == fam]
        for label, group in [(f"{title}, reference", refs), (f"{title}, {len(muts)} mutants", muts)]:
            if not group:
                continue
            cells = []
            for a in approaches:
                ks = [r["killed"] for r in rows if r["mutant"] in group and r["approach"] == a]
                cells.append(f"{sum(ks)}/{len(ks)}")
            lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    totals = []
    mutants = [n for n in names if not n.startswith("reference:")]
    for a in approaches:
        per = [any(r["killed"] for r in rows if r["mutant"] == n and r["approach"] == a and r["seed"] == s)
               for n in mutants for s in range(data["seeds"])]
        killed_all = sum(all(r["killed"] for r in rows if r["mutant"] == n and r["approach"] == a) for n in mutants)
        totals.append(f"{killed_all}")
        key = {"dsp-props": "Dsp", "dsp-props/uniform": "Uni", "dsp-props/fixed-tol": "Fix", "generic-pbt": "Gen"}[a]
        macros[f"hand{key}Always"] = str(killed_all)
        macros[f"hand{key}Never"] = str(sum(not any(r["killed"] for r in rows if r["mutant"] == n and r["approach"] == a)
                                            for n in mutants))
        kills = [r["failures"][0] for r in rows if r["approach"] == a and r["killed"]
                 and not r["mutant"].startswith("reference:") and r["failures"]]
        first = np.array([f["first_failure"] for f in kills if f.get("first_failure") is not None])
        cex = np.array([f["counterexample_length"] for f in kills if f.get("counterexample_length") is not None])
        macros[f"hand{key}FirstMed"] = f"{np.median(first):g}"
        macros[f"hand{key}FirstPnn"] = f"{np.percentile(first, 90):.0f}"
        macros[f"hand{key}CexPnn"] = f"{np.percentile(cex, 90):.0f}"
        macros[f"hand{key}FalseAlarms"] = str(sum(r["killed"] for r in rows
                                                  if r["approach"] == a and r["mutant"].startswith("reference:")))
    keys = [{"dsp-props": "Dsp", "dsp-props/uniform": "Uni", "dsp-props/fixed-tol": "Fix", "generic-pbt": "Gen"}[a]
            for a in approaches]
    head = " & ".join(APPROACH_LABELS[a] for a in approaches)
    table = r"""\begin{table}[t]
\caption{Hand-written mutants by family: runs that report a failure, over mutants and """ + str(data["seeds"]) + r""" seeds. A correct reference must never fail}
\label{tab:hand}
\centering\scriptsize
\begin{tabular}{@{}lrrrr@{}}
\toprule
Mutant & """ + head + r""" \\
\midrule
""" + "\n".join(lines) + r"""
\midrule
mutants always detected & """ + " & ".join(totals) + r""" \\
first failure, median (p90) & """ + " & ".join(f"{macros['hand' + k + 'FirstMed']} ({macros['hand' + k + 'FirstPnn']})" for k in keys) + r""" \\
counterexample length, p90 & """ + " & ".join(macros['hand' + k + 'CexPnn'] for k in keys) + r""" \\
\bottomrule
\end{tabular}
\end{table}
"""
    (HERE / "tables" / "hand.tex").write_text(table)


def validation(macros):
    v = json.loads((BENCH / "validation.json").read_text())
    env = v["environment"]
    macros["scipyVersion"] = env["scipy"]
    fp = v["false-positives"]
    macros["fpSeeds"] = str(fp["rows"][0]["seeds"])
    macros["fpRuns"] = f"{fp['runs']:,}".replace(",", "\\,")
    macros["fpFailed"] = "no" if fp["failed_runs"] == 0 else str(fp["failed_runs"])
    macros["fpComparisons"] = f"{fp['comparisons']:,}".replace(",", "\\,")
    q = fp["margin_quantiles"]
    macros["marginExact"] = pct(fp["exact_comparisons"] / fp["comparisons"])
    macros["marginPnn"] = f"${tex_float(q['99'])}$"
    macros["marginMax"] = f"${tex_float(q['100'])}$"
    macros["marginFactor"] = f"{1 / q['100']:.0f}"
    kb = v["known-bug"]["runs"]
    macros["knownBugRuns"] = str(len(kb))
    macros["knownBugFlagged"] = str(sum(not r["passed"] for r in kb))
    ref = v["reference"]
    macros["refDftMax"] = f"${tex_float(ref['dft_max_in_eps64'])}$"
    macros["refLfilterMax"] = f"${tex_float(ref['lfilter_max_in_eps64'])}$"
    g = v["generators"]
    macros["genPathoCount"] = str(len(g["lengths"]["pathological_values_hit"]))
    runs = g["lengths"]["runs_of_200"]
    macros["genPathoShare"] = pct(runs["mean_fraction_pathological"])
    macros["genPathoShareMin"] = pct(runs["min"])
    macros["genPathoShareMax"] = pct(runs["max"])
    macros["genPathoRuns"] = str(runs["runs"])
    macros["genPathoLongDraws"] = str(g["lengths"]["draws"])
    macros["genPathoLong"] = pct(g["lengths"]["fraction_pathological"])
    macros["genEmptyBlock"] = pct(g["block_splits"]["fraction_with_empty_block"])
    macros["genOneBlock"] = pct(g["block_splits"]["fraction_with_one_sample_block"])
    macros["genCloseBelowOne"] = pct(g["signals"]["close_tones"]["fraction_below_one_bin"])
    macros["genOnGrid"] = pct(g["signals"]["offgrid_sine"]["fraction_on_dft_grid"])
    macros["genZeroNyq"] = pct(g["signals"]["offgrid_sine"]["fraction_freq_0_or_nyquist"])
    macros["envDescription"] = (f"Python {env['python']}, NumPy {env['numpy']}, SciPy {env['scipy']}, "
                                f"Hypothesis {env['hypothesis']}, on a 4-core Linux x86-64 machine")


def hand_mutants(macros):
    m = json.loads((BENCH / "mutant_results.json").read_text())["mutants"]
    macros["nMutHand"] = str(len(m))
    macros["handKilled"] = "all" if all(x["killed"] for x in m) else str(sum(x["killed"] for x in m))
    macros["handFirstThree"] = str(sum((x.get("inputs_before_first_failure") or 99) <= 3 for x in m))
    lens = sorted(x["min_counterexample_length"] for x in m if x.get("min_counterexample_length") is not None
                  and x["mutant"] != "iir_wrong_gain")
    macros["handMedianLen"] = str(lens[len(lens) // 2]) if len(lens) % 2 else f"{(lens[len(lens)//2-1] + lens[len(lens)//2]) / 2:g}"


def error_profile(macros):
    rows = list(csv.DictReader((BENCH / "error_profile.csv").open()))
    torch = max(rows, key=lambda r: float(r["torch_eps"]))
    macros["torchMaxEps"] = f"{float(torch['torch_eps']):.0f}"
    macros["torchWorstN"] = torch["N"]
    macros["numpyMaxEps"] = f"{max(float(r['numpy_eps']) for r in rows):.1f}"


def main():
    (HERE / "tables").mkdir(exist_ok=True)
    macros = dict(CAMPAIGN)
    survey(macros)
    comparison(macros)
    validation(macros)
    hand_mutants(macros)
    hand_ablation(macros)
    error_profile(macros)
    text = ["% Generated by make_numbers.py; do not edit."]
    for k, v in sorted(macros.items()):
        text.append(rf"\newcommand{{\{k}}}{{{v}}}")
    (HERE / "numbers.tex").write_text("\n".join(text) + "\n")
    print("\n".join(text))


if __name__ == "__main__":
    main()

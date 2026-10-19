"""How do DSP libraries test themselves? A static survey of their test suites.

Usage: python benchmarks/survey_test_suites.py ROOT [--json out.json]

ROOT contains sparse checkouts of the libraries (see SUITES below for the
paths). benchmarks/survey_results.json was produced from the commits listed
in SOURCES. The script parses every test file with ``ast`` and counts, per
library:

* test functions (``def test_*``) and those that use Hypothesis (``@given``);
* numerical comparisons (``assert_allclose``, ``assert_array_almost_equal``,
  ``allclose``, ``torch.testing.assert_close``, ``assertEqual`` with a
  tolerance, ...) and how their tolerance is set: library default, or a
  hand-written ``rtol`` / ``atol`` / ``decimal`` / ``nulp`` value;
* the distinct hand-written tolerance values, converted to a relative or
  absolute magnitude (``decimal=d`` means 1.5 * 10**-d);
* tests that feed an empty signal, and tests that combine an initial state
  (``zi=``) with an empty signal, the case behind scipy/scipy#26406;
* tests that compare against a hard-coded array of at least 4 floating-point
  literals (expected values typed in, often produced by another tool).

The counts are heuristics on the syntax tree, not a semantic analysis; the
paper reports them as such.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import pathlib
from collections import Counter

SUITES = {
    "NumPy (numpy.fft)": ["numpy/numpy/fft/tests"],
    "SciPy (scipy.fft)": ["scipy/scipy/fft/tests", "scipy/scipy/fft/_pocketfft/tests"],
    "SciPy (scipy.signal)": ["scipy/scipy/signal/tests"],
    "librosa": ["librosa/tests"],
    "torchaudio (functional)": ["torchaudio/test/torchaudio_unittest/functional"],
}

# The checkouts behind benchmarks/survey_results.json (main branches on 14 October 2026 for SciPy
# and librosa, release tags otherwise).
SOURCES = {
    "numpy": ("https://github.com/numpy/numpy", "v2.5.3"),
    "scipy": ("https://github.com/scipy/scipy", "19a262b04562aac08b22412fe5c7b7c11ff6085a"),
    "librosa": ("https://github.com/librosa/librosa", "08e2c0a"),
    "torchaudio": ("https://github.com/pytorch/audio", "v2.11.0"),
}

CLOSE_FUNCS = {
    "assert_allclose", "assert_array_almost_equal", "assert_almost_equal",
    "assert_array_almost_equal_nulp", "assert_array_max_ulp", "allclose", "isclose",
    "xp_assert_close", "assert_close", "assert_approx_equal",
}
TOL_KEYWORDS = {"rtol", "atol", "decimal", "nulp", "maxulp", "significant", "rel", "abs"}
# Relative and absolute tolerances do not measure the same thing: an atol is in the units of the
# signal, so its magnitude depends on the test data. They are reported separately.
EPS64 = 2.220446049250313e-16
TOL_KINDS = {"relative": {"rtol", "rel", "nulp", "maxulp", "significant"},
             "absolute": {"atol", "abs", "decimal"}}


def _name(func):
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _const(node):
    """Numeric value of a literal such as 1e-7, -3 or 10**-6 (None otherwise)."""
    try:
        value = ast.literal_eval(node)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    except Exception:  # noqa: BLE001 - not a literal
        pass
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
        base, exp = _const(node.left), _const(node.right)
        if base is not None and exp is not None:
            return base ** exp
    return None


def _magnitude(keyword, value):
    if value is None:
        return None
    if keyword == "decimal":
        return 1.5 * 10 ** (-value)
    if keyword in ("nulp", "maxulp"):
        return value * 2.220446049250313e-16
    return value


def _is_empty_signal(node):
    """np.array([]), [], np.zeros(0), np.empty(0), torch.tensor([]) and similar."""
    if isinstance(node, ast.List) and not node.elts:
        return True
    if isinstance(node, ast.Call) and _name(node.func) in {"array", "asarray", "tensor", "zeros", "empty", "ones"}:
        if node.args:
            arg = node.args[0]
            if isinstance(arg, ast.List) and not arg.elts:
                return True
            if _const(arg) == 0:
                return True
            if isinstance(arg, ast.Tuple) and arg.elts and _const(arg.elts[0]) == 0:
                return True
    return False


def _float_literal_array(node, minimum=4):
    """Number of float literals (e.g. 0.7071, -1.25e-3) in a list/tuple display, if >= minimum."""
    if isinstance(node, (ast.List, ast.Tuple)):
        count = 0
        for e in node.elts:
            if isinstance(e, (ast.List, ast.Tuple)):
                count += _float_literal_array(e, 1)
            elif isinstance(e, ast.UnaryOp) and isinstance(e.operand, ast.Constant):
                count += isinstance(e.operand.value, float)
            elif isinstance(e, ast.Constant):
                count += isinstance(e.value, float)
        return count if count >= minimum else 0
    if isinstance(node, ast.Call) and _name(node.func) in {"array", "asarray", "tensor"} and node.args:
        return _float_literal_array(node.args[0], minimum)
    return 0


class TestVisitor(ast.NodeVisitor):
    def __init__(self):
        self.tests = []

    def visit_FunctionDef(self, node):
        if node.name.startswith("test"):
            self.tests.append(node)
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef


def analyse_test(fn):
    out = {"comparisons": 0, "explicit": 0, "tolerances": [], "empty": False, "zi": False,
           "hardcoded": False, "hypothesis": False}
    for dec in fn.decorator_list:
        target = dec.func if isinstance(dec, ast.Call) else dec
        if _name(target) == "given":
            out["hypothesis"] = True
    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            name = _name(node.func)
            kws = {k.arg: k.value for k in node.keywords if k.arg}
            is_close = name in CLOSE_FUNCS or (name == "assertEqual" and (TOL_KEYWORDS & set(kws)))
            if is_close:
                out["comparisons"] += 1
                explicit = False
                for key in TOL_KEYWORDS & set(kws):
                    explicit = True
                    mag = _magnitude(key, _const(kws[key]))
                    if mag is not None and mag > 0:
                        out["tolerances"].append((key, mag))
                # positional decimal of assert_array_almost_equal(x, y, decimal)
                if name in ("assert_array_almost_equal", "assert_almost_equal") and len(node.args) >= 3:
                    explicit = True
                    mag = _magnitude("decimal", _const(node.args[2]))
                    if mag is not None:
                        out["tolerances"].append(("decimal", mag))
                if name in ("assert_allclose", "allclose", "isclose") and len(node.args) >= 3:
                    explicit = True
                    mag = _const(node.args[2])
                    if mag is not None and mag > 0:
                        out["tolerances"].append(("rtol", mag))
                out["explicit"] += explicit
                for arg in node.args[1:2]:
                    if _float_literal_array(arg):
                        out["hardcoded"] = True
            if "zi" in kws:
                out["zi"] = True
            if any(_is_empty_signal(a) for a in node.args) or any(_is_empty_signal(v) for v in kws.values()):
                out["empty"] = True
        if isinstance(node, ast.Assign) and _float_literal_array(node.value, 4):
            out["hardcoded"] = True
    return out


def survey(root: pathlib.Path):
    results = {}
    for lib, paths in SUITES.items():
        files = []
        for p in paths:
            files += sorted((root / p).glob("*.py"))
        stats = Counter()
        tolerances = []
        for f in files:
            try:
                tree = ast.parse(f.read_text(encoding="utf-8", errors="replace"))
            except SyntaxError:
                continue
            visitor = TestVisitor()
            visitor.visit(tree)
            stats["files"] += 1
            for fn in visitor.tests:
                a = analyse_test(fn)
                stats["tests"] += 1
                stats["tests_hypothesis"] += a["hypothesis"]
                stats["comparisons"] += a["comparisons"]
                stats["explicit_tolerance"] += a["explicit"]
                stats["tests_with_comparison"] += a["comparisons"] > 0
                stats["tests_empty_input"] += a["empty"]
                stats["tests_zi"] += a["zi"]
                stats["tests_zi_and_empty"] += a["zi"] and a["empty"]
                stats["tests_hardcoded_expected"] += a["hardcoded"]
                tolerances += a["tolerances"]
        mags = [m for _, m in tolerances]
        by_kind = {}
        for kind, keys in TOL_KINDS.items():
            km = [m for k, m in tolerances if k in keys]
            by_kind[kind] = {
                "count": len(km),
                "distinct": len({round(m, 300) for m in km}),
                "min": min(km) if km else None,
                "max": max(km) if km else None,
                "decades": math.log10(max(km) / min(km)) if km else None,
                # below the double-precision epsilon, a relative tolerance asks for exact equality
                "below_eps": sum(m < EPS64 for m in km),
                "min_from_eps": min((m for m in km if m >= EPS64), default=None),
                "histogram": dict(Counter(f"1e{round(math.log10(m))}" for m in km).most_common()),
            }
        results[lib] = {
            **stats,
            "distinct_tolerance_values": len({(k, round(m, 300)) for k, m in tolerances}),
            "tolerance_min": min(mags) if mags else None,
            "tolerance_max": max(mags) if mags else None,
            "tolerance_decades": (math.log10(max(mags) / min(mags)) if mags else None),
            "tolerance_histogram": dict(Counter(f"1e{round(math.log10(m))}" for m in mags).most_common()),
            "tolerance_by_kind": by_kind,
        }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--json")
    args = parser.parse_args()
    res = survey(pathlib.Path(args.root))
    for lib, s in res.items():
        print(f"{lib}: {s['tests']} tests in {s['files']} files, {s['comparisons']} numerical comparisons "
              f"({s['explicit_tolerance']} with a hand-set tolerance, {s['distinct_tolerance_values']} distinct "
              f"values from {s['tolerance_min']:.1e} to {s['tolerance_max']:.1e}); "
              f"hypothesis {s['tests_hypothesis']}, empty input {s['tests_empty_input']}, zi {s['tests_zi']}, "
              f"zi+empty {s['tests_zi_and_empty']}, hard-coded expected {s['tests_hardcoded_expected']}")
    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()

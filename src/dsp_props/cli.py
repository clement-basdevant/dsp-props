"""Command line: find every declared operator in modules or files and verify it.

    dsp-props check mypkg.filters             # a module
    dsp-props check src/filters.py            # a file
    dsp-props check mypkg.filters:my_fft      # one operator
    dsp-props check mypkg --json report.json --examples 500
    dsp-props check mypkg --profile fast      # 30 inputs per relation
    dsp-props check mypkg --regressions tests # write tests replaying each counterexample
    dsp-props calibrate mypkg --tolerances tolerances.json            # measure, store
    dsp-props check mypkg --tolerances tolerances.json                # verify with it
    dsp-props calibrate mypkg --tolerances tolerances.json --compare  # precision regression?

Exit status: 0 if every operator passes, 1 if any relation fails (``calibrate``: if
calibration is refused, or ``--compare`` finds a precision regression), 2 on usage errors.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import pathlib
import sys
import time

from . import __version__
from .api import PROFILES, get_profile, verify


def _load_module(target):
    path = pathlib.Path(target)
    if path.suffix == ".py" and path.exists():
        name = "_dsp_props_target_" + path.stem
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot import {path}")
        module = importlib.util.module_from_spec(spec)
        sys.path.insert(0, str(path.resolve().parent))
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    return importlib.import_module(target)


def discover(target):
    """``[(qualified name, operator)]`` for a ``module``, ``file.py`` or ``module:attribute`` target."""
    module_part, _, attr = target.partition(":")
    module = _load_module(module_part)
    label = module_part if module_part.endswith(".py") else module.__name__
    if attr:
        obj = module
        for part in attr.split("."):
            obj = getattr(obj, part)
        if not hasattr(obj, "__dsp_props__"):
            raise SystemExit(f"dsp-props: {target} has no dsp-props declaration")
        return [(f"{label}:{attr}", obj)]
    found, seen = [], set()
    for name in sorted(vars(module)):
        obj = vars(module)[name]
        if name.startswith("_") or id(obj) in seen:
            continue
        if getattr(obj, "__dsp_props__", None) is not None:
            seen.add(id(obj))
            found.append((f"{label}:{name}", obj))
        elif isinstance(obj, dict):  # registries such as {"name": operator}
            for key, value in obj.items():
                if getattr(value, "__dsp_props__", None) is not None and id(value) not in seen:
                    seen.add(id(value))
                    found.append((f"{label}:{name}[{key!r}]", value))
    return found


def _write_regressions(op, report, directory):
    from .regression import write_regression_tests

    try:
        written = write_regression_tests(op, report, directory)
    except (ValueError, OSError) as exc:
        print(f"    no regression test written: {exc}")
        return
    for name in written.added:
        print(f"    regression test written: {written.path}::{name}")
    for relation, why in written.skipped:
        print(f"    no regression test for {relation}: {why}")


def _targets(names):
    sys.path.insert(0, ".")
    targets = []
    for t in names:
        try:
            targets += discover(t)
        except (ImportError, AttributeError, FileNotFoundError) as exc:
            print(f"dsp-props: cannot load {t}: {exc}", file=sys.stderr)
            return None
    if not targets:
        print("dsp-props: no declared operator found (use @dp.transform, @dp.lti_filter, ...)", file=sys.stderr)
        return None
    return targets


def _lengths(text):
    """``"1-256"``, ``"1-64,100,128"`` -> a sorted list of lengths."""
    out: set[int] = set()
    for part in text.split(","):
        first, _, last = part.strip().partition("-")
        out.update(range(int(first), int(last or first) + 1))
    if not out or min(out) < 1:
        raise argparse.ArgumentTypeError(f"invalid lengths {text!r}")
    return sorted(out)


def _calibrate(args):
    from .core.calibration import CalibrationError
    from .tolerances import NoReferenceError, ToleranceFile

    targets = _targets(args.targets)
    if targets is None:
        return 2
    try:
        file = ToleranceFile(args.tolerances)
    except (ValueError, OSError) as exc:
        print(f"dsp-props: {exc}", file=sys.stderr)
        return 2
    status, changed, results = 0, False, []
    for name, op in targets:
        t0 = time.perf_counter()
        try:
            if args.compare:
                cmp = file.compare(op, factor=args.factor)
                results.append(cmp.to_dict())
                status |= not cmp.ok
                print(f"{'ok  ' if cmp.ok else 'LOSS'} {cmp}  ({time.perf_counter() - t0:.1f}s)")
            else:
                tol = file.calibrate(op, lengths=args.lengths, trials=args.trials, margin=args.margin,
                                     floor=args.floor, seed=args.seed)
                changed = True
                print(f"ok   {name}: worst error {tol.worst:.3g} eps over {len(tol.ratios)} lengths "
                      f"({time.perf_counter() - t0:.1f}s)")
        except NoReferenceError as exc:
            print(f"skip {name}: {exc}")
        except CalibrationError as exc:
            status = 1
            print(f"FAIL {name}: {exc}")
        except KeyError as exc:
            print(f"skip {name}: {exc.args[0]}")
    if changed:
        print(f"\nwrote {file.save()}")
    if args.json:
        pathlib.Path(args.json).write_text(json.dumps({"dsp_props": __version__, "operators": results}, indent=2))
    return status


def _check(args):
    targets = _targets(args.targets)
    if targets is None:
        return 2

    reports, failed = [], 0
    for name, op in targets:
        t0 = time.perf_counter()
        report = verify(op, max_examples=args.examples, raise_on_failure=False, derandomize=args.derandomize,
                        profile=args.profile, database=args.database, tolerances=args.tolerances)
        dt = time.perf_counter() - t0
        failed += not report.ok
        reports.append((name, report))
        calibrated = ", calibrated tolerance" if report.tolerance.startswith("calibrated") else ""
        print(f"{'ok  ' if report.ok else 'FAIL'} {name}  ({len(report.checked)} relations, {dt:.1f}s{calibrated})")
        if not report.ok or args.verbose:
            print("\n".join("    " + line for line in str(report).splitlines()[1:]))
        if not report.ok and args.regressions:
            _write_regressions(op, report, args.regressions)

    print(f"\n{len(targets) - failed}/{len(targets)} operators pass")
    if args.json:
        data = {"dsp_props": __version__, "examples": args.examples or get_profile(args.profile).max_examples,
                "operators": [{**r.to_dict(), "target": n} for n, r in reports]}
        pathlib.Path(args.json).write_text(json.dumps(data, indent=2))
    if args.markdown:
        pathlib.Path(args.markdown).write_text("\n\n".join(r.to_markdown() for _, r in reports) + "\n")
    if args.html:
        from .report import render_html

        for n, r in reports:
            r.func_name = n
        pathlib.Path(args.html).write_text(render_html([r for _, r in reports]))
    return 1 if failed else 0


def main(argv=None):
    """Entry point of the ``dsp-props`` command; returns the exit status."""
    parser = argparse.ArgumentParser(prog="dsp-props", description="Property-based checks for DSP code.")
    parser.add_argument("--version", action="version", version=f"dsp-props {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check", help="verify every declared operator in modules or files")
    check.add_argument("targets", nargs="+", help="module, file.py, or module:attribute")
    check.add_argument("--examples", type=int, default=None,
                       help="inputs per relation (default: set by the profile, 200)")
    check.add_argument("--profile", choices=sorted(PROFILES), default=None,
                       help="cost profile: fast (30 inputs), default (200), nightly (2000); "
                            "default from DSP_PROPS_PROFILE")
    check.add_argument("--database", metavar="DIR",
                       help="Hypothesis example database: save failing inputs there, replay them first")
    check.add_argument("--regressions", metavar="DIR",
                       help="write test_regression_<operator>.py files replaying each minimal "
                            "counterexample (plain pytest, no Hypothesis) into DIR")
    check.add_argument("--tolerances", metavar="FILE",
                       help="tolerance file written by `dsp-props calibrate`: its calibrations replace the "
                            "default tolerance of the operators it names (default from DSP_PROPS_TOLERANCES)")
    check.add_argument("--derandomize", action="store_true", help="reproducible inputs (for CI)")
    check.add_argument("--json", help="write the full report as JSON")
    check.add_argument("--markdown", help="write the report as Markdown")
    check.add_argument("--html", help="write the report as a standalone HTML page")
    check.add_argument("-v", "--verbose", action="store_true", help="list relations of passing operators too")
    cal = sub.add_parser("calibrate", help="measure declared operators against an extended-precision reference "
                                           "and store the result in a tolerance file, or compare with it")
    cal.add_argument("targets", nargs="+", help="module, file.py, or module:attribute")
    cal.add_argument("--tolerances", metavar="FILE", default="tolerances.json",
                     help="the tolerance file (default tolerances.json); entries are added or replaced")
    cal.add_argument("--compare", action="store_true",
                     help="measure again on the same probes and report precision regressions; writes nothing")
    cal.add_argument("--factor", type=float, default=2.0,
                     help="with --compare: a length regresses when its error grows by more than this (2)")
    cal.add_argument("--lengths", type=_lengths, default=None,
                     help="lengths to measure, e.g. 1-256 or 1-64,100,128 (default 1 to min(256, max_n))")
    cal.add_argument("--trials", type=int, default=6, help="Gaussian-noise probes per length (6)")
    cal.add_argument("--margin", type=float, default=8.0, help="safety factor on the measured error (8)")
    cal.add_argument("--floor", type=float, default=0.0,
                     help="c of the analytic floor c eps log2 N kept under the measurement (0: none; "
                          "20: never tighter than the default transform model)")
    cal.add_argument("--seed", type=int, default=0, help="seed of the probe signals (0)")
    cal.add_argument("--json", help="with --compare: write the comparison as JSON")
    args = parser.parse_args(argv)
    if args.command == "calibrate":
        return _calibrate(args)
    return _check(args)


if __name__ == "__main__":
    sys.exit(main())

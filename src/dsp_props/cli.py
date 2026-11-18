"""Command line: find every declared operator in modules or files and verify it.

    dsp-props check mypkg.filters             # a module
    dsp-props check src/filters.py            # a file
    dsp-props check mypkg.filters:my_fft      # one operator
    dsp-props check mypkg --json report.json --examples 500
    dsp-props check mypkg --profile fast      # 30 inputs per relation
    dsp-props check mypkg --regressions tests # write tests replaying each counterexample

Exit status: 0 if every operator passes, 1 if any relation fails, 2 on usage errors.
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


def _check(args):
    sys.path.insert(0, ".")
    targets = []
    for t in args.targets:
        try:
            targets += discover(t)
        except (ImportError, AttributeError, FileNotFoundError) as exc:
            print(f"dsp-props: cannot load {t}: {exc}", file=sys.stderr)
            return 2
    if not targets:
        print("dsp-props: no declared operator found (use @dp.transform, @dp.lti_filter, ...)", file=sys.stderr)
        return 2

    reports, failed = [], 0
    for name, op in targets:
        t0 = time.perf_counter()
        report = verify(op, max_examples=args.examples, raise_on_failure=False, derandomize=args.derandomize,
                        profile=args.profile, database=args.database)
        dt = time.perf_counter() - t0
        failed += not report.ok
        reports.append((name, report))
        print(f"{'ok  ' if report.ok else 'FAIL'} {name}  ({len(report.checked)} relations, {dt:.1f}s)")
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
    check.add_argument("--derandomize", action="store_true", help="reproducible inputs (for CI)")
    check.add_argument("--json", help="write the full report as JSON")
    check.add_argument("--markdown", help="write the report as Markdown")
    check.add_argument("--html", help="write the report as a standalone HTML page")
    check.add_argument("-v", "--verbose", action="store_true", help="list relations of passing operators too")
    args = parser.parse_args(argv)
    return _check(args)


if __name__ == "__main__":
    sys.exit(main())

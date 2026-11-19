"""Lines of the target that each suite executes, on the unmutated copy (plan 3.2).

Usage: python benchmarks/automutation/lines_reached.py   (after run.py; same environment, plus coverage)

A mutant on a line that a suite never executes cannot be killed by that suite, whatever its
relations or assertions. Writes ``lines_reached.json``: for each suite, the executed lines of
``target/scipy_subset.py``. ``analyse.py`` uses it to tell "not reached" from "reached, not seen".
"""

from __future__ import annotations

import json
import pathlib
import shutil
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from run import SUITES  # noqa: E402


def main():
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        work = pathlib.Path(tmp)
        shutil.copytree(HERE / "target", work / "target")
        for suite in ("upstream", "dsp_default"):
            cmd, _ = SUITES[suite]
            data = work / f".coverage.{suite}"
            pytest_args = [a for a in cmd[3:] if a != "-x"]  # after "python -m pytest"
            env = {"PYTHONPATH": str(work / "target"), "PATH": "/usr/bin:/bin",
                   "COVERAGE_FILE": str(data)}
            subprocess.run([sys.executable, "-m", "coverage", "run", "--include", "*/scipy_subset.py",
                            "-m", "pytest", *pytest_args], cwd=work, env=env, capture_output=True, check=False)
            report = work / f"{suite}.json"
            subprocess.run([sys.executable, "-m", "coverage", "json", "-o", str(report)], cwd=work,
                           env=env, capture_output=True, check=True)
            files = json.loads(report.read_text())["files"]
            (info,) = files.values()
            out[suite] = sorted(info["executed_lines"])
            print(f"{suite}: {len(out[suite])} lines executed, {len(info['missing_lines'])} not", flush=True)
    (HERE / "lines_reached.json").write_text(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()

"""Verification reports: per-relation outcome, minimal counterexample, exports."""

from __future__ import annotations

import html
import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

SHOWN_SAMPLES = 8


def describe(value) -> Any:
    """A short, JSON-friendly description of a relation input (signals become summaries)."""
    if isinstance(value, np.ndarray):
        v = value.ravel()
        info = {"N": int(v.size), "dtype": str(v.dtype)}
        if v.size and np.all(np.isfinite(v)):
            info["max_abs"] = float(np.max(np.abs(v)))
            info["nonzero"] = int(np.count_nonzero(v))
        if v.size <= SHOWN_SAMPLES:
            info["samples"] = [_scalar(s) for s in v]
        return info
    if isinstance(value, (list, tuple)):
        return [describe(v) for v in value]
    if isinstance(value, (np.generic, int, float, complex, bool)) or value is None:
        return _scalar(value)
    return repr(value)


def _scalar(s):
    if isinstance(s, (complex, np.complexfloating)):
        return {"re": float(s.real), "im": float(s.imag)}
    if isinstance(s, (np.generic,)):
        return s.item()
    return s


def _short(desc) -> str:
    """One-line rendering of ``describe`` output."""
    if isinstance(desc, dict) and "N" in desc:
        if "samples" in desc:
            vals = ", ".join(_fmt(s) for s in desc["samples"])
            return f"[{vals}]"
        parts = [f"N={desc['N']}"]
        if "max_abs" in desc:
            parts.append(f"max|x|={desc['max_abs']:.3g}")
            parts.append(f"{desc['nonzero']} non-zero")
        return "signal(" + ", ".join(parts) + ")"
    if isinstance(desc, list):
        inner = ", ".join(_short(d) for d in desc)
        return f"({inner},)" if len(desc) == 1 else f"({inner})"
    return _fmt(desc)


def _fmt(s):
    if isinstance(s, dict) and "re" in s:
        return f"{s['re']:.4g}{s['im']:+.4g}j"
    if isinstance(s, float):
        return f"{s:.4g}"
    return str(s)


@dataclass
class Failure:
    """One violated relation: the error raised, the shrunk input and when it first failed."""

    relation: str
    error: BaseException
    example: Any = None  # the shrunk input, as passed to the relation's check
    examples_tried: int | None = None  # inputs generated before the first failure

    def __str__(self):
        return f"{self.relation}: {self.error}"

    @property
    def message(self):
        return str(self.error).splitlines()[0] if str(self.error) else type(self.error).__name__

    def to_dict(self):
        return {
            "relation": self.relation,
            "error": type(self.error).__name__,
            "message": self.message,
            "examples_tried": self.examples_tried,
            "counterexample": describe(self.example),
        }


@dataclass
class Report:
    """Outcome of ``verify`` for one operator.

    ``checked`` lists the relation names that were run, ``failures`` the
    ``Failure`` objects for those that were violated.
    ``str(report)`` gives the text summary; ``to_dict``, ``to_json``,
    ``to_markdown`` and ``to_html`` export it.
    """

    func_name: str
    checked: list
    failures: list
    kind: str = ""
    examples_per_relation: int = 0
    seconds: dict = field(default_factory=dict)

    @property
    def ok(self):
        return not self.failures

    def __str__(self):
        lines = [f"dsp-props report for {self.func_name}"]
        for name in self.checked:
            bad = [f for f in self.failures if f.relation == name]
            lines.append(f"  {'FAIL' if bad else 'ok  '} {name}")
            for f in bad:
                lines.append(f"       {f.message}")
                if f.example is not None:
                    lines.append(f"       minimal input: {_short(describe(f.example))}")
        return "\n".join(lines)

    def to_dict(self):
        return {
            "function": self.func_name,
            "kind": self.kind,
            "ok": self.ok,
            "examples_per_relation": self.examples_per_relation,
            "relations": [
                {
                    "name": name,
                    "ok": not any(f.relation == name for f in self.failures),
                    "seconds": round(self.seconds.get(name, 0.0), 3),
                }
                for name in self.checked
            ],
            "failures": [f.to_dict() for f in self.failures],
        }

    def to_json(self, **kwargs):
        return json.dumps(self.to_dict(), **{"indent": 2, **kwargs})

    def to_markdown(self):
        status = "passes every relation" if self.ok else f"fails {len(self.failures)} relation(s)"
        lines = [f"### `{self.func_name}` {status}", "", "| Relation | Result | Time (s) |", "| --- | --- | --- |"]
        for name in self.checked:
            bad = any(f.relation == name for f in self.failures)
            lines.append(f"| {name} | {'**FAIL**' if bad else 'ok'} | {self.seconds.get(name, 0.0):.2f} |")
        for f in self.failures:
            lines += ["", f"**{f.relation}**: {f.message}", ""]
            if f.example is not None:
                lines.append(f"Minimal input: `{_short(describe(f.example))}`")
            if f.examples_tried is not None:
                lines.append(f"First failure after {f.examples_tried} generated input(s).")
        return "\n".join(lines)

    def to_html(self):
        """A self-contained HTML fragment (one table, failures with their minimal input)."""
        return render_html([self])


_CSS = """
body{font:15px/1.5 system-ui,sans-serif;margin:24px auto;max-width:900px;padding:0 16px;color:#1d1d1f;background:#fff}
table{border-collapse:collapse;width:100%;margin:8px 0 24px}
td,th{border-bottom:1px solid #ddd;padding:4px 8px;text-align:left}
.ok{color:#1a7f37}.fail{color:#c62828;font-weight:600}
code{background:#f3f3f3;padding:1px 4px;border-radius:3px;word-break:break-all}
@media (prefers-color-scheme:dark){body{background:#161618;color:#e8e8ea}td,th{border-color:#333}
code{background:#26262a}.ok{color:#5fd07f}.fail{color:#ff7a7a}}
"""


def render_html(reports, title="dsp-props report"):
    """A standalone HTML page for one or more reports."""
    esc = html.escape
    passed = sum(r.ok for r in reports)
    parts = [f"<!doctype html><meta charset='utf-8'><meta name='viewport' content='width=device-width'>"
             f"<title>{esc(title)}</title><style>{_CSS}</style>",
             f"<h1>{esc(title)}</h1><p>{passed}/{len(reports)} operators pass.</p>"]
    for r in reports:
        state = "<span class='ok'>passes</span>" if r.ok else "<span class='fail'>fails</span>"
        parts.append(f"<h2><code>{esc(r.func_name)}</code> {state}</h2>")
        parts.append("<table><tr><th>Relation</th><th>Result</th><th>Time (s)</th></tr>")
        for name in r.checked:
            bad = any(f.relation == name for f in r.failures)
            cls, word = ("fail", "FAIL") if bad else ("ok", "ok")
            parts.append(f"<tr><td>{esc(name)}</td><td class='{cls}'>{word}</td>"
                         f"<td>{r.seconds.get(name, 0.0):.2f}</td></tr>")
        parts.append("</table>")
        for f in r.failures:
            parts.append(f"<p><b>{esc(f.relation)}</b>: {esc(f.message)}")
            if f.example is not None:
                parts.append(f"<br>Minimal input: <code>{esc(_short(describe(f.example)))}</code>")
            if f.examples_tried is not None:
                parts.append(f"<br>First failure after {f.examples_tried} generated input(s).")
            parts.append("</p>")
    return "\n".join(parts) + "\n"


class PropertyViolation(AssertionError):
    """Raised by ``verify`` when a relation fails; ``.report`` holds the full ``Report``."""

    def __init__(self, report: Report):
        super().__init__(str(report))
        self.report = report

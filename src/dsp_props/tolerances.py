"""Versioned calibration profiles: ``tolerances.json``.

Calibration (``dp.calibrate``) is the way to a tight tolerance, but a
tolerance measured in a notebook is lost, and one passed by hand to each
declaration is easy to forget. A tolerance file stores, per operator, the
error measured length by length against an extended-precision reference,
with the library versions it was measured with. Commit it next to the tests:

    dsp-props calibrate mypkg.filters --tolerances tolerances.json   # measure, write
    dsp-props check mypkg.filters --tolerances tolerances.json       # verify with it
    dsp-props calibrate mypkg.filters --tolerances tolerances.json --compare

``dp.verify(op, tolerances=...)``, ``dsp-props check --tolerances``, the pytest
option ``--dsp-props-tolerances`` and the ``DSP_PROPS_TOLERANCES`` environment
variable make the stored calibration the tolerance of every operator the
file names, in place of the default model (a tolerance given explicitly in
the declaration still wins).

``--compare`` (``ToleranceFile.compare``) is the precision-regression mode:
it measures the operator again on the same probe signals and reports the
lengths where it is now less accurate than recorded, for instance after
upgrading the library that implements it.

Which reference is used: the extended DFT (``extended_dft``) for
``kind="dft"`` and ``kind="rdft"`` transforms, and the ``long double``
direct-form recursion for an LTI filter declared with ``transfer=(b, a)``
where ``long double`` is extended. Other declarations need an explicit
``reference=`` (Python API only).
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import math
import os
import platform
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType

import numpy as np

from .core.calibration import CalibratedTolerance, CalibrationError, calibrate, extended_dft
from .core.doubledouble import DoubleDouble
from .core.tolerance import FFTTolerance

__all__ = ["ToleranceFile", "NoReferenceError", "PrecisionComparison", "operator_name", "TOLERANCES_ENV", "default_reference"]

FORMAT = 1
TOLERANCES_ENV = "DSP_PROPS_TOLERANCES"
DEFAULT_LENGTHS = range(1, 257)
REGRESSION_FACTOR = 2.0
_LIBRARIES = ("numpy", "scipy", "torch", "librosa", "torchaudio", "pyfftw", "mkl_fft")


# --- naming operators ---------------------------------------------------------------------------


def operator_name(operator, modules: Iterable[ModuleType] = ()) -> str:
    """The key of ``operator`` in a tolerance file: ``module:attribute`` or ``module:REGISTRY[key]``.

    The module is the one that defines the operator, or, if it does not name
    it, one of ``modules``. A file loaded by ``dsp-props check path/to/file.py``
    is named by its stem, as when pytest imports it.
    """
    from .regression import _find_in

    own = sys.modules.get(getattr(operator, "__module__", "") or "")
    candidates = ([own] if own is not None else []) + [m for m in modules if m is not own]
    for module in candidates:
        found = _find_in(module, operator)
        if found is not None:
            attribute, key = found
            label = attribute if key is None else f"{attribute}[{key}]"
            return f"{_module_label(module)}:{label}"
    qualname = getattr(operator, "__qualname__", None) or getattr(operator, "__name__", repr(operator))
    return f"{_module_label(own) if own is not None else '?'}:{qualname}"


def _module_label(module: ModuleType) -> str:
    name = module.__name__
    for prefix in ("_dsp_props_target_", "_dsp_props_regression_"):
        if name.startswith(prefix):
            file = getattr(module, "__file__", None)
            return Path(file).stem if file else name[len(prefix):]
    return name


# --- references ---------------------------------------------------------------------------------


class NoReferenceError(CalibrationError):
    """No built-in extended-precision reference for this declaration: pass ``reference=``."""


def _half(reference, n_out):
    def half(x):
        ref = reference(x)
        m = n_out(len(x))
        if isinstance(ref, DoubleDouble):
            return DoubleDouble(ref.hi[:m], ref.lo[:m])
        return np.asarray(ref)[:m]

    return half


def default_reference(spec) -> tuple[str, Callable]:
    """``(name, reference)``: the extended-precision reference for a declaration.

    Raises ``CalibrationError`` when there is none (pass ``reference=``).
    """
    from .api import LTISpec, TransformSpec

    if isinstance(spec, TransformSpec) and spec.kind == "dft":
        return "extended_dft", extended_dft
    if isinstance(spec, TransformSpec) and spec.kind == "rdft":
        return "extended_dft (first N//2+1 bins)", _half(extended_dft, lambda n: n // 2 + 1)
    if isinstance(spec, LTISpec) and spec.transfer is not None:
        if not np.finfo(np.longdouble).eps < np.finfo(np.float64).eps:
            raise NoReferenceError("long double is not extended on this platform, so longdouble_lfilter "
                                   "is no reference; calibrate on Linux x86-64 or pass reference=")
        from .core.iir import longdouble_lfilter

        b, a = spec.transfer
        return "longdouble_lfilter(b, a)", lambda x: longdouble_lfilter(b, a, x)
    raise NoReferenceError(f"no built-in extended-precision reference for this {type(spec).__name__} "
                           "declaration (only dft and rdft transforms, and LTI filters with transfer=); "
                           "pass reference=")


def _measured(spec) -> Callable:
    from .api import StreamingSpec

    if isinstance(spec, StreamingSpec):
        return lambda x: spec.make().process(x)  # one call on the whole signal
    return spec.func


# --- entries ------------------------------------------------------------------------------------


def _round_up(r: float, digits: int = 4) -> float:
    """``r`` to ``digits`` significant digits, rounded up (a stored tolerance must not shrink)."""
    if r <= 0 or not math.isfinite(r):
        return float(r)
    scale = 10 ** (digits - 1 - math.floor(math.log10(r)))
    return math.ceil(r * scale) / scale


def _environment(operator=None) -> dict[str, str]:
    """Versions that can change an operator's accuracy: Python, platform, the usual numerical
    libraries when imported, and the top-level package that defines the operator."""
    env = {"python": platform.python_version(), "platform": f"{platform.system()}-{platform.machine()}"}
    from . import __version__

    env["dsp_props"] = __version__
    own = (getattr(operator, "__module__", "") or "").split(".")[0]
    for name in (*_LIBRARIES, own):
        module = sys.modules.get(name)
        if module is not None and getattr(module, "__version__", None):
            env[name] = str(module.__version__)
    return env


def _tolerance_from(entry: dict) -> CalibratedTolerance:
    return CalibratedTolerance(ratios={int(n): float(r) for n, r in entry["ratios"].items()},
                               margin=float(entry["margin"]), chained=int(entry["chained"]),
                               base=FFTTolerance(c=float(entry["base_c"])))


def _lengths_list(lengths) -> list[int]:
    return sorted({int(n) for n in lengths})


# --- comparison ---------------------------------------------------------------------------------


@dataclass
class PrecisionComparison:
    """An operator measured again against its stored calibration.

    ``regressions`` and ``improvements`` list ``(N, recorded, now)`` ratios
    (in units of eps) for the lengths where the error grew, or shrank, by
    more than ``factor`` (ratios below 1 eps count as 1).
    """

    name: str
    factor: float
    recorded_with: dict
    measured_with: dict
    regressions: list = field(default_factory=list)
    improvements: list = field(default_factory=list)
    worst_before: float = 0.0
    worst_now: float = 0.0

    @property
    def ok(self) -> bool:
        return not self.regressions

    def __str__(self):
        head = (f"{self.name}: worst error {self.worst_before:.3g} eps recorded, {self.worst_now:.3g} eps now; "
                + (f"{len(self.regressions)} length(s) less accurate by more than x{self.factor:g}"
                   if self.regressions else "no precision regression"))
        lines = [head]
        changed = sorted(self.regressions, key=lambda r: -r[2] / max(r[1], 1.0))[:8]
        lines += [f"    N={n:<6} {before:10.3g} eps -> {now:10.3g} eps" for n, before, now in changed]
        if self.improvements:
            lines.append(f"    ({len(self.improvements)} length(s) more accurate by more than x{self.factor:g})")
        versions = _version_changes(self.recorded_with, self.measured_with)
        if versions:
            lines.append("    since the calibration: " + ", ".join(versions))
        return "\n".join(lines)

    def to_dict(self):
        return {"name": self.name, "ok": self.ok, "factor": self.factor,
                "worst_before": self.worst_before, "worst_now": self.worst_now,
                "regressions": [list(r) for r in self.regressions],
                "improvements": [list(r) for r in self.improvements],
                "recorded_with": self.recorded_with, "measured_with": self.measured_with}


def _version_changes(before: dict, after: dict) -> list[str]:
    keys = [k for k in after if k != "date"] + [k for k in before if k not in after and k != "date"]
    return [f"{k} {before.get(k, '-')} -> {after.get(k, '-')}" for k in keys if before.get(k) != after.get(k)]


# --- the file -----------------------------------------------------------------------------------


class ToleranceFile:
    """A ``tolerances.json`` file: calibrations of named operators, with their provenance.

    ``ToleranceFile(path)`` reads the file if it exists. ``calibrate`` measures
    an operator and stores the result (``save`` writes the file),
    ``tolerance_for`` returns the stored ``CalibratedTolerance`` of an
    operator (or ``None``), and ``compare`` measures it again and reports
    precision regressions.
    """

    def __init__(self, path):
        self.path = Path(path)
        self.entries: dict[str, dict] = {}
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if data.get("format") != FORMAT:
                raise ValueError(f"{self.path}: unsupported tolerance file format {data.get('format')!r}")
            self.entries = dict(data.get("operators", {}))

    def save(self) -> Path:
        data = {"format": FORMAT,
                "about": "Calibrated tolerances written by `dsp-props calibrate`: errors in units of eps "
                         "against an extended-precision reference, per length N. Commit this file.",
                "operators": {k: self.entries[k] for k in sorted(self.entries)}}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # one line per ratio: a recalibration shows as a short, readable diff
        self.path.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
        return self.path

    # names --------------------------------------------------------------------------------------

    def key(self, operator, modules: Iterable[ModuleType] = (), name: str | None = None) -> str | None:
        """The entry for ``operator``: by its full name, else by its attribute alone if that is unique."""
        if name is not None:
            return name if name in self.entries else None
        full = operator_name(operator, modules)
        if full in self.entries:
            return full
        short = full.split(":", 1)[1]
        matches = [k for k in self.entries if k.split(":", 1)[1] == short]
        return matches[0] if len(matches) == 1 else None

    def tolerance_for(self, operator, modules: Iterable[ModuleType] = (), name: str | None = None):
        """The stored ``CalibratedTolerance`` of ``operator``, or ``None`` if the file has none."""
        key = self.key(operator, modules, name)
        return None if key is None else _tolerance_from(self.entries[key])

    # measuring ----------------------------------------------------------------------------------

    def calibrate(self, operator, *, reference=None, lengths=None, trials=6, margin=8.0, seed=0,
                  floor: float = 0.0, modules: Iterable[ModuleType] = (), name: str | None = None,
                  max_plausible=1e6) -> CalibratedTolerance:
        """Measure ``operator`` and store its calibration (call ``save`` to write the file).

        ``reference`` defaults to the declaration's built-in reference
        (``default_reference``). ``lengths`` defaults to N = 1..256, capped
        by the declaration's ``max_n``; lengths the relations draw beyond
        them get the worst measured error. ``floor`` is the ``c`` of the
        analytic floor ``FFTTolerance(c=floor)`` kept under the measurement:
        0 (the default) is ``dp.calibrate(base=None)``, the tight tolerance;
        20 keeps the default transform model as a minimum. Raises
        ``CalibrationError`` for an error no rounding explains, and for a
        declaration in another precision than float64 (the probes are float64).
        """
        spec = _spec(operator)
        if np.dtype(spec.precision) != np.float64:
            raise CalibrationError("calibration measures float64 operators only (precision=float64)")
        ref_name = getattr(reference, "__name__", "custom reference")
        if reference is None:
            ref_name, reference = default_reference(spec)
        if lengths is None:
            lengths = range(getattr(spec, "min_n", 1), min(max(DEFAULT_LENGTHS), spec.max_n) + 1)
        lengths = _lengths_list(lengths)
        tol = calibrate(_measured(spec), reference=reference, lengths=lengths, trials=trials, margin=margin,
                        seed=seed, max_plausible=max_plausible, base=FFTTolerance(c=floor))
        key = name or operator_name(operator, modules)
        self.entries[key] = {
            "kind": type(spec).__name__.replace("Spec", "").lower(),
            "reference": ref_name,
            "lengths": _compress(lengths), "trials": trials, "seed": seed,
            "margin": margin, "chained": tol.chained, "base_c": floor,
            "worst": _round_up(tol.worst),
            "ratios": {str(n): _round_up(r) for n, r in sorted(tol.ratios.items())},
            "calibrated_with": {**_environment(operator), "date": datetime.date.today().isoformat()},
        }
        return _tolerance_from(self.entries[key])

    def compare(self, operator, *, reference=None, factor: float = REGRESSION_FACTOR,
                modules: Iterable[ModuleType] = (), name: str | None = None) -> PrecisionComparison:
        """Measure ``operator`` again, on the probe signals of its calibration, and compare.

        A length is a regression when the error now exceeds ``factor`` times
        the recorded one (both taken as at least 1 eps, so that a length
        computed exactly does not turn rounding noise into a regression).
        The probes are the same (same lengths, trials and seed), so an
        unchanged implementation measures the same errors, up to the
        platform. Raises ``KeyError`` if the file has no entry for ``operator``.
        """
        key = self.key(operator, modules, name)
        if key is None:
            raise KeyError(f"{self.path} has no calibration for {operator_name(operator, modules)}; "
                           "run `dsp-props calibrate` first")
        entry = self.entries[key]
        spec = _spec(operator)
        if reference is None:
            _, reference = default_reference(spec)
        now = calibrate(_measured(spec), reference=reference, lengths=_expand(entry["lengths"]),
                        trials=int(entry["trials"]), seed=int(entry["seed"]), max_plausible=math.inf,
                        base=None)
        result = PrecisionComparison(key, factor, entry.get("calibrated_with", {}),
                                     {**_environment(operator), "date": datetime.date.today().isoformat()},
                                     worst_before=float(entry["worst"]), worst_now=now.worst)
        for n_str, before in entry["ratios"].items():
            n = int(n_str)
            after = now.ratios.get(n)
            if after is None:
                continue
            if max(after, 1.0) > factor * max(before, 1.0):
                result.regressions.append((n, before, after))
            elif max(before, 1.0) > factor * max(after, 1.0):
                result.improvements.append((n, before, after))
        return result


def _spec(operator):
    spec = getattr(operator, "__dsp_props__", None)
    if spec is None:
        raise TypeError(f"{operator!r} has no dsp-props declaration")
    return spec


def _compress(lengths: list[int]) -> str:
    """``[1, 2, 3, 7]`` -> ``"1-3,7"``, the syntax of ``dsp-props calibrate --lengths``."""
    runs: list[list[int]] = []
    for n in lengths:
        if runs and runs[-1][1] == n - 1:
            runs[-1][1] = n
        else:
            runs.append([n, n])
    return ",".join(str(a) if a == b else f"{a}-{b}" for a, b in runs)


def _expand(lengths: str) -> list[int]:
    out: list[int] = []
    for part in str(lengths).split(","):
        first, _, last = part.strip().partition("-")
        out += range(int(first), int(last or first) + 1)
    return out


# --- using a file in verify ---------------------------------------------------------------------


def resolve(tolerances) -> ToleranceFile | None:
    """``tolerances`` as given to ``dp.verify`` (path, ``ToleranceFile``, ``None``: the environment variable)."""
    if isinstance(tolerances, ToleranceFile):
        return tolerances
    if tolerances is None:
        tolerances = os.environ.get(TOLERANCES_ENV) or None
        if tolerances is None:
            return None
    if isinstance(tolerances, str) and tolerances.lower() == "none":
        return None
    path = Path(tolerances)
    if not path.exists():
        raise FileNotFoundError(f"tolerance file {path} does not exist; write it with `dsp-props calibrate`")
    return _cached(path.resolve(), path.stat().st_mtime_ns)


_CACHE: dict[tuple, ToleranceFile] = {}


def _cached(path: Path, mtime: int) -> ToleranceFile:
    key = (path, mtime)
    if key not in _CACHE:
        _CACHE.clear()
        _CACHE[key] = ToleranceFile(path)
    return _CACHE[key]


def apply(spec, operator, tolerances, modules: Iterable[ModuleType] = ()):
    """``(spec, source)``: ``spec`` with the stored calibration of ``operator`` as its tolerance.

    The declaration is returned unchanged when there is no file, no entry
    for the operator, or when the declaration gives its own tolerance.
    """
    file = resolve(tolerances)
    if file is None or getattr(spec, "tolerance_declared", False):
        return spec, "declared" if getattr(spec, "tolerance_declared", False) else "default"
    tol = file.tolerance_for(operator, modules)
    if tol is None:
        return spec, "default"
    return dataclasses.replace(spec, tolerance=tol), f"calibrated ({file.path.name})"

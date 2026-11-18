"""Regression tests generated from minimal counterexamples.

When a relation fails, ``write_regression_tests`` writes (or extends)
``test_regression_<operator>.py``: one plain pytest function per failure,
which replays the shrunk input exactly, without Hypothesis::

    # inversibility: L2 error 1.000e+00 exceeds tolerance 1.776e-15 (N=1, ...)
    def test_inversibility_5d41402a():
        inputs = np.array([1.0])
        replay(OPERATOR, "inversibility", inputs)

The test fails as long as the bug is there and passes once it is fixed, at
the cost of one call to the relation. Inputs are written as exact literals
(every float round-trips bit for bit), and a case is only written if
replaying it from its source fails now, so a non-reproducible failure
(hidden state, randomness) gives no test.

The same file can be produced from ``dp.verify`` (``write_regression_tests``),
``dsp-props check --regressions DIR`` and the pytest plugin
(``--dsp-props-regressions``).
"""

from __future__ import annotations

import ast
import datetime
import hashlib
import importlib.util
import os
import re
import sys
import textwrap
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

__all__ = ["replay", "load_operator", "to_source", "write_regression_tests", "RegressionFile"]

_ARRAY_KINDS = "biufc"
_WIDTH = 96


# --- replay -------------------------------------------------------------------------------------


def _relation(spec, name):
    wanted = name.replace("-", " ")
    for rel in spec.relations:
        if rel.name == wanted:
            return rel
    raise ValueError(f"{name!r} is not a relation of this declaration: {[r.name for r in spec.relations]}")


def replay(operator, relation: str, inputs) -> None:
    """Check ``relation`` of the declared ``operator`` on ``inputs`` once, without Hypothesis.

    Raises the relation's ``AssertionError`` (or whatever the operator
    raises) if it is violated, and returns ``None`` otherwise. ``relation``
    is the relation's name, with spaces or hyphens (``"block invariance"``
    or ``"block-invariance"``).
    """
    spec = getattr(operator, "__dsp_props__", None)
    if spec is None:
        raise TypeError(f"{operator!r} has no dsp-props declaration")
    _relation(spec, relation).check(spec, inputs)


def load_operator(path, attribute: str, key: Any = None):
    """The operator ``attribute`` (or ``attribute[key]``) of the Python file ``path``.

    Used by generated tests whose operator lives in a file that is not
    importable by name (a script given to ``dsp-props check``, say). The
    file's directory is put on ``sys.path`` so that its own imports work,
    and the module is loaded once per process.
    """
    path = Path(path).resolve()
    name = "_dsp_props_regression_" + hashlib.sha1(str(path).encode()).hexdigest()[:12]
    module = sys.modules.get(name)
    if module is None:
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot import {path}")
        module = importlib.util.module_from_spec(spec)
        if str(path.parent) not in sys.path:
            sys.path.insert(0, str(path.parent))
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            del sys.modules[name]
            raise
    obj: Any = module
    for part in attribute.split("."):
        obj = getattr(obj, part)
    return obj if key is None else obj[key]


# --- exact source of an input -------------------------------------------------------------------


def _float_source(v: float) -> str:
    if np.isnan(v):
        return 'float("nan")'
    if np.isinf(v):
        return 'float("inf")' if v > 0 else '-float("inf")'
    return repr(float(v))  # shortest repr, exact for a double


def _element_source(v, dtype: np.dtype) -> str:
    """Source of one array element, read back exactly by ``np.array(..., dtype=dtype)``."""
    if dtype.kind == "b":
        return repr(bool(v))
    if dtype.kind in "iu":
        return repr(int(v))
    if dtype.kind == "c":
        return f"complex({_element_source(v.real, np.dtype(v.real.dtype))}, " \
               f"{_element_source(v.imag, np.dtype(v.imag.dtype))})"
    if dtype == np.float64 or not np.isfinite(v):
        return _float_source(float(v))
    short = str(v)  # numpy's shortest repr for this dtype (0.1 for float32(0.1))
    try:
        if np.array(float(short), dtype=dtype).tobytes() == np.array(v, dtype=dtype).tobytes():
            return short
    except ValueError:
        pass
    return _float_source(float(v))  # exact anyway: a float32 is a double


def _nested(a: np.ndarray) -> str:
    if a.ndim == 1:
        return "[" + ", ".join(_element_source(v, a.dtype) for v in a) + "]"
    return "[" + ", ".join(_nested(sub) for sub in a) + "]"


def to_source(value) -> str:
    """Python source that evaluates to ``value`` exactly (with ``import numpy as np``).

    Handles ``None``, booleans, Python and NumPy numbers, NumPy arrays of
    booleans, integers, real or complex floats, and tuples and lists of these,
    which covers the inputs of every built-in relation. Raises ``TypeError``
    for anything else.
    """
    if value is None or isinstance(value, (bool, str)):
        return repr(value)
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in _ARRAY_KINDS:
            raise TypeError(f"cannot write an array of dtype {value.dtype} as source")
        dtype = f"np.{value.dtype.name}"
        if value.tobytes() == np.zeros_like(value).tobytes():  # all +0 (or empty)
            shape = value.shape[0] if value.ndim == 1 else value.shape
            return f"np.zeros({shape}, dtype={dtype})"
        flat = value.ravel()
        if flat.size > 1 and flat.tobytes() == np.full_like(flat, flat[0]).tobytes():
            shape = value.shape[0] if value.ndim == 1 else value.shape
            return f"np.full({shape}, {_element_source(flat[0], value.dtype)}, dtype={dtype})"
        return f"np.array({_nested(value)}, dtype={dtype})"
    if isinstance(value, np.generic):
        if value.dtype.kind not in _ARRAY_KINDS:
            raise TypeError(f"cannot write a NumPy scalar of dtype {value.dtype} as source")
        return f"np.{value.dtype.name}({_element_source(value, value.dtype)})"
    if isinstance(value, int):
        return repr(value)
    if isinstance(value, float):
        return _float_source(value)
    if isinstance(value, complex):
        return f"complex({_float_source(value.real)}, {_float_source(value.imag)})"
    if isinstance(value, tuple):
        items = [to_source(v) for v in value]
        return "(" + items[0] + ",)" if len(items) == 1 else "(" + ", ".join(items) + ")"
    if isinstance(value, list):
        return "[" + ", ".join(to_source(v) for v in value) + "]"
    raise TypeError(f"cannot write {type(value).__name__} as source")


def _same(a, b) -> bool:
    """Exact equality of two inputs: same types, dtypes, shapes and bits (any NaN equals any NaN)."""
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        if not (isinstance(a, np.ndarray) and isinstance(b, np.ndarray)):
            return False
        if a.dtype != b.dtype or a.shape != b.shape:
            return False
        if a.dtype.kind in "fc":
            nan_a, nan_b = np.isnan(a), np.isnan(b)
            if not np.array_equal(nan_a, nan_b):
                return False
            a, b = np.where(nan_a, 0, a), np.where(nan_b, 0, b)
        return a.tobytes() == b.tobytes()
    if type(a) is not type(b):
        return False
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, (float, complex, np.generic)):
        return _same(np.asarray(a), np.asarray(b))
    return bool(a == b)


def _evaluate(source: str):
    return eval(source, {"np": np, "__builtins__": {"float": float, "complex": complex}})  # noqa: S307


# --- where the operator comes from -------------------------------------------------------------


def _find_in(module: ModuleType, operator) -> tuple[str, Any] | None:
    """``(attribute, key)`` under which ``module`` holds ``operator`` (key ``None`` if direct)."""
    names = vars(module)
    preferred = getattr(operator, "__name__", None)
    if preferred in names and names[preferred] is operator:
        return preferred, None
    for name, obj in names.items():
        if obj is operator and not name.startswith("__"):
            return name, None
    for name, obj in names.items():
        if isinstance(obj, dict) and not name.startswith("__"):
            for key, value in obj.items():
                if value is operator and _literal_key(key):
                    return name, key
    return None


def _literal_key(key) -> bool:
    try:
        return isinstance(key, (str, int)) and ast.literal_eval(repr(key)) == key
    except (ValueError, SyntaxError):
        return False


def _importable_by_name(module: ModuleType, directory: Path) -> bool:
    """Whether ``import <module>`` from a test file in ``directory`` reaches this very file."""
    name, file = module.__name__, getattr(module, "__file__", None)
    if file is None or name == "__main__" or name.startswith(("_dsp_props_", "_")):
        return False
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError):
        return False
    if spec is None or spec.origin is None or Path(spec.origin).resolve() != Path(file).resolve():
        return False
    if "." in name:
        return True  # a package module: importable wherever the package is
    # a top-level module is only importable from its own directory (pytest puts a test file's
    # directory on sys.path) or if it is installed
    here = Path(file).resolve().parent
    return here == directory.resolve() or here.name in ("site-packages", "dist-packages")


def _header_lines(operator, directory: Path, modules: Iterable[ModuleType]) -> tuple[str, str, list[str]]:
    """``(label, target, lines)``: the operator's name in its module (``FILTERS[gain]``), an
    identifier of the operator, and the source lines binding ``OPERATOR``."""
    candidates = []
    own = sys.modules.get(getattr(operator, "__module__", "") or "")
    if own is not None:
        candidates.append(own)
    candidates += [m for m in modules if m is not own]
    for module in candidates:
        found = _find_in(module, operator)
        if found is None:
            continue
        attribute, key = found
        index = "" if key is None else f"[{_key_source(key)}]"
        label = attribute if key is None else f"{attribute}[{key}]"
        if _importable_by_name(module, directory):
            target = f"{module.__name__}:{attribute}{index}"
            lines = [f"from {module.__name__} import {attribute}", "", f"OPERATOR = {attribute}{index}"]
            if key is None:
                lines = [f"from {module.__name__} import {attribute} as OPERATOR"]
            return label, target, lines
        file = getattr(module, "__file__", None)
        if file is None:
            continue
        file_path = Path(file).resolve()
        try:
            rel = Path(os.path.relpath(file_path, directory.resolve())).as_posix()
            path_src = f"Path(__file__).parent / {_quote(rel)}"
        except ValueError:  # another drive on Windows
            path_src = f"Path({_quote(str(file_path))})"
        target = f"{file_path.name}:{attribute}{index}"
        key_src = "" if key is None else f", {_key_source(key)}"
        return label, target, [f"OPERATOR = load_operator({path_src}, {_quote(attribute)}{key_src})"]
    raise ValueError(f"cannot locate {operator!r} in an importable module: declare it at module level "
                     "(or in a module-level dictionary) to get regression tests")


# --- writing the file ---------------------------------------------------------------------------


def _slug(text: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "_", text).strip("_").lower() or "operator"


def _case_name(relation: str, source: str) -> str:
    digest = hashlib.sha1(f"{relation}\0{source}".encode()).hexdigest()[:8]
    return f"test_{_slug(relation)}_{digest}"


def _case(name: str, relation: str, source: str, message: str) -> str:
    from . import __version__

    today = datetime.date.today().isoformat()
    comment = " ".join(message.split())
    if not comment.startswith(relation):
        comment = f"{relation}: {comment}"
    body = textwrap.fill(f"inputs = {source}", width=_WIDTH, initial_indent="    ",
                         subsequent_indent="        ", break_long_words=False, break_on_hyphens=False)
    lines = [textwrap.fill(comment, width=_WIDTH, initial_indent="# ",
                           subsequent_indent="# ", break_long_words=False, break_on_hyphens=False),
             f"# found by dsp-props {__version__} on {today}",
             f"def {name}():",
             body,
             f"    replay(OPERATOR, {_quote(relation)}, inputs)"]
    return "\n".join(lines) + "\n"


def _quote(text: str) -> str:
    return f'"{text}"' if not set(text) & set('"\\\n') else repr(text)


def _key_source(key) -> str:
    return _quote(key) if isinstance(key, str) else repr(key)


_TARGET = "# dsp-props operator: "


def _header(operator_name: str, target: str, binding: list[str]) -> str:
    uses_path = any("load_operator(" in line for line in binding)
    imports = ["from pathlib import Path", ""] if uses_path else []
    imports += ["import numpy as np", "",
                "from dsp_props.regression import " + ("load_operator, replay" if uses_path else "replay")]
    doc = (f'"""Regression tests for ``{operator_name}``, generated by dsp-props from minimal counterexamples.\n\n'
           "Each test replays, without Hypothesis, one input on which a relation failed: it fails\n"
           "while the bug is there and passes once it is fixed. New counterexamples are appended to\n"
           'this file; keep it under version control, and delete a test that no longer guards anything.\n"""')
    return "\n".join([doc, "", *imports, "", f"{_TARGET}{target}", *binding]) + "\n"


@dataclass
class RegressionFile:
    """What ``write_regression_tests`` did: the file, the tests added, and the failures skipped (with why)."""

    path: Path | None
    added: list[str] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)


def write_regression_tests(operator, report, directory, *, modules: Iterable[ModuleType] = ()) -> RegressionFile:
    """Write one regression test per failure of ``report`` into ``directory``.

    ``report`` comes from ``dp.verify(operator, ...)``. The tests go to
    ``test_regression_<operator>.py``, created if needed; a counterexample
    already in the file is not added twice. A failure is skipped (and listed
    in ``skipped``) if its input cannot be written as source or if replaying
    it does not fail. ``modules`` are other modules in which to look for the
    operator when its own module does not name it (the pytest plugin passes
    the test module that lists it). Nothing is written when there is nothing
    to add.
    """
    directory = Path(directory)
    result = RegressionFile(None)
    cases = []
    for f in getattr(report, "failures", []):
        if not getattr(f, "has_example", f.example is not None):
            result.skipped.append((f.relation, "no counterexample recorded"))
            continue
        try:
            source = to_source(f.example)
        except TypeError as exc:
            result.skipped.append((f.relation, str(exc)))
            continue
        replayed = _evaluate(source)
        if not _same(replayed, f.example):
            result.skipped.append((f.relation, "the input does not round-trip through its source"))
            continue
        try:
            replay(operator, f.relation, replayed)
        except Exception:  # noqa: BLE001 - any failure reproduces it
            cases.append((f.relation, source, f.message))
        else:
            result.skipped.append((f.relation, "replaying the counterexample passes (non-deterministic operator?)"))
    if not cases:
        return result

    label, target, binding = _header_lines(operator, directory, modules)
    path = directory / f"test_regression_{_slug(label)}.py"
    if path.exists() and f"{_TARGET}{target}\n" not in path.read_text(encoding="utf-8"):
        module = target.split(":")[0].removesuffix(".py")
        path = directory / f"test_regression_{_slug(module)}_{_slug(label)}.py"
    text = path.read_text(encoding="utf-8") if path.exists() else _header(label, target, binding)
    for relation, source, message in cases:
        name = _case_name(relation, source)
        if f"def {name}(" in text:
            continue
        text = text.rstrip("\n") + "\n\n\n" + _case(name, relation, source, message)
        result.added.append(name)
    result.path = path
    if result.added:
        directory.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return result

"""pytest plugin: every declared operator in a test module becomes one test per relation.

Installed with dsp-props (entry point ``pytest11``), so nothing needs configuring::

    # test_filters.py
    import dsp_props as dp
    import numpy as np

    @dp.transform(inverse=np.fft.ifft, parseval=True)
    def my_fft(x):
        return np.fft.fft(x)

``pytest`` then collects ``test_filters.py::my_fft[linearity]``,
``test_filters.py::my_fft[parseval]``, and so on; a failing relation reports
its minimal counterexample. What is collected, in test modules only:

- operators declared in the module itself (``@dp.transform``, ``dp.lti_filter(...)(f)``...),
  and dictionaries of them, such as ``{"butter2": op, "butter4": op}``;
- anything listed in a module-level ``dsp_props_operators`` (a list or a
  dictionary), for operators defined elsewhere.

Operators imported from another module are not collected, so that importing
a buggy operator to test that it fails does not add a failing test.

Options (command line, or ``ini`` keys without the leading dashes and with ``_``):

- ``--dsp-props-profile fast|default|nightly``: inputs per relation (30, 200,
  2000), default from ``DSP_PROPS_PROFILE``, else ``default``;
- ``--dsp-props-derandomize``: the same inputs on every run;
- ``--dsp-props-database DIR``: where failing inputs are saved and replayed
  first on the next run (default ``.hypothesis/examples`` in the root
  directory, Hypothesis's own; ``none`` disables it). Keep this directory
  between CI runs (a cache) to replay past counterexamples;
- ``-p no:dsp_props`` disables the plugin.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

REGISTRY_NAME = "dsp_props_operators"


def pytest_addoption(parser):
    group = parser.getgroup("dsp-props", "property-based checks of declared DSP operators")
    group.addoption("--dsp-props-profile", default=None, choices=["fast", "default", "nightly"],
                    help="inputs per relation: fast 30, default 200, nightly 2000")
    group.addoption("--dsp-props-derandomize", action="store_true", default=None,
                    help="draw the same inputs on every run")
    group.addoption("--dsp-props-database", default=None, metavar="DIR",
                    help="example database (default .hypothesis/examples; 'none' disables it)")
    parser.addini("dsp_props_profile", "dsp-props cost profile (fast, default, nightly)", default="")
    parser.addini("dsp_props_derandomize", "draw the same inputs on every run", type="bool", default=False)
    parser.addini("dsp_props_database", "dsp-props example database directory, or 'none'", default="")


def _declared(obj):
    return getattr(obj, "__dsp_props__", None) is not None


def _defined_in(obj, module):
    return getattr(obj, "__module__", None) == module.__name__


def _operators(name, obj, module):
    """``[(test name, operator)]`` that ``name = obj`` contributes to ``module``'s tests."""
    if name == REGISTRY_NAME:
        items = obj.items() if isinstance(obj, dict) else ((getattr(o, "__name__", str(i)), o)
                                                            for i, o in enumerate(obj))
        return [(str(k), o) for k, o in items if _declared(o)]
    if _declared(obj):
        return [(name, obj)] if _defined_in(obj, module) else []
    if isinstance(obj, dict) and obj:
        ops = [(f"{name}[{k}]", o) for k, o in obj.items() if _declared(o)]
        if ops and all(_defined_in(o, module) for _, o in ops):
            return ops
    return []


@pytest.hookimpl(tryfirst=True)
def pytest_pycollect_makeitem(collector, name, obj):
    if not isinstance(collector, pytest.Module) or name.startswith("_"):
        return None
    ops = _operators(name, obj, collector.module)
    if not ops:
        return None
    return [OperatorCollector.from_parent(collector, name=n, operator=o) for n, o in ops]


class OperatorCollector(pytest.Collector):
    """One declared operator; its children are its relations."""

    def __init__(self, *, operator, **kwargs):
        super().__init__(**kwargs)
        self.operator = operator

    def collect(self):
        spec = self.operator.__dsp_props__
        for rel in spec.relations:
            yield RelationItem.from_parent(self, name=rel.name.replace(" ", "-"), relation=rel.name)

    def reportinfo(self):
        return _location(self.operator, self.fspath, self.name)


def _location(operator, default_path, name):
    target = getattr(operator, "__wrapped__", operator)
    try:
        path = inspect.getsourcefile(target)
        line = inspect.getsourcelines(target)[1] - 1
        if path is None:
            raise TypeError(name)
        return Path(path), line, name
    except (OSError, TypeError):
        return default_path, None, name


class RelationFailure(AssertionError):
    """A violated relation, with the ``Failure`` that ``dp.verify`` recorded."""

    def __init__(self, failure):
        super().__init__(str(failure))
        self.failure = failure


class RelationItem(pytest.Item):
    """One relation of one declared operator, checked with ``dp.verify``."""

    def __init__(self, *, relation, **kwargs):
        super().__init__(**kwargs)
        self.relation = relation

    @property
    def collector(self) -> OperatorCollector:
        assert isinstance(self.parent, OperatorCollector)
        return self.parent

    @property
    def operator(self):
        return self.collector.operator

    def _settings(self):
        config = self.config
        profile = config.getoption("dsp_props_profile") or config.getini("dsp_props_profile") or None
        derandomize = config.getoption("dsp_props_derandomize")
        if derandomize is None:
            derandomize = config.getini("dsp_props_derandomize")
        database = config.getoption("dsp_props_database") or config.getini("dsp_props_database") or ""
        if database.lower() == "none":
            database = None
        else:
            database = Path(database) if database else Path(config.rootpath) / ".hypothesis" / "examples"
            if not database.is_absolute():
                database = Path(config.rootpath) / database
        return profile, bool(derandomize), database

    def runtest(self):
        from .api import verify

        profile, derandomize, database = self._settings()
        report = verify(self.operator, relations=[self.relation], raise_on_failure=False,
                        profile=profile, derandomize=derandomize, database=database)
        if report.failures:
            raise RelationFailure(report.failures[0])

    def repr_failure(self, excinfo, style=None):
        if not isinstance(excinfo.value, RelationFailure):
            return super().repr_failure(excinfo, style)
        from .report import _short, describe

        f = excinfo.value.failure
        lines = [f"{self.collector.name} violates {f.relation}", f"  {f.message}"]
        if f.example is not None:
            lines.append(f"  minimal input: {_short(describe(f.example))}")
            import numpy as np

            with np.printoptions(threshold=64, precision=17):
                lines.append(f"  full input:    {f.example!r}")
        if f.examples_tried is not None:
            lines.append(f"  first failure after {f.examples_tried} generated input(s)")
        if not isinstance(f.error, AssertionError):
            lines.append(f"  raised {type(f.error).__name__}: {f.error}")
        return "\n".join(lines)

    def reportinfo(self):
        path, line, _ = _location(self.operator, self.fspath, self.name)
        return path, line, f"{self.collector.name}[{self.name}]"

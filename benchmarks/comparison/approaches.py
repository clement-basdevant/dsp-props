"""The test approaches compared in the paper, applied to declared operators.

* ``sop``: the state-of-practice suite of the subject (see suites.py).
* ``dsp-props``: every relation of every declaration, default settings.
* ``dsp-props/uniform``: same relations and tolerances, but the domain
  generators are replaced by what a generic property-based test would draw:
  uniform lengths in [1, max_n] and arrays of floats uniform in [-1, 1].
* ``dsp-props/fixed-tol``: same relations and generators, but every
  tolerance model is replaced by a fixed relative tolerance (1e-7, the
  default of numpy.testing.assert_allclose), applied to the L2 error.
* ``generic-pbt``: both replacements at once, i.e. the same relations
  written with Hypothesis alone.
"""

from __future__ import annotations

import contextlib
import importlib
import time
from dataclasses import dataclass

import numpy as np
from hypothesis import strategies as st
from hypothesis.errors import HypothesisException
from hypothesis.extra import numpy as hnp

import dsp_props.relations.lti as rel_lti
import dsp_props.relations.multirate as rel_multirate
import dsp_props.relations.streaming as rel_streaming
import dsp_props.relations.transforms as rel_transforms
from dsp_props.api import _hypothesis_test
from dsp_props.core.tolerance import ToleranceModel

gen_signals = importlib.import_module("dsp_props.generators.signals")  # the package re-exports a function named signals

APPROACHES = ["sop", "dsp-props", "dsp-props/uniform", "dsp-props/fixed-tol", "generic-pbt"]
RANDOMISED = APPROACHES[1:]


@dataclass(frozen=True)
class FixedTolerance(ToleranceModel):
    """||error||_2 <= rtol * ||reference||_2 + atol * sqrt(N), whatever the operator."""

    rtol: float = 1e-7
    atol: float = 0.0

    def bound(self, n, dtype, ref_norm):
        return self.rtol * ref_norm + self.atol * np.sqrt(n)


def _uniform_lengths(max_n: int = 512):
    return st.integers(1, max_n)


def _uniform_signals(n: int, dtype=np.float64):
    return hnp.arrays(dtype, n, elements=st.floats(-1.0, 1.0, allow_nan=False, allow_infinity=False,
                                                   width=np.finfo(dtype).bits))


def _uniform_complex_signals(n: int, dtype=np.float64):
    """Real and imaginary parts uniform in [-1, 1] (``dtype`` is the real dtype, as in the generators)."""
    return st.tuples(_uniform_signals(n, dtype), _uniform_signals(n, dtype)).map(lambda ri: ri[0] + 1j * ri[1])


@st.composite
def _uniform_pairs(draw, max_n: int = 512, complex: bool = False, dtype=np.float64):
    n = draw(_uniform_lengths(max_n))
    kind = _uniform_complex_signals if complex else _uniform_signals
    return draw(kind(n, dtype)), draw(kind(n, dtype))


@contextlib.contextmanager
def uniform_generators():
    """Swap the domain generators for uniform ones in every module that uses them."""
    patches = [(gen_signals, "lengths", _uniform_lengths), (gen_signals, "signals", _uniform_signals)]
    for mod in (rel_lti, rel_multirate, rel_streaming, rel_transforms):
        for name, new in (("lengths", _uniform_lengths), ("signals", _uniform_signals),
                          ("complex_signals", _uniform_complex_signals), ("signal_pairs", _uniform_pairs)):
            if hasattr(mod, name):
                patches.append((mod, name, new))
    saved = [(m, n, getattr(m, n)) for m, n, _ in patches]
    try:
        for m, n, new in patches:
            setattr(m, n, new)
        yield
    finally:
        for m, n, old in saved:
            setattr(m, n, old)


@contextlib.contextmanager
def fixed_tolerance(specs, rtol=1e-7, atol=0.0):
    saved = [(s, s.tolerance) for s in specs]
    try:
        for s in specs:
            s.tolerance = FixedTolerance(rtol, atol)
        yield
    finally:
        for s, tol in saved:
            s.tolerance = tol


def _signal_length(example):
    if isinstance(example, np.ndarray) and example.ndim >= 1:
        return int(example.shape[0])
    if isinstance(example, (list, tuple)):
        for item in example:
            n = _signal_length(item)
            if n is not None:
                return n
    if isinstance(example, int) and not isinstance(example, bool):
        return example
    return None


def run_relations(decls: dict, *, seed: int, max_examples: int = 200, stop_at_first=True):
    """Run every relation of every declaration; return the outcome (stops at the first failure)."""
    t0 = time.perf_counter()
    failures = []
    examples = 0
    for dname, func in decls.items():
        spec = func.__dsp_props__
        for rel in spec.relations:
            run, trace = _hypothesis_test(rel, spec, max_examples, False, seed)
            try:
                run()
            except HypothesisException:
                raise
            except BaseException as exc:  # noqa: BLE001 - any failure kills the mutant
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
                failures.append({"declaration": dname, "relation": rel.name, "error": type(exc).__name__,
                                 "message": str(exc).splitlines()[0][:200] if str(exc) else "",
                                 "first_failure": trace["first_failure"],
                                 "counterexample_length": _signal_length(trace["example"])})
                examples += trace["calls"]
                if stop_at_first:
                    return _outcome(failures, examples, t0)
                continue
            examples += trace["calls"]
    return _outcome(failures, examples, t0)


def _outcome(failures, examples, t0):
    return {"killed": bool(failures), "failures": failures, "examples": examples,
            "seconds": round(time.perf_counter() - t0, 3)}


def run_sop(tests):
    t0 = time.perf_counter()
    failed = []
    for test in tests:
        try:
            test()
        except BaseException as exc:  # noqa: BLE001
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            failed.append({"test": test.__name__, "error": type(exc).__name__})
            break
    return {"killed": bool(failed), "failures": failed, "seconds": round(time.perf_counter() - t0, 3)}


def run_approach(approach, decls, sop_tests, seed, stop_at_first=True):
    if approach == "sop":
        return run_sop(sop_tests)
    specs = [f.__dsp_props__ for f in decls.values()]
    with contextlib.ExitStack() as stack:
        if approach in ("dsp-props/uniform", "generic-pbt"):
            stack.enter_context(uniform_generators())
        if approach in ("dsp-props/fixed-tol", "generic-pbt"):
            stack.enter_context(fixed_tolerance(specs))
        return run_relations(decls, seed=seed, stop_at_first=stop_at_first)

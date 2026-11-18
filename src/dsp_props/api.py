"""Declarative API: describe an operator once, check every applicable relation."""

from __future__ import annotations

import copy
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import seed as _seed
from hypothesis.errors import FlakyFailure, HypothesisException

from .core.tolerance import FFTTolerance, ToleranceModel
from .relations import LTI_RELATIONS, MULTIRATE_RELATIONS, STREAMING_RELATIONS, TRANSFORM_RELATIONS
from .report import Failure, PropertyViolation, Report

KINDS = ("dft", "rdft", "generic")


def _precision(dtype):
    """Validate a declared ``precision`` and return it as a real floating dtype."""
    dt = np.dtype(dtype)
    if dt.kind not in "fc":
        raise ValueError(f"precision must be a floating dtype such as np.float32 or np.float64, got {dt}")
    return np.finfo(np.result_type(dt, np.float32)).dtype  # float16 is checked as float32, like the bounds


def hermitian_extend(half, n):
    """Full length-``n`` DFT from the ``n // 2 + 1`` bins of a real-input DFT.

    The result is the complex dtype of ``half``'s precision (complex128 for
    float64 or complex128 bins, complex64 for single precision): the
    extension never lowers the precision the relations then check.
    """
    half = np.asarray(half)
    full = np.empty(n, dtype=np.result_type(half.dtype, np.complex64))
    m = n // 2 + 1
    full[:m] = half[:m]
    if n > m:
        full[m:] = np.conj(half[1 : n - m + 1][::-1])
    return full


@dataclass
class TransformSpec:
    """Declaration of a transform, stored on ``func.__dsp_props__`` by ``transform``.

    Fields mirror the arguments of ``transform``; ``relations`` lists the
    relations that apply to this declaration.
    """

    func: Callable
    inverse: Callable | None = None
    kind: str = "dft"
    linear: bool = True
    parseval: bool = False
    parseval_scale: Callable[[int], float] = lambda n: n
    real_input_hermitian: bool = False
    real_input_only: bool = False
    inverse_takes_length: bool = False
    tolerance: ToleranceModel = field(default_factory=FFTTolerance)
    max_n: int = 512
    precision: np.dtype = np.dtype(np.float64)
    tolerance_declared: bool = False  # a tolerance given in the declaration wins over a tolerance file
    float32: bool = False  # the operator also runs in single precision (float32 consistency relation)

    @property
    def relations(self):
        return [r for r in TRANSFORM_RELATIONS if r.applies(self)]

    @property
    def is_fourier(self):
        return self.kind in ("dft", "rdft")

    def full(self, x):
        """The transform of ``x`` as a full spectrum (rdft outputs are Hermitian-extended)."""
        y = np.asarray(self.func(x))
        if self.kind == "rdft":
            n = len(x)
            if y.shape != (n // 2 + 1,):
                raise AssertionError(f"rdft output has shape {y.shape}, expected ({n // 2 + 1},) for N={n}")
            return hermitian_extend(y, n)
        return y

    def invert(self, X, n):
        assert self.inverse is not None
        return np.asarray(self.inverse(X, n) if self.inverse_takes_length else self.inverse(X))


@dataclass
class StreamingSpec:
    """Declaration of a block processor, stored by ``streaming`` and ``stateful``.

    ``make`` is called with no argument to build a fresh processor whose
    ``process(block)`` method returns the output for that block.
    """

    make: Callable
    tolerance: ToleranceModel = field(default_factory=lambda: FFTTolerance(c=1000.0))
    max_n: int = 512
    precision: np.dtype = np.dtype(np.float64)
    tolerance_declared: bool = False  # a tolerance given in the declaration wins over a tolerance file
    float32: bool = False  # the operator also runs in single precision (float32 consistency relation)

    @property
    def relations(self):
        return [r for r in STREAMING_RELATIONS if r.applies(self)]


@dataclass
class LTISpec:
    """Declaration of an LTI filter, stored on ``func.__dsp_props__`` by ``lti_filter``.

    Fields mirror the arguments of ``lti_filter``.
    """

    func: Callable
    causal: bool = True
    transfer: tuple | None = None
    max_gain: float | None = None
    tolerance: ToleranceModel = field(default_factory=lambda: FFTTolerance(c=1000.0))
    max_n: int = 512
    precision: np.dtype = np.dtype(np.float64)
    tolerance_declared: bool = False  # a tolerance given in the declaration wins over a tolerance file
    float32: bool = False  # the operator also runs in single precision (float32 consistency relation)

    @property
    def relations(self):
        return [r for r in LTI_RELATIONS if r.applies(self)]


def lti_filter(*, causal=True, transfer=None, max_gain=None, tolerance=None, max_n=512,
               precision=np.float64, float32=False):
    """Declare a linear time-invariant filter applied with zero initial state.

    Checks linearity, time invariance and (if ``causal``) causality. With
    ``transfer=(b, a)`` it also checks the measured frequency response against
    B(z)/A(z), and the tolerance defaults to ``IIRTolerance(b, a)``, which
    follows the conditioning of a direct-form I recursion. With ``max_gain`` it checks
    that no input bounded by 1 gives an output above ``max_gain`` (a bounded
    gain on N <= ``max_n``, e.g. the l1 norm of the impulse response; not BIBO
    stability, since longer inputs are not tried).
    An unstable ``transfer`` (a pole on or outside the unit circle) raises
    ``ValueError``: no relation could fail against it.

    ``precision`` (every declaration takes it) is the dtype of the generated
    inputs, ``np.float64`` by default. The tolerance uses its epsilon, and an
    output less precise than it (float32 out of float64 in) is a failure.
    Declare ``precision=np.float32`` for an operator meant to run in single
    precision: it then gets float32 inputs within float32 range.

    ``float32=True`` (every declaration takes it, with the default float64
    precision) declares that the operator also runs in single precision: the
    ``float32 consistency`` relation checks that its output on a float32 input
    agrees with its output on the same values in float64, to float32
    accuracy, and is at least float32.
    """
    precision = _precision(precision)
    declared = tolerance is not None
    if transfer is not None:
        from .core.iir import check_stable

        check_stable(*transfer)
    if tolerance is None:
        if transfer is not None:
            from .core.iir import IIRTolerance

            tolerance = IIRTolerance(*transfer)
        else:
            tolerance = FFTTolerance(c=1000.0)

    def decorate(func):
        func.__dsp_props__ = LTISpec(func=func, causal=causal, transfer=transfer, max_gain=max_gain,
                                     tolerance=tolerance, max_n=max_n, precision=precision,
                                     tolerance_declared=declared, float32=float32)
        return func

    return decorate


@dataclass
class MultirateSpec:
    """Declaration of a rational resampler, stored on ``func.__dsp_props__`` by ``resampler``.

    Fields mirror the arguments of ``resampler``; ``length`` is the resolved
    output-length rule (a callable ``N -> length``, or ``None``).
    """

    func: Callable
    up: int = 1
    down: int = 1
    causal: bool = False
    length: Callable[[int], int] | None = None
    dc_gain: float | None = 1.0
    ripple: float = 1e-2
    stopband: float | None = 0.1
    tolerance: ToleranceModel = field(default_factory=lambda: FFTTolerance(c=1000.0))
    max_n: int = 512
    precision: np.dtype = np.dtype(np.float64)
    tolerance_declared: bool = False  # a tolerance given in the declaration wins over a tolerance file
    float32: bool = False  # the operator also runs in single precision (float32 consistency relation)

    @property
    def relations(self):
        return [r for r in MULTIRATE_RELATIONS if r.applies(self)]

    def expected_length(self, n):
        assert self.length is not None
        return self.length(n)


def resampler(up=1, down=1, *, causal=False, length="ceil", dc_gain=1.0, ripple=1e-2,
              stopband=0.1, tolerance=None, max_n=512, precision=np.float64, float32=False):
    """Declare a rational resampler (up by ``up``, down by ``down``), like ``resample_poly``.

    Checks linearity and periodic time invariance (an input delay of
    ``down`` samples gives an output delay of ``up`` samples). Also, unless
    disabled with ``None``: the output length (``"ceil"``: ceil(N up / down),
    ``"floor"``, or a callable), the DC gain within ``ripple`` away from the
    edges, and, when decimating, that a tone above the output Nyquist
    frequency comes out below ``stopband`` (relative amplitude). ``precision``
    and ``float32`` as for ``lti_filter``.
    """
    precision = _precision(precision)
    if up < 1 or down < 1:
        raise ValueError("up and down must be positive integers")
    rules = {
        "ceil": lambda n: -(-n * up // down),
        "floor": lambda n: n * up // down,
        None: None,
    }
    rule = rules[length] if (length is None or isinstance(length, str)) else length

    def decorate(func):
        func.__dsp_props__ = MultirateSpec(func=func, up=up, down=down, causal=causal, length=rule,
                                           dc_gain=dc_gain, ripple=ripple, stopband=stopband,
                                           tolerance=tolerance or FFTTolerance(c=1000.0), max_n=max_n,
                                           precision=precision, tolerance_declared=tolerance is not None,
                                           float32=float32)
        return func

    return decorate


def streaming(*, tolerance=None, max_n=512, precision=np.float64, float32=False):
    """Declare a block processor: a class (or factory) whose instances have ``process(block)``.

    ``dp.verify`` then checks that a split into blocks gives the same output
    as one call. The splits drawn mix arbitrary cut points (up to
    ``max(N, 16)`` blocks, empty and one-sample blocks included), blocks of
    one sample, and blocks of a fixed size (see ``generators.block_splits``).
    ``precision`` and ``float32`` as for ``lti_filter``.
    """
    precision = _precision(precision)

    def decorate(make):
        make.__dsp_props__ = StreamingSpec(make=make, tolerance=tolerance or FFTTolerance(c=1000.0),
                                           max_n=max_n, precision=precision,
                                           tolerance_declared=tolerance is not None, float32=float32)
        return make

    return decorate


class _StatefulProcessor:
    def __init__(self, func, init_state):
        self.func = func
        self.state = init_state() if callable(init_state) else np.array(init_state, copy=True)

    def process(self, block):
        y, self.state = self.func(block, self.state)
        return y


def stateful(func, init_state, **kwargs):
    """Wrap ``func(block, state) -> (out, state)`` (e.g. lfilter with zi) as a streaming operator."""

    def make():
        return _StatefulProcessor(func, init_state)

    # report and collect it under the wrapped function's name and module (pytest plugin)
    for attr in ("__module__", "__name__", "__qualname__"):
        if hasattr(func, attr):
            setattr(make, attr, getattr(func, attr))
    return streaming(**kwargs)(make)


def transform(inverse=None, *, kind="dft", linear=True, parseval=False, parseval_scale=None,
              real_input_hermitian=False, real_input_only=None, inverse_takes_length=None,
              tolerance=None, max_n=512, precision=np.float64, float32=False):
    """Declare a function as a transform; ``dp.verify`` then checks it.

    ``kind="dft"``: complex DFT (shift and convolution theorems apply).
    ``kind="rdft"``: real-input DFT returning ``N // 2 + 1`` bins, like
    ``numpy.fft.rfft``; its inverse is called as ``inverse(X, N)``.
    ``kind="generic"``: any linear transform (DCT, wavelets, STFT...): only
    linearity, Parseval and inversibility are checked.
    ``precision`` and ``float32`` as for ``lti_filter``.
    """
    precision = _precision(precision)
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    if real_input_only is None:
        real_input_only = kind == "rdft"
    if inverse_takes_length is None:
        inverse_takes_length = kind == "rdft"

    def decorate(func):
        func.__dsp_props__ = TransformSpec(
            func=func,
            inverse=inverse,
            kind=kind,
            linear=linear,
            parseval=parseval,
            parseval_scale=parseval_scale or (lambda n: n),
            real_input_hermitian=real_input_hermitian,
            real_input_only=real_input_only,
            inverse_takes_length=inverse_takes_length,
            tolerance=tolerance or FFTTolerance(),
            max_n=max_n,
            precision=precision,
            tolerance_declared=tolerance is not None,
            float32=float32,
        )
        return func

    return decorate


@dataclass(frozen=True)
class Profile:
    """A cost profile: how hard ``verify`` searches when no ``max_examples`` is given."""

    name: str
    max_examples: int
    description: str = ""


PROFILES = {
    "fast": Profile("fast", 30, "a few seconds per operator, for every commit"),
    "default": Profile("default", 200, "the default search"),
    "nightly": Profile("nightly", 2000, "bug hunting, for scheduled runs"),
}
PROFILE_ENV = "DSP_PROPS_PROFILE"
_active_profile: str | None = None


def set_profile(name: str | None) -> None:
    """Make ``name`` (``"fast"``, ``"default"`` or ``"nightly"``) the profile ``verify`` uses.

    ``None`` goes back to the ``DSP_PROPS_PROFILE`` environment variable, then ``"default"``.
    An explicit ``max_examples`` always wins over the profile.
    """
    global _active_profile
    if name is not None:
        get_profile(name)
    _active_profile = name


def get_profile(name: str | None = None) -> Profile:
    """The profile named ``name``, or the active one (``set_profile``, ``DSP_PROPS_PROFILE``, ``"default"``)."""
    import os

    name = name or _active_profile or os.environ.get(PROFILE_ENV) or "default"
    try:
        return PROFILES[name]
    except KeyError:
        raise ValueError(f"unknown dsp-props profile {name!r}; choose one of {sorted(PROFILES)}") from None


def _database(database):
    if database is None or not isinstance(database, (str, os.PathLike)):
        return database
    from hypothesis.database import DirectoryBasedExampleDatabase

    return DirectoryBasedExampleDatabase(os.fspath(database))


def _hypothesis_test(rel, spec, max_examples, derandomize, seed=None, database=None, key=b""):
    def run_inner(inputs):
        trace["calls"] += 1
        snapshot = copy.deepcopy(inputs)  # as drawn, even if the operator writes into its input
        try:
            rel.check(spec, inputs)
        except Exception:
            # Hypothesis replays the shrunk example last, so this ends as the minimal input.
            trace["example"] = snapshot
            trace["has_example"] = True
            if trace["first_failure"] is None:
                trace["first_failure"] = trace["calls"]
            raise

    # Every relation shares this source, so the example database needs its own key for each
    # (operator, relation) pair, or saved counterexamples would be replayed on the wrong test.
    run_inner._hypothesis_internal_add_digest = key + b"\0" + rel.name.encode()  # type: ignore[attr-defined]
    run = settings(max_examples=max_examples, deadline=None, database=database,
                   derandomize=derandomize, suppress_health_check=list(HealthCheck),
                   print_blob=False)(given(rel.inputs(spec))(run_inner))
    if seed is not None:
        run = _seed(seed)(run)
    trace: dict[str, Any] = {"calls": 0, "example": None, "has_example": False, "first_failure": None}
    return run, trace


def _operator_key(func):
    module = getattr(func, "__module__", "") or ""
    name = getattr(func, "__qualname__", None) or getattr(func, "__name__", None) or repr(func)
    return f"{module}:{name}".encode()


def verify(func, *, max_examples=None, raise_on_failure=True, derandomize=False, seed=None,
           profile=None, relations=None, database=None, tolerances=None) -> Report:
    """Run every relation declared for ``func`` and return a report.

    ``max_examples`` is the number of inputs per relation; without it, the
    cost profile decides (``profile``, else ``set_profile``, else the
    ``DSP_PROPS_PROFILE`` environment variable): ``"fast"`` 30, ``"default"``
    200, ``"nightly"`` 2000.
    ``seed`` makes the run reproducible (the same inputs are drawn), which
    the benchmarks use to repeat experiments over independent seeds.
    ``relations`` restricts the run to the relations with these names.
    ``database`` is a Hypothesis example database, or a directory for one:
    failing inputs are saved there and replayed first on the next run
    (``None``, the default, saves nothing).
    ``tolerances`` is a tolerance file (``tolerances.json``, written by
    ``dsp-props calibrate``, or a ``ToleranceFile``): if it holds a
    calibration of ``func``, that calibration replaces the default tolerance
    model (a tolerance given in the declaration still wins). ``None`` reads
    the path from the ``DSP_PROPS_TOLERANCES`` environment variable, if set;
    ``"none"`` ignores it.
    """
    spec = getattr(func, "__dsp_props__", None)
    if spec is None:
        raise TypeError(f"{func!r} has no dsp-props declaration (use @dp.transform, @dp.lti_filter, @dp.resampler or @dp.streaming)")
    from .tolerances import apply

    spec, tolerance_source = apply(spec, func, tolerances)
    if max_examples is None:
        max_examples = get_profile(profile).max_examples
    selected = spec.relations
    if relations is not None:
        wanted = [relations] if isinstance(relations, str) else list(relations)
        unknown = set(wanted) - {r.name for r in selected}
        if unknown:
            raise ValueError(f"{sorted(unknown)} not among the relations of this declaration: "
                             f"{[r.name for r in selected]}")
        selected = [r for r in selected if r.name in wanted]
    database = _database(database)
    key = _operator_key(func)
    failures, checked, seconds = [], [], {}
    for rel in selected:
        checked.append(rel.name)
        run, trace = _hypothesis_test(rel, spec, max_examples, derandomize, seed, database, key)
        t0 = time.perf_counter()
        error: BaseException | None = None
        try:
            run()
        except FlakyFailure as exc:
            # the operator gave different results for the same input: hidden state
            error = AssertionError(f"{rel.name}: non-deterministic operator (same input, different result): {exc}")
        except HypothesisException:
            raise  # misuse of the framework, not a property violation
        except Exception as exc:  # noqa: BLE001 - every failure is reported
            error = exc
        seconds[rel.name] = time.perf_counter() - t0
        if error is not None:
            failures.append(Failure(rel.name, error, trace["example"], trace["first_failure"],
                                    has_example=trace["has_example"]))
    report = Report(getattr(func, "__name__", repr(func)), checked, failures,
                    kind=type(spec).__name__.replace("Spec", "").lower(),
                    examples_per_relation=max_examples, seconds=seconds, tolerance=tolerance_source)
    if failures and raise_on_failure:
        raise PropertyViolation(report)
    return report

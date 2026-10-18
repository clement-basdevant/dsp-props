"""Declarative API: describe an operator once, check every applicable relation."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from hypothesis import HealthCheck, given, seed as _seed, settings
from hypothesis.errors import FlakyFailure, HypothesisException

from .core.tolerance import FFTTolerance, ToleranceModel
from .report import Failure, PropertyViolation, Report
from .relations import LTI_RELATIONS, MULTIRATE_RELATIONS, STREAMING_RELATIONS, TRANSFORM_RELATIONS


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
    inverse: Optional[Callable] = None
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
    transfer: Optional[tuple] = None
    max_gain: Optional[float] = None
    tolerance: ToleranceModel = field(default_factory=lambda: FFTTolerance(c=1000.0))
    max_n: int = 512
    precision: np.dtype = np.dtype(np.float64)

    @property
    def relations(self):
        return [r for r in LTI_RELATIONS if r.applies(self)]


def lti_filter(*, causal=True, transfer=None, max_gain=None, tolerance=None, max_n=512,
               precision=np.float64):
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
    """
    precision = _precision(precision)
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
                                     tolerance=tolerance, max_n=max_n, precision=precision)
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
    length: Optional[Callable[[int], int]] = None
    dc_gain: Optional[float] = 1.0
    ripple: float = 1e-2
    stopband: Optional[float] = 0.1
    tolerance: ToleranceModel = field(default_factory=lambda: FFTTolerance(c=1000.0))
    max_n: int = 512
    precision: np.dtype = np.dtype(np.float64)

    @property
    def relations(self):
        return [r for r in MULTIRATE_RELATIONS if r.applies(self)]

    def expected_length(self, n):
        return self.length(n)


def resampler(up=1, down=1, *, causal=False, length="ceil", dc_gain=1.0, ripple=1e-2,
              stopband=0.1, tolerance=None, max_n=512, precision=np.float64):
    """Declare a rational resampler (up by ``up``, down by ``down``), like ``resample_poly``.

    Checks linearity and periodic time invariance (an input delay of
    ``down`` samples gives an output delay of ``up`` samples). Also, unless
    disabled with ``None``: the output length (``"ceil"``: ceil(N up / down),
    ``"floor"``, or a callable), the DC gain within ``ripple`` away from the
    edges, and, when decimating, that a tone above the output Nyquist
    frequency comes out below ``stopband`` (relative amplitude). ``precision``
    as for ``lti_filter``.
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
                                           precision=precision)
        return func

    return decorate


def streaming(*, tolerance=None, max_n=512, precision=np.float64):
    """Declare a block processor: a class (or factory) whose instances have ``process(block)``.

    ``dp.verify`` then checks that a split into blocks gives the same output
    as one call. The splits drawn mix arbitrary cut points (up to
    ``max(N, 16)`` blocks, empty and one-sample blocks included), blocks of
    one sample, and blocks of a fixed size (see ``generators.block_splits``).
    ``precision`` as for ``lti_filter``.
    """
    precision = _precision(precision)

    def decorate(make):
        make.__dsp_props__ = StreamingSpec(make=make, tolerance=tolerance or FFTTolerance(c=1000.0),
                                           max_n=max_n, precision=precision)
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
    return streaming(**kwargs)(lambda: _StatefulProcessor(func, init_state))


def transform(inverse=None, *, kind="dft", linear=True, parseval=False, parseval_scale=None,
              real_input_hermitian=False, real_input_only=None, inverse_takes_length=None,
              tolerance=None, max_n=512, precision=np.float64):
    """Declare a function as a transform; ``dp.verify`` then checks it.

    ``kind="dft"``: complex DFT (shift and convolution theorems apply).
    ``kind="rdft"``: real-input DFT returning ``N // 2 + 1`` bins, like
    ``numpy.fft.rfft``; its inverse is called as ``inverse(X, N)``.
    ``kind="generic"``: any linear transform (DCT, wavelets, STFT...): only
    linearity, Parseval and inversibility are checked.
    ``precision`` as for ``lti_filter``.
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
        )
        return func

    return decorate


def _hypothesis_test(rel, spec, max_examples, derandomize, seed=None):
    @settings(max_examples=max_examples, deadline=None, database=None,
              derandomize=derandomize, suppress_health_check=list(HealthCheck),
              print_blob=False)
    @given(rel.inputs(spec))
    def run(inputs):
        trace["calls"] += 1
        try:
            rel.check(spec, inputs)
        except Exception:
            # Hypothesis replays the shrunk example last, so this ends as the minimal input.
            trace["example"] = inputs
            if trace["first_failure"] is None:
                trace["first_failure"] = trace["calls"]
            raise

    if seed is not None:
        run = _seed(seed)(run)
    trace = {"calls": 0, "example": None, "first_failure": None}
    return run, trace


def verify(func, *, max_examples=200, raise_on_failure=True, derandomize=False, seed=None) -> Report:
    """Run every relation declared for ``func`` and return a report.

    ``seed`` makes the run reproducible (the same inputs are drawn), which
    the benchmarks use to repeat experiments over independent seeds.
    """
    spec = getattr(func, "__dsp_props__", None)
    if spec is None:
        raise TypeError(f"{func!r} has no dsp-props declaration (use @dp.transform, @dp.lti_filter, @dp.resampler or @dp.streaming)")
    failures, checked, seconds = [], [], {}
    for rel in spec.relations:
        checked.append(rel.name)
        run, trace = _hypothesis_test(rel, spec, max_examples, derandomize, seed)
        t0 = time.perf_counter()
        error = None
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
            failures.append(Failure(rel.name, error, trace["example"], trace["first_failure"]))
    report = Report(getattr(func, "__name__", repr(func)), checked, failures,
                    kind=type(spec).__name__.replace("Spec", "").lower(),
                    examples_per_relation=max_examples, seconds=seconds)
    if failures and raise_on_failure:
        raise PropertyViolation(report)
    return report

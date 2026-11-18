"""Operator families: the parameters are drawn too, not only the input signal.

A declaration such as ``@dp.lti_filter(transfer=(b, a))`` fixes its
parameters, and a fault that shows only for other values (an order-0 filter,
a numerator shorter than the denominator, a rate of 1) is invisible. A
family declares how to build the operator from its parameters and a
strategy for them::

    @dp.family(dp.generators.iir_coefficients(max_order=6))
    def lfilter(b, a):
        return dp.lti_filter(transfer=(b, a))(lambda x: signal.lfilter(b, a, x))

``dp.verify(lfilter)`` then checks every relation of the declarations the
factory returns, on parameters and signals drawn together, and the
counterexample is ``(parameters, input)``, both shrunk.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
from hypothesis import HealthCheck, assume, find, settings
from hypothesis import strategies as st
from hypothesis.control import currently_in_test_context
from hypothesis.errors import NoSuchExample

from .relations.base import Relation

__all__ = ["family", "FamilySpec", "FamilyRelation"]


def _kind_relations(member) -> list:
    """Every relation of the member's kind (whether or not it applies to this member)."""
    from .api import LTISpec, MultirateSpec, StreamingSpec, TransformSpec
    from .relations import LTI_RELATIONS, MULTIRATE_RELATIONS, STREAMING_RELATIONS, TRANSFORM_RELATIONS

    kinds = {TransformSpec: TRANSFORM_RELATIONS, LTISpec: LTI_RELATIONS, MultirateSpec: MULTIRATE_RELATIONS,
             StreamingSpec: STREAMING_RELATIONS}
    return kinds.get(type(member), [])


def _all_relations():
    from .relations import LTI_RELATIONS, MULTIRATE_RELATIONS, STREAMING_RELATIONS, TRANSFORM_RELATIONS

    unique: list[Relation] = []
    for rel in [*TRANSFORM_RELATIONS, *LTI_RELATIONS, *MULTIRATE_RELATIONS, *STREAMING_RELATIONS]:
        if all(rel is not r for r in unique):
            unique.append(rel)
    return unique


def _applies(rel, member) -> bool:
    return any(rel is r for r in _kind_relations(member)) and rel.applies(member)


class FamilyRelation(Relation):
    """A relation of the family's members, checked on drawn ``(parameters, input)`` pairs."""

    def __init__(self, base: Relation):
        self.base = base
        self.name = base.name

    def applies(self, spec) -> bool:
        return True

    def inputs(self, spec):
        def for_parameters(p):
            member = spec.build(p)
            inner = self.base.inputs(member) if _applies(self.base, member) else st.just(None)
            return st.tuples(st.just(p), inner)

        return spec.params.flatmap(for_parameters)

    def check(self, spec, inputs):
        p, x = inputs
        member = spec.build(p)
        if not _applies(self.base, member):  # e.g. alias rejection needs a decimating pair
            if currently_in_test_context():
                assume(False)
            return  # replayed outside Hypothesis: nothing to check for this member
        self.base.check(member, x)


@dataclass
class FamilySpec:
    """Declaration of an operator family, stored on ``factory.__dsp_props__`` by ``family``."""

    factory: Callable
    params: st.SearchStrategy
    max_tries: int = 100
    tolerance_declared: bool = True  # each member has its own tolerance; tolerance files do not apply
    precision: np.dtype = field(default_factory=lambda: np.dtype(np.float64))

    def build(self, p):
        """The declaration of the member with parameters ``p`` (a tuple or dict is unpacked)."""
        if isinstance(p, tuple):
            op = self.factory(*p)
        elif isinstance(p, dict):
            op = self.factory(**p)
        else:
            op = self.factory(p)
        member = getattr(op, "__dsp_props__", None)
        if member is None or isinstance(member, FamilySpec):
            raise TypeError(f"the factory of a family must return a declared operator, got {op!r}")
        return member

    @cached_property
    def relations(self):
        """The relations that apply to at least one member (searched over ``max_tries`` parameters)."""
        quiet = settings(max_examples=self.max_tries, database=None, derandomize=True,
                         suppress_health_check=list(HealthCheck))
        found = []
        for rel in _all_relations():
            def applies_to(p, rel: Relation = rel) -> bool:
                return _applies(rel, self.build(p))

            try:
                find(self.params, applies_to, settings=quiet)
            except NoSuchExample:
                continue
            found.append(FamilyRelation(rel))
        if not found:
            raise TypeError("no relation applies to any member of this family")
        return found


def family(params: st.SearchStrategy, *, max_tries: int = 100):
    """Declare a family of operators: ``factory(*params)`` returns a declared operator.

    ``params`` is a Hypothesis strategy for the parameters (a tuple is
    unpacked into positional arguments, a dict into keyword arguments), such
    as ``dp.generators.iir_coefficients()`` or ``dp.generators.rate_pairs()``.
    ``dp.verify`` checks each relation that applies to some member on
    parameters and signals drawn together; a relation that does not apply to
    the drawn member (alias rejection for an upsampler) skips that draw.
    """

    def decorate(factory):
        factory.__dsp_props__ = FamilySpec(factory=factory, params=params, max_tries=max_tries)
        return factory

    return decorate

"""Relations for stateful, block-processing operators."""

from __future__ import annotations

import numpy as np
from hypothesis import strategies as st

from ..core.tolerance import assert_close, check_precision
from ..generators import block_splits, lengths, signals, split_blocks
from .base import Relation


@st.composite
def _signal_and_splits(draw, max_n, dtype=np.float64):
    n = draw(lengths(max_n))
    x = draw(signals(n, dtype))
    return x, draw(block_splits(n))


class BlockInvariance(Relation):
    """Processing in blocks of arbitrary sizes (including 0 and 1) equals one-shot processing."""

    name = "block invariance"

    def inputs(self, spec):
        return _signal_and_splits(spec.max_n, spec.precision)

    def check(self, spec, inputs):
        x, cuts = inputs
        whole = np.asarray(spec.make().process(x))
        proc = spec.make()
        parts = [np.asarray(proc.process(b)) for b in split_blocks(x, cuts)]
        streamed = np.concatenate(parts) if parts else np.zeros(0, x.dtype)
        check_precision(whole, spec.precision, f"{self.name} (one call)")
        assert_close(streamed, whole, spec.tolerance, what=f"{self.name} (cuts={cuts})", precision=spec.precision)


STREAMING_RELATIONS = [BlockInvariance()]

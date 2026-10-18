# Relations catalogue

A declaration fixes what an operator is; `dp.verify` then runs every relation
that applies to that declaration, each on up to `max_examples` generated inputs
(200 by default). The relations are proved in the companion article (and in Coq
in the repository's `proofs/` folder); they are stated here on finite sequences
of length N, as implementations compute them. Notation: T a transform, H a
filter, R a resampler, D_s a delay by s samples with zero filling, S_s a
circular shift.

All comparisons use the ℓ2 norm of the error against a [tolerance
model](tolerances.md), never an element-wise `rtol`.

## Transforms: `dp.transform`

| Relation (report name) | Checked property | Applies when |
| --- | --- | --- |
| `linearity` | T(αx + βy) = αTx + βTy, with complex α, β (real if the transform takes real input only) | `linear=True` (default) |
| `circular shift` | T(S_s x)[k] = e^{-2iπks/N} Tx[k], with s in [-1024, 1024] reduced modulo N in integers | `kind="dft"` or `"rdft"` |
| `parseval` | ‖Tx‖² = c(N) ‖x‖², c given by `parseval_scale` (default N, as for an unnormalised DFT) | `parseval=True` |
| `hermitian symmetry` | real x ⇒ Tx[k] = conj(Tx[-k mod N]) | `kind="dft"` and `real_input_hermitian=True` |
| `inversibility` | inverse(Tx) = x | `inverse` given |
| `convolution theorem` | inverse(Tx · Ty) = x ⊛ y, the circular convolution computed directly | `kind="dft"` or `"rdft"`, and `inverse` given |
| `real DC/Nyquist bins` | the DC bin, and the Nyquist bin for even N, are real | `kind="rdft"` |

For `kind="rdft"` the transform must return `N // 2 + 1` bins (any other shape
fails), and the relations act on the Hermitian extension of the output to N
bins. Its inverse is called as `inverse(X, N)`; set `inverse_takes_length` to
override this.

```python
import numpy as np
from scipy import fft
import dsp_props as dp

# Complex DFT
dp.transform(inverse=np.fft.ifft, parseval=True, real_input_hermitian=True)(np.fft.fft)

# Real-input DFT: the inverse receives the length N
dp.transform(inverse=np.fft.irfft, kind="rdft", parseval=True)(np.fft.rfft)

# Orthonormal DCT-II: generic linear transform, energy-preserving
@dp.transform(kind="generic", inverse=lambda X: fft.idct(X, norm="ortho"),
              parseval=True, parseval_scale=lambda n: 1.0)
def dct(x):
    return fft.dct(x, norm="ortho")
```

## LTI filters: `dp.lti_filter`

The function is applied from zero initial state. Tolerances are relative to the
norm of the *input*, which is what filter error models bound.

| Relation | Checked property | Applies when |
| --- | --- | --- |
| `linearity` | H(αx + βy) = αHx + βHy, real α, β | always |
| `time invariance` | H(D_s x) = D_s(Hx), s in [1, 32]; for a non-causal filter, the first s output samples are dropped before comparing | always |
| `causality` | changing x[n] for n ≥ k leaves y[n] unchanged for n < k | `causal=True` (default) |
| `transfer function` | the DFT of the impulse response, measured over L samples (enough to decay, but at most 2^16), equals B(e^{iθ})/A(e^{iθ}) on the DFT grid, within the tolerance plus a bound on the truncated tail | `transfer=(b, a)` (needs SciPy) |
| `bounded gain` | for each generated length N ≤ `max_n`, the worst-case input with \|x[n]\| ≤ 1 gives \|y[n]\| ≤ G (a bounded gain on N ≤ `max_n`, not BIBO stability) | `max_gain=G` |

## Rational resamplers: `dp.resampler(up, down)`

A resampler by L/M is not time-invariant but periodically time-varying.

| Relation | Checked property | Applies when |
| --- | --- | --- |
| `linearity` | as for filters | always |
| `output length` | len(Rx) = ⌈N L / M⌉ (`length="ceil"`), ⌊N L / M⌋ (`"floor"`), or a callable | `length` not `None` |
| `periodic time invariance` | R(D_{Ms} x) = D_{Ls}(Rx), s in [1, 8]; a non-causal resampler is compared after dropping the first Ls output samples | always |
| `DC gain` | a constant input gives the declared constant `dc_gain` within `ripple`, in the middle half of the output | `dc_gain` not `None` |
| `alias rejection` | a tone between 1.25 × the output Nyquist frequency and 0.95 × the input Nyquist frequency leaves an rms below `stopband / √2` in the middle half of the output | `stopband` not `None` and decimating enough for such a tone to exist (1.25 L/M < 0.9) |

```python
from scipy import signal

dp.resampler(2, 3)(lambda x: signal.resample_poly(x, 2, 3))
```

The DC-gain and alias-rejection relations draw signals of at least
64 × max(L, M) samples, so that the middle of the output is free of edge
transients.

## Streaming processors: `dp.streaming`, `dp.stateful`

| Relation | Checked property |
| --- | --- |
| `block invariance` | for a split drawn by `block_splits`, concatenating `process(block)` over the blocks of a fresh processor equals `process(x)` on another fresh processor. A split is one of three kinds: arbitrary cut points (up to max(N, 16) blocks, empty blocks and single samples included), blocks of one sample, or blocks of a fixed size B (the last one possibly shorter) |

## Generated signals

Every relation draws its signals from
[`dsp_props.generators`](api/generators.md):

| Family | Aimed at |
| --- | --- |
| impulse | indexing and edge errors |
| constant | DC gain, normalisation |
| off-grid sinusoid (any frequency in [0, 0.5]) | leakage, twiddle errors |
| near-Nyquist sinusoid ([0.45, 0.5]) | aliasing, sign alternation |
| two tones less than one bin apart | resolution, accumulated rounding |
| sparse (up to four impulses) | superposition |
| white noise | generic coverage |
| complex exponential (any frequency in [-0.5, 0.5]), x + iy from two real signals, complex noise | code that drops or mixes up the imaginary part; transforms only, unless declared `real_input_only` |

Amplitudes are drawn from {1, 1e-3, 1e3, 1e-100, 1e100}, to reach underflow
and overflow paths ({1, 1e-3, 1e3} for an operator declared with
`precision=np.float32`). Lengths mix uniform values up to `max_n` with
1, 2, 3, 5, 7, 8, 15, 16, 17, 31, 32, 33, 97, 127, 128, 129, 255, 256 and 257;
in runs of 200 examples with `max_n = 512`, about 15 % of the lengths drawn are
in that list.
The families are ordered so that shrinking converges to short impulses of unit
amplitude, which is why counterexamples are usually one to three samples long.

## Anatomy of a relation

Each relation is a subclass of [`Relation`](api/relations.md) with a `name`, an
`applies(spec)` precondition on the declaration, an `inputs(spec)` Hypothesis
strategy and a `check(spec, inputs)` method that raises `AssertionError` (or
`ToleranceError`, through `dsp_props.core.assert_close`) when the relation is
violated. A declaration object selects its relations from the lists
`TRANSFORM_RELATIONS`, `LTI_RELATIONS`, `MULTIRATE_RELATIONS` and
`STREAMING_RELATIONS` of `dsp_props.relations`, and the value passed to
`check` is what a report prints as the minimal input.

---
title: 'dsp-props: property-based testing for digital signal processing code'
tags:
  - Python
  - signal processing
  - property-based testing
  - metamorphic testing
  - floating point
authors:
  - name: Clément Basdevant
    orcid: 0009-0004-7496-2365
    affiliation: 1
affiliations:
  - name: Independent Researcher
    index: 1
date: 14 October 2026
bibliography: paper.bib
---

# Summary

Signal processing code is hard to test because its correct output is rarely
known: the result of a filter, a transform or a resampler on an arbitrary
input has no independent oracle [@barr2015oracle]. Tests therefore compare
against a handful of hand-computed cases, or against another implementation
that may share the same assumptions. `dsp-props` replaces those cases with
the mathematical properties every correct implementation must satisfy:
linearity, the shift and convolution theorems, Parseval's identity, time
invariance, causality, consistency with a declared transfer function,
periodic time invariance of a rational resampler, and invariance of a
streaming processor to the way its input is cut into blocks
[@oppenheim2010dtsp; @vaidyanathan1993multirate]. The user declares what an
operator is in one decorator; `dsp-props` draws adversarial signals with
Hypothesis [@maciver2019hypothesis], checks every applicable relation within
a floating-point tolerance that scales with the signal length, the input
precision and, for recursive filters, the filter's conditioning, and reports
the smallest input that breaks one.

```python
import numpy as np
import dsp_props as dp

@dp.transform(inverse=np.fft.ifft, parseval=True, real_input_hermitian=True)
def my_fft(x):
    return np.fft.fft(x)

dp.verify(my_fft)   # raises PropertyViolation with a minimal counterexample
```

# Statement of need

Property-based testing [@claessen2000quickcheck] and metamorphic testing
[@chen2018metamorphic] address the oracle problem in general, and Hypothesis
makes them practical in Python. Applying them to numerical signal processing
still requires three pieces of domain knowledge that every project
re-implements, usually partially:

1. **Which relations hold.** The catalogue depends on the operator: a complex
   DFT satisfies the circular shift theorem, a real-input DFT returns
   `N // 2 + 1` bins whose DC and Nyquist bins are real, an IIR filter applied
   from rest is causal and time-invariant, a resampler by `L/M` is only
   periodically time-varying, and a block processor must give the same output
   however its input is split: `dsp-props` draws arbitrary cuts (up to
   `max(N, 16)` blocks, empty and one-sample blocks included), one-sample
   blocks throughout, and fixed-size blocks.
2. **Which signals expose bugs.** Uniform random arrays rarely do. `dsp-props`
   generates impulses, constants, off-grid and near-Nyquist tones, close tone
   pairs, sparse signals and white noise, over pathological lengths (1, 2,
   primes, powers of two plus or minus one), amplitudes from 1e-100 to
   1e100, and block splits that include empty and one-sample blocks.
3. **How close is close enough.** A fixed `rtol` is either too loose to catch
   a single-precision intermediate or too tight for a large transform.
   `dsp-props` uses three kinds of tolerance. Transforms are checked against
   $c\,\varepsilon \log_2 N$ times the norm of the result, the shape of the
   worst-case $O(\varepsilon \log N)$ bound of the FFT
   [@higham2002accuracy, Theorem 24.2]; the root-mean-square error grows only
   as $O(\varepsilon \sqrt{\log N})$ [@schatzman1996accuracy]. The constant,
   $c = 20$ for transforms and $c = 1000$ for filters without a declared
   transfer function, resamplers and streaming processors, is an engineering
   choice that avoids false alarms, not a derived bound. Recursive filters
   use a forward-error bound for direct form I, derived in the manner of
   @higham2002accuracy, that grows with the order and the $\ell_1$ norms of
   the recursion's impulse responses. Finally, `dp.calibrate` gives a
   measured tolerance: it compares an implementation with an
   extended-precision reference, length by length, and refuses to calibrate
   an operator whose error is implausibly large instead of absorbing the bug
   into the tolerance.

`dsp-props` packages these three pieces for NumPy, SciPy, PyTorch and similar
libraries [@harris2020numpy; @virtanen2020scipy; @paszke2019pytorch], and for C
code through `ctypes` adapters. It targets library developers who want to
test new kernels, users who need to trust a third-party implementation, and
researchers who need executable specifications of DSP operators.

# Features

- Declarations for transforms (complex, real-input and generic linear
  transforms such as the DCT), LTI filters with an optional transfer function
  and declared gain bound, rational resamplers, and streaming or stateful
  block processors.
- Tolerances for transforms and recursive filters, and calibration against
  an extended-precision reference (`long double` where it is extended, a
  double-double DFT elsewhere). The expected precision is that of the
  declared input dtype, so an operator that silently returns a less precise
  dtype fails.
- Reports that keep the shrunk counterexample [@maciver2020reduction] and the
  number of inputs generated before the first failure, exported as text,
  JSON, Markdown or HTML.
- A command line, `dsp-props check module_or_file.py`, that discovers declared
  operators and exits non-zero on failure, for continuous integration.
- A mutation suite [@jia2011mutation] that measures how well the relations
  catch realistic bugs.

# Evaluation

The repository includes 31 deliberately faulty operators in six families
(complex FFT, real FFT, DCT-II, a fourth-order resonant IIR filter, a 2/3
resampler and streaming processors): a forgotten 1/N, a twiddle sign error,
single-precision intermediates, a filter redesigned mid-signal, a missing
interpolation gain, a weak anti-aliasing filter, state reset between blocks
and others. All 31 are detected with the default 200 inputs per relation. For
27 of the 30 mutants whose counterexample is a signal, the shrunk signal has
at most three samples (median: two), and 27 of the 31 fail within the first
three generated inputs. The only documented blind spot is a filter that is
wrong but still linear, time-invariant and causal; it is caught as soon as
its transfer function is declared.

Applied to 46 targets in SciPy, NumPy, PyTorch, librosa [@mcfee2015librosa]
and torchaudio, the relations found that `scipy.signal.lfilter` returns an
uninitialised final state for a recursive filter, and raises for an FIR
filter, when given an empty block with an initial state, so recursive block
processing silently diverges after an empty block; `sosfilt` raises on the
same input (reported together as scipy/scipy#26406, not yet
confirmed by a maintainer). The `lfilter` part is an independent rediscovery
of a known issue (scipy/scipy#22571), already fixed in the development
version after SciPy 1.18.1; its FIR case propagates to librosa's streaming
pre-emphasis. The `sosfilt` part is new, but it is an exception on any input
that is empty along the filtered axis, not a silent divergence. The relations
also flagged two documented behaviours that break a natural expectation:
`tf2sos` drops pure delays, and torchaudio's `deemphasis` clips its output to
[-1, 1] by default, so it does not invert `preemphasis`. An error profile
over lengths 2 to 1200 shows that PyTorch's CPU FFT reaches 195 ε for some
lengths with large prime factors (N = 596 = 4 × 149), against at most about
4 ε for NumPy and SciPy, which illustrates why tolerances must be calibrated
per implementation rather than fixed.

# Related work

Hypothesis provides the generation and shrinking engine; `dsp-props` is a
domain layer on top of it. Differential testing of numerical libraries
[@vanover2020fpdiff] compares implementations with each other and needs at
least two of them; metamorphic relations need none. Recent work uses language
models to propose properties across the Python ecosystem
[@maaz2025agentic]; `dsp-props` instead provides a curated, verified
catalogue for one domain, with explicit tolerances that can be replaced by
calibrated ones.

# Acknowledgements

I thank Nicolas Boganski for discussions during the development of `dsp-props`.

# AI usage disclosure

Most of the code, tests and this draft were written with the help of an AI
coding assistant (Claude, Anthropic), under the direction of the author, who
chose the scope and design, reviewed the results and reported the upstream
bugs. Every number in this paper is produced by scripts in the repository
(`mutants/run_mutants.py`, `benchmarks/campaign.py`,
`benchmarks/error_profile.py`).

# References

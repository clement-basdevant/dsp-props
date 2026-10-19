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
input has no independent oracle [@barr2015oracle]. Tests compare against a few
hand-computed cases, or another implementation sharing the same assumptions. `dsp-props` replaces those cases with
the mathematical properties every correct implementation must satisfy:
linearity, the shift and convolution theorems, Parseval's identity, time
invariance, causality, consistency with a declared transfer function,
periodic time invariance of a rational resampler, and invariance of a
streaming processor to the way its input is cut into blocks
[@oppenheim2010dtsp; @vaidyanathan1993multirate]. The user declares what an
operator is in one decorator; `dsp-props` draws adversarial signals with
Hypothesis [@maciver2019hypothesis], checks every applicable relation within
a floating-point tolerance, and reports the smallest input that breaks one.

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
[@chen2018metamorphic] address the oracle problem, and Hypothesis makes them
practical in Python. Applying them to signal processing still requires three
pieces of domain knowledge that projects re-implement, usually partially:

1. **Which relations hold.** The catalogue depends on the operator: a complex
   DFT satisfies the circular shift theorem, a real-input DFT returns
   `N // 2 + 1` bins whose DC and Nyquist bins are real, an IIR filter applied
   from rest is causal and time-invariant, a resampler by `L/M` is only
   periodically time-varying, and a block processor must not depend on how its
   input is split.
2. **Which signals expose bugs.** Uniform random arrays rarely do. `dsp-props`
   generates impulses, constants, off-grid and near-Nyquist tones, close tone
   pairs, sparse signals and white noise, over pathological lengths (1, 2,
   primes, powers of two plus or minus one), amplitudes from 1e-100 to
   1e100, and block splits: arbitrary cuts (up to `max(N, 16)` blocks,
   empty and one-sample blocks included), one-sample blocks, fixed-size
   blocks.
3. **How close is close enough.** A fixed `rtol` is either too loose to catch
   a single-precision intermediate or too tight for a large transform.
   The default tolerance, $c\,\varepsilon \log_2 N$ relative to the result,
   has the shape of the worst-case FFT bound [@higham2002accuracy, Theorem
   24.2]; the root-mean-square error grows only as
   $O(\varepsilon \sqrt{\log N})$ [@schatzman1996accuracy]. The constant
   ($c = 20$ for transforms, $c = 1000$ for filters without a declared
   transfer function, resamplers and streaming processors) is an engineering
   choice against false alarms, not a derived bound. Only recursive filters
   get a derived forward-error bound (direct form I), growing with the order
   and the $\ell_1$ norms of the recursion's impulse responses.
   `dp.calibrate` instead measures an implementation against an
   extended-precision reference, length by length, and refuses errors too
   large to be rounding.

`dsp-props` packages these three pieces for NumPy, SciPy, PyTorch and similar
libraries [@harris2020numpy; @virtanen2020scipy; @paszke2019pytorch], and for C
code through `ctypes` adapters. It targets library developers testing new
kernels, users who must trust a third-party implementation, and researchers
who need executable specifications of DSP operators.

# Features

- Declarations for transforms (complex, real-input and generic linear
  transforms such as the DCT), LTI filters with an optional transfer function
  and declared gain bound, rational resamplers, and streaming or stateful
  block processors.
- Calibration against `long double`, or a double-double DFT where
  `long double` is not extended. The expected precision is the declared
  input dtype, so an operator that silently loses precision fails.
- Reports with the shrunk counterexample [@maciver2020reduction] and the
  number of inputs before the first failure, as text, JSON, Markdown or HTML.
- A command line, `dsp-props check module_or_file.py`, that discovers declared
  operators and exits non-zero on failure, for continuous integration.
- A mutation suite [@jia2011mutation] that measures how well the relations
  catch realistic bugs.
- Coq proofs of the relations in exact arithmetic (`proofs/`), and
  documentation at <https://dsp-props.readthedocs.io/>.

# Evaluation

The repository includes 31 deliberately faulty operators in six families
(complex FFT, real FFT, DCT-II, a fourth-order resonant IIR filter, a 2/3
resampler and streaming processors), such as a forgotten 1/N,
single-precision intermediates or state reset between blocks. All 31 are detected with the default 200 inputs per relation.
With Hypothesis seed 0, 26 of the 30 signal counterexamples shrink to at
most three samples (median: two) and 28 of the 31 mutants fail within the
first three inputs (26–27 and 25–28 over seeds 0 to 3). The only documented blind spot
is a filter that is wrong but still linear, time-invariant and causal; it is
caught as soon as its transfer function is declared.

Applied to 46 targets in SciPy, NumPy, PyTorch, librosa [@mcfee2015librosa]
and torchaudio, the relations found that, given an empty block and an initial state,
`scipy.signal.lfilter` returns a wrong final state (recursive filters) or
raises (FIR filters, which breaks librosa's streaming pre-emphasis), and
`sosfilt` raises (scipy/scipy#26406, not yet confirmed by a maintainer; proposed
fix scipy/scipy#26415). The
`lfilter` part rediscovers scipy/scipy#22571, already fixed in the
development version after SciPy 1.18.1; the `sosfilt` part is new, but is an
exception rather than a silent divergence. They also flagged two documented
behaviours: `tf2sos` drops pure delays (scipy/scipy#26417), and torchaudio's
`deemphasis` clips to [-1, 1] by default, so it does not invert `preemphasis`
(pytorch/audio#4238). An error profile
over lengths 2 to 1200 shows PyTorch's CPU FFT reaching about 200 ε at some
lengths with a large prime factor (195 ε at N = 596 in the committed profile,
214 ε at N = 1048 on another x86-64 machine), against at most about 4 ε for
NumPy and SciPy: tolerances must be calibrated per implementation and
platform rather than fixed.

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

# Automatic mutation of SciPy code

Can generated faults in real DSP code be caught by the relations, and how do they compare with
the tests the project already has? Our own mutation suites cannot answer this, because we wrote
their faults. Here the faults come from a mutation tool, the code comes from SciPy, and the
comparison is SciPy's own tests.

**Result.** We checked <!-- r:automutation.mutants -->701<!-- /r --> mutants of four `scipy.signal`
functions:

| Suite | Mutants killed | Share |
| --- | --- | --- |
| SciPy's tests | <!-- r:automutation.upstream -->550<!-- /r --> | <!-- r:automutation.upstream_rate -->78 %<!-- /r --> |
| dsp-props | <!-- r:automutation.dsp_default -->398<!-- /r --> | <!-- r:automutation.dsp_default_rate -->57 %<!-- /r --> |
| either | <!-- r:automutation.union -->575<!-- /r --> | <!-- r:automutation.union_rate -->82 %<!-- /r --> |

SciPy's tests are stronger overall, and the two sets are complementary rather than nested:

- <!-- r:automutation.upstream_only -->177<!-- /r --> mutants are killed by SciPy's tests only.
  <!-- r:automutation.upstream_only_not_reached -->113<!-- /r --> of them sit on lines that the
  declarations never execute: argument checks, `axis` and n-D handling, and options that the
  declarations do not draw (`window`, `padtype`, `domain`, `t`).
- <!-- r:automutation.dsp_only -->25<!-- /r --> mutants are killed by dsp-props only.
  <!-- r:automutation.hilbert_dsp_only -->23<!-- /r --> of them are in `hilbert`'s branch for an
  odd number of points.
- <!-- r:automutation.neither -->126<!-- /r --> mutants survive both. Some are equivalent mutants;
  their number was not estimated.

Some SciPy defects come from code that no test executes. A property-based declaration reaches the
code its declaration covers, and no more: its strength is on the paths it draws.

## Protocol

- **Target.** `benchmarks/automutation/extract.py` copies `resample`, `resample_poly`, `decimate`
  and `hilbert` verbatim from SciPy 1.18.1 into `target/scipy_subset.py`, with the BSD notice. A
  pytest plugin (`target/scipy_patch.py`) points `scipy.signal` at the copy, so that SciPy's tests
  and the declarations both call the mutated code. A call to `upfirdn` or `firwin` still goes to
  the installed SciPy.
- **Mutants.** `cosmic-ray` 8.9 generates the mutants (`cosmic-ray.toml`). Its operator filter
  keeps arithmetic, comparison, boolean, number and unary replacements. It drops bitwise, shift,
  power and modulo replacements, and drops the operators that insert or rename variables at
  random. Of 1181 generated mutants, 702 are kept, and one of those, outside the four functions, is
  left out of the counts.
- **Suites.** Each mutant is applied in a private copy of the target and checked three ways:
  - `upstream`: the `TestResample`, `TestDecimate` and `TestHilbert` classes of
    `scipy/signal/tests/test_signaltools.py`, 254 tests, median 2.7 s per mutant.
  - `dsp_fast`: the declarations of `dsp_tests/test_declarations.py` through the pytest plugin,
    30 inputs per relation, derandomized, median 6 s.
  - `dsp_default`: the same declarations with 200 inputs, run only on the mutants that `dsp_fast`
    did not kill. Going from 30 to 200 inputs killed one more mutant
    (<!-- r:automutation.dsp_fast -->397<!-- /r --> → <!-- r:automutation.dsp_default -->398<!-- /r -->).
  - A failing run, an import error or a timeout kills the mutant, as in cosmic-ray.
- **Declarations.** They were written from the documentation, before the run, with one family per
  function:
  - `resample`: upsampling by 2, 3 or 4, inverted by downsampling.
  - `resample_poly`: a rational resampler over drawn rates.
  - `decimate`: q from 2 to 8, IIR and FIR, `zero_phase=False`, with the IIR filter's documented
    tolerance.
  - `hilbert`: an analytic signal whose real part is the input.

  The declarations were not changed after the run. They exercise
  <!-- r:automutation.lines_dsp -->132<!-- /r --> lines of the copy, against
  <!-- r:automutation.lines_upstream -->185<!-- /r --> for SciPy's tests
  (`lines_reached.py`, `coverage` on the unmutated copy).

Run it with `python benchmarks/automutation/run.py` (about 2 hours with 3 workers; `--limit N` for
a quick check; a rerun resumes from `partial.jsonl`). Then run `lines_reached.py` and `analyse.py`.
The run needs SciPy 1.18.1 and cosmic-ray 8.9. Results go to `results.json`, one row per mutant
with its diff, and to `summary.json`.

## What each side misses

**SciPy's tests only.** Of the 177 mutants that only SciPy's tests kill,
<!-- r:automutation.upstream_only_not_reached -->113<!-- /r --> are on lines the declarations never
reach. The other <!-- r:automutation.upstream_only_reached -->64<!-- /r --> are reached but not seen.
Most of those change a value that SciPy's tests compare with a reference, and that no relation
constrains:

- the length of the default `resample_poly` filter (`10 * max_rate` becomes `10 + max_rate`, or
  `2 * half_len + 1` becomes `2 * half_len + 2`);
- the padding constant (`cval` 0 becomes 1).

These operators are still a resampler of the right rate, so this is the blind spot of
[What the relations don't see](blind-spots.md): a wrong operator of the right class. A few others
mutate an argument check that the generators never trigger, for example
`up < 1 or down < 1` becoming `and`.

**dsp-props only.** The upstream `TestHilbert` tests compare values on even lengths (1, 6, 8, 20,
512), and use odd lengths only in an n-D test along other axes. Mutants that break the odd-length
branch of `hilbert` therefore survive them, for instance factor 2 becoming
3 or 1, or a slice bound moved by one. The relation "real part = input", over lengths drawn with
primes and 2^k ± 1, kills
<!-- r:automutation.hilbert_dsp_only -->23<!-- /r --> of these. On `hilbert`, dsp-props kills
<!-- r:automutation.hilbert_dsp_rate -->83 %<!-- /r --> of the mutants and SciPy's tests kill
<!-- r:automutation.hilbert_upstream_rate -->65 %<!-- /r -->. The other two are `decimate`
filter-design constants (order `2 * half_len` becoming `3 * half_len`, ripple 0.05 dB becoming
1.05 dB), which the filter's declared tolerance catches.

**Neither.** Some of these mutants are equivalent on the inputs that matter: a default `axis` of
0 or -1 on a 1-D signal, or `copy=False` on an array that is not modified. Others are real faults
that both sides miss. One example: the cutoff of `decimate`'s Chebyshev filter moved from 0.8/q to
1.8/q. The declaration did not ask for alias rejection (`stopband=None`), and SciPy's tests did not
catch it either.

## What this changes

- A declaration is only as strong as the paths it draws. Options, `axis` and argument checks are
  where SciPy's example-based tests are better. A declaration per option would close part of the
  gap, and it would cost one decorator each.
- Where both reach the code, the two kinds of tests find different faults. SciPy's tests check
  values; the relations check structure across lengths that example tests rarely pick. Running both
  kills <!-- r:automutation.union_rate -->82 %<!-- /r --> of the mutants, against
  <!-- r:automutation.upstream_rate -->78 %<!-- /r --> for SciPy's tests alone. The companion
  article's comparison on 166 generated faults found the same complementarity.

# Generated results

Written by `python benchmarks/make_results.py` from the result files of the benchmark
scripts; do not edit. A number quoted in the documentation is marked
`<!-- r:KEY -->value<!-- /r -->` and checked against this table in CI.

| Key | Value | Source |
| --- | --- | --- |
| `mutants.total` | 31 | `mutant_results.json` |
| `mutants.killed` | 31 | `mutant_results.json` |
| `mutants.examples` | 200 | `mutant_results.json` |
| `mutants.small_counterexamples` | 26 | `mutant_results.json` |
| `mutants.with_counterexample` | 30 | `mutant_results.json` |
| `mutants.counterexample_median` | 2 | `mutant_results.json` |
| `mutants.fail_within_3` | 26 | `mutant_results.json` |
| `precision.examples` | 200 | `precision_mutants.json` |
| `precision.seeds` | 3 | `precision_mutants.json` |
| `precision.fft_twiddles.default` | 90 ε | `precision_mutants.json` |
| `precision.fft_twiddles.passes_up_to` | 21 ε | `precision_mutants.json` |
| `precision.iir_coefficients.default` | 7.1e5 ε | `precision_mutants.json` |
| `precision.iir_coefficients.passes_up_to` | 1.9e5 ε | `precision_mutants.json` |
| `precision.iir_arithmetic.default` | 3.0e6 ε | `precision_mutants.json` |
| `precision.iir_arithmetic.passes_up_to` | 6.9e5 ε | `precision_mutants.json` |
| `precision.resampler_taps.default` | never | `precision_mutants.json` |
| `precision.resampler_taps.passes_up_to` | 6.2e7 ε | `precision_mutants.json` |
| `precision.resampler_arithmetic.default` | 1.1e5 ε | `precision_mutants.json` |
| `precision.resampler_arithmetic.passes_up_to` | 6.7e3 ε | `precision_mutants.json` |
| `precision.streaming_state.default` | 7.1e3 ε | `precision_mutants.json` |
| `precision.streaming_state.passes_up_to` | 750 ε | `precision_mutants.json` |
| `precision.realistic.fft_radix2_recursive_twiddles.loss` | 19 ε | `precision_mutants.json` |
| `precision.realistic.fft_radix2_recursive_twiddles.caught` | 0/3 | `precision_mutants.json` |
| `precision.realistic.fft_radix2_single_twiddles.loss` | 2.5e8 ε | `precision_mutants.json` |
| `precision.realistic.fft_radix2_single_twiddles.caught` | 3/3 | `precision_mutants.json` |
| `precision.realistic.fft_naive_dft_unreduced.loss` | 805 ε | `precision_mutants.json` |
| `precision.realistic.fft_naive_dft_unreduced.caught` | 3/3 | `precision_mutants.json` |
| `precision.realistic.fft_goertzel.loss` | 1.4e4 ε | `precision_mutants.json` |
| `precision.realistic.fft_goertzel.caught` | 3/3 | `precision_mutants.json` |
| `precision.realistic.iir_single_coefficients.loss` | 7.1e9 ε | `precision_mutants.json` |
| `precision.realistic.iir_single_coefficients.caught` | 3/3 | `precision_mutants.json` |
| `precision.reference.numpy_fft.loss` | 2.0 ε | `precision_mutants.json` |
| `precision.reference.radix2_accurate_twiddles.loss` | 2.0 ε | `precision_mutants.json` |
| `precision.reference.dft_accurate_twiddles.loss` | 3.5 ε | `precision_mutants.json` |
| `oracles.fft_twiddles.default` | 90 ε | `precision_oracles.json` |
| `oracles.fft_twiddles.fixed` | never | `precision_oracles.json` |
| `oracles.fft_twiddles.calibrated` | 21 ε | `precision_oracles.json` |
| `oracles.fft_twiddles.differential` | 240 ε | `precision_oracles.json` |
| `oracles.iir_coefficients.default` | 7.1e5 ε | `precision_oracles.json` |
| `oracles.iir_coefficients.fixed` | never | `precision_oracles.json` |
| `oracles.iir_coefficients.calibrated` | 1.7e3 ε | `precision_oracles.json` |
| `oracles.iir_coefficients.differential` | 7.1e5 ε | `precision_oracles.json` |
| `oracles.iir_arithmetic.default` | 3.0e6 ε | `precision_oracles.json` |
| `oracles.iir_arithmetic.fixed` | never | `precision_oracles.json` |
| `oracles.iir_arithmetic.calibrated` | 2.9e3 ε | `precision_oracles.json` |
| `oracles.iir_arithmetic.differential` | 190 ε | `precision_oracles.json` |
| `oracles.resampler_taps.default` | never | `precision_oracles.json` |
| `oracles.resampler_taps.fixed` | never | `precision_oracles.json` |
| `oracles.resampler_taps.calibrated` | never | `precision_oracles.json` |
| `oracles.resampler_taps.differential` | 1.5e7 ε | `precision_oracles.json` |
| `oracles.resampler_arithmetic.default` | 1.1e5 ε | `precision_oracles.json` |
| `oracles.resampler_arithmetic.fixed` | never | `precision_oracles.json` |
| `oracles.resampler_arithmetic.calibrated` | 430 ε | `precision_oracles.json` |
| `oracles.resampler_arithmetic.differential` | 100 ε | `precision_oracles.json` |
| `oracles.streaming_state.default` | 7.1e3 ε | `precision_oracles.json` |
| `oracles.streaming_state.fixed` | never | `precision_oracles.json` |
| `oracles.streaming_state.calibrated` | 28 ε | `precision_oracles.json` |
| `oracles.streaming_state.differential` | 750 ε | `precision_oracles.json` |
| `oracles.false_alarms` | 0 | `precision_oracles.json` |
| `oracles.trusted.fft` | 2.2 ε | `precision_oracles.json` |
| `oracles.trusted.iir` | 27 ε | `precision_oracles.json` |
| `oracles.trusted.resampler` | 17 ε | `precision_oracles.json` |
| `oracles.trusted.streaming` | 2.0 ε | `precision_oracles.json` |
| `calibrated.other_implementations` | 14 | `calibrated_tolerance.json` |
| `calibrated.correct_without_alarm` | 13 | `calibrated_tolerance.json` |
| `calibrated.correct_alarms` | 0 | `calibrated_tolerance.json` |
| `calibrated.torch_fft` | 156 ε | `calibrated_tolerance.json` |
| `calibrated.torch_fft_seeds_failing` | 3/3 | `calibrated_tolerance.json` |
| `calibrated.numpy_fft` | 2.7 ε | `calibrated_tolerance.json` |
| `calibrated.gain_min` | 7 | `calibrated_tolerance.json` |
| `calibrated.gain_max` | 3400 | `calibrated_tolerance.json` |
| `calibrated.filters` | 5 | `calibrated_tolerance.json` |
| `calibrated.ratios` | 4 | `calibrated_tolerance.json` |
| `validation.runs` | 2000 | `validation.json` |
| `validation.failed_runs` | 0 | `validation.json` |
| `validation.comparisons` | 248000 | `validation.json` |
| `profile.n_max` | 1200 | `error_profile.csv` |
| `profile.numpy.worst` | 4.1 ε | `error_profile.csv` |
| `profile.numpy.worst_n` | 1094 | `error_profile.csv` |
| `profile.scipy.worst` | 3.7 ε | `error_profile.csv` |
| `profile.scipy.worst_n` | 1091 | `error_profile.csv` |
| `profile.torch.worst` | 195 ε | `error_profile.csv` |
| `profile.torch.worst_n` | 596 | `error_profile.csv` |
| `comparison.lfilter.dsp_props` | 86 % | `comparison/summary.json` |
| `comparison.lfilter.params` | 100 % | `comparison/summary.json` |
| `comparison.lfilter.sop` | 100 % | `comparison/summary.json` |
| `comparison.lfilter.mutants` | 51 | `comparison/summary.json` |
| `comparison.upfirdn.dsp_props` | 72 % | `comparison/summary.json` |
| `comparison.upfirdn.params` | 97 % | `comparison/summary.json` |
| `comparison.upfirdn.sop` | 89 % | `comparison/summary.json` |
| `comparison.upfirdn.mutants` | 36 | `comparison/summary.json` |
| `iir_structure.named_ratio_min` | 1.1 | `iir_structure.json` |
| `iir_structure.named_ratio_max` | 2.8 | `iir_structure.json` |
| `iir_structure.random_ratio_min` | 0.62 | `iir_structure.json` |
| `iir_structure.random_ratio_median` | 2.0 | `iir_structure.json` |
| `iir_structure.random_ratio_max` | 5.7 | `iir_structure.json` |
| `iir_structure.random_gap_median` | 220 | `iir_structure.json` |
| `iir_structure.iir_coefficients.df1` | 7.1e5 ε | `iir_structure.json` |
| `iir_structure.iir_coefficients.df2t` | 1.9e5 ε | `iir_structure.json` |
| `iir_structure.iir_arithmetic.df1` | 1.5e6 ε | `iir_structure.json` |
| `iir_structure.iir_arithmetic.df2t` | 1.5e6 ε | `iir_structure.json` |
| `rediscovery.candidates` | 246 | `bugcorpus/triage.tsv` |
| `rediscovery.in_scope` | 14 | `bugcorpus/triage.tsv` |
| `rediscovery.bugs` | 12 | `rediscovery.json` |
| `rediscovery.found` | 7 | `rediscovery.json` |
| `rediscovery.found_all_seeds` | 6 | `rediscovery.json` |
| `rediscovery.seeds` | 3 | `rediscovery.json` |
| `rediscovery.examples` | 200 | `rediscovery.json` |
| `rediscovery.inputs_median` | 4.5 | `rediscovery.json` |
| `rediscovery.counterexample_median` | 2 | `rediscovery.json` |
| `rediscovery.found_seeds` | 20 | `rediscovery.json` |
| `rediscovery.small_counterexamples` | 15 | `rediscovery.json` |

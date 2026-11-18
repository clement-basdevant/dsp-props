# API reference

Generated from the docstrings of the source code.

| Module | Content |
| --- | --- |
| [`dsp_props`](dsp_props.md) | The declarative API: `transform`, `lti_filter`, `resampler`, `streaming`, `stateful`, `verify`, `calibrate`, and the declaration and report classes. This is what `import dsp_props as dp` gives you. |
| [`dsp_props.core`](core.md) | Tolerance models (`FFTTolerance`, `IIRTolerance`, `CalibratedTolerance`), calibration and extended-precision references. |
| [`dsp_props.generators`](generators.md) | Hypothesis strategies for signals, lengths and block splits. |
| [`dsp_props.relations`](relations.md) | One class per metamorphic relation. |
| [`dsp_props.report`](report.md) | `Report`, `Failure`, `PropertyViolation` and the HTML renderer. |
| [`dsp_props.tolerances`](tolerances.md) | Versioned calibrations (`tolerances.json`): `ToleranceFile`, `PrecisionComparison`. |
| [`dsp_props.regression`](regression.md) | Regression tests written from minimal counterexamples: `write_regression_tests`, `replay`, `to_source`. |
| [`dsp_props.adapters`](adapters.md) | Loading C functions through `ctypes`. |
| [`dsp_props.cli`](cli.md) | The `dsp-props check` command. |

"""Does dsp-props measure correctly?  Four checks of the tool itself.

1. ``reference``: the extended-precision references used by ``dp.calibrate``
   and the IIR tests (``longdouble_dft``, ``longdouble_lfilter``) are compared
   with a 50-digit mpmath computation.
2. ``false-positives``: every declaration is run on correct implementations
   (NumPy, SciPy and the four reference subjects of benchmarks/comparison)
   with many independent seeds; a correct operator must never fail.
3. ``margin``: during those runs, every comparison records error / tolerance.
   Values close to 1 would mean false alarms are one unlucky input away.
4. ``generators``: the signal generators are sampled through Hypothesis and
   checked against what they claim (pathological lengths, amplitude decades,
   one non-zero sample per impulse, off-grid and near-Nyquist frequencies,
   close tones, empty and one-sample blocks).

    python benchmarks/validation.py                 # all checks, writes benchmarks/validation.json
    python benchmarks/validation.py --only generators --seeds 3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE / "comparison"), str(ROOT / "src"), str(ROOT)]

import numpy as np  # noqa: E402
import scipy.signal as ss  # noqa: E402
from hypothesis import HealthCheck, given, seed as hseed, settings, strategies as st  # noqa: E402

import dsp_props as dp  # noqa: E402
import dsp_props.relations.lti as rel_lti  # noqa: E402
import dsp_props.relations.multirate as rel_multirate  # noqa: E402
import dsp_props.relations.streaming as rel_streaming  # noqa: E402
import dsp_props.relations.transforms as rel_transforms  # noqa: E402
from dsp_props.core import tolerance as tolmod  # noqa: E402
from dsp_props.core.calibration import longdouble_dft  # noqa: E402
from dsp_props.core.iir import longdouble_lfilter  # noqa: E402
import importlib  # noqa: E402

gen = importlib.import_module("dsp_props.generators.signals")  # the package re-exports a function named signals

EPS = np.finfo(float).eps


# ----------------------------------------------------------------------------- 1. references

def check_reference(seed=0):
    import mpmath

    mpmath.mp.dps = 50
    rng = np.random.default_rng(seed)
    out = {"dft": [], "lfilter": []}
    for n in [1, 2, 3, 4, 5, 7, 8, 16, 17, 31, 32, 64, 97, 127, 128, 129, 255, 256, 257]:
        x = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        xs = [mpmath.mpc(complex(v)) for v in x]
        w = [mpmath.exp(-2j * mpmath.pi * k / n) for k in range(n)]
        ref = [mpmath.fsum(xs[t] * w[(k * t) % n] for t in range(n)) for k in range(n)]
        got = longdouble_dft(x)
        # str() keeps every digit of the long double value
        num = mpmath.sqrt(mpmath.fsum(abs(mpmath.mpc(str(g.real), str(g.imag)) - r) ** 2 for g, r in zip(got, ref)))
        den = mpmath.sqrt(mpmath.fsum(abs(r) ** 2 for r in ref))
        out["dft"].append({"n": n, "rel_error_in_eps64": float(num / den) / EPS})
    poles = 0.95 * np.exp(1j * np.array([0.3, -0.3, 1.2, -1.2]))
    a = np.real(np.poly(poles))
    b = np.array([0.2, 0.0, -0.4, 0.0, 0.2])
    for n in [1, 2, 5, 16, 64, 256]:
        x = rng.standard_normal(n)
        am = [mpmath.mpf(float(v)) for v in a]
        bm = [mpmath.mpf(float(v)) for v in b]
        xm = [mpmath.mpf(float(v)) for v in x]
        ym = []
        for i in range(n):
            acc = mpmath.fsum(bm[k] * xm[i - k] for k in range(min(len(bm), i + 1)))
            acc -= mpmath.fsum(am[k] * ym[i - k] for k in range(1, min(len(am), i + 1)))
            ym.append(acc / am[0])
        got = longdouble_lfilter(b, a, x)
        num = mpmath.sqrt(mpmath.fsum((mpmath.mpf(str(g)) - r) ** 2 for g, r in zip(got, ym)))
        den = mpmath.sqrt(mpmath.fsum(r ** 2 for r in ym))
        out["lfilter"].append({"n": n, "rel_error_in_eps64": float(num / den) / EPS})
    for k in ("dft", "lfilter"):
        out[k + "_max_in_eps64"] = max(r["rel_error_in_eps64"] for r in out[k])
    out["longdouble_eps_in_eps64"] = float(np.finfo(np.longdouble).eps) / EPS
    out["mpmath_version"] = mpmath.__version__
    return out


# ----------------------------------------------------------------------------- 2 + 3. false positives, margin

_MARGINS = defaultdict(list)
_CURRENT = {"rel": None}


def _instrumented_assert_close(actual, expected, model, *, ref_norm=None, what="", precision=None):
    actual_a, expected_a = np.asarray(actual), np.asarray(expected)
    if actual_a.shape == expected_a.shape and np.all(np.isfinite(actual_a)):
        rn = tolmod.norm(expected_a) if ref_norm is None else ref_norm
        err = tolmod.norm(actual_a - expected_a)
        tol = model.bound(max(actual_a.size, 1), actual_a.dtype if precision is None else precision, rn)
        if tol > 0:
            _MARGINS[_CURRENT["rel"]].append(err / tol)
    return _ORIGINAL_ASSERT(actual, expected, model, ref_norm=ref_norm, what=what, precision=precision)


_ORIGINAL_ASSERT = tolmod.assert_close


def _install_probe():
    for mod in (tolmod, rel_lti, rel_multirate, rel_streaming, rel_transforms):
        if hasattr(mod, "assert_close"):
            mod.assert_close = _instrumented_assert_close


def correct_targets():
    """(name, declarations) of implementations that are correct, as far as anyone knows."""
    import importlib.util

    import suites

    def load_subject(name):
        spec = importlib.util.spec_from_file_location(f"subj_{name}", HERE / "comparison" / "subjects" / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    class ScipyLfilter1D:  # the comparison subjects' interface, backed by SciPy
        @staticmethod
        def lfilter(b, a, x, zi=None):
            return ss.lfilter(b, a, x) if zi is None else ss.lfilter(b, a, x, zi=zi)

    rfft = dp.transform(inverse=np.fft.irfft, kind="rdft", parseval=True)(lambda x: np.fft.rfft(x))
    targets = [
        ("numpy.fft", suites.fft_decls(np.fft)),
        ("numpy.fft.rfft", {"rfft": rfft}),
        ("scipy.signal.upfirdn", suites.upfirdn_decls(ss)),
        # SciPy's lfilter on an empty block with zi is a known bug (gh-22571): keep it out of the
        # false-positive count and report it separately.
        ("scipy.signal.lfilter", {k: v for k, v in suites.lfilter_decls(ScipyLfilter1D).items()
                                  if k != "lfilter_streaming"}),
        ("scipy.signal.sosfilt", {"sosfilt_butter4": dp.lti_filter(transfer=ss.butter(4, 0.2))(
            lambda x, sos=ss.butter(4, 0.2, output="sos"): ss.sosfilt(sos, x))}),
        ("scipy.signal.resample_poly", {f"resample_poly_{u}_{d}": dp.resampler(u, d)(
            lambda x, u=u, d=d: ss.resample_poly(x, u, d)) for u, d in [(2, 3), (3, 1), (1, 2)]}),
    ]
    for name in ["fft", "lfilter", "upfirdn", "ola"]:
        targets.append((f"subject:{name}", suites.SUBJECTS[name][0](load_subject(name))))
    return targets


def check_false_positives(seeds=20, max_examples=200):
    _install_probe()
    from dsp_props.api import _hypothesis_test

    rows = []
    for tname, decls in correct_targets():
        for dname, func in decls.items():
            spec = func.__dsp_props__
            for rel in spec.relations:
                fails = []
                t0 = time.perf_counter()
                for s in range(seeds):
                    _CURRENT["rel"] = (tname, dname, rel.name)
                    run, trace = _hypothesis_test(rel, spec, max_examples, False, 1000 + s)
                    try:
                        with warnings.catch_warnings(), np.errstate(all="ignore"):
                            warnings.simplefilter("ignore")
                            run()
                    except Exception as exc:  # noqa: BLE001
                        fails.append({"seed": 1000 + s, "error": type(exc).__name__, "message": str(exc)[:300]})
                m = np.asarray(_MARGINS.get((tname, dname, rel.name), []))
                rows.append({"target": tname, "declaration": dname, "relation": rel.name, "seeds": seeds,
                             "failures": len(fails), "failure_details": fails[:3],
                             "comparisons": int(m.size),
                             "margin_median": float(np.median(m)) if m.size else None,
                             "margin_p99": float(np.percentile(m, 99)) if m.size else None,
                             "margin_max": float(m.max()) if m.size else None,
                             "seconds": round(time.perf_counter() - t0, 1)})
                print(f"  {tname:28s} {dname:22s} {rel.name:28s} fails {len(fails)}/{seeds}"
                      f"  max err/tol {rows[-1]['margin_max']}", flush=True)
    all_m = np.concatenate([np.asarray(v) for v in _MARGINS.values() if v]) if _MARGINS else np.zeros(0)
    nonzero = all_m[all_m > 0]
    return {"rows": rows, "runs": int(sum(r["seeds"] for r in rows)),
            "failed_runs": int(sum(r["failures"] for r in rows)),
            "comparisons": int(all_m.size),
            "margin_quantiles": {q: float(np.percentile(all_m, q)) for q in (50, 90, 99, 99.9, 100)} if all_m.size else {},
            "margin_histogram_log10": dict(Counter(int(np.floor(np.log10(v))) for v in nonzero)),
            "exact_comparisons": int((all_m == 0).sum())}


def check_known_bug(seeds=5):
    """SciPy lfilter on empty blocks with zi (gh-22571): dsp-props must flag it on SciPy <= 1.18."""
    import scipy

    class ScipyLfilter1D:
        @staticmethod
        def lfilter(b, a, x, zi=None):
            return ss.lfilter(b, a, x) if zi is None else ss.lfilter(b, a, x, zi=zi)

    import suites

    func = suites.lfilter_decls(ScipyLfilter1D)["lfilter_streaming"]
    out = []
    for s in range(seeds):
        r = dp.verify(func, raise_on_failure=False, seed=2000 + s)
        out.append({"seed": 2000 + s, "passed": not r.failures,
                    "report": str(r)[:400]})
    return {"scipy": scipy.__version__, "runs": out}


# ----------------------------------------------------------------------------- 4. generators

def check_generators(n_draws=4000, seed=0):
    stats = {}

    # lengths
    drawn = []

    @hseed(seed)
    @settings(max_examples=n_draws, database=None, deadline=None, suppress_health_check=list(HealthCheck))
    @given(gen.lengths(512))
    def collect_len(n):
        drawn.append(n)

    collect_len()
    d = np.array(drawn)
    patho = set(gen.PATHOLOGICAL_LENGTHS)
    stats["lengths"] = {"draws": int(d.size), "min": int(d.min()), "max": int(d.max()),
                        "fraction_pathological": float(np.mean([v in patho for v in d])),
                        "pathological_values_hit": sorted(int(v) for v in set(d) & patho),
                        "pathological_values_missed": sorted(patho - set(int(v) for v in d)),
                        "fraction_prime": float(np.mean([_is_prime(int(v)) for v in d])),
                        "fraction_power_of_two": float(np.mean([(v & (v - 1)) == 0 for v in d]))}

    # The run above asks for many draws and Hypothesis then explores the uniform branch at length;
    # a relation draws 200 examples, the default, so measure the same fraction in runs of that size.
    per_run = []
    for s in range(20):
        run = []

        @hseed(s)
        @settings(max_examples=200, database=None, deadline=None, suppress_health_check=list(HealthCheck))
        @given(gen.lengths(512))
        def collect_run(n):
            run.append(n)

        collect_run()
        per_run.append(float(np.mean([v in patho for v in run])))
    stats["lengths"]["runs_of_200"] = {"runs": len(per_run), "mean_fraction_pathological": float(np.mean(per_run)),
                                       "min": min(per_run), "max": max(per_run)}

    # signal families, through the public signals(n) strategy (as the relations use it)
    per_kind = defaultdict(list)
    freqs = defaultdict(list)  # frequencies actually drawn, recorded at the source
    current = {"kind": None}
    original_sine = gen._sine

    def recording_sine(n, f, phi, a):
        freqs[current["kind"]].append((n, f))
        return original_sine(n, f, phi, a)

    @st.composite
    def labelled(draw, n):
        k = draw(st.integers(0, len(gen.SIGNAL_KINDS) - 1))
        kind = gen.SIGNAL_KINDS[k]
        current["kind"] = kind.__name__
        return kind.__name__, draw(kind(n))

    gen._sine = recording_sine
    for n in [1, 2, 7, 64, 257]:
        draws = []

        @hseed(seed + n)
        @settings(max_examples=n_draws // 5, database=None, deadline=None, suppress_health_check=list(HealthCheck))
        @given(labelled(n))
        def collect(item):
            draws.append(item)

        collect()
        for name, x in draws:
            per_kind[name].append((n, x))
    gen._sine = original_sine
    checks = {}
    for name, items in per_kind.items():
        amps = [float(np.max(np.abs(x))) for _, x in items if np.any(x)]
        decades = sorted({int(np.round(np.log10(a))) for a in amps if a > 0})
        c = {"draws": len(items), "all_finite": bool(all(np.all(np.isfinite(x)) for _, x in items)),
             "length_ok": bool(all(len(x) == n for n, x in items)), "amplitude_decades": decades}
        if name == "impulse":
            c["exactly_one_nonzero"] = float(np.mean([np.count_nonzero(x) == 1 for _, x in items]))
        if name == "dc":
            c["constant"] = float(np.mean([np.ptp(x) == 0 for _, x in items]))
        if name in ("offgrid_sine", "near_nyquist_sine"):
            nf = freqs[name]
            f = np.array([v for _, v in nf])
            on_grid = np.array([abs(v * n - round(v * n)) < 1e-9 for n, v in nf])
            c["freq_range"] = [float(f.min()), float(f.max())]
            c["fraction_on_dft_grid"] = float(on_grid.mean())
            c["fraction_freq_0_or_nyquist"] = float(np.mean((f == 0.0) | (f == 0.5)))
            c["distinct_freqs"] = int(len(set(f.tolist())))
        if name == "close_tones":
            nf = freqs[name]
            sep = np.array([(nf[i + 1][1] - nf[i][1]) * nf[i][0] for i in range(0, len(nf) - 1, 2)])
            c["separation_in_bins"] = [float(np.min(sep)), float(np.median(sep)), float(np.max(sep))]
            c["fraction_below_one_bin"] = float(np.mean(sep < 1.0))
            c["fraction_identical_tones"] = float(np.mean(sep == 0.0))
        if name == "sparse":
            nnz = [np.count_nonzero(x) for _, x in items]
            c["nonzeros_range"] = [int(min(nnz)), int(max(nnz))]
        checks[name] = c
    stats["signals"] = checks

    # block splits
    splits = []

    @hseed(seed)
    @settings(max_examples=n_draws, database=None, deadline=None, suppress_health_check=list(HealthCheck))
    @given(st.integers(0, 64).flatmap(lambda n: st.tuples(st.just(n), gen.block_splits(n))))
    def collect_split(item):
        splits.append(item)

    collect_split()
    sizes = [np.diff([0, *cuts, n]) for n, cuts in splits]
    stats["block_splits"] = {"draws": len(splits),
                             "fraction_with_empty_block": float(np.mean([np.any(s == 0) for s in sizes])),
                             "fraction_with_one_sample_block": float(np.mean([np.any(s == 1) for s in sizes])),
                             "blocks_cover_signal": bool(all(s.sum() == n and np.all(s >= 0)
                                                             for s, (n, _) in zip(sizes, splits)))}
    return stats


def _is_prime(n):
    return n >= 2 and all(n % p for p in range(2, int(n ** 0.5) + 1))


# ----------------------------------------------------------------------------- main

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", nargs="+", choices=["reference", "false-positives", "known-bug", "generators"])
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--out", default=str(HERE / "validation.json"))
    args = p.parse_args(argv)
    which = args.only or ["reference", "generators", "known-bug", "false-positives"]
    out = json.loads(Path(args.out).read_text()) if Path(args.out).exists() else {}
    import hypothesis
    import scipy

    out["environment"] = {"python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__,
                          "hypothesis": hypothesis.__version__}
    for name in which:
        print(f"== {name}", flush=True)
        t0 = time.perf_counter()
        if name == "reference":
            out[name] = check_reference()
        elif name == "false-positives":
            out[name] = check_false_positives(seeds=args.seeds)
        elif name == "known-bug":
            out[name] = check_known_bug()
        else:
            out[name] = check_generators()
        out[name]["seconds"] = round(time.perf_counter() - t0, 1)
        Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "false-positives"}, indent=1)[:6000])
    if "false-positives" in out:
        fp = out["false-positives"]
        print({k: fp[k] for k in ("runs", "failed_runs", "comparisons", "margin_quantiles", "exact_comparisons")})


if __name__ == "__main__":
    main()

"""Error profile of GPU FFTs against an extended-precision DFT (plan 3.5).

Usage: python benchmarks/gpu_fft_profile.py [--max-n 1200] [--trials 3] [--json benchmarks/gpu_fft_profile.json]
       python benchmarks/gpu_fft_profile.py --cpu   # check the script on the CPU backends only

Each backend found on the machine is measured: PyTorch on CUDA (cuFFT) or Apple MPS, CuPy (cuFFT),
JAX on GPU (with ``jax_enable_x64``); NumPy and PyTorch on the CPU are measured too, as the baseline.
Inputs are complex random signals in float64 and in float32 (complex128 and complex64), moved to
the device through ``dp.array_api``. The error is ``||fft(x) - DFT(x)|| / (eps ||DFT(x)||)`` in
epsilons of the input's precision, against ``extended_dft`` (long double FFT where it is extended,
double-double DFT otherwise), the worst of ``--trials`` signals per length. A GPU that computes a
float64 transform in single precision shows up as about 1e8 to 1e9 eps.

The JSON keeps the device names and library versions, so profiles from several machines can be
compared. Needs no GPU to run, but measures nothing new without one.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import platform
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src")]

import numpy as np  # noqa: E402

import dsp_props as dp  # noqa: E402
from dsp_props.core.calibration import extended_dft, relative_error  # noqa: E402


def backends(cpu_only):
    """name -> (fft taking NumPy arrays, description); only what this machine has."""
    out = {"numpy (cpu)": (np.fft.fft, f"numpy {np.__version__}")}
    try:
        import torch

        out["torch (cpu)"] = (dp.array_api(torch.fft.fft, xp=torch, device="cpu"), f"torch {torch.__version__}")
        if not cpu_only and torch.cuda.is_available():
            out["torch (cuda)"] = (dp.array_api(torch.fft.fft, xp=torch, device="cuda"),
                                   f"torch {torch.__version__}, {torch.cuda.get_device_name(0)}, "
                                   f"CUDA {torch.version.cuda}")
        if not cpu_only and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            # MPS has no float64: only the complex64 rows are measured
            out["torch (mps)"] = (dp.array_api(torch.fft.fft, xp=torch, device="mps"), f"torch {torch.__version__}, MPS")
    except ImportError:
        pass
    if not cpu_only:
        try:
            import cupy

            name = cupy.cuda.runtime.getDeviceProperties(0)["name"].decode()
            out["cupy (cuda)"] = (dp.array_api(cupy.fft.fft, xp=cupy), f"cupy {cupy.__version__}, {name}")
        except Exception:  # noqa: BLE001 - not installed, or no CUDA device
            pass
        try:
            import jax

            jax.config.update("jax_enable_x64", True)
            gpus = [d for d in jax.devices() if d.platform == "gpu"]
            if gpus:
                import jax.numpy as jnp

                out["jax (gpu)"] = (dp.array_api(jnp.fft.fft, xp=jnp, device=gpus[0]),
                                    f"jax {jax.__version__}, {gpus[0].device_kind}")
        except Exception:  # noqa: BLE001 - not installed, or no GPU
            pass
    return out


def measure(fft, n, dtype, trials, rng):
    worst = 0.0
    for _ in range(trials):
        x = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(dtype)
        worst = max(worst, relative_error(fft, extended_dft, x))
    return worst


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-n", type=int, default=1200)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--cpu", action="store_true", help="CPU backends only (a check of the script)")
    parser.add_argument("--json", type=pathlib.Path, default=ROOT / "benchmarks" / "gpu_fft_profile.json")
    args = parser.parse_args()
    lengths = list(range(2, args.max_n + 1))
    found = backends(args.cpu)
    result = {"machine": platform.platform(), "trials": args.trials, "lengths": [2, args.max_n],
              "backends": {}}
    for name, (fft, description) in found.items():
        rows = {}
        for dtype in (np.complex128, np.complex64):
            rng = np.random.default_rng(0)  # same signals for every backend
            try:
                errors = {n: measure(fft, n, dtype, args.trials, rng) for n in lengths}
            except (TypeError, RuntimeError) as exc:  # e.g. no float64 on the device
                rows[np.dtype(dtype).name] = {"error": f"{type(exc).__name__}: {exc}"[:200]}
                print(f"{name:14} {np.dtype(dtype).name:10} not measured: {exc}"[:120], flush=True)
                continue
            worst_n = max(errors, key=errors.get)
            pow2 = [errors[n] for n in lengths if n & (n - 1) == 0]
            rows[np.dtype(dtype).name] = {
                "worst_eps": round(errors[worst_n], 2), "worst_n": worst_n,
                "median_eps": round(float(np.median(list(errors.values()))), 2),
                "worst_power_of_two_eps": round(max(pow2), 2),
                "per_length_eps": {str(n): round(e, 3) for n, e in errors.items()},
            }
            print(f"{name:14} {np.dtype(dtype).name:10} worst {errors[worst_n]:8.1f} eps at N={worst_n:<5} "
                  f"median {rows[np.dtype(dtype).name]['median_eps']:6.2f} eps", flush=True)
        result["backends"][name] = {"description": description, "rows": rows}
    args.json.write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()

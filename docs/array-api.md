# Other array libraries and GPUs

The relations call an operator with NumPy arrays. An operator written for PyTorch, CuPy, JAX or any
[Array API](https://data-apis.org/array-api/) namespace is wrapped with `dp.array_api`, which moves
the inputs to that library (and to a device) and brings the outputs back as NumPy arrays:

```python
import numpy as np
import torch
import dsp_props as dp

@dp.transform(inverse=dp.array_api(torch.fft.ifft, xp=torch, device="cuda"), parseval=True)
@dp.array_api(xp=torch, device="cuda")
def gpu_fft(x):
    return torch.fft.fft(x)

dp.verify(gpu_fft)
```

- Every NumPy array argument is converted, including those inside tuples and lists, such as the state of
  `dp.stateful`. Other arguments, such as a length, pass unchanged. Outputs come back the same way,
  so a `(y, state)` pair stays a pair.
- The dtype is kept in both directions. JAX computes in single precision unless
  `jax.config.update("jax_enable_x64", True)` is set. A float32 result then fails the precision check
  of a float64 declaration. This is the intended outcome: declare such an operator with
  `precision=np.float32`, or enable x64.
- Moving an IEEE array to a device and back is exact. Every difference the relations see therefore
  comes from the operator itself.
- Apple's MPS has no float64. Declare operators that run there with `precision=np.float32`.

The tests cover NumPy as a namespace and PyTorch on the CPU. CuPy and JAX go through the same two
paths (`xp.asarray(x, device=...)` on the way in; DLPack or the array protocol on the way out). They
have not been run on a GPU in CI.

## Error profile of GPU FFTs

`python benchmarks/gpu_fft_profile.py` measures each FFT the machine has against an
extended-precision DFT. The backends are PyTorch on CUDA or MPS, CuPy, and JAX on GPU, with NumPy and
PyTorch on the CPU as the baseline. It covers N = 2 to 1200, in complex128 and complex64. It writes the
worst and median error per backend, in epsilons of the input's precision, to
`benchmarks/gpu_fft_profile.json`, with the device names and library versions. `--cpu` checks the
script without a GPU.

The profile tells you which tolerance a GPU FFT needs: the default model (20 ε log2 N), or a
calibration (`dp.calibrate` with the wrapped operator, see [Choosing a tolerance](tolerances.md)). On
the CPU, PyTorch's FFT already needs a calibration at some lengths: see the
[gallery](gallery.md#pytorchs-fft-at-lengths-with-a-large-prime-factor).

### Measured: CuPy on an RTX 5070

One profile, run on 20 November 2026 (`benchmarks/gpu_fft_profile.json`): CuPy 14.2.0 with CUDA
12.9 on an NVIDIA GeForce RTX 5070 under Windows 11, and NumPy 2.5.3 on the same machine's CPU.
PyTorch was not installed, so its GPU FFT is not measured yet. The table gives the worst error over
N = 2 to 1200 (3 random signals per length) and the median over all lengths, in epsilons of the
input's precision:

| Backend | complex128 worst | complex128 median | complex64 worst | complex64 median |
| --- | --- | --- | --- | --- |
| CuPy (cuFFT, GPU) | <!-- r:gpu_fft.cupy_gpu.c128.worst -->3.6 ε<!-- /r --> (N = <!-- r:gpu_fft.cupy_gpu.c128.worst_n -->1094<!-- /r -->) | <!-- r:gpu_fft.cupy_gpu.c128.median -->1.5 ε<!-- /r --> | <!-- r:gpu_fft.cupy_gpu.c64.worst -->3.0 ε<!-- /r --> (N = <!-- r:gpu_fft.cupy_gpu.c64.worst_n -->124<!-- /r -->) | <!-- r:gpu_fft.cupy_gpu.c64.median -->1.5 ε<!-- /r --> |
| NumPy (CPU) | <!-- r:gpu_fft.numpy_cpu.c128.worst -->3.6 ε<!-- /r --> (N = <!-- r:gpu_fft.numpy_cpu.c128.worst_n -->1094<!-- /r -->) | <!-- r:gpu_fft.numpy_cpu.c128.median -->1.2 ε<!-- /r --> | <!-- r:gpu_fft.numpy_cpu.c64.worst -->0.34 ε<!-- /r --> | <!-- r:gpu_fft.numpy_cpu.c64.median -->0.22 ε<!-- /r --> |

- cuFFT computes float64 transforms in double precision. It does not fall back to single
  precision, and its errors stay within a few ε of NumPy's.
- The default tolerance (20 ε log2 N) covers it with a wide margin at every length measured. A
  GPU FFT behaves here like a CPU one and needs no calibration.
- NumPy's complex64 transforms come out at a third of an ε. This is consistent with an FFT
  computed in double precision and rounded once at the end, which a float32 declaration would
  not notice.

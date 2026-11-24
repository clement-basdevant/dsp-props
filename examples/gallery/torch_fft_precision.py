"""The CPU FFT of PyTorch is less accurate than NumPy's at some lengths.

``dp.calibrate`` measures an FFT length by length against an extended
precision DFT and reports the worst error in units of eps. For lengths with
a large prime factor, PyTorch's CPU FFT (float64) exceeds 100 eps where NumPy
stays under 3 eps (N = 596 = 4 x 149; 195 eps at worst in
``benchmarks/error_profile.py``). The gap depends on the hardware, not
only on the versions: on some CPUs PyTorch's FFT is as accurate as NumPy's
at the same length (1.6 eps on an AMD Ryzen 7 9800X3D, torch 2.15.0+cpu).
Both pass the default
tolerance, 20 eps log2 N: this is a precision difference, not a bug, and the
calibration is the tool that shows it. ``dsp-props calibrate --compare``
turns the same measurement into a check between two versions of a library.

Needs ``torch``.
"""

import numpy as np

import dsp_props as dp

LENGTHS = [512, 596, 600, 1021]


def torch_fft(x):
    import torch

    return torch.fft.fft(torch.from_numpy(np.asarray(x, dtype=np.complex128))).numpy()


def worst_eps(op, lengths=LENGTHS):
    """Worst measured error of ``op`` per length, in eps."""
    return dp.calibrate(op, lengths=lengths, trials=4, base=None).ratios


if __name__ == "__main__":
    numpy_eps, torch_eps = worst_eps(np.fft.fft), worst_eps(torch_fft)
    for n in LENGTHS:
        print(f"N={n:5d}  numpy {numpy_eps[n]:6.1f} eps   torch {torch_eps[n]:6.1f} eps")

"""Operators written for another array library: PyTorch, CuPy, JAX, any Array API namespace.

The relations call an operator with NumPy arrays and compare NumPy arrays. ``array_api`` wraps a
function written for the namespace ``xp`` so that its inputs are moved to ``xp`` (and to
``device``, a GPU for example) and its outputs come back as NumPy arrays::

    import torch

    @dp.transform(inverse=dp.array_api(torch.fft.ifft, xp=torch, device="cuda"))
    @dp.array_api(xp=torch, device="cuda")
    def gpu_fft(x):
        return torch.fft.fft(x)

Arguments that are NumPy arrays are converted, recursively inside tuples and lists (the state of
``dp.stateful`` included); other arguments (lengths, parameters) pass unchanged. Outputs are
converted back the same way, so a ``(y, state)`` pair stays a pair. The dtype is kept on both
sides: a library that computes in single precision by default (JAX without ``jax_enable_x64``)
returns float32 to a float64 declaration, which then fails its precision check, as it should;
declare ``precision=np.float32`` for such an operator.

The conversion only moves data; it is not part of what is tested, except that a device transfer
must be exact, which it is for IEEE types.
"""

from __future__ import annotations

import functools

import numpy as np


def _to_xp(value, xp, device):
    if isinstance(value, np.ndarray):
        value = np.ascontiguousarray(value)  # negative strides (x[::-1]) are not portable
        return xp.asarray(value) if device is None else xp.asarray(value, device=device)
    if isinstance(value, tuple):
        return tuple(_to_xp(v, xp, device) for v in value)
    if isinstance(value, list):
        return [_to_xp(v, xp, device) for v in value]
    return value


def to_numpy(value):
    """A NumPy copy of an array of any library and device; tuples and lists are converted item by item."""
    if isinstance(value, tuple):
        return tuple(to_numpy(v) for v in value)
    if isinstance(value, list):
        return [to_numpy(v) for v in value]
    if value is None or isinstance(value, (np.ndarray, np.generic, bool, int, float, complex, str)):
        return value
    if hasattr(value, "detach") and hasattr(value, "cpu"):  # PyTorch
        value = value.detach().cpu()
        if getattr(value, "is_conj", lambda: False)():
            value = value.resolve_conj()
        return value.numpy()
    if hasattr(value, "get") and type(value).__module__.startswith("cupy"):  # CuPy
        return value.get()
    if hasattr(value, "to_device"):  # Array API standard: move to the host first
        try:
            value = value.to_device("cpu")
        except Exception:  # noqa: BLE001 - not every library names its host device "cpu"
            pass
    try:
        return np.from_dlpack(value)
    except Exception:  # noqa: BLE001 - fall back on the array protocol (JAX, older libraries)
        return np.asarray(value)


def array_api(func=None, *, xp, device=None):
    """Wrap ``func``, written for the array namespace ``xp``, so that it takes and returns NumPy arrays.

    ``xp`` is the module of the library (``torch``, ``cupy``, ``jax.numpy``, or an Array API
    namespace such as ``array_api_strict``); ``device`` is passed to ``xp.asarray`` when given.
    Usable as a decorator, ``@dp.array_api(xp=torch, device="cuda")``, or as a call,
    ``dp.array_api(torch.fft.ifft, xp=torch)``.
    """
    if func is None:
        return functools.partial(array_api, xp=xp, device=device)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        args = _to_xp(args, xp, device)
        kwargs = {k: _to_xp(v, xp, device) for k, v in kwargs.items()}
        return to_numpy(func(*args, **kwargs))

    wrapper.xp = xp  # type: ignore[attr-defined]
    wrapper.device = device  # type: ignore[attr-defined]
    return wrapper

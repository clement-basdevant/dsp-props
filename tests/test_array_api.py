"""dp.array_api: operators written for another array library (plan 3.5)."""

import numpy as np
import pytest

import dsp_props as dp
from dsp_props.array_api import to_numpy


class _Box:
    """A minimal foreign array: data kept apart from NumPy, converted back through __array__."""

    def __init__(self, data):
        self.data = np.array(data, copy=True)

    def __array__(self, dtype=None, copy=None):
        return self.data if dtype is None else self.data.astype(dtype)


class _Namespace:
    """A stand-in array namespace recording the device it was asked for."""

    def __init__(self):
        self.devices = []

    def asarray(self, x, device=None):
        self.devices.append(device)
        return _Box(x)


def test_arguments_are_converted_and_outputs_come_back_as_numpy():
    xp = _Namespace()
    seen = []

    @dp.array_api(xp=xp, device="gpu:0")
    def op(x, state, n):
        seen.append((type(x), type(state[0]), n))
        return _Box(np.asarray(x) * 2), (_Box(np.asarray(state[0]) + 1),)

    y, (s,) = op(np.arange(3.0)[::-1], (np.zeros(2),), 3)
    assert seen == [(_Box, _Box, 3)]
    assert xp.devices == ["gpu:0", "gpu:0"]
    assert isinstance(y, np.ndarray) and isinstance(s, np.ndarray)
    np.testing.assert_array_equal(y, [4.0, 2.0, 0.0])
    np.testing.assert_array_equal(s, [1.0, 1.0])
    assert op.__name__ == "op" and op.xp is xp


def test_numpy_as_namespace_verifies():
    fft = dp.transform(inverse=dp.array_api(np.fft.ifft, xp=np))(dp.array_api(np.fft.fft, xp=np))
    assert dp.verify(fft, max_examples=20, seed=0, raise_on_failure=False).ok


def test_dtype_is_kept():
    x32 = np.ones(4, np.float32)
    assert dp.array_api(lambda x: x, xp=np)(x32).dtype == np.float32


def test_torch_fft_and_stateful_filter():
    torch = pytest.importorskip("torch")

    @dp.transform(inverse=dp.array_api(torch.fft.ifft, xp=torch), parseval=True)
    @dp.array_api(xp=torch)
    def torch_fft(x):
        return torch.fft.fft(x)

    assert dp.verify(torch_fft, max_examples=30, seed=0, raise_on_failure=False).ok

    def one_pole(x, state):  # y[n] = x[n] + 0.5 y[n-1], state = last output
        ys, y = [], state[0]
        for v in x:
            y = v + 0.5 * y
            ys.append(y)
        out = torch.stack(ys) if ys else torch.zeros(0, dtype=x.dtype)
        return out, torch.reshape(y, (1,))

    block = dp.stateful(dp.array_api(one_pole, xp=torch), np.zeros(1))
    assert dp.verify(block, max_examples=30, seed=0, raise_on_failure=False).ok


def test_torch_conjugate_view_is_resolved():
    torch = pytest.importorskip("torch")
    z = torch.tensor([1 + 2j, 3 - 1j]).conj()
    np.testing.assert_array_equal(to_numpy(z), [1 - 2j, 3 + 1j])

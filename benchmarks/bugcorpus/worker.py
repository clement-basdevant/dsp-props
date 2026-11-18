"""Worker run inside an isolated environment holding an old library version.

The rediscovery campaign (``benchmarks/rediscovery.py``) runs dsp-props in the current
environment and the library under test in a separate interpreter, so that a release from 2018
can be tested with today's Hypothesis. This file is executed by that interpreter: it must stay
compatible with Python 3.7 and must not import dsp-props.

Protocol, over stdin/stdout, one message per frame (4-byte big-endian length, then a pickle of
plain Python objects, protocol 2):

* ``("define", name, source)``: evaluate ``source`` (an expression, usually a lambda) with the
  imports of ``("setup", code)`` in scope, and keep it under ``name``;
* ``("setup", code)``: execute ``code`` (imports, helper definitions) once;
* ``("call", name, args)``: call the function; ``args`` is a list; the answer is the result and
  the arguments the function modified in place, by position;
* ``("versions",)``: the versions of Python and of the imported libraries.

Arrays travel as ``{"__nd__": (dtype.str, shape, bytes)}``, not as pickled ndarrays: a NumPy 2
pickle names ``numpy._core``, which NumPy 1.x cannot load. Answers are ``("ok", value)`` or
``("error", exception type name, message)``.
"""

import pickle
import struct
import sys
import traceback

import numpy as np

NAMESPACE = {"np": np, "numpy": np}
FUNCTIONS = {}


def encode(value):
    if isinstance(value, np.ndarray):
        a = np.ascontiguousarray(value)
        return {"__nd__": (a.dtype.str, a.shape, a.tobytes())}
    if isinstance(value, np.generic):
        return encode(np.asarray(value))
    if isinstance(value, (list, tuple)):
        return type(value)(encode(v) for v in value)
    if isinstance(value, dict):
        return {k: encode(v) for k, v in value.items()}
    if hasattr(value, "numpy") and callable(value.numpy):  # torch tensors
        return encode(value.detach().cpu().numpy())
    return value


def decode(value):
    if isinstance(value, dict) and "__nd__" in value:
        dtype, shape, data = value["__nd__"]
        return np.frombuffer(data, dtype=np.dtype(dtype)).reshape(shape).copy()
    if isinstance(value, (list, tuple)):
        return type(value)(decode(v) for v in value)
    if isinstance(value, dict):
        return {k: decode(v) for k, v in value.items()}
    return value


def read_frame(stream):
    head = stream.read(4)
    if len(head) < 4:
        return None
    (size,) = struct.unpack(">I", head)
    return pickle.loads(stream.read(size))


def write_frame(stream, obj):
    data = pickle.dumps(obj, protocol=2)
    stream.write(struct.pack(">I", len(data)))
    stream.write(data)
    stream.flush()


def versions():
    out = {"python": sys.version.split()[0]}
    for name in ("numpy", "scipy", "librosa", "torch", "torchaudio"):
        module = sys.modules.get(name)
        if module is not None:
            out[name] = getattr(module, "__version__", "?")
    return out


def handle(message):
    kind = message[0]
    if kind == "setup":
        exec(message[1], NAMESPACE)
        return None
    if kind == "define":
        FUNCTIONS[message[1]] = eval(message[2], NAMESPACE)
        return None
    if kind == "call":
        args = decode(message[2])
        before = [a.copy() if isinstance(a, np.ndarray) else None for a in args]
        result = FUNCTIONS[message[1]](*args)
        # Arguments the function wrote into travel back, so that an in-place modification of the
        # caller's array (a bug in itself) is reproduced on the caller's side.
        written = {i: encode(a) for i, (a, b) in enumerate(zip(args, before))
                   if b is not None and (a.shape != b.shape or a.tobytes() != b.tobytes())} if args else {}
        return encode(result), written
    if kind == "versions":
        return versions()
    raise ValueError(f"unknown message {kind!r}")


def main():
    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer
    sys.stdout = sys.stderr  # stray prints must not corrupt the protocol
    while True:
        message = read_frame(stdin)
        if message is None:
            return
        try:
            write_frame(stdout, ("ok", handle(message)))
        except Exception as exc:  # every error goes back to the caller, as the operator's error
            detail = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            write_frame(stdout, ("error", type(exc).__name__, detail))


if __name__ == "__main__":
    main()

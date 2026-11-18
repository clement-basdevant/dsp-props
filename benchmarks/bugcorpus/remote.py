"""Run functions of an old library release in a separate interpreter (client side).

``Environment`` describes an isolated environment (a Python version and pinned wheels) and builds
it once with ``uv`` under a cache directory. ``Remote`` starts ``worker.py`` in it and turns
source strings into local callables: the arrays drawn by Hypothesis are sent to the old release
and its answer comes back as a NumPy array, so dsp-props declarations wrap these callables like
any other function.
"""

from __future__ import annotations

import hashlib
import os
import pathlib
import pickle
import shutil
import struct
import subprocess
from dataclasses import dataclass, field

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
WORKER = HERE / "worker.py"
CACHE = pathlib.Path(os.environ.get("DSP_PROPS_REDISCOVERY_ENVS",
                                    pathlib.Path.home() / ".cache" / "dsp-props" / "rediscovery"))


COMPILED = ("numpy", "scipy", "numba", "llvmlite", "scikit-learn", "torch", "torchaudio", "soundfile")


class RemoteError(RuntimeError):
    """An exception raised by the function under test, in the worker."""


@dataclass(frozen=True)
class Environment:
    python: str
    packages: tuple[str, ...]
    index_url: str | None = None  # e.g. the PyTorch CPU index
    extra: tuple[str, ...] = field(default_factory=tuple)  # extra ``uv pip install`` arguments

    @property
    def name(self) -> str:
        text = repr((self.python, self.packages, self.index_url, self.extra))
        label = "-".join(p.replace("==", "") for p in self.packages)[:60]
        return f"py{self.python}-{label}-{hashlib.sha1(text.encode()).hexdigest()[:8]}"

    @property
    def path(self) -> pathlib.Path:
        return CACHE / self.name

    @property
    def interpreter(self) -> pathlib.Path:
        return self.path / "bin" / "python"

    def ensure(self) -> pathlib.Path:
        """Build the environment if it does not exist yet; return its interpreter."""
        marker = self.path / ".complete"
        if marker.exists():
            return self.interpreter
        uv = shutil.which("uv")
        if uv is None:
            raise RuntimeError("the rediscovery campaign needs uv (https://docs.astral.sh/uv/)")
        if self.path.exists():
            shutil.rmtree(self.path)
        subprocess.run([uv, "venv", "-q", "--python", self.python, str(self.path)], check=True)
        # Compiled packages come from wheels only (a 2019 source build would not compile today);
        # pure-Python ones (resampy 0.2.2 has no wheel) may be built from their source archive.
        cmd = [uv, "pip", "install", "-q", "--python", str(self.interpreter)]
        for name in COMPILED:
            cmd += ["--only-binary", name]
        if self.index_url:
            cmd += ["--index-url", self.index_url, "--extra-index-url", "https://pypi.org/simple"]
        cmd += [*self.extra, *self.packages]
        subprocess.run(cmd, check=True)
        marker.write_text("ok\n")
        return self.interpreter


def _encode(value):
    if isinstance(value, np.ndarray):
        a = np.ascontiguousarray(value)
        return {"__nd__": (a.dtype.str, a.shape, a.tobytes())}
    if isinstance(value, np.generic):
        return _encode(np.asarray(value))
    if isinstance(value, (list, tuple)):
        return type(value)(_encode(v) for v in value)
    return value


def _decode(value):
    if isinstance(value, dict) and "__nd__" in value:
        dtype, shape, data = value["__nd__"]
        return np.frombuffer(data, dtype=np.dtype(dtype)).reshape(shape).copy()
    if isinstance(value, (list, tuple)):
        return type(value)(_decode(v) for v in value)
    return value


class Remote:
    """A worker process in ``env``; ``fn(source)`` returns a local callable."""

    def __init__(self, env: Environment, setup: str = ""):
        self.env = env
        self.proc = subprocess.Popen([str(env.ensure()), "-I", str(WORKER)], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, cwd=str(env.path))
        self._count = 0
        if setup:
            self._request(("setup", setup))
        self.versions = self._request(("versions",))

    def _request(self, message):
        data = pickle.dumps(message, protocol=2)
        assert self.proc.stdin is not None and self.proc.stdout is not None
        self.proc.stdin.write(struct.pack(">I", len(data)) + data)
        self.proc.stdin.flush()
        head = self.proc.stdout.read(4)
        if len(head) < 4:
            raise RuntimeError(f"worker in {self.env.name} died (exit code {self.proc.poll()})")
        (size,) = struct.unpack(">I", head)
        answer = pickle.loads(self.proc.stdout.read(size))
        if answer[0] == "error":
            raise RemoteError(answer[2])
        return answer[1]

    def setup(self, code: str):
        """Execute ``code`` (imports, helpers) in the worker's namespace."""
        self._request(("setup", code))

    def fn(self, source: str, name: str | None = None):
        """A local function that calls ``eval(source)`` in the worker."""
        self._count += 1
        key = f"f{self._count}"
        self._request(("define", key, source))

        def call(*args):
            result, written = self._request(("call", key, _encode(list(args))))
            for i, value in written.items():  # reproduce an in-place modification of an argument
                if isinstance(args[i], np.ndarray) and args[i].flags.writeable:
                    args[i][...] = _decode(value)
            return _decode(result)

        call.__name__ = call.__qualname__ = name or key
        call.source = source  # type: ignore[attr-defined]
        return call

    def close(self):
        if self.proc.poll() is None:
            assert self.proc.stdin is not None
            self.proc.stdin.close()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

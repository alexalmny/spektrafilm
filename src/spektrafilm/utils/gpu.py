"""GPU acceleration backend (optional CuPy / CUDA).

Off by default. When ``params.settings.gpu`` is true and CuPy with a
usable CUDA device is installed, supported stages can run on the GPU
in float64. All other stages, including large-sigma Gaussian filters,
remain on the original NumPy / Numba path.

Enablement travels through a :class:`contextvars.ContextVar` that
:class:`~spektrafilm.runtime.pipeline.SimulationPipeline.process` sets
for the duration of one render. Context vars are per-thread/per-task,
so concurrent renders cannot leak the flag into each other.

The GPU Gaussian path uses the CPU's small-sigma FIR kernel and reflect
boundary mode; device accumulation can still differ by floating-point
rounding. See ``tests/test_gpu.py`` for the CPU/GPU tolerance checks.
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from contextvars import ContextVar

import numpy as np

__all__ = [
    "cupy_module",
    "get_array_module",
    "use",
    "active",
    "as_device_array",
    "as_host_array",
]

_use_gpu: ContextVar[bool] = ContextVar("spektrafilm_use_gpu", default=False)

_cupy = None
_cupy_probed = False


def cupy_module():
    """Return the ``cupy`` module when a usable CUDA device answers, else None.

    Probes lazily and only once per process: importing cupy is expensive
    and pointless for users who never enable the flag.
    """
    global _cupy, _cupy_probed
    if not _cupy_probed:
        _cupy_probed = True
        try:
            import cupy as cp

            if cp.cuda.runtime.getDeviceCount() > 0:
                _cupy = cp
        except Exception:  # no CuPy, no CUDA runtime/driver, or no device
            _cupy = None
    return _cupy


def get_array_module(arr):
    """Return ``cupy`` when ``arr`` lives on the device, else ``numpy``.

    Detects cupy arrays by type without importing cupy, so the numpy
    path never pays for the probe.
    """
    if type(arr).__module__.startswith("cupy"):
        import cupy

        return cupy
    return np


@contextmanager
def use(enabled: bool):
    """Scope in which GPU dispatch is requested.

    Falls back to CPU (with a one-time warning) when the flag is set
    but CuPy/CUDA is unavailable, so enabling the setting can never
    break a render.
    """
    enabled = bool(enabled)
    if enabled and cupy_module() is None:
        warnings.warn(
            "settings.gpu=True but CuPy/CUDA is unavailable — rendering on CPU. "
            "Install a CuPy wheel matching your CUDA driver (e.g. pip install "
            "cupy-cuda12x).",
            RuntimeWarning,
        )
        enabled = False
    token = _use_gpu.set(enabled)
    try:
        yield
    finally:
        _use_gpu.reset(token)


def active() -> bool:
    """True when GPU dispatch is requested *and* usable."""
    return _use_gpu.get() and cupy_module() is not None


def as_device_array(arr):
    """Host → device copy (no-op for device arrays)."""
    cp = cupy_module()
    return arr if cp is not None and isinstance(arr, cp.ndarray) else cp.asarray(arr)


def as_host_array(arr):
    """Device → host copy (no-op for host arrays)."""
    cp = cupy_module()
    return cp.asnumpy(arr) if cp is not None and isinstance(arr, cp.ndarray) else arr

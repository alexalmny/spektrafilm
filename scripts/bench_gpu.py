"""Benchmark CPU vs GPU rendering.

Run on a machine with a CUDA GPU and CuPy installed:

    pip install cupy-cuda12x   # wheel matching your driver
    python scripts/bench_gpu.py

Prints wall-clock times for identical renders with settings.gpu=False
and True, plus per-stage timings from the last GPU run.
"""
import time

import numpy as np

from spektrafilm.runtime import init_params, simulate
from spektrafilm.utils import gpu


def synthetic_image(h=1024, w=1536, seed=42):
    rng = np.random.default_rng(seed)
    xx = np.linspace(0, 1, w)[None, :]
    yy = np.linspace(0, 1, h)[:, None]
    img = np.stack([
        0.2 + 0.5 * xx * np.ones_like(yy) + 0.02 * rng.standard_normal((h, w)),
        0.25 + 0.4 * yy * np.ones_like(yy)[::-1] + 0.02 * rng.standard_normal((h, w)),
        0.3 + 0.3 * ((xx + yy) / 2) + 0.02 * rng.standard_normal((h, w)),
    ], axis=-1)
    return np.clip(img, 0, 1)


def build_params():
    params = init_params(print_profile="kodak_portra_endura")
    params.io.upscale_factor = 1.0
    params.io.scan_film = False
    params.camera.auto_exposure = True
    params.settings.use_fast_stats = True
    params.settings.use_enlarger_lut = True
    params.settings.use_scanner_lut = True
    params.settings.lut_resolution = 17
    params.debug.deactivate_stochastic_effects = False
    return params


def timed(image, params, runs=2):
    best = float("inf")
    out = None
    for _ in range(runs):
        t0 = time.perf_counter()
        out = simulate(image, params)
        best = min(best, time.perf_counter() - t0)
    return best, out


def main():
    if gpu.cupy_module() is None:
        print("No usable CUDA device / CuPy — benchmarking CPU only.")
    image = synthetic_image()
    params = build_params()

    simulate(image[:64, :64].copy(), params)  # warm numba (and cupy JIT below)

    params.settings.gpu = False
    t_cpu, out_cpu = timed(image, params)
    print(f"CPU (settings.gpu=False): {t_cpu:.2f} s")

    params.settings.gpu = True
    simulate(image[:256, :256].copy(), params)  # warm cupy kernel compiles
    t_gpu, out_gpu = timed(image, params)
    print(f"GPU (settings.gpu=True):  {t_gpu:.2f} s")
    if gpu.cupy_module() is not None:
        print(f"speedup: {t_cpu / t_gpu:.2f}x")
        d = np.abs(out_gpu - out_cpu)
        print(f"max |gpu-cpu| = {d.max():.3e}, mean = {d.mean():.3e}")


if __name__ == "__main__":
    main()

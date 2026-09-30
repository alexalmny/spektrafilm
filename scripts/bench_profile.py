import numpy as np
import time
import json
import sys
from spektrafilm.runtime import init_params, simulate

# Deterministic synthetic image (no external asset dependency)
rng = np.random.default_rng(42)
H, W = 1024, 1536
# smooth-ish image with gradients + noise
xx = np.linspace(0, 1, W)[None, :]
yy = np.linspace(0, 1, H)[:, None]
img = np.stack([
    0.2 + 0.5 * xx * np.ones_like(yy) + 0.02 * rng.standard_normal((H, W)),
    0.25 + 0.4 * yy * np.ones_like(yy)[::-1] + 0.02 * rng.standard_normal((H, W)),
    0.3 + 0.3 * ((xx + yy) / 2) + 0.02 * rng.standard_normal((H, W)),
], axis=-1)
img = np.clip(img, 0, 1)

params = init_params(print_profile='kodak_portra_endura')
params.io.upscale_factor = 1.0
params.io.scan_film = False
params.camera.auto_exposure = True
params.settings.use_fast_stats = True
params.settings.use_enlarger_lut = True
params.settings.use_scanner_lut = True
params.settings.lut_resolution = 17
params.debug.deactivate_stochastic_effects = False
params.debug.print_timings = False

# warmup numba
_ = simulate(img[:64, :64].copy(), params)
print("warmup done", flush=True)

# timed runs
for run in range(2):
    t0 = time.perf_counter()
    out = simulate(img, params, print_timings=(run == 1))
    t1 = time.perf_counter()
    print(f"run {run}: total {t1-t0:.2f}s", flush=True)

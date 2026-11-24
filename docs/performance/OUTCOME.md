# Performance roadmap outcome — Windows / Ryzen 7 5800X / RTX 3070

**Decision: keep the FP64 CPU and GUI changes. Do not integrate CUDA.** `performance-roadmap` is the delivery branch; the CuPy experiment remains on `performance-roadmap-cuda` (`8b7a09c` for the prototype), with no new GPU dependency in the normal install. The CPU passes both warm targets. GPU FP32 passes the measured color and 45 Mpx tests, but misses the 24 Mpx export and interactive preview speed gates.

## Follow-up: fused CPU grain sampling

`e669bf3` fuses the existing FP64 Poisson and binomial fast-stat paths per layer, removing intermediate arrays without changing the sampling algorithms or grain settings. Against the optimized CPU reference below, at 8 Numba threads the default print ran in 10.52 instead of 11.36 s at 6 Mpx, 39.30 instead of 42.69 s at 24 Mpx, and 77.38 instead of 84.77 s at 45 Mpx (two warm samples for the last candidate case). At 45 Mpx, peak process commit was 10.85 instead of 10.99 GiB, both full frames were finite, and the largest mean encoded RGB difference across channels was below 0.00002 with independent random draws. On repeated neutral patches at the 24 Mpx pixel pitch, grain RMS differed by at most 0.20% per channel.

Validation also covered a 32.95 Mpx Sony ARW rendered as a Portra 400 negative (51.24 s on the one warm pass, and a re-opened 16-bit RGB TIFF), and Ektachrome 100 reversal film scanned directly at 6 Mpx. The full suite showed **874 passed and the same 3 inherited print snapshot failures**. No source photos or output images are published. The original CPU and GPU measurements below predate this follow-up.

## CPU and GUI

Same Windows host, CPython 3.13.14, pinned dependencies, eight Numba threads, one BLAS thread, same actual sizes, stock/print and default effects. Drive cache and OS cache are not assumed cold. “First” below means a new process with an empty Numba disk-cache directory. The warm number is the median ± median absolute deviation over 10 previews or 3 scans in one simulator. Commit is the Windows process high-water commit, before display and export.

| Workload | First, before → CPU (s) | Warm, before → CPU (s) | Peak commit, before → CPU (GiB) |
|---|---:|---:|---:|
| 640 × 426 preview | 6.761 → 8.041 | 0.640 ± 0.009 → **0.333 ± 0.008** | 0.61 → 0.44 |
| 1280 × 853 preview | 9.820 → 9.807 | 2.466 ± 0.034 → 1.179 ± 0.018 | 1.06 → 0.62 |
| 6 Mpx print | 28.172 → 21.938 | 18.738 ± 0.004 → 11.313 ± 0.007 | 4.10 → 1.83 |
| 6 Mpx negative | 26.483 → 20.306 | 17.640 ± 0.037 → 10.429 ± 0.001 | 3.83 → 1.82 |
| 24 Mpx print | 81.882 → 54.385 | 72.605 ± 0.016 → **42.884 ± 0.052** | 15.19 → **6.06** |
| 45 Mpx print | 159.851 → 94.249 | 147.621 ± 0.906 → 84.774 ± 0.042 | 28.12 → 10.99 |
| Native Sony RAW (32.95 Mpx) | 112.712 → 70.509 | 103.297 ± 0.316 → 59.881 ± 0.022 | 21.06 → 8.54 |

The warm 24 Mpx scan takes **41% less time** and **60% less peak commit**. The copy/lifetime patch alone took 73.327 s at 13.04 GiB peak commit; it earned its place on RAM, not speed. The 24 Mpx CAM16-UCS median fell from 31.205 to 1.717 s. The first disk-cold 640 engine run took 1.28 s longer; the extra JIT compile is not a warm-regression claim. The preview sweep at 2 / 4 / 8 / 16 Numba threads produced 0.394 / 0.320 / 0.333 / 0.377 s; this is not enough to choose a global thread default, so the project default is unchanged.

A matched cache-control rerun gives first-pass 640 at 6.674 → 7.774 s with an empty Numba disk cache, and 1.090 → 0.820 s in new processes with those caches populated. Warm medians gated on exposure / filter / stock / alternating scan mode are 0.650 / 0.642 / 0.649 / 0.612 s before and 0.330 / 0.319 / 0.323 / 0.300 s after.

The separate 24 Mpx monitor transform, export conversion, TIFF write and EXIF copy took 5.527 / 4.897 / 0.653 / 0.597 s on CPU, versus 5.777 / 4.950 / 0.728 / 0.604 s before. The lazy fallback avoids one output clip/quantize, but does not remove the ICC transform. Input decode, ICC and file I/O were not moved to new workers. Grain, halation, PCHIP, and the presets' size/effects were not reduced to obtain these gains. After the CPU changes, 24 Mpx grain, halation, couplers and density interpolation cost 12.318 / 5.190 / 3.918 / 0.159 s respectively. At 640, preparing both 17³ PCHIP tables spent ~28 ms in the monotonicity diagnostic and just 0.55 ms in slopes and cell limits; no extra slope cache was added for that sub-millisecond computation.

### Actual GUI response

These comparisons reran the same six ordered edits, scheduled 25 ms apart, on all three backends with the same harness. A real VisPy draw is observed, not physical panel presentation. The 1000 × 667 source TIFF is unchanged. All edit columns summarize ten bursts per process. Each CPU or CUDA burst starts/publishes one render, down from two before. Open includes normal app warmup and animation; its disk-cold figure is one observation per backend, not a tail estimate.

| GUI metric | Before | CPU | CUDA FP32 |
|---|---:|---:|---:|
| Disk-cold open to first draw | 12.809 s | **14.187 s** | 12.857 s |
| Disk-warm open to first draw | 2.444 s | 2.177 s | 2.267 s |
| Warm last edit to first draw, median | 1.409 s | **0.586 s** | 0.567 s |
| Warm last edit to settled draw, median ± MAD | 1.834 ± 0.018 s | **1.016 ± 0.017 s** | 0.998 ± 0.022 s |
| Warm manual scan to settled draw | 4.561 s | 3.779 s | 3.791 s |
| Save after scan | 0.052 s | 0.050 s | 0.051 s |

CPU edits reach the first draw **2.4× sooner** and the settled draw **1.8× sooner**. The startup trade-off is explicitly separate. CUDA FP32 changes the settled edit median by ~2% versus the optimized CPU, not the required 20%. The initial timings in [the baseline README](README.md) used six independent timers; the table above uses the matched rerun for all branches.

## CUDA test and decision

CuPy 14.2.0 `cupy-cuda12x[ctk]`, CUDA runtime and NVRTC 12.9, driver 610.62, RTX 3070 / 8 GiB. `cupy.show_config()`, a standalone FP64 allocation and kernel smoke, and the actual CAM16 kernel all ran without a system CUDA toolkit. CuPy warns that no CUDA root was detected; the bundled runtime and NVRTC worked. The prototype uploads the boundary table and fixed transforms once per active configuration and transfers one RGB buffer in each direction per image/band. No package import or RTC compile is required for the normal CPU path.

| 24 Mpx, warm | Optimized CPU | CUDA FP64 | CUDA FP32 |
|---|---:|---:|---:|
| Entire engine, median | 42.884 s | 41.266 s | 40.983 s |
| Estimated process + real export phases | 49.031 s | 47.482 s | 47.400 s |
| CAM16-UCS, host total | 1.717 s | 0.891 s | 0.572 s |
| Warm host pack/cast | — | <0.001 s | 0.108 s |
| Warm H→D / kernel / D→H (CUDA events) | — | 0.199 / 0.372 / 0.063 s | 0.100 / 0.045 / 0.088 s |
| Cold device init / first kernel compile or module load | — | 0.261 / 0.288 s | 0.268 / 0.276 s |
| Peak device used, including desktop / CuPy pool | — | 2.13 / 1.07 GiB | 1.60 / 0.54 GiB |
| Peak process commit | 6.06 GiB | 8.61 GiB | 7.58 GiB |

The export estimate adds measured conversion, TIFF write and EXIF copy to the warm process median; I/O is sampled on the final output, not a median or an optimized GUI scan at 24 Mpx. The engine itself is only **1.046× faster** in FP32 and including I/O gives about **1.034×**, both well below 1.3×. Once the fused CPU implementation is in place, even removing all of its 1.717 s CAM16 time could not yield 1.3× on this scan. That argues against expanding this port. Timings in a row are independently summarized and not necessarily additive; CUDA events are synchronized on the working stream, and sampled VRAM high-water includes non-CuPy device usage (not a continuous GPU trace).

The 45 Mpx FP32 scan completed three warm executions (median 82.172 s; CPU 84.774 s), with 12 bands of at most 4 million pixels and zero nonfinite kernel pixels. Peak device use was 1.15 GiB including the desktop, of which ~0.09 GiB was reserved by the CuPy pool. Test fixtures require banded/full equality in both precisions across three output spaces. An actual forced zero-budget run emitted the CPU fallback warning and its 640 reference matched the CPU exactly. Unit tests separately check the unavailable device/memory path and that unexpected kernel errors propagate rather than silently retry on CPU. An actual 8 GiB exhaustion was not induced.

## Color and file validation

The deterministic portrait references compare the original commit's current-default Colour output with the revised pipeline. Encoded sRGB is decoded and adapted to D65 Lab for ΔE00; the report separately checks full-image linear RGB, out-of-range values, new clip counts and finite masks. On this portrait all pixels were displayable. The mire tests additionally cover black, neutral ramps, saturated primaries, HDR, negatives and nonfinite/singular input; the FP64 print and negative pipeline checks cover density taps and `rgb_out` on Gold/Endura, Fuji/Crystal Archive and Portra/Endura.

| Deterministic comparison | Max abs linear RGB | ΔE00 p95 | ΔE00 p99 | Worst ΔE00 |
|---|---:|---:|---:|---:|
| Original → CPU, 640 | 1.04e−14 | 1.34e−13 | 2.43e−13 | 9.73e−13 |
| Original → CPU, 24 Mpx | 1.81e−14 | 1.92e−13 | 3.52e−13 | 1.86e−12 |
| CPU → CUDA FP64, 640 | 7.22e−15 | 1.09e−13 | 1.86e−13 | 7.32e−13 |
| CPU → CUDA FP32, 640 | 3.81e−6 | 0.0000654 | 0.000119 | 0.000405 |
| CPU → CUDA FP32, 24 Mpx | 7.14e−6 | 0.0000972 | 0.000178 | 0.000765 |

No new clips, NaNs, or infinities on these frames. FP32 stays below the predeclared ΔE00 limits (p95 ≤ 0.5, p99 ≤ 1.0). For the repo color chart and a second photograph decoded from native RAW, the CPU had maximum linear errors of 6.44e−15 and 7.66e−15; CUDA FP32 p99 ΔE00 was 0.000145 and 0.000140, respectively. Side-by-side inspection of those two fixtures, the deterministic 640 and 24 Mpx portrait, and a 3× nearest-neighbor crop around the latter's worst pixel `(4581, 672)` showed no visible color shifts, halos, or added banding at those views. This was not a blinded study or exhaustive inspection of every saturated image or 45 Mpx band join.

For stochastic validation, eight independent renders per effect and backend used a flat linear ProPhoto neutral at 0.18, sampling the 192² interior of a 256² Gold 200 / Supra Endura print. Grain and glare were enabled separately, each measured against that backend's noise-free control. The mean residual standard deviation per channel changed by at most 0.43% for grain and 0.66% for glare versus baseline. The largest differences among the three radial FFT power bands were 7.2% (grain, smallest low-frequency band) and 1.4% (glare); mean linear RGB differed by less than 0.003% across all channels. This characterizes these repeated neutral samples, not every profile or a guarantee of identical random draws.

The actual 24 Mpx TIFF exports from baseline, CPU, GPU FP64 and GPU FP32 all have 6000 × 4000 RGB uint16 pixels, the original encoded ProPhoto ICC bytes, EXIF orientation 1, the correct color-space tag, software and dimensions. The TIFF round-trip test independently checks pixel quantization, unchanged ICC after EXIF copy, original camera/lens tags and XMP. The EXR test reads half-float values back, including negative and HDR channels, and checks the existing no-ICC/EXIF behavior. No saver or metadata-writing production code was changed.

There were no edits to the profiles, preset assets, formats, or public API for speed; the LUT bug fix changes stale-cache behavior to match a fresh simulator. The code remains GPLv3, and no profile or LUT artifacts are redistributed here. Original profiles by Andrea Volpato, [upstream Spektrafilm](https://github.com/andreavolpato/spektrafilm), remain covered by [CC BY-SA 4.0 with attribution](../../SPEKTRAFILM_LICENSE.txt).

## Verification and raw files

The CPU full suite ended with **869 passed, 3 inherited failures**, all the previously diagnosed stale print snapshots; the `.npz` files and their tolerances are unchanged. [Baseline history and exceptions](README.md#inherited-tests) identifies the two intentional color changes and the older dependency stack tested. The focused CPU suite had **224 passed**; actual CUDA parity, banding and fallback tests had **8 passed** on the RTX 3070. A built CUDA-branch wheel contains `spektrafilm/data/cuda/cam16ucs.cu`; no wheel or binary is committed. See the [test summary](results/test-summary.json) and [TIFF validation](results/tiff-validation.json).

- [Original engine and GUI](results/before/), [copy/lifetime patch only](results/after-copy/), [optimized CPU and photo parity](results/after-cpu/), [CUDA and parity](results/after-cuda/).
- [Matched GUI: disk-cold and disk-warm](results/gui-matched/), [thread/cache/parameter diagnostics](results/variants/), [additional chart/RAW references](results/color-fixtures/), [noise samples](results/noise/).
- The raw JSON records the settings, image hashes, all repetitions and memory samples. The `.npy`, `.tif`, private RAW and inspection images remain local under `.venv/audit/`; the report does not rely on embedding a photo in Git. The portable fixtures are in `img/`; use your own camera RAW for the native run.

## Reproduce the comparisons

Use the PowerShell environment and standard case matrix in [README.md](README.md) on each branch/revision. `af1ede2` provides the baseline runner with `3bb2c2d` renderer code; `420d860` isolates the copy/lifetime change on top of the correctness fixes. All other production figures are after `c4d0df0` and `e2759e9`. For the matched GUI measurements on the baseline, copy only `scripts/gui_response_timing.py` from `5109b45` into the baseline worktree. Run it twice on each revision with the same initially empty `NUMBA_CACHE_DIR`, different output paths and the interactive desktop visible. No need to copy application files or change personal settings.

Deterministic photo comparisons can be reproduced by running the same cases with `--deterministic --reference` in the two worktrees, followed by:

```powershell
& $python scripts/compare_render_references.py .venv/audit/before/reference-24mp.json .venv/audit/after-cpu/reference-24mp.json --json .venv/audit/parity-24mp.json
& $python scripts/render_noise_audit.py --repeats 8 --json .venv/audit/noise/cpu.json
```

For baseline noise, run the same noise script with `PYTHONPATH` set to the `af1ede2` worktree's `src` (use an absolute path, then remove the environment override). For additional photos, decode the RAW with `load_and_process_raw_file(..., output_colorspace='ProPhoto RGB')`; for the `cc07.png` chart, decode sRGB and convert to linear ProPhoto RGB. Resize the decoded image to long edge 640 and save as 32-bit linear ProPhoto TIFF with `save_image_oiio`. Pass the same file to both benchmark commands with `preview640 --image <fixture.tif> --deterministic --reference`. These are quality probes; one warm sample suffices to retain a reference, not to infer throughput.

For CUDA, enter a checkout of `performance-roadmap-cuda`, create a separate `uv venv --python 3.13.14 .venv-cuda` and use its Python. Install the full [CUDA freeze](requirements-win-py313-cuda.txt) with `uv pip install --python $gpuPython -r <absolute-path-to-freeze>`, then install the CUDA checkout with `uv pip install --python $gpuPython --no-deps -e '.[dev]'`. From that checkout, for example:

```powershell
$gpuPython = '.venv-cuda\Scripts\python.exe'
& $gpuPython -c 'import cupy; cupy.show_config()'
& $gpuPython -m pytest -q tests/test_cam16ucs_cuda.py
$env:NUMBA_CACHE_DIR = '.venv-cuda/audit/cache/numba/24mp-fp32'
$env:CUPY_CACHE_DIR = '.venv-cuda/audit/cache/cupy/24mp-fp32'
& $gpuPython scripts/runtime_process_timing.py 24mp --cam16-backend cuda-fp32 --display --export --reference --json .venv-cuda/audit/24mp-fp32.json
& $gpuPython scripts/runtime_process_timing.py 45mp --cam16-backend cuda-fp32 --cuda-band-pixels 4000000 --reference --json .venv-cuda/audit/45mp-fp32.json
& $gpuPython scripts/runtime_process_timing.py preview640 --cam16-backend cuda-fp32 --cuda-band-pixels 0 --deterministic --reference --json .venv-cuda/audit/forced-fallback.json
```

Choose unused explicit cache directories per cold case for both Numba and CuPy. For FP64, use `--cam16-backend cuda-fp64`; for CUDA GUI or noise timing, set `SPEKTRAFILM_EXPERIMENTAL_CAM16=cuda-fp32` in that process's environment, run the same harness, then remove the variable. Headless options apply only to the benchmark process; the production branch has no such backend. The forced fallback is a reproducible budget refusal, not destructive VRAM exhaustion.

# Spektrafilm performance measurements

Scope and gates: [performance roadmap](../PERFORMANCE_ROADMAP.md). Results and decision: [CPU/GUI and CUDA outcome](OUTCOME.md). Renderer baseline: `3bb2c2d`; the benchmark commit changes no runtime or profiles.

## Reproduce on Windows PowerShell

From the repository root with `uv` installed:

```powershell
uv venv --python 3.13.14 .venv
$python = '.venv\Scripts\python.exe'
uv pip install --python $python -r docs/performance/requirements-win-py313.txt
uv pip install --python $python --no-deps -e '.[dev]'
$env:QT_QPA_PLATFORM = 'offscreen'
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:NUMBA_NUM_THREADS = '8'
& $python -m pytest -q
```

One fresh process per case. Give each invocation an unused explicit Numba cache directory for disk-cold measurements. Run the same command with the same cache path and a different JSON path for a disk-warm first sample. This controls Numba's disk cache, not the OS file cache; the first engine sample is not application startup. Each process then reuses one `Simulator` for at least 10 warm previews or 3 warm large scans. Preview repetitions call `update_params` even without an edit; large scans keep unchanged parameters unless `--update` or a variant is requested.

```powershell
$run = '.venv/audit/before'
foreach ($case in 'preview640', 'preview1280', '6mp', '24mp', '45mp') {
    $env:NUMBA_CACHE_DIR = "$run/cache/$case"
    & $python scripts/runtime_process_timing.py $case --reference --json "$run/$case.json"
    if ($LASTEXITCODE) { throw "Benchmark failed: $case" }
}
$env:NUMBA_CACHE_DIR = "$run/cache/raw"
& $python scripts/runtime_process_timing.py raw --raw 'C:\path\to\fixture.ARW' --reference --json "$run/raw.json"
```

Use `--mode negative`, `--mode alternate`, and `--change exposure|filter|stock` to benchmark unchanged versus alternating settings. Use `--grain-off` and `--halation-off` on a scan with both effects active for one-variable comparisons. `--deterministic` disables grain and glare for pixel references, **not** for the default throughput measurements. `--display` measures ICC conversion separately and records whether an actual display profile was available. `--export` separately times conversion to encoded ProPhoto RGB, TIFF write, and EXIF copy. These are diagnostic phases, not GUI latency. Use `--numba-threads` and `--blas-threads` for controlled sensitivity checks.

The included portrait is a linear ProPhoto RGB TIFF; large scans are synthetic resamples of it. The 1280 case explicitly resamples to that long edge rather than benchmarking the original 1000-pixel image. `raw` preserves the camera file's decoded size and uses the production RAW loader (as-shot white balance, lens correction off, linear ProPhoto RGB). Do not commit private RAW files or their rendered images. Keep the local `.npy` output for comparison; JSON records the last variant, hash, shape, dtype, and effective settings. Random-on references are for visual inspection, not pixel parity. Windows working-set and commit peaks are OS process high-water marks before display/export; the per-sample series is cumulative. Stage and nested timings overlap; do not add them together. Warm spread is median absolute deviation (MAD), with min/max and all samples recorded. This sample count does not establish a p95.

## Baseline results

[Raw JSON by case](results/before/) includes all trials, dimensions, effective options and reference-image checksums. The local `.npy` references and TIFF exports are in `.venv/audit/before/` and `.venv/audit/gui/`; they are not committed. Lower is better for all timings below.

| CPU case | First engine pass, disk cache empty | Warm median ± MAD (n) | Peak working set / commit |
|---|---:|---:|---:|
| 640 × 426 preview | 6.761 s | 0.640 ± 0.009 s (10) | 0.63 / 0.61 GiB |
| 1280 × 853 preview | 9.820 s | 2.466 ± 0.034 s (10) | 1.08 / 1.06 GiB |
| 6 Mpx print | 28.172 s | 18.738 ± 0.004 s (3) | 4.12 / 4.10 GiB |
| 6 Mpx negative | 26.483 s | 17.640 ± 0.037 s (3) | 3.84 / 3.83 GiB |
| 24 Mpx print | 81.882 s | 72.605 ± 0.016 s (3) | 15.18 / 15.19 GiB |
| 45 Mpx stress print | 159.851 s | 147.621 ± 0.906 s (3) | 27.11 / 28.12 GiB |
| Native Sony RAW, 32.95 Mpx | 112.712 s | 103.297 ± 0.316 s (3) | 21.05 / 21.06 GiB |

RAW decode/demosaic and conversion to linear ProPhoto RGB took 4.639 s separately. At 24 Mpx, the warm CAM16-UCS median was 31.205 s (11.519 s forward and 11.703 s inverse, nested), and grain alone took 12.280 s. At 6 Mpx, the diagnostics were 14.520 s with grain off and 17.445 s with halation off, versus 18.738 s with both on. These settings are not used for the speed comparison. At 24 Mpx, separate display conversion took 5.777 s with the installed ICC profile active; converting to encoded ProPhoto RGB for export took 4.950 s, writing TIFF 0.728 s, and EXIF read/write 0.604 s.

### Real GUI timing

Run on the interactive Windows desktop (not Qt offscreen). The harness uses project settings without reading/saving personal defaults, opens the real viewer, sends six exposure edits 25 ms apart, waits for the real VisPy draw with each requested setting, and measures a manual scan and normal TIFF save with only the chooser stubbed. The first visible draw and the draw after animation/crossfade finishes are reported separately.

```powershell
Remove-Item Env:QT_QPA_PLATFORM -ErrorAction SilentlyContinue
$env:NUMBA_CACHE_DIR = '.venv/audit/gui/cache'
& $python scripts/gui_response_timing.py --warm 1 --json .venv/audit/gui/cold.json
& $python scripts/gui_response_timing.py --json .venv/audit/gui/warm.json
```

This is the repository TIFF at full native resolution for the GUI scan (1000 × 667), not the 24 Mpx engine benchmark. Time from open to the first preview draw was **13.06 s disk-cold / 2.43 s disk-warm**; after the opening animation it was 15.28 / 4.56 s. For the ten warm bursts, the last edit to first draw median was **1.479 s**; the settled draw took **1.920 ± 0.032 s**. The baseline rendered and published twice in every burst. Disk-warm GUI manual scan to draw was 4.666 s and the separate save took 0.052 s. The cold GUI sample includes the normal simultaneous startup warmup; it is not a cold engine sample.

## Inherited tests

The unmodified checkout reproduced **829 passed, 5 failed** under Qt offscreen. The layout test double (container, stretch and action bar) and the `lut_mode` assert on the nonexistent `GrainParams.mult_usm_amount` were test-only fixes. Grain remains covered by the original disabled-effect and deterministic-render assertions.

Three inherited print snapshots remain open **pending approval to rebaseline**. They predate two intentional rendering changes:

- `cd254ba` (parent of `6ec371b`): all five snapshots pass, both with the pinned environment and with NumPy 2.4.6 / SciPy 1.17.1 / Numba 0.64.0 / colour-science 0.4.6, on the same Python 3.13.14.
- `6ec371b`: changing the lightness roll-off from two-sided to black-anchored makes the two neutral print ramps divergent. Maximum encoded RGB errors are 0.007131 (Portra/Endura) and 0.008971 (Fuji/Crystal Archive); the green patch still passes.
- `6deeec3` changes the default perceptual gamut algorithm from OkLCh to CAM16-UCS. At `83278a5` and `3bb2c2d`, the same three cases fail with respective maxima 0.007131, 0.009070, and 0.05176. Forcing only OkLCh in a diagnostic run at `3bb2c2d` restores the green snapshot and reproduces exactly the two earlier ramp errors (including their mean errors). The older dependencies reproduce all three failures at `3bb2c2d` as well.

The stored `.npz` files, their tolerance, and the current default renderer are unchanged. Reproduce the matrix above with `python -m pytest -q tests/test_regression_baselines.py --tb=short` on the named revisions in a clean worktree. The two pre-scan snapshots still pass. The performance parity checks use deterministic `3bb2c2d` output with the current CAM16-UCS and lightness settings, not stale `.npz` print pixels.

The seven new regressions for the spectral cache fail on unmodified `3bb2c2d` and pass with the fixes. They cover resolution changes in both directions, individual min/max bound changes (in-place as well as through parameter updates), an intermediate tap, stock changes, and alternation between negative and print against a fresh simulator.

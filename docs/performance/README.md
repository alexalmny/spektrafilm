# Spektrafilm performance measurements

Scope and gates: [performance roadmap](../PERFORMANCE_ROADMAP.md). Renderer baseline: `3bb2c2d`; the benchmark commit changes no runtime or profiles.

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

## Inherited tests

The unmodified checkout reproduced **829 passed, 5 failed** under Qt offscreen with the environment above before benchmark changes. The failures are a layout test double with a stale `addWidget` signature, a `lut_mode` test reading the absent `GrainParams.mult_usm_amount`, and the three print snapshots `print_rgb_portra_endura_gray_ramp16`, `print_rgb_fuji_crystal_gray_ramp16`, and `print_rgb_portra_endura_green_patch8`. The snapshots and their tolerances have not been changed; their cause still needs verification.

# CPU grain: binomial underflow experiment

`525d694` is an experiment on top of the fused CPU grain sampler already deployed on `main` (`9f5f22f`). In the inverse-CDF path, `(1 - p) ** n` can round to zero before iteration starts. For a positive uniform draw the old recurrence then spends `n + 1` iterations adding zero and returns `n`. The candidate consumes the same uniform draw and returns that result immediately; it does not correct the old approximation. It also hoists the fixed probability ratio out of the loop.

A spatial stride-eight sample of the default Gold 200 / Supra Endura 6 Mpx portrait found 13,768 zero-start inversions out of 843,750 layer pixels (1.63%). Sampling used independently drawn Poisson counts at those pixels, so this estimates branch incidence, not exact runtime events. The affected samples alone represent about 39.5 million inert iterations on the sampled grid.

Measurements used 8 Numba threads and identical fixture, effective parameters and dependency versions. The 6 Mpx candidate JSON records the deployed HEAD with a dirty worktree containing exactly the tested `525d694` patch; the 24 Mpx case was measured after the local commit.

| Size | Deployed CPU → candidate engine | Fused particle sampling | Peak process commit |
|---|---:|---:|---:|
| 6 Mpx, three warm repeats each | 10.658 → 8.536 s (−19.9%) | 2.607 → 0.567 s | 1.68 → 1.77 GiB |
| 24 Mpx, three reference and two candidate repeats | 39.301 → 37.097 s (−5.6%) | 5.990 → 3.535 s | 5.92 → 5.92 GiB |

First-pass times are not comparable because the two disk caches had different contents. Raw measurements: [results/binomial](results/binomial/).

With one Numba thread and the same explicitly seeded generator, 176,000 direct binomial draws across all branches and 10,000 FP64 fused particle draws on strided input matched bit-for-bit. The following 100 draws matched as well. The regression test checks that the zero-start shortcut still consumes exactly one uniform draw. The full candidate suite had 875 passes and only the 3 inherited print-snapshot failures. No new binary or source photo is tracked.

## Promotion checks

| Case | Warm engine | Fused particle sampling | Peak process commit |
|---|---:|---:|---:|
| 45 Mpx Gold 200 / Supra Endura print (2 baseline, 3 optimized repeats) | 77.383 → 75.666 s (−2.2%) | 12.773 → 10.664 s | 10.85 → 11.00 GiB |
| 32.95 Mpx Sony RAW, Ektachrome 100 direct scan (2 repeats each) | 66.069 → 50.953 s (−22.9%) | 23.733 → 8.867 s | 8.52 → 8.41 GiB |

The 45 Mpx baseline uses `531a557` (same renderer as deployed `9f5f22f`); the RAW baseline and candidate were rerun in fresh processes. Inputs, settings, packages and threads were matched, and the first pass is excluded. All four saved FP64 frames were finite; maximum absolute frame mean RGB delta was below `1.75e-5` over independent noise draws. Both Sony TIFFs reopened as 7028 × 4688 RGB uint16 with the expected encoded ProPhoto ICC and EXIF orientation/dimensions. The full candidate suite again passed 875 tests with only the three inherited snapshots failing. No private images are tracked. These checks gate promotion of `525d694` to `main`.

To reproduce timing, use `scripts/runtime_process_timing.py 6mp --json <file>` (or `24mp --warm 2` / `45mp --warm 3`) in the pinned environment described in [README.md](README.md). For the Sony case use `raw --mode negative --warm 2 --reference --export --raw <file>` and select `kodak_ektachrome_100` when initializing the film in `params_for`. The release artifacts and checks are under [results/binomial](results/binomial/).

"""Measure grain and glare separately on repeated full-effect neutral print scans."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


def noise_metrics(image, control):
    import numpy as np

    if image.shape != control.shape or not np.isfinite(image).all() or not np.isfinite(control).all():
        raise ValueError('noise audit requires matching finite linear RGB images')
    residual = image - control
    centered = residual - residual.mean(axis=(0, 1))
    spectrum = np.abs(np.fft.fft2(centered, axes=(0, 1)) / centered.shape[0] / centered.shape[1]) ** 2
    fy, fx = (np.fft.fftfreq(n) for n in centered.shape[:2])
    radius = np.hypot(fy[:, None], fx[None, :])
    bands = [(0, 0.0625), (0.0625, 0.25), (0.25, 0.75)]
    return {
        'mean_linear_rgb': image.mean(axis=(0, 1)).tolist(),
        'mean_residual': residual.mean(axis=(0, 1)).tolist(),
        'std_residual': centered.std(axis=(0, 1)).tolist(),
        'radial_power': {f'({lo}, {hi}]': spectrum[(radius > lo) & (radius <= hi)].sum(axis=0).tolist()
                         for lo, hi in bands},
    }


def run(repeats):
    from runtime_process_timing import configure_threads

    configure_threads(1, 8)
    import numpy as np
    import spektrafilm
    from spektrafilm.runtime import Simulator, digest_params, init_params

    p = init_params(film_profile='kodak_gold_200', print_profile='kodak_supra_endura')
    p.io.input_color_space = 'ProPhoto RGB'
    p.io.input_cctf_decoding = p.io.output_cctf_encoding = p.camera.auto_exposure = False
    p.settings.use_scanner_lut = p.settings.use_enlarger_lut = p.settings.use_fast_stats = True
    p.settings.lut_resolution = 17
    p = digest_params(p)
    fixture = np.full((256, 256, 3), 0.18, dtype=np.float32)
    crop = np.s_[32:-32, 32:-32]
    cases = {}
    for grain, glare in [(False, False), (True, False), (False, True)]:
        p.film_render.grain.active, p.print_render.glare.active = grain, glare
        sim = Simulator(p)
        if not grain and not glare:
            control = sim.process(fixture)[crop].copy()
            continue
        label = 'grain' if grain else 'glare'
        cases[label] = [noise_metrics(sim.process(fixture)[crop], control) for _ in range(repeats)]
    repo = Path(spektrafilm.__file__).resolve().parents[2]
    return {'commit': subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip(),
            'backend_requested': os.environ.get('SPEKTRAFILM_EXPERIMENTAL_CAM16', 'cpu'),
            'fixture': {'shape': fixture.shape, 'linear_prophoto_rgb': 0.18, 'crop_border': 32},
            'film_and_paper': ['kodak_gold_200', 'kodak_supra_endura'], 'numba_threads': 8,
            'repeats': repeats, 'frequency_unit': 'cycles/pixel', 'samples': cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats', type=int, default=8)
    parser.add_argument('--json', type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 3 or args.json.exists():
        parser.error('need at least three repeats and a new JSON path')
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(run(args.repeats), indent=2, allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()

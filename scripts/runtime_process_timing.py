"""Headless end-to-end engine benchmark; see docs/performance/README.md."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
from collections import defaultdict
from contextlib import ExitStack
from dataclasses import asdict, fields, is_dataclass
from functools import wraps
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
PORTRAIT = ROOT / 'img/test/portrait_leaves_32bit_linear_prophoto_rgb.tif'
SCAN_SIZES = {'6mp': (2000, 3000), '24mp': (4000, 6000), '45mp': (5477, 8216)}
CASES = ('preview640', 'preview1280', *SCAN_SIZES, 'raw')
THREADS = ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'NUMBA_NUM_THREADS')


def configure_threads(blas, numba):
    for name in THREADS:
        os.environ[name] = str(numba if name == 'NUMBA_NUM_THREADS' else blas)


def command_output(*args):
    try:
        return subprocess.check_output(args, cwd=ROOT, text=True, timeout=10, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def dimensions(case, shape):
    if case in SCAN_SIZES:
        return SCAN_SIZES[case]
    if case.startswith('preview'):
        edge = int(case.removeprefix('preview'))
        return tuple(int(side * edge / max(shape[:2])) for side in shape[:2])
    return shape[:2]


def params_for(args, i):
    from spektrafilm.runtime import digest_params, init_params

    alternate = i % 2 == 1
    stock = 'kodak_portra_400' if alternate and args.change == 'stock' else 'kodak_gold_200'
    params = init_params(film_profile=stock, print_profile='kodak_supra_endura')
    params.io.input_color_space = 'ProPhoto RGB'
    params.io.input_cctf_decoding = False
    params.io.scan_film = args.mode == 'negative' or (args.mode == 'alternate' and alternate)
    params.settings.use_enlarger_lut = True
    params.settings.use_scanner_lut = True
    params.settings.lut_resolution = 17
    params.settings.use_fast_stats = True
    params.settings.preview_mode = args.case.startswith('preview')
    params.debug.deactivate_stochastic_effects = args.deterministic
    if alternate and args.change == 'exposure':
        params.camera.exposure_compensation_ev += 0.25
    if alternate and args.change == 'filter':
        params.enlarger.y_filter_shift += 0.05
    params = digest_params(params)
    if args.grain_off:
        params.film_render.grain.active = False
    if args.halation_off:
        params.film_render.halation.active = False
    return params


def dump_params(params):
    return {
        f.name: (value.info.stock if f.name in ('film', 'print') else
                 asdict(value) if is_dataclass(value) else value)
        for f in fields(params) if (value := getattr(params, f.name)) is not None
    }


def instrument(stack, timings):
    from spektrafilm.model import develop
    from spektrafilm.runtime.stages import filming
    from spektrafilm.utils import gamut_compression as gamut

    targets = {
        'filming.halation': (filming, 'apply_halation_um'),
        'develop.density_interpolation': (develop, 'develop_simple'),
        'develop.couplers': (develop, 'apply_density_correction_dir_couplers'),
        'develop.grain': (develop, 'apply_grain'),
        'gamut.cam16ucs': (gamut, 'compress_rgb_cam16ucs_chroma'),
        # Includes small whitepoint calls and boundary-table construction when it is cold.
        'cam16ucs.forward': (gamut.colour, 'XYZ_to_CAM16UCS'),
        'cam16ucs.inverse': (gamut.colour, 'CAM16UCS_to_XYZ'),
    }

    def timed(fn, label):
        @wraps(fn)
        def run(*args, **kwargs):
            start = perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                timings[label] += perf_counter() - start
        return run

    for label, (module, name) in targets.items():
        stack.enter_context(patch.object(module, name, timed(getattr(module, name), label)))


def memory_peaks(process):
    memory = process.memory_info()
    return {'working_set_bytes': getattr(memory, 'peak_wset', None),
            'commit_bytes': getattr(memory, 'peak_pagefile', None)}


def summary(samples):
    durations = [s['engine_seconds'] for s in samples if s['phase'] == 'warm']
    median = statistics.median(durations)
    return {'warm_samples': len(durations), 'median_seconds': median,
            'mad_seconds': statistics.median(abs(x - median) for x in durations),
            'min_seconds': min(durations), 'max_seconds': max(durations)}


def run(args):
    configure_threads(args.blas_threads, args.numba_threads)
    cache_arg = os.environ.get('NUMBA_CACHE_DIR')
    cache_path = Path(cache_arg) if cache_arg else None
    cache_files = sum(path.is_file() for path in cache_path.rglob('*')) if cache_path else None

    import colour
    import numba
    import numpy as np
    import psutil
    from scipy.ndimage import zoom
    from threadpoolctl import threadpool_info

    from spektrafilm.runtime import Simulator
    from spektrafilm.utils.io import load_image_oiio, read_image_metadata, save_image_oiio, write_image_metadata
    from spektrafilm.utils.preview import resize_for_preview

    source = args.raw if args.case == 'raw' else args.image
    if args.case == 'raw':
        from spektrafilm.utils.raw_file_processor import load_and_process_raw_file
        start = perf_counter()
        image = load_and_process_raw_file(source, output_colorspace='ProPhoto RGB')
    else:
        start = perf_counter()
        image = load_image_oiio(str(source))[..., :3]
    source_shape = image.shape
    load_seconds = perf_counter() - start
    start = perf_counter()
    target = dimensions(args.case, image.shape)
    if max(target) < max(image.shape[:2]) and args.case.startswith('preview'):
        image = resize_for_preview(image, max(target))
    elif image.shape[:2] != target:
        image = zoom(image, (target[0] / image.shape[0], target[1] / image.shape[1], 1), order=1)
    resize_seconds = perf_counter() - start

    process = psutil.Process()
    timings = defaultdict(float)
    samples, configurations = [], {}
    result = simulator = None
    with ExitStack() as stack:
        instrument(stack, timings)
        for i in range(args.warm + 1):
            result = None  # Previous output is not an input to this engine-only measurement.
            timings.clear()
            variant = i % 2 if args.change != 'repeat' or args.mode == 'alternate' else 0
            update = (simulator is None or args.update or args.case.startswith('preview')
                      or samples[-1]['variant'] != variant)
            prepare_seconds = 0.0
            if update:
                start = perf_counter()
                params = params_for(args, i)
                if simulator is None:
                    simulator = Simulator(params)
                else:
                    simulator.update_params(params)
                prepare_seconds = perf_counter() - start
                configurations[variant] = dump_params(params)
            start = perf_counter()
            result = simulator.process(image)
            engine_seconds = perf_counter() - start
            samples.append({'phase': 'cold_process' if i == 0 else 'warm', 'variant': variant,
                            'prepare_seconds': prepare_seconds, 'engine_seconds': engine_seconds,
                            'node_seconds': dict(simulator.get_timings()), 'nested_seconds': dict(timings),
                            'process_peaks_to_date': memory_peaks(process)})
            print(f'{args.case} [{i}/{args.warm}] {engine_seconds:.3f}s', flush=True)

    peaks = memory_peaks(process)
    io_timings = {}
    if args.display:
        from PIL import Image, ImageCms
        from spektrafilm_gui.controller_runtime import prepare_output_display_image
        start = perf_counter()
        display, status = prepare_output_display_image(np.asarray(result, dtype=np.float32),
            output_color_space=params.io.output_color_space, use_display_transform=True,
            imagecms_module=ImageCms, colour_module=colour, pil_image_module=Image)
        io_timings.update(display_seconds=perf_counter() - start, display_status=status)
        del display
    if args.export:
        # Exercise a real save-space conversion independently from the renderer.
        save_space = 'ProPhoto RGB'
        path = args.json.with_suffix('.tif')
        start = perf_counter()
        converted = colour.RGB_to_RGB(result, params.io.output_color_space, save_space,
                                      apply_cctf_decoding=params.io.output_cctf_encoding,
                                      apply_cctf_encoding=True)
        io_timings.update(export_space=save_space, export_cctf=True, export_conversion_seconds=perf_counter() - start)
        start = perf_counter()
        save_image_oiio(str(path), converted, color_space=save_space)
        io_timings['tiff_write_seconds'] = perf_counter() - start
        del converted
        start = perf_counter()
        write_image_metadata(str(path), read_image_metadata(str(source)), saving_color_space=save_space)
        io_timings['exif_read_write_seconds'] = perf_counter() - start

    with source.open('rb') as file:
        source_hash = hashlib.file_digest(file, 'sha256').hexdigest()
    output = {
        'commit': command_output('git', 'rev-parse', 'HEAD'),
        'tracked_worktree_dirty': bool(command_output('git', 'status', '--porcelain', '--untracked-files=no')),
        'platform': platform.platform(), 'python': platform.python_version(),
        'packages': {key: version(key) for key in ('numpy', 'scipy', 'colour-science', 'numba', 'rawpy',
                                                   'OpenImageIO', 'psutil', 'threadpoolctl', 'spektrafilm')},
        'cpu': command_output('powershell', '-NoProfile', '-Command',
                              '(Get-CimInstance Win32_Processor).Name') or platform.processor(),
        'cpu_physical': psutil.cpu_count(logical=False), 'cpu_logical': psutil.cpu_count(),
        'ram_bytes': psutil.virtual_memory().total,
        'gpu': command_output('nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'),
        'threads_env': {key: os.environ[key] for key in THREADS},
        'threadpools': [dict(pool, filepath=Path(pool['filepath']).name) for pool in threadpool_info()],
        'numba_threads': numba.get_num_threads(), 'numba_threading_layer': numba.threading_layer(),
        'numba_cache': {'location': 'explicit' if cache_arg else 'package defaults', 'files_before': cache_files},
        'options': {k: v for k, v in vars(args).items() if k not in ('raw', 'image', 'json')},
        'input': {'kind': 'native raw' if args.case == 'raw' else 'TIFF', 'sha256': source_hash,
                  'source_shape': source_shape, 'engine_shape': image.shape, 'engine_dtype': str(image.dtype),
                  'load_seconds': load_seconds, 'resize_seconds': resize_seconds},
        'configurations': configurations, 'samples': samples, 'summary': summary(samples),
        'engine_process_peaks': peaks, 'separate_io': io_timings,
        'output_shape': result.shape, 'output_dtype': str(result.dtype),
    }
    if args.reference:
        reference = args.json.with_suffix('.npy')
        np.save(reference, result, allow_pickle=False)
        with reference.open('rb') as file:
            output['last_output_reference'] = {'file': reference.name, 'sha256': hashlib.file_digest(file, 'sha256').hexdigest(),
                                                'variant': samples[-1]['variant']}
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case', choices=CASES)
    parser.add_argument('--json', type=Path, required=True)
    parser.add_argument('--image', type=Path, default=PORTRAIT, help='linear ProPhoto RGB TIFF')
    parser.add_argument('--raw', type=Path, help='native camera RAW for the raw case')
    parser.add_argument('--mode', choices=('print', 'negative', 'alternate'), default='print')
    parser.add_argument('--change', choices=('repeat', 'exposure', 'filter', 'stock'), default='repeat')
    parser.add_argument('--warm', type=int, help='warm samples after the first process pass (default 10 preview / 3 scan)')
    parser.add_argument('--blas-threads', type=int, default=1)
    parser.add_argument('--numba-threads', type=int, default=8)
    for flag in ('update', 'deterministic', 'grain-off', 'halation-off', 'display', 'export', 'reference'):
        parser.add_argument(f'--{flag}', action='store_true')
    args = parser.parse_args()
    if args.warm is None:
        args.warm = 10 if args.case.startswith('preview') else 3
    if min(args.warm, args.blas_threads, args.numba_threads) < 1:
        parser.error('sample and thread counts must be positive')
    if (args.case == 'raw') != (args.raw is not None):
        parser.error('--raw is required for the raw case and not accepted for other cases')
    args.json.parent.mkdir(parents=True, exist_ok=True)
    output = run(args)
    args.json.write_text(json.dumps(output, indent=2, allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()

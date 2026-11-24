"""Compare deterministic benchmark images in linear RGB and D65 CIE Lab."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import colour
import numpy as np


D65 = colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D65']
FP32_DE00_LIMITS = {'p95': 0.5, 'p99': 1.0}


def load_reference(path):
    record = json.loads(path.read_bytes())
    if not record['options']['deterministic']:
        raise ValueError(f'{path}: rerun using --deterministic --reference for pixel parity')
    reference = record['last_output_reference']
    config = record['configurations'][str(reference['variant'])]
    frame_path = path.with_name(reference['file'])
    with frame_path.open('rb') as file:
        if hashlib.file_digest(file, 'sha256').hexdigest() != reference['sha256']:
            raise ValueError(f'{frame_path}: reference image does not match benchmark SHA-256')
    frame = np.load(frame_path, mmap_mode='r', allow_pickle=False)
    return frame, config, record['input']['sha256']


def compare(before, after, color_space, encoded, rows=128):
    if before.shape != after.shape or before.ndim != 3 or before.shape[-1] != 3:
        raise ValueError(f'expected matching H×W×3 arrays, got {before.shape} and {after.shape}')
    space = colour.RGB_COLOURSPACES[color_space]
    count = before.shape[0] * before.shape[1]
    delta_e = np.empty(count)
    compared = finite_count = 0
    square_error = max_error = 0.0
    counts = dict(new_nan=0, removed_nan=0, new_inf=0, removed_inf=0, opposite_inf_sign=0,
                  before_below_black=0, after_below_black=0, before_above_white=0, after_above_white=0,
                  new_below_black=0, new_above_white=0, new_at_black=0, new_at_white=0)
    worst = []
    # Track out-of-range elements separately from perceptual metrics on displayable pixels.
    # A numerical tolerance avoids treating rounding noise at 0 or 1 as new clipping.
    tolerance = 1e-8
    for y in range(0, before.shape[0], rows):
        ref, new = np.asarray(before[y:y + rows]), np.asarray(after[y:y + rows])
        nan_ref, nan_new = np.isnan(ref), np.isnan(new)
        inf_ref, inf_new = np.isinf(ref), np.isinf(new)
        counts['new_nan'] += np.count_nonzero(nan_new & ~nan_ref)
        counts['removed_nan'] += np.count_nonzero(nan_ref & ~nan_new)
        counts['new_inf'] += np.count_nonzero(inf_new & ~inf_ref)
        counts['removed_inf'] += np.count_nonzero(inf_ref & ~inf_new)
        counts['opposite_inf_sign'] += np.count_nonzero(inf_ref & inf_new & (np.signbit(ref) != np.signbit(new)))
        if encoded:
            with np.errstate(over='ignore', invalid='ignore'):
                ref, new = space.cctf_decoding(ref), space.cctf_decoding(new)
        finite = np.isfinite(ref) & np.isfinite(new)
        difference = new[finite] - ref[finite]
        finite_count += difference.size
        square_error += float(np.sum(difference**2))
        max_error = max(max_error, float(np.max(np.abs(difference), initial=0.0)))
        for label, ref_oob, new_oob in (
            ('below_black', ref < -tolerance, new < -tolerance),
            ('above_white', ref > 1.0 + tolerance, new > 1.0 + tolerance),
        ):
            counts[f'before_{label}'] += np.count_nonzero(ref_oob & np.isfinite(ref))
            counts[f'after_{label}'] += np.count_nonzero(new_oob & np.isfinite(new))
            counts[f'new_{label}'] += np.count_nonzero(new_oob & ~ref_oob & finite)
        counts['new_at_black'] += np.count_nonzero(finite & (np.abs(new) <= tolerance) & (ref > tolerance))
        counts['new_at_white'] += np.count_nonzero(finite & (np.abs(new - 1.0) <= tolerance) & (ref < 1.0 - tolerance))
        display = (finite & (ref >= -tolerance) & (ref <= 1.0 + tolerance)
                   & (new >= -tolerance) & (new <= 1.0 + tolerance)).all(axis=-1)
        if not display.any():
            continue
        # Respect the recorded R'G'B' transfer functions above, then adapt both to D65.
        ref_lab = colour.XYZ_to_Lab(colour.RGB_to_XYZ(ref[display], space, illuminant=D65), illuminant=D65)
        new_lab = colour.XYZ_to_Lab(colour.RGB_to_XYZ(new[display], space, illuminant=D65), illuminant=D65)
        values = np.asarray(colour.difference.delta_E_CIE2000(ref_lab, new_lab))
        delta_e[compared:compared + values.size] = values
        compared += values.size
        coordinates = np.argwhere(display)
        top = min(5, len(values))
        for index in np.argpartition(values, -top)[-top:]:
            row, col = coordinates[index]
            worst.append({'xy': [int(col), y + int(row)], 'delta_e00': float(values[index]),
                          'before_linear': ref[row, col].tolist(), 'after_linear': new[row, col].tolist()})
        worst = sorted(worst, key=lambda item: item['delta_e00'], reverse=True)[:5]

    de00 = None
    if compared:
        percentiles = np.percentile(delta_e[:compared], [50, 95, 99, 100], overwrite_input=True)
        de00 = dict(zip(('p50', 'p95', 'p99', 'max'), percentiles.tolist()))
    return {'shape': before.shape, 'color_space': color_space, 'encoded': encoded, 'total_pixels': count,
            'displayable_pixels_compared': compared, 'finite_elements_compared': finite_count,
            'linear_rgb_max_abs': max_error if finite_count else None,
            'linear_rgb_rmse': (square_error / finite_count)**0.5 if finite_count else None,
            'counts': counts, 'displayable_delta_e00_D65': de00, 'worst_displayable_pixels': worst,
            'fp32_perceptual_threshold_met': bool(de00 and all(de00[p] <= limit for p, limit in FP32_DE00_LIMITS.items()))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('before', type=Path, help='benchmark JSON with a deterministic .npy reference')
    parser.add_argument('after', type=Path, help='candidate benchmark JSON with the same input/options')
    parser.add_argument('--json', type=Path)
    args = parser.parse_args()
    before, config, fixture = load_reference(args.before)
    after, after_config, after_fixture = load_reference(args.after)
    if config != after_config or fixture != after_fixture:
        parser.error('input hash and effective render configuration must match')
    io = config['io']
    report = compare(before, after, io['output_color_space'], io['output_cctf_encoding'])
    text = json.dumps(report, indent=2, allow_nan=False) + '\n'
    if args.json:
        args.json.write_text(text, encoding='utf-8')
    else:
        print(text, end='')


if __name__ == '__main__':
    main()

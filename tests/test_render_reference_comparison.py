import colour
import numpy as np
import pytest

from scripts.compare_render_references import compare


def test_comparison_decodes_rgb_and_separates_hdr_nan_and_clip_metrics():
    linear = np.array([[[0.25] * 3, [3.] * 3, [0.8] * 3], [[0.3] * 3, [0.4] * 3, [0.5] * 3]])
    changed = linear.copy()
    changed[0, 1, 0] = 4.
    changed[0, 2, 0] = 1.
    changed[1, 0, 0] = np.nan
    changed[1, 1, 0] = np.inf
    changed[1, 2, 0] = -0.01
    curve = colour.RGB_COLOURSPACES['sRGB'].cctf_encoding
    report = compare(curve(linear), curve(changed), 'sRGB', encoded=True, rows=1)

    counts = report['counts']
    assert counts['new_nan'] == counts['new_inf'] == 1
    assert counts['new_below_black'] == counts['new_at_white'] == 1
    assert counts['before_above_white'] == counts['after_above_white'] == 3
    assert report['linear_rgb_max_abs'] == pytest.approx(1.)
    assert report['displayable_pixels_compared'] == 2
    assert report['worst_displayable_pixels'][0]['xy'] == [2, 0]
    assert not report['fp32_perceptual_threshold_met']


def test_identical_aces_reference_has_zero_colour_error_after_d65_conversion():
    rgb = np.array([[[0.38, 0.5, 0.62], [0.5, 0.5, 0.5]]])
    report = compare(rgb, rgb, 'ACES2065-1', encoded=False)
    assert report['displayable_delta_e00_D65']['p99'] == 0.
    assert report['linear_rgb_rmse'] == 0.
    assert report['fp32_perceptual_threshold_met']

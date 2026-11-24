import numpy as np
import pytest

from spektrafilm.runtime import Simulator
from spektrafilm.runtime.topology import Tap
from spektrafilm.utils import gamut_compression as gamut
from tests.conftest import make_fast_test_params


@pytest.mark.parametrize('space', ['sRGB', 'ProPhoto RGB', 'ACES2065-1'])
@pytest.mark.parametrize('lightness', [None, (0.7, 1.0, 2.2)])
def test_fused_cam16_matches_colour_on_fp64_colour_mires(space, lightness, monkeypatch):
    rng = np.random.default_rng(345)
    rgb = rng.uniform(0.0, 2.0, (72, 66, 3))
    rgb[0] = np.linspace(0., 4., 66)[:, None]
    rgb[1, :6] = [(0., 0., 0.), (1., 0., 0.), (0., 1., 0.), (0., 0., 1.),
                    (-0.05, 0.5, 0.2), (32., 32., 32.)]
    rgb = np.rot90(rgb)
    options = dict(threshold=0.25, limit=1.0, power=3.0, lightness_compression=lightness)
    reference = gamut._compress_rgb_cam16ucs_reference
    fallback_sizes = []

    def track(batch, *args, **kwargs):
        fallback_sizes.append(len(batch))
        return reference(batch, *args, **kwargs)

    monkeypatch.setattr(gamut, '_compress_rgb_cam16ucs_reference', track)
    with np.errstate(invalid='ignore'):
        expected = reference(rgb, space, **options)
        actual = gamut.compress_rgb_cam16ucs_chroma(rgb, space, **options)
    # Saturated blues in wide working spaces can be outside CAM16's valid domain.
    assert sum(fallback_sizes) < rgb.shape[0] * rgb.shape[1] * 0.01
    np.testing.assert_array_equal(np.isfinite(actual), np.isfinite(expected))
    np.testing.assert_allclose(actual, expected, rtol=3e-12, atol=2e-12)
    np.testing.assert_array_equal(actual[-1, 1], [0., 0., 0.])


def test_singular_and_nonfinite_pixels_use_reference_without_rerendering_frame(monkeypatch):
    rng = np.random.default_rng(4)
    rgb = rng.uniform(0.01, 1.5, (5000, 3))
    rgb[:5] = [(np.nan, 0., 0.), (np.inf, 0., 0.), (0., 0., -np.inf),
               (-2.0, -4.0, 0.05), (-0.2, 0.0, -0.15)]
    options = dict(threshold=0., limit=1., power=6., lightness_compression=(0.7, 1.0, 2.2))
    reference = gamut._compress_rgb_cam16ucs_reference
    calls = []

    def track(batch, *args, **kwargs):
        calls.append(len(batch))
        return reference(batch, *args, **kwargs)

    monkeypatch.setattr(gamut, '_compress_rgb_cam16ucs_reference', track)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        expected = reference(rgb, 'sRGB', **options)
        result = gamut.compress_rgb_cam16ucs_chroma(rgb, 'sRGB', **options)
    assert len(calls) == 1 and 4 <= calls[0] <= 6
    np.testing.assert_array_equal(np.isnan(result), np.isnan(expected))
    np.testing.assert_array_equal(np.isinf(result), np.isinf(expected))
    np.testing.assert_allclose(result, expected, rtol=3e-12, atol=2e-12)


@pytest.mark.parametrize('film,paper,negative', [
    ('kodak_gold_200', 'kodak_supra_endura', False),
    ('fujifilm_pro_400h', 'fujifilm_crystal_archive_typeii', False),
    ('kodak_portra_400', 'kodak_portra_endura', True),
])
def test_deterministic_pipeline_preserves_tap_and_output(film, paper, negative, monkeypatch):
    p = make_fast_test_params(film_profile=film, print_profile=paper)
    p.io.scan_film = negative
    p.settings.use_scanner_lut = p.settings.use_enlarger_lut = True
    rng = np.random.default_rng(4)
    image = rng.uniform(0.01, 1.5, (70, 63, 3))
    image[0] = np.linspace(0.001, 1.2, 63)[:, None]
    p.taps.collect = Tap.CMY_FILM if negative else Tap.CMY_PRINT
    density = Simulator(p).process(image)
    p.taps.collect = Tap.RGB_OUT
    output = Simulator(p).process(image)

    monkeypatch.setattr(gamut, 'compress_rgb_cam16ucs_chroma', gamut._compress_rgb_cam16ucs_reference)
    expected = Simulator(p).process(image)
    p.taps.collect = Tap.CMY_FILM if negative else Tap.CMY_PRINT
    np.testing.assert_array_equal(Simulator(p).process(image), density)
    np.testing.assert_allclose(output, expected, rtol=3e-12, atol=2e-12)

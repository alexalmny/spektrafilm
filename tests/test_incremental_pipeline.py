"""Explicit tap reuse must match a fresh render and never reuse changed input."""

import numpy as np
import pytest

from spektrafilm.runtime.pipeline import SimulationPipeline
from spektrafilm.runtime.topology import Tap


def test_scanner_change_reuses_print_without_changing_output(default_params, small_rgb_image):
    pipe = SimulationPipeline(default_params)
    pipe.process_cached(small_rgb_image)
    pipe.io.output_cctf_encoding = not pipe.io.output_cctf_encoding
    cached = pipe.process_cached(small_rgb_image, reuse_through=Tap.CMY_PRINT)
    assert "scanning.scan_print" in pipe.get_timings()
    assert "filming.develop" not in pipe.get_timings()
    assert "printing.develop" not in pipe.get_timings()
    full = pipe.process(small_rgb_image)
    np.testing.assert_array_equal(cached, full)


def test_scan_film_reuses_film_density(default_params, small_rgb_image):
    default_params.io.scan_film = True
    pipe = SimulationPipeline(default_params)
    pipe.process_cached(small_rgb_image)
    pipe.io.output_cctf_encoding = not pipe.io.output_cctf_encoding
    cached = pipe.process_cached(small_rgb_image, reuse_through=Tap.CMY_FILM)
    assert "filming.develop" not in pipe.get_timings()
    np.testing.assert_array_equal(cached, pipe.process(small_rgb_image))


def test_soft_update_limits_reusable_prefix(default_params, small_rgb_image):
    pipe = SimulationPipeline(default_params)
    pipe.process_cached(small_rgb_image)
    pipe.soft_update(print_exposure=pipe.enlarger.print_exposure * 1.1)
    with pytest.raises(ValueError, match="invalidated"):
        pipe.process_cached(small_rgb_image, reuse_through=Tap.CMY_PRINT)
    cached = pipe.process_cached(small_rgb_image, reuse_through=Tap.CMY_FILM)
    assert "filming.develop" not in pipe.get_timings()
    np.testing.assert_array_equal(cached, pipe.process(small_rgb_image))


def test_input_mutation_forces_full_render(default_params, small_rgb_image):
    pipe = SimulationPipeline(default_params)
    pipe.process_cached(small_rgb_image)
    small_rgb_image[0, 0, :] = 0.85
    cached = pipe.process_cached(small_rgb_image, reuse_through=Tap.CMY_PRINT)
    assert "filming.develop" in pipe.get_timings()
    np.testing.assert_array_equal(cached, pipe.process(small_rgb_image))


def test_invalid_taps_and_cache_reset(default_params, small_rgb_image):
    pipe = SimulationPipeline(default_params)
    with pytest.raises(ValueError, match="invalid reusable tap"):
        pipe.process_cached(small_rgb_image, reuse_through=Tap.RGB_OUT)
    pipe.process_cached(small_rgb_image)
    pipe.clear_cache()
    pipe.process_cached(small_rgb_image, reuse_through=Tap.CMY_PRINT)
    assert "filming.develop" in pipe.get_timings()

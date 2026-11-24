import numpy as np
import pytest

from spektrafilm.runtime.pipeline import SimulationPipeline
from spektrafilm.runtime.topology import Tap


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('channels', [3, 4])
@pytest.mark.parametrize('rotated', [False, True])
def test_render_accepts_readonly_rgb_views_without_mutation(small_rgb_image, default_params, dtype, channels, rotated):
    image = small_rgb_image[:, :12].astype(dtype)
    if channels == 4:
        image = np.concatenate((image, np.zeros_like(image[..., :1])), axis=-1)
    if rotated:
        image = np.rot90(image)
        assert any(stride < 0 for stride in image.strides)
    original = image.copy()
    image.flags.writeable = False
    reference = np.ascontiguousarray(original[..., :3], dtype=np.float64)
    pipeline = SimulationPipeline(default_params)

    expected = pipeline.process(reference)
    np.testing.assert_array_equal(pipeline.process(image), expected)
    np.testing.assert_array_equal(image, original)


def test_collected_preprocess_tap_has_independent_pixels(small_rgb_image, default_params):
    pipeline = SimulationPipeline(default_params)
    preprocessed = pipeline.process(small_rgb_image, collect=Tap.RGB_PRE)
    np.testing.assert_array_equal(preprocessed, small_rgb_image)
    assert not np.shares_memory(preprocessed, small_rgb_image)

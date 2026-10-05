"""Check GPU blur transfer boundaries without requiring a CUDA device."""
import sys
import types

import numpy as np
from scipy.ndimage import correlate1d

from spektrafilm.utils import gpu
from spektrafilm.utils.fast_gaussian_filter import _gpu_gaussian_filter, _gaussian_kernel_1d


def test_gpu_gaussian_copies_image_once(monkeypatch):
    transfers = {"image_in": 0, "image_out": 0}

    def asarray(array):
        if np.ndim(array) > 1:
            transfers["image_in"] += 1
        return np.asarray(array)

    def asnumpy(array):
        transfers["image_out"] += 1
        return np.asarray(array)

    fake_cp = types.SimpleNamespace(asarray=asarray, asnumpy=asnumpy, empty_like=np.empty_like)
    monkeypatch.setattr(gpu, "cupy_module", lambda: fake_cp)
    monkeypatch.setitem(sys.modules, "cupyx", types.ModuleType("cupyx"))
    monkeypatch.setitem(sys.modules, "cupyx.scipy", types.ModuleType("cupyx.scipy"))
    fake_ndimage = types.ModuleType("cupyx.scipy.ndimage")
    fake_ndimage.correlate1d = correlate1d
    monkeypatch.setitem(sys.modules, "cupyx.scipy.ndimage", fake_ndimage)

    rng = np.random.default_rng(5)
    image = rng.random((20, 24, 3))
    sigmas = [0.8, 1.3, 0.8]
    result = _gpu_gaussian_filter(image, sigmas)
    expected = np.empty_like(image)
    for ch, sigma in enumerate(sigmas):
        kernel, _ = _gaussian_kernel_1d(sigma, 3.0)
        tmp = correlate1d(image[:, :, ch], kernel, axis=0, mode="reflect")
        expected[:, :, ch] = correlate1d(tmp, kernel, axis=1, mode="reflect")
    np.testing.assert_allclose(result, expected, rtol=0, atol=0)
    assert transfers == {"image_in": 1, "image_out": 1}

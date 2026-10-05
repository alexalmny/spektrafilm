"""PCHIP preparation may be reused only for the matching spectral LUT."""

import numpy as np
import pytest

from spektrafilm.runtime.services import spectral_lut_compute as service_module
from spektrafilm.utils.lut import compute_with_lut


@pytest.mark.parametrize("stage", ["enlarger", "scanner"])
def test_cached_lut_preparation_matches_uncached_and_invalidates(monkeypatch, stage):
    service = service_module.SpectralLUTService(lut_resolution=5)
    calculate = lambda x: x ** 2
    image = np.random.default_rng(1).random((4, 6, 3))
    call = getattr(service, f"spectral_compute_{stage}")
    kwargs = dict(data_min=0, data_max=1, use_lut=True)
    original = service_module.prepare_lut_pchip_3d
    count = [0]

    def counted(lut):
        count[0] += 1
        return original(lut)

    monkeypatch.setattr(service_module, "prepare_lut_pchip_3d", counted)
    first = call(image, calculate, **kwargs)
    second = call(image, calculate, **kwargs)
    third = call(image, calculate, **kwargs)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first, third)
    assert count == [1]  # first cache hit prepares; second hit reuses

    memory_name = f"{stage}_lut_memory"
    reference, _ = compute_with_lut(image, calculate, lut=getattr(service, memory_name), **{
        "xmin": 0, "xmax": 1,
    })
    np.testing.assert_array_equal(second, reference)

    # A changed calculation rebuilds the LUT and must discard the old slopes.
    changed = lambda x: x ** 3
    rebuilt = call(image, changed, **kwargs)
    next_result = call(image, changed, **kwargs)
    np.testing.assert_array_equal(rebuilt, next_result)
    assert count == [2]

    # LUT memory is public and may be replaced by a caller.
    replacement = getattr(service, memory_name).copy()
    replacement *= 0.5
    setattr(service, memory_name, replacement)
    replaced = call(image, changed, **kwargs)
    expected, _ = compute_with_lut(image, changed, lut=replacement, xmin=0, xmax=1)
    np.testing.assert_array_equal(replaced, expected)
    assert count == [3]

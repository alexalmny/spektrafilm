import numpy as np
import pytest

from spektrafilm.profiles.io import Hanatos2025SensitivityAdaptation
from spektrafilm.runtime.services import spectral_lut_compute as spectral_lut_compute_module


pytestmark = pytest.mark.unit


def test_filming_tc_lut_recomputes_when_spectral_gaussian_blur_changes(monkeypatch) -> None:
    calls: list[float] = []

    def fake_compute_hanatos2025_tc_lut(sensitivity, adaptation, gamut_compress=None):
        del sensitivity, gamut_compress
        calls.append(float(adaptation.spectral_gaussian_blur))
        return np.full((2, 2, 3), adaptation.spectral_gaussian_blur + 1.0, dtype=float)

    monkeypatch.setattr(
        spectral_lut_compute_module,
        'compute_hanatos2025_tc_lut',
        fake_compute_hanatos2025_tc_lut,
    )

    service = spectral_lut_compute_module.SpectralLUTService(lut_resolution=17)
    sensitivity = np.ones((4, 3), dtype=float)
    adaptation = Hanatos2025SensitivityAdaptation(
        window_params=np.empty((0,), dtype=float),
        surface_params=np.empty((0, 3), dtype=float),
        spectral_gaussian_blur=0.0,
        reference_illuminant='D55',
        apply_window=True,
        apply_surface=True,
    )

    service.set_hanatos2025_adaptation(adaptation)
    first = service.get_filming_tc_lut(sensitivity)

    adaptation.spectral_gaussian_blur = 4.0
    service.set_hanatos2025_adaptation(adaptation)
    second = service.get_filming_tc_lut(sensitivity)

    assert calls == [0.0, 4.0]
    assert np.array_equal(first, second) is False


@pytest.mark.parametrize('stage', ['enlarger', 'scanner'])
@pytest.mark.parametrize('changed_bound', ['min', 'max'])
def test_spectral_lut_rebuilds_if_only_density_domain_changes(stage, changed_bound):
    service_type = spectral_lut_compute_module.SpectralLUTService
    compute = getattr(service_type(9), f'spectral_compute_{stage}')
    fresh = getattr(service_type(9), f'spectral_compute_{stage}')
    data = np.array([[[0.2, 0.35, 0.7], [0.6, 0.5, 0.3]]])
    bounds = {'min': np.zeros(3), 'max': np.ones(3)}
    calls = []

    def spectral_calculation(cmy):
        calls.append(cmy.shape)
        return cmy * np.array([2., 3., 4.])

    before = compute(data, spectral_calculation, bounds['min'], bounds['max'], use_lut=True)
    np.testing.assert_array_equal(
        compute(data, spectral_calculation, bounds['min'], bounds['max'], use_lut=True), before,
    )
    assert calls.count((81, 9, 3)) == 1  # Unchanged bounds reuse the cube.

    # The caller may mutate the original bounds after the cache has seen them.
    bounds[changed_bound][1] += -0.25 if changed_bound == 'min' else 0.25
    updated = compute(data, spectral_calculation, bounds['min'], bounds['max'], use_lut=True)
    assert calls.count((81, 9, 3)) == 2
    np.testing.assert_array_equal(
        updated, fresh(data, spectral_calculation, bounds['min'], bounds['max'], use_lut=True),
    )
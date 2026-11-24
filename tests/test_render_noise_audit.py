import numpy as np

from scripts.render_noise_audit import noise_metrics


def test_noise_spectrum_keeps_residual_variance_and_frequency():
    phase = 2 * np.pi * np.arange(16) / 8
    pattern = np.broadcast_to(np.sin(phase)[None, :, None], (16, 16, 3))
    control = np.full_like(pattern, 0.2)
    result = noise_metrics(control + 0.015 + 0.03 * pattern, control)

    np.testing.assert_allclose(result['mean_residual'], 0.015)
    np.testing.assert_allclose(result['std_residual'], 0.03 / np.sqrt(2))
    np.testing.assert_allclose(list(result['radial_power'].values()), [[0.] * 3, [0.00045] * 3, [0.] * 3], atol=1e-15)

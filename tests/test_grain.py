import numpy as np
import pytest

from spektrafilm.model.density_curves import interp_density_cmy_layers
from spektrafilm.model.grain import apply_grain_to_density, apply_grain_to_density_layers
from spektrafilm.model.grain import apply_grain
from spektrafilm.runtime.params_schema import GrainParams


pytestmark = pytest.mark.unit


class TestApplyGrain:
    def test_apply_grain_returns_input_when_bypassed_or_inactive(self):
        density_cmy = np.full((3, 3, 3), 0.4)
        density_curves = np.tile(np.linspace(0.0, 2.0, 8)[:, None], (1, 3))
        density_curves_layers = np.tile(density_curves[:, None, :] / 3.0, (1, 3, 1))
        grain = GrainParams(active=False)

        inactive = apply_grain(
            density_cmy.copy(),
            4.0,
            grain,
            density_curves,
            density_curves_layers,
            "negative",
        )
        bypassed = apply_grain(
            density_cmy.copy(),
            4.0,
            GrainParams(active=True),
            density_curves,
            density_curves_layers,
            "negative",
            bypass_grain=True,
        )

        np.testing.assert_allclose(inactive, density_cmy, atol=1e-10)
        np.testing.assert_allclose(bypassed, density_cmy, atol=1e-10)

    def test_apply_grain_matches_single_layer_pipeline(self):
        density_cmy = np.full((4, 4, 3), [0.3, 0.6, 0.9], dtype=np.float64)
        density_curves = np.column_stack([
            np.linspace(0.0, 2.4, 12),
            np.linspace(0.0, 2.2, 12),
            np.linspace(0.0, 2.0, 12),
        ])
        density_curves_layers = np.tile(density_curves[:, None, :] / 3.0, (1, 3, 1))
        grain = GrainParams(
            active=True,
            sublayers_active=False,
            particle_area_um2=0.25,
            particle_scale=(0.9, 1.1, 1.4),
            density_min=(0.05, 0.07, 0.09),
            uniformity=(0.98, 0.97, 0.96),
            blur=0.0,
            n_sub_layers=2,
        )

        result = apply_grain(
            density_cmy.copy(),
            5.0,
            grain,
            density_curves,
            density_curves_layers,
            "negative",
        )

        expected = apply_grain_to_density(
            density_cmy.copy(),
            pixel_size_um=5.0,
            particle_area_um2=grain.particle_area_um2,
            particle_scale=grain.particle_scale,
            density_min=grain.density_min,
            density_max_curves=np.nanmax(density_curves, axis=0),
            grain_uniformity=grain.uniformity,
            grain_blur=grain.blur,
            n_sub_layers=grain.n_sub_layers,
        )

        np.testing.assert_allclose(result, expected, atol=1e-10)

    @pytest.mark.parametrize("profile_type", ["negative", "positive"])
    def test_apply_grain_matches_layered_pipeline(self, profile_type):
        density_cmy = np.full((4, 4, 3), [0.35, 0.55, 0.75], dtype=np.float64)
        density_curves = np.column_stack([
            np.linspace(0.0, 2.1, 10),
            np.linspace(0.0, 1.9, 10),
            np.linspace(0.0, 1.7, 10),
        ])
        density_curves_layers = np.stack([
            density_curves * np.array([0.55, 0.50, 0.45]),
            density_curves * np.array([0.30, 0.33, 0.35]),
            density_curves * np.array([0.15, 0.17, 0.20]),
        ], axis=1)
        grain = GrainParams(
            active=True,
            sublayers_active=True,
            particle_area_um2=0.18,
            particle_scale=(0.8, 1.0, 1.2),
            particle_scale_layers=(2.2, 1.0, 0.5),
            density_min=(0.04, 0.06, 0.08),
            uniformity=(0.99, 0.98, 0.97),
            blur=0.0,
            blur_dye_clouds_um=0.0,
            micro_structure=(0.0, 0.0),
        )

        result = apply_grain(
            density_cmy.copy(),
            4.0,
            grain,
            density_curves,
            density_curves_layers,
            profile_type,
            use_fast_stats=False,
        )

        density_cmy_layers = interp_density_cmy_layers(
            density_cmy.copy(),
            density_curves,
            density_curves_layers,
            positive_film=profile_type == "positive",
        )
        expected = apply_grain_to_density_layers(
            density_cmy_layers,
            density_max_layers=np.nanmax(density_curves_layers, axis=0),
            pixel_size_um=4.0,
            particle_area_um2=grain.particle_area_um2,
            particle_scale=grain.particle_scale,
            particle_scale_layers=grain.particle_scale_layers,
            density_min=grain.density_min,
            grain_uniformity=grain.uniformity,
            grain_blur=grain.blur,
            grain_blur_dye_clouds_um=grain.blur_dye_clouds_um,
            grain_micro_structure=grain.micro_structure,
            use_fast_stats=False,
        )

        np.testing.assert_allclose(result, expected, atol=1e-10)

@pytest.mark.parametrize('p,particles', [(0.05, 12.0), (0.12, 40.0), (0.42, 90.0), (0.95, 150.0)])
def test_fused_particle_model_preserves_distribution(p, particles):
    from spektrafilm.model.grain import layer_particle_model
    from spektrafilm.utils.fast_stats import fast_binomial, fast_poisson

    maximum, uniformity = 2.2, 0.97
    density = np.full((256, 768), p * maximum)[:, ::-2]
    draws = np.full(density.shape, p)
    saturation = 1 - draws * uniformity * (1 - 1e-6)
    seeds = fast_poisson(particles / saturation)
    reference = fast_binomial(seeds, draws) * (maximum / particles) * saturation
    fused = layer_particle_model(density, density_max=maximum, n_particles_per_pixel=particles,
                                 grain_uniformity=uniformity, use_fast_stats=True)

    assert fused.dtype == np.float64 and fused.shape == density.shape
    assert np.all(density == p * maximum)  # strided input was not modified
    assert abs(fused.mean() - reference.mean()) <= 7 * reference.std() / np.sqrt(fused.size)
    assert fused.std() == pytest.approx(reference.std(), rel=0.03)


def test_fused_sampler_preserves_nonfinite_and_clipped_density():
    from spektrafilm.utils.fast_stats import sample_grain_particles

    density = np.tile([np.nan, -np.inf, np.inf], (32, 1))
    result = sample_grain_particles(density, 2.2, 120.0, 0.97)
    assert np.isnan(result[:, 0]).all()
    assert np.isfinite(result[:, 1:]).all()

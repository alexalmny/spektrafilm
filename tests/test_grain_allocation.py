"""Avoid a discarded full-frame grain allocation without changing samples."""

import numpy as np
from scipy.stats import binom, poisson

from spektrafilm.model.grain import layer_particle_model


def test_poisson_binomial_matches_seeded_reference():
    density = np.random.default_rng(1).uniform(0.1, 2, (32, 48))
    density_max = 2.2
    particles = 50.0
    uniformity = 0.98
    probability = np.clip(density / density_max, 1e-6, 1 - 1e-6)
    saturation = 1 - probability * uniformity * (1 - 1e-6)
    np.random.seed(123)
    seeds = poisson.rvs(particles / saturation)
    expected = np.double(binom.rvs(seeds, probability)) * (density_max / particles) * saturation

    result = layer_particle_model(
        density, density_max=density_max, n_particles_per_pixel=particles,
        grain_uniformity=uniformity, seed=123,
    )
    np.testing.assert_array_equal(result, expected)


def test_unknown_method_still_returns_zeros():
    result = layer_particle_model(np.ones((3, 4)), method="unknown")
    np.testing.assert_array_equal(result, np.zeros((3, 4)))

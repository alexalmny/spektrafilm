"""GPU acceleration tests.

The GPU path (CuPy/CUDA) must track the CPU path to float64 rounding —
it replays the same operation sequence, so any tolerance breach means a
real semantic drift, not just accelerator noise. Every test that needs a
device skips cleanly on machines without CuPy or CUDA (including CI).
"""
from __future__ import annotations

import numpy as np
import pytest

import spektrafilm.utils.gpu as gpu
from spektrafilm.utils.fast_gaussian_filter import fast_gaussian_filter
from spektrafilm.utils.gamut_compression import (
    OutputGamutCompressSpec,
    compress_rgb,
    compress_rgb_cam16ucs_chroma,
)


def _require_gpu():
    pytest.importorskip("cupy", reason="CuPy not installed")
    if gpu.cupy_module() is None:
        pytest.skip("no usable CUDA device")


def _random_rgb(shape, seed=7, lo=-0.05, hi=1.3):
    rng = np.random.default_rng(seed)
    return (hi - lo) * rng.random(shape) + lo


class TestGpuContext:
    def test_off_by_default(self):
        # Without an active context nothing may dispatch to the GPU —
        # this is what keeps the bit-exact regression baselines valid.
        assert gpu.active() is False

    def test_context_sets_and_resets(self, monkeypatch):
        # With a usable backend the raw flag follows the request...
        fake = type("M", (), {})()
        monkeypatch.setattr(gpu, "cupy_module", lambda: fake)
        with gpu.use(True):
            assert gpu._use_gpu.get() is True
            assert gpu.active() is True
        assert gpu._use_gpu.get() is False
        # ...and the flag is per-context, not global state.
        assert gpu.active() is False

    def test_unavailable_falls_back_to_cpu(self, monkeypatch):
        monkeypatch.setattr(gpu, "cupy_module", lambda: None)
        with pytest.warns(RuntimeWarning, match="CuPy/CUDA is unavailable"):
            with gpu.use(True):
                assert gpu.active() is False


class TestNumpyPathUnchanged:
    """The xp refactor of fast_cam16 must leave the numpy path untouched."""

    def test_compress_matches_reference(self):
        ref = np.load(_REF_FILE)
        out = compress_rgb_cam16ucs_chroma(
            ref["rgb"], "sRGB", threshold=0.8, limit=0.97, power=2.2,
            lightness_compression=(0.7, 1.0, 2.2),
        )
        assert np.array_equal(out, ref["out"], equal_nan=True)

    def test_gaussian_numpy_path_returns_numpy(self):
        img = _random_rgb((64, 80, 3), seed=3)
        out = fast_gaussian_filter(img, 1.5)
        assert type(out) is np.ndarray


class TestGpuCam16Chain:
    def test_forward_inverse_matches_cpu(self):
        _require_gpu()
        from spektrafilm.utils.fast_cam16 import (
            cam16ucs_to_xyz, xyz_to_cam16ucs,
        )
        from spektrafilm.utils.gamut_compression import cam16ucs_conditions

        import cupy as cp

        ref = np.load(_REF_FILE)
        vc = cam16ucs_conditions("sRGB")
        with gpu.use(True):
            xyz_dev = cp.asarray(ref["xyz"])
            jab_dev = xyz_to_cam16ucs(xyz_dev, vc)
            back_dev = cam16ucs_to_xyz(jab_dev, vc)
            jab = cp.asnumpy(jab_dev)
            back = cp.asnumpy(back_dev)
        np.testing.assert_allclose(jab, ref["jab"], rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(back, ref["back"], rtol=1e-10, atol=1e-10)

    def test_compress_rgb_matches_cpu(self):
        _require_gpu()
        rgb = _random_rgb((513, 509, 3), seed=11)
        spec = OutputGamutCompressSpec(algorithm="cam16ucs")
        cpu = compress_rgb(
            rgb, spec, output_color_space="sRGB",
        )
        with gpu.use(True):
            dev = compress_rgb(
                rgb, spec, output_color_space="sRGB",
            )
        # Same math on a different accelerator: float64 rounding across
        # ~40 chained transcendental ops. 1e-9 relative is far below any
        # visible (8-bit LSB ≈ 4e-3) and below LUT-bake tolerances.
        np.testing.assert_allclose(dev, cpu, rtol=1e-9, atol=1e-12)


class TestGpuGaussian:
    def test_small_sigma_matches_cpu(self):
        _require_gpu()
        img = _random_rgb((400, 700, 3), seed=5)
        cpu = fast_gaussian_filter(img, 1.2)
        with gpu.use(True):
            dev = fast_gaussian_filter(img, 1.2)
        # Same FIR kernel, same boundary mode — different summation order.
        np.testing.assert_allclose(dev, cpu, rtol=1e-10, atol=1e-14)

    def test_per_channel_sigma_matches_cpu(self):
        _require_gpu()
        img = _random_rgb((400, 700, 3), seed=6)
        sigmas = np.array([0.7, 1.3, 2.1])
        cpu = fast_gaussian_filter(img, sigmas)
        with gpu.use(True):
            dev = fast_gaussian_filter(img, sigmas)
        np.testing.assert_allclose(dev, cpu, rtol=1e-10, atol=1e-14)

    def test_large_sigma_matches_cpu_within_iir_tolerance(self):
        _require_gpu()
        img = _random_rgb((400, 700, 3), seed=8)
        cpu = fast_gaussian_filter(img, 6.0)  # CPU: Young-van Vliet IIR
        with gpu.use(True):
            dev = fast_gaussian_filter(img, 6.0)  # GPU: exact FIR
        # CPU IIR documents ~1e-3 max error vs the analytic Gaussian;
        # the GPU FIR sits at ~1e-12. Tolerance vs CPU therefore 2e-3.
        np.testing.assert_allclose(dev, cpu, rtol=2e-3, atol=1e-6)

    def test_2d_input_matches_cpu(self):
        _require_gpu()
        img = _random_rgb((400, 700), seed=9)
        cpu = fast_gaussian_filter(img, 1.0)
        with gpu.use(True):
            dev = fast_gaussian_filter(img, 1.0)
        np.testing.assert_allclose(dev, cpu, rtol=1e-10, atol=1e-14)

    def test_small_image_stays_on_cpu(self):
        _require_gpu()
        img = _random_rgb((64, 80, 3), seed=10)
        with gpu.use(True):
            out = fast_gaussian_filter(img, 1.0)
        assert type(out) is np.ndarray  # below _GPU_MIN_PIXELS


class TestGpuEndToEnd:
    @pytest.mark.slow
    def test_simulate_gpu_tracks_cpu(self):
        _require_gpu()
        from spektrafilm.runtime import digest_params, init_params, simulate

        rng = np.random.default_rng(42)
        img = np.clip(rng.random((300, 450, 3)) * 0.9 + 0.05, 0, 1)

        params = init_params(print_profile="kodak_portra_endura")
        params.io.upscale_factor = 1.0
        params.settings.use_fast_stats = True
        params.settings.use_enlarger_lut = True
        params.settings.use_scanner_lut = True
        params.debug.deactivate_stochastic_effects = True  # deterministic
        params = digest_params(params)

        cpu = simulate(img, params, digest_params_first=False)

        params.settings.gpu = True
        gpu_out = simulate(img, params, digest_params_first=False)

        # Grain and other stochastic stages stay on CPU; the GPU covers
        # the gamut compression + blur chain in float64.
        np.testing.assert_allclose(gpu_out, cpu, rtol=1e-6, atol=1e-8)


_REF_FILE = __import__("pathlib").Path(__file__).parent / "baselines" / "gpu_cam16_reference.npz"

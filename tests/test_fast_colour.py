import numpy as np
import pytest

import colour

from spektrafilm.utils.fast_cam16 import (
    _viewing_conditions,
    cam16ucs_to_xyz,
    xyz_to_cam16ucs,
)
from spektrafilm.utils.fast_conversions import rgb_to_xyz, xyz_to_rgb
from spektrafilm.utils.gamut_compression import compress_rgb_cam16ucs_chroma

pytestmark = pytest.mark.unit

_XYZ_W = colour.xy_to_XYZ(colour.RGB_COLOURSPACES["sRGB"].whitepoint)
_VC = _viewing_conditions(_XYZ_W, 64.0, 20.0)


def _sample_xyz(rng, n):
    return rng.uniform(0.0, 1.2, (n, 3))


def test_fast_cam16_forward_matches_colour_science() -> None:
    rng = np.random.default_rng(7)
    xyz = _sample_xyz(rng, 20_000)
    jab_ref = np.asarray(colour.XYZ_to_CAM16UCS(xyz, XYZ_w=_XYZ_W, L_A=64.0, Y_b=20.0))
    jab_fast = xyz_to_cam16ucs(xyz, _VC)
    # NaN patterns must agree elementwise (out-of-domain inputs).
    np.testing.assert_array_equal(np.isfinite(jab_ref), np.isfinite(jab_fast))
    both = np.isfinite(jab_ref) & np.isfinite(jab_fast)
    np.testing.assert_allclose(
        jab_fast[both], jab_ref[both], rtol=1e-9, atol=1e-9
    )


def test_fast_cam16_inverse_matches_colour_science() -> None:
    rng = np.random.default_rng(7)
    xyz = _sample_xyz(rng, 20_000)
    jab = np.asarray(colour.XYZ_to_CAM16UCS(xyz, XYZ_w=_XYZ_W, L_A=64.0, Y_b=20.0))
    xyz_ref = np.asarray(colour.CAM16UCS_to_XYZ(jab, XYZ_w=_XYZ_W, L_A=64.0, Y_b=20.0))
    xyz_fast = cam16ucs_to_xyz(jab, _VC)
    np.testing.assert_array_equal(np.isfinite(xyz_ref), np.isfinite(xyz_fast))
    both = np.isfinite(xyz_ref) & np.isfinite(xyz_fast)
    np.testing.assert_allclose(
        xyz_fast[both], xyz_ref[both], rtol=1e-8, atol=1e-11
    )


def test_compress_rgb_cam16ucs_chroma_matches_colour_science() -> None:
    rng = np.random.default_rng(11)
    rgb = rng.uniform(0.0, 1.0, (12_000, 3))
    kwargs = dict(
        threshold=0.8,
        limit=0.6,
        power=0.9,
        lightness_compression=(0.02, 1.4, 0.7),
    )
    ref = np.asarray(compress_rgb_cam16ucs_chroma(rgb, "sRGB", **kwargs))
    assert np.isfinite(ref).all()

    # Cross-check the fast path against the legacy colour-science path
    # (monkeypatch the CAM16 conversion functions back to colour).
    import spektrafilm.utils.gamut_compression as gc

    xyz_w = gc._output_cs_whitepoint_xyz("sRGB")
    cam_kwargs = dict(XYZ_w=xyz_w, L_A=gc._CAM16UCS_L_A, Y_b=gc._CAM16UCS_Y_B)

    def legacy_forward(xyz, conditions=None):
        return np.asarray(colour.XYZ_to_CAM16UCS(xyz, **cam_kwargs))

    def legacy_inverse(jab, conditions=None):
        return np.asarray(colour.CAM16UCS_to_XYZ(jab, **cam_kwargs))

    original = gc.xyz_to_cam16ucs, gc.cam16ucs_to_xyz
    gc.xyz_to_cam16ucs = legacy_forward
    gc.cam16ucs_to_xyz = legacy_inverse
    try:
        legacy = compress_rgb_cam16ucs_chroma(rgb, "sRGB", **kwargs)
    finally:
        gc.xyz_to_cam16ucs, gc.cam16ucs_to_xyz = original
    np.testing.assert_array_equal(np.isfinite(ref), np.isfinite(legacy))
    both = np.isfinite(ref) & np.isfinite(legacy)
    np.testing.assert_allclose(ref[both], legacy[both], rtol=1e-9, atol=1e-11)


def test_fast_conversions_rgb_xyz_matrices() -> None:
    rng = np.random.default_rng(13)
    rgb = rng.uniform(0.0, 1.0, (8_000, 3))
    xyz = rng.uniform(0.0, 1.2, (8_000, 3))
    illu = np.array([0.328, 0.344])

    # Plain conversion with CCTF decoding.
    ref = np.asarray(colour.RGB_to_XYZ(rgb, colourspace="sRGB", apply_cctf_decoding=True))
    np.testing.assert_allclose(
        rgb_to_xyz(rgb, "sRGB", apply_cctf_decoding=True), ref, rtol=1e-12, atol=1e-14
    )

    # Chromatic adaptation (what scanning._density_to_rgb uses; colour
    # default adaptation is CAT02).
    ref = np.asarray(
        colour.XYZ_to_RGB(xyz, colourspace="sRGB", apply_cctf_encoding=False, illuminant=illu)
    )
    np.testing.assert_allclose(
        xyz_to_rgb(xyz, "sRGB", illuminant_xy=illu), ref, rtol=1e-12, atol=1e-14
    )

    # CAT16 adaptation (what spectral upsampling uses).
    ref = np.asarray(
        colour.RGB_to_XYZ(
            rgb,
            colourspace="ITU-R BT.2020",
            apply_cctf_decoding=True,
            illuminant=illu,
            chromatic_adaptation_transform="CAT16",
        )
    )
    np.testing.assert_allclose(
        rgb_to_xyz(
            rgb,
            "ITU-R BT.2020",
            apply_cctf_decoding=True,
            illuminant_xy=illu,
            adaptation="CAT16",
        ),
        ref,
        rtol=1e-12,
        atol=1e-14,
    )


def test_fast_conversions_caching() -> None:
    from spektrafilm.utils.fast_conversions import _MATRIX_CACHE

    rgb = np.ones((3, 3))
    rgb_to_xyz(rgb, "sRGB")
    keys_before = set(_MATRIX_CACHE)
    rgb_to_xyz(rgb, "sRGB")
    assert set(_MATRIX_CACHE) == keys_before

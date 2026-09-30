"""Specialized fused CAM16-UCS forward/inverse for fixed viewing conditions.

The output gamut compression (:mod:`spektrafilm.utils.gamut_compression`,
``algorithm='cam16ucs'``) runs the full CAM16 forward and inverse on every
output pixel. Going through :mod:`colour` costs ~4x more time than the raw
math because the generic implementations also compute correlates the UCS
path never uses (Q, s, C/H quadrature), and pay per-call wrapper overhead
(domain scaling, tstack/tsplit copies, generic vecmul/spow).

This module re-implements the exact same chain (Li et al. 2017; CAM16 per
Li et al. 2017 with CAT16) for the fixed viewing conditions used by the
gamut compressor (Average surround, discount_illuminant=False, hue
quadrature skipped), fused into flat vectorized numpy ops:

    - viewing-condition constants are precomputed once and cached
    - RGB<->XYZ and XYZ->RGB use the colourspace matrices directly
    - every step is elementwise or a (3, 3) matmul

The op shape (elementwise + small matmuls) is intentionally the same as
the GPU equivalent, so a CuPy port is a drop-in `numpy` -> `cupy` swap.

Numerics: formula-for-formula match of :func:`colour.XYZ_to_CAM16UCS` /
:func:`colour.CAM16UCS_to_XYZ` at domain-range scale ``reference``
(XYZ in [0, 1]-ish units, J' in 0..100), including sign-preserving
``spow`` semantics for out-of-gamut negative values. Validated against
colour-science to ~1e-12 relative; see tests.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# CAT16 forward matrix (colour.science: colour.appearance.cam16 MATRIX_16).
_MATRIX_16 = np.array(
    [
        [0.401288, 0.650173, -0.051461],
        [-0.250268, 1.204414, 0.045854],
        [-0.002079, 0.048952, 0.953127],
    ]
)
_MATRIX_16_INV = np.linalg.inv(_MATRIX_16)

# CAM16-UCS coefficients (Li et al. 2017, "CAM02-UCS" set).
_UCS_C1 = 0.007
_UCS_C2 = 0.0228

# CAM16 Average surround induction factors.
_SURROUND_F = 1.0
_SURROUND_C = 0.69
_SURROUND_N_C = 1.0

# Inverse opponent-dimensions constants (CIECAM02/CAM16 step).
_INV_460_1403 = 460.0 / 1403.0
_INV_220_1403 = 220.0 / 1403.0
_INV_27_1403 = 27.0 / 1403.0
_INV_6300_1403 = 6300.0 / 1403.0
_P3 = 21.0 / 20.0

_EPSILON = 2.220446049250313e-16


def _spow(a: np.ndarray, p: float) -> np.ndarray:
    """Sign-preserving power, matching colour.science's ``spow``."""
    return np.sign(a) * np.abs(a) ** p


@dataclass(frozen=True)
class Cam16UcsConditions:
    """Precomputed CAM16 viewing-condition constants (100-domain)."""

    d_rgb: np.ndarray  # (3,) per-channel adaptation gains
    fl: float  # luminance level adaptation factor F_L
    n_bb: float
    n_cb: float
    z: float
    a_w: float  # achromatic response of the whitepoint
    fl_025: float  # F_L ** 0.25
    n_pow: float  # (1.64 - 0.29 ** n) ** 0.73
    c_z: float  # surround c * z
    inv_c_z: float  # 1 / (c * z)

    @property
    def chroma_adaptation_row(self) -> np.ndarray:
        """D_RGB * Y_w / RGB_w per channel (kept for completeness)."""
        return self.d_rgb


def _viewing_conditions(xyz_w: np.ndarray, la: float, y_b: float) -> Cam16UcsConditions:
    """Precompute the constant part of CAM16 for fixed conditions.

    ``xyz_w`` is the whitepoint in [0, 1]-domain units; internally the
    100-domain is used, matching colour-science's reference scale.
    """
    xyz_w = np.asarray(xyz_w, dtype=np.float64) * 100.0
    y_w = float(xyz_w[1])

    rgb_w = _MATRIX_16 @ xyz_w

    # Degree of adaptation D = clip(F * (1 - (1/3.6) * exp((-L_A - 42) / 92)), 0, 1)
    d = float(np.clip(_SURROUND_F * (1.0 - (1.0 / 3.6) * np.exp((-la - 42.0) / 92.0)), 0.0, 1.0))

    n = y_b / y_w
    k = 1.0 / (5.0 * la + 1.0)
    k4 = k**4
    fl = 0.2 * k4 * (5.0 * la) + 0.1 * (1.0 - k4) ** 2 * (5.0 * la) ** (1.0 / 3.0)
    n_bb = 0.725 * (1.0 / n) ** 0.2
    n_cb = n_bb
    z = 1.48 + np.sqrt(n)

    d_rgb = d * y_w / rgb_w + 1.0 - d
    rgb_wc = d_rgb * rgb_w
    rgb_aw = _response_compression_forward(rgb_wc, fl)
    a_w = float((2.0 * rgb_aw[0] + rgb_aw[1] + rgb_aw[2] / 20.0 - 0.305) * n_bb)

    return Cam16UcsConditions(
        d_rgb=d_rgb,
        fl=fl,
        n_bb=n_bb,
        n_cb=n_cb,
        z=z,
        a_w=a_w,
        fl_025=fl**0.25,
        n_pow=(1.64 - 0.29**n) ** 0.73,
        c_z=_SURROUND_C * z,
        inv_c_z=1.0 / (_SURROUND_C * z),
    )


def _response_compression_forward(rgb: np.ndarray, fl: float) -> np.ndarray:
    """Post-adaptation non-linear response compression (forward)."""
    fl_rgb = _spow(fl * np.abs(rgb) / 100.0, 0.42)
    return (400.0 * np.sign(rgb) * fl_rgb) / (27.13 + fl_rgb) + 0.1


def _response_compression_inverse(rgb_a: np.ndarray, fl: float) -> np.ndarray:
    """Post-adaptation non-linear response compression (inverse)."""
    delta = np.abs(rgb_a - 0.1)
    return (
        np.sign(rgb_a - 0.1)
        * 100.0
        / fl
        * _spow((27.13 * delta) / (400.0 - delta), 1.0 / 0.42)
    )


def xyz_to_cam16ucs(xyz: np.ndarray, vc: Cam16UcsConditions) -> np.ndarray:
    """XYZ (0..1-domain) -> CAM16-UCS J'a'b' (J' in 0..100).

    Matches :func:`colour.XYZ_to_CAM16UCS` at reference domain-range
    scale with Average surround, ``discount_illuminant=False``.
    """
    xyz100 = np.asarray(xyz, dtype=np.float64) * 100.0

    rgb = xyz100 @ _MATRIX_16.T
    rgb_a = _response_compression_forward(vc.d_rgb * rgb, vc.fl)

    r, g, b = rgb_a[..., 0], rgb_a[..., 1], rgb_a[..., 2]
    a = r - 12.0 * g / 11.0 + b / 11.0
    b_dim = (r + g - 2.0 * b) / 9.0

    h = np.degrees(np.arctan2(b_dim, a)) % 360.0
    h_rad = np.radians(h)
    e_t = 0.25 * (np.cos(2.0 + h_rad) + 3.8)

    achro = (2.0 * r + g + b / 20.0 - 0.305) * vc.n_bb
    j = 100.0 * _spow(achro / vc.a_w, vc.c_z)

    # t = (50000/13) * N_c * N_cb * e_t * |ab| / (Ra + Ga + 21 * Ba / 20)
    denom = r + g + 21.0 * b / 20.0
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (50000.0 / 13.0) * _SURROUND_N_C * vc.n_cb * e_t * np.sqrt(
            a * a + b_dim * b_dim
        ) / denom
        chroma = _spow(t, 0.9) * _spow(j / 100.0, 0.5) * vc.n_pow
    m = chroma * vc.fl_025

    j_p = (1.0 + 100.0 * _UCS_C1) * j / (1.0 + _UCS_C1 * j)
    m_p = np.log1p(_UCS_C2 * m) / _UCS_C2
    a_p = m_p * np.cos(h_rad)
    b_p = m_p * np.sin(h_rad)

    return np.stack([j_p, a_p, b_p], axis=-1)


def cam16ucs_to_xyz(jab: np.ndarray, vc: Cam16UcsConditions) -> np.ndarray:
    """CAM16-UCS J'a'b' (J' in 0..100) -> XYZ (0..1-domain).

    Matches :func:`colour.CAM16UCS_to_XYZ` at reference domain-range
    scale with Average surround, ``discount_illuminant=False``.
    """
    jab = np.asarray(jab, dtype=np.float64)
    j_p = jab[..., 0]
    a_p = jab[..., 1]
    b_p = jab[..., 2]

    # UCS -> JMh
    j = -j_p / (_UCS_C1 * j_p - 1.0 - 100.0 * _UCS_C1)
    m_p = np.hypot(a_p, b_p)
    h = np.degrees(np.arctan2(b_p, a_p)) % 360.0
    m = np.expm1(_UCS_C2 * m_p) / _UCS_C2

    # JMh -> XYZ (CAM16 inverse)
    c = m / vc.fl_025
    h_rad = np.radians(h)
    e_t = 0.25 * (np.cos(2.0 + h_rad) + 3.8)

    j_safe = np.maximum(j, _EPSILON)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = _spow(
            c / (np.sqrt(j_safe / 100.0) * vc.n_pow),
            1.0 / 0.9,
        )
    achro = vc.a_w * _spow(j / 100.0, vc.inv_c_z)

    p_1 = (50000.0 / 13.0) * _SURROUND_N_C * vc.n_cb * e_t / t
    p_2 = achro / vc.n_bb + 0.305

    sin_hr = np.sin(h_rad)
    cos_hr = np.cos(h_rad)
    with np.errstate(divide="ignore", invalid="ignore"):
        cos_over_sin = cos_hr / sin_hr
        sin_over_cos = sin_hr / cos_hr
        p_4 = p_1 / sin_hr
        p_5 = p_1 / cos_hr

        n_qty = p_2 * (2.0 + _P3) * _INV_460_1403
        # Two complementary (for non-NaN) masks, mirroring colour.science:
        # a NaN hue satisfies neither, so a/b stay 0 — matching the
        # reference implementation's fall-through to zeros.
        sin_dominates = np.abs(sin_hr) >= np.abs(cos_hr)
        cos_dominates = np.abs(sin_hr) < np.abs(cos_hr)

        b_dim = np.where(
            sin_dominates,
            n_qty
            / (
                p_4
                + (2.0 + _P3) * _INV_220_1403 * cos_over_sin
                - _INV_27_1403
                + _P3 * _INV_6300_1403
            ),
            0.0,
        )
        a = np.where(sin_dominates, b_dim * cos_over_sin, 0.0)
        a = np.where(
            cos_dominates,
            n_qty
            / (
                p_5
                + (2.0 + _P3) * _INV_220_1403
                - (_INV_27_1403 - _P3 * _INV_6300_1403) * sin_over_cos
            ),
            a,
        )
        b_dim = np.where(cos_dominates, a * sin_over_cos, b_dim)

    # t == 0 -> zero opponent dimensions (matches colour's masking).
    ab_mask = np.where(t == 0.0, 0.0, 1.0)
    a = a * ab_mask
    b_dim = b_dim * ab_mask

    # matrix_post_adaptation_non_linear_response_compression
    rgb_a = (
        np.stack(
            [
                460.0 * p_2 + 451.0 * a + 288.0 * b_dim,
                460.0 * p_2 - 891.0 * a - 261.0 * b_dim,
                460.0 * p_2 - 220.0 * a - 6300.0 * b_dim,
            ],
            axis=-1,
        )
        / 1403.0
    )

    rgb_c = _response_compression_inverse(rgb_a, vc.fl)
    rgb = rgb_c / vc.d_rgb
    xyz100 = rgb @ _MATRIX_16_INV.T
    return xyz100 / 100.0

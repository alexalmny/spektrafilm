"""Pixel-local FP64 CAM16-UCS gamut compression (Li & Luo 2017).

The Colour implementation in gamut_compression is the reference. Combine
only the fixed transforms and compute useful correlates (J, M, h) per pixel.
"""
from __future__ import annotations

import math

import colour
import numpy as np
from colour.appearance.cam16 import MATRIX_16, MATRIX_INVERSE_16, VIEWING_CONDITIONS_CAM16
from colour.appearance.ciecam02 import (
    achromatic_response_forward,
    degree_of_adaptation,
    post_adaptation_non_linear_response_compression_forward,
    viewing_conditions_dependent_parameters,
)
from colour.constants import EPSILON
from numba import njit, prange


_C1 = 0.007
_C2 = 0.0228
_P3 = 21.0 / 20.0


def prepare(output_space, white_xyz, L_A, Y_b):
    """Fold fixed CAT16 adaptation and RGB/XYZ matrices into the two transforms."""
    white = colour.RGB_COLOURSPACES[output_space].whitepoint
    basis = np.eye(3)
    to_xyz = colour.RGB_to_XYZ(basis, output_space, illuminant=white).T
    from_xyz = colour.XYZ_to_RGB(basis, output_space, illuminant=white).T
    surround = VIEWING_CONDITIONS_CAM16['Average']
    white_xyz100 = white_xyz * 100.0
    white_cat = MATRIX_16 @ white_xyz100
    white_Y = white_xyz100[1]
    D = np.clip(degree_of_adaptation(surround.F, L_A), 0, 1)
    D_rgb = D * white_Y / white_cat + 1 - D
    n, F_L, N_bb, N_cb, z = viewing_conditions_dependent_parameters(Y_b, white_Y, L_A)
    white_adapted = post_adaptation_non_linear_response_compression_forward(D_rgb * white_cat, F_L)
    A_white = achromatic_response_forward(white_adapted, N_bb)
    # Fold F_L and the normalized XYZ convention into the matrices.
    forward = np.ascontiguousarray((F_L * D_rgb)[:, None] * (MATRIX_16 @ to_xyz))
    inverse = np.ascontiguousarray((from_xyz @ MATRIX_INVERSE_16) / (F_L * D_rgb))
    constants = (A_white, N_bb, surround.c * z, (50000.0 / 13.0) * surround.N_c * N_cb,
                 (1.64 - 0.29**n)**0.73 * F_L**0.25)
    return forward, inverse, tuple(float(value) for value in constants)


@njit(cache=True, inline='always')
def _signed_power(x, exponent):
    return math.copysign(abs(x)**exponent, x)


@njit(cache=True, inline='always')
def _reinhard(x, knee):
    threshold, limit, power = knee
    if x > threshold:
        scale = limit - threshold
        u = (x - threshold) / scale
        return threshold + scale * (u / (1.0 + u**power)**(1.0 / power))
    return x


@njit(cache=True, inline='always')
def _row(matrix, i, r, g, b):
    return matrix[i, 0] * r + matrix[i, 1] * g + matrix[i, 2] * b


@njit(cache=True, inline='always')
def _adapt(channel):
    u = abs(channel)**0.42
    return 400.0 * math.copysign(u, channel) / (27.13 + u) + 0.1


@njit(cache=True, inline='always')
def _unadapt(channel):
    delta = channel - 0.1
    magnitude = 27.13 * abs(delta) / (400.0 - abs(delta))
    return math.copysign(1.0, delta) * _signed_power(magnitude, 1.0 / 0.42)


@njit(cache=True, inline='always')
def _boundary(Jp, h, J_min, J_max, h_min, h_step, table):
    # Same periodic hue and clamped lightness bilinear interpolation as the reference.
    j = (min(max(Jp, J_min), J_max) - J_min) / (J_max - J_min) * (table.shape[0] - 1)
    k = (h - h_min) / h_step
    j0 = min(math.floor(j), table.shape[0] - 2)
    h0 = math.floor(k) % table.shape[1]
    h1 = (h0 + 1) % table.shape[1]
    fj, fh = j - j0, k - math.floor(k)
    return (table[j0, h0] * (1.0 - fj) * (1.0 - fh) + table[j0, h1] * (1.0 - fj) * fh
            + table[j0 + 1, h0] * fj * (1.0 - fh) + table[j0 + 1, h1] * fj * fh)


@njit(cache=True, error_model='numpy')
def _pixel(r, g, b, forward, inverse, viewing, knee, lightness, J_white, J_min, J_max, h_min, h_step, table):
    A_white, N_bb, exp_J, factor_t, factor_M = viewing
    Ra = _adapt(_row(forward, 0, r, g, b))
    Ga = _adapt(_row(forward, 1, r, g, b))
    Ba = _adapt(_row(forward, 2, r, g, b))
    a, b = Ra - 12.0 * Ga / 11.0 + Ba / 11.0, (Ra + Ga - 2.0 * Ba) / 9.0
    h = math.atan2(b, a)
    e = 0.25 * (math.cos(2.0 + h) + 3.8)
    A = (2.0 * Ra + Ga + Ba / 20.0 - 0.305) * N_bb
    J = 100.0 * _signed_power(A / A_white, exp_J)
    t = e * math.sqrt(a * a + b * b) / (Ra + Ga + _P3 * Ba)
    t = factor_t * t if math.isfinite(t) else 0.0
    M = _signed_power(t, 0.9) * _signed_power(J / 100.0, 0.5) * factor_M

    Jp = (1.0 + 100.0 * _C1) * J / (1.0 + _C1 * J)
    if J_white != 0.0:
        Jp = _reinhard(Jp / J_white, lightness) * J_white
    Mp = math.log1p(_C2 * M) / _C2
    # The reference constructs cartesian UCS before looking up polar hue.
    ap, bp = Mp * math.cos(h), Mp * math.sin(h)
    hue = math.atan2(bp, ap)
    if not (math.isfinite(Jp) and math.isfinite(Mp)):
        return np.nan, np.nan, np.nan
    cp_max = np.fmax(1e-9, _boundary(Jp, hue, J_min, J_max, h_min, h_step, table))
    cp = _reinhard(math.hypot(ap, bp) / cp_max, knee) * cp_max

    J = -Jp / (_C1 * Jp - 1.0 - 100.0 * _C1)
    a, b = cp * math.cos(hue), cp * math.sin(hue)
    h = math.atan2(b, a)
    M = math.expm1(_C2 * math.hypot(a, b)) / _C2
    t = _signed_power(M / (math.sqrt(max(J, EPSILON) / 100.0) * factor_M), 1.0 / 0.9)
    P2 = A_white * _signed_power(J / 100.0, 1.0 / exp_J) / N_bb + 0.305
    if t == 0.0:
        a = b = 0.0
    else:
        si, co = math.sin(h), math.cos(h)
        P1 = factor_t * (0.25 * (math.cos(2.0 + h) + 3.8)) / t
        # Solve the reference opponent-coordinate equations for radial magnitude,
        # eliminating both divisions by sin/cos and their quadrant branch.
        radius = P2 * (2.0 + _P3) * (460.0 / 1403.0) / (
            P1 + (2.0 + _P3) * (220.0 / 1403.0) * co + (_P3 * (6300.0 / 1403.0) - 27.0 / 1403.0) * si)
        a, b = radius * co, radius * si
    R = _unadapt((460.0 * P2 + 451.0 * a + 288.0 * b) / 1403.0)
    G = _unadapt((460.0 * P2 - 891.0 * a - 261.0 * b) / 1403.0)
    B = _unadapt((460.0 * P2 - 220.0 * a - 6300.0 * b) / 1403.0)
    return _row(inverse, 0, R, G, B), _row(inverse, 1, R, G, B), _row(inverse, 2, R, G, B)


@njit(cache=True, parallel=True)
def compress(rgb, forward, inverse, viewing, knee, lightness, J_white, J_min, J_max, h_min, h_step, table):
    result = np.empty_like(rgb)
    invalid = 0
    for i in prange(rgb.shape[0]):
        r, g, b = _pixel(rgb[i, 0], rgb[i, 1], rgb[i, 2], forward, inverse, viewing,
                         knee, lightness, J_white, J_min, J_max, h_min, h_step, table)
        result[i, 0], result[i, 1], result[i, 2] = r, g, b
        invalid += int(not (math.isfinite(r) and math.isfinite(g) and math.isfinite(b)))
    return result, invalid

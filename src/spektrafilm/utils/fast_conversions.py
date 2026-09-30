"""Fast RGB<->XYZ conversions with precomputed (cached) fused matrices.

`colour.RGB_to_XYZ` / `colour.XYZ_to_RGB` pay heavy per-call wrapper
overhead (validate_method, dict juggling, domain-scale contexts, and a
generic `vecmul` per matrix) which dominates on multi-megapixel images.
The actual math is always:

    optional CCTF decode -> one 3x3 matrix -> optional CAT matrix
    -> one 3x3 matrix -> optional CCTF encode

The matrix product is constant for a given (colourspace, illuminant,
adaptation) triple, so it is computed once with colour-science itself
(scalar-sized work) and cached; the per-pixel work reduces to a single
`rgb @ M.T` plus the elementwise CCTF, which is also what a GPU port
(CuPy) needs.

Matrix construction and CCTF semantics match colour-science's
`RGB_to_XYZ` / `XYZ_to_RGB` / `RGB_to_RGB` formula-for-formula.
"""

from __future__ import annotations

import numpy as np

import colour
from colour.adaptation import matrix_chromatic_adaptation_VonKries

_MATRIX_CACHE: dict[tuple, np.ndarray] = {}


def _von_kries(source_xy: np.ndarray, target_xy: np.ndarray, transform: str) -> np.ndarray:
    """CAT matrix from source whitepoint xy to target whitepoint xy."""
    return matrix_chromatic_adaptation_VonKries(
        colour.xyY_to_XYZ(colour.xy_to_xyY(source_xy)),
        colour.xyY_to_XYZ(colour.xy_to_xyY(target_xy)),
        transform=transform,
    )


def _build_matrix(
    direction: str,
    color_space: str,
    illuminant_xy: np.ndarray | None,
    adaptation: str,
) -> np.ndarray:
    """Build the fused matrix for one (direction, space, illuminant,
    adaptation) combination. Scalar-sized work; called once per key."""
    cs = colour.RGB_COLOURSPACES[color_space]
    if direction == "rgb_to_xyz":
        m = np.asarray(cs.matrix_RGB_to_XYZ)
        if illuminant_xy is not None:
            m = _von_kries(cs.whitepoint, illuminant_xy, adaptation) @ m
    else:
        m = np.asarray(cs.matrix_XYZ_to_RGB)
        if illuminant_xy is not None:
            m = m @ _von_kries(illuminant_xy, cs.whitepoint, adaptation)
    return m


def rgb_to_xyz_matrix(
    color_space: str,
    illuminant_xy: np.ndarray | None = None,
    adaptation: str = "CAT02",
) -> np.ndarray:
    """Fused RGB->XYZ matrix for a colourspace, optionally adapted from
    the colourspace whitepoint to ``illuminant_xy``.

    Matches ``colour.RGB_to_XYZ`` matrix semantics (default adaptation
    ``CAT02`` when an illuminant is given).
    """
    illuminant_key = None if illuminant_xy is None else np.asarray(illuminant_xy).tobytes()
    key = ("rgb_to_xyz", color_space, illuminant_key, adaptation)
    m = _MATRIX_CACHE.get(key)
    if m is None:
        m = _build_matrix("rgb_to_xyz", color_space, illuminant_xy, adaptation)
        _MATRIX_CACHE[key] = m
    return m


def xyz_to_rgb_matrix(
    color_space: str,
    illuminant_xy: np.ndarray | None = None,
    adaptation: str = "CAT02",
) -> np.ndarray:
    """Fused XYZ->RGB matrix for a colourspace, optionally adapted from
    ``illuminant_xy`` back to the colourspace whitepoint.

    Matches ``colour.XYZ_to_RGB`` matrix semantics.
    """
    illuminant_key = None if illuminant_xy is None else np.asarray(illuminant_xy).tobytes()
    key = ("xyz_to_rgb", color_space, illuminant_key, adaptation)
    m = _MATRIX_CACHE.get(key)
    if m is None:
        m = _build_matrix("xyz_to_rgb", color_space, illuminant_xy, adaptation)
        _MATRIX_CACHE[key] = m
    return m


def rgb_to_xyz(
    rgb: np.ndarray,
    color_space: str,
    apply_cctf_decoding: bool = False,
    illuminant_xy: np.ndarray | None = None,
    adaptation: str = "CAT02",
) -> np.ndarray:
    """Vectorized `colour.RGB_to_XYZ` equivalent (see module docstring)."""
    if apply_cctf_decoding:
        cs = colour.RGB_COLOURSPACES[color_space]
        if cs.cctf_decoding is not None:
            rgb = cs.cctf_decoding(rgb)
    return np.asarray(rgb) @ rgb_to_xyz_matrix(color_space, illuminant_xy, adaptation).T


def xyz_to_rgb(
    xyz: np.ndarray,
    color_space: str,
    apply_cctf_encoding: bool = False,
    illuminant_xy: np.ndarray | None = None,
    adaptation: str = "CAT02",
) -> np.ndarray:
    """Vectorized `colour.XYZ_to_RGB` equivalent (see module docstring)."""
    rgb = np.asarray(xyz) @ xyz_to_rgb_matrix(color_space, illuminant_xy, adaptation).T
    if apply_cctf_encoding:
        cs = colour.RGB_COLOURSPACES[color_space]
        if cs.cctf_encoding is not None:
            rgb = cs.cctf_encoding(rgb)
    return rgb

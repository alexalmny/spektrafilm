from __future__ import annotations

from enum import Enum

from spektrafilm.model.diffusion import DIFFUSION_FILTER_FAMILIES


class RGBColorSpaces(Enum):
    sRGB = "sRGB"
    DCI_P3 = "DCI-P3"
    DisplayP3 = "Display P3"
    AdobeRGB = "Adobe RGB (1998)"
    ITU_R_BT2020 = "ITU-R BT.2020"
    ProPhotoRGB = "ProPhoto RGB"
    ACES2065_1 = "ACES2065-1"


class RGBtoRAWMethod(Enum):
    hanatos2025 = "hanatos2025"
    mallett2019 = "mallett2019"


class RawWhiteBalance(Enum):
    as_shot = "as_shot"
    daylight = "daylight"
    tungsten = "tungsten"
    custom = "custom"


class AutoExposureMethods(Enum):
    center_weighted = "center_weighted"
    matrix = "matrix"
    multi_zone = "multi_zone"
    partial = "partial"
    highlight_weighted = "highlight_weighted"
    median = "median"
    average = "average"


class NapariInterpolationModes(Enum):
    nearest = "nearest"
    linear = "linear"
    cubic = "cubic"
    spline16 = "spline16"
    spline36 = "spline36"
    lanczos = "lanczos"
    blackman = "blackman"


# Derived from the model's family table so the GUI can never drift from it.
DiffusionFilterFamilies = Enum("DiffusionFilterFamilies", {n: n for n in DIFFUSION_FILTER_FAMILIES})


class InputGamutCompressAlgorithms(Enum):
    xy = "xy"
    oklch = "oklch"


class OutputGamutCompressAlgorithms(Enum):
    off = "off"
    oklch = "oklch"
    aces_rgc = "aces_rgc"
    oklrab = "oklrab"
    jzazbz = "jzazbz"
    cam16ucs = "cam16ucs"



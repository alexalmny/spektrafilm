from __future__ import annotations

from typing import Callable
import numpy as np

from spektrafilm.profiles.io import Hanatos2025SensitivityAdaptation
from spektrafilm.utils.gamut_compression import InputGamutCompressSpec
from spektrafilm.utils.lut import compute_with_lut
from spektrafilm.utils.spectral_upsampling import compute_hanatos2025_tc_lut
from spektrafilm.utils.timings import timeit


class SpectralLUTService:
    def __init__(self, lut_resolution: int):
        self._lut_resolution = lut_resolution

        self.timings = {}
        self.hanatos2025_adaptation = None # to be set by filming stage with info from film profile and settings
        # Input gamut compression spec (set by the pipeline from
        # params.io.input_gamut_compress). Drives the build-time bake
        # of the filming tc_lut; changes invalidate the cache below.
        self.input_gamut_compress: InputGamutCompressSpec = InputGamutCompressSpec()

        # external memory
        self.filming_tc_lut_memory : np.ndarray | None = None # tc_lut memory
        self._spectral_luts: dict[str, tuple[tuple[np.ndarray, ...], np.ndarray]] = {}

        # local memory
        self._film_sensitivity = None # to track if tc_lut needs to be recomputed when film sensitivity changes
        self._cached_filming_adaptation = None # full adaptation state for which the cached tc_lut was computed
        self._cached_input_gamut_compress: InputGamutCompressSpec | None = None

        # shortcut: probes identify the built-in spectral transforms; use explicit keys if adding arbitrary transforms.
        self._cmy_test_values = np.array([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
                                          [[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]]]) # to test if LUTs are identical

    def set_hanatos2025_adaptation(self, adaptation: Hanatos2025SensitivityAdaptation) -> None:
        adaptation_copy = self._copy_hanatos2025_adaptation(adaptation)
        self.hanatos2025_adaptation = adaptation_copy
        if not self._same_hanatos2025_adaptation(self._cached_filming_adaptation, adaptation_copy):
            self.filming_tc_lut_memory = None
            self._film_sensitivity = None
            self._cached_filming_adaptation = None

    def set_input_gamut_compress(self, spec: InputGamutCompressSpec) -> None:
        """Set the input-gamut-compression spec used when baking the
        filming tc_lut. A change invalidates the cached LUT.
        """
        if spec != self.input_gamut_compress:
            self.input_gamut_compress = spec
            # Force a rebuild of the filming tc_lut on the next call.
            self.filming_tc_lut_memory = None
            self._film_sensitivity = None
            self._cached_filming_adaptation = None
            self._cached_input_gamut_compress = None

    @staticmethod
    def _copy_hanatos2025_adaptation(
        adaptation: Hanatos2025SensitivityAdaptation | None,
    ) -> Hanatos2025SensitivityAdaptation | None:
        if adaptation is None:
            return None
        return Hanatos2025SensitivityAdaptation(
            window_params=np.array(adaptation.window_params, copy=True),
            surface_params=np.array(adaptation.surface_params, copy=True),
            spectral_gaussian_blur=float(adaptation.spectral_gaussian_blur),
            reference_illuminant=adaptation.reference_illuminant,
            apply_window=bool(adaptation.apply_window),
            apply_surface=bool(adaptation.apply_surface),
            active=adaptation.active,
        )

    @staticmethod
    def _same_hanatos2025_adaptation(
        left: Hanatos2025SensitivityAdaptation | None,
        right: Hanatos2025SensitivityAdaptation | None,
    ) -> bool:
        if left is None or right is None:
            return left is right
        return (
            bool(left.apply_window) == bool(right.apply_window)
            and bool(left.apply_surface) == bool(right.apply_surface)
            and float(left.spectral_gaussian_blur) == float(right.spectral_gaussian_blur)
            and left.reference_illuminant == right.reference_illuminant
            and np.array_equal(left.window_params, right.window_params)
            and np.array_equal(left.surface_params, right.surface_params)
        )

    @timeit("spectral_compute_enlarger")
    def spectral_compute_enlarger(self,
        cmy_data,
        spectral_calculation: Callable,
        data_min,
        data_max,
        *,
        use_lut: bool = False,
    ):
        return self._spectral_compute('enlarger', cmy_data, spectral_calculation, data_min, data_max, use_lut)

    @timeit("spectral_compute_scanner")
    def spectral_compute_scanner(self,
        cmy_data,
        spectral_calculation: Callable,
        data_min,
        data_max,
        *,
        use_lut: bool = False,
    ):
        return self._spectral_compute('scanner', cmy_data, spectral_calculation, data_min, data_max, use_lut)

    def _spectral_compute(self, stage, cmy_data, spectral_calculation, data_min, data_max, use_lut):
        if not use_lut:
            return spectral_calculation(cmy_data)

        # The same transform sampled over a different domain is a different cube.
        key = (data_min, data_max, spectral_calculation(self._cmy_test_values.copy()))
        cached_key, cached_lut = self._spectral_luts.get(stage, ((), None))
        hit = cached_lut is not None and all(np.array_equal(a, b) for a, b in zip(key, cached_key))
        data_out, lut = compute_with_lut(cmy_data, spectral_calculation, xmin=data_min, xmax=data_max,
                                         steps=self._lut_resolution, lut=cached_lut if hit else None)
        if data_out is None:
            raise RuntimeError('LUT computation did not produce an output')
        if not hit:
            self._spectral_luts[stage] = (tuple(np.array(value, copy=True) for value in key), lut)
        return data_out

    @timeit("get_filming_tc_lut")
    def get_filming_tc_lut(self, sensitivity):
        sensitivity = np.asarray(sensitivity)
        if (
            self.filming_tc_lut_memory is not None
            and self._film_sensitivity is not None
            and self._same_hanatos2025_adaptation(self._cached_filming_adaptation, self.hanatos2025_adaptation)
            and self._cached_input_gamut_compress == self.input_gamut_compress
            and np.array_equal(self._film_sensitivity, sensitivity)
        ):
            return self.filming_tc_lut_memory

        self._film_sensitivity = np.array(sensitivity, copy=True)
        self._cached_filming_adaptation = self._copy_hanatos2025_adaptation(self.hanatos2025_adaptation)
        self._cached_input_gamut_compress = self.input_gamut_compress
        self.filming_tc_lut_memory = compute_hanatos2025_tc_lut(
            sensitivity,
            self.hanatos2025_adaptation,
            gamut_compress=self.input_gamut_compress,
        )
        return self.filming_tc_lut_memory

from __future__ import annotations

import copy
from time import perf_counter

import numpy as np

from spektrafilm.runtime.services import (
    EnlargerService,
    ResizingService,
    SpectralLUTService,
    ColorReferenceService,
)
from spektrafilm.runtime.stages import FilmingStage, PrintingStage, ScanningStage
from spektrafilm.runtime.topology import Node, Tap, run_topology
from spektrafilm.utils import gpu
from spektrafilm.utils.timings import format_timings



class SimulationPipeline:
    """Thin runtime orchestrator that composes stage objects around a
    tap-based topology dispatcher.

    The pipeline declares its stages as a list of :class:`Node` objects via
    :meth:`_build_topology`. Calling :meth:`process` walks the topology in
    declared order, firing each node whose input taps are present in state,
    and returns the value at the requested ``collect`` tap.
    """

    def __init__(self, params, update_params=False):
        self._params = copy.deepcopy(params)

        self.camera = self._params.camera
        self.film = self._params.film
        self.film_render = self._params.film_render
        self.enlarger = self._params.enlarger
        self.print = self._params.print
        self.print_render = self._params.print_render
        self.scanner = self._params.scanner
        self.io = self._params.io
        self.debug = self._params.debug
        self.settings = self._params.settings
        self.taps = self._params.taps

        self.timings = {}
        self._last_elapsed_time = None

        self._resize_service = ResizingService(self.io, self.camera.film_format_mm)
        if not update_params:
            self._lut_service = SpectralLUTService(self.settings.lut_resolution)
        self._enlarger_service = EnlargerService(self.enlarger)
        self._color_reference_service = ColorReferenceService(self.film, self.film_render,
                                                              self.print, self.print_render,
                                                              self.scanner.black_correction, self.scanner.white_correction,
                                                              self.scanner.black_level, self.scanner.white_level,
                                                              self.io)

        self._filming_stage = FilmingStage(
            self.film,
            self.film_render,
            self.camera,
            self.io,
            self.settings,
            self._lut_service,
            self._resize_service,
            self._enlarger_service,
            self._color_reference_service,
        )
        self._printing_stage = PrintingStage(
            self.film,
            self.film_render,
            self.print,
            self.print_render,
            self.enlarger,
            self.settings,
            self._lut_service,
            self._enlarger_service,
            self._resize_service,
            self._color_reference_service,
        )
        self._scanning_stage = ScanningStage(
            self.film,
            self.film_render,
            self.print,
            self.print_render,
            self.scanner,
            self.io,
            self.settings,
            self._lut_service,
            self._color_reference_service,
        )

        # timing communication
        self._filming_stage.timings = self.timings
        self._printing_stage.timings = self.timings
        self._scanning_stage.timings = self.timings
        self._lut_service.timings = self.timings

        self._topology: list[Node] = self._build_topology()
        self._cached_taps: dict[str, np.ndarray] = {}
        self._cached_input: np.ndarray | None = None
        self._cache_allowed_through = len(self._topology) - 2

    def process(self, image, *, inject: str | None = None, collect: str | None = None):
        """Run the pipeline.

        Defaults run end-to-end (``rgb_in`` → ``rgb_out``). Pass
        ``inject`` / ``collect`` to enter or exit at a named tap.
        Call-site kwargs override the persistent :class:`TapsParams`
        defaults.
        """
        inject = inject or self.taps.inject or Tap.RGB_IN
        collect = collect or self.taps.collect or Tap.RGB_OUT

        self.timings.clear()
        start = perf_counter()
        try:
            return run_topology(
                self._topology, inject, collect, image,
                on_fire=self._record_node_timing,
            )
        finally:
            self._last_elapsed_time = perf_counter() - start

    def process_cached(self, image, *, reuse_through: str | None = None):
        """Render end-to-end, optionally reusing an unchanged prefix of taps.

        ``reuse_through`` explicitly certifies that the image and parameters
        feeding the named tap have not changed. For a scanner-only change use
        ``Tap.CMY_PRINT`` (or ``Tap.CMY_FILM`` in scan-film mode). Omit it to
        run the full pipeline. The input is checked by value, including
        in-place edits. Direct mutations of stage parameters are *not*
        observable: callers must specify a conservative tap after such edits.
        ``soft_update`` automatically limits which cached taps are eligible.
        Cached arrays consume roughly one image per intermediate tap; use
        ``clear_cache`` to release them.
        """
        if (self.taps.inject not in (None, Tap.RGB_IN)
                or self.taps.collect not in (None, Tap.RGB_OUT)):
            raise ValueError("process_cached requires rgb_in → rgb_out topology")
        available = [node.writes[0] for node in self._topology[:-1]]
        if reuse_through is not None and reuse_through not in available:
            raise ValueError(f"invalid reusable tap: {reuse_through!r}")
        if reuse_through is not None and available.index(reuse_through) > self._cache_allowed_through:
            raise ValueError("cached tap was invalidated by soft_update")

        input_array = np.asarray(image)
        same_input = (self._cached_input is not None
                      and np.array_equal(input_array, self._cached_input, equal_nan=True))
        if not same_input:
            self.clear_cache()
        if reuse_through is not None and same_input and reuse_through in self._cached_taps:
            inject = reuse_through
            source = self._cached_taps[inject].copy()
            new_cache = {tap: value for tap, value in self._cached_taps.items()
                         if available.index(tap) <= available.index(inject)}
        else:
            inject = Tap.RGB_IN
            source = image
            new_cache = {}

        def snapshot(node, state):
            for tap in node.writes:
                if tap in available:
                    new_cache[tap] = np.array(state[tap], copy=True)

        self.timings.clear()
        start = perf_counter()
        try:
            with gpu.use(getattr(self.settings, "gpu", False)):
                result = run_topology(
                    self._topology, inject, Tap.RGB_OUT, source,
                    on_fire=self._record_node_timing, on_write=snapshot,
                )
        finally:
            self._last_elapsed_time = perf_counter() - start
        self._cached_taps = new_cache
        self._cached_input = np.array(input_array, copy=True)
        self._cache_allowed_through = len(available) - 1
        return result

    def clear_cache(self):
        """Discard cached image/taps (required after untracked parameter edits)."""
        self._cached_taps = {}
        self._cached_input = None
        self._cache_allowed_through = len(self._topology) - 2

    def get_timings(self):
        return self.timings

    def get_total_elapsed_time(self):
        return self._last_elapsed_time

    def format_timings(self):
        return format_timings(
            self.get_timings(),
            total_elapsed_time=self.get_total_elapsed_time(),
        )

    def print_timings(self):
        print(self.format_timings())

    def update_params(self, params):
        """Update params and re-initialize stages that depend on them."""
        self.__init__(params, update_params=True)

    def soft_update(self,
                    exposure_compensation_ev=None,
                    print_exposure=None,
                    c_filter_neutral=None,
                    m_filter_neutral=None,
                    y_filter_neutral=None,
                    film_density_curves=None,
                    print_density_curves=None,):
        invalidates_print_balance_reference = False
        if exposure_compensation_ev is not None:
            self.camera.exposure_compensation_ev = exposure_compensation_ev
            invalidates_print_balance_reference = True
        if print_exposure is not None:
            self.enlarger.print_exposure = print_exposure
        if c_filter_neutral is not None:
            self.enlarger.c_filter_neutral = c_filter_neutral
        if m_filter_neutral is not None:
            self.enlarger.m_filter_neutral = m_filter_neutral
        if y_filter_neutral is not None:
            self.enlarger.y_filter_neutral = y_filter_neutral
        if film_density_curves is not None:
            self.film.data.density_curves = film_density_curves
            invalidates_print_balance_reference = True
        if print_density_curves is not None:
            self.print.data.density_curves = print_density_curves
        if invalidates_print_balance_reference:
            (
                self._enlarger_service.density_spectral_midgray,
                self._enlarger_service.density_spectral_midgray_comp,
            ) = self._filming_stage._compute_density_spectral_midgray_to_balance_print()

        # Prevent an incremental render from using a prefix that a known
        # parameter edit may have changed. Film-profile edits can influence
        # filming reference corrections, so invalidate from the first stage.
        if exposure_compensation_ev is not None or film_density_curves is not None:
            self._cache_allowed_through = min(self._cache_allowed_through, 0)  # rgb_pre
        elif any(v is not None for v in (
            print_exposure, c_filter_neutral, m_filter_neutral,
            y_filter_neutral, print_density_curves,
        )):
            self._cache_allowed_through = min(self._cache_allowed_through, 2)  # cmy_film

    # private methods

    def _build_topology(self) -> list[Node]:
        f, p, s = self._filming_stage, self._printing_stage, self._scanning_stage
        common = [
            Node((Tap.RGB_IN,),     (Tap.RGB_PRE,),    self._preprocess, "preprocess"),
            Node((Tap.RGB_PRE,),    (Tap.LOG_E_FILM,), f.expose,         "filming.expose"),
            Node((Tap.LOG_E_FILM,), (Tap.CMY_FILM,),   f.develop,        "filming.develop"),
        ]
        if self.io.scan_film:
            return common + [
                Node((Tap.CMY_FILM,), (Tap.RGB_OUT,), s.scan, "scanning.scan_film"),
            ]
        return common + [
            Node((Tap.CMY_FILM,),    (Tap.LOG_E_PRINT,), p.expose,  "printing.expose"),
            Node((Tap.LOG_E_PRINT,), (Tap.CMY_PRINT,),   p.develop, "printing.develop"),
            Node((Tap.CMY_PRINT,),   (Tap.RGB_OUT,),     s.scan,    "scanning.scan_print"),
        ]

    def _preprocess(self, image):
        image = np.double(np.array(image)[:, :, 0:3])
        image = self._filming_stage.auto_exposure(image)
        image = self._resize_service.crop_and_rescale(image)
        return image

    def _record_node_timing(self, node: Node, elapsed: float) -> None:
        self.timings[node.label] = self.timings.get(node.label, 0.0) + elapsed

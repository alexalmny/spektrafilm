"""Measure the real Qt controller/worker/VisPy draw path with project defaults."""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from unittest.mock import patch

from runtime_process_timing import PORTRAIT, THREADS, command_output, configure_threads, memory_peaks


def run(args):
    configure_threads(1, 8)
    from napari.settings import get_settings
    get_settings(path=None)  # Do not read or change personal settings.
    import psutil
    from qtpy import QtCore, QtTest, QtWidgets
    from spektrafilm_gui import app, controller as controller_module
    from spektrafilm_gui.persistence import gui_state_to_dict
    from spektrafilm_gui.state import PROJECT_DEFAULT_GUI_STATE, clone_gui_state

    def project_state():
        state = clone_gui_state(PROJECT_DEFAULT_GUI_STATE)
        state.input_image.io.input_color_space = 'ProPhoto RGB'
        state.input_image.io.input_cctf_decoding = False
        state.gui_only.display.settings.preview_max_size = 640
        state.simulation.workflow.auto_preview = True
        return state

    started = perf_counter()
    def now():
        return perf_counter() - started

    requests, frames, errors = [], [], []
    by_output = {}
    def wait_for(predicate, label):
        until = now() + args.timeout
        while not predicate() and not errors and now() < until:
            QtTest.QTest.qWait(5)
        if errors or not predicate():
            detail = errors or f'timed out; recent workers={requests[-3:]}, published={frames[-3:]}'
            raise RuntimeError(f'{label}: {detail}')

    with patch.object(app, 'load_default_gui_state', project_state), \
         patch.object(QtWidgets.QMessageBox, 'critical', lambda *call: errors.append(str(call[-1]))):
        gui = app.create_app()
        controller = gui.controller
        canvas = gui.viewer.window._qt_viewer.canvas
        process = psutil.Process()
        original_execute = controller._execute_simulation_request
        original_publish = controller._set_or_add_output_layer

        def execute(request):
            info = {'mode': request.mode_label, 'shape': request.image.shape,
                    'exposure_ev': request.params.camera.exposure_compensation_ev,
                    'worker_started_at': now()}
            requests.append(info)
            result = original_execute(request)
            info['worker_finished_at'] = now()
            by_output[id(result.float_image)] = info
            return result

        def publish(image, **kwargs):
            info = by_output.pop(id(kwargs['float_image']))
            original_publish(image, **kwargs)
            frames.append(dict(info, published_at=now()))

        def draw(_event):
            if not frames or 'settled_draw_at' in frames[-1]:
                return
            layer = controller._output_layer()
            if layer is None:
                return
            frames[-1].setdefault('first_draw_at', now())
            if id(layer) not in controller._layers._output_animations:
                frames[-1]['settled_draw_at'] = now()

        def rendered_since(first, mode, *, exposure=None):
            return next((f for f in reversed(frames[first:]) if f['mode'] == mode
                         and (exposure is None or math.isclose(f['exposure_ev'], exposure, abs_tol=1e-9))
                         and 'settled_draw_at' in f), None)

        def latency(frame, since):
            return {key.removesuffix('_at') + '_seconds': frame[key] - since
                    for key in ('published_at', 'first_draw_at', 'settled_draw_at')}

        try:
            with patch.object(controller, '_execute_simulation_request', execute), \
                 patch.object(controller, '_set_or_add_output_layer', publish):
                canvas.events.draw.connect(draw, position='last')
                app.show_viewer_window(gui.viewer)
                shown_at = now()
                QtTest.QTest.qWait(args.open_after_ms)
                opened_at = now()
                (controller.load_raw_image if args.raw else controller.load_input_image)(str(args.raw or args.image))
                load_returned_at = now()
                wait_for(lambda: rendered_since(0, 'Preview'), 'open to first preview')
                open_latency = latency(rendered_since(0, 'Preview'), opened_at)
                print(f'first preview settled: {open_latency["settled_draw_seconds"]:.3f}s', flush=True)

                editor = gui.widgets.camera.exposure_compensation_ev
                anchor, step = editor.value, editor.singleStep()
                samples = []
                for i in range(args.warm):
                    first, workers_before = len(frames), len(requests)
                    edits = []
                    def edit():
                        tick = len(edits) + 1
                        target = anchor + (tick if i % 2 == 0 else 6 - tick) * step
                        edits.append(now())
                        editor.setValue(target)
                        if tick < 6:
                            QtCore.QTimer.singleShot(25, edit)
                    edit()
                    wait_for(lambda: len(edits) == 6, 'slider events')
                    exposure = editor.value
                    wait_for(lambda: rendered_since(first, 'Preview', exposure=exposure),
                             f'final slider preview ({exposure:g} EV)')
                    frame = rendered_since(first, 'Preview', exposure=exposure)
                    samples.append(dict(latency(frame, edits[-1]), exposure_ev=exposure,
                                        edit_span_seconds=edits[-1] - edits[0],
                                        workers_started=len(requests) - workers_before,
                                        outputs_published=len(frames) - first))
                    print(f'burst {i + 1}/{args.warm}: {samples[-1]["settled_draw_seconds"]:.3f}s', flush=True)

                first = len(frames)
                scan_started = now()
                controller.run_scan()
                wait_for(lambda: rendered_since(first, 'Scan'), 'manual scan')
                scan_latency = latency(rendered_since(first, 'Scan'), scan_started)
                destination = args.json.with_suffix('.tif')
                # Skip only the native file chooser; use the normal save path.
                dialog = SimpleNamespace(get_save_file_name=lambda *unused: (str(destination), ''))
                with patch.object(controller_module, '_DirMemoryDialog', return_value=dialog):
                    before_save = now()
                    controller.save_output_layer()
                    save_seconds = now() - before_save
                if errors or not destination.is_file():
                    raise RuntimeError(f'export: {errors or "output missing"}')
                pre_close_peaks = memory_peaks(process)
        finally:
            canvas.events.draw.disconnect(draw)
            controller._thread_pool.waitForDone()
            if app._background_warmup_pool is not None:
                app._background_warmup_pool.waitForDone()
            gui.viewer.close()
            for window in QtWidgets.QApplication.topLevelWidgets():
                window.close()

    times = [s['settled_draw_seconds'] for s in samples]
    median = statistics.median(times)
    return {
        'commit': command_output('git', 'rev-parse', 'HEAD'), 'platform': platform.platform(),
        'python': platform.python_version(), 'qt_platform': os.environ.get('QT_QPA_PLATFORM', 'default'),
        'threads_env': {k: os.environ[k] for k in THREADS},
        'numba_cache': 'explicit' if os.environ.get('NUMBA_CACHE_DIR') else 'package defaults',
        'settings': gui_state_to_dict(project_state()), 'source': 'native raw' if args.raw else 'linear ProPhoto TIFF',
        'startup_to_show_request_seconds': shown_at, 'show_to_open_seconds': opened_at - shown_at,
        'open_call_seconds': load_returned_at - opened_at, 'open_to_first_preview': open_latency,
        'edit_bursts': samples, 'edit_settled_median_seconds': median,
        'edit_settled_mad_seconds': statistics.median(abs(x - median) for x in times),
        'manual_scan': scan_latency, 'save_seconds': save_seconds, 'save_bytes': destination.stat().st_size,
        'worker_trace': requests, 'published_trace': frames, 'process_peaks': pre_close_peaks,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, default=PORTRAIT)
    parser.add_argument('--raw', type=Path)
    parser.add_argument('--json', type=Path, required=True)
    parser.add_argument('--warm', type=int, default=10)
    parser.add_argument('--open-after-ms', type=int, default=100)
    parser.add_argument('--timeout', type=int, default=240, help='seconds allowed per UI step')
    args = parser.parse_args()
    if min(args.warm, args.timeout) < 1 or args.open_after_ms < 0:
        parser.error('warm and timeout must be positive and open-after-ms nonnegative')
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json = args.json.resolve()
    if args.json.exists() or args.json.with_suffix('.tif').exists():
        parser.error('choose a new --json path; benchmark artifacts already exist')
    result = run(args)
    args.json.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()

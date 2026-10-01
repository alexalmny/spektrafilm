from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, cast

from spektrafilm_gui.state import (
    GuiState,
    PROJECT_DEFAULT_GUI_STATE,
    clone_gui_state,
)
from spektrafilm_gui.widgets import WidgetBundle


class SupportsSectionState(Protocol):
    def set_state(self, state: object) -> None:
        ...

    def get_state(self) -> object:
        ...


GUI_STATE_SECTION_NAMES = (
    'display',
    'input_image',
    'load_raw',
    'grain',
    'preflashing',
    'halation',
    'couplers',
    'chemistry',
    'camera',
    'enlarger_diffusion',
    'camera_diffusion',
    'glare',
    'scanner',
    'input_gamut_compress',
    'output_gamut_compress',
    'special',
    'simulation',
)


@dataclass(frozen=True)
class SectionStateAccessor:
    get: Callable[[GuiState], object]
    set: Callable[[GuiState, object], None]


# Most sections live at the top level of GuiState; 'display' and 'load_raw'
# live under `gui_only`.
_GUI_ONLY_SECTIONS = ('display', 'load_raw')


def _section_accessor(section_name: str) -> SectionStateAccessor:
    if section_name in _GUI_ONLY_SECTIONS:
        return SectionStateAccessor(
            get=lambda state, name=section_name: getattr(state.gui_only, name),
            set=lambda state, value, name=section_name: setattr(state.gui_only, name, value),
        )
    return SectionStateAccessor(
        get=lambda state, name=section_name: getattr(state, name),
        set=lambda state, value, name=section_name: setattr(state, name, value),
    )


SECTION_STATE_ACCESSORS = {name: _section_accessor(name) for name in GUI_STATE_SECTION_NAMES}


def _get_stateful_widget(widgets: WidgetBundle, section_name: str) -> SupportsSectionState:
    return cast(SupportsSectionState, getattr(widgets, section_name))


def _get_section_state(state: GuiState, section_name: str) -> object:
    return SECTION_STATE_ACCESSORS[section_name].get(state)


def _set_section_state(state: GuiState, section_name: str, value: object) -> None:
    SECTION_STATE_ACCESSORS[section_name].set(state, value)


def apply_gui_state(state: GuiState, *, widgets: WidgetBundle) -> None:
    apply_gui_state_sections(state, widgets=widgets, section_names=GUI_STATE_SECTION_NAMES)


def apply_gui_state_sections(
    state: GuiState,
    *,
    widgets: WidgetBundle,
    section_names: tuple[str, ...],
) -> None:
    for section_name in section_names:
        _get_stateful_widget(widgets, section_name).set_state(_get_section_state(state, section_name))
    if 'simulation' in section_names:
        widgets.simulation.set_auto_preview_value(state.simulation.workflow.auto_preview)
        widgets.simulation.set_scan_film_value(state.simulation.io.scan_film)
        widgets.simulation.reset_scan_for_print_value()


def collect_gui_state(
    *,
    widgets: WidgetBundle,
) -> GuiState:
    gui_state = clone_gui_state(PROJECT_DEFAULT_GUI_STATE)
    for section_name in GUI_STATE_SECTION_NAMES:
        _set_section_state(gui_state, section_name, _get_stateful_widget(widgets, section_name).get_state())
    gui_state.simulation.workflow.auto_preview = widgets.simulation.auto_preview_value()
    gui_state.simulation.io.scan_film = widgets.simulation.scan_film_value()
    return gui_state
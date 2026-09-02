"""Stubbed LaserAutofocusController used by the laser AF tests.

Lives on its own so the headless tests can build a controller without importing
control.widgets, which pulls in Qt and the whole GUI layer.
"""

from unittest.mock import MagicMock

from control.core.laser_auto_focus_controller import LaserAutofocusController
from control.models import LaserAFConfig

SENSOR_WIDTH = 3088
SENSOR_HEIGHT = 2064


def _make_controller(config: LaserAFConfig) -> LaserAutofocusController:
    """A controller with every collaborator stubbed, for exercising crop logic alone."""
    controller = LaserAutofocusController.__new__(LaserAutofocusController)
    controller._log = MagicMock()
    controller.camera = MagicMock()
    controller.camera.get_region_of_interest.return_value = (
        int(config.x_offset),
        int(config.y_offset),
        int(config.width),
        int(config.height),
    )
    # Real collaborators for the persistence path, mocked out: saving is exercised by the
    # config repository's own tests, and these cases are about crop geometry.
    controller.objectiveStore = MagicMock()
    controller.liveController = MagicMock()
    controller.laser_af_properties = config
    controller.reference_crop = None
    controller.is_initialized = True
    controller._sensor_size = (SENSOR_WIDTH, SENSOR_HEIGHT)
    controller.signal_reference_changed = MagicMock()
    return controller


def _controller_for_search(config, z_um=1000.0, piezo=None):
    controller = _make_controller(config)
    controller.microcontroller = MagicMock()
    controller.piezo = piezo
    controller.stage = MagicMock()
    controller.stage.get_pos.return_value = MagicMock(z_mm=z_um / 1000.0)
    controller._move_z = MagicMock()
    controller._restore_to_position = MagicMock()
    return controller

"""Unit tests for laser AF crop placement.

Covers utils.clamp_roi, the ROI clamping in LaserAutofocusController, and the
pixel-to-um calibration sanity guard. The controller tests build minimal stubs rather
than a full Microscope so they stay fast and hardware-free.
"""

from unittest.mock import MagicMock, patch

import pytest

import control._def
from control import utils
from control.core.laser_auto_focus_controller import LaserAutofocusController
from control.models import LaserAFConfig
from control.widgets import LaserAutofocusSettingWidget

SENSOR_WIDTH = 3088
SENSOR_HEIGHT = 2064


class TestClampRoi:
    def test_in_bounds_roi_is_only_grid_aligned(self):
        assert utils.clamp_roi(752, 694, 1536, 256, SENSOR_WIDTH, SENSOR_HEIGHT) == (752, 694, 1536, 256)

    def test_negative_offsets_clamp_to_zero(self):
        # truncate_to_interval floors, so a naive truncation of -3 gives -8, not 0.
        assert utils.clamp_roi(-3, -1, 1536, 256, SENSOR_WIDTH, SENSOR_HEIGHT) == (0, 0, 1536, 256)

    def test_offset_running_off_the_right_edge_is_pulled_back(self):
        # The failure seen on hardware: a spot at x=2819 asks for offset 2051 at width 1536,
        # and 2051 + 1536 > 3088, so the camera rejects the ROI.
        assert utils.clamp_roi(2051, 0, 1536, 256, SENSOR_WIDTH, SENSOR_HEIGHT) == (1552, 0, 1536, 256)

    def test_oversized_roi_clamps_to_sensor(self):
        assert utils.clamp_roi(0, 0, 5000, 5000, SENSOR_WIDTH, SENSOR_HEIGHT) == (0, 0, SENSOR_WIDTH, SENSOR_HEIGHT)

    def test_result_is_grid_aligned(self):
        x, y, w, h = utils.clamp_roi(101, 101, 1001, 101, SENSOR_WIDTH, SENSOR_HEIGHT)
        assert (x % 8, w % 8, y % 2, h % 2) == (0, 0, 0, 0)

    def test_sub_minimum_sizes_are_raised(self):
        _, _, w, h = utils.clamp_roi(0, 0, 0, 0, SENSOR_WIDTH, SENSOR_HEIGHT)
        assert w >= 8 and h >= 2

    @pytest.mark.parametrize(
        "offset_x, offset_y, width, height",
        [(-500, -500, 1536, 256), (9999, 9999, 1536, 256), (2051, 1950, 1536, 256), (0, 0, 5000, 5000)],
    )
    def test_result_always_fits_on_the_sensor(self, offset_x, offset_y, width, height):
        x, y, w, h = utils.clamp_roi(offset_x, offset_y, width, height, SENSOR_WIDTH, SENSOR_HEIGHT)
        assert 0 <= x and x + w <= SENSOR_WIDTH
        assert 0 <= y and y + h <= SENSOR_HEIGHT


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


class TestApplyCrop:
    def test_applies_and_returns_the_clamped_roi(self):
        controller = _make_controller(LaserAFConfig(x_offset=752, y_offset=694, width=1536, height=256))

        applied = controller.apply_crop(2051, 0, 1536, 256)

        assert applied == (1552, 0, 1536, 256)
        controller.camera.set_region_of_interest.assert_called_once_with(1552, 0, 1536, 256)

    def test_x_reference_keeps_pointing_at_the_same_sensor_pixel(self):
        # x_reference is crop-relative in memory: full-sensor 1520 inside a crop at 752.
        controller = _make_controller(
            LaserAFConfig(x_offset=752, y_offset=694, width=1536, height=256, x_reference=768.0)
        )

        controller.apply_crop(552, 694, 1536, 256)

        assert controller.laser_af_properties.x_offset == 552
        assert controller.laser_af_properties.x_reference == pytest.approx(968.0)  # 1520 - 552

    def test_x_reference_survives_repeated_shifts(self):
        controller = _make_controller(
            LaserAFConfig(x_offset=752, y_offset=694, width=1536, height=256, x_reference=768.0)
        )

        for offset in (552, 1000, 8, 1552):
            controller.apply_crop(offset, 694, 1536, 256)
            full_sensor_x = controller.laser_af_properties.x_reference + controller.laser_af_properties.x_offset
            assert full_sensor_x == pytest.approx(1520.0)

    def test_missing_x_reference_stays_missing(self):
        controller = _make_controller(LaserAFConfig(x_offset=752, width=1536, height=256, x_reference=None))

        controller.apply_crop(552, 0, 1536, 256)

        assert controller.laser_af_properties.x_reference is None

    def test_reference_image_is_dropped(self):
        # The correlation template is anchored to the crop's vertical center, so it cannot
        # survive a crop change even when x_reference is carried correctly.
        config = LaserAFConfig(x_offset=752, y_offset=694, width=1536, height=256, x_reference=768.0)
        controller = _make_controller(config)
        controller.reference_crop = object()
        controller.laser_af_properties = config.model_copy(update={"has_reference": True})

        controller.apply_crop(552, 694, 1536, 256)

        assert controller.laser_af_properties.has_reference is False
        assert controller.laser_af_properties.reference_image is None
        assert controller.reference_crop is None

    def test_calibration_is_untouched(self):
        controller = _make_controller(
            LaserAFConfig(x_offset=752, width=1536, height=256, pixel_to_um=0.564, calibration_timestamp="2026-04-28")
        )

        controller.apply_crop(552, 0, 1536, 256)

        assert controller.laser_af_properties.pixel_to_um == pytest.approx(0.564)
        assert controller.laser_af_properties.calibration_timestamp == "2026-04-28"
        assert controller.is_initialized is True


class TestCenterCropOnPoint:
    def test_centers_a_spot_sitting_at_the_crop_edge(self):
        # The hardware case: crop at offset 0, correct spot 14 px from the right edge.
        controller = _make_controller(LaserAFConfig(x_offset=0, y_offset=694, width=1536, height=256))

        applied = controller.center_crop_on_point(1521.5, 128.0, source_roi=(0, 694, 1536, 256))

        assert applied[0] == 752  # 0 + 1521.5 - 768, grid-aligned
        # The spot now sits at the middle of the crop, with room to travel either way.
        assert applied[0] <= 1521.5 - 100 and 1521.5 + 100 <= applied[0] + applied[2]

    def test_uses_the_source_roi_offset_not_the_current_one(self):
        controller = _make_controller(LaserAFConfig(x_offset=1000, y_offset=694, width=1536, height=256))

        applied = controller.center_crop_on_point(768.0, 128.0, source_roi=(0, 694, 1536, 256))

        assert applied[0] == 0  # measured against a crop at 0, so it is already centered

    def test_falls_back_to_the_current_camera_roi(self):
        controller = _make_controller(LaserAFConfig(x_offset=752, y_offset=694, width=1536, height=256))

        controller.center_crop_on_point(768.0, 128.0)

        controller.camera.get_region_of_interest.assert_called()
        assert controller.laser_af_properties.x_offset == 752


class TestCalibrationGuard:
    def _controller_measuring(self, x0, x1, calibration_distance=6.0):
        controller = _make_controller(LaserAFConfig(pixel_to_um_calibration_distance=calibration_distance))
        controller.microcontroller = MagicMock()
        controller.piezo = None
        controller._move_z = MagicMock()
        controller._get_laser_spot_centroid = MagicMock(side_effect=[(x0, 100.0), (x1, 100.0)])
        return controller

    def test_sub_pixel_displacement_fails_instead_of_dividing(self):
        # The observed failure: 0.06 px of travel over 6 um produced -98.6 um/pixel.
        controller = self._controller_measuring(772.9, 772.84)

        assert controller._calibrate_pixel_to_um() is False
        # The bogus factor must not be stored.
        assert controller.laser_af_properties.pixel_to_um == pytest.approx(1.0)

    def test_real_displacement_calibrates(self):
        controller = self._controller_measuring(700.0, 710.64)

        assert controller._calibrate_pixel_to_um() is True
        assert controller.laser_af_properties.pixel_to_um == pytest.approx(6.0 / 10.64)

    def test_negative_displacement_is_fine(self):
        # The sign encodes which way the spot travels with defocus; only tiny travel is bad.
        controller = self._controller_measuring(710.64, 700.0)

        assert controller._calibrate_pixel_to_um() is True
        assert controller.laser_af_properties.pixel_to_um == pytest.approx(-6.0 / 10.64)

    def test_implausible_factor_warns_but_still_calibrates(self):
        # Above the 1 px minimum, so it calibrates -- but 20 / 1.5 is far outside the
        # 0.4 - 2.0 um/pixel range these objectives actually produce.
        controller = self._controller_measuring(700.0, 701.5, calibration_distance=20.0)

        assert controller._calibrate_pixel_to_um() is True
        assert abs(controller.laser_af_properties.pixel_to_um) > control._def.LASER_AF_MAX_PLAUSIBLE_PIXEL_TO_UM
        assert any("implausibly large" in str(c) for c in controller._log.warning.call_args_list)


class TestInitializeWithinCurrentCrop:
    def _controller(self, config, spot=(768.0, 128.0)):
        controller = _make_controller(config)
        controller.microcontroller = MagicMock()
        controller.piezo = None
        controller._move_z = MagicMock()
        # First call is the spot search, the next two are the calibration positions.
        controller._get_laser_spot_centroid = MagicMock(side_effect=[spot, (700.0, 100.0), (710.64, 100.0)])
        return controller

    def test_search_region_is_the_crop_not_a_sensor_centered_window(self):
        controller = self._controller(LaserAFConfig(x_offset=2048, y_offset=694, width=512, height=256))

        assert controller.initialize_auto(search_within_current_crop=True) is True

        # No full-sensor ROI: the crop stays applied, so the search only sees inside it.
        assert (0, 0, SENSOR_WIDTH, SENSOR_HEIGHT) not in [
            c.args for c in controller.camera.set_region_of_interest.call_args_list
        ]
        search_call = controller._get_laser_spot_centroid.call_args_list[0]
        assert search_call.kwargs["use_center_crop"] is None

    def test_crop_is_left_exactly_where_it_was(self):
        controller = self._controller(LaserAFConfig(x_offset=2048, y_offset=694, width=512, height=256))

        controller.initialize_auto(search_within_current_crop=True)

        config = controller.laser_af_properties
        assert (config.x_offset, config.y_offset, config.width, config.height) == (2048, 694, 512, 256)

    def test_off_center_crop_survives_that_the_default_search_could_not_reach(self):
        # A 512-wide crop at 2560 frames the spot at x~2819, which the sensor-centered
        # search window cannot be positioned over at all.
        controller = self._controller(LaserAFConfig(x_offset=2560, y_offset=694, width=512, height=256))

        assert controller.initialize_auto(search_within_current_crop=True) is True
        assert controller.laser_af_properties.x_offset == 2560

    def test_calibration_still_runs(self):
        controller = self._controller(LaserAFConfig(x_offset=2048, width=512, height=256))

        controller.initialize_auto(search_within_current_crop=True)

        assert controller.laser_af_properties.pixel_to_um == pytest.approx(6.0 / 10.64)

    def test_reference_is_cleared_and_the_stale_position_dropped(self):
        config = LaserAFConfig(x_offset=2048, width=512, height=256, x_reference=100.0, has_reference=True)
        controller = self._controller(config)

        controller.initialize_auto(search_within_current_crop=True)

        assert controller.laser_af_properties.has_reference is False
        assert controller.laser_af_properties.x_reference is None
        assert controller.reference_crop is None

    def test_failed_search_reports_failure(self):
        controller = self._controller(LaserAFConfig(x_offset=2048, width=512, height=256), spot=None)

        assert controller.initialize_auto(search_within_current_crop=True) is False

    def test_default_mode_still_resets_to_the_full_sensor(self):
        controller = self._controller(LaserAFConfig(x_offset=2048, y_offset=694, width=512, height=256))
        controller._get_laser_spot_centroid = MagicMock(
            side_effect=[(1520.0, 822.0), (700.0, 100.0), (710.64, 100.0)]
        )

        assert controller.initialize_auto() is True

        first_roi = controller.camera.set_region_of_interest.call_args_list[0].args
        assert first_roi == (0, 0, SENSOR_WIDTH, SENSOR_HEIGHT)
        # And it re-places the crop around what it found, as before.
        assert controller.laser_af_properties.x_offset != 2048


class _WidgetStub:
    """LaserAutofocusSettingWidget-shaped stub, so the crop slots can be exercised without
    building a Qt widget tree. Mirrors the stub style in test_per_region_laser_af_offset.py.
    """

    def __init__(self, config, last_detection=None):
        self._log = MagicMock()
        self.laserAutofocusController = _make_controller(config)
        self._last_spot_detection = last_detection
        self.crop_status_label = MagicMock()
        self.center_crop_button = MagicMock()
        self.spinboxes = {
            name: MagicMock(value=MagicMock(return_value=getattr(config, name)))
            for name in ("width", "height", "x_offset", "y_offset")
        }
        self.update_values = MagicMock()
        self.signal_apply_settings = MagicMock()

    apply_crop = LaserAutofocusSettingWidget.apply_crop
    center_crop_on_last_detection = LaserAutofocusSettingWidget.center_crop_on_last_detection
    reset_crop_to_full_sensor = LaserAutofocusSettingWidget.reset_crop_to_full_sensor
    _apply_crop_and_refresh = LaserAutofocusSettingWidget._apply_crop_and_refresh
    _update_crop_status = LaserAutofocusSettingWidget._update_crop_status


class TestWidgetCropSlots:
    def test_apply_crop_pushes_spinbox_values_to_the_camera(self):
        widget = _WidgetStub(LaserAFConfig(x_offset=752, y_offset=694, width=1536, height=256))
        widget.spinboxes["x_offset"].value.return_value = 552

        widget.apply_crop()

        widget.laserAutofocusController.camera.set_region_of_interest.assert_called_once_with(552, 694, 1536, 256)
        widget.update_values.assert_called_once()
        widget.signal_apply_settings.emit.assert_called_once()

    def test_apply_crop_resyncs_the_spinboxes_when_the_camera_rejects_it(self):
        widget = _WidgetStub(LaserAFConfig(x_offset=752, y_offset=694, width=1536, height=256))
        widget.laserAutofocusController.camera.set_region_of_interest.side_effect = RuntimeError("bad roi")

        with patch("control.widgets.QMessageBox") as message_box:
            widget.apply_crop()

        message_box.warning.assert_called_once()
        widget.update_values.assert_called_once()  # snap back to the ROI actually in effect
        widget.signal_apply_settings.emit.assert_not_called()

    def test_center_on_last_detection_requires_a_detection(self):
        widget = _WidgetStub(LaserAFConfig(x_offset=0, width=1536, height=256), last_detection=None)

        with patch("control.widgets.QMessageBox") as message_box:
            widget.center_crop_on_last_detection()

        message_box.information.assert_called_once()
        widget.laserAutofocusController.camera.set_region_of_interest.assert_not_called()

    def test_center_on_last_detection_recenters_and_clears_the_stale_detection(self):
        widget = _WidgetStub(
            LaserAFConfig(x_offset=0, y_offset=694, width=1536, height=256),
            last_detection=(1521.5, 128.0, (0, 694, 1536, 256)),
        )

        widget.center_crop_on_last_detection()

        widget.laserAutofocusController.camera.set_region_of_interest.assert_called_once_with(752, 694, 1536, 256)
        assert widget._last_spot_detection is None
        widget.center_crop_button.setEnabled.assert_called_with(False)

    def test_reset_to_full_sensor(self):
        widget = _WidgetStub(LaserAFConfig(x_offset=752, y_offset=694, width=1536, height=256))

        widget.reset_crop_to_full_sensor()

        widget.laserAutofocusController.camera.set_region_of_interest.assert_called_once_with(
            0, 0, SENSOR_WIDTH, SENSOR_HEIGHT
        )

    def test_crop_status_reports_travel_headroom(self):
        widget = _WidgetStub(
            LaserAFConfig(x_offset=752, y_offset=694, width=1536, height=256, pixel_to_um=0.5, laser_af_range=40.0),
            last_detection=(768.0, 128.0, (752, 694, 1536, 256)),
        )

        widget._update_crop_status()

        text = widget.crop_status_label.setText.call_args[0][0]
        assert "full-sensor x=1520.0" in text
        assert "768 px left / 768 px right" in text
        assert "-384 um / +384 um" in text
        assert widget.crop_status_label.setStyleSheet.call_args[0][0] == ""

    def test_crop_status_flags_a_spot_with_less_headroom_than_the_af_range(self):
        # The hardware case: the spot 14 px from the crop edge, at 0.5 um/px, leaves 7 um of
        # travel against a 40 um search range.
        widget = _WidgetStub(
            LaserAFConfig(x_offset=0, y_offset=694, width=1536, height=256, pixel_to_um=0.5, laser_af_range=40.0),
            last_detection=(1521.5, 128.0, (0, 694, 1536, 256)),
        )

        widget._update_crop_status()

        assert "Less headroom" in widget.crop_status_label.setText.call_args[0][0]
        assert "red" in widget.crop_status_label.setStyleSheet.call_args[0][0]

"""Tests for the laser AF diagnostic sweep widget and the calibration read back from it.

Split out of test_laser_af_search.py: these drive LaserAFSweepWidget and the fit helpers in
control.widgets, so they need Qt.  The headless search/controller tests stay in that file.
"""

from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

import control._def
from control._def import SpotDetectionMode
from control.core.laser_auto_focus_controller import LaserAutofocusController, SweepSample
from control.models import LaserAFConfig

from tests.control.test_utils import create_test_image
from tests.control.laser_af_helpers import (
    SENSOR_HEIGHT,
    SENSOR_WIDTH,
    _controller_for_search,
    _make_controller,
)


class TestCurrentZMarker:
    """The sweep plot's x-axis is offset from where the sweep started, so a marker showing current
    z has to be placed against that origin -- and in the same piezo-vs-stage frame the sweep used.
    """

    def _widget(self, piezo=None, start_z_um=None, last_marked=None):
        from control.widgets import LaserAFSweepWidget

        widget = MagicMock()
        widget.laserAutofocusController.piezo = piezo
        widget._sweep_start_z_um = start_z_um
        widget._last_marked_z_um = last_marked
        widget._samples = []
        widget._set_current_z = lambda z: LaserAFSweepWidget._set_current_z(widget, z)
        return widget

    def _on_sweep_sample(self, widget, sample):
        from control.widgets import LaserAFSweepWidget

        LaserAFSweepWidget.on_sweep_sample(widget, sample)

    def test_origin_comes_from_the_first_sample(self):
        # Taken from the sample rather than read live, so it is the origin the sweep actually used
        # even if the sweep was cancelled partway.
        widget = self._widget()
        self._on_sweep_sample(widget, SweepSample(z_um=3380.0, dz_um=-20.0))

        assert widget._sweep_start_z_um == pytest.approx(3400.0)

    def test_later_samples_do_not_move_the_origin(self):
        widget = self._widget()
        for dz in (-20.0, -10.0, 0.0, 10.0):
            self._on_sweep_sample(widget, SweepSample(z_um=3400.0 + dz, dz_um=dz))

        assert widget._sweep_start_z_um == pytest.approx(3400.0)

    def test_marker_is_placed_as_an_offset_from_the_sweep_origin(self):
        widget = self._widget(start_z_um=3400.0)

        widget._set_current_z(3418.0)

        widget.current_z_line.setPos.assert_called_once_with(pytest.approx(18.0))
        widget.current_z_line.setVisible.assert_called_once_with(True)

    def test_nothing_is_drawn_before_a_sweep_exists(self):
        widget = self._widget(start_z_um=None)

        widget._set_current_z(3418.0)

        widget.current_z_line.setPos.assert_not_called()

    def test_a_stationary_stage_costs_no_qt_call(self):
        # This runs at 10 Hz whether or not anything moved.
        widget = self._widget(start_z_um=3400.0, last_marked=3418.0)

        widget._set_current_z(3418.01)

        widget.current_z_line.setPos.assert_not_called()

    def test_a_real_move_is_drawn(self):
        widget = self._widget(start_z_um=3400.0, last_marked=3418.0)

        widget._set_current_z(3419.0)

        widget.current_z_line.setPos.assert_called_once_with(pytest.approx(19.0))

    def test_stage_signal_drives_the_marker_when_there_is_no_piezo(self):
        from control.widgets import LaserAFSweepWidget

        widget = self._widget(piezo=None, start_z_um=3400.0)

        LaserAFSweepWidget.on_stage_position(widget, MagicMock(z_mm=3.418))

        widget.current_z_line.setPos.assert_called_once_with(pytest.approx(18.0))

    def test_stage_signal_is_ignored_when_a_piezo_defines_the_axis(self):
        # The sweep recorded piezo z; stage z is a different axis, off by thousands of microns.
        from control.widgets import LaserAFSweepWidget

        widget = self._widget(piezo=MagicMock(), start_z_um=150.0)

        LaserAFSweepWidget.on_stage_position(widget, MagicMock(z_mm=3.418))

        widget.current_z_line.setPos.assert_not_called()

    def test_piezo_signal_drives_the_marker_when_a_piezo_is_present(self):
        from control.widgets import LaserAFSweepWidget

        widget = self._widget(piezo=MagicMock(), start_z_um=150.0)

        LaserAFSweepWidget.on_piezo_position(widget, 168.0)

        widget.current_z_line.setPos.assert_called_once_with(pytest.approx(18.0))

    def test_piezo_signal_is_ignored_when_there_is_no_piezo(self):
        from control.widgets import LaserAFSweepWidget

        widget = self._widget(piezo=None, start_z_um=3400.0)

        LaserAFSweepWidget.on_piezo_position(widget, 168.0)

        widget.current_z_line.setPos.assert_not_called()

    def test_clear_drops_the_origin_and_hides_the_marker(self):
        from control.widgets import LaserAFSweepWidget

        widget = self._widget(start_z_um=3400.0, last_marked=3418.0)
        widget._samples = [SweepSample(z_um=3400.0, dz_um=0.0)]

        LaserAFSweepWidget.clear(widget)

        assert widget._sweep_start_z_um is None
        widget.current_z_line.setVisible.assert_called_with(False)
        # And a stale origin cannot then place a marker against an axis that no longer exists.
        widget._set_current_z(3418.0)
        widget.current_z_line.setPos.assert_not_called()


class TestRunAfSweep:
    def _controller(self, config, frames):
        controller = _controller_for_search(config)
        controller._turn_on_laser = MagicMock()
        controller._turn_off_laser = MagicMock()
        controller.get_new_frame = MagicMock(side_effect=frames)
        controller.signal_af_sweep_sample = MagicMock()
        controller.signal_af_sweep_finished = MagicMock()
        controller.camera.get_region_of_interest.return_value = (0, 0, 640, 480)
        return controller

    def _config(self, **kwargs):
        base = dict(laser_af_search_range_um=20.0, laser_af_search_step_um=10.0, width=640, height=480)
        base.update(kwargs)
        return LaserAFConfig(**base)

    def test_emits_one_sample_per_position_in_ascending_z(self):
        image = create_test_image([(200, 240), (440, 240)])
        controller = self._controller(self._config(), frames=[image] * 10)

        samples = controller.run_af_sweep()

        assert [round(s.dz_um) for s in samples] == [-20, -10, 0, 10, 20]
        assert controller.signal_af_sweep_sample.emit.call_count == len(samples)
        assert all(len(s.candidates) == 2 for s in samples)

    def test_records_every_candidate_not_just_the_selected_one(self):
        image = create_test_image([(200, 240), (440, 240)])
        controller = self._controller(
            self._config(spot_detection_mode=SpotDetectionMode.DUAL_RIGHT), frames=[image] * 10
        )

        samples = controller.run_af_sweep()

        assert [round(c["x"]) for c in samples[0].candidates] == [200, 440]
        assert samples[0].selected_x == pytest.approx(440, abs=1)

    def test_restores_z_and_kills_the_laser_on_an_exception(self):
        controller = self._controller(self._config(), frames=RuntimeError("camera died"))
        controller.get_new_frame = MagicMock(side_effect=RuntimeError("camera died"))

        with pytest.raises(RuntimeError):
            controller.run_af_sweep()

        controller._turn_off_laser.assert_called_once()
        controller._restore_to_position.assert_called_once_with(1000.0)
        controller.signal_af_sweep_finished.emit.assert_called_once()

    def test_honours_cancellation(self):
        import threading

        image = create_test_image([(200, 240)])
        controller = self._controller(self._config(), frames=[image] * 10)
        keep_running = threading.Event()  # never set -> stop before the first position

        samples = controller.run_af_sweep(keep_running=keep_running)

        assert samples == []
        controller._restore_to_position.assert_called_once_with(1000.0)

    def test_aborts_when_the_crop_changes_underneath_it(self):
        image = create_test_image([(200, 240)])
        controller = self._controller(self._config(), frames=[image] * 10)
        controller.camera.get_region_of_interest.side_effect = [(0, 0, 640, 480), (0, 0, 640, 480), (99, 0, 640, 480)]

        samples = controller.run_af_sweep()

        # Coordinates measured in two different frames must never be spliced into one result.
        assert len(samples) < 5

    def test_never_writes_configuration(self):
        """A diagnostic must be safe to run at any time, including on a tuned objective."""
        image = create_test_image([(200, 240)])
        controller = self._controller(self._config(pixel_to_um=0.5, x_reference=100.0), frames=[image] * 10)

        controller.run_af_sweep()

        controller._config_repo.save_laser_af_config.assert_not_called()
        assert controller.laser_af_properties.pixel_to_um == pytest.approx(0.5)
        assert controller.laser_af_properties.x_reference == pytest.approx(100.0)

    def test_summarize_reads_the_slope_back_as_a_calibration(self):
        """The sweep's headline output: an independent check on pixel_to_um."""
        from control.widgets import LaserAFSweepWidget

        widget = MagicMock()
        widget.laserAutofocusController.laser_af_properties = LaserAFConfig(pixel_to_um=0.5)
        # A spot translating at exactly 1/0.5 = 2 px per um.
        samples = [
            SweepSample(z_um=1000.0 + dz, dz_um=dz, candidates=[{"x": 100.0 + 2 * dz}], selected_x=100.0 + 2 * dz)
            for dz in (-20.0, -10.0, 0.0, 10.0, 20.0)
        ]

        text = LaserAFSweepWidget._summarize(widget, samples)

        assert "2.00 px/um" in text
        # Three decimals in the readout; the fourth belongs on the Apply Found Slope dialog, which
        # is where the number is actually committed.
        assert "0.500 um/px" in text
        assert "agrees" in text

    def test_summarize_calls_out_a_branch_that_does_not_move(self):
        from control.widgets import LaserAFSweepWidget

        widget = MagicMock()
        widget.laserAutofocusController.laser_af_properties = LaserAFConfig(pixel_to_um=0.5)
        samples = [
            SweepSample(z_um=1000.0 + dz, dz_um=dz, candidates=[{"x": 100.0}], selected_x=100.0)
            for dz in (-20.0, -10.0, 0.0, 10.0, 20.0)
        ]

        text = LaserAFSweepWidget._summarize(widget, samples)

        assert "does NOT move with z" in text
        assert "static reflection" in text

    def test_summarize_flags_a_slope_disagreeing_with_the_stored_calibration(self):
        from control.widgets import LaserAFSweepWidget

        widget = MagicMock()
        widget.laserAutofocusController.laser_af_properties = LaserAFConfig(pixel_to_um=0.5)
        # Translating at 12.5 px/um -> 0.08 um/px, nothing like the stored 0.5.
        samples = [
            SweepSample(z_um=1000.0 + dz, dz_um=dz, candidates=[{"x": 100.0 + 12.5 * dz}], selected_x=100.0 + 12.5 * dz)
            for dz in (-20.0, -10.0, 0.0, 10.0, 20.0)
        ]

        text = LaserAFSweepWidget._summarize(widget, samples)

        assert "off " in text and "%" in text
        assert "agrees" not in text

    def test_summarize_handles_an_empty_sweep(self):
        from control.widgets import LaserAFSweepWidget

        widget = MagicMock()
        widget.laserAutofocusController.laser_af_properties = LaserAFConfig(pixel_to_um=0.5)
        samples = [SweepSample(z_um=1000.0 + dz, dz_um=dz) for dz in (-10.0, 0.0, 10.0)]

        text = LaserAFSweepWidget._summarize(widget, samples)

        assert "outside the crop" in text

    def test_missing_frames_still_produce_a_sample(self):
        controller = self._controller(self._config(), frames=[None] * 20)

        samples = controller.run_af_sweep()

        assert len(samples) == 5
        assert all(s.candidates == [] for s in samples)


class TestSweepFit:
    """The line fitted through a sweep, which is both the summary text and a calibration."""

    def _samples(self, slope_px_per_um=2.0, noise=None, dzs=(-20.0, -10.0, 0.0, 10.0, 20.0)):
        noise = noise or [0.0] * len(dzs)
        return [
            SweepSample(
                z_um=1000.0 + dz,
                dz_um=dz,
                candidates=[{"x": 100.0 + slope_px_per_um * dz + n}],
                selected_x=100.0 + slope_px_per_um * dz + n,
            )
            for dz, n in zip(dzs, noise)
        ]

    def test_fit_measures_the_slope_span_and_residual(self):
        from control.widgets import _fit_sweep_slope

        fit = _fit_sweep_slope(self._samples(slope_px_per_um=2.0))

        assert fit.slope_px_per_um == pytest.approx(2.0)
        assert fit.um_per_px == pytest.approx(0.5)
        assert fit.n_points == 5
        assert fit.dz_span_um == pytest.approx(40.0)
        assert fit.residual_rms_px == pytest.approx(0.0, abs=1e-9)
        assert fit.is_usable_calibration
        assert not fit.residual_is_high

    def test_fit_is_none_when_there_is_no_line_to_fit(self):
        from control.widgets import _fit_sweep_slope

        assert _fit_sweep_slope([]) is None
        # Every detection at one z: a vertical scatter has no slope, however many points it holds.
        assert _fit_sweep_slope(self._samples(dzs=(0.0, 0.0, 0.0))) is None

    def test_a_bent_branch_is_flagged_rather_than_averaged_into_one_slope(self):
        """A sweep across the whole search range leaves the linear region; one slope then fits
        neither end, and adopting it as a calibration would be wrong everywhere."""
        from control.widgets import _fit_sweep_slope

        # A parabola, which no straight line describes.
        samples = [
            SweepSample(
                z_um=1000.0 + dz,
                dz_um=dz,
                candidates=[{"x": 100.0 + 0.02 * dz * dz}],
                selected_x=100.0 + 0.02 * dz * dz,
            )
            for dz in range(-100, 101, 10)
        ]

        fit = _fit_sweep_slope(samples)

        assert fit.residual_is_high

    def test_summarize_reports_how_the_fit_was_obtained(self):
        """Whether the slope is worth adopting is a question about the fit, not the slope."""
        from control.widgets import LaserAFSweepWidget

        widget = MagicMock()
        widget.laserAutofocusController.laser_af_properties = LaserAFConfig(pixel_to_um=0.5)

        text = LaserAFSweepWidget._summarize(widget, self._samples(slope_px_per_um=2.0))

        assert "5 pts / 40 um / 0.00 px RMS" in text


class TestApplySweepFitAsCalibration:
    """Reading the swept slope back into pixel_to_um -- the only path in that window that writes."""

    def _widget(self, fit, stored=1.0, acquisition_running=False):
        from control.widgets import LaserAFSweepWidget

        widget = MagicMock()
        widget._fit = fit
        widget._log = MagicMock()
        widget.laserAutofocusController = MagicMock()
        widget.laserAutofocusController.laser_af_properties = LaserAFConfig(pixel_to_um=stored)
        widget.multipointController = MagicMock()
        widget.multipointController.acquisition_in_progress.return_value = acquisition_running
        widget.laserAutofocusSettingWidget = MagicMock()
        return widget

    def _fit(self, slope_px_per_um=2.0, residual_rms_px=0.0):
        from control.widgets import _SweepFit

        return _SweepFit(slope_px_per_um, residual_rms_px, 41, 400.0)

    def _answer(self, message_box, answer):
        """Make the confirmation dialog return a given standard button."""
        message_box.return_value.exec_.return_value = answer

    def test_adopting_writes_the_measured_factor_and_refreshes_the_panel(self):
        from control.widgets import LaserAFSweepWidget

        widget = self._widget(self._fit(slope_px_per_um=1.0 / 29.7), stored=1.0)

        with patch("control.widgets.QMessageBox") as message_box:
            self._answer(message_box, message_box.Yes)
            LaserAFSweepWidget.apply_fit_as_calibration(widget)

        (measured,), kwargs = widget.laserAutofocusController.set_pixel_to_um_calibration.call_args
        assert measured == pytest.approx(29.7)
        assert "sweep" in kwargs["source"]
        widget.laserAutofocusSettingWidget.refresh_calibration_display.assert_called_once()
        # Nothing here re-initializes: the reference is a pixel position and rescaling um does not
        # move it, so throwing it away would be pure loss.
        widget.laserAutofocusController.initialize_auto.assert_not_called()

    def test_declining_the_confirmation_writes_nothing(self):
        from control.widgets import LaserAFSweepWidget

        widget = self._widget(self._fit())

        with patch("control.widgets.QMessageBox") as message_box:
            self._answer(message_box, message_box.No)
            LaserAFSweepWidget.apply_fit_as_calibration(widget)

        widget.laserAutofocusController.set_pixel_to_um_calibration.assert_not_called()

    def test_a_flat_branch_is_never_adopted(self):
        """1/slope on a static back-reflection is an enormous number, not a calibration."""
        from control.widgets import LaserAFSweepWidget

        widget = self._widget(self._fit(slope_px_per_um=0.0))

        with patch("control.widgets.QMessageBox") as message_box:
            self._answer(message_box, message_box.Yes)
            LaserAFSweepWidget.apply_fit_as_calibration(widget)

        widget.laserAutofocusController.set_pixel_to_um_calibration.assert_not_called()

    def test_refuses_to_change_the_calibration_under_a_running_acquisition(self):
        from control.widgets import LaserAFSweepWidget

        widget = self._widget(self._fit(), acquisition_running=True)

        with patch("control.widgets.QMessageBox") as message_box:
            self._answer(message_box, message_box.Yes)
            LaserAFSweepWidget.apply_fit_as_calibration(widget)

        widget.laserAutofocusController.set_pixel_to_um_calibration.assert_not_called()
        message_box.warning.assert_called_once()

    def test_a_curved_branch_is_adopted_only_with_the_curvature_spelled_out(self):
        from control.widgets import LaserAFSweepWidget

        widget = self._widget(self._fit(slope_px_per_um=0.1, residual_rms_px=20.0))

        with patch("control.widgets.QMessageBox") as message_box:
            self._answer(message_box, message_box.Yes)
            LaserAFSweepWidget.apply_fit_as_calibration(widget)

        shown = message_box.return_value.setText.call_args[0][0]
        assert "not straight" in shown
        widget.laserAutofocusController.set_pixel_to_um_calibration.assert_called_once()

    def test_apply_found_slope_is_armed_only_by_a_sweep_that_yields_a_usable_slope(self):
        """The button sits on the settings panel; the sweep is what knows whether to arm it."""
        from control.widgets import LaserAFSweepWidget

        widget = MagicMock()
        widget.laserAutofocusController.laser_af_properties = LaserAFConfig(pixel_to_um=0.5)
        widget._was_main_live = False
        widget._set_slope_available = LaserAFSweepWidget._set_slope_available.__get__(widget)
        panel = widget.laserAutofocusSettingWidget

        moving = [
            SweepSample(z_um=1000.0 + dz, dz_um=dz, candidates=[{"x": 100.0 + 2 * dz}], selected_x=100.0 + 2 * dz)
            for dz in (-10.0, 0.0, 10.0)
        ]
        LaserAFSweepWidget.on_sweep_finished(widget, moving)
        assert panel.set_sweep_slope_available.call_args[0][0] is True

        static = [
            SweepSample(z_um=1000.0 + dz, dz_um=dz, candidates=[{"x": 100.0}], selected_x=100.0)
            for dz in (-10.0, 0.0, 10.0)
        ]
        LaserAFSweepWidget.on_sweep_finished(widget, static)
        assert panel.set_sweep_slope_available.call_args[0][0] is False

    def test_a_resync_disarms_the_slope_button(self):
        """A fit belongs to the objective it was measured on. update_values is where either can
        have just changed underneath it, so a stale fit must not stay one click from being written."""
        from control.widgets import LaserAutofocusSettingWidget

        widget = MagicMock()
        widget.set_sweep_slope_available = LaserAutofocusSettingWidget.set_sweep_slope_available.__get__(widget)
        widget.spinboxes = {}
        widget.laserAutofocusController.laser_af_properties = LaserAFConfig(pixel_to_um=0.5)
        # findData feeds a `>= 0` index check, which a bare MagicMock cannot satisfy.
        widget.spot_mode_combo.findData.return_value = 0
        widget.confirm_mode_combo.findData.return_value = 0

        LaserAutofocusSettingWidget.update_values(widget)

        widget.apply_slope_button.setEnabled.assert_called_with(False)

    def test_the_panel_being_absent_is_not_an_error(self):
        """The sweep plot is usable on its own; the settings widget is an optional collaborator."""
        from control.widgets import LaserAFSweepWidget

        widget = MagicMock()
        widget.laserAutofocusSettingWidget = None

        LaserAFSweepWidget._set_slope_available(widget, True)  # must not raise
        LaserAFSweepWidget._set_sweep_running(widget, True)


class TestFalseColorLut:
    def test_offered_luts_all_resolve(self):
        import pyqtgraph as pg

        from control.core.core import ImageDisplayWindow

        assert ImageDisplayWindow.FALSE_COLOR_LUTS[0] == "Grayscale"
        for name in ImageDisplayWindow.FALSE_COLOR_LUTS[1:]:
            lut = pg.colormap.get(name).getLookupTable(nPts=256)
            assert lut.shape == (256, 3)

    def test_a_dim_pixel_is_visible_under_false_color(self):
        """The point of the feature: a spot peaking near the bottom of the range is nearly black
        in grayscale but has its own hue under a colormap."""
        import pyqtgraph as pg

        lut = pg.colormap.get("inferno").getLookupTable(nPts=256)
        dim = lut[20]
        # Grayscale would render intensity 20 as (20, 20, 20) -- indistinguishable from black.
        assert max(int(c) for c in dim) > 20 or max(abs(int(dim[0]) - int(dim[2])), 0) > 20

    def test_grayscale_entry_clears_the_lookup_table(self):
        from control.core.core import ImageDisplayWindow

        window = MagicMock()
        window.show_LUT = False
        ImageDisplayWindow.set_false_color_lut(window, "Grayscale")
        window.graphics_widget.img.setLookupTable.assert_called_once_with(None)

    def test_unknown_colormap_leaves_the_display_untouched(self):
        from control.core.core import ImageDisplayWindow

        window = MagicMock()
        window.show_LUT = False
        ImageDisplayWindow.set_false_color_lut(window, "not-a-colormap")
        window.graphics_widget.img.setLookupTable.assert_not_called()

    def test_named_colormap_is_applied(self):
        from control.core.core import ImageDisplayWindow

        window = MagicMock()
        window.show_LUT = False
        ImageDisplayWindow.set_false_color_lut(window, "inferno")
        lut = window.graphics_widget.img.setLookupTable.call_args[0][0]
        assert lut.shape == (256, 3)

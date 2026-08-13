"""Unit tests for the laser AF z-search, the diagnostic sweep, and the motion confirm guard.

The simulated focus camera renders uniform noise rather than a spot, so detection against it
always fails. Every test here that cares about detection behaviour stubs the frame source or
_get_laser_spot_centroid directly; the simulated camera is only useful for smoke-testing control
flow, not spot finding.
"""

import glob
import math
import os
from unittest.mock import MagicMock, call

import numpy as np
import pytest
import yaml
from pydantic import ValidationError

import control._def
from control import utils
from control._def import LaserAFConfirmMotionMode, SpotDetectionMode
from control.core.laser_auto_focus_controller import LaserAutofocusController, SweepSample
from control.models import LaserAFConfig

from tests.control.test_laser_af_crop import SENSOR_HEIGHT, SENSOR_WIDTH, _make_controller
from tests.control.test_utils import create_test_image

REAL_CONFIG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "user_profiles",
    "Test",
    "laser_af_configs",
)


def _legacy_search_positions(current_z_um, range_um, step_um, down_first=True):
    """The position list the hard-coded search built before it became configurable.

    Pinned here as a literal so the refactor cannot quietly change which z positions a working
    objective visits.
    """
    lower, upper = current_z_um - range_um, current_z_um + range_um
    downward, pos = [], current_z_um - step_um
    while pos >= lower:
        downward.append(pos)
        pos -= step_um
    upward, pos = [], current_z_um + step_um
    while pos <= upper:
        upward.append(pos)
        pos += step_um
    return downward + [current_z_um] + upward if down_first else upward + [current_z_um] + downward


def _controller_for_search(config, z_um=1000.0, piezo=None):
    controller = _make_controller(config)
    controller.microcontroller = MagicMock()
    controller.piezo = piezo
    controller.stage = MagicMock()
    controller.stage.get_pos.return_value = MagicMock(z_mm=z_um / 1000.0)
    controller._move_z = MagicMock()
    controller._restore_to_position = MagicMock()
    return controller


class TestBackCompat:
    """The regression gate: existing objectives must search exactly as they did before."""

    @pytest.mark.parametrize("path", sorted(glob.glob(os.path.join(REAL_CONFIG_DIR, "*.yaml"))))
    def test_real_configs_keep_their_search_span_and_default_step(self, path):
        config = LaserAFConfig(**yaml.safe_load(open(path)))
        # laser_af_range used to bound the search; taking the new field's default instead would
        # silently widen an objective that was deliberately set narrower.
        assert config.laser_af_search_range_um == config.laser_af_range
        assert config.laser_af_search_step_um == float(control._def.LASER_AF_SEARCH_STEP_UM)
        assert config.confirm_motion_mode == LaserAFConfirmMotionMode.OFF

    def test_explicit_search_range_is_not_overwritten(self):
        config = LaserAFConfig(laser_af_range=100.0, laser_af_search_range_um=25.0)
        assert config.laser_af_search_range_um == 25.0

    @pytest.mark.parametrize("range_um, step_um", [(100.0, 10.0), (40.0, 10.0)])
    def test_positions_match_the_previous_hardcoded_algorithm(self, range_um, step_um):
        controller = _controller_for_search(
            LaserAFConfig(laser_af_search_range_um=range_um, laser_af_search_step_um=step_um)
        )
        current_z, step_used, positions = controller._build_search_positions()

        assert current_z == pytest.approx(1000.0)
        assert step_used == pytest.approx(step_um)
        assert positions == pytest.approx(_legacy_search_positions(1000.0, range_um, step_um))


class TestBuildSearchPositions:
    def test_step_is_honoured(self):
        controller = _controller_for_search(
            LaserAFConfig(laser_af_search_range_um=10.0, laser_af_search_step_um=2.5)
        )
        _, _, positions = controller._build_search_positions()
        deltas = {round(b - a, 6) for a, b in zip(sorted(positions), sorted(positions)[1:])}
        assert deltas == {2.5}

    def test_explicit_arguments_override_the_config(self):
        controller = _controller_for_search(
            LaserAFConfig(laser_af_search_range_um=100.0, laser_af_search_step_um=10.0)
        )
        _, step_used, positions = controller._build_search_positions(range_um=5.0, step_um=1.0)
        assert step_used == pytest.approx(1.0)
        assert min(positions) == pytest.approx(995.0)
        assert max(positions) == pytest.approx(1005.0)

    def test_zero_step_terminates_rather_than_hanging(self):
        # LaserAFConfig forbids it, but the search runs on the GUI thread, so a value arriving by
        # any other route must not spin forever.
        controller = _controller_for_search(LaserAFConfig())
        _, step_used, positions = controller._build_search_positions(range_um=10.0, step_um=0)
        assert step_used > 0
        assert len(positions) >= 1

    def test_step_larger_than_range_degenerates_to_one_position(self):
        controller = _controller_for_search(LaserAFConfig())
        _, _, positions = controller._build_search_positions(range_um=2.0, step_um=50.0)
        assert positions == [1000.0]

    def test_piezo_bounds_are_clamped(self):
        piezo = MagicMock()
        piezo.position = 5.0
        piezo.range_um = 300.0
        controller = _controller_for_search(LaserAFConfig(), piezo=piezo)
        _, _, positions = controller._build_search_positions(range_um=100.0, step_um=10.0)
        assert min(positions) >= 0
        assert max(positions) <= 300.0

    @pytest.mark.parametrize("bad_step", [0, -1, -0.5])
    def test_config_rejects_a_non_positive_step(self, bad_step):
        with pytest.raises(ValidationError):
            LaserAFConfig(laser_af_search_step_um=bad_step)

    @pytest.mark.parametrize("bad_range", [0, -10])
    def test_config_rejects_a_non_positive_range(self, bad_range):
        with pytest.raises(ValidationError):
            LaserAFConfig(laser_af_search_range_um=bad_range)


class TestAcceptWindow:
    @pytest.mark.parametrize("step_um, expected", [(10.0, 14.0), (3.0, 4.2), (0.5, 0.7)])
    def test_window_scales_with_step(self, step_um, expected):
        # At the historical 10 um step this must reproduce the historical `step + 4`.
        window = step_um * (1.0 + control._def.LASER_AF_SEARCH_ACCEPT_TOLERANCE_FRACTION)
        assert window == pytest.approx(expected)


class TestFindAllSpotLocations:
    def test_returns_every_candidate_sorted_by_x(self):
        image = create_test_image([(200, 240), (320, 240), (440, 240)])
        candidates = utils.find_all_spot_locations(image)
        assert len(candidates) == 3
        assert [round(c["x"]) for c in candidates] == [200, 320, 440]

    def test_does_not_carry_the_full_frame_mask(self):
        # Retained across a long sweep, one boolean frame per candidate is a real memory cost.
        candidate = utils.find_all_spot_locations(create_test_image([(200, 240)]))[0]
        assert "mask" not in candidate
        assert set(candidate) == {"x", "y", "col", "row", "area", "intensity", "aspect_ratio"}

    def test_blank_frame_returns_empty_rather_than_raising(self):
        # A z position where nothing is visible is ordinary sweep data, not an error.
        assert utils.find_all_spot_locations(np.zeros((480, 640), dtype=np.uint8)) == []

    def test_agrees_with_find_spot_location_on_which_spot_a_mode_picks(self):
        image = create_test_image([(200, 240), (320, 240), (440, 240)])
        candidates = utils.find_all_spot_locations(image)
        for mode in (SpotDetectionMode.DUAL_LEFT, SpotDetectionMode.DUAL_RIGHT):
            selected = utils.select_spot_by_mode(candidates, mode)
            assert selected["x"] == pytest.approx(utils.find_spot_location(image, mode=mode)[0])

    def test_candidate_count_is_capped(self):
        image = create_test_image([(30 * i + 20, 240) for i in range(20)])
        assert len(utils.find_all_spot_locations(image, max_candidates=5)) == 5

    def test_invalid_image_still_raises(self):
        with pytest.raises(ValueError):
            utils.find_all_spot_locations(None)


class TestConfirmSpotMovesWithZ:
    def _controller(self, pixel_to_um=0.5, confirm_step_um=2.0, piezo=None, z_um=1000.0):
        controller = _controller_for_search(
            LaserAFConfig(pixel_to_um=pixel_to_um, confirm_step_um=confirm_step_um), z_um=z_um, piezo=piezo
        )
        return controller

    def test_spot_that_translates_is_accepted(self):
        controller = self._controller(pixel_to_um=0.5, confirm_step_um=2.0)
        predicted = 2.0 / 0.5  # 4 px
        controller._get_laser_spot_centroid = MagicMock(return_value=(100.0 + predicted, 50.0))

        accepted, reason = controller._confirm_spot_moves_with_z(100.0)

        assert accepted is True
        assert "confirmed" in reason

    def test_static_spot_is_rejected(self):
        controller = self._controller(pixel_to_um=0.5, confirm_step_um=20.0)  # predicts 40 px
        controller._get_laser_spot_centroid = MagicMock(return_value=(100.0, 50.0))

        accepted, reason = controller._confirm_spot_moves_with_z(100.0)

        assert accepted is False
        assert "rejected" in reason

    def test_spot_moving_the_wrong_way_is_rejected(self):
        # pixel_to_um is signed, so direction is part of the prediction.
        controller = self._controller(pixel_to_um=0.5, confirm_step_um=20.0)
        controller._get_laser_spot_centroid = MagicMock(return_value=(100.0 - 40.0, 50.0))

        accepted, _ = controller._confirm_spot_moves_with_z(100.0)

        assert accepted is False

    def test_lost_spot_is_rejected(self):
        controller = self._controller(pixel_to_um=0.5, confirm_step_um=20.0)
        controller._get_laser_spot_centroid = MagicMock(return_value=None)

        accepted, reason = controller._confirm_spot_moves_with_z(100.0)

        assert accepted is False
        assert "lost" in reason

    def test_step_predicting_sub_pixel_motion_is_skipped_not_guessed(self):
        # The real 10x: 2 um at 1.98 um/px is 1 px, below the discrimination floor.
        controller = self._controller(pixel_to_um=1.9827, confirm_step_um=2.0)
        controller._get_laser_spot_centroid = MagicMock(return_value=(100.0, 50.0))

        accepted, reason = controller._confirm_spot_moves_with_z(100.0)

        assert accepted is True
        assert "skipped" in reason
        controller._move_z.assert_not_called()

    @pytest.mark.parametrize("bad_pixel_to_um", [0.0, float("nan"), 1e6])
    def test_untrustworthy_calibration_is_skipped(self, bad_pixel_to_um):
        controller = self._controller(pixel_to_um=bad_pixel_to_um)
        controller._get_laser_spot_centroid = MagicMock(return_value=(100.0, 50.0))

        accepted, reason = controller._confirm_spot_moves_with_z(100.0)

        assert accepted is True
        assert "not trustworthy" in reason
        controller._move_z.assert_not_called()

    def test_steps_down_when_the_piezo_has_no_headroom_up(self):
        piezo = MagicMock()
        piezo.position = 299.0
        piezo.range_um = 300.0
        controller = self._controller(pixel_to_um=0.5, confirm_step_um=2.0, piezo=piezo)
        controller._get_laser_spot_centroid = MagicMock(return_value=(100.0 - 4.0, 50.0))

        accepted, _ = controller._confirm_spot_moves_with_z(100.0)

        assert controller._move_z.call_args[0][0] < 0
        assert accepted is True  # moved by the predicted amount for a downward step

    def test_no_headroom_either_way_is_skipped(self):
        piezo = MagicMock()
        piezo.position = 1.0
        piezo.range_um = 2.0
        controller = self._controller(pixel_to_um=0.5, confirm_step_um=20.0, piezo=piezo)
        controller._get_laser_spot_centroid = MagicMock(return_value=(100.0, 50.0))

        accepted, reason = controller._confirm_spot_moves_with_z(100.0)

        assert accepted is True
        assert "headroom" in reason
        controller._move_z.assert_not_called()

    @pytest.mark.parametrize("centroid", [(104.0, 50.0), (100.0, 50.0), None])
    def test_z_is_always_restored_to_the_entry_position(self, centroid):
        """The invariant that makes this safe to run once per candidate during an acquisition."""
        controller = self._controller(pixel_to_um=0.5, confirm_step_um=2.0, z_um=1234.0)
        controller._get_laser_spot_centroid = MagicMock(return_value=centroid)

        controller._confirm_spot_moves_with_z(100.0)

        controller._restore_to_position.assert_called_once_with(1234.0)

    def test_z_is_restored_even_when_detection_raises(self):
        controller = self._controller(pixel_to_um=0.5, confirm_step_um=2.0, z_um=1234.0)
        controller._get_laser_spot_centroid = MagicMock(side_effect=RuntimeError("camera died"))

        with pytest.raises(RuntimeError):
            controller._confirm_spot_moves_with_z(100.0)

        controller._restore_to_position.assert_called_once_with(1234.0)


class TestMeasureDisplacementIntegration:
    def _controller(self, config, centroids):
        controller = _controller_for_search(config)
        controller._get_laser_spot_centroid = MagicMock(side_effect=centroids)
        controller._turn_on_laser = MagicMock()
        controller._turn_off_laser = MagicMock()
        controller.signal_displacement_um = MagicMock()
        return controller

    def test_off_mode_never_takes_a_confirm_step(self):
        """The 'did we change the working objectives' test."""
        config = LaserAFConfig(
            pixel_to_um=0.5, x_reference=100.0, has_reference=True, confirm_motion_mode=LaserAFConfirmMotionMode.OFF
        )
        controller = self._controller(config, centroids=[(100.0, 50.0)])

        result = controller.measure_displacement()

        assert result == pytest.approx(0.0)
        controller._move_z.assert_not_called()
        controller._restore_to_position.assert_not_called()

    def test_always_mode_confirms_the_first_try_detection(self):
        config = LaserAFConfig(
            pixel_to_um=0.5,
            confirm_step_um=20.0,  # predicts 40 px
            x_reference=100.0,
            has_reference=True,
            confirm_motion_mode=LaserAFConfirmMotionMode.ALWAYS,
        )
        # first-try detection, then the confirm frame showing the spot translated correctly
        controller = self._controller(config, centroids=[(100.0, 50.0), (140.0, 50.0)])

        result = controller.measure_displacement()

        assert result == pytest.approx(0.0)
        assert controller._move_z.called  # the confirm step was taken

    def test_always_mode_falls_through_to_the_search_when_the_first_try_is_static(self):
        config = LaserAFConfig(
            pixel_to_um=0.5,
            confirm_step_um=20.0,
            laser_af_search_range_um=20.0,
            laser_af_search_step_um=10.0,
            x_reference=100.0,
            has_reference=True,
            confirm_motion_mode=LaserAFConfirmMotionMode.ALWAYS,
        )
        # first-try spot, a confirm frame showing no motion (rejected), then the search finds
        # nothing at any position.
        controller = self._controller(config, centroids=[(100.0, 50.0), (100.0, 50.0)] + [None] * 20)

        result = controller.measure_displacement()

        assert math.isnan(result)
        # It went looking rather than returning the static spot's displacement.
        assert controller._get_laser_spot_centroid.call_count > 2

    def test_search_gives_up_after_repeated_confirm_failures(self):
        config = LaserAFConfig(
            pixel_to_um=0.5,
            confirm_step_um=20.0,
            laser_af_search_range_um=100.0,
            laser_af_search_step_um=10.0,
            x_reference=100.0,
            has_reference=True,
            confirm_motion_mode=LaserAFConfirmMotionMode.SEARCH_ONLY,
        )
        # First-try fails, then every search position shows the same static spot.
        controller = self._controller(config, centroids=[None] + [(100.0, 50.0)] * 100)

        result = controller.measure_displacement()

        assert math.isnan(result)
        controller._turn_off_laser.assert_called()


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
        samples = [SweepSample(z_um=1000.0 + dz, dz_um=dz, candidates=[{"x": 100.0 + 2 * dz}], selected_x=100.0 + 2 * dz)
                   for dz in (-20.0, -10.0, 0.0, 10.0, 20.0)]

        text = LaserAFSweepWidget._summarize(widget, samples)

        assert "2.00 px/um" in text
        assert "0.5000 um/px" in text
        assert "agrees within" in text

    def test_summarize_calls_out_a_branch_that_does_not_move(self):
        from control.widgets import LaserAFSweepWidget

        widget = MagicMock()
        widget.laserAutofocusController.laser_af_properties = LaserAFConfig(pixel_to_um=0.5)
        samples = [SweepSample(z_um=1000.0 + dz, dz_um=dz, candidates=[{"x": 100.0}], selected_x=100.0)
                   for dz in (-20.0, -10.0, 0.0, 10.0, 20.0)]

        text = LaserAFSweepWidget._summarize(widget, samples)

        assert "does NOT move with z" in text
        assert "static reflection" in text

    def test_summarize_flags_a_slope_disagreeing_with_the_stored_calibration(self):
        from control.widgets import LaserAFSweepWidget

        widget = MagicMock()
        widget.laserAutofocusController.laser_af_properties = LaserAFConfig(pixel_to_um=0.5)
        # Translating at 12.5 px/um -> 0.08 um/px, nothing like the stored 0.5.
        samples = [SweepSample(z_um=1000.0 + dz, dz_um=dz, candidates=[{"x": 100.0 + 12.5 * dz}],
                               selected_x=100.0 + 12.5 * dz)
                   for dz in (-20.0, -10.0, 0.0, 10.0, 20.0)]

        text = LaserAFSweepWidget._summarize(widget, samples)

        assert "DISAGREES" in text

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

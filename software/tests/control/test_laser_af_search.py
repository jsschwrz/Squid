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
    def test_real_configs_load_and_back_fill_only_what_is_missing(self, path):
        """Every config on the machine loads, and only pre-split ones inherit the old span.

        Asserts against the raw file rather than fixed numbers: once an objective has been tuned
        through the GUI the two fields diverge legitimately, and a test pinned to today's values
        would fail the moment someone used the feature.
        """
        raw = yaml.safe_load(open(path))
        config = LaserAFConfig(**raw)

        if "laser_af_search_range_um" in raw:
            assert config.laser_af_search_range_um == raw["laser_af_search_range_um"]
        else:
            # Pre-split config: laser_af_range used to bound the search, so taking the field
            # default instead would silently widen an objective deliberately set narrower.
            assert config.laser_af_search_range_um == config.laser_af_range

        if "laser_af_search_step_um" not in raw:
            assert config.laser_af_search_step_um == float(control._def.LASER_AF_SEARCH_STEP_UM)
        assert config.laser_af_search_step_um > 0
        assert config.correlation_threshold <= control._def.MAX_CORRELATION_THRESHOLD

    def test_pre_split_config_inherits_its_search_span(self):
        """The upgrade path, pinned against a synthetic pre-split config rather than live state."""
        pre_split = {"pixel_to_um": 0.5, "laser_af_range": 40.0}
        config = LaserAFConfig(**pre_split)
        assert config.laser_af_search_range_um == 40.0
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


class TestCorrelationThreshold:
    def test_unsatisfiable_threshold_is_clamped_not_rejected(self):
        # The check is `correlation >= threshold` and a live frame never correlates to exactly
        # 1.0, so 1.0 rejects every measurement including perfect ones. Clamping rather than
        # raising matters because ConfigRepository swallows ValidationError and would drop the
        # whole objective's calibration.
        config = LaserAFConfig(correlation_threshold=1.0)
        assert config.correlation_threshold == control._def.MAX_CORRELATION_THRESHOLD

    @pytest.mark.parametrize("threshold", [0.7, 0.75, 0.9, 0.99])
    def test_reachable_thresholds_are_left_alone(self, threshold):
        assert LaserAFConfig(correlation_threshold=threshold).correlation_threshold == threshold

    def test_default_is_reachable_by_a_real_measurement(self):
        # Good locks observed on a 0.09 um/px objective spanned 0.749-0.991, so 0.75 is a working
        # default but only barely clears the bottom of that spread -- a noisier objective is
        # expected to need it lowered per-objective rather than relying on this.
        default = LaserAFConfig().correlation_threshold
        assert default == 0.75
        assert 0.1 <= default <= control._def.MAX_CORRELATION_THRESHOLD

    def test_max_is_below_one(self):
        assert control._def.MAX_CORRELATION_THRESHOLD < 1.0


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


class TestSearchAcceptsTheFirstDetection:
    """There is deliberately no displacement window inside the search loop.

    A step-derived window silently tightened to 2.8 um when the step was set to 2 um, discarding
    real detections at 3-25 um and making the search succeed only if it happened to land within
    one step of focus. The frame-level pixel window, the laser_af_range ceiling and the
    cross-correlation check cover what it was doing.
    """

    def _controller(self, centroids, **config_kwargs):
        config = LaserAFConfig(
            pixel_to_um=0.5,
            x_reference=100.0,
            has_reference=True,
            laser_af_search_range_um=30.0,
            laser_af_search_step_um=2.0,
            **config_kwargs,
        )
        controller = _controller_for_search(config)
        controller._get_laser_spot_centroid = MagicMock(side_effect=centroids)
        controller._turn_on_laser = MagicMock()
        controller._turn_off_laser = MagicMock()
        controller.signal_displacement_um = MagicMock()
        return controller

    @pytest.mark.parametrize("displacement_px", [10.0, 30.0, 50.0])
    def test_a_distant_detection_is_returned_rather_than_stepped_past(self, displacement_px):
        # 50 px at 0.5 um/px is 25 um -- far outside any step-derived window, and exactly the kind
        # of detection that was being discarded.
        controller = self._controller(centroids=[None, (100.0 + displacement_px, 50.0)])

        result = controller.measure_displacement()

        assert result == pytest.approx(displacement_px * 0.5)

    def test_the_search_stops_at_the_first_detection(self):
        controller = self._controller(centroids=[None, (140.0, 50.0), (100.0, 50.0)])

        controller.measure_displacement()

        # Two calls: the failed first try, then the first search position that saw anything.
        assert controller._get_laser_spot_centroid.call_count == 2

    def test_a_search_that_never_detects_still_fails(self):
        controller = self._controller(centroids=[None] * 60)

        assert math.isnan(controller.measure_displacement())
        controller._restore_to_position.assert_called()


class TestIterativeCorrection:
    """A large correction extrapolates a curve with a straight line and lands short.

    Modelled here as a spot whose apparent displacement only shrinks by a fraction of each move,
    which is the observable signature of the calibration bending away from focus.
    """

    def _controller(self, displacements, enabled=True, **config_kwargs):
        settings = dict(
            pixel_to_um=0.09,
            x_reference=100.0,
            has_reference=True,
            laser_af_range=50.0,
            iterative_correction_enabled=enabled,
        )
        settings.update(config_kwargs)
        config = LaserAFConfig(**settings)
        controller = _controller_for_search(config)
        controller.measure_displacement = MagicMock(side_effect=displacements)
        controller._verify_spot_alignment = MagicMock(return_value=(True, 0.95))
        controller.signal_cross_correlation = MagicMock()
        return controller

    def test_disabled_makes_exactly_one_move(self):
        """The default path, unchanged: measure once, move once, verify."""
        controller = self._controller([41.0], enabled=False)

        assert controller.move_to_target(0.0) is True

        controller._move_z.assert_called_once_with(-41.0)
        assert controller.measure_displacement.call_count == 1

    def test_small_correction_does_not_iterate_even_when_enabled(self):
        # Below the engage threshold the linear move is accurate, and re-measuring would cost a
        # frame grab per FOV for nothing.
        controller = self._controller([5.0], iterative_correction_min_displacement_um=10.0)

        assert controller.move_to_target(0.0) is True

        controller._move_z.assert_called_once_with(-5.0)
        assert controller.measure_displacement.call_count == 1

    def test_large_correction_converges(self):
        # 41 um measured, but each move only closes ~75% of the gap.
        controller = self._controller(
            [41.0, 10.0, 2.0, 0.5],
            iterative_correction_min_displacement_um=10.0,
            iterative_correction_tolerance_um=1.0,
        )

        assert controller.move_to_target(0.0) is True

        assert [c.args[0] for c in controller._move_z.call_args_list] == [-41.0, -10.0, -2.0]
        assert controller.measure_displacement.call_count == 4

    def test_stops_as_soon_as_it_is_within_tolerance(self):
        controller = self._controller(
            [41.0, 0.4], iterative_correction_min_displacement_um=10.0, iterative_correction_tolerance_um=1.0
        )

        controller.move_to_target(0.0)

        controller._move_z.assert_called_once_with(-41.0)  # no second move needed

    def test_honours_a_nonzero_target(self):
        controller = self._controller(
            [41.0, 8.0, 5.2], iterative_correction_min_displacement_um=10.0, iterative_correction_tolerance_um=1.0
        )

        controller.move_to_target(5.0)

        # Each move closes the gap to the target, not to zero.
        assert [round(c.args[0], 2) for c in controller._move_z.call_args_list] == [-36.0, -3.0]

    def test_gives_up_after_the_pass_limit_and_lets_the_check_judge(self):
        # Never converges; must not loop forever, and must still reach the alignment check.
        controller = self._controller(
            [41.0] * 10, iterative_correction_min_displacement_um=10.0, iterative_correction_tolerance_um=1.0
        )

        controller.move_to_target(0.0)

        assert controller._move_z.call_count == 1 + 3  # first move plus the pass limit
        controller._verify_spot_alignment.assert_called_once()

    def test_a_lost_spot_mid_iteration_stops_without_restoring(self):
        # Leaving z where the previous move put it is deliberate: the alignment check decides, and
        # it restores on failure. Bailing out to the original z would discard a nearly-good move.
        controller = self._controller([41.0, float("nan")], iterative_correction_min_displacement_um=10.0)

        assert controller.move_to_target(0.0) is True

        controller._move_z.assert_called_once_with(-41.0)
        controller._restore_to_position.assert_not_called()

    def test_does_not_run_a_spot_search_while_iterating(self):
        # After the first move we are near focus; sweeping z again would be slow and could wander.
        controller = self._controller(
            [41.0, 0.2], iterative_correction_min_displacement_um=10.0, iterative_correction_tolerance_um=1.0
        )

        controller.move_to_target(0.0)

        assert controller.measure_displacement.call_args_list[-1].kwargs == {"search_for_spot": False}

    def test_an_implausible_re_measurement_stops_the_loop(self):
        controller = self._controller([41.0, 500.0], iterative_correction_min_displacement_um=10.0)

        controller.move_to_target(0.0)

        controller._move_z.assert_called_once_with(-41.0)


class TestDebrisWarning:
    @pytest.mark.parametrize(
        "pixel_to_um, offset_px, expect_warning",
        [
            # 40x: a 20 px offset is 1.8 um, well within a normal lock -- must not warn.
            (0.0911, 20.0, False),
            (0.0911, 150.0, True),  # 13.7 um -- genuinely off
            # 10x: 20 px is 40 um, badly off -- the old fixed pixel threshold barely caught it.
            (1.9827, 20.0, True),
            (1.9827, 3.0, False),  # 5.9 um
        ],
    )
    def test_threshold_means_the_same_distance_on_every_objective(self, pixel_to_um, offset_px, expect_warning):
        offset_um = offset_px * abs(pixel_to_um)
        assert (offset_um > control._def.LASER_AF_DEBRIS_WARNING_OFFSET_UM) is expect_warning


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

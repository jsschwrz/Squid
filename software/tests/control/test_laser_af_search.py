"""Unit tests for the laser AF z-search, the diagnostic sweep and the calibration read back from
it, and the motion confirm guard.

The simulated focus camera renders uniform noise rather than a spot, so detection against it
always fails. Every test here that cares about detection behaviour stubs the frame source or
_get_laser_spot_centroid directly; the simulated camera is only useful for smoke-testing control
flow, not spot finding.
"""

import glob
import math
import os
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest
import yaml
from pydantic import ValidationError

import control._def
from control import utils
from control._def import LaserAFConfirmMotionMode, SpotDetectionMode
from control.core.laser_auto_focus_controller import LaserAutofocusController, SweepSample
from control.models import LaserAFConfig

from tests.control.laser_af_helpers import (
    SENSOR_HEIGHT,
    SENSOR_WIDTH,
    _controller_for_search,
    _make_controller,
)
from tests.control.test_utils import create_test_image

# Two sources, on purpose. The committed fixtures give the regression gate below something to
# check everywhere, including CI, where user_profiles/ is gitignored and holds no laser AF
# configs at all -- globbing only the machine made the gate collect zero cases and pass
# vacuously (it originally pointed at a "Test" profile that has never existed on disk). The
# machine profiles are then added on top, so a developer's own tuned objectives are covered too.
FIXTURE_CONFIG_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "laser_af_configs")
MACHINE_CONFIG_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "user_profiles",
)
REAL_CONFIG_PATHS = sorted(glob.glob(os.path.join(FIXTURE_CONFIG_ROOT, "*.yaml"))) + sorted(
    glob.glob(os.path.join(MACHINE_CONFIG_ROOT, "*", "laser_af_configs", "*.yaml"))
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


class TestBackCompat:
    """The regression gate: existing objectives must search exactly as they did before."""

    def test_there_are_real_configs_to_check(self):
        """The gate below is parametrized over a glob; an empty glob would pass silently."""
        assert REAL_CONFIG_PATHS, f"no laser AF configs found under {FIXTURE_CONFIG_ROOT} or {MACHINE_CONFIG_ROOT}"

    @pytest.mark.parametrize("path", REAL_CONFIG_PATHS)
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


class TestBuildSearchPositions:
    def test_step_is_honoured(self):
        controller = _controller_for_search(LaserAFConfig(laser_af_search_range_um=10.0, laser_af_search_step_um=2.5))
        _, _, positions = controller._build_search_positions()
        deltas = {round(b - a, 6) for a, b in zip(sorted(positions), sorted(positions)[1:])}
        assert deltas == {2.5}

    def test_explicit_arguments_override_the_config(self):
        controller = _controller_for_search(LaserAFConfig(laser_af_search_range_um=100.0, laser_af_search_step_um=10.0))
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


class TestXReferenceFrameConversion:
    """x_reference is crop-relative in memory and full-sensor on disk.

    Saving the model directly writes the wrong frame, and the next load subtracts x_offset a
    second time -- moving the reference off the crop entirely, where nothing can ever match it.
    """

    @staticmethod
    def _in_memory(x_reference_full_sensor=1543.0, **kwargs):
        """A config in the in-memory frame, built the way initialize_manual builds it.

        model_copy rather than direct construction, because construction runs the repair validator
        -- which is correct to do for a value read off disk, and wrong for one already converted.
        Production only ever reaches the in-memory frame through model_copy, so this matches it.
        """
        settings = dict(x_offset=1000.0, width=1000, x_reference=x_reference_full_sensor)
        settings.update(kwargs)
        on_disk = LaserAFConfig(**settings)
        return on_disk.model_copy(update={"x_reference": on_disk.x_reference - on_disk.x_offset})

    def _controller(self, config):
        controller = _make_controller(config)
        controller.objectiveStore.current_objective = "40x"
        return controller

    def test_save_converts_back_to_the_full_sensor_frame(self):
        controller = self._controller(self._in_memory(1543.0))
        assert controller.laser_af_properties.x_reference == pytest.approx(543.0)  # crop-relative

        controller._save_current_config()

        saved = controller._config_repo.save_laser_af_config.call_args[0][2]
        assert saved.x_reference == pytest.approx(1543.0)

    def test_a_round_trip_through_save_and_load_is_stable(self):
        """The property that was broken: tuning repeatedly must not walk the reference away."""
        controller = self._controller(self._in_memory(1543.0))

        for _ in range(5):
            controller._save_current_config()
            saved = controller._config_repo.save_laser_af_config.call_args[0][2]
            # initialize_manual's disk -> memory conversion
            controller.laser_af_properties = saved.model_copy(
                update={"x_reference": saved.x_reference - saved.x_offset}
            )

        assert controller.laser_af_properties.x_reference == pytest.approx(543.0)

    def test_update_threshold_properties_no_longer_corrupts_the_reference(self):
        # This is the "Apply without Re-initialization" path, pressed repeatedly while tuning.
        controller = self._controller(self._in_memory(1543.0))

        controller.update_threshold_properties({"correlation_threshold": 0.8})

        saved = controller._config_repo.save_laser_af_config.call_args[0][2]
        assert saved.x_reference == pytest.approx(1543.0)
        assert saved.correlation_threshold == 0.8

    def test_missing_reference_stays_missing(self):
        controller = self._controller(LaserAFConfig(x_offset=1000.0, width=1000, x_reference=None))

        controller._save_current_config()

        assert controller._config_repo.save_laser_af_config.call_args[0][2].x_reference is None

    def test_load_repairs_a_reference_stored_in_the_crop_relative_frame(self):
        # The value actually found on the machine: crop 1000..2000, reference stored as 543.66.
        config = LaserAFConfig(x_offset=1000.0, width=1000, x_reference=543.6563556340321)
        assert config.x_reference == pytest.approx(1543.6563556340321)

    def test_load_leaves_a_correct_reference_alone(self):
        config = LaserAFConfig(x_offset=1000.0, width=1000, x_reference=1543.0)
        assert config.x_reference == pytest.approx(1543.0)

    def test_load_does_not_guess_where_the_frames_overlap(self):
        # crop 100..1636, so 500 is a plausible full-sensor reference. Rewriting it would break a
        # config that was fine.
        config = LaserAFConfig(x_offset=100.0, width=1536, x_reference=500.0)
        assert config.x_reference == pytest.approx(500.0)

    def test_load_reports_but_does_not_rewrite_an_unrepairable_reference(self):
        # Below the crop, but adding x_offset overshoots it too -- no safe correction exists.
        config = LaserAFConfig(x_offset=1000.0, width=100, x_reference=500.0)
        assert config.x_reference == pytest.approx(500.0)


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
        assert set(candidate) == {
            "x",
            "y",
            "col",
            "row",
            "area",
            "intensity",
            "peak_intensity",
            "aspect_ratio",
        }

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


class TestSetPixelToUmCalibration:
    """The controller side of adopting an externally measured factor."""

    def _controller(self, **config_kwargs):
        controller = _controller_for_search(LaserAFConfig(**config_kwargs))
        controller._save_current_config = MagicMock()
        return controller

    def test_writes_the_factor_stamps_it_and_saves(self):
        controller = self._controller(pixel_to_um=1.0, calibration_timestamp="")

        controller.set_pixel_to_um_calibration(29.7, source="AF sweep fit")

        assert controller.laser_af_properties.pixel_to_um == pytest.approx(29.7)
        assert controller.laser_af_properties.calibration_timestamp != ""
        controller._save_current_config.assert_called_once()

    def test_leaves_the_reference_and_the_initialized_state_alone(self):
        controller = self._controller(pixel_to_um=1.0, x_reference=768.0, has_reference=True)
        controller.is_initialized = True

        controller.set_pixel_to_um_calibration(0.4, source="AF sweep fit")

        assert controller.laser_af_properties.x_reference == pytest.approx(768.0)
        assert controller.laser_af_properties.has_reference
        assert controller.is_initialized

    @pytest.mark.parametrize("bad", [0.0, float("inf"), float("nan")])
    def test_refuses_a_factor_that_is_not_a_number_of_microns(self, bad):
        controller = self._controller(pixel_to_um=1.0)

        with pytest.raises(ValueError):
            controller.set_pixel_to_um_calibration(bad, source="test")

        assert controller.laser_af_properties.pixel_to_um == pytest.approx(1.0)
        controller._save_current_config.assert_not_called()


class TestCalibrationDistanceFitsThePiezo:
    """PiezoStage.move_to raises rather than clipping, so an over-range calibration distance would
    otherwise fail halfway through the sequence with the AF laser still on."""

    def _controller(self, distance_um, piezo=None):
        controller = _controller_for_search(LaserAFConfig(pixel_to_um_calibration_distance=distance_um))
        controller.piezo = piezo
        return controller

    def test_a_stage_machine_is_never_blocked(self):
        controller = self._controller(600.0, piezo=None)

        assert controller._calibration_distance_fits()

    def test_a_move_that_fits_the_piezo_travel_is_allowed(self):
        controller = self._controller(200.0, piezo=MagicMock(position=150.0, range_um=300.0))

        assert controller._calibration_distance_fits()

    def test_a_move_off_the_end_of_the_piezo_is_refused_before_the_laser_is_lit(self):
        controller = self._controller(600.0, piezo=MagicMock(position=150.0, range_um=300.0))
        controller._move_z = MagicMock()
        controller._turn_on_laser = MagicMock()

        assert controller._calibrate_pixel_to_um() is False
        controller._move_z.assert_not_called()

    def test_an_off_center_piezo_is_refused_even_for_a_short_move(self):
        """Half the move goes down: the travel that matters is the travel on both sides."""
        controller = self._controller(100.0, piezo=MagicMock(position=10.0, range_um=300.0))

        assert not controller._calibration_distance_fits()


class _StrictImageSignal:
    """Stands in for image_to_display, which is typed numpy.ndarray and rejects None.

    A MagicMock would swallow the exact call that took down a whole acquisition FOV, so this
    reproduces PyQt's own type check instead.
    """

    def __init__(self):
        self.emitted = []

    def emit(self, image):
        if not isinstance(image, np.ndarray):
            raise TypeError(
                "LaserAutofocusController.image_to_display[numpy.ndarray].emit(): "
                f"argument 1 has unexpected type '{type(image).__name__}'"
            )
        self.emitted.append(image)


def _controller_for_frames(frames, **config_overrides):
    """A controller whose focus camera hands back exactly `frames`, one per read."""
    config = LaserAFConfig(
        spot_detection_mode=SpotDetectionMode.SINGLE,
        laser_af_averaging_n=len(frames),
        **config_overrides,
    )
    controller = _make_controller(config)
    controller.microcontroller = MagicMock()
    controller.camera.read_frame.side_effect = list(frames)
    controller.image_to_display = _StrictImageSignal()
    controller.signal_displacement_um = MagicMock()
    return controller


class TestDroppedFrameDuringAveraging:
    """A frame the camera fails to deliver must cost one pass of averaging, nothing more.

    From the 2026-08-21 acquisition: the focus camera missed frame 3 of 3, `image` was left None,
    and the display emit below the loop raised TypeError -- discarding two good detections that
    were already averaged, failing AF for that FOV, and leaving the laser on because the raise
    skipped the turn-off.
    """

    def test_a_missed_last_frame_still_returns_the_good_detections(self):
        spot = create_test_image([(320, 240)])
        controller = _controller_for_frames([spot, spot, None])

        result = controller._get_laser_spot_centroid()

        assert result is not None, "two good frames must still produce a centroid"
        assert result[0] == pytest.approx(320, abs=2)
        assert result[1] == pytest.approx(240, abs=2)

    def test_a_missed_last_frame_displays_nothing_rather_than_raising(self):
        spot = create_test_image([(320, 240)])
        controller = _controller_for_frames([spot, spot, None])

        controller._get_laser_spot_centroid()

        assert controller.image_to_display.emitted == []

    def test_a_frame_that_arrives_is_still_displayed(self):
        """The guard must not cost the normal case its preview."""
        spot = create_test_image([(320, 240)])
        controller = _controller_for_frames([spot, spot, spot])

        controller._get_laser_spot_centroid()

        assert len(controller.image_to_display.emitted) == 1

    def test_every_frame_missed_reports_failure_without_raising(self):
        controller = _controller_for_frames([None, None, None])

        assert controller._get_laser_spot_centroid() is None
        assert controller.image_to_display.emitted == []


class TestLaserIsAlwaysTurnedOff:
    """The laser must not outlive the measurement, on any path.

    It used to be turned off once per return path and not at all on a raise, so an unexpected
    exception left it lit through the FOV's own exposure.
    """

    def _controller(self):
        controller = _make_controller(LaserAFConfig(x_reference=100.0, has_reference=True))
        controller.microcontroller = MagicMock()
        controller.signal_displacement_um = MagicMock()
        return controller

    def test_turned_off_when_the_measurement_raises(self):
        controller = self._controller()
        controller._get_laser_spot_centroid = MagicMock(side_effect=RuntimeError("dropped frame"))

        with pytest.raises(RuntimeError):
            controller.measure_displacement()

        controller.microcontroller.turn_on_AF_laser.assert_called_once()
        controller.microcontroller.turn_off_AF_laser.assert_called_once()

    def test_turned_off_on_a_normal_measurement(self):
        controller = self._controller()
        controller._get_laser_spot_centroid = MagicMock(return_value=(120.0, 50.0))

        controller.measure_displacement()

        controller.microcontroller.turn_off_AF_laser.assert_called_once()

    def test_turned_off_when_no_spot_is_found_and_no_search_is_allowed(self):
        controller = self._controller()
        controller._get_laser_spot_centroid = MagicMock(return_value=None)

        assert math.isnan(controller.measure_displacement(search_for_spot=False))
        controller.microcontroller.turn_off_AF_laser.assert_called_once()

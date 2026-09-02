"""Tests for the live laser AF spot diagnostics.

The overlay says a frame failed. These say *why*, in terms of the cc_* settings that decide it,
and what value would let the spot back. Layered the same way test_laser_af_spot_overlay is: the
detector decides what a rejection means (no Qt), the controller decides when to ask, and the
settings widget decides how to say it.

The load-bearing property throughout is the round trip -- a suggestion that does not actually
admit the blob is worse than no suggestion, because the operator clicks Relax, presses Apply, and
sees nothing change.
"""

import time
from unittest.mock import MagicMock

import numpy as np
import pytest

from control import utils
from control.core.laser_auto_focus_controller import LaserAutofocusController, SpotOverlayResult
from control.models import LaserAFConfig
from control.utils import SpotDetectionMode
from control.widgets import LaserAutofocusSettingWidget
from tests.control.test_utils import create_test_image


def _relaxed(params: dict, suggested: dict) -> dict:
    """Apply a suggestion to a detector parameter dict, as the Relax button plus Apply would."""
    out = dict(params)
    for name, value in suggested.items():
        out[utils._CC_FIELD_TO_PARAM[name]] = value
    return out


def _named(criteria, name):
    return next(c for c in criteria if c.name == name)


class TestRejectionIsMeasured:
    """One test per filter: the diagnosis names the setting and the number that turned it away."""

    def test_a_blob_below_the_area_floor_names_cc_min_area(self):
        diagnosis = utils.diagnose_spot_detection(
            create_test_image([(320, 240)], spot_size=6), params={"min_area": 400}
        )
        best = diagnosis.best
        assert [c.name for c in best.failures] == ["cc_min_area"]
        assert _named(best.criteria, "cc_min_area").measured == best.area
        assert _named(best.criteria, "cc_min_area").limit == 400

    def test_a_blob_above_the_area_ceiling_names_cc_max_area(self):
        diagnosis = utils.diagnose_spot_detection(create_test_image([(320, 240)]), params={"max_area": 20})
        assert [c.name for c in diagnosis.best.failures] == ["cc_max_area"]

    def test_a_blob_off_the_centre_row_names_cc_row_tolerance(self):
        diagnosis = utils.diagnose_spot_detection(create_test_image([(320, 60)]), params={"row_tolerance": 20})
        best = diagnosis.best
        assert [c.name for c in best.failures] == ["cc_row_tolerance"]
        # expected_row is the crop centre, so a spot at y=60 in a 480-tall frame is 180 px off.
        assert _named(best.criteria, "cc_row_tolerance").measured == pytest.approx(180, abs=2)

    def test_an_elongated_blob_names_cc_max_aspect_ratio(self):
        image = np.zeros((480, 640), dtype=np.uint8)
        image[236:244, 200:440] = 200  # 240 x 8, aspect ratio 30
        diagnosis = utils.diagnose_spot_detection(image)
        assert [c.name for c in diagnosis.best.failures] == ["cc_max_aspect_ratio"]

    def test_a_dim_blob_names_cc_threshold(self):
        # create_test_image normalises to a peak of 255, so scaling is what makes a dim spot.
        dim = (create_test_image([(320, 240)]) * 0.06).astype(np.uint8)
        diagnosis = utils.diagnose_spot_detection(dim, params={"threshold": 30})
        assert [c.name for c in diagnosis.best.failures] == ["cc_threshold"]
        assert diagnosis.best.peak_intensity == pytest.approx(dim.max(), abs=1)

    def test_every_failure_is_recorded_not_just_the_first(self):
        """The detector stops at the first failing filter. A diagnosis that did the same would
        send the operator round the loop once per filter."""
        diagnosis = utils.diagnose_spot_detection(
            create_test_image([(320, 60)], spot_size=6), params={"min_area": 400, "row_tolerance": 20}
        )
        assert {c.name for c in diagnosis.best.failures} == {"cc_min_area", "cc_row_tolerance"}

    def test_a_passing_frame_offers_nothing_to_fix(self):
        diagnosis = utils.diagnose_spot_detection(create_test_image([(320, 240)]))
        assert diagnosis.rejects == [] and diagnosis.note is None


class TestSuggestionsActuallyAdmit:
    """The round trip. A suggestion is only worth making if feeding it back finds the spot."""

    @pytest.mark.parametrize(
        "params",
        [
            {"min_area": 400},
            {"max_area": 20},
            {"row_tolerance": 20},
            {"min_area": 400, "row_tolerance": 20},
        ],
    )
    def test_feeding_the_suggestion_back_finds_the_spot(self, params):
        image = create_test_image([(320, 60)], spot_size=6)
        assert utils.find_all_spot_locations(image, params=params) == []

        suggested = utils.diagnose_spot_detection(image, params=params).suggested_params()
        assert suggested
        assert utils.find_all_spot_locations(image, params=_relaxed(params, suggested))

    def test_a_dim_spot_round_trips_through_the_threshold_suggestion(self):
        dim = (create_test_image([(320, 240)]) * 0.06).astype(np.uint8)
        params = {"threshold": 30}
        assert utils.find_all_spot_locations(dim, params=params) == []

        suggested = utils.diagnose_spot_detection(dim, params=params).suggested_params()
        assert "cc_threshold" in suggested
        assert utils.find_all_spot_locations(dim, params=_relaxed(params, suggested))

    def test_rounding_goes_the_way_that_admits(self):
        """One decimal on the aspect ratio spinbox: 2.63 suggested as 2.6 still rejects."""
        assert utils._round_to_admit("cc_max_aspect_ratio", 2.63, 1) == pytest.approx(2.7)
        assert utils._round_to_admit("cc_min_area", 45.0, -1) == 45.0  # already on the grid
        assert utils._round_to_admit("cc_min_area", 1.0, -1) == 1.0  # and at the very bottom of it
        assert utils._round_to_admit("cc_max_aspect_ratio", 2.5, 1) == pytest.approx(2.5)

    def test_a_suggestion_the_spinbox_cannot_hold_is_refused_not_clamped(self):
        """A clamped value looks like a fix and changes nothing. Say where the real answer is."""
        image = np.zeros((480, 640), dtype=np.uint8)
        image[236:244, 200:440] = 200  # 240 x 8, aspect ratio 30, past the ceiling of 10
        criterion = _named(utils.diagnose_spot_detection(image).best.criteria, "cc_max_aspect_ratio")
        assert not criterion.passed
        assert criterion.suggested is None
        assert "streak" in criterion.alternative

    @pytest.mark.parametrize("height", [256, 480, 2064])
    def test_row_tolerance_can_always_reach_a_blob_inside_the_crop(self, height):
        """Row deviation cannot exceed half the crop height, so a ceiling that tracks the crop can
        always admit a blob that is actually in frame. A fixed ceiling could not: on a full sensor
        it refused a spot plainly visible near the top of the image."""
        image = np.zeros((height, 640), dtype=np.uint8)
        image[10:30, 300:320] = 200  # near the top edge, so the deviation is close to its maximum
        criterion = _named(
            utils.diagnose_spot_detection(image, params={"row_tolerance": 5}).best.criteria, "cc_row_tolerance"
        )
        assert not criterion.passed
        assert criterion.suggested is not None
        assert criterion.suggested >= criterion.measured

    @pytest.mark.parametrize(
        "height, expected",
        [
            (None, 200.0),  # nothing known about the frame: fall back to the static floor
            (256, 200.0),  # half of 256 is under the floor, and the floor never drops
            (480, 240.0),
            (2064, 1032.0),
            (0, 200.0),  # degenerate, not a division
        ],
    )
    def test_the_row_tolerance_ceiling_tracks_the_crop(self, height, expected):
        assert utils._row_tolerance_ceiling(height) == expected

    def test_narrowing_the_crop_never_lowers_the_ceiling(self):
        """Otherwise loading an objective saved against a wide crop would silently clamp its
        tolerance down, and Apply would then write the clamped value back."""
        assert utils._row_tolerance_ceiling(64) >= utils._CC_SPINBOX_LIMITS["cc_row_tolerance"][1]

    def test_a_merged_blob_is_told_to_raise_the_threshold_not_the_ceiling(self):
        """Raising cc_max_area to admit a spot fused with its halo admits the halo."""
        image = np.zeros((256, 1536), dtype=np.uint8)
        image[100:160, 680:760] = 60  # halo
        image[120:135, 710:725] = 220  # the spot inside it
        criterion = _named(utils.diagnose_spot_detection(image, params={"max_area": 900}).best.criteria, "cc_max_area")
        assert not criterion.passed
        assert "CC Threshold" in criterion.alternative


class TestFramesWithNoBlobAnswer:
    """Cases where no cc_* setting is the fix -- which is the most useful thing to be told."""

    def test_a_uniform_frame_reports_no_signal(self):
        note = utils.diagnose_spot_detection(np.zeros((480, 640), dtype=np.uint8)).note
        assert "uniform" in note and "laser" in note

    def test_a_spot_outside_the_crop_sends_you_to_the_crop_not_a_spinbox(self):
        """The failure that most misleads today: indistinguishable from a slightly tight setting."""
        # Sensor noise and nothing else: enough contrast to be worth looking at, but no pixel
        # stands out from the noise floor, which is what an empty crop actually looks like.
        frame = np.random.default_rng(3).integers(14, 28, (256, 1536), dtype=np.uint8)
        diagnosis = utils.diagnose_spot_detection(frame, params={"threshold": 200})
        assert diagnosis.note is not None
        assert "not in this crop" in diagnosis.note
        assert diagnosis.suggested_params() == {}

    def test_a_frame_of_pure_noise_says_so_rather_than_listing_blobs(self):
        rng = np.random.default_rng(0)
        frame = rng.integers(0, 2, (480, 640), dtype=np.uint8) * 60
        diagnosis = utils.diagnose_spot_detection(frame, params={"threshold": 30})
        assert diagnosis.note is None or "noise" in diagnosis.note


class TestConsentAndCost:
    def test_it_counts_what_else_the_suggestion_would_admit(self):
        """Relaxing until something appears is how a back-reflection gets locked onto."""
        image = create_test_image([(200, 240), (320, 240), (440, 240)], spot_size=6)
        diagnosis = utils.diagnose_spot_detection(image, params={"min_area": 400})
        assert diagnosis.also_admits == 2  # the other two blobs the same relaxation lets in

    def test_a_surgical_suggestion_admits_nothing_else(self):
        diagnosis = utils.diagnose_spot_detection(
            create_test_image([(320, 240)], spot_size=6), params={"min_area": 400}
        )
        assert diagnosis.also_admits == 0

    def test_the_reject_list_is_capped(self):
        image = create_test_image([(30 * i + 20, 240) for i in range(20)], spot_size=6)
        diagnosis = utils.diagnose_spot_detection(image, params={"min_area": 400})
        assert 0 < len(diagnosis.rejects) <= utils.LASER_AF_DIAG_MAX_REJECTS

    def test_a_noisy_full_sensor_frame_does_not_stall_the_gui_thread(self):
        """The regression that matters most: per-component work here is a full-frame mask, and
        Reset to Full Sensor is exactly when someone is diagnosing."""
        rng = np.random.default_rng(1)
        frame = rng.integers(0, 60, (2064, 3088), dtype=np.uint8)

        started = time.perf_counter()
        utils.analyze_frame(frame, params={"threshold": 8}, filter_sigma=1, diagnose=True)
        diagnosed = time.perf_counter() - started

        started = time.perf_counter()
        utils.analyze_frame(frame, params={"threshold": 8}, filter_sigma=1, diagnose=False)
        detected = time.perf_counter() - started

        # Generous, because CI timing is noisy -- this catches an order of magnitude, which is what
        # a full-frame mask per component would cost, not a percentage.
        assert diagnosed < max(detected * 3.0, 2.0)

    def test_the_same_frame_twice_gives_the_same_answer(self):
        image = create_test_image([(200, 240), (440, 240)], spot_size=6)
        params = {"min_area": 400}
        assert utils.diagnose_spot_detection(image, params=params) == utils.diagnose_spot_detection(
            image, params=params
        )


class TestDetectionIsUnchanged:
    def test_candidates_still_come_back_the_same(self):
        image = create_test_image([(200, 240), (320, 240), (440, 240)])
        assert [round(c["x"]) for c in utils.find_all_spot_locations(image)] == [200, 320, 440]

    def test_analyze_frame_agrees_with_find_all_spot_locations(self):
        image = create_test_image([(200, 240), (440, 240)])
        candidates, diagnosis = utils.analyze_frame(image)
        assert [c["x"] for c in candidates] == [c["x"] for c in utils.find_all_spot_locations(image)]
        assert diagnosis is None


def _make_controller(**config_overrides) -> LaserAutofocusController:
    controller = LaserAutofocusController.__new__(LaserAutofocusController)
    controller._log = MagicMock()
    controller.laser_af_properties = LaserAFConfig(**config_overrides)
    return controller


class TestClassifyFrameSpotsDiagnosis:
    def test_a_passing_frame_reports_margins_and_no_diagnosis(self):
        controller = _make_controller(spot_detection_mode=SpotDetectionMode.SINGLE)
        result = controller.classify_frame_spots(create_test_image([(320, 240)]), diagnose=True)
        assert result.failure_reason is None
        assert result.diagnosis is None
        assert [c.name for c in result.criteria] == [
            "cc_threshold",
            "cc_min_area",
            "cc_max_area",
            "cc_row_tolerance",
            "cc_max_aspect_ratio",
        ]
        assert all(c.passed for c in result.criteria)

    def test_margins_are_reported_even_without_asking_for_a_diagnosis(self):
        """Headroom while it still works is the half that makes a dropout visible coming."""
        controller = _make_controller(spot_detection_mode=SpotDetectionMode.SINGLE)
        result = controller.classify_frame_spots(create_test_image([(320, 240)]))
        assert result.criteria and result.diagnosis is None

    def test_diagnose_is_off_by_default(self, monkeypatch):
        calls = []
        monkeypatch.setattr(utils, "diagnose_frame", lambda *a, **k: calls.append(1))
        controller = _make_controller(cc_min_area=400)
        controller.classify_frame_spots(create_test_image([(320, 240)], spot_size=6))
        assert calls == []

    def test_a_failed_frame_explains_itself_when_asked(self):
        controller = _make_controller(cc_min_area=400, spot_detection_mode=SpotDetectionMode.SINGLE)
        result = controller.classify_frame_spots(create_test_image([(320, 240)], spot_size=6), diagnose=True)
        assert result.failure_reason == "no spot detected"
        assert result.diagnosis is not None
        assert result.diagnosis.suggested_params() == {"cc_min_area": pytest.approx(result.diagnosis.best.area)}

    def test_the_reference_is_used_to_rank_which_blob_is_worth_explaining(self):
        controller = _make_controller(
            cc_min_area=400,
            spot_detection_mode=SpotDetectionMode.SINGLE,
            has_reference=True,
            x_reference=440.0,
            pixel_to_um=0.5,
        )
        image = create_test_image([(200, 240), (440, 240)], spot_size=6)
        result = controller.classify_frame_spots(image, diagnose=True)
        assert result.diagnosis.best.x == pytest.approx(440, abs=3)

    def test_an_unchoosable_frame_gets_no_blob_advice(self):
        """Every candidate passed every filter; the mode is what cannot choose, and no margin
        explains a mode."""
        controller = _make_controller(spot_detection_mode=SpotDetectionMode.SINGLE)
        result = controller.classify_frame_spots(create_test_image([(200, 240), (440, 240)]), diagnose=True)
        assert "cannot choose" in result.failure_reason
        assert result.diagnosis is None


class _SettingsWidgetStub:
    """LaserAutofocusSettingWidget-shaped stub for the live readout slots.

    Same approach as _WidgetStub in test_laser_af_crop: bind the real methods onto an object
    carrying only what they touch, so the readouts can be exercised without building a Qt widget
    tree. Class attributes have to be copied across by hand, so a removed one shows up here as an
    AttributeError rather than a silent pass.
    """

    _MARGIN_REFRESH_INTERVAL_S = LaserAutofocusSettingWidget._MARGIN_REFRESH_INTERVAL_S

    on_live_detection_result = LaserAutofocusSettingWidget.on_live_detection_result
    _clear_live_readouts = LaserAutofocusSettingWidget._clear_live_readouts
    # Rewrapped: reading it off the class yields the plain function, which would then bind
    # `self` as its first argument.
    _criterion_tooltip = staticmethod(LaserAutofocusSettingWidget._criterion_tooltip)
    _apply_criteria_to_readouts = LaserAutofocusSettingWidget._apply_criteria_to_readouts
    _update_live_readouts = LaserAutofocusSettingWidget._update_live_readouts
    _update_row_tolerance_range = LaserAutofocusSettingWidget._update_row_tolerance_range

    def __init__(self, initialized=True, crop_height=256):
        self._last_detection_result = None
        self._next_margin_refresh_s = 0.0
        self.measured_labels = {name: MagicMock() for name in utils._CC_SPINBOX_LIMITS}
        self.candidate_count_label = MagicMock()
        self.update_threshold_button = MagicMock()
        self.update_threshold_button.isEnabled.return_value = initialized
        self.spinboxes = {name: MagicMock() for name in utils._CC_SPINBOX_LIMITS}
        self.laserAutofocusController = MagicMock()
        self.laserAutofocusController.laser_af_properties = LaserAFConfig(height=crop_height)

    def shown(self, name) -> str:
        return self.measured_labels[name].setText.call_args[0][0]

    def is_red(self, name) -> bool:
        return "#C00000" in self.measured_labels[name].setStyleSheet.call_args[0][0]

    def tooltip(self, name) -> str:
        return self.measured_labels[name].setToolTip.call_args[0][0]


def _result_for(image, diagnose=True, **config_overrides):
    return _make_controller(**config_overrides).classify_frame_spots(image, diagnose=diagnose)


ALL_CC = tuple(utils._CC_SPINBOX_LIMITS)


class TestMeasuredReadouts:
    """The measured value sits beside the control that judges it, in that control's own units."""

    def test_a_working_frame_fills_every_readout_and_reddens_none(self):
        widget = _SettingsWidgetStub()
        widget.on_live_detection_result(_result_for(create_test_image([(320, 240)])))

        assert all(widget.shown(name) for name in ALL_CC)
        assert not any(widget.is_red(name) for name in ALL_CC)
        assert widget.candidate_count_label.setText.call_args[0][0] == "1"

    def test_only_the_failing_control_goes_red(self):
        widget = _SettingsWidgetStub()
        widget.on_live_detection_result(_result_for(create_test_image([(320, 240)], spot_size=6), cc_min_area=400))

        assert widget.is_red("cc_min_area")
        assert not any(widget.is_red(name) for name in ALL_CC if name != "cc_min_area")

    def test_an_off_row_blob_reddens_cc_row_tolerance_and_names_it(self):
        """The reported confusion, pinned: the measurement is called "row offset" but the control
        that governs it is CC Row Tolerance, and only the control name is actionable."""
        widget = _SettingsWidgetStub()
        widget.on_live_detection_result(_result_for(create_test_image([(320, 60)]), cc_row_tolerance=20))

        assert widget.is_red("cc_row_tolerance")
        assert widget.shown("cc_row_tolerance") == "180"
        assert "CC Row Tolerance" in widget.tooltip("cc_row_tolerance")

    def test_readouts_use_the_same_decimals_as_their_spinbox(self):
        widget = _SettingsWidgetStub()
        widget.on_live_detection_result(_result_for(create_test_image([(320, 240)])))

        assert "." not in widget.shown("cc_min_area")  # pixels are whole numbers
        assert "." in widget.shown("cc_max_aspect_ratio")  # the spinbox shows one decimal

    def test_both_area_controls_show_the_blob_area(self):
        widget = _SettingsWidgetStub()
        widget.on_live_detection_result(_result_for(create_test_image([(320, 240)])))
        assert widget.shown("cc_min_area") == widget.shown("cc_max_area")

    def test_a_passing_tooltip_states_the_limit(self):
        widget = _SettingsWidgetStub()
        widget.on_live_detection_result(_result_for(create_test_image([(320, 240)])))
        assert "CC Threshold" in widget.tooltip("cc_threshold")

    def test_an_unreachable_suggestion_puts_the_alternative_in_the_tooltip(self):
        """The one piece of the old advice paragraph with nowhere else to go."""
        image = np.zeros((480, 640), dtype=np.uint8)
        image[236:244, 200:440] = 200  # a streak, aspect ratio 30 against a ceiling of 10
        widget = _SettingsWidgetStub()
        widget.on_live_detection_result(_result_for(image))

        assert "streak" in widget.tooltip("cc_max_aspect_ratio")
        assert "cc_max_aspect_ratio" not in widget._last_detection_result.diagnosis.suggested_params()

    @pytest.mark.parametrize("height, expected_max", [(256, 200.0), (2064, 1032.0)])
    def test_the_row_tolerance_spinbox_is_widened_to_suit_the_crop(self, height, expected_max):
        widget = _SettingsWidgetStub(crop_height=height)
        widget._update_row_tolerance_range()

        assert widget.spinboxes["cc_row_tolerance"].setMaximum.call_args[0][0] == expected_max
        assert f"{int(height)} px crop" in widget.spinboxes["cc_row_tolerance"].setToolTip.call_args[0][0]

    def test_a_saved_tolerance_wider_than_the_crop_is_not_clamped_away(self):
        """Set at a full sensor, then the crop narrows: the value must survive into the spinbox,
        or the next Apply writes the clamped number back over it."""
        widget = _SettingsWidgetStub(crop_height=256)
        widget.laserAutofocusController.laser_af_properties = LaserAFConfig(height=256, cc_row_tolerance=900)
        widget._update_row_tolerance_range()

        assert widget.spinboxes["cc_row_tolerance"].setMaximum.call_args[0][0] >= 900

    def test_redraws_are_paced_not_run_per_frame(self):
        widget = _SettingsWidgetStub()
        result = _result_for(create_test_image([(320, 240)]))
        for _ in range(10):
            widget.on_live_detection_result(result)
        assert widget.measured_labels["cc_threshold"].setText.call_count == 1

    def test_turning_detection_off_clears_the_readouts_immediately(self):
        widget = _SettingsWidgetStub()
        widget.on_live_detection_result(_result_for(create_test_image([(320, 240)])))
        widget.on_live_detection_result(None)

        assert all(widget.shown(name) == "" for name in ALL_CC)
        assert widget._last_detection_result is None

    def test_a_frame_level_note_blanks_the_readouts_rather_than_showing_stale_numbers(self):
        """No number beside a control describes anything on a frame with no spot in it at all.
        The reason goes to the live detection status line instead."""
        widget = _SettingsWidgetStub()
        widget.on_live_detection_result(_result_for(np.zeros((256, 1536), dtype=np.uint8)))

        assert all(widget.shown(name) == "" for name in ALL_CC)


class TestTheSuggestionInTheTooltip:
    """The Relax button is gone; what it knew now lives on the readout it applies to.

    A rejected blob still needs to say which value would admit it -- and, crucially, what that
    value would cost, because relaxing a filter until something appears is exactly how a static
    back-reflection gets locked onto.
    """

    def test_a_rejected_blob_names_the_value_that_would_admit_it(self):
        widget = _SettingsWidgetStub()
        widget.on_live_detection_result(_result_for(create_test_image([(320, 240)], spot_size=6), cc_min_area=400))

        assert "Set CC Min Area to" in widget.tooltip("cc_min_area")

    def test_every_failing_control_carries_its_own_suggestion(self):
        """The detector stops at the first filter a blob fails, so one at a time is a loop."""
        widget = _SettingsWidgetStub()
        widget.on_live_detection_result(
            _result_for(create_test_image([(320, 60)], spot_size=6), cc_min_area=400, cc_row_tolerance=20)
        )

        assert "Set CC Min Area to" in widget.tooltip("cc_min_area")
        assert "Set CC Row Tolerance to" in widget.tooltip("cc_row_tolerance")

    def test_a_surgical_relaxation_says_nothing_extra(self):
        widget = _SettingsWidgetStub()
        widget.on_live_detection_result(_result_for(create_test_image([(320, 240)], spot_size=6), cc_min_area=400))

        assert "would also admit" not in widget.tooltip("cc_min_area")

    @pytest.mark.parametrize(
        "spots, expected",
        [
            ([(200, 240), (440, 240)], "would also admit 1 other blob"),
            ([(140, 240), (280, 240), (420, 240), (540, 240)], "would also admit 3 other blobs"),
        ],
    )
    def test_it_says_how_many_others_the_suggestion_would_admit(self, spots, expected):
        """The guard against relaxing your way onto a back-reflection, kept where the number is."""
        widget = _SettingsWidgetStub()
        widget.on_live_detection_result(_result_for(create_test_image(spots, spot_size=6), cc_min_area=400))

        assert expected in widget.tooltip("cc_min_area")

    def test_a_passing_frame_suggests_nothing(self):
        widget = _SettingsWidgetStub()
        widget.on_live_detection_result(_result_for(create_test_image([(320, 240)])))

        assert "Set CC Min Area to" not in widget.tooltip("cc_min_area")


class TestFrameNotesReachTheStatusLine:
    def test_a_note_is_preferred_over_the_bare_failure_reason(self):
        """ "no spot detected" leaves you guessing; "the spot is not in this crop" tells you no
        detection setting is the fix."""
        from control.widgets import LaserAFSpotOverlay

        # filter_sigma off, so the frame reaches the detector as the sensor noise it is. With the
        # default Gaussian the noise smooths into something blob-shaped and the frame gets a
        # per-blob answer instead -- a real behaviour, but not the one under test here.
        frame = np.random.default_rng(3).integers(14, 28, (256, 1536), dtype=np.uint8)
        controller = _make_controller(cc_threshold=200, filter_sigma=None)
        overlay = LaserAFSpotOverlay(controller, MagicMock(), rate_hz=1000.0)

        statuses = []
        overlay.signal_status.connect(statuses.append)
        overlay.set_enabled(True)
        overlay.on_frame(frame)
        overlay._next_allowed_s = 0.0
        overlay.on_frame(frame)

        assert "not in this crop" in statuses[-1]

    def test_a_frame_with_a_blob_answer_keeps_the_plain_reason(self):
        from control.widgets import LaserAFSpotOverlay

        controller = _make_controller(cc_min_area=400)
        overlay = LaserAFSpotOverlay(controller, MagicMock(), rate_hz=1000.0)

        statuses = []
        overlay.signal_status.connect(statuses.append)
        overlay.set_enabled(True)
        overlay.on_frame(create_test_image([(320, 240)], spot_size=6))

        assert statuses[-1] == "no spot detected"


class TestOverlayDrawsRejects:
    def test_rejected_blobs_and_a_label_reach_the_display(self):
        from control.widgets import LaserAFSpotOverlay

        controller = _make_controller(cc_min_area=400, spot_detection_mode=SpotDetectionMode.SINGLE)
        display = MagicMock()
        overlay = LaserAFSpotOverlay(controller, display, rate_hz=1000.0)
        overlay.set_enabled(True)

        image = create_test_image([(320, 240)], spot_size=6)
        overlay.on_frame(image)  # first failure: nothing to diagnose from yet
        # The duty cycle spaces the next run by the cost of the last one, so step past it rather
        # than sleeping. The pacing itself is covered in test_laser_af_spot_overlay.
        overlay._next_allowed_s = 0.0
        overlay.on_frame(image)  # now the previous frame is known to have failed

        kwargs = display.set_spot_overlay.call_args.kwargs
        assert kwargs["failed"] is True
        assert len(kwargs["rejects"]) == 1
        assert kwargs["reject_label"] is not None and "area" in kwargs["reject_label"][2]

    def test_a_working_frame_draws_no_rejects(self):
        from control.widgets import LaserAFSpotOverlay

        controller = _make_controller(spot_detection_mode=SpotDetectionMode.SINGLE)
        display = MagicMock()
        overlay = LaserAFSpotOverlay(controller, display, rate_hz=1000.0)
        overlay.set_enabled(True)
        overlay.on_frame(create_test_image([(320, 240)]))

        kwargs = display.set_spot_overlay.call_args.kwargs
        assert kwargs["rejects"] == [] and kwargs["reject_label"] is None

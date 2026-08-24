"""Tests for the live laser AF spot overlay.

Three layers, tested separately because they are separable: the controller decides what a frame
means (classify_frame_spots, no Qt at all), LaserAFSpotOverlay decides how often to ask, and
ImageDisplayWindow draws the answer.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest
from qtpy.QtCore import Qt

from control.core.core import ImageDisplayWindow
from control.core.laser_auto_focus_controller import LaserAutofocusController, SpotOverlayResult
from control.models import LaserAFConfig
from control.utils import SpotDetectionMode
from control.widgets import LaserAFSpotOverlay
from tests.control.test_utils import create_test_image


def _make_controller(**config_overrides) -> LaserAutofocusController:
    """A controller with only what classify_frame_spots touches, as test_laser_af_crop does."""
    controller = LaserAutofocusController.__new__(LaserAutofocusController)
    controller._log = MagicMock()
    controller.laser_af_properties = LaserAFConfig(**config_overrides)
    return controller


class TestClassifyFrameSpots:
    def test_empty_frame_reports_no_spot(self):
        result = _make_controller().classify_frame_spots(np.zeros((480, 640), dtype=np.uint8))
        assert result.candidates == []
        assert result.selected_x is None
        assert result.failure_reason == "no spot detected"

    def test_unusable_frame_reports_no_frame(self):
        # find_all_spot_locations raises on a zero-size array; the stream can hand us one
        # between a crop change and the first frame in the new geometry.
        result = _make_controller().classify_frame_spots(np.zeros((0, 0), dtype=np.uint8))
        assert result.failure_reason == "no frame"

    def test_single_spot_is_selected_and_does_not_fail(self):
        controller = _make_controller(spot_detection_mode=SpotDetectionMode.SINGLE)
        result = controller.classify_frame_spots(create_test_image([(320, 240)]))
        assert len(result.candidates) == 1
        assert result.selected_x == pytest.approx(320, abs=2)
        assert result.selected_y == pytest.approx(240, abs=2)
        assert result.failure_reason is None

    def test_single_mode_with_two_spots_keeps_candidates_but_cannot_choose(self):
        controller = _make_controller(spot_detection_mode=SpotDetectionMode.SINGLE)
        result = controller.classify_frame_spots(create_test_image([(200, 240), (440, 240)]))
        # The candidates still matter -- that the mode cannot choose between them is the finding.
        assert len(result.candidates) == 2
        assert result.selected_x is None
        assert "cannot choose" in result.failure_reason

    @pytest.mark.parametrize(
        "mode, expected_x",
        [(SpotDetectionMode.DUAL_LEFT, 200), (SpotDetectionMode.DUAL_RIGHT, 440)],
    )
    def test_mode_picks_its_side(self, mode, expected_x):
        controller = _make_controller(spot_detection_mode=mode)
        result = controller.classify_frame_spots(create_test_image([(200, 240), (440, 240)]))
        assert result.selected_x == pytest.approx(expected_x, abs=2)
        assert result.failure_reason is None

    def test_no_reference_means_no_displacement(self):
        controller = _make_controller(spot_detection_mode=SpotDetectionMode.SINGLE, has_reference=False)
        result = controller.classify_frame_spots(create_test_image([(320, 240)]))
        assert (result.reference_x, result.displacement_um) == (None, None)

    def test_a_reference_turns_the_selection_into_a_signed_displacement(self):
        controller = _make_controller(
            spot_detection_mode=SpotDetectionMode.SINGLE,
            has_reference=True,
            x_reference=300.0,
            pixel_to_um=0.5,
        )
        result = controller.classify_frame_spots(create_test_image([(320, 240)]))
        assert result.reference_x == 300.0
        assert result.displacement_um == pytest.approx((result.selected_x - 300.0) * 0.5)
        assert result.failure_reason is None

    def test_a_spot_far_from_the_reference_is_reported_not_rejected(self):
        """Distance from the reference is no longer a verdict this layer makes.

        The crop bounds where a spot may be found, and how large a displacement is worth acting
        on is move_to_target's call against laser_af_range. A frame the operator can see is a
        frame worth reporting honestly.
        """
        controller = _make_controller(
            spot_detection_mode=SpotDetectionMode.SINGLE,
            has_reference=True,
            x_reference=100.0,
            pixel_to_um=0.5,
        )
        result = controller.classify_frame_spots(create_test_image([(600, 240)]))
        assert result.selected_x == pytest.approx(600, abs=2)
        assert result.displacement_um == pytest.approx((result.selected_x - 100.0) * 0.5)
        assert result.failure_reason is None


class _FakeClock:
    """A hand-cranked stand-in for perf_counter, so throttling is tested without sleeping."""

    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds


@pytest.fixture
def overlay_parts(qtbot):
    """An overlay wired to a mock controller and a mock display, on a hand-cranked clock."""
    controller = MagicMock()
    # A real SpotOverlayResult, not a MagicMock: a mock auto-creates whatever the overlay reads,
    # so a field added to the result later would be silently satisfied here and the test would go
    # on passing while the overlay mishandled it.
    controller.classify_frame_spots.return_value = SpotOverlayResult(
        candidates=[{"x": 10.0, "y": 20.0}],
        selected_x=10.0,
        selected_y=20.0,
    )
    display = MagicMock()
    overlay = LaserAFSpotOverlay(controller, display, rate_hz=5.0)
    clock = _FakeClock()
    overlay._now = clock
    return overlay, controller, display, clock


class TestSpotOverlayThrottle:
    def test_disabled_overlay_never_detects(self, overlay_parts):
        overlay, controller, _, _ = overlay_parts
        for _ in range(10):
            overlay.on_frame(np.zeros((16, 16), dtype=np.uint8))
        controller.classify_frame_spots.assert_not_called()

    def test_none_frame_is_ignored(self, overlay_parts):
        overlay, controller, _, _ = overlay_parts
        overlay.set_enabled(True)
        overlay.on_frame(None)
        controller.classify_frame_spots.assert_not_called()

    def test_frames_inside_the_interval_are_dropped(self, overlay_parts):
        overlay, controller, _, clock = overlay_parts
        overlay.set_enabled(True)
        for _ in range(10):
            overlay.on_frame(np.zeros((16, 16), dtype=np.uint8))
            clock.advance(0.01)  # 100 fps of frames against a 5 Hz overlay
        assert controller.classify_frame_spots.call_count == 1

    def test_detection_resumes_once_the_interval_passes(self, overlay_parts):
        overlay, controller, _, clock = overlay_parts
        overlay.set_enabled(True)
        overlay.on_frame(np.zeros((16, 16), dtype=np.uint8))
        clock.advance(0.21)  # just past 1/5 s
        overlay.on_frame(np.zeros((16, 16), dtype=np.uint8))
        assert controller.classify_frame_spots.call_count == 2

    def test_an_expensive_detection_spaces_itself_out_beyond_the_rate(self, overlay_parts):
        overlay, controller, _, clock = overlay_parts
        overlay.set_enabled(True)
        # A full-sensor crop: detection itself burns 1 s, so the 5 Hz rate is not the binding
        # constraint -- the duty cycle is, and it must hold the next run off for 1/0.25 = 4 s.
        controller.classify_frame_spots.side_effect = lambda image, diagnose=False: clock.advance(
            1.0
        ) or SpotOverlayResult()
        overlay.on_frame(np.zeros((16, 16), dtype=np.uint8))

        clock.advance(3.0)
        overlay.on_frame(np.zeros((16, 16), dtype=np.uint8))
        assert controller.classify_frame_spots.call_count == 1, "should still be backed off"

        clock.advance(1.1)
        overlay.on_frame(np.zeros((16, 16), dtype=np.uint8))
        assert controller.classify_frame_spots.call_count == 2

    def test_a_raising_detector_backs_off_instead_of_spinning(self, overlay_parts):
        overlay, controller, _, clock = overlay_parts
        overlay.set_enabled(True)
        controller.classify_frame_spots.side_effect = RuntimeError("camera went away")

        statuses = []
        overlay.signal_status.connect(statuses.append)

        for _ in range(5):
            overlay.on_frame(np.zeros((16, 16), dtype=np.uint8))
            clock.advance(0.25)  # past the 5 Hz interval, but well inside the error backoff
        assert controller.classify_frame_spots.call_count == 1
        assert statuses and "failed" in statuses[-1]

    def test_disabling_clears_the_overlay(self, overlay_parts):
        overlay, _, display, _ = overlay_parts
        overlay.set_enabled(True)
        overlay.set_enabled(False)
        display.clear_spot_overlay.assert_called_once()

    def test_result_is_forwarded_to_the_display(self, overlay_parts):
        overlay, controller, display, _ = overlay_parts
        controller.classify_frame_spots.return_value = SpotOverlayResult(
            candidates=[{"x": 10.0, "y": 20.0}, {"x": 30.0, "y": 21.0}],
            selected_x=30.0,
            selected_y=21.0,
            reference_x=25.0,
        )
        overlay.set_enabled(True)
        overlay.on_frame(np.zeros((16, 16), dtype=np.uint8))

        kwargs = display.set_spot_overlay.call_args.kwargs
        assert kwargs["candidates"] == [(10.0, 20.0), (30.0, 21.0)]
        assert kwargs["selected"] == (30.0, 21.0)
        assert (kwargs["reference_x"], kwargs["failed"]) == (25.0, False)

    def test_a_failing_frame_is_flagged_and_explained_once(self, overlay_parts):
        overlay, controller, display, clock = overlay_parts
        controller.classify_frame_spots.return_value = SpotOverlayResult(failure_reason="no spot detected")
        statuses = []
        overlay.signal_status.connect(statuses.append)
        overlay.set_enabled(True)

        for _ in range(3):
            overlay.on_frame(np.zeros((16, 16), dtype=np.uint8))
            clock.advance(0.25)

        assert display.set_spot_overlay.call_args.kwargs["failed"] is True
        # Emitted on change only: three failing frames with the same reason is one status update.
        assert statuses == ["no spot detected"]


@pytest.fixture
def focus_display(qtbot):
    win = ImageDisplayWindow()
    qtbot.addWidget(win)
    win.display_image(np.zeros((256, 512), dtype=np.uint8))
    return win


class TestImageDisplayWindowOverlay:
    def test_overlay_items_are_not_built_until_used(self, focus_display):
        assert focus_display.spot_candidates_item is None
        focus_display.clear_spot_overlay()  # safe before anything has been drawn
        assert focus_display.spot_candidates_item is None

    def test_candidates_and_selection_land_in_their_items(self, focus_display):
        focus_display.set_spot_overlay(candidates=[(10, 20), (30, 21), (50, 19)], selected=(30, 21))
        assert len(focus_display.spot_candidates_item.getData()[0]) == 3
        assert len(focus_display.spot_selected_item.getData()[0]) == 1
        assert focus_display.spot_candidates_item.isVisible()
        assert focus_display.spot_selected_item.isVisible()

    def test_the_reference_line_marks_the_focus_plane(self, focus_display):
        focus_display.set_spot_overlay(selected=(300, 20), reference_x=250.0)
        assert focus_display.spot_reference_line.value() == pytest.approx(250.0)
        assert focus_display.spot_reference_line.isVisible()

    def test_no_reference_hides_the_line(self, focus_display):
        focus_display.set_spot_overlay(candidates=[(10, 20)], selected=(10, 20))
        assert not focus_display.spot_reference_line.isVisible()

    def test_failure_recolors_the_selection(self, focus_display):
        focus_display.set_spot_overlay(selected=(10, 20), failed=False)
        passing = focus_display.spot_selected_item.opts["pen"].color().getRgb()
        focus_display.set_spot_overlay(selected=(10, 20), failed=True)
        failing = focus_display.spot_selected_item.opts["pen"].color().getRgb()
        assert passing != failing
        assert failing[:3] == ImageDisplayWindow._SPOT_FAILED_BRUSH[:3]

    def test_empty_result_hides_the_markers(self, focus_display):
        focus_display.set_spot_overlay(candidates=[(10, 20)], selected=(10, 20))
        focus_display.set_spot_overlay(candidates=[], selected=None)
        assert not focus_display.spot_candidates_item.isVisible()
        assert not focus_display.spot_selected_item.isVisible()

    def test_clear_hides_everything(self, focus_display):
        focus_display.set_spot_overlay(candidates=[(10, 20)], selected=(10, 20), reference_x=250.0)
        focus_display.clear_spot_overlay()
        for item in (
            focus_display.spot_candidates_item,
            focus_display.spot_selected_item,
            focus_display.spot_reference_line,
        ):
            assert not item.isVisible()

    def test_overlay_items_never_take_mouse_events(self, focus_display):
        """Clicks on this view start a stage move, a profiler line or a crop drag.

        A ScatterPlotItem accepts a left-click landing on one of its points by default, which
        would swallow the click before the view saw it -- and a marker sits exactly where the
        interesting part of the image is.
        """
        focus_display.set_spot_overlay(candidates=[(10, 20)], selected=(10, 20), reference_x=250.0)
        for item in (
            focus_display.spot_candidates_item,
            focus_display.spot_selected_item,
            focus_display.spot_reference_line,
        ):
            assert item.acceptedMouseButtons() == Qt.NoButton
            assert not item.acceptHoverEvents()

    def test_mark_spot_leaves_the_frame_single_channel(self, focus_display):
        """The old implementation painted a BGR copy, which silently defeats the LUT and autolevel."""
        focus_display.mark_spot(np.zeros((256, 512), dtype=np.uint8), 100, 50)
        assert focus_display.graphics_widget.img.image.ndim == 2
        assert len(focus_display.spot_selected_item.getData()[0]) == 1


class TestEndToEnd:
    """Real controller, real overlay, real display, wired as gui_hcs wires them.

    The full-GUI suite cannot construct HighContentScreeningGui in every environment, so this
    covers the same chain -- frame in, markers on the image -- without booting the GUI.
    """

    @pytest.fixture
    def chain(self, qtbot):
        controller = _make_controller(
            spot_detection_mode=SpotDetectionMode.DUAL_LEFT,
            has_reference=True,
            x_reference=210.0,
        )
        display = ImageDisplayWindow()
        qtbot.addWidget(display)
        overlay = LaserAFSpotOverlay(controller, display, rate_hz=5.0)
        clock = _FakeClock()
        overlay._now = clock
        return overlay, display, clock

    def test_a_frame_puts_markers_where_the_spots_are(self, chain):
        overlay, display, _ = chain
        overlay.set_enabled(True)
        overlay.on_frame(create_test_image([(200, 240), (440, 240)]))

        xs, ys = display.spot_candidates_item.getData()
        assert sorted(round(x) for x in xs) == [200, 440]
        assert all(y == pytest.approx(240, abs=2) for y in ys)
        # DUAL_LEFT selects the leftmost, 10 px from x_reference=210.
        assert display.spot_selected_item.getData()[0][0] == pytest.approx(200, abs=2)
        assert display.spot_reference_line.value() == pytest.approx(210.0)

    def test_a_frame_with_nothing_in_it_shows_up_as_a_failure(self, chain):
        overlay, display, _ = chain
        statuses = []
        overlay.signal_status.connect(statuses.append)
        overlay.set_enabled(True)
        overlay.on_frame(np.zeros((480, 640), dtype=np.uint8))

        assert "no spot detected" in statuses[-1]
        assert not display.spot_candidates_item.isVisible()
        assert not display.spot_selected_item.isVisible()

    def test_a_spot_far_from_the_reference_is_still_a_good_frame(self, chain):
        """What used to be the window rejection. The crop is what decides this now."""
        overlay, display, _ = chain
        statuses = []
        overlay.signal_status.connect(statuses.append)
        overlay.set_enabled(True)
        # 390 px right of x_reference=210 -- outside the window that used to be enforced here.
        overlay.on_frame(create_test_image([(600, 240)]))

        assert display.spot_selected_item.getData()[0][0] == pytest.approx(600, abs=2)
        assert statuses[-1] == ""

    def test_turning_it_off_removes_the_markers(self, chain):
        overlay, display, _ = chain
        overlay.set_enabled(True)
        overlay.on_frame(create_test_image([(200, 240)]))
        assert display.spot_candidates_item.isVisible()

        overlay.set_enabled(False)
        assert not display.spot_candidates_item.isVisible()
        assert not display.spot_reference_line.isVisible()

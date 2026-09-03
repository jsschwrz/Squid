"""Tests for the manual focus overlay on the laser AF sweep plot.

Two separable layers, tested separately. _manual_focus_spread and _summarize_manual_focus are
arithmetic over a list of z values and need no Qt at all. LaserAFSweepWidget decides which list to
read, what frame to place it in, and what range to measure it against -- and that last decision is
the one worth pinning down, because a cloud drawn against the wrong origin still looks like data.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from control.core.laser_auto_focus_controller import SweepSample
from control.models import LaserAFConfig
from control.widgets import LaserAFSweepWidget, _manual_focus_spread, _summarize_manual_focus


class TestManualFocusSpread:
    def test_spread_is_measured_from_the_given_origin(self):
        spread = _manual_focus_spread([4000.0, 4020.0, 4040.0], origin_um=4020.0)
        assert spread.offsets_um == [-20.0, 0.0, 20.0]
        assert spread.min_um == -20.0
        assert spread.max_um == 20.0
        assert spread.span_um == 40.0
        assert spread.mean_um == pytest.approx(0.0)

    def test_one_point_has_zero_span(self):
        # Not an error: zero span is a true statement about one point.
        spread = _manual_focus_spread([4000.0], origin_um=3990.0)
        assert spread.span_um == 0.0
        assert spread.mean_um == 10.0

    def test_no_points_gives_no_spread(self):
        assert _manual_focus_spread([], origin_um=0.0) is None

    def test_non_finite_values_are_dropped(self):
        # An un-focused point can reach the list as NaN, and one NaN would otherwise poison the
        # mean, the SD and both ends of the band.
        spread = _manual_focus_spread([4000.0, float("nan"), 4010.0], origin_um=4000.0)
        assert spread.offsets_um == [0.0, 10.0]

    def test_all_non_finite_gives_no_spread(self):
        assert _manual_focus_spread([float("nan")], origin_um=0.0) is None


class TestManualFocusSummary:
    def _spread(self):
        return _manual_focus_spread([-30.0, 0.0, 30.0, 60.0], origin_um=0.0)

    def test_span_ends_and_mean_and_nothing_else(self):
        # Everything dropped from this sentence is dropped because the plot says it better. If any
        # of it creeps back the line goes over one row again, which is what this pins.
        assert _summarize_manual_focus(self._spread()) == "Manual focus 90.0 um: -30.0 to +60.0, mean +15.0."

    def test_mean_centred_says_so_and_drops_the_mean(self):
        # The axis is labelled "z offset from sweep start", which this case is not measured from --
        # so the marker has to be here. The mean is zero by construction, so quoting it is noise.
        text = _summarize_manual_focus(_manual_focus_spread([-30.0, 30.0], origin_um=0.0), mean_centred=True)
        assert text == "Manual focus 60.0 um about the mean: -30.0 to +30.0."

    def test_a_mean_just_below_the_origin_is_not_written_as_minus_zero(self):
        # Floating point puts the mean of a symmetric spread a hair either side of zero, and
        # "mean -0.0" reads as a defect.
        text = _summarize_manual_focus(_manual_focus_spread([-20.0, 20.0], origin_um=1e-14))
        assert "mean +0.0." in text

    def test_one_point_reads_as_zero_span(self):
        assert _summarize_manual_focus(_manual_focus_spread([5.0], origin_um=0.0)) == (
            "Manual focus 0.0 um: +5.0 to +5.0, mean +5.0."
        )


def _make_widget(qtbot, piezo=None, current_z_um=4000.0, search_range_um=300.0):
    """A sweep widget over a stub controller, as test_laser_af_spot_overlay stubs its own."""
    controller = MagicMock()
    controller.piezo = piezo
    controller.laser_af_properties = LaserAFConfig(laser_af_search_range_um=search_range_um)
    controller.get_current_z_um.return_value = current_z_um
    widget = LaserAFSweepWidget(controller, MagicMock())
    qtbot.addWidget(widget)
    return widget


def _focus_map(*z_mm):
    return SimpleNamespace(focus_points=[(f"R{i}", 0.0, 0.0, z) for i, z in enumerate(z_mm)])


def _flexible(*z_mm):
    locations = np.zeros((len(z_mm), 3), dtype=float)
    locations[:, 2] = z_mm
    return SimpleNamespace(location_list=locations)


class TestReadingThePointLists:
    def test_focus_map_z_is_converted_to_um(self, qtbot):
        widget = _make_widget(qtbot)
        widget.set_manual_focus_sources(focusMapWidget=_focus_map(4.0, 4.02))
        assert widget._focus_map_z_um() == pytest.approx([4000.0, 4020.0])

    def test_flexible_z_comes_from_the_array_in_mm(self, qtbot):
        # The table alongside it displays um; reading that instead would be off by 1000x.
        widget = _make_widget(qtbot)
        widget.set_manual_focus_sources(flexibleMultiPointWidget=_flexible(4.0, 3.98))
        assert widget._flexible_multipoint_z_um() == pytest.approx([4000.0, 3980.0])

    def test_absent_sources_read_as_empty(self, qtbot):
        widget = _make_widget(qtbot)
        assert widget._focus_map_z_um() == []
        assert widget._flexible_multipoint_z_um() == []

    def test_the_only_populated_list_is_used_without_asking(self, qtbot):
        widget = _make_widget(qtbot)
        widget.set_manual_focus_sources(focusMapWidget=_focus_map(), flexibleMultiPointWidget=_flexible(4.0))
        source = widget._read_manual_focus_points(allow_prompt=True)
        assert source.key == "flexible"

    def test_a_remembered_choice_is_reused_rather_than_re_asked(self, qtbot):
        widget = _make_widget(qtbot)
        widget.set_manual_focus_sources(focusMapWidget=_focus_map(4.0), flexibleMultiPointWidget=_flexible(4.01))
        widget._manual_focus_source = "flexible"
        # allow_prompt=True and both populated: without the memory this would raise a modal chooser
        # and hang the test.
        assert widget._read_manual_focus_points(allow_prompt=True).key == "flexible"


class TestChoosingTheOrigin:
    def test_a_sweep_start_places_points_where_they_really_are(self, qtbot):
        widget = _make_widget(qtbot)
        widget._sweep_start_z_um = 4010.0
        assert widget._manual_focus_origin([4000.0, 4020.0]) == (4010.0, False)

    def test_before_any_sweep_the_origin_is_where_z_is_now(self, qtbot):
        widget = _make_widget(qtbot, current_z_um=4005.0)
        assert widget._manual_focus_origin([4000.0, 4020.0]) == (4005.0, False)

    def test_a_piezo_forces_centring_on_the_mean(self, qtbot):
        # The sweep runs on the piezo, so its axis cannot carry an absolute stage z at all.
        widget = _make_widget(qtbot, piezo=MagicMock())
        widget._sweep_start_z_um = 150.0
        assert widget._manual_focus_origin([4000.0, 4020.0]) == (4010.0, True)

    def test_mean_centring_puts_the_mean_at_zero(self, qtbot):
        widget = _make_widget(qtbot, piezo=MagicMock())
        origin_um, _ = widget._manual_focus_origin([4000.0, 4020.0, 4060.0])
        assert _manual_focus_spread([4000.0, 4020.0, 4060.0], origin_um).mean_um == pytest.approx(0.0)

    def test_an_unreadable_z_falls_back_to_the_mean(self, qtbot):
        widget = _make_widget(qtbot)
        widget.laserAutofocusController.get_current_z_um.side_effect = RuntimeError("no stage")
        assert widget._manual_focus_origin([4000.0, 4020.0]) == (4010.0, True)


class TestTheOverlay:
    def _items(self, widget):
        return (widget.manual_focus_region, widget.manual_focus_mean_line, widget.manual_focus_rug)

    def test_toggling_on_draws_all_three_items(self, qtbot):
        widget = _make_widget(qtbot)
        widget.set_manual_focus_sources(flexibleMultiPointWidget=_flexible(3.98, 4.0, 4.02))
        widget.btn_show_manual_focus.setChecked(True)

        assert all(item.isVisible() for item in self._items(widget))
        assert widget.manual_focus_region.getRegion() == pytest.approx((-20.0, 20.0))
        assert widget.manual_focus_mean_line.value() == pytest.approx(0.0)
        # One two-point segment per focus point.
        assert len(widget.manual_focus_rug.getData()[0]) == 6
        assert widget.status_label.text() == "Manual focus 40.0 um: -20.0 to +20.0, mean +0.0."

    def test_toggling_off_hides_them(self, qtbot):
        widget = _make_widget(qtbot)
        widget.set_manual_focus_sources(flexibleMultiPointWidget=_flexible(4.0, 4.02))
        widget.btn_show_manual_focus.setChecked(True)
        widget.btn_show_manual_focus.setChecked(False)

        assert not any(item.isVisible() for item in self._items(widget))
        assert widget._manual_focus_source is None

    def test_no_points_anywhere_unchecks_the_button(self, qtbot):
        widget = _make_widget(qtbot)
        widget.set_manual_focus_sources(focusMapWidget=_focus_map(), flexibleMultiPointWidget=_flexible())
        widget.btn_show_manual_focus.setChecked(True)

        assert not widget.btn_show_manual_focus.isChecked()
        assert not any(item.isVisible() for item in self._items(widget))
        assert "No focus points" in widget.status_label.text()

    def test_the_first_sample_re_places_the_overlay_on_the_new_origin(self, qtbot):
        widget = _make_widget(qtbot, current_z_um=4000.0)
        widget.set_manual_focus_sources(flexibleMultiPointWidget=_flexible(4.0, 4.02))
        widget.btn_show_manual_focus.setChecked(True)
        assert widget.manual_focus_region.getRegion() == pytest.approx((0.0, 20.0))

        # A sweep that started 10 um lower moves every point up by 10 on the axis.
        widget.on_sweep_sample(SweepSample(z_um=3995.0, dz_um=5.0))
        assert widget._sweep_start_z_um == 3990.0
        assert widget.manual_focus_region.getRegion() == pytest.approx((10.0, 30.0))

    def test_clear_re_places_the_overlay_rather_than_stranding_it(self, qtbot):
        widget = _make_widget(qtbot, current_z_um=4000.0)
        widget.set_manual_focus_sources(flexibleMultiPointWidget=_flexible(4.0, 4.02))
        widget.btn_show_manual_focus.setChecked(True)
        widget.on_sweep_sample(SweepSample(z_um=3995.0, dz_um=5.0))
        assert widget.manual_focus_region.getRegion() == pytest.approx((10.0, 30.0))

        # Clear drops the sweep origin, so the band goes back to being measured from current z.
        widget.clear()
        assert widget._sweep_start_z_um is None
        assert widget.btn_show_manual_focus.isChecked()
        assert widget.manual_focus_region.getRegion() == pytest.approx((0.0, 20.0))

    def test_the_sweep_summary_keeps_the_spread_sentence(self, qtbot):
        widget = _make_widget(qtbot)
        widget.set_manual_focus_sources(flexibleMultiPointWidget=_flexible(4.0, 4.02))
        widget.btn_show_manual_focus.setChecked(True)

        samples = [SweepSample(z_um=4000.0 + dz, dz_um=dz, selected_x=100.0 + dz) for dz in (-10.0, 0.0, 10.0)]
        widget.on_sweep_finished(samples)

        text = widget.status_label.text()
        assert "px RMS" in text  # the fit summary
        assert "Manual focus" in text  # and the spread, not one replacing the other
        # One line under the plot, not a paragraph. The two halves were each written to stand
        # alone and together ran to 362 characters before they were cut back.
        assert len(text) < 160, text
        assert "z positions" not in text  # the scatter says that

    def test_a_band_outside_the_swept_range_widens_the_plot(self, qtbot):
        # The overlay used to be drawn ignoreBounds, so a spread far from the sweep was simply not
        # on screen and only a sentence said so. The band now carries itself into view.
        widget = _make_widget(qtbot, current_z_um=4000.0)
        widget.set_manual_focus_sources(flexibleMultiPointWidget=_flexible(4.5, 4.6))
        for dz in (-10.0, 0.0, 10.0):
            widget.on_sweep_sample(SweepSample(z_um=4000.0 + dz, dz_um=dz, selected_x=300.0 + dz))
        widget.btn_show_manual_focus.setChecked(True)

        x_bounds, y_bounds = widget.plot.getViewBox().childrenBounds()
        assert x_bounds[0] <= 500.0 and x_bounds[1] >= 600.0
        # The mean line reports (0, 0) as its y bounds whatever its position, so letting it into
        # the bounds would drag the y scale down to 0 and squash the swept branch.
        assert y_bounds[0] > 100.0

    def test_hiding_the_band_gives_the_width_back(self, qtbot):
        widget = _make_widget(qtbot, current_z_um=4000.0)
        widget.set_manual_focus_sources(flexibleMultiPointWidget=_flexible(4.5, 4.6))
        for dz in (-10.0, 0.0, 10.0):
            widget.on_sweep_sample(SweepSample(z_um=4000.0 + dz, dz_um=dz, selected_x=300.0 + dz))
        widget.btn_show_manual_focus.setChecked(True)
        widget.btn_show_manual_focus.setChecked(False)

        x_bounds, _ = widget.plot.getViewBox().childrenBounds()
        assert x_bounds[1] < 100.0

    def test_the_overlay_stays_hidden_while_the_button_is_up(self, qtbot):
        widget = _make_widget(qtbot)
        widget.set_manual_focus_sources(flexibleMultiPointWidget=_flexible(4.0, 4.02))
        widget.on_sweep_sample(SweepSample(z_um=4000.0, dz_um=0.0))
        assert not any(item.isVisible() for item in self._items(widget))

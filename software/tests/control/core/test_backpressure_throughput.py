"""Tests for the throughput-driven additions to backpressure:

- cumulative captured/written counters + RateSampler,
- the adaptive effective byte cap, and
- MultiPointWorker format-aware writer routing.
"""

import time
from unittest.mock import MagicMock

import pytest

import squid.abc
from control.core.backpressure import BackpressureController, RateSampler
from control.core.job_processing import CaptureInfo
from control.models import AcquisitionChannel, CameraSettings, IlluminationSettings


def make_capture_info(region_id: str = "A1", fov: int = 0) -> CaptureInfo:
    return CaptureInfo(
        position=squid.abc.Pos(x_mm=0.0, y_mm=0.0, z_mm=0.0, theta_rad=None),
        z_index=0,
        capture_time=time.time(),
        configuration=AcquisitionChannel(
            name="BF LED matrix full",
            display_color="#FFFFFF",
            camera=1,
            illumination_settings=IlluminationSettings(illumination_channel="LED", intensity=50.0),
            camera_settings=CameraSettings(exposure_time_ms=10.0, gain_mode=1.0),
            z_offset_um=0.0,
        ),
        save_directory="/tmp/test",
        file_id=f"test_{fov}",
        region_id=region_id,
        fov=fov,
        configuration_idx=0,
    )


class TestRateSampler:
    def test_first_sample_seeds_silently(self):
        s = RateSampler()
        s.update(1000, 500, now=10.0)
        assert s.capture_mb_s == 0.0
        assert s.write_mb_s == 0.0

    def test_positive_rate_after_second_sample(self):
        s = RateSampler(alpha=1.0)  # pure instantaneous -> easy to assert
        s.update(0, 0, now=0.0)
        s.update(1024 * 1024, 512 * 1024, now=1.0)  # 1 MiB / 0.5 MiB over 1 s
        assert s.capture_mb_s == pytest.approx(1.0, rel=1e-3)
        assert s.write_mb_s == pytest.approx(0.5, rel=1e-3)

    def test_dt_guard_ignores_too_frequent_samples(self):
        s = RateSampler(alpha=1.0, min_dt_s=0.05)
        s.update(0, 0, now=0.0)
        s.update(10 * 1024 * 1024, 0, now=0.01)  # dt below guard -> ignored
        assert s.capture_mb_s == 0.0

    def test_negative_delta_clamped(self):
        s = RateSampler(alpha=1.0)
        s.update(10 * 1024 * 1024, 10 * 1024 * 1024, now=0.0)
        s.update(0, 0, now=1.0)  # counters reset -> no negative rate
        assert s.capture_mb_s == 0.0
        assert s.write_mb_s == 0.0

    def test_reset_clears_state(self):
        s = RateSampler(alpha=1.0)
        s.update(0, 0, now=0.0)
        s.update(1024 * 1024, 1024 * 1024, now=1.0)
        assert s.write_mb_s > 0
        s.reset()
        assert s.write_mb_s == 0.0
        s.update(5 * 1024 * 1024, 5 * 1024 * 1024, now=2.0)  # seeds silently again
        assert s.write_mb_s == 0.0


class TestCumulativeCounters:
    def test_job_dispatched_increments_captured(self):
        c = BackpressureController(max_jobs=100, max_mb=1000.0)
        c.job_dispatched(2 * 1024 * 1024)
        stats = c.get_stats()
        assert stats.captured_bytes == 2 * 1024 * 1024
        assert stats.captured_count == 1
        assert stats.written_bytes == 0
        c.close()

    def test_job_completed_increments_written_and_drains_pending(self):
        c = BackpressureController(max_jobs=100, max_mb=1000.0)
        c.job_dispatched(3 * 1024 * 1024)
        c.job_completed(3 * 1024 * 1024)
        stats = c.get_stats()
        assert stats.written_bytes == 3 * 1024 * 1024
        assert stats.written_count == 1
        assert stats.pending_jobs == 0
        assert stats.pending_bytes_mb == pytest.approx(0.0)
        c.close()

    def test_reset_clears_cumulative(self):
        c = BackpressureController(max_jobs=100, max_mb=1000.0)
        c.job_dispatched(1024 * 1024)
        c.job_completed(1024 * 1024)
        c.reset()
        stats = c.get_stats()
        assert stats.captured_bytes == 0
        assert stats.written_bytes == 0
        c.close()

    def test_closed_stats_have_zeroed_new_fields(self):
        c = BackpressureController(max_jobs=1, max_mb=1.0)
        c.close()
        stats = c.get_stats()
        assert stats.captured_bytes == 0
        assert stats.written_bytes == 0
        assert stats.write_mb_s == 0.0


class TestAdaptiveCap:
    def test_static_when_target_backlog_zero(self):
        c = BackpressureController(max_jobs=10, max_mb=500.0, target_backlog_s=0.0)
        assert c._effective_max_bytes() == int(500.0 * 1024 * 1024)
        c.close()

    def test_cap_tracks_write_rate(self):
        c = BackpressureController(max_jobs=10, max_mb=10000.0, target_backlog_s=30.0, floor_mb=100.0)
        c._rate_sampler._write_mb_s = 200.0  # 200 * 30 = 6000 MB, within [100, 10000]
        assert c._effective_max_bytes() == int(6000.0 * 1024 * 1024)
        c.close()

    def test_cap_respects_floor(self):
        c = BackpressureController(max_jobs=10, max_mb=10000.0, target_backlog_s=30.0, floor_mb=500.0)
        c._rate_sampler._write_mb_s = 0.0
        assert c._effective_max_bytes() == int(500.0 * 1024 * 1024)
        c.close()

    def test_cap_respects_ceiling(self):
        c = BackpressureController(max_jobs=10, max_mb=1000.0, target_backlog_s=30.0, floor_mb=100.0)
        c._rate_sampler._write_mb_s = 1000.0  # 1000 * 30 = 30000 > ceiling
        assert c._effective_max_bytes() == int(1000.0 * 1024 * 1024)
        c.close()


class TestWriterRouting:
    def _worker(self, region_only):
        w = MagicMock()
        w._route_by_region_only = region_only
        return w

    def test_single_writer_always_zero(self):
        from control.core.multi_point_worker import MultiPointWorker

        info = make_capture_info(region_id="A1", fov=7)
        assert MultiPointWorker._writer_index(self._worker(False), info, 1) == 0

    def test_fov_level_deterministic_and_in_range(self):
        from control.core.multi_point_worker import MultiPointWorker

        w = self._worker(False)
        info = make_capture_info(region_id="A1", fov=7)
        idx = MultiPointWorker._writer_index(w, info, 4)
        assert 0 <= idx < 4
        assert idx == MultiPointWorker._writer_index(w, info, 4)  # pure function

    def test_region_only_ignores_fov(self):
        from control.core.multi_point_worker import MultiPointWorker

        w = self._worker(True)
        a = make_capture_info(region_id="B2", fov=0)
        b = make_capture_info(region_id="B2", fov=9)
        # 6D zarr: every FOV of a region must map to the same writer
        assert MultiPointWorker._writer_index(w, a, 8) == MultiPointWorker._writer_index(w, b, 8)

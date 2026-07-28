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


class TestJobRunnerWarmupOffFramePath:
    """Writer-subprocess warmup must not block the image callback.

    The image callback sets _ready_for_next_trigger; the acquisition thread only allows
    5 * total_frame_time + 2 seconds for a frame. Waiting for a cold subprocess inside
    the callback pushed the first frame past that deadline and aborted the run whenever
    ACQUISITION_WRITER_PROCESSES > 1 (only the first runner is pre-warmed).
    """

    def test_wait_ready_is_not_called_from_image_callback(self):
        import inspect

        from control.core.multi_point_worker import MultiPointWorker

        callback_src = inspect.getsource(MultiPointWorker._image_callback)
        assert "wait_ready" not in callback_src, (
            "wait_ready() must not be called from _image_callback -- it blocks frame "
            "delivery and trips the frame-arrival timeout."
        )

    def test_warmup_helper_waits_for_every_runner(self):
        from control.core.multi_point_worker import MultiPointWorker

        class FakeRunner:
            def __init__(self, ready=True):
                self.ready = ready
                self.calls = 0

            def wait_ready(self, timeout_s):
                self.calls += 1
                return self.ready

        runners = [FakeRunner(), FakeRunner(), FakeRunner()]
        worker = MagicMock()
        worker._job_runners = [(SlowJobStub, runners)]
        worker._log = MagicMock()

        MultiPointWorker._wait_for_job_runners_ready(worker, timeout_s=5.0)

        # Every runner is waited on exactly once, including a None entry being skipped.
        assert [r.calls for r in runners] == [1, 1, 1]

    def test_unready_runner_does_not_raise(self):
        from control.core.multi_point_worker import MultiPointWorker

        class NeverReady:
            def wait_ready(self, timeout_s):
                return False

        worker = MagicMock()
        worker._job_runners = [(SlowJobStub, [NeverReady(), None])]
        worker._log = MagicMock()

        # A runner that never comes up is logged and skipped, not fatal.
        MultiPointWorker._wait_for_job_runners_ready(worker, timeout_s=0.01)
        assert worker._log.warning.called


class SlowJobStub:
    __name__ = "SlowJobStub"


class TestSpanFloor:
    """The cap must stay large enough for parallel writers to actually engage.

    Writers are routed per FOV, so if the cap is smaller than one z-stack only one
    writer is ever busy -- and the low write rate that causes holds the adaptive cap
    down, which keeps it that way. Observed live: cap pinned at the 1024 MB floor with
    a 1505 MB z-stack, 4 writers configured, only 1 ever busy.
    """

    def _controller(self, min_span_images, floor_mb=1024.0, max_mb=49152.0):
        return BackpressureController(
            max_jobs=10000,
            max_mb=max_mb,
            target_backlog_s=30.0,
            floor_mb=floor_mb,
            min_span_images=min_span_images,
        )

    def test_no_span_floor_when_single_writer(self):
        c = self._controller(min_span_images=0)
        c.job_dispatched(50 * 1024 * 1024)
        assert c._span_floor_bytes() == 0
        c.close()

    def test_span_floor_zero_before_any_image(self):
        # No image dispatched yet -> unknown mean size -> plain floor applies.
        c = self._controller(min_span_images=116)
        assert c._span_floor_bytes() == 0
        assert c._effective_max_bytes() == int(1024.0 * 1024 * 1024)
        c.close()

    def test_mean_image_bytes_from_counters(self):
        c = self._controller(min_span_images=4)
        c.job_dispatched(10 * 1024 * 1024)
        c.job_dispatched(20 * 1024 * 1024)
        assert c.mean_image_bytes() == pytest.approx(15 * 1024 * 1024)
        c.close()

    def test_span_floor_lifts_cap_above_one_zstack(self):
        # The real case: 4 writers x 29 z-levels, ~51.9 MB per image.
        img = int(51.9 * 1024 * 1024)
        c = self._controller(min_span_images=4 * 29)
        c.job_dispatched(img)
        c._rate_sampler._write_mb_s = 25.0  # 25 * 30 = 750 MB, below the old 1024 floor

        one_stack = 29 * img
        assert c._effective_max_bytes() > one_stack, "cap must exceed a single z-stack"
        assert c._effective_max_bytes() == pytest.approx(4 * 29 * img, rel=1e-6)
        c.close()

    def test_ceiling_still_wins_over_span_floor(self):
        # Never blow past the absolute RAM limit just to chase parallelism.
        img = int(51.9 * 1024 * 1024)
        c = self._controller(min_span_images=4 * 29, max_mb=2048.0)
        c.job_dispatched(img)
        assert c._effective_max_bytes() == int(2048.0 * 1024 * 1024)
        c.close()
